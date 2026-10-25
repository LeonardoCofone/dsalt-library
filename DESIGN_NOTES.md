# DSALT, Engineering Design Notes

How the DSALT attention (an adaptive local window `W(i)` plus landmark tokens `L(i)`
selected by a hybrid score) is implemented so that it stays sparse on the GPU, can be
trained end to end, and runs under DDP and `torch.compile`.

The library was developed on a node with two T4 GPUs (`sm_75`) running
`DistributedDataParallel`; the measurements below come from that node. Tile sizes,
warp counts, pipeline depth and shared-memory budgets are resolved at runtime for
each device.

---

## 1. Differentiable selectors in a sparse kernel

Attention is computed over `A(i) = W(i) ∪ L(i)`. Computing it with a soft mask over
all positions would need the dense `L × L` score matrix. The kernels instead keep the
computation block-sparse and make two terms differentiable, so that the Triton
kernels work with PyTorch autograd.

### 1.1 Soft window edge

The band mask must be a hard boundary: a query block only loads the key tiles inside
its window, so tiles outside it are never read. Only the border is soft. For distance
`d` and continuous window `w̃(i)`, the kernel adds a log-bias:

```
z         = (w̃(i) − d) / τ_win        # signed distance to the edge
in_core   = d ≤ (w̃(i) − win_edge)      # inside tokens: bias 0
edge_bias = log σ(z)                   # border tokens: differentiable in w̃(i)
```

The window predictor `win_gate` receives gradient only from the `win_edge` border
tokens: in the backward `∂bias/∂w̃ = (1 − σ(z)) / τ_win`, masked to zero on the core.
The extra work is `O(win_edge)` per query.

### 1.2 Landmarks: detached selection, differentiable weight

Top-k selection is not differentiable, and DSALT does not relax it. It separates
which tokens are selected from how much they weigh:

- **Selection, detached.** The score `s = α·z(‖x·W_V‖₂) + (1−α)·z(‖x‖₂)` is evaluated
  under `torch.no_grad` to pick the `k` landmark indices per (head, sequence). The
  indices carry no gradient.
- **Weight, differentiable in α.** A second, gradient-carrying evaluation of the same
  score gives a log-bias `log σ(s_j(α) / τ_lmk)` at the selected indices, added to the
  landmark logits in the kernel. The gradient with respect to `α = σ(α̃)` flows
  through this term.

The score formula is defined once in `landmark_tokens_ker.py` and shared by the
kernel path, the selector code and the dense reference. The union `W ∪ L` is the
elementwise `max` of the two log-biases.

The selection standardizes the scores over the whole sequence and takes the top-k
over all positions, so it is not causal in the strict sense: a query attends only to
earlier tokens, but which earlier tokens are landmarks depends on later ones.

---

## 2. DDP and `torch.compile`

DSALT trains under `DistributedDataParallel` and `torch.compile` together. The
attention forward and backward are Triton kernels inside a custom
`torch.autograd.Function` (`DSALTTrainFunction`) that Dynamo must not trace.

### 2.1 Opaque kernels

The Triton entry points are marked `torch._dynamo.disable`. Dynamo breaks the graph
at each kernel and compiles the code around it: RoPE, selectors, RMSNorm, residual
adds, SwiGLU FFN and loss. On the T4 node this raised single-GPU throughput from
2.278 to 3.171 it/s and lowered peak memory, with the kernel outputs unchanged
(checked against the dense reference).

### 2.2 DDPOptimizer disabled under DDP

Under DDP, Dynamo's `DDPOptimizer` splits the compiled graph at the all-reduce bucket
boundaries. With the custom autograd `Function` in the middle of the graph, the split
breaks the backward link between the loss and the parameters before the kernel, and
`loss.backward()` fails with:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The trainer therefore disables it under DDP before compiling:

```python
if world_size > 1:
    torch._dynamo.config.optimize_ddp = False
model = torch.compile(model, fullgraph=False)
```

The graph stays whole, with breaks at the kernel, and DDP's all-reduce hooks run
normally. The overlap of communication and compute that `DDPOptimizer` provides is
lost; on the T4 node the backward kernel, not communication, was the bottleneck. A
single-GPU run never takes this path, so it has to be tested on a distributed job.

If compilation fails, the trainer falls back to eager execution.

---

## 3. Key-parallel backward

A custom attention backward usually accumulates `dk, dv` with `atomic_add`, because
many query blocks contribute to the same key. DSALT splits the backward into two
kernels with different parallel axes.

### 3.1 Two kernels

- **`_train_bwd_dkdv_kernel`, key-parallel, no atomics.** The window-band `dk, dv` are
  computed with the keys as the parallel axis. Each key tile accumulates the
  contribution of every query that sees it in registers and writes the result once.
- **`_train_bwd_kernel`, query-parallel.** Computes `dq`, the window gradient `d_w̃`
  and the landmark weight gradient `d_logw`. The only atomics left are for the
  landmark `dk, dv` and `d_logw` (`k` landmarks per sequence), outside the inner loop.

`BLOCK_M` is shared by forward and backward, so both use the same `seq_block_map`
(the packed sequence-to-block table).

### 3.2 Tuning `BLOCK_N_BWD` by shared memory

The backward key tile `BLOCK_N_BWD` is chosen at runtime within the shared-memory
capacity of the device, in two phases:

1. **Phase 1** fixes the forward axes (`BLOCK_M`, `BLOCK_N`, warps, pipeline stages),
   with candidates filtered by an estimated shared-memory footprint
   (`_fits_shared_memory`) and `BLOCK_N_BWD = 32`.
2. **Phase 2** (`_bwd_tile_candidates`) varies only `BLOCK_N_BWD` around the phase-1
   config, keeping the tiles that fit the backward budget of the device
   (`_bwd_tile_fits_smem`). Every `tl.dot` dimension is kept at least 16, as `sm_75`
   requires.

The budget is read from the device (`_smem_budget`, about 75% of the per-block
maximum, `shared_memory_per_block_optin` when available, otherwise 48 KB). The
pipeline uses 3 stages on `sm_80` and newer and 2 on smaller GPUs. The search runs
once per `(head_dim, compute capability)`; `@triton.autotune` would re-tune on every
new packed shape.

---

## 4. Profiling

Measurements come from per-category CUDA profiling on the T4 node: self-CUDA time is
attributed to Triton attention, GEMM, cast/copy, softmax/reduce, elementwise and
norm, and `aten::copy_` is broken down by tensor shape.

### 4.1 Sparse forward against fused dense SDPA

At sequence length 1024 the sparse forward kernel takes about 25 ms per step, against
about 30 ms for fused dense `scaled_dot_product_attention` on the same hardware. In
that setup the remaining speed gap to the dense baseline came from other choices (DDP
all-reduce against `DataParallel`, chunked loss against materialized logits), not
from the attention. Peak memory in the two-rank job was about 3.35 GB.

### 4.2 Throughput and window size

The kernel scans, for each query block, back to the largest window in the block, so
throughput depends on how the predicted windows are distributed, and it changes
during training as the windows change. The `scan_block_max` and `scan_ratio` metrics
logged by the trainer track this directly.

### 4.3 Verification

Kernel changes are checked on GPU against the dense SDPA reference
(`python -m dsalt.kernels._verify_train_kernel`). The reference uses the same
selectors as the kernel, and the check compares outputs and gradients in fp16, with
landmark counts that include the padding to 16 required on `sm_75`.
