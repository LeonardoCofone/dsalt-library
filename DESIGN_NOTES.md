# DSALT, Engineering Design Notes

This document records the engineering rationale behind DSALT (**Dynamic Sparse
Attention with Landmark Tokens**). Where the paper states *what* the model computes
(§4.2 the adaptive local window, §4.3 the hybrid-energy landmarks), these notes
state *how* that computation is realised on real hardware without sacrificing
end-to-end differentiability, distributed-training correctness, or throughput.

The reference deployment target is a dual-T4 node (compute capability `sm_75`)
running `DistributedDataParallel` over two ranks, but every decision below is
**device-agnostic**: tile sizes, warp counts, pipeline depth and shared-memory
budgets are resolved at runtime per device, never hard-coded for one GPU.

---

## 1. Differentiable Approximations for Hardware-Efficient Attention

The paper defines attention over a *continuous* admissible set
`A(i) = W(i) ∪ L(i)` (eq. 32). A literal continuous implementation would force a
dense `L × L` score matrix and defeat the sub-quadratic claim. DSALT instead keeps
the on-chip computation **structurally sparse** (block-sparse over SRAM tiles) and
recovers learnability through two carefully scoped differentiable approximations,
so that the discrete Triton kernels remain compatible with PyTorch Autograd.

### 1.1 Window Soft-Edge (border smoothing)

To preserve sub-quadratic complexity, the per-block band mask must be a **hard
boundary**: a query block only ever touches the key tiles that fall inside its
window radius, so out-of-window tiles are never loaded into SRAM. A naively
differentiable (soft) window would attend everywhere with vanishing weight and
reintroduce the `O(L²)` cost.

The resolution is to make the boundary differentiable **only at the margin**.
For relative distance `d` and continuous window `w̃(i)`, the kernel computes a
log-bias with a hard interior and a smooth transition band:

```
z        = (w̃(i) − d) / τ_win          # signed, normalised distance to the edge
in_core  = d ≤ (w̃(i) − win_edge)        # strictly-inside tokens: rigid, bias = 0
edge_bias = log σ(z)                     # marginal tokens: differentiable in w̃(i)
```

Interior tokens carry zero bias (full, rigid membership); only the `win_edge`-wide
band around the boundary is soft. The window predictor `win_gate` therefore
receives a gradient **exclusively from the marginal tokens**, in the backward the
sensitivity is `∂bias/∂w̃ = (1 − σ(z)) / τ_win` and is masked to zero on the core
(`in_core`). This yields a learnable `w̃(i)` at the cost of `O(win_edge)` extra
work per query, not `O(L)`. The temperature `τ_win` controls the sharpness of the
transition.

### 1.2 Landmark Soft-Reweight (detached selection, differentiable gating)

Landmark membership is a **top-k** operation, which is not differentiable. DSALT
does not attempt to relax the selection itself; instead it separates *which* tokens
are admitted from *how strongly* they participate:

- **Selection, detached, non-differentiable.** The hybrid-energy score
  `s = α·z(‖x·W_V‖₂) + (1−α)·z(‖x‖₂)` (§4.3) is evaluated under `torch.no_grad`
  to pick the `k` landmark indices per (head, sequence). These indices are hard and
  carry no gradient, they only address memory.

- **Re-weight, differentiable in α.** A second, gradient-carrying evaluation of
  the same score formula produces a per-landmark log-bias
  `log σ(s_j(α) / τ_lmk)` **at the admitted indices only**. This gating term is
  added to the landmark logits inside the kernel, so the gradient with respect to
  the per-head balancing parameter `α = σ(α̃)` flows through the log-bias of the
  surviving tokens, exactly as the paper's differentiable selector prescribes.

The single source of the score formula lives in `landmark_tokens_ker.py` and is
shared by the kernel path, the selector prelude and the dense reference, so the
three paths cannot diverge. The union `W ∪ L` is realised as an elementwise `max`
of the two log-biases.

---

## 2. Distributed Training Architecture and Graph Compilation

DSALT trains under `DistributedDataParallel` (DDP) **and** `torch.compile`
simultaneously. Their coexistence is non-trivial because the attention forward and
backward are hand-written Triton kernels wrapped in a custom
`torch.autograd.Function` (`DSALTTrainFunction`) that Dynamo cannot and must not
trace into.

### 2.1 Opaque kernels, fused surroundings

The Triton entry points are marked `torch._dynamo.disable` (opaque). Dynamo then
takes a clean **graph break** at each kernel and compiles everything around it,
RoPE, the selectors, RMSNorm, residual adds, the SwiGLU FFN and the loss, which is
where the per-op launch overhead lived. Measured effect on the reference node:
**+39 % iterations/s** single-GPU (2.278 → 3.171 it/s) and a lower peak footprint,
with the kernel numerics unchanged (verified against the dense reference).

### 2.2 Strategic exclusion of the DDPOptimizer

Under DDP, Dynamo enables `DDPOptimizer`, which splits the compiled graph at the
DDP all-reduce bucket boundaries to overlap gradient communication with compute.
With a custom autograd `Function` sitting **mid-graph**, that splitter severs the
backward link between the post-kernel subgraph (the loss) and the pre-kernel
parameters, producing at `loss.backward()`:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The engineered fix is to disable the optimizer **only under DDP**, before
compiling:

```python
if world_size > 1:
    torch._dynamo.config.optimize_ddp = False
model = torch.compile(model, fullgraph=False)
```

This keeps a single fused graph with clean graph-breaks at the kernel while DDP's
eager all-reduce hooks fire normally. The bucket/compute overlap that
`DDPOptimizer` would have provided is forfeited deliberately, profiling shows the
backward kernel, not gradient communication, is the critical path, so the trade is
correctness-for-free. The fix was confirmed on the live two-rank job: the loss
regains its `grad_fn` on every rank and training converges. A single-GPU profiler
never exercises this path (no DDP, no `DDPOptimizer`), so the regression test must
run on a real distributed job.

`torch.compile` is treated strictly as a performance knob: any hard compile failure
falls back to eager execution and never gates correctness.

---

## 3. Asymmetric Key-Parallel Backward Design

A custom attention backward typically accumulates the key/value gradients
`dk, dv` with `atomic_add`, because many query blocks contribute to the same key.
On `sm_75` that atomic contention serialises writes and inflates the backward.
DSALT decomposes the backward pass into two specialised kernels with **asymmetric
parallelisation axes** to remove the contention entirely on the dominant term.

### 3.1 Two kernels, two axes

- **`_train_bwd_dkdv_kernel`, key-parallel, zero-atomic.** The window-band
  `dk, dv` are computed with the keys as the parallel axis. Each key tile owns its
  output region and accumulates the contribution of every query that sees it in
  local registers, writing the result to global memory **exactly once**. There is
  no `atomic_add` on the band path, the previous query-parallel scheme had multiple
  query blocks writing the same key location and required atomics; flipping the axis
  makes each write exclusive.

- **`_train_bwd_kernel`, query-parallel, light.** This kernel computes `dq`, the
  window-size gradient `d_w̃` and the landmark gate gradient `d_logw`. The only
  remaining atomics are for the **landmark** `dk, dv` and `d_logw`, which are rare
  (`k` landmarks per sequence, gathered indices) and live **outside** the inner
  loop, so their cost is negligible. The band, the bulk of the work, is
  atomic-free.

`BLOCK_M` is shared between forward and backward so the two passes consume the same
`seq_block_map` (the packed sequence→block table), avoiding a second host-side
build.

### 3.2 SRAM-capacity-bound autotuning of `BLOCK_N_BWD`

The backward key tile `BLOCK_N_BWD` is selected at runtime, bound by the **shared
memory capacity of the device**, not by a constant. Tuning proceeds in two phases:

1. **Phase 1** fixes the forward axes (`BLOCK_M`, `BLOCK_N`, warps, pipeline
   stages), with every candidate filtered by an estimated shared-memory footprint
   (`_fits_shared_memory`); the backward tile is pinned to a safe `BLOCK_N_BWD = 32`
   to keep this phase small.
2. **Phase 2** (`_bwd_tile_candidates`) varies *only* `BLOCK_N_BWD` around the fixed
   phase-1 config, keeping every tile that fits the backward shared-memory budget on
   *this* device (`_bwd_tile_fits_smem`). Each dimension of any `tl.dot` is held
   `≥ 16` for `sm_75` legality.

The usable budget is read from the device at runtime (`_smem_budget` ≈ 75 % of the
declared per-block maximum, with `shared_memory_per_block_optin` preferred when
available, falling back to a conservative 48 KB). The pipeline depth follows the
same logic, 3 stages where more SRAM is available (`sm_80`+), 2 on the tighter
T4, so the kernel transports cleanly to newer GPUs without code changes. The whole
search runs **once** per `(head_dim, compute capability)` and is then fixed for the
run, unlike `@triton.autotune`, which would re-tune on every new packed-sequence
shape.

---

## 4. Algorithmic Micro-Benchmarks & Profiling Veracity

Performance claims are backed by per-category CUDA profiling on the reference node,
not by wall-clock anecdote. The profiler attributes self-CUDA time to buckets
(Triton attention, GEMM, cast/copy, softmax/reduce, elementwise, norm) and breaks
down `aten::copy_` by tensor shape, so each statement below is traceable to a
measured kernel.

### 4.1 The sparse attention forward is iso-speed with fused dense SDPA

The sparse **forward** kernel costs ~**25 ms/step** on the reference node, at or
below an industrial fused dense SDPA (`scaled_dot_product_attention`, ~30 ms on the
same hardware). The DSALT speed gap versus a dense baseline is therefore **not** in
the attention: it is attributable to orthogonal training-setup choices (a true DDP
all-reduce versus `DataParallel`, and a memory-frugal chunked loss versus a single
materialised logits tensor). The attention mechanism itself, the contribution of
the paper, is already competitive on speed while consuming **drastically less
memory** (peak ~**3.35 GB** in the real two-rank job, against a much larger dense
footprint). The asymptotic advantage of `W ∪ L` over dense `O(L²)` only widens as
the sequence length grows beyond the benchmarked 1024.

### 4.2 Throughput saturation is head specialisation, not a regression

Iterations/s dips slightly during the early phase of training and then stabilises.
This is **not** a performance leak: it is the empirical signature of the adaptive
window specialising per head. The kernel's scan cost is bound by the per-block
maximum window `max(w̃)`, each query block must scan as far as its widest member
requires. Early in training the heads have not yet differentiated, so `w̃` is broad;
as the heads specialise, blocks that do not need long context contract their window
and the per-block scan shrinks. The `scan` metric logged each step makes this
observable directly: it tracks `max(w̃)` and the realised scan ratio, and it
saturates as specialisation completes. In other words, the model dynamically
expands the scan block only where the contextual complexity demands it, the
"slowdown" is the optimiser purchasing sparsity, and the metric proves it converges
rather than drifting.

### 4.3 Verification gate

Every kernel-affecting change is validated on GPU against the dense SDPA reference
before it is trusted. The reference consumes the *exact same* selectors as the
kernel (shared prelude), so a passing check certifies that the hand-written Triton
math matches the differentiable specification, absolute deltas in fp16, not
relative, across landmark counts including the padded-to-16 minimum required by
`sm_75`'s `tl.dot` constraint.
