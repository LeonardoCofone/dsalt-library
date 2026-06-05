# DSALT, Features & Hyperparameters Guide

A complete reference for **every** public component, its real constructor
signature, every default, every option, and the recommended usage. Everything
here is verified against the source, no placeholders, no planned-but-missing
APIs. Reading this file takes you from zero knowledge of the library to full
working command of it.

---

## Table of Contents

1. [What DSALT computes](#1-what-dsalt-computes)
2. [Installation & extras](#2-installation--extras)
3. [`DSALTLMHeadModel`, the causal language model](#3-dsaltlmheadmodel--the-causal-language-model)
4. [`DSALTConfig`, serializable configuration](#4-dsaltconfig--serializable-configuration)
5. [`DSALTAttention`, the sparse attention module](#5-dsaltattention--the-sparse-attention-module)
6. [`DSALTTransformerBlock` & `SwiGLUFFN`](#6-dsalttransformerblock--swigluffn)
7. [The loss functions](#7-the-loss-functions)
8. [`DSALTTrainer`, the training loop](#8-dsalttrainer--the-training-loop)
9. [Data format: packed sequences](#9-data-format-packed-sequences)
10. [Mixed precision & device portability](#10-mixed-precision--device-portability)
11. [Distributed training (DDP) & torch.compile](#11-distributed-training-ddp--torchcompile)
12. [Diagnostics & checkpointing](#12-diagnostics--checkpointing)
13. [Low-level Triton kernels](#13-low-level-triton-kernels)
14. [Top-level exports](#14-top-level-exports)
15. [End-to-end example](#15-end-to-end-example)
16. [License](#license)

---

## 1. What DSALT computes

**DSALT** (Dynamic Sparse Attention with Landmark Tokens) replaces dense `O(N²)`
attention with the union of two sparse sets per query `i`:

```
A(i) = W(i) ∪ L(i)            (paper eq. 32)
```

- **`W(i)`, adaptive local causal window (§4.2).** A learned linear projection
  `win_gate: R^d → R` predicts, per token, a continuous window size

  ```
  w̃(i) = n_min + σ(f(x_i)) · (n_max − n_min)
  ```

  from the block input (the previous layer's hidden state, so there is no
  circular dependency). The window core is a **hard** mask, a query block only
  loads the key tiles inside its radius, which is what keeps the cost
  sub-quadratic, but the boundary carries a **soft, differentiable** edge of
  width `win_edge` so that `win_gate` is trained. At inference `w̃` is floored to
  an integer window (paper eq. 29).

- **`L(i)`, global landmark tokens (§4.3).** A per-head **hybrid-energy** score

  ```
  s_j = α · z(‖x_j · W_V‖₂) + (1 − α) · z(‖x_j‖₂),   α = σ(α̃)  (per head)
  ```

  (where `z(·)` is standardisation over the candidate tokens) ranks tokens; the
  top-`k_lmk` per head become landmarks. The **selection is hard** (top-k,
  detached, it only addresses memory), while a **soft re-weight** `log σ(s_j/τ)`
  added to the admitted landmarks' logits makes the per-head balance `α` trainable.

Both predictors, `win_gate` (§4.2) and the per-head `α` (§4.3), are **trained**.
The union `W ∪ L` is realised as the elementwise `max` of the two additive
log-biases; the raw `q·k` score itself is never biased by the selector. There is
**no separate auxiliary loss**: the predictors receive their gradient directly
inside the main forward through the soft edge / soft weight.

On CUDA this runs as fused **Triton** kernels (FlashAttention-2-style forward and
a split backward, online softmax, one-shot autotuned block sizes). Where Triton is
unavailable, a masked **SDPA** path implements the *same* math and doubles as the
correctness reference. The package always stays importable (CPU, Windows included).

---

## 2. Installation & extras

```bash
pip install dsalt                 # core, depends only on torch>=2.0.0
pip install "dsalt[triton]"       # + Triton GPU kernels (triton>=2.0.0)
pip install "dsalt[dev]"          # + pytest, pytest-cov, black, isort, flake8, mypy, pre-commit
pip install "dsalt[docs]"         # + sphinx, sphinx-rtd-theme, myst-parser
pip install "dsalt[build]"        # + build, twine (packaging/release)
pip install "dsalt[all]"          # triton + dev + docs + build
```

- **Python** ≥ 3.10, **PyTorch** ≥ 2.0.
- Triton is optional and only matters on Linux/CUDA. Without it, `loss_fn="liger"`
  is unavailable and the attention uses the SDPA fallback, everything else works.

---

## 3. `DSALTLMHeadModel`, the causal language model

`dsalt.model.DSALTLMHeadModel`, token embedding → a stack of
`DSALTTransformerBlock` → final RMSNorm → (optionally tied) LM head.

The optimized training path consumes **packed** sequences (concatenated tokens +
a `cu_seqlens` offset tensor). A plain `[B, T]` tensor is accepted for inference.

### Required parameters

```python
vocab_size:  int   # Vocabulary size
d_model:     int   # Hidden dimension; must be divisible by n_heads
n_layers:    int   # Number of transformer blocks
n_heads:     int   # Number of attention heads
n_min:       int   # Lower bound of the adaptive window (§4.2)
n_max:       int   # Upper bound of the adaptive window (§4.2)
k_lmk:       int   # Number of landmark tokens per query/head (§4.3)
max_seq_len: int   # Max length for the RoPE/YaRN cache
```

### Optional parameters (defaults match the constructor exactly)

```python
d_ff:               int | None = None      # FFN hidden dim; if None → round(8/3 · d_model) up to a multiple of 128
dropout:            float      = 0.0       # Dropout on embeddings / blocks
yarn_scale:         float      = 1.0       # RoPE / YaRN positional scaling factor
tie_weights:        bool       = True      # Share embedding and LM-head weights
padding_idx:        int | None = None      # Embedding padding index
lm_head_chunk_size: int        = 2048      # Chunk size for the "chunked" cross-entropy
loss_fn:            str        = "chunked" # "chunked" | "liger" | "auto"  (see §7)
aux_loss_weight:    float      = 0.0       # Weight of the auxiliary term (inert: predictors train in the main forward)
```

- `d_ff=None` resolves to `((⌈8/3·d_model⌉ + 127) // 128) · 128`, the standard
  SwiGLU ~2.67× width rounded to a multiple of 128.
- `loss_fn="liger"` requires the Triton fused cross-entropy kernel; if Triton is
  not available the constructor raises a clear `RuntimeError` pointing you to
  `loss_fn="chunked"`.
- `aux_loss_weight` is kept for signature compatibility; the auxiliary term is an
  inert zero because `win_gate` and `α` are trained directly in the main forward
  (not via an extra loss). Leaving it at `0.0` is correct.

### Forward signature & return value

```python
out = model(
    input_ids,                       # packed [total_len] or [B, T]
    cu_seqlens=None,                 # int32 offsets [num_seqs + 1] for the packed path
    max_seqlen=None,                 # longest sequence in the batch (defaults to input_ids.shape[-1])
    labels=None,                     # packed targets; -100 is ignored
    gradient_checkpointing=False,    # recompute blocks in backward to save activation memory
    padding_mask=None,               # optional bool mask
)
```

The forward **always returns a dict**:

| key        | with `labels`                  | without `labels`         |
|------------|--------------------------------|--------------------------|
| `loss`     | scalar `main + aux_weight·aux` | `None`                   |
| `logits`   | `None` (loss computed fused)   | `[*, vocab_size]`        |
| `aux_loss` | detached aux term (zero)       | `None`                   |

> When `labels` is given, `logits` is `None` **by design**: the loss is fused so
> the full `[tokens, vocab]` logits tensor is never materialised, which is the
> main memory win at large vocabularies.

### Helper methods

```python
DSALTLMHeadModel.from_config(cfg)            # build from a DSALTConfig
model.num_parameters(trainable_only=True)    # parameter count
```

---

## 4. `DSALTConfig`, serializable configuration

`dsalt.model.DSALTConfig` is a dataclass holding **every** model argument, so an
experiment's configuration can be saved and reloaded reproducibly.

```python
from dsalt.model import DSALTConfig, DSALTLMHeadModel

cfg = DSALTConfig(
    vocab_size=50257, d_model=512, n_layers=6, n_heads=8,
    n_min=64, n_max=256, k_lmk=16, max_seq_len=1024,
    # any optional field from §3 is also accepted, e.g.:
    loss_fn="chunked", lm_head_chunk_size=2048, tie_weights=True,
)
model = DSALTLMHeadModel.from_config(cfg)

cfg.save("config.json")                  # → JSON
cfg2 = DSALTConfig.load("config.json")   # ← JSON
d    = cfg.to_dict()                      # plain dict
cfg3 = DSALTConfig.from_dict(d)           # ignores unknown keys
```

It has the same fields and defaults as the model constructor (§3), and validates
on construction:

- `d_model % n_heads == 0`
- `0 <= n_min <= n_max`
- `k_lmk >= 0`
- `loss_fn in {"auto", "chunked", "liger"}`

A clear `ValueError` is raised otherwise.

---

## 5. `DSALTAttention`, the sparse attention module

`dsalt.modules.DSALTAttention`, multi-head attention over `W(i) ∪ L(i)` with
RoPE/YaRN positions.

### Constructor

```python
d_model:     int
n_heads:     int
n_min:       int
n_max:       int
k_lmk:       int
max_seq_len: int
dropout:     float = 0.0
yarn_scale:  float = 1.0
layer_idx:   int   = 0
```

### Learnable parameters

- **`win_gate`**, `nn.Linear(d_model, 1)`: the §4.2 window-size predictor.
  Trained through the *soft window edge* (`_soft_window_logbias`): inside the
  window the bias is exactly `0` (hard core), and only the last `win_edge` keys
  before the boundary carry `log σ((w̃ − d)/τ_win)`, so `∂/∂w̃` is non-zero only on
  the boundary band.
- **`alpha_w`**, a per-head vector, initialised so that `σ(alpha_w) ≈ 0.6`. It
  balances value-energy vs. context-energy in the landmark score and is trained
  through the *soft landmark re-weight* (`_soft_landmark_logbias`): the top-k
  selection is detached, but each admitted landmark's logit is biased by
  `log σ(s_j(α)/τ_lmk)`.

### Fixed temperatures (module attributes)

| attribute  | value | meaning                                                       |
|------------|-------|---------------------------------------------------------------|
| `tau_win`  | `1.0` | sharpness of the differentiable window border                 |
| `tau_lmk`  | `2.0` | sharpness of the α-dependent landmark re-weight               |
| `win_edge` | `4`   | width (in tokens) of the soft boundary band; core is hard     |

### Two execution paths

- **Packed** (`cu_seqlens` provided): the Triton training kernel
  `dsalt_triton_train_attention` (or the inference kernel
  `dsalt_triton_attention`), with a differentiable masked-SDPA fallback when
  Triton is unavailable.
- **Batched** (`[B, T, d]`): masked SDPA, used for inference.

In `eval` mode the module caches the dense attention matrix of the first sequence
in `_last_P`, which the trainer consumes for the rank/entropy/attention-sink
diagnostics.

---

## 6. `DSALTTransformerBlock` & `SwiGLUFFN`

- **`dsalt.modules.DSALTTransformerBlock`**, one pre-norm block:

  ```
  x = x + DSALTAttention(RMSNorm(x))
  x = x + SwiGLUFFN(RMSNorm(x))
  ```

  Constructed as `DSALTTransformerBlock(d_model, n_heads, n_min, n_max, k_lmk,
  max_seq_len, d_ff, dropout=0.0, yarn_scale=1.0, layer_idx=0)`. All
  architectural hyperparameters are inherited from `DSALTLMHeadModel`. Its
  `forward` returns `(x, aux)` where `aux` is the inert zero auxiliary term.

- **`dsalt.modules.SwiGLUFFN`**, gated SwiGLU feed-forward,
  `SwiGLUFFN(d_model, d_ff, dropout=0.0)`: `down(silu(gate(x)) * up(x))`.

---

## 7. The loss functions

Set via `loss_fn` on the model (or config). All three compute the same causal LM
cross-entropy with `-100` ignored; they differ only in memory/speed trade-offs.

| `loss_fn`   | what it does                                                                 | when to use                                                        |
|-------------|------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `"chunked"` | Pure-PyTorch cross-entropy over the LM head in chunks of `lm_head_chunk_size`; never materialises the full `[tokens, vocab]` logits. **Default.** | The memory-safe default; best on tighter-VRAM GPUs (e.g. T4).     |
| `"liger"`   | Liger fused linear cross-entropy (Triton); no logits materialisation.        | Big-VRAM GPUs (A100+) where the fused kernel wins. Needs Triton.  |
| `"auto"`    | Benchmarks chunked vs. liger **once** per `(device, vocab)` and caches the winner (same one-shot pattern as the kernel autotune). | When you don't want to choose; picks by speed per GPU.            |

Notes:
- `"chunked"` with a large chunk is fast on T4 but materialises `[chunk, vocab]`
  fp32 logits, a larger memory peak. Lower `lm_head_chunk_size` to trade speed
  for memory.
- The chunked path passes fp16 logits straight to `F.cross_entropy` (which already
  upcasts to fp32 internally for the log-softmax), so there is no redundant fp32
  copy.

---

## 8. `DSALTTrainer`, the training loop

`dsalt.training.DSALTTrainer`, single- and multi-GPU (DDP) training with AMP
autodetect, cosine LR schedule, checkpointing, and rich diagnostics. It expects
**packed** batches shaped `(input_ids, labels, cu_seqlens, max_seqlen)` (see §9).

### Required parameters

```python
model:        nn.Module
train_loader: DataLoader
val_loader:   DataLoader
```

### Optional parameters (defaults match the constructor exactly)

```python
# Distributed identity (filled in by the launcher for multi-GPU)
rank:        int = 0
local_rank:  int = 0
world_size:  int = 1            # > 1 wraps the model in DistributedDataParallel

# Optimisation (AdamW)
lr:            float = 3e-4     # base LR; the alpha_w parameters get 2× this LR
weight_decay:  float = 0.1
max_grad_norm: float = 0.5      # gradient clipping (0 disables)
grad_accum:    int   = 1        # micro-steps accumulated per optimizer step

# Cosine schedule with linear warm-up (decays to 0.1× base LR)
warmup_steps: int = 1000
total_steps:  int = 10000

# Logging / checkpointing
log_every:  int = 100
val_every:  int = 500
save_every: int = 1000
save_dir:   str = "./checkpoints_dsalt"

# Precision & performance
mixed_precision:        str  = "auto"   # "auto" | "bf16" | "fp16" | "none"  (§10)
gradient_checkpointing: bool = False    # recompute blocks in backward to save memory
compile_model:          bool = False    # apply torch.compile (after the DDP wrap)  (§11)
ddp_backend:            str  = "nccl"
seed:                   int  = 42        # per-rank seed = seed + rank
```

### Public methods

```python
trainer.train()                 # run the full training loop
trainer.load_checkpoint(path)   # resume model + optimizer + scheduler + step + history
```

---

## 9. Data format: packed sequences

The training path is optimised for **packed** batches: many variable-length
sequences concatenated along the token axis, with an offset table.

A batch is `(input_ids, labels, cu_seqlens, max_seqlen)`:

- `input_ids`, `[total_len]` (or `[B, T]` for the inference path).
- `labels`, same shape as `input_ids`; positions set to `-100` are ignored by
  the loss (use this for padding / prompt masking).
- `cu_seqlens`, `int32` tensor `[num_seqs + 1]`, the cumulative sequence offsets
  (`cu_seqlens[0] = 0`, `cu_seqlens[-1] = total_len`), exactly the FlashAttention
  convention. Sequence `b` occupies `input_ids[cu_seqlens[b] : cu_seqlens[b+1]]`.
- `max_seqlen`, the longest sequence length in the batch (an `int`).

Your `DataLoader` must yield this 4-tuple.

---

## 10. Mixed precision & device portability

With `mixed_precision="auto"` the trainer picks the autocast dtype from the GPU's
**compute capability**, not from `torch.cuda.is_bf16_supported()`, which returns
`True` even on `sm_75`/T4 where bf16 is software-emulated and does not compile:

- `sm_80+` (A100 / H100 / L4 / …) → **bf16** (no GradScaler needed)
- below `sm_80` (e.g. T4 `sm_75`) → **fp16** (with a `GradScaler`)
- CPU → no autocast

You can force `"bf16"`, `"fp16"`, or `"none"` explicitly. TF32 matmul/cuDNN are
enabled (a real win on Ampere+, inert on T4). Nothing in the kernels is hard-coded
for a specific GPU: tile sizes, warp counts, pipeline stages and shared-memory
budgets are all resolved at runtime per device.

---

## 11. Distributed training (DDP) & torch.compile

- **DDP only** (no FSDP in this release). When `world_size > 1` the trainer wraps
  the model in `DistributedDataParallel` with `gradient_as_bucket_view=True` and
  uses `no_sync()` during gradient accumulation (all-reduce fires only on the last
  micro-step). Launch one process per GPU:

  ```bash
  torchrun --nproc_per_node=2 your_train_script.py
  ```

  and pass `rank` / `local_rank` / `world_size` through to the trainer.

- **torch.compile** (`compile_model=True`). The Triton kernels are marked opaque to
  Dynamo, so `torch.compile` fuses all the eager code around them (RoPE, selectors,
  RMSNorm, residuals, SwiGLU FFN, loss). It is applied **after** the DDP wrap, and
  under DDP the trainer sets `torch._dynamo.config.optimize_ddp = False` to keep the
  backward graph intact across the custom autograd Function (otherwise the loss
  loses its `grad_fn`). `torch.compile` is a pure performance knob: any hard compile
  failure falls back to eager and never affects correctness. See
  [DESIGN_NOTES.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/DESIGN_NOTES.md)
  §2 for the full rationale.

---

## 12. Diagnostics & checkpointing

### Diagnostics (logged every `log_every` steps)

Besides loss, perplexity, LR, it/s and tok/s, the trainer computes a suite of
per-layer **representation-health** metrics in `eval` mode from the cached
attention matrices (`_last_P`):

- `σ²`, second singular value of the attention matrix (rank-collapse signal)
- `eff_rank`, effective rank of the representations
- `res_norm`, residual-stream norm ratio
- `attn_entropy`, attention entropy `H`
- `noise_norm`, noise propagation under a token perturbation
- `token_dist`, token distinguishability
- `head_spec_std`, head-specialisation spread
- `attn_sink`, attention mass on token 0 (sink)
- `alpha_{min,mean,max}` and per-head `alpha`
- `win_{min,mean,max}` and per-layer window sizes
- `scan_block_max`, `scan_ratio`, the per-block max window `max(w̃)` and realised
  scan ratio (this is what makes the early throughput dip = head specialisation
  visible; it saturates as heads specialise)
- `oow_mass_per_layer`, out-of-window attention mass

### Checkpointing

The trainer saves to `save_dir`:

- `checkpoint_best.pt`, on a new best validation perplexity
- `checkpoint_step_<n>.pt`, every `save_every` steps
- `checkpoint_final.pt`, at the end

Each checkpoint stores the **unwrapped** model state (DDP/compile peeled off),
optimizer state, scheduler state, best validation perplexity, and the full metric
history. Resume with `trainer.load_checkpoint(path)`.

---

## 13. Low-level Triton kernels

For advanced / packed use (CUDA + Triton required). The **inference** kernel:

```python
from dsalt.kernels import dsalt_triton_attention

out = dsalt_triton_attention(
    q, k, v,        # [total_len, n_heads, head_dim]
    lmk_indices,    # selected landmark indices per head
    lmk_bias,       # log-sigmoid landmark bias (carries the α gradient in training)
    w_sizes,        # per-token window size
    cu_seqlens,     # int32 sequence offsets [num_seqs + 1]
)
```

The **training** kernel (`dsalt_triton_train_attention`, used inside
`DSALTAttention`) implements a split FlashAttention-2-style forward + backward:

- forward with online softmax (band + landmarks),
- a **key-parallel, atomic-free** `dk/dv` backward for the window band
  (`_train_bwd_dkdv_kernel`), plus a light query-parallel kernel for `dq`, the
  window-size gradient `d_w̃` and the landmark gate gradient `d_logw`.

Block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_N_BWD`, `num_warps`, `num_stages`) are
chosen **once** per `(head_dim, compute capability)` at the first launch by an
autotuner that benchmarks a small set of valid candidates (filtered by the device
shared-memory budget) and prints a debug table; if nothing can be measured it
falls back to portable heuristics. Constraint on `sm_75`: every `tl.dot` axis must
be ≥ 16, so landmark blocks are padded to 16 when needed.

When Triton is absent, `dsalt_triton_attention` (and the other Triton symbols) are
`None`, and the model uses the SDPA fallback automatically, the SDPA path mirrors
the exact same math (shared selectors), so it doubles as the kernel's correctness
reference.

---

## 14. Top-level exports

```python
from dsalt import (
    # config & model
    DSALTConfig, DSALTLMHeadModel,
    # modules
    DSALTAttention, DSALTTransformerBlock, SwiGLUFFN,
    # training
    DSALTTrainer,
    # landmark scoring (single source of the §4.3 formula)
    hybrid_scores_per_head, compute_hybrid_scores,
    select_landmarks, soft_landmark_weights, HybridEnergyLandmarkSelector,
    # dense sparse-attention helpers
    sparse_attention_forward, sparse_attention_forward_packed,
    # window / RoPE utilities
    compute_window_sizes, build_local_window_mask, build_local_window_mask_packed,
    apply_rotary_emb, build_rope_cache,
    # norm & fused CE
    RMSENorm, LigerFusedLinearCrossEntropyFunction,
    # Triton attention (None when Triton is unavailable)
    dsalt_triton_attention,
)
```

---

## 15. End-to-end example

```python
import torch
from dsalt.model import DSALTConfig, DSALTLMHeadModel
from dsalt.training import DSALTTrainer

# 1) Build the model from a (serializable) config.
cfg = DSALTConfig(
    vocab_size=50257, d_model=512, n_layers=6, n_heads=8,
    n_min=64, n_max=256, k_lmk=16, max_seq_len=1024,
    loss_fn="chunked",
)
model = DSALTLMHeadModel.from_config(cfg)

# 2) Your DataLoaders must yield packed 4-tuples:
#    (input_ids [total_len], labels [total_len], cu_seqlens [num_seqs+1] int32, max_seqlen int)
#    -100 labels are ignored by the loss.

# 3) Train (single GPU here; for DDP launch with torchrun and pass rank/local_rank/world_size).
trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=3e-4,
    total_steps=10_000,
    warmup_steps=1_000,
    grad_accum=5,
    mixed_precision="auto",
    compile_model=True,
    save_dir="./checkpoints_dsalt",
    log_every=100,
)
trainer.train()

# 4) Resume later:
# trainer.load_checkpoint("./checkpoints_dsalt/checkpoint_final.pt")
```

---

## License

Apache 2.0, see
[LICENSE](https://github.com/LeonardoCofone/dsalt-library/blob/main/LICENSE).
