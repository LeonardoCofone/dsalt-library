# DSALT Features & Hyperparameters Guide

A complete reference for every public component, its real constructor signature,
defaults, and recommended usage. Everything documented here is verified against
the source, no placeholders, no planned-but-missing APIs.

---

## Overview

**DSALT** (Dynamic Sparse Attention with Landmark Tokens) replaces dense
`O(N²)` attention with the union of two sparse sets per query:

- an **adaptive local causal window** `W(i)` (§4.2), and
- a small set of **global landmark tokens** `L(i)` selected by a learnable
  hybrid-energy score (§4.3).

The library is GPU-portable by design: on CUDA it uses custom **Triton** kernels
(fused forward/backward with online softmax, one-shot autotuned block sizes); if
Triton is unavailable it transparently falls back to a masked **SDPA** path, so
the package stays importable and runnable everywhere (including CPU).

> **Implementation note.** In this release the local window is *frozen* to a
> constant `(n_min + n_max) // 2`, there is no learnable `WindowSizePredictor`.
> The demonstrated adaptivity is the per-head learnable `alpha` used by the


---

## `DSALTLMHeadModel`, Causal Language Model

`dsalt.model.DSALTLMHeadModel`, token embedding + a stack of
`DSALTTransformerBlock` + final RMSNorm + (optionally tied) LM head.

The optimized training path consumes **packed** sequences (concatenated tokens
plus a `cu_seqlens` offset tensor). A plain `[B, T]` tensor is accepted for
inference.

### Required parameters

```python
vocab_size:  int   # Vocabulary size
d_model:     int   # Hidden dimension; must be divisible by n_heads
n_layers:    int   # Number of transformer blocks
n_heads:     int   # Number of attention heads
n_min:       int   # Lower bound of the local window (§4.2)
n_max:       int   # Upper bound of the local window (§4.2)
k_lmk:       int   # Number of landmark tokens per query (§4.3)
max_seq_len: int   # Max length for the RoPE cache
```

### Optional parameters (defaults match the constructor exactly)

```python
d_ff:               int | None = None      # FFN hidden dim; if None → round(8/3 · d_model) up to a multiple of 128
dropout:            float      = 0.0       # Dropout on embeddings / blocks
yarn_scale:         float      = 1.0       # RoPE / YaRN positional scaling factor
tie_weights:        bool       = True      # Share embedding and LM-head weights
padding_idx:        int | None = None      # Embedding padding index
lm_head_chunk_size: int        = 2048      # Chunk size for the "chunked" cross-entropy
loss_fn:            str        = "chunked" # "chunked" (pure PyTorch) or "liger" (fused Triton kernel)
aux_loss_weight:    float      = 0.0       # Weight of the auxiliary term (inert: window is frozen)
```

`loss_fn="liger"` requires the Triton fused cross-entropy kernel; if Triton is
not available the constructor raises a clear `RuntimeError` telling you to use
`loss_fn="chunked"`.

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

| key        | with `labels`                 | without `labels`         |
|------------|-------------------------------|--------------------------|
| `loss`     | scalar `main + aux_weight·aux`| `None`                   |
| `logits`   | `None` (loss computed fused)  | `[*, vocab_size]`        |
| `aux_loss` | detached aux term             | `None`                   |

Helpers: `DSALTLMHeadModel.from_config(cfg)` builds from a `DSALTConfig`;
`model.num_parameters(trainable_only=True)` counts parameters.

---

## `DSALTConfig`, Serializable Configuration

`dsalt.model.DSALTConfig` is a dataclass holding every model argument so an
experiment can be saved and reloaded.

```python
from dsalt.model import DSALTConfig, DSALTLMHeadModel

cfg = DSALTConfig(
    vocab_size=50257, d_model=512, n_layers=6, n_heads=8,
    n_min=64, n_max=256, k_lmk=16, max_seq_len=1024,
)
model = DSALTLMHeadModel.from_config(cfg)

cfg.save("config.json")               # → JSON
cfg2 = DSALTConfig.load("config.json")
```

It validates on construction: `d_model % n_heads == 0`, `0 <= n_min <= n_max`,
`k_lmk >= 0`, and `loss_fn in {"chunked", "liger"}`.

---

## `DSALTAttention`, Sparse Attention Module

`dsalt.modules.DSALTAttention`, multi-head attention over `W(i) ∪ L(i)`.

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

The only attention-specific learnable parameter is `alpha_w`, a per-head vector
initialised so that `sigmoid(alpha_w) ≈ 0.6`; it weights value-energy vs.
context-energy in the landmark score. It is **not** a constructor flag.

### Two execution paths

- **Packed** (`cu_seqlens` provided): the Triton kernel `dsalt_triton_attention`
  in training, with a masked-SDPA fallback when Triton is unavailable.
- **Batched** (`[B, T, d]`): masked SDPA, used for inference.

In `eval` mode the module caches the dense attention matrix of the first
sequence in `_last_P`, which the trainer consumes for rank/entropy/attention-sink
metrics.

---

## `DSALTTransformerBlock` & `SwiGLUFFN`

`dsalt.modules.DSALTTransformerBlock` is one pre-norm block: RMSNorm →
`DSALTAttention` → residual, then RMSNorm → `SwiGLUFFN` → residual. All
architectural hyperparameters are inherited from `DSALTLMHeadModel`.
`SwiGLUFFN` is the gated SwiGLU feed-forward network.

---

## `DSALTTrainer`, Training Loop

`dsalt.training.DSALTTrainer`, single- and multi-GPU (DDP) training with
automatic mixed precision, cosine LR schedule, checkpointing, and a rich set of
diagnostic metrics. It expects **packed** batches shaped as
`(input_ids, labels, cu_seqlens, max_seqlen)`.

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

# Optimisation
lr:            float = 3e-4     # AdamW base LR; alpha_w parameters get 2× this LR
weight_decay:  float = 0.1
max_grad_norm: float = 0.5      # gradient clipping (0 disables)
grad_accum:    int   = 1

# Cosine schedule with linear warm-up (decays to 0.1× base LR)
warmup_steps: int = 1000
total_steps:  int = 10000

# Logging / checkpointing
log_every:  int = 100
val_every:  int = 500
save_every: int = 1000
save_dir:   str = "./checkpoints_dsalt"

# Precision & performance
mixed_precision:        str  = "auto"   # "auto" | "bf16" | "fp16" | "none"
gradient_checkpointing: bool = False
compile_model:          bool = False    # torch.compile the whole model
ddp_backend:            str  = "nccl"
seed:                   int  = 42
```

### Mixed-precision autodetect (GPU-portable)

With `mixed_precision="auto"` the trainer picks the dtype from the **compute
capability**, not from `torch.cuda.is_bf16_supported()` (which returns `True`
even on sm_75 / T4, where bf16 is software-emulated and does not compile):

- `sm_80+` (A100 / H100 / L4 / …) → **bf16**
- below sm_80 (e.g. T4 sm_75) → **fp16** (with a `GradScaler`)
- CPU → no autocast

### Distributed training

Only **DDP** is supported (`world_size > 1` wraps the model in
`DistributedDataParallel` with `gradient_as_bucket_view=True` and gradient
`no_sync()` during accumulation). There is no FSDP path in this release.

### Diagnostics

On every `log_every` step the trainer logs loss/perplexity/LR plus per-layer
representation-health metrics computed in `eval` mode: second singular value
`σ²`, effective rank, residual norm, attention entropy, noise propagation, token
distinguishability, head-specialisation spread, attention-sink mass, per-head
`alpha`, and out-of-window attention mass.

### Checkpointing

`save_dir/checkpoint_best.pt`, `checkpoint_step_<n>.pt`, and `checkpoint_final.pt`
store the (unwrapped) model, optimizer, scheduler, best validation perplexity,
and full metric history. Resume with `trainer.load_checkpoint(path)`.

---

## Low-Level Triton Kernel

For advanced/packed use only (CUDA + Triton required):

```python
from dsalt.kernels import dsalt_triton_attention

out = dsalt_triton_attention(
    q, k, v,        # [total_len, n_heads, head_dim]
    lmk_indices,    # selected landmark indices per head
    lmk_bias,       # log-sigmoid landmark bias (carries the alpha gradient)
    w_sizes,        # per-token window size
    cu_seqlens,     # int32 sequence offsets [num_seqs + 1]
)
```

Block sizes (`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`) are chosen **once**
per `(head_dim, GPU)` at the first launch by an autotuner that benchmarks a small
set of valid candidates (filtered by shared-memory budget) and prints a debug
table; if nothing can be measured it falls back to portable heuristics. When
Triton is absent, `dsalt_triton_attention` and the other Triton symbols are
`None`, and the model uses the SDPA fallback automatically.

---

## Installation extras

```bash
pip install dsalt                 # core: torch
pip install "dsalt[triton]"       # + Triton GPU kernels (Linux/CUDA)
pip install "dsalt[dev]"          # + black / isort / flake8 / mypy / pytest
```

Triton is declared `sys_platform != "win32"`: on Windows the SDPA fallback is
used.

---

## Performance (illustrative)

DSALT trades a small constant overhead for asymptotic memory savings versus dense
attention, growing with sequence length: each query attends to a bounded window
plus `k_lmk` landmarks instead of the full `O(N²)` set. Exact numbers depend on
`n_min`, `n_max`, `k_lmk`, sequence length, GPU, and dtype, benchmark on your
own configuration rather than relying on a single headline ratio.

---

## License

Apache 2.0, see
<https://github.com/LeonardoCofone/dsalt-library/blob/main/LICENSE>.
