# DSALT: Dynamic Sparse Attention with Landmark Tokens

[![PyPI](https://img.shields.io/pypi/v/dsalt)](https://pypi.org/project/dsalt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**DSALT** is a PyTorch library implementing *Dynamic Sparse Attention with
Landmark Tokens*, a memory-efficient attention mechanism for transformers. Each
query attends to an **adaptive local causal window** plus a small set of
**global landmark tokens**, instead of the full `O(N²)` set. On CUDA it runs
custom **Triton** kernels; everywhere else it falls back to a masked **SDPA**
path, so the package stays importable and runnable on any platform (including
CPU and Windows).

> **Install**: `pip install dsalt`
> **Source**: <https://github.com/LeonardoCofone/dsalt-library>
> **Paper**: <https://zenodo.org/records/19312826>

## ✨ Key Features

- **Sparse attention** — adaptive local causal window `∪` top-`k` landmark tokens per head (`A(i) = W(i) ∪ L(i)`).
- **Fully differentiable selectors** — the *hard* window/landmark selection stays non-differentiable, but the gradient still reaches both predictors through soft weights: a **soft window edge** trains the per-token window size, and a **soft landmark re-weight** trains the per-head balance `α`.
- **GPU-portable** — Triton kernels on CUDA, transparent SDPA fallback otherwise; AMP dtype is auto-selected from the GPU's compute capability (bf16 on `sm_80+`, fp16 on T4-class cards, none on CPU).
- **One-shot autotune** — Triton block sizes are benchmarked once per `(head_dim, GPU)` at the first launch, then reused for the whole run; portable heuristics if benchmarking is impossible.
- **Packed-sequence training** — concatenated sequences + `cu_seqlens`, fused FlashAttention-2-style forward/backward with online softmax and a key-parallel, atomic-free `dk/dv` backward.
- **Flexible loss** — memory-frugal chunked cross-entropy (default), optional Liger fused linear cross-entropy, or `"auto"` to pick the fastest per GPU.
- **DDP + torch.compile** — single- and multi-GPU via `DistributedDataParallel`, co-existing with `torch.compile`; gradient accumulation, cosine schedule with warm-up, checkpointing, and rich representation-health diagnostics.

---

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [🎯 Training](#-training)
- [📚 API Reference](#-api-reference)
- [📖 Documentation](#-documentation)
- [📄 License](#-license)
- [🤝 Contributing](#-contributing)
- [📝 Citation](#-citation)

---

## 🛠️ Installation

### Requirements
- Python **3.10+** (the codebase uses `X | None` / `tuple[...]` syntax)
- PyTorch **2.0+** (the only required dependency)
- CUDA 11.0+ for the GPU path (CPU fallback always available)
- Triton 2.0+ (optional; enables the GPU kernels, Linux/CUDA)

### From PyPI
```bash
pip install dsalt                 # core (torch only)
pip install "dsalt[triton]"       # + Triton GPU kernels
pip install "dsalt[dev]"          # + lint/type/test tooling
pip install "dsalt[all]"          # everything
```

### From source
```bash
git clone https://github.com/LeonardoCofone/dsalt-library.git
cd dsalt-library
pip install -e .
```

---

## 🚀 Quick Start

### Inference

```python
import torch
from dsalt.model import DSALTLMHeadModel

model = DSALTLMHeadModel(
    vocab_size=32000,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_min=32,
    n_max=512,
    k_lmk=64,
    max_seq_len=2048,   # required: sizes the RoPE cache
)

input_ids = torch.randint(0, 32000, (1, 1024))   # [batch, seq_len]
out = model(input_ids)                           # dict
logits = out["logits"]                           # [1, 1024, 32000]
print(logits.shape)
```

### Computing the loss

When `labels` are given the forward computes the loss internally (fused), and
returns a **dict** `{"loss", "logits", "aux_loss"}` (here `logits` is `None`
because the loss is fused to save memory):

```python
labels = torch.randint(0, 32000, (1, 1024))
out  = model(input_ids, labels=labels)
loss = out["loss"]
loss.backward()
```

### Building from a config

```python
from dsalt.model import DSALTConfig, DSALTLMHeadModel

cfg = DSALTConfig(
    vocab_size=50257, d_model=512, n_layers=6, n_heads=8,
    n_min=64, n_max=256, k_lmk=16, max_seq_len=1024,
)
model = DSALTLMHeadModel.from_config(cfg)
cfg.save("config.json")               # reload with DSALTConfig.load("config.json")
```

---

## 🏗️ Architecture Overview

Each query's attention set is the union of two sparse sets:

```
┌─ Local window W(i) (adaptive) ─┬─ Global landmarks L(i) ─┐
│  Recent causal tokens up to    │  Top-k informative      │
│  a per-token learned size      │  tokens per head        │
└────────────────────────────────┴─────────────────────────┘
                 ↓                            ↓
              Sparse attention output  over  A(i) = W(i) ∪ L(i)
```

- **Adaptive window (§4.2).** A small learned projection `win_gate` predicts a
  per-token continuous window `w̃(i) = n_min + σ(f(x_i))·(n_max − n_min)` from the
  block input. The window core is a hard mask (so the cost stays sub-quadratic),
  but a thin differentiable band at the boundary lets gradients train `win_gate`.
- **Landmark tokens (§4.3).** A per-head hybrid-energy score
  `s = α·z(‖x·W_V‖₂) + (1−α)·z(‖x‖₂)` ranks tokens; the top-`k` are admitted as
  landmarks. The selection is hard (detached), while a soft re-weight on the
  admitted tokens' logits trains the per-head balance `α = σ(α̃)`.
- **Blocks.** Pre-norm `DSALTTransformerBlock` = RMSNorm → `DSALTAttention`
  (with RoPE/YaRN positions) → residual, then RMSNorm → `SwiGLUFFN` → residual.
- **Model.** `DSALTLMHeadModel` = token embeddings → block stack → RMSNorm →
  (optionally tied) LM head, with a fused/chunked cross-entropy loss.
- **Kernels.** On CUDA, fused Triton forward/backward with online softmax and
  one-shot autotuned block sizes; a masked-SDPA path mirrors the exact same math
  on CPU / no-Triton environments and serves as the correctness reference.

For the engineering rationale (differentiable approximations, DDP + `torch.compile`
graph integrity, the key-parallel backward, and profiling evidence) see
[DESIGN_NOTES.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/DESIGN_NOTES.md).

---

## 🎯 Training

`DSALTTrainer` drives single- and multi-GPU (DDP) training. It expects **packed**
batches `(input_ids, labels, cu_seqlens, max_seqlen)`, where `cu_seqlens` is an
`int32` offset tensor of shape `[num_seqs + 1]` and `-100` labels are ignored.

```python
from dsalt.model import DSALTLMHeadModel
from dsalt.training import DSALTTrainer

model = DSALTLMHeadModel(
    vocab_size=32000, d_model=768, n_layers=12, n_heads=12,
    n_min=32, n_max=256, k_lmk=32, max_seq_len=1024,
)

trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,   # yields (ids, labels, cu_seqlens, max_seqlen)
    val_loader=val_loader,
    lr=3e-4,
    total_steps=10_000,
    warmup_steps=1_000,
    mixed_precision="auto",      # bf16 on sm_80+, fp16 on T4-class, none on CPU
    save_dir="./checkpoints_dsalt",
    log_every=100,
)
trainer.train()
```

**Multi-GPU (DDP).** Launch one process per GPU; the trainer wraps the model in
`DistributedDataParallel` when `world_size > 1`, and (optionally) applies
`torch.compile` *after* the DDP wrap:

```bash
torchrun --nproc_per_node=2 your_train_script.py
```

Only DDP is supported in this release (no FSDP). The trainer handles gradient
accumulation, gradient clipping, cosine LR decay with warm-up, AMP autodetect,
checkpointing (`checkpoint_best.pt` / `checkpoint_step_<n>.pt` /
`checkpoint_final.pt`), and per-layer representation-health metrics. Resume with
`trainer.load_checkpoint(path)`.

Every constructor argument, default, and the full metric list are documented in
[FEATURE.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/FEATURE.md).

---

## 📚 API Reference

```python
# Top-level exports
from dsalt import (
    DSALTConfig, DSALTLMHeadModel,
    DSALTAttention, DSALTTransformerBlock, SwiGLUFFN,
    DSALTTrainer,
    dsalt_triton_attention,            # None when Triton is unavailable
    hybrid_scores_per_head,            # single source of the landmark score (§4.3)
    compute_hybrid_scores, select_landmarks, soft_landmark_weights,
    HybridEnergyLandmarkSelector,
    sparse_attention_forward, sparse_attention_forward_packed,
    RMSENorm, compute_window_sizes, apply_rotary_emb, build_rope_cache,
    build_local_window_mask, build_local_window_mask_packed,
    LigerFusedLinearCrossEntropyFunction,
)
```

`q, k, v` for the low-level kernel are `[total_len, n_heads, head_dim]`;
`cu_seqlens` is the `int32` sequence-offset tensor. The complete, source-verified
signature and semantics of **every** component live in
[FEATURE.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/FEATURE.md).

---

## 📖 Documentation

- [FEATURE.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/FEATURE.md) — complete feature & hyperparameter reference (every public API, every option).
- [DESIGN_NOTES.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/DESIGN_NOTES.md) — engineering design rationale and profiling evidence.
- [STRUCTURE.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/STRUCTURE.md) — repository layout and intra-package usage map.
- [CONTRIBUTING.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/CONTRIBUTING.md) — how to contribute.

---

## 📄 License

Apache 2.0 — see
[LICENSE](https://github.com/LeonardoCofone/dsalt-library/blob/main/LICENSE).

---

## 🤝 Contributing

Contributions are welcome — see
[CONTRIBUTING.md](https://github.com/LeonardoCofone/dsalt-library/blob/main/CONTRIBUTING.md).
Especially valuable: Triton kernel optimisation, new architectures (encoder /
encoder-decoder), additional training strategies, documentation, and bug fixes.

- **Issues**: <https://github.com/LeonardoCofone/dsalt-library/issues>
- **Discussions**: <https://github.com/LeonardoCofone/dsalt-library/discussions>

---

## 📝 Citation

If you use DSALT in your research, please cite the paper:

```bibtex
@software{dsalt,
  author  = {Cofone, Leonardo},
  title   = {DSALT: Dynamic Sparse Attention with Landmark Tokens},
  url     = {https://github.com/LeonardoCofone/dsalt-library},
  note    = {https://zenodo.org/records/19312826},
}
```
