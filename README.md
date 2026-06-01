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

- **Sparse attention**, local adaptive window `∪` top-`k` landmark tokens per head.
- **GPU-portable**, Triton kernels on CUDA, transparent SDPA fallback otherwise; correct AMP autodetect across GPU generations (bf16 only where natively supported, fp16 on T4-class cards).
- **One-shot autotune**, Triton block sizes are benchmarked once per `(head_dim, GPU)` at the first launch, then reused for the whole run; heuristic fallback if benchmarking is impossible.
- **Packed-sequence training**, concatenated sequences + `cu_seqlens`, fused forward/backward with online softmax.
- **Fused cross-entropy**, optional Liger fused linear cross-entropy, or a memory-frugal chunked pure-PyTorch loss.
- **DDP training**, single- and multi-GPU via `DistributedDataParallel`, gradient accumulation, cosine schedule with warm-up, checkpointing, and rich representation-health diagnostics.

---

## 📋 Table of Contents
- [DSALT: Dynamic Sparse Attention with Landmark Tokens](#dsalt-dynamic-sparse-attention-with-landmark-tokens)
  - [✨ Key Features](#-key-features)
  - [📋 Table of Contents](#-table-of-contents)
  - [🛠️ Installation](#️-installation)
    - [Requirements](#requirements)
    - [From PyPI](#from-pypi)
    - [From source](#from-source)
  - [🚀 Quick Start](#-quick-start)
    - [Inference](#inference)
    - [Computing the loss](#computing-the-loss)
    - [Building from a config](#building-from-a-config)
  - [🏗️ Architecture Overview](#️-architecture-overview)
  - [🎯 Training](#-training)
    - [Mixed precision](#mixed-precision)
    - [Multi-GPU (DDP)](#multi-gpu-ddp)
  - [📚 API Reference](#-api-reference)
  - [📖 Hyperparameter Guide](#-hyperparameter-guide)
  - [📄 License](#-license)
  - [🤝 Contributing](#-contributing)
  - [📝 Citation](#-citation)

---

## 🛠️ Installation

### Requirements
- Python **3.10+** (the codebase uses `X | None` / `tuple[...]` syntax)
- PyTorch 2.0+
- CUDA 11.0+ for the GPU path (CPU fallback always available)
- Triton 2.0+ (optional; enables the GPU kernels, Linux/CUDA)

### From PyPI
```bash
pip install dsalt                 # core
pip install "dsalt[triton]"       # + Triton GPU kernels
pip install "dsalt[dev]"          # + lint/type/test tooling
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
    max_seq_len=2048,   # required
)

input_ids = torch.randint(0, 32000, (1, 1024))   # [batch, seq_len]
out = model(input_ids)                           # dict
logits = out["logits"]                           # [1, 1024, 32000]
print(logits.shape)
```

### Computing the loss

The forward computes the loss internally (fused) when `labels` are given, and
returns a **dict** `{"loss", "logits", "aux_loss"}`:

```python
labels = torch.randint(0, 32000, (1, 1024))
out  = model(input_ids, labels=labels)
loss = out["loss"]          # logits is None here (loss is fused)
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
cfg.save("config.json")               # reload with DSALTConfig.load(...)
```

---

## 🏗️ Architecture Overview

DSALT combines a per-token **local causal window** with **global landmark
tokens** selected per head:

```
┌─ Local window (adaptive) ──┬─ Global landmarks ──┐
│  Recent tokens up to       │  Top-k informative  │
│  window size               │  tokens per head    │
└────────────────────────────┴─────────────────────┘
                 ↓                      ↓
              Sparse attention output  (W(i) ∪ L(i))
```

Components:
1. `DSALTAttention`, multi-head sparse attention over `W(i) ∪ L(i)` with RoPE/YaRN positions.
2. `hybrid_scores_per_head`, the single source of the hybrid-energy landmark score (§4.3), shared by both the SDPA path and the Triton kernel.
3. `DSALTTransformerBlock` / `SwiGLUFFN`, pre-norm block with a gated SwiGLU FFN.
4. `DSALTLMHeadModel`, embeddings + block stack + RMSNorm + (tied) LM head.
5. Triton kernels, fused forward (`dsalt_triton_attention`) and backward with online softmax and one-shot autotuned block sizes.

> **Note.** The local window is frozen to `(n_min + n_max) // 2` in this release
> (no learnable window predictor). The learned adaptivity is the per-head
---

## 🎯 Training

`DSALTTrainer` drives single- and multi-GPU (DDP) training. It expects **packed**
batches: `(input_ids, labels, cu_seqlens, max_seqlen)`, where `cu_seqlens` is an
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

### Mixed precision

`mixed_precision="auto"` selects the dtype from the GPU's **compute capability**:
bf16 on `sm_80+` (A100/H100/L4/…), fp16 (with a `GradScaler`) below that
(e.g. T4 sm_75), and no autocast on CPU. You can force `"bf16"`, `"fp16"`, or
`"none"` explicitly.

### Multi-GPU (DDP)

Launch one process per GPU and pass the distributed identity through; the trainer
wraps the model in `DistributedDataParallel` when `world_size > 1`:

```bash
torchrun --nproc_per_node=2 your_train_script.py
```

```python
trainer = DSALTTrainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    rank=rank, local_rank=local_rank, world_size=world_size,
    ddp_backend="nccl", total_steps=100_000,
)
trainer.train()
```

Only DDP is supported in this release (no FSDP). The trainer also handles
gradient accumulation, gradient clipping, cosine LR decay with warm-up,
checkpointing (`checkpoint_best/step_N/final.pt`), and per-layer
representation-health metrics. Resume with `trainer.load_checkpoint(path)`.

---

## 📚 API Reference

```python
# Top-level exports
from dsalt import (
    DSALTConfig, DSALTLMHeadModel,
    DSALTAttention, DSALTTransformerBlock, SwiGLUFFN,
    DSALTTrainer,
    dsalt_triton_attention,            # None when Triton is unavailable
    hybrid_scores_per_head,            # single source of the landmark score
    sparse_attention_forward, sparse_attention_forward_packed,
    RMSENorm, compute_window_sizes, apply_rotary_emb, build_rope_cache,
)

# Low-level Triton kernel (packed sequences, CUDA + Triton only)
from dsalt.kernels import dsalt_triton_attention
out = dsalt_triton_attention(q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens)
```

`q, k, v` are `[total_len, n_heads, head_dim]`; `cu_seqlens` is the `int32`
sequence-offset tensor. See `FEATURE.md` for the full signature and semantics of
every component.

---

## 📖 Hyperparameter Guide

Full, source-verified defaults for `DSALTLMHeadModel`, `DSALTConfig`,
`DSALTAttention`, and `DSALTTrainer` live in [`FEATURE.md`](FEATURE.md). Highlights:

| Component            | Required                                                            | Notable defaults                                                                 |
|----------------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `DSALTLMHeadModel`   | `vocab_size, d_model, n_layers, n_heads, n_min, n_max, k_lmk, max_seq_len` | `d_ff=None` (→ 8/3·d_model), `loss_fn="chunked"`, `tie_weights=True`, `yarn_scale=1.0` |
| `DSALTTrainer`       | `model, train_loader, val_loader`                                   | `lr=3e-4`, `max_grad_norm=0.5`, `warmup_steps=1000`, `mixed_precision="auto"`     |

`alpha` is a learnable per-head parameter (init `sigmoid ≈ 0.6`), not a
constructor flag. The auxiliary loss term is inert in this release (frozen
window) and kept only for signature compatibility.

---

## 📄 License

Apache 2.0, see
<https://github.com/LeonardoCofone/dsalt-library/blob/main/LICENSE>.

---

## 🤝 Contributing

Contributions are welcome, see [`CONTRIBUTING.md`](CONTRIBUTING.md). Especially
valuable: Triton kernel optimisation, new architectures (encoder /
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
  url      = {https://github.com/LeonardoCofone/dsalt-library},
  note    = {https://zenodo.org/records/19312826},
}
```
