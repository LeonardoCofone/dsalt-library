# DSALT: Dynamic Sparse Attention with Landmark Tokens

[![PyPI](https://img.shields.io/pypi/v/dsalt)](https://pypi.org/project/dsalt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

DSALT is a high‑performance PyTorch library that implements **Dynamic Sparse Attention with Landmark Tokens** – a memory‑efficient attention mechanism for transformers. It relies on Triton kernels and supports distributed training.

> **Install**: `pip install dsalt`  
> **Source**: <https://github.com/LeonardoCofone/dsalt-library>  
> **Paper**: <https://zenodo.org/records/19312826>  
> **Feature guide**: See `FEATURE.md` here: <https://github.com/LeonardoCofone/dsalt-library/blob/main/FEATURE.md>

## 🚀 Key Features

- **Memory‑efficient sparse attention** – Triton‑accelerated kernels provide 4–8× memory savings compared to dense attention.
- **Adaptive local windows** – Token‑wise window sizes that grow with sequence position.
- **Global landmark tokens** – Top‑k informative tokens per head selected via a hybrid energy scoring function.
- **Production‑ready training** – Mixed‑precision, gradient checkpointing, and validation support.
- **Distributed training** – Full DDP and FSDP support for multi‑GPU setups.
- **Numerical verification** – CPU/GPU equivalence tests and gradient stability checks.

---

## 📋 Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture)
4. [Training & Generation](#training--generation)
5. [API Reference](#api-reference)
6. [Hyperparameter Guide](#hyperparameter-guide)
7. [Testing](#testing)
8. [Performance & Benchmarks](#performance--benchmarks)
9. [Citation](#citation)
10. [Contributing](#contributing)
11. [License](#license)

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU) – CPU fallback is available
- Triton 2.0+ (optional, enables GPU kernels)

### From PyPI
```bash
pip install dsalt
```

### From source
```bash
git clone https://github.com/LeonardoCofone/dsalt-library.git
cd dsalt-pytorch
pip install -e .
```

### Development setup
```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1. Language‑model inference
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
)

input_ids = torch.randint(0, 32000, (1, 1024))  # [batch, seq_len]
logits = model(input_ids)                     # [1, 1024, 32000]
print(logits.shape)

# With labels – loss is computed internally
labels = torch.randint(0, 32000, (1, 1024))
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()
```

### 2. Single‑GPU training
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from dsalt.model import DSALTLMHeadModel
from dsalt.training import DSALTTrainer

vocab_size = 32000
seq_len = 512
train_dataset = TensorDataset(
    torch.randint(0, vocab_size, (1000, seq_len))
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = DSALTLMHeadModel(
    vocab_size=vocab_size,
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_min=32,
    n_max=256,
    k_lmk=32,
)

trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,
    lr=3e-4,
    total_steps=10_000,
    save_dir="checkpoints",
    dtype=torch.bfloat16,
    log_every=50,
)
trainer.train()
```

### 3. Multi‑GPU with DataParallel
```python
import torch
import torch.nn as nn
from dsalt.model import DSALTLMHeadModel
from dsalt.training import DSALTTrainer

model = DSALTLMHeadModel(...).to("cuda")
model = nn.DataParallel(model)  # uses all available GPUs

trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,
    lr=3e-4,
    total_steps=100_000,
    dtype=torch.bfloat16,
    save_dir="checkpoints",
)
trainer.train()
```

### 4. Multi‑GPU with FSDP (model sharding)
```bash
torchrun --nproc_per_node=2 train.py
```
Then configure the trainer with `fsdp=True`.

---

## 🏗️ Architecture Overview

DSALT combines **local causal windows** (adaptive per token) with **global landmark tokens** (top‑k per head):
```
┌─ Local window (adaptive) ──┬─ Global landmarks ──┐
│ Recent N tokens            │ Top‑K informative │
│ (window size grows)        │ tokens per head   │
└────────────────────────────┴────────────────────┘
                ↓                     ↓
            Sparse attention output
```
Key components:
1. `DSALTAttention` – multi‑head sparse attention with adaptive windows and landmark selection.
2. `WindowSizePredictor` – learns per‑token window sizes.
3. `HybridEnergyScorer` (kernel) – computes landmark scores.
4. `DSALTTransformer` – stack of attention + feed‑forward layers.
5. Triton kernels – fused forward and backward passes for speed and memory efficiency.

---

## 🎯 Training & Generation

See the code snippets above for full training loops. The `DSALTTrainer` handles:
- Mixed‑precision (BF16 default)
- Gradient checkpointing
- Learning‑rate warm‑up and cosine decay
- Optional window‑entropy regularisation (`window_reg_coef`)
- Checkpointing and logging utilities

---

## 📚 API Reference (excerpt)

```python
from dsalt.model import DSALTLMHeadModel
model = DSALTLMHeadModel(vocab_size=32000, d_model=1024, n_layers=24,
                         n_heads=16, n_min=32, n_max=512, k_lmk=64)
logits, windows = model(input_ids, return_window=True)
```
Low‑level kernel call:
```python
from dsalt.kernels import dsalt_attention
out = dsalt_attention(Q, K, V, window_sizes, landmark_idx)
```

---

## 📊 Performance & Benchmarks (May 2026)

| Attention type | Approx. memory (GB) | Relative speed |
|----------------|--------------------|----------------|
| Dense (O(N²))  | ~3.5               | 1.0× |
| FlashAttention 2| ~1.8               | 0.5× |
| **DSALT**      | ~0.6               | **0.17×** |

---

## 📖 Hyperparameter Guide

All hyperparameters are documented in `FEATURE.md`. Typical configurations are provided for:
- **Mobile / Edge** – tiny models, low memory.
- **Consumer GPU** – e.g., RTX 4090, 24 GB.
- **Enterprise** – H100 80 GB, optional FSDP.
- **Research** – multi‑node, large models.

---

## 🧪 Testing

```bash
make test-cov          # Full test suite with coverage report
pytest tests/ -v       # Run tests directly
```
Key test modules:
- `tests/test_sparse_attn.py` – kernel equivalence and backward.
- `tests/test_hybrid_energy.py` – landmark scoring.
- `tests/test_dsalt_lm.py` – language‑model wrapper.
- `tests/test_main.py` – end‑to‑end smoke test.

---

## 📄 License

See here: <https://github.com/LeonardoCofone/dsalt-library/blob/main/LICENSE>

---

## 🤝 Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for guidelines. Areas where help is especially valuable:
- Triton kernel optimisation
- New model architectures (encoder, encoder‑decoder)
- Additional training strategies and samplers
- Documentation and tutorials
- Bug reports and fixes

---

## 📞 Support & Questions

- **Issues**: <https://github.com/LeonardoCofone/dsalt-library/issues>
- **Discussions**: <https://github.com/LeonardoCofone/dsalt-library/discussions>
- **Paper**: <https://zenodo.org/records/19312826>

---

**Last Updated**: May 2026
