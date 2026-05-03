# DSALT: Dynamic Sparse Attention with Landmark Tokens

[![PyPI](https://img.shields.io/pypi/v/dsalt)](https://pypi.org/project/dsalt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance PyTorch library implementing **DSALT** (Dynamic Sparse Attention with Landmark Tokens)—a memory-efficient sparse attention mechanism for transformers, built with Triton kernels and optimized for distributed training.

> **Install from PyPI**: `pip install dsalt`  
> **GitHub**: [dsalt-pytorch](https://github.com/LeonardoCofone/dsalt-pytorch)  
> **Paper**: [Zenodo Preprint](https://zenodo.org/records/19312827)  
> **Feature Roadmap**: See [FEATURE.md](FEATURE.md)

## 🚀 Key Features

- **Memory-Efficient Sparse Attention**: Triton-accelerated kernels (4–8× memory savings vs. dense attention)
- **Adaptive Local Windows**: Token-by-token dynamic window sizing that grows with sequence position
- **Global Landmark Tokens**: Top-k informative tokens selected per head via hybrid energy scoring
- **Production-Ready Training**: Complete trainer with mixed precision, gradient checkpointing, and validation
- **Distributed Training**: Full support for DDP and FSDP (model sharding across 2+ GPUs)
- **Numerically Verified**: CPU/GPU equivalence tests ensure correctness; gradient stability validated

### Recent Optimizations (2024)

✅ **Eliminated Silent Memory Replication**
- Landmark tensor shape: `[B, H, K]` (was `[B, H, N, K]`) — saves O(N) allocation  
- Hidden state input: `[B, N, D]` (was `[B, H, N, D]`) — eliminates `H×` copy per head  
- Combined effect: **4–8× memory reduction** in landmark computation

✅ **Fixed Correctness Issues**
- Gradient checkpointing now properly checkpoints full attention block (not lambda-wrapped)
- Backward kernel signatures cleaned: removed dead code and unused parameters  
- Distributed training fixed: `_is_main` no longer silently defined twice

✅ **Enhanced Distributed Training**
- FSDP support for 2+ GPU model sharding: `torchrun --nproc_per_node=2 train.py --fsdp`
- Gradient accumulation optimized: `no_sync()` eliminates intermediate all-reduce cost
- DataParallel removed: unsuitable overhead for sparse patterns

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Training Examples](#training-examples)
5. [API Reference](#api-reference)
6. [Testing](#testing)
7. [Citation](#citation)
8. [Contributing](#contributing)
9. [License](#license)

---

## 🛠️ Installation

### Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.0+ (for GPU acceleration; CPU fallback available)
- **Triton**: 2.0+ (optional; enables GPU kernels; CPU fallback via PyTorch)

### From PyPI
```bash
pip install dsalt
```

### From PyPI with GPU Acceleration
```bash
# Includes Triton for GPU kernels
pip install dsalt
```

### From Source
```bash
git clone https://github.com/LeonardoCofone/dsalt-pytorch.git
cd dsalt-pytorch
pip install -e .
```

### Development Setup
```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1. Minimal Example: Language Model Inference
```python
import torch
from dsalt.model import DSALTLMHeadModel

# Create a DSALT language model
model = DSALTLMHeadModel(
    vocab_size=32000,      # Size of vocabulary
    d_model=1024,          # Hidden dimension
    n_layers=24,           # Depth: 24 transformer blocks
    n_heads=16,            # Multi-head attention heads
    n_min=32,              # Minimum local window
    n_max=512,             # Maximum local window
    k_lmk=64,              # Number of landmark tokens per head
)

# Forward pass (inference)
input_ids = torch.randint(0, 32000, (1, 1024))  # [batch=1, seq_len=1024]
logits = model(input_ids)                        # [1, 1024, 32000]
print(f"Output shape: {logits.shape}")

# With labels: direct loss computation
input_ids = torch.randint(0, 32000, (4, 512))  # [batch=4, seq_len=512]
labels = torch.randint(0, 32000, (4, 512))
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()
```

### 2. Training: Single GPU
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from dsalt.model import DSALTLMHeadModel
from dsalt.training import DSALTTrainer

# Prepare dataset
vocab_size = 32000
seq_len = 512
train_dataset = TensorDataset(
    torch.randint(0, vocab_size, (1000, seq_len)),  # 1000 sequences
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Create model
model = DSALTLMHeadModel(
    vocab_size=vocab_size,
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_min=32,
    n_max=256,
    k_lmk=32,
)

# Train
trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,
    lr=3e-4,
    total_steps=10000,
    save_dir="checkpoints",
    dtype=torch.bfloat16,  # Mixed precision: BF16
    log_every=50,
)
trainer.train()
```

### 3. Training: Multi-GPU with FSDP (Fully Sharded Data Parallel)
```python
# Command: torchrun with FSDP enabled
# torchrun --nproc_per_node=2 train.py

import torch
from dsalt.training import DSALTTrainer

trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=3e-4,
    total_steps=100000,
    fsdp=True,              # ← Enable FSDP for 2+ GPU sharding
    dtype=torch.bfloat16,
    save_dir="checkpoints",
)
trainer.train()
```

---

## 🏗️ Architecture

### Overview

DSALT combines **local causal windows** (adaptive, growing with position) with **global landmark tokens** (top-k per head):

```
┌─ Local Attention ──────┬─ Global Landmarks ────┐
│ Recent N tokens        │ Top-K informative     │
│ (adaptive window)      │ (hybrid energy score) │
└───────────────────────┴──────────────────────┘
         ↓                         ↓
         └─────────────┬───────────┘
                       ↓
            Sparse Attention Output
```

### Key Components

1. **`DSALTAttention`**: Multi-head sparse attention module
   - Adaptive window size prediction per token
   - Landmark token selection (no gradient)
   - Sparse kernel computation (Triton or CPU fallback)

2. **`WindowSizePredictor`**: Learned dynamic window module
   - Predicts continuous window sizes
   - Enables attention scope to adapt to token importance
   - Regularization: entropy loss on window decisions

3. **`HybridEnergyScorer`** (in kernels): Landmark selection
   - Computes energy scores per token (norm-based)
   - Z-score normalization
   - Top-k selection per head
   - Excludes tokens in local window (redundancy-aware)

4. **`DSALTTransformer`**: Decoder-only stack
   - Pre-norm RMSNorm for stability
   - SwiGLU feed-forward networks
   - Residual connections and dropout

5. **Sparse Attention Kernel** (Triton)
   - Fused forward pass: avoids materializing full attention matrix
   - Backward pass: gradient stability for Q, K, V
   - CPU fallback: functional equivalence on devices without GPU

### Memory Profile

**Dense Attention (standard transformer)**:
- Attention matrix: O(N²) memory, O(N²) compute

**DSALT Sparse Attention**:
- Local window: O(w·N) memory (w = adaptive window)
- Landmarks: O(K·N) memory (K = constant landmark count, independent of N)
- Total: O((w+K)·N) ≪ O(N²) for long sequences

---

## 🎯 Training

### Configuration

```python
trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    
    # Optimization
    lr=3e-4,
    weight_decay=0.1,
    max_grad_norm=1.0,
    grad_accum=2,                  # Gradient accumulation: effective batch 2x
    
    # Schedule
    warmup_steps=500,
    total_steps=100_000,
    
    # Checkpointing & Logging
    save_every=1000,
    log_every=50,
    val_every=500,
    save_dir="checkpoints",
    
    # Precision & Device
    dtype=torch.bfloat16,          # Mixed precision training
    device=torch.device("cuda"),
    
    # Parallelism (choose ONE)
    ddp=False,                     # Standard DDP
    fsdp=True,                     # FSDP: model sharding
    fsdp_cpu_offload=False,        # CPU offload (slow, for very large models)
    
    # Memory Optimization
    gradient_checkpointing=True,   # Gradient checkpointing: saves activation memory
    
    # Regularization
    window_reg_coef=0.01,          # Window entropy regularization
)

trainer.train()
```

### Training with Gradient Accumulation + Distributed

```bash
# 2 GPUs, batch=4 per GPU, accumulate 2 steps = effective batch 16
torchrun --nproc_per_node=2 train.py \
    --batch_size 4 \
    --grad_accum 2 \
    --fsdp true \
    --dtype bfloat16
```

---

## 📚 API Reference

### Core Classes

#### `DSALTLMHeadModel`
Language model wrapper for autoregressive training/inference.

```python
model = DSALTLMHeadModel(
    vocab_size=32000,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_min=32,           # Min window size
    n_max=512,          # Max window size
    k_lmk=64,           # Landmarks per head
    norm_eps=1e-6,
    dropout=0.1,
    bias=False,
)

# Forward: returns logits or (loss, logits) if labels provided
outputs = model(input_ids, labels=None)
logits = outputs.logits
```

#### `DSALTTransformer`
Core transformer architecture (without LM head).

```python
transformer = DSALTTransformer(
    d_model=1024,
    n_heads=16,
    n_layers=24,
    n_min=32,
    n_max=512,
    k_lmk=64,
)

# Forward: returns [batch, seq_len, d_model]
x = transformer(input_embeddings)
```

#### `DSALTAttention`
Single attention layer.

```python
attn = DSALTAttention(
    d_model=1024,
    n_heads=16,
    n_min=32,
    n_max=512,
    k_lmk=64,
    dropout=0.1,
    gradient_checkpointing=False,
)

# Returns (output, window_sizes) if return_window=True
out, _ = attn(x, x_prev=None, return_window=True)
```

#### `DSALTTrainer`
Training loop with mixed precision, DDP/FSDP, checkpointing.

```python
trainer = DSALTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=3e-4,
    total_steps=100_000,
    dtype=torch.bfloat16,
    ddp=False,
    fsdp=True,
    gradient_checkpointing=True,
)

trainer.train()  # Blocking: runs until total_steps
```

### Kernel Functions

#### `dsalt_attention(Q, K, V, window_sizes, landmark_idx)`
Low-level sparse attention computation.

```python
from dsalt.kernels import dsalt_attention

# Q, K, V: [batch, n_heads, seq_len, d_head]
# window_sizes: [batch, n_heads, seq_len] int32
# landmark_idx: [batch, n_heads, k_landmarks] int32
# Returns: [batch, n_heads, seq_len, d_head]

out = dsalt_attention(Q, K, V, window_sizes, landmark_idx)
```

#### `compute_hybrid_energy_scores(X, WV, window_sizes, k, alpha)`
Compute landmark scores and select top-k.

```python
from dsalt.kernels import compute_hybrid_energy_scores

# X: [batch, seq_len, d_model]
# WV: [n_heads, d_model, d_head]
# Returns: [batch, n_heads, k]

landmark_idx = compute_hybrid_energy_scores(
    X=hidden_states,
    WV=value_projections,
    window_sizes=window_sizes,
    k=64,
    alpha=torch.tensor([0.6] * n_heads),
)
```

---

## 🧪 Testing

### Run Full Test Suite

```bash
# All tests with coverage
make test-cov

# Or directly with pytest
pytest tests/ -v --cov=dsalt --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Modules

- `tests/test_sparse_attn.py`: Attention kernel CPU/GPU equivalence, backward pass
- `tests/test_hybrid_energy.py`: Landmark scoring and selection
- `tests/test_dsalt_lm.py`: Language model wrapper, loss computation
- `tests/test_main.py`: End-to-end training smoke test

### CI/CD

Tests automatically run on:
- Push to any branch
- Pull requests
- Scheduled nightly builds

---

## 📊 Performance & Benchmarks

### Memory Usage (Approximate)

Model: `d_model=1024, n_heads=16, n_layers=12, seq_len=1024, batch=4`

| Attention Type | Memory (GB) | Relative |
|:---|---:|---:|
| Dense (Q×K^T) | ~3.5 | 1.0× |
| FlashAttention 2 | ~1.8 | 0.51× |
| **DSALT** | ~0.6 | **0.17×** |

### Compute Efficiency

- **Forward**: ~95% of time spent in Triton kernels (minimal Python overhead)
- **Backward**: Full gradient support with automatic differentiation
- **Mixed Precision (BF16)**: 1.5–2× speedup vs. FP32 on modern GPUs

---

## 📖 Citation

If DSALT is useful in your research, please cite:

```bibtex
@article{dsalt2024,
  title={Noise Accumulation and Rank Collapse in Dense Self-Attention: DSALT},
  author={Leonardo Cofone},
  journal={Zenodo Preprint},
  year={2026},
  url={https://zenodo.org/records/19312827},
  note={Dynamic Sparse Attention with Landmark Tokens}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Performance tuning (Triton kernel optimization)
- Additional model architectures (encoder, encoder-decoder)
- New training strategies and samplers
- Documentation and tutorials
- Bug reports and fixes

---

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Triton**: GPU kernel framework by OpenAI
- **FlashAttention**: Inspiration for fused kernel design (Dao et al.)
- **PyTorch**: Deep learning framework and distributed training infrastructure

---

## 📞 Support & Questions

- **Issues**: [GitHub Issues](https://github.com/LeonardoCofone/dsalt-pytorch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LeonardoCofone/dsalt-pytorch/discussions)
- **Paper**: [Zenodo Preprint](https://zenodo.org/records/19312827)

---

**Last Updated**: May 2026  
**Version**: 1.2.0  
**Status**: ✅ Production-Ready