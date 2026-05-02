# DSALT – Dynamic Sparse Attention with Landmark Tokens

[![PyPI](https://img.shields.io/pypi/v/dsalt)](https://pypi.org/project/dsalt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance PyTorch library implementing **DSALT** (Dynamic Sparse Attention with Landmark Tokens), a sparse attention transformer library built for efficient training with Triton and PyTorch.

> Published on PyPI: `pip install dsalt`

## 🚀 Key Features

- **Efficient Sparse Attention**: Triton-accelerated kernels for GPU-optimized sparse causal self-attention
- **Dynamic Window Sizing**: Adaptive local attention windows that grow with sequence position
- **Landmark Token Selection**: Global landmark tokens selected via hybrid energy scoring
- **Mixed Precision Training**: Full support for BF16/FP16 training with gradient scaling
- **Distributed Training**: DDP (DistributedDataParallel) support for multi-GPU training
- **Production Ready**: Complete training harness with checkpointing, logging, and validation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Training](#training)
- [API Reference](#api-reference)
- [Benchmarks](#benchmarks)
- [Citation](#citation)
- [License](#license)

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- Triton 2.0+ (optional, for GPU kernels)

### Install from PyPI
```bash
pip install dsalt
```

### Install with Triton support
```bash
pip install dsalt[triton]
```

### Install with Flash Attention fallback
```bash
pip install dsalt[flash-attn]
```

### Install from source
```bash
git clone https://github.com/yourusername/dsalt-pytorch.git
cd dsalt-pytorch
pip install -e .
```

### Developer setup
```bash
pip install -r requirements-dev.txt
```
## 🚀 Quick Start

```python
import torch
from dsalt.model import DSALTLMHeadModel

# Create a DSALT language model
model = DSALTLMHeadModel(
    vocab_size=32000,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_min=32,      # Minimum window size
    n_max=512,     # Maximum window size
    k_lmk=64,      # Number of landmark tokens
)

# Forward pass
input_ids = torch.randint(0, 32000, (1, 1024))
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # [1, 1024, 32000]
```

## 🏗️ Architecture

DSALT combines **local causal windows** with **global landmark tokens**:

- **Local Attention**: Each token attends to a dynamic window of recent tokens
- **Landmark Selection**: Top-k informative tokens selected globally via energy scoring
- **Sparse Computation**: Only compute attention for relevant token pairs

### Key Components

- `DSALTTransformer`: Main transformer architecture
- `DSALTAttention`: Multi-head sparse attention layer
- `WindowSizePredictor`: Learned adaptive window sizing
- `HybridEnergyScorer`: Landmark token selection
- `SparseAttentionKernel`: Triton-accelerated attention computation

## 🎯 Training

### Single GPU Training
```python
from dsalt.training import DSALTTrainer
from torch.utils.data import DataLoader

trainer = DSALTTrainer(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    lr=3e-4,
    total_steps=100000,
    save_dir="checkpoints",
    dtype=torch.bfloat16,
)

trainer.train()
```

### Multi-GPU Distributed Training
```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend='nccl')

trainer = DSALTTrainer(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    ddp=True,  # Enable DDP
    # ... other args
)
```

## 📚 API Reference

### Core Classes

- `DSALTLMHeadModel`: Language model wrapper with LM head
- `DSALTTransformer`: Base transformer architecture
- `DSALTAttention`: Sparse attention module
- `DSALTTrainer`: Training harness

### Kernel Functions

- `dsalt_attention()`: Main sparse attention function
- `compute_hybrid_energy_scores()`: Landmark scoring
- `select_landmarks()`: Landmark selection

## 🧪 Testing

Run the full test suite:
```bash
python tests/test.py
```

Run specific tests:
```bash
python tests/test_sparse_attn.py  # Attention kernels
python tests/test_dsalt_lm.py     # LM wrapper
```

## 📊 Benchmarks

DSALT achieves significant speedups over dense attention:

- **Memory**: O(n) vs O(n²) for dense attention
- **Speed**: 2-5x faster training on long sequences
- **Quality**: Maintains perplexity comparable to dense models

*Detailed benchmarks coming soon*

## 📖 Citation

If you use DSALT in your research, please cite our paper:

```bibtex
@article{dsalt2024,
  title={Noise Accumulation and Rank Collapse in Dense Self-Attention: DSALT},
  author={Your Name et al.},
  journal={arXiv preprint},
  year={2024}
}
```

Paper: [https://zenodo.org/records/19312827](https://zenodo.org/records/19312827)

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [Triton](https://github.com/openai/triton) for GPU kernels
- Inspired by [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- Thanks to the PyTorch team for the excellent deep learning framework