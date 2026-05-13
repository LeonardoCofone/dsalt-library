# DSALT: Dynamic Sparse Attention with Landmark Tokens

[![PyPI](https://img.shields.io/pypi/v/dsalt)](https://pypi.org/project/dsalt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

DSALT is a high‑performance PyTorch library that implements **Dynamic Sparse Attention with Landmark Tokens** – a memory‑efficient attention mechanism for transformers. It relies on Triton kernels and supports distributed training.

> **Install**: `pip install dsalt`  
> **Source**: <https://github.com/LeonardoCofone/dsalt-library>  
> **Paper**: <https://zenodo.org/records/19312826>  

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
┌─ Local window (adaptive) ──┬─ Global landmarks ─┐
│ Recent N tokens            │ Top‑K informative  │ 
│ (window size grows)        │ tokens per head    │
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
    -   Gradient checkpointing to save activation memory at the cost of recomputing some layers.
    -   Learning‑rate warm‑up and cosine decay for stable and effective optimisation.
    -   Optional window‑entropy regularisation (`window_reg_coef`) to encourage diverse window size predictions.
    -   Comprehensive checkpointing and logging utilities.
    -   Support for various distributed training strategies (DDP, FSDP).

For detailed configuration, refer to the Hyperparameter Guide below, especially the `DSALTTrainer` section.

---

## 📖 Hyperparameter Guide

This section provides a comprehensive reference for all key hyperparameters across DSALT components, their defaults, and recommended usage.

### DSALTLMHeadModel (Language Model)

The main language‑model wrapper that combines embeddings, transformer blocks, and an output head.

#### Required Parameters
```python
vocab_size: int          # Vocabulary size (e.g., 32000 for GPT‑2)
d_model: int            # Hidden dimension, must be divisible by `n_heads`
n_layers: int           # Number of transformer blocks
n_heads: int            # Number of attention heads (d_model // n_heads must be a power of two and ≥ 16)
```

#### Architecture Hyperparameters
```python
d_ff: int | None = None   # Feed‑forward hidden dim (default = 4 × d_model)
max_seq_len: int = 2048   # Maximum sequence length for positional embeddings
dropout: float = 0.0     # Dropout rate applied after attention and FFN
use_fa2: bool = True      # Enable FlashAttention 2 when Triton is available
tie_weights: bool = True # Share embedding and output‑projection weights
```

#### Sparse‑Attention Hyperparameters
```python
n_min: int = 32            # Minimum local window size (causal sliding window)
n_max: int = 256           # Maximum local window size (grows with token position)
k_lmk: int = 16           # Number of global landmark tokens per head
```

*Note*: `alpha` is a learnable per‑head weight automatically initialised; it is **not** exposed as a configuration flag.

### DSALTAttention (Attention Module)

A multi‑head sparse‑attention layer with adaptive windows and landmark selection.

#### Required Parameters
```python
d_model: int
n_heads: int
```

#### Sparse‑Attention Hyperparameters (inherited from the model)
```python
n_min: int
n_max: int
k_lmk: int
alpha: float = 0.6   # Initial value for the learnable weight per head
```

#### Regularisation & Optimisation
```python
dropout: float = 0.0
use_fa2: bool = True                # FlashAttention 2 fallback when the whole sequence fits the local window
gradient_checkpointing: bool = False # Enable gradient checkpointing for memory savings
compile_attention: bool = False     # Enable `torch.compile` for the attention block (requires PyTorch 2.0+)
```

### WindowSizePredictor (Dynamic Window Module)

Learns a per‑token window size that adapts between `n_min` and `n_max`.

### Embedded Parameters (no constructor arguments)
-   `d_model`, `n_heads`, `n_min`, `n_max` are automatically inferred from the parent `DSALTAttention`.

### Output
```text
output: [batch, n_heads, seq_len]   # Predicted window size per token per head
```
The module also returns a continuous regularisation term used by the trainer when `window_reg_coef > 0`.

### DSALTTransformer (Core Stack)

Stack of `DSALTAttention` + feed‑forward blocks.

All architectural hyperparameters are inherited from `DSALTLMHeadModel`.

### DSALTTrainer (Training Configuration)

High‑level training loop with mixed‑precision, distributed training, and checkpointing.

#### Optimisation Hyperparameters
```python
lr: float = 3e-4                 # Initial learning rate
weight_decay: float = 0.1        # L2 regularisation strength
max_grad_norm: float = 1.0       # Maximum gradient norm for clipping
grad_accum: int = 1              # Number of gradient accumulation steps
```

#### Learning‑Rate Schedule
```python
warmup_steps: int = 500          # Number of steps for linear learning rate warm-up
total_steps: int = 10_000        # Total number of training steps for cosine decay
```

#### Logging & Checkpointing
```python
log_every: int = 50              # Log training metrics every N steps
val_every: int = 500             # Run validation every N steps
save_every: int = 1000           # Save model checkpoint every N steps
save_dir: str = "checkpoints"    # Directory to save checkpoints and logs
```

#### Precision & Device
```python
dtype: torch.dtype = torch.bfloat16   # Data type for training (BF16 default for speed/stability)
device: torch.device = "cuda:0"       # Device to run training on (e.g., "cuda:0", "cpu")
```

#### Distributed Training (choose **one**)
```python
ddp: bool = False                     # Enable standard DistributedDataParallel
fsdp: bool = False                    # Enable Fully‑sharded Data Parallel (model sharding)
fsdp_cpu_offload: bool = False        # Optional CPU off‑load for very large models with FSDP
```

#### Memory Optimisation
```python
gradient_checkpointing: bool = False   # Save ~30 % activation memory at the cost of extra compute
```

#### Regularisation
```python
window_reg_coef: float = 0.0   # Entropy penalty on the predicted window distribution (0.0 disables)
```

---

## 📚 API Reference (excerpt)

```python
# Model creation
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

# Forward pass
input_ids = torch.randint(0, 32000, (1, 1024))
logits, windows = model(input_ids, return_window=True)

# Low‑level kernel call (for advanced users)
from dsalt.kernels import dsalt_attention
Q, K, V = torch.randn(1, 16, 1024, 64), torch.randn(1, 16, 1024, 64), torch.randn(1, 16, 1024, 64)
window_sizes = torch.full((1, 16, 1024), 64, dtype=torch.int32)
landmark_idx = torch.randint(0, 1024, (1, 16, 1024, 16), dtype=torch.int32)
out = dsalt_attention(Q, K, V, window_sizes, landmark_idx)
```

---

## 🧪 Testing


Key test modules include:
-   `tests/test_sparse_attn.py` – CPU/GPU equivalence and backward pass.
-   `tests/test_hybrid_energy.py` – Landmark scoring and selection.
-   `tests/test_dsalt_lm.py` – Language‑model wrapper and loss.
-   `tests/test_main.py` – End‑to‑end smoke test.

---

## 📄 License

See here: <https://github.com/LeonardoCofone/dsalt-library/blob/main/LICENSE>

---

## 🤝 Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for guidelines. Areas where help is especially valuable:
-   Triton kernel optimisation
-   New model architectures (encoder, encoder‑decoder)
-   Additional training strategies and samplers
-   Documentation and tutorials
-   Bug reports and fixes

---

## 📞 Support & Questions

-   **Issues**: <https://github.com/LeonardoCofone/dsalt-library/issues>
-   **Discussions**: <https://github.com/LeonardoCofone/dsalt-library/discussions>
-   **Paper**: <https://zenodo.org/records/19312826>

---
