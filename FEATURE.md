# DSALT Features & Hyperparameters Guide

Comprehensive reference for all hyperparameters, their defaults, and recommended usage.

---

## 📦 Component Hyperparameters

### 1. `DSALTLMHeadModel` (Language Model)

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

---

## 🧩 `DSALTAttention` (Attention Module)

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
gradient_checkpointing: bool = False
compile_attention: bool = False     # Enable `torch.compile` for the attention block (requires PyTorch 2.0+)
```

---

## 🪟 `WindowSizePredictor` (Dynamic Window Module)

Learns a per‑token window size that adapts between `n_min` and `n_max`.

### Embedded Parameters (no constructor arguments)
- `d_model`, `n_heads`, `n_min`, `n_max` are automatically inferred from the parent `DSALTAttention`.

### Output
```text
output: [batch, n_heads, seq_len]   # Predicted window size per token per head
```
The module also returns a continuous regularisation term used by the trainer when `window_reg_coef > 0`.

---

## 🏗️ `DSALTTransformer` (Core Stack)

Stack of `DSALTAttention` + feed‑forward blocks.

All architectural hyperparameters are inherited from `DSALTLMHeadModel`.

---

## 🚀 `DSALTTrainer` (Training Configuration)

High‑level training loop with mixed‑precision, distributed training, and checkpointing.

### Optimisation Hyperparameters
```python
lr: float = 3e-4
weight_decay: float = 0.1
max_grad_norm: float = 1.0
grad_accum: int = 1
```

### Learning‑Rate Schedule
```python
warmup_steps: int = 500
total_steps: int = 10_000
```

### Logging & Checkpointing
```python
log_every: int = 50
val_every: int = 500
save_every: int = 1000
save_dir: str = "checkpoints"
```

### Precision & Device
```python
dtype: torch.dtype = torch.bfloat16   # BF16 is the default for best speed‑stability trade‑off
device: torch.device = "cuda:0"
```

### Distributed Training (choose **one**)
```python
ddp: bool = False                     # Standard DistributedDataParallel
fsdp: bool = False                    # Fully‑sharded Data Parallel (model sharding)
fsdp_cpu_offload: bool = False        # Optional CPU off‑load for very large models
```

### Memory Optimisation
```python
gradient_checkpointing: bool = False   # Save ~30 % activation memory at the cost of extra compute
```

### Regularisation
```python
window_reg_coef: float = 0.0   # Entropy penalty on the predicted window distribution
```

---

## 📚 API Reference (excerpt)

```python
# Model creation
model = DSALTLMHeadModel(
    vocab_size=32000,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_min=32,
    n_max=512,
    k_lmk=64,
)

# Low‑level kernel call
from dsalt.kernels import dsalt_attention
out = dsalt_attention(Q, K, V, window_sizes, landmark_idx)
```

---

## 🧪 Testing

```bash
make test-cov          # Run full test suite with coverage
pytest tests/ -v       # Run tests directly
```

Key test modules include:
- `tests/test_sparse_attn.py` – CPU/GPU equivalence and backward pass.
- `tests/test_hybrid_energy.py` – Landmark scoring and selection.
- `tests/test_dsalt_lm.py` – Language‑model wrapper and loss.
- `tests/test_main.py` – End‑to‑end smoke test.

---

## 📊 Performance & Benchmarks (May 2026)

| Attention type | Approx. memory (GB) | Relative speed |
|----------------|--------------------|----------------|
| Dense (O(N²))  | ~3.5                | 1.0× |
| FlashAttention 2| ~1.8                | 0.5× |
| **DSALT**      | ~0.6                | **0.17×** |

---

## 📄 License

See here: <https://github.com/LeonardoCofone/dsalt-library/blob/main/LICENSE>

---

*Last updated*: May 2026
