# DSALT Features & Hyperparameters Guide

Complete reference for all hyperparameters, their defaults, and use cases.

---

## 🏗️ Component Hyperparameters

### 1. DSALTLMHeadModel (Language Model)

Language model wrapper combining embeddings, transformer blocks, and output layer.

#### Required Parameters
```python
vocab_size: int
    Size of the vocabulary / token space
    Examples: 32000 (LLaMA), 50257 (GPT-2), 128000 (GPT-4)

d_model: int
    Hidden dimension of the transformer
    Must be divisible by n_heads
    Examples: 512, 768, 1024, 1536, 2048
    
    Constraint: d_model % n_heads == 0

n_layers: int
    Number of transformer blocks (depth)
    Examples: 6, 12, 24, 32

n_heads: int
    Number of attention heads
    d_head = d_model / n_heads must be:
      - Power of 2: 32, 64, 128, 256
      - At least 16
    Examples: 8, 12, 16, 32
```

#### Architecture Hyperparameters
```python
d_ff: int | None = None
    Feed-forward hidden dimension (intermediate layer in SwiGLU)
    If None: defaults to 4 * d_model
    
    Typical values:
    - Small model: 2048 (d_model=512)
    - Medium model: 3072 (d_model=768)
    - Large model: 4096 (d_model=1024)
    
    Effect: Larger → more parameters & compute; typically 4× is optimal

max_seq_len: int = 2048
    Maximum sequence length (for positional embeddings)
    Must be >= longest sequence in training
    Examples: 512, 1024, 2048, 4096
    
    Memory impact: Linear with seq_len in embeddings

dropout: float = 0.0
    Dropout rate across all layers
    Range: [0.0, 1.0]
    Examples:
    - 0.0: No regularization (use for inference)
    - 0.1: Light regularization (typical)
    - 0.2: Medium regularization (for overfitting)
    
    Note: Applied after attention and FFN

use_fa2: bool = True
    Use FlashAttention 2 for faster/memory-efficient attention
    Requires: triton>=2.0 and CUDA-capable GPU
    Fall back to normal attention if unavailable
    
    Performance impact: ~2-3× faster if available

tie_weights: bool = True
    Share weights between token embedding and output layer
    Reduces parameters: saves vocab_size * d_model parameters
    
    Effect:
    - True: Smaller model, slightly lower capacity
    - False: Independent embeddings, more parameters
```

#### Sparse Attention Hyperparameters
```python
n_min: int = 32
    Minimum local window size (causal sliding window)
    Range: [4, 256]
    Recommended: 32–64
    
    Effect:
    - Smaller: Fewer local tokens, potentially missing short-term context
    - Larger: More local compute, more local bias
    - Typical: 32–64

n_max: int = 256
    Maximum local window size (allows window to grow with position)
    Range: [n_min, seq_len]
    Recommended: 128–512
    
    Effect:
    - Smaller: Limited long-range awareness in late positions
    - Larger: More memory per token, but longer-range context
    - Typical: 256–512

k_lmk: int = 16
    Number of landmark (global) tokens per head
    Range: [1, 64]
    Recommended: 16–64
    
    Effect:
    - Smaller: Fewer global tokens, lower recall of important tokens
    - Larger: More global coverage, more compute
    - Typical: 16–32 per head
    
    Total sparse attention complexity:
    - Local: O(n_max * N) ≈ O(256 * N)
    - Global: O(k_lmk * N) ≈ O(16 * N)
    - Total: O((n_max + k_lmk) * N) ≈ O(272 * N) vs O(N²) dense

```

**Note**: The `alpha` parameter (learnable per-head weight) is automatically initialized and trained internally. It is not a configuration hyperparameter.

---

### 2. DSALTAttention (Attention Module)

Single multi-head sparse attention layer with adaptive windows.

#### Required Parameters
```python
d_model: int
    Hidden dimension (must match DSALTTransformer)

n_heads: int
    Number of attention heads
    d_head = d_model / n_heads must be power of 2, ≥ 16
```

#### Sparse Attention Hyperparameters
```python
n_min: int = 32
    (Same as DSALTLMHeadModel.n_min)
    Minimum window size

n_max: int = 256
    (Same as DSALTLMHeadModel.n_max)
    Maximum window size

k_lmk: int = 16
    (Same as DSALTLMHeadModel.k_lmk)
    Number of landmarks per head

alpha: float = 0.6
    (Same as DSALTLMHeadModel.alpha)
    Initial learnable per-head weight
```

#### Regularization & Optimization
```python
dropout: float = 0.0
    Attention pattern dropout
    Applied to attention weights before value multiplication
    
    Effect:
    - 0.0: No dropout
    - 0.1: Light regularization
    - 0.2+: Strong regularization (rare)

use_fa2: bool = True
    Use FlashAttention 2 for faster computation
    Automatic fallback if unavailable

gradient_checkpointing: bool = False
    Gradient checkpointing (trade compute for memory)
    
    Effect:
    - False: Store all activations (faster backward, more memory)
    - True: Recompute activations in backward (slower, saves ~30% memory)
    
    Recommendation: Enable if OOM during backward

compile_attention: bool = False
    Use torch.compile on attention kernel
    Requires: PyTorch 2.0+
    
    Effect:
    - False: Normal Python dispatch
    - True: JIT-compiled attention (5-10% faster, first run slower)
```

---

### 3. WindowSizePredictor (Dynamic Window Module)

Learns per-token window sizes adaptively.

#### Embedded Parameters (No constructor config)
```python
# Automatically configured from DSALTAttention:
d_model: int
    (From parent attention)

n_heads: int
    (From parent attention)

n_min: int
    (From DSALTAttention)

n_max: int
    (From DSALTAttention)
```

#### Learned Behavior
```python
output: [batch, n_heads, seq_len]
    Predicted window size per token per head
    Range: [n_min, n_max]
    
    Regularization term:
    - Entropy of window distribution
    - Used if window_reg_coef > 0 in trainer
```

---

### 4. DSALTTransformer (Core Stack)

Stack of DSALT attention + feed-forward blocks.

#### Configuration
```python
# All parameters inherited from DSALTLMHeadModel:
vocab_size, d_model, n_layers, n_heads, n_min, n_max, k_lmk,
alpha, d_ff, max_seq_len, dropout, use_fa2, tie_weights
```

#### Block Structure (per layer)
```
[Pre-LN] → Attention → [Residual Add]
  ↓
[Pre-LN] → SwiGLU(FFN) → [Residual Add]
```

---

## 🎯 DSALTTrainer (Training Configuration)

Training loop with mixed precision, distributed training, checkpointing.

### Optimization Hyperparameters
```python
lr: float = 3e-4
    Learning rate
    Typical range:
    - Small model (768): 1e-4 to 5e-4
    - Large model (1024+): 1e-4 to 3e-4
    
    Recommendation: Start with 3e-4, decay with scheduler

weight_decay: float = 0.1
    AdamW weight decay (L2 regularization on weights)
    Typical range: [0.0, 0.2]
    
    Effect:
    - 0.0: No regularization
    - 0.1: Standard (recommended)
    - 0.2+: Strong regularization (for overfitting)
    
    Note: Not applied to bias, norm, or embedding layers

max_grad_norm: float = 1.0
    Gradient clipping threshold
    Range: [0.1, 10.0]
    
    Effect:
    - None (0.0): No clipping → potential exploding gradients
    - 1.0: Standard clipping (recommended)
    - Higher: Less clipping, more gradient variance

grad_accum: int = 1
    Gradient accumulation steps
    Effective batch size = batch_size * grad_accum
    Range: [1, 256]
    
    Example:
    - batch_size=4, grad_accum=1 → eff_batch=4
    - batch_size=4, grad_accum=8 → eff_batch=32 (same GPU memory)
    
    Use when batch_size limited by GPU memory
```

### Learning Rate Schedule
```python
warmup_steps: int = 500
    Linear warmup steps
    LR goes from 0 → lr over these steps
    Range: [0, 10000]
    
    Typical: 0.05 * total_steps
    Example: 500–1000 for 10K–100K total_steps
    
    Effect:
    - 0: No warmup (can cause instability)
    - 500: Standard warmup
    - Higher: Longer ramp-up

total_steps: int = 10_000
    Total training steps (optimizer updates)
    Range: [1000, 1_000_000]
    
    Equivalence: steps = (num_samples * epochs) / eff_batch_size
    
    After warmup, LR decays with cosine schedule toward min_lr_ratio
```

### Logging & Checkpointing
```python
log_every: int = 50
    Logging interval (steps between log prints)
    Typical range: [10, 200]
    
    Effect:
    - Smaller: More frequent logs, slightly slower
    - Larger: Less I/O, less detailed tracking

val_every: int = 500
    Validation interval (steps between validation runs)
    Typical range: [100, 1000]
    
    Effect:
    - Smaller: More validation, slower training
    - Larger: Less validation, faster training
    
    Tip: val_every should be multiple of log_every

save_every: int = 1000
    Checkpoint save interval
    Typical range: [500, 5000]
    
    Effect:
    - Smaller: More checkpoints (disk space)
    - Larger: Fewer checkpoints (risk of losing progress)
    
    Default: Usually equals val_every

save_dir: str = "checkpoints"
    Directory to save checkpoints and logs
    Checkpoints include:
    - Model state_dict
    - Optimizer state
    - Scheduler state
    - Training history
```

### Validation & Metrics
```python
val_loader: DataLoader | None = None
    Validation dataset loader
    Used to compute val_ppl every val_every steps
    
    If None: No validation (training loss only)

compute_metrics_fn: Callable | None = None
    Custom function fn(model, x) → dict
    Called every log_every steps
    
    Example metrics:
    - sigma2: Attention matrix singular value
    - eff_rank: Effective rank of attention
    - attn_entropy: Attention weight entropy
    - res_norm: Residual norm per layer
    
    If None: Skip detailed metrics (faster)
```

### Precision & Device
```python
dtype: torch.dtype = torch.bfloat16
    Training precision
    Options:
    - torch.float32: Full precision (slower, more memory)
    - torch.float16: Half precision (faster, less stable)
    - torch.bfloat16: Brain float (recommended; good for stability)
    
    Performance impact (relative to FP32):
    - FP16: ~1.5–2× faster (less stable)
    - BF16: ~1.2–1.8× faster (more stable)
    
    Recommendation: BF16 for most cases

device: torch.device = "cuda:0"
    Device to train on
    Examples:
    - torch.device("cuda:0"): First GPU
    - torch.device("cuda"): Auto-detect first GPU
    - torch.device("cpu"): CPU training (slow)
```

### Distributed Training (choose ONE)
```python
ddp: bool = False
    Standard Distributed Data Parallel
    Requires: torchrun or torch.distributed
    
    Usage: torchrun --nproc_per_node=2 train.py
    Effect:
    - One process per GPU
    - Each GPU handles own batch
    - Gradients synced across GPUs
    - Overhead: ~5–10% slower than single-GPU

fsdp: bool = False
    Fully Sharded Data Parallel (model sharding)
    Requires: torchrun + torch.distributed
    
    Usage: torchrun --nproc_per_node=2 train.py
    Effect:
    - Model parameters sharded across GPUs
    - Each GPU sees full batch
    - Reduces per-GPU memory by ~N (N = num GPUs)
    - Overhead: ~10–15% slower than single-GPU
    
    Use when model too large for single GPU

fsdp_cpu_offload: bool = False
    CPU offload for FSDP (advanced)
    Offloads unused parameters to CPU
    
    Effect:
    - Much slower (frequent CPU↔GPU transfers)
    - Saves GPU memory (rarely needed)
    - Only use as last resort
```

### Multi-GPU Without Distribution (Simple)
```python
# Wrap model with DataParallel
import torch.nn as nn
model = DSALTLMHeadModel(...)
model = nn.DataParallel(model)  # ← Automatic multi-GPU

trainer = DSALTTrainer(model=model, ...)
# No distributed setup needed!
```

### Memory Optimization
```python
gradient_checkpointing: bool = False
    Gradient checkpointing (trade compute for memory)
    
    Effect (vs. no checkpointing):
    - Memory: ~30% reduction
    - Speed: ~10–20% slower (recompute in backward)
    
    When to use:
    - Large models (>1B params)
    - Long sequences (>2K tokens)
    - Limited GPU memory
    
    Typical: False for most cases, True if OOM
```

### Regularization
```python
window_reg_coef: float = 0.0
    Window size regularization coefficient
    Adds entropy penalty to window distribution
    
    Loss: total_loss = ce_loss + window_reg_coef * entropy_loss
    
    Range: [0.0, 1.0]
    Typical: 0.0 (no regularization)
    
    Use cases:
    - 0.0: Let window sizes adapt freely (recommended)
    - 0.01–0.1: Encourage more uniform window sizes
    - 0.1+: Force diversity in window decisions
```

### Resume Training
```python
resume_from: str | None = None
    Path to checkpoint file to resume from
    Loads:
    - Model state
    - Optimizer state
    - Scheduler state
    - Training history
    
    Example:
    trainer = DSALTTrainer(
        ...,
        resume_from="checkpoints/step_0010000.pt"
    )
    # Continues from step 10001
```

---

## 📊 Complete Config Example

```python
CFG = {
    # === MODEL CONFIG (DSALTLMHeadModel) ===
    "vocab_size": 32000,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_ff": None,              # Auto = 4 * 768 = 3072
    "max_seq_len": 2048,
    "dropout": 0.1,
    "use_fa2": True,
    "tie_weights": True,
    
    # Sparse Attention
    "N_min": 32,
    "N_max": 256,
    "k_lm": 16,
    # alpha: automatically initialized & trained internally
    
    # === TRAINING CONFIG (DSALTTrainer) ===
    # Optimization
    "lr": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "grad_accum": 5,
    
    # Schedule
    "warmup_steps": 500,
    "max_steps": 100_000,
    
    # Logging & Checkpoints
    "log_every": 50,
    "val_every": 500,
    "save_dir": "./checkpoints",
    
    # Data
    "batch_size": 8,
    "seq_len": 2048,
    
    # Precision
    "dtype": "bfloat16",
    
    # Advanced
    "window_reg_coef": 0.0,
    "gradient_checkpointing": False,
}
```

---

## 🔍 Hyperparameter Tuning Guide

### For Different Scenarios

#### 📱 Mobile / Edge (Small Model)
```python
d_model = 256
n_layers = 6
n_heads = 4
N_min = 16
N_max = 64
k_lm = 8
grad_accum = 1
batch_size = 2
```

#### 💻 Consumer GPU (e.g., RTX 4090, 24GB)
```python
d_model = 768
n_layers = 12
n_heads = 12
N_min = 32
N_max = 256
k_lm = 16
grad_accum = 4
batch_size = 8
```

#### 🖥️ Enterprise (H100 80GB)
```python
d_model = 1024
n_layers = 24
n_heads = 16
N_min = 64
N_max = 512
k_lm = 32
grad_accum = 8
batch_size = 16–32
fsdp = True
```

#### 🚀 Research (Multi-GPU 8× H100)
```python
d_model = 2048
n_layers = 32–48
n_heads = 32
N_min = 64
N_max = 1024
k_lm = 64
grad_accum = 16
batch_size = 64
fsdp = True
gradient_checkpointing = True
```

### Tuning Tips

1. **Start Simple**: Single GPU, standard config, small model
2. **Scale Gradually**: Increase d_model → n_layers → batch_size
3. **LR Schedule**: Reduce LR slightly when scaling model size
4. **Batch Size**: Sweet spot usually 8–32 per GPU
5. **Sparse Params**: Keep ratio `k_lm ≈ n_max / 16` (e.g., n_max=256 → k_lm=16)

---

**Last Updated**: May 2026
