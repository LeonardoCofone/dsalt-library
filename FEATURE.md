# DSALT Feature Catalog

This document provides a comprehensive reference for all DSALT features—both currently implemented and planned extensions. It serves as a single source of truth for capabilities, modules, and future development roadmap.

---

## 1. Core Project

- `dsalt` package distributed on PyPI (`pip install dsalt`)
- Source installation support (`pip install -e .`)
- Modern package metadata via `pyproject.toml`
- Python 3.8+ compatibility
- Apache 2.0 license
- Complete documentation in `README.md`
- Development files: `requirements.txt`, `requirements-dev.txt`, `Makefile`, `MANIFEST.in`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`

## 2. Exported Public API

- `dsalt.DSALTAttention`
- `dsalt.DSALTTransformer`
- `dsalt.dsalt_attention`
- `dsalt.model.DSALTLMHeadModel`
- `dsalt.training.DSALTTrainer`
- `dsalt.kernels.compute_hybrid_energy_scores`
- `dsalt.kernels.select_landmarks`

## 3. High-Performance Kernels

### 3.1 Sparse Attention Kernels

- **Triton-optimized forward pass** for DSALT sparse causal attention
- **Dual-path attention pipeline**:
  - Adaptive local causal window (token-by-token)
  - Global landmark tokens (top-k per head)
- **GPU acceleration** with Triton (when available)
- **CPU fallback** using native PyTorch operations
- **Mixed-precision support** (FP16, BF16, FP32)
- **Verified numerics**: CPU/Triton equivalence tests ensure correctness
- **Gradient stability**: Comprehensive backward pass tests for gradient consistency

**Recent Optimization (2024):**
- `landmark_idx` tensor shape changed from `[B, H, N, K]` to `[B, H, K]`
  - Eliminates `N` replication: same top-k landmarks shared across all queries of a head
  - Semantically correct per paper: landmarks are head-specific globals, not query-specific
  - Memory savings: O(N) reduction in landmark tensor allocation

### 3.2 Hybrid Energy Scoring

- Hybrid energy calculation for landmark token selection
- Z-score normalization of energy scores
- Top-k global landmark selection per head
- Intelligent exclusion of tokens already in local window
- GPU/Triton support with CPU fallback

**Recent Optimization (2024):**
- Input shape standardization: now accepts `X` as `[B, N, D]` instead of `[B, H, N, D]`
  - Eliminates `H×` memory replica of hidden states
  - Kernel processes all heads from same shared representation
  - Reduces memory pressure by 4–8× on typical configurations (H=8–16, N=512–1024)

### 3.3 Adaptive Window Utilities

- `WindowSizePredictor`: learned module for dynamic attention windows
- Per-token adaptive window size computation
- Continuous window size output for entropy regularization
- Enables attention scope to grow naturally with sequence position

## 4. Transformer Architecture

- **`DSALTTransformer`**: Decoder-only stack of DSALT blocks
- **`DSALTBlock`** components:
  - Pre-norm RMSNorm for stable gradients
  - Multi-head DSALTAttention with hybrid sparsity
  - SwiGLU feed-forward network
  - Residual connections and configurable dropout
- **`DSALTLMHeadModel`**: Complete language model wrapper featuring:
  - Token + positional embeddings
  - Learnable LM head (optional weight sharing with embeddings)
  - Direct label input and cross-entropy loss computation
  - Attention window regularization loss (via `windows` return)

## 5. Training Infrastructure

- **`DSALTTrainer`** with integrated training loop:
  - AdamW optimizer with per-layer weight decay groups
  - Cosine annealing schedule with linear warmup phase
  - Gradient clipping and normalization
  - Mixed-precision training (`torch.autocast`) for BF16/FP16
  - **DDP support** (DistributedDataParallel) for standard multi-GPU
  - **FSDP support** (FullyShardedDataParallel) for 2+ GPU model sharding
  - **Gradient accumulation** with `no_sync()` optimization to avoid redundant all-reduce
  - Automatic CPU/GPU device management
  - Periodic checkpointing with resume capability
  - Validation loop with perplexity calculation
  - Window entropy regularization

**Recent Enhancements (2024):**
- Removed DataParallel: replica overhead unsuitable for sparse attention
- Gradient checkpointing correctly integrated: full attention block checkpoint (not lambda-wrapped)
- Fixed duplicate `_is_main` definition that caused silent override
- FSDP integration for distributed training: `torchrun --nproc_per_node=2 train.py --fsdp`
- Synchronized gradient accumulation without intermediate all-reduce cost

- PyTorch dataset and DataLoader compatibility
- Single-GPU and multi-GPU training examples included

## 6. Test Suite

- Comprehensive unit tests in `tests/` directory
- Test coverage includes:
  - Sparse attention: CPU/Triton forward/backward equivalence
  - Gradient correctness and numerical stability
  - Hybrid energy scoring
  - LM wrapper inference and loss computation
  - End-to-end training smoke tests
- Pytest integration via `pyproject.toml`
- HTML coverage reports with detailed per-module breakdowns

## 7. Packaging & Distribution

- **Build**: `python -m build` produces `.whl` and `.tar.gz`
- **Distribution**: PyPI package `dsalt` with automated `twine` upload
- **Compatibility**: Legacy `setup.py` support + modern `pyproject.toml`
- **Makefile automation**:
  - `make install` / `make install-dev`
  - `make test` / `make test-cov`
  - `make lint` / `make format`
  - `make clean` / `make build` / `make publish`

## 8. Code Quality & Development

- **Formatting**: Black (code style)
- **Import management**: isort (import sorting)
- **Linting**: Flake8 (style & best practices)
- **Type checking**: Mypy (static type validation)
- **Pre-commit hooks**: Optional integration for CI/CD
- **.gitignore**: Python, build artifacts, checkpoints

## 9. Documentation

- **README**: Installation, quick-start, architecture overview, API reference
- **CONTRIBUTING**: Guidelines for project contributions
- **CHANGELOG**: Release history and version notes
- **LICENSE**: Apache 2.0 legal text
- **py.typed**: Marker for type-aware IDE support

## 10. Current Implementation Status

### ✅ Fully Implemented & Optimized
- Memory-efficient landmark indexing (`[B, H, K]` shape)
- Shared hidden state across heads (no `H×` replication)
- Corrected gradient checkpointing in attention
- FSDP support for distributed model sharding
- Fixed backward pass kernels (removed dead code, unused parameters)
- Distributed training without redundant gradient synchronization

### ⚙️ Ready for Enhancement
- Robust checkpoint resumption logic
- Expanded DDP/FSDP logging (per-rank metrics)
- Integration with Weights & Biases / TensorBoard

## 11. Roadmap: Planned Extensions

### 11.1 Model Architectures

- `DSALTEncoder` for encoder-only and encoder-decoder setups
- Full GPT-style model with built-in tokenizer and config serialization
- Task-specific heads: `DSALTForSequenceClassification`, `DSALTForQuestionAnswering`

### 11.2 Kernel Optimization

- Token grouping / cluster attention for hierarchical sparsity
- Ultra-long sequence support (>8K tokens)
- Automatic fallback to FlashAttention or native CUDA kernels
- Advanced Triton: 2D/3D tiling, mixed-precision kernels

### 11.3 Advanced Training

- Multiple optimizer schedules: Adam, Adafactor, AdaBound
- Flexible LR schedulers: lambda, cosine, linear, exponential decay
- Optional gradient checkpointing per layer
- FP8 / int8 quantization during training
- Pipeline parallelism and model parallelism support
- Integration with Weights & Biases / TensorBoard

### 11.4 Data & Sampling

- High-level dataset classes for tokenized text (autoregressive)
- Dynamic masking & padding collators
- HuggingFace Datasets integration
- Inference sampling: beam search, top-k, top-p decoding

### 11.5 Documentation & Examples

- Sphinx + ReadTheDocs full documentation site
- `examples/` folder with training & inference scripts
- How-to tutorials for DSALT fine-tuning
- Interactive Jupyter notebooks

### 11.6 CI/CD & DevOps

- GitHub Actions: automated test, lint, build, publish
- Automatic PyPI releases on tag
- Build/test/coverage badges in README

### 11.7 User Experience

- CLI tool `dsalt` for:
  - Training launch
  - Model evaluation
  - Inference serving
  - Checkpoint inspection
- YAML/JSON config files for experiment reproducibility

## 12. Repository Structure

```
dsalt/
├── kernels/
│   ├── hybrid_energy.py       # Landmark scoring (optimized input shape)
│   ├── sparse_attn.py         # Triton attention kernels
│   └── window_utils.py        # Adaptive window prediction
├── modules/
│   ├── dsalt_attention.py     # Multi-head attention layer
│   ├── dsalt_transformer.py   # Transformer block stack
│   └── __init__.py
├── model/
│   ├── dsalt_lm.py            # Language model wrapper
│   └── __init__.py
├── training/
│   ├── trainer.py             # Training harness (DDP/FSDP support)
│   └── __init__.py
├── utils/                     # Reserved for future utilities
└── __init__.py
tests/                         # Test suite with high coverage
```

## 13. Key Technical Achievements (2024)

1. **Memory Optimization**: Eliminated implicit tensor replications in landmark and hidden state handling
2. **Correctness**: Fixed backward pass kernel signatures and gradient checkpointing logic
3. **Distributed Training**: Added FSDP support for model sharding across 2+ GPUs
4. **Training Stability**: Optimized gradient accumulation to avoid spurious synchronization

---

## 14. Development Priorities (Next Steps)

1. Expand test suite to cover all PyTorch/Triton edge cases
2. Add `all` extras group to `pyproject.toml` for optional dependencies
3. Implement `dsalt` CLI entrypoint for easy training/inference
4. Complete `utils/` module with data loading helpers
5. Publish comprehensive tutorials and benchmark results

---

This document evolves with each release. For the latest status, check the GitHub repository and CHANGELOG.