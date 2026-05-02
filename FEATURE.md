# DSALT Feature Catalog

This file lists all features of the DSALT project, both those currently implemented and natural/possible extensions for the library. The goal is to have a single point of reference for all capabilities, modules, and future extensions.

---

## 1. Core Project

- `dsalt` package distributed as a Python library on PyPI (`pip install dsalt`).
- Support for source installation with `pip install -e .`.
- Modern package metadata based on `pyproject.toml`.
- Python 3.8+ compatibility.
- Apache 2.0 license.
- Basic documentation in `README.md`.
- Development files: `requirements.txt`, `requirements-dev.txt`, `Makefile`, `MANIFEST.in`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`.

## 2. Exported API

- `dsalt.DSALTAttention`
- `dsalt.DSALTTransformer`
- `dsalt.dsalt_attention`
- `dsalt.model.DSALTLMHeadModel`
- `dsalt.training.DSALTTrainer`
- `dsalt.kernels.compute_hybrid_energy_scores`
- `dsalt.kernels.select_landmarks`

## 3. Kernels and Sparse Computation

### 3.1. Sparse Attention

- Triton kernel for DSALT sparse causal attention.
- Attention pipeline that combines:
  - variable local causal window, token-by-token
  - global landmark tokens
- GPU support with Triton when available.
- CPU fallback via PyTorch when Triton is not available.
- Mixed precision support for FP16/BF16/FP32.
- CPU vs Triton matching tests for numerical accuracy.
- Backward gradient tests for gradient consistency.

### 3.2. Hybrid Energy

- Hybrid score calculation for landmark selection.
- Z-score normalization of energy values.
- Top-k global landmark selection.
- Exclusion of tokens already covered by local window.
- GPU/Triton support and CPU fallback.

### 3.3. Window Utils

- `WindowSizePredictor`: module for predicting continuous attention windows.
- Calculation of adaptive window sizes for each token.
- Continuous window size output for regularization.

## 4. Transformer Model

- `DSALTTransformer` as a stack of decoder-only blocks.
- `DSALTBlock` with:
  - pre-norm RMSNorm
  - Multi-head DSALTAttention
  - SwiGLU feed-forward
  - dropout and residual connections
- `DSALTLMHeadModel` complete LM wrapper with:
  - token + positional embeddings
  - LM head shared with embedding (optional)
  - labels support and cross-entropy loss
  - `windows` return for regularization

## 5. Training and Fine-tuning

- `DSALTTrainer` with:
  - AdamW optimizer
  - cosine schedule with linear warmup
  - gradient clipping
  - mixed precision (`torch.autocast`) for BF16/FP16
  - DDP (DistributedDataParallel) support
  - automatic CPU/GPU device management
  - periodic checkpointing and resume loading
  - validation with perplexity calculation
  - window entropy regularization
- Support for standard PyTorch datasets and batching.
- Examples for single-GPU and multi-GPU training.

## 6. Testing

- Unit test suite in `tests/`.
- Tests for:
  - sparse attention CPU/Triton
  - forward/backward consistency
  - hybrid energy
  - LM wrapper
  - training smoke test
- `pyproject.toml` configured with Pytest.
- HTML coverage report available.

## 7. Packaging and Distribution

- PyPI `dsalt` package built with `python -m build`.
- Generated `.whl` and `.tar.gz` files.
- Automatic upload to PyPI with `twine`.
- `setup.py` compatible for legacy installation.
- `pyproject.toml` as main configuration.
- `Makefile` with commands:
  - `install`
  - `install-dev`
  - `test`
  - `test-cov`
  - `lint`
  - `format`
  - `clean`
  - `build`
  - `publish`
  - `docs`

## 8. Development and Code Quality

- Style formatting with Black.
- Import sorting with isort.
- Linting with Flake8.
- Type checking with Mypy.
- Pre-commit hook mentioned in dev requirements.
- Custom `.gitignore` for Python, build artifacts, and checkpoints.

## 9. Documentation and Support

- README with installation instructions, quick start, API, testing, and citation.
- CONTRIBUTING for project contribution guidelines.
- CHANGELOG for tracking releases.
- Apache 2.0 LICENSE.

## 10. Implemented Features Needing Refinement

- `dsalt.training.trainer`:
  - more robust checkpointing with `resume_from`
  - DDP support and rank-0 logging
  - deprecation warning `torch.cuda.amp.GradScaler` to update
- `tests/test.py`: working `tests/test.py` file but not aligned with standard `test_*.py` pattern.

## 11. Extra Features and Possible Extensions

### 11.1. Model Extensions

- `DSALTEncoder` for encoder-only or encoder-decoder architectures.
- Complete GPT-style implementation with tokenizer and config.
- `DSALTForSequenceClassification` / `DSALTForQuestionAnswering` models.

### 11.2. Kernels and Performance

- careful support for token grouping / cluster attention
- optimizations for ultra-long sequences (> 8k)
- automatic fallback to FlashAttention or native CUDA kernels
- advanced Triton usage for 2D/3D layouts and mixed-precision kernels

### 11.3. Advanced Training

- multiple learning rate schedules (AdamW, Adam, Adafactor)
- configurable warmup/decay with lambda scheduler, cosine, linear, step
- gradient checkpointing to save memory
- FP8 / int8 quantization during training
- support for pipeline parallelism and model parallelism
- logging with Weights & Biases / TensorBoard

### 11.4. Dataset and Data Loading

- dataset classes for tokenized text and autoregressive tasks
- data collator with dynamic masking and padding
- support for HuggingFace datasets
- sample generation at inference with beam search, top-k, top-p

### 11.5. Documentation & UX

- Sphinx / ReadTheDocs documentation
- `examples/` for training and inference
- how-to tutorials for DSALT training
- demo notebooks

### 11.6. DevOps and CI/CD

- GitHub Actions for test, lint, build, and publish
- packaging for automatic PyPI releases
- build/test/coverage status badges in README

### 11.7. Package UX

- `dsalt` CLI for:
  - training
  - evaluation
  - inference
  - checkpoint management
- YAML / JSON configuration for experiments

## 12. Features Documented in Codebase

- `return_windows` support in model for regularization
- global landmark selection with `landmark_idx` broadcasting
- continuous window size handling for regularization
- CPU fallback for all main kernels
- `py.typed` to signal type hints in package

## 13. Repository Overview

- `dsalt/`
  - `kernels/`: Triton kernels and CPU fallback
  - `modules/`: attention, transformer, and DSALT block
  - `model/`: LM wrapper
  - `training/`: trainer and scheduler
  - `utils/`: reserved space for future utilities
- `tests/`: automated test suite
- `dist/`: built releases
- `pyproject.toml`: main configuration
- `setup.py`: legacy compatibility
- `README.md`: main documentation
- `CONTRIBUTING.md`: contribution guidelines
- `CHANGELOG.md`: release history
- `MANIFEST.in`: package file inclusion

---

## 14. Future Priorities

1. make `tests/` fully compatible with automatic pytest;
2. update `pyproject.toml` configuration for `all` extras;
3. add a `dsalt` CLI entrypoint;
4. complete features in `utils/`;
5. extend documentation with examples and tutorials.

---

This document is intended to be the complete map of DSALT features, useful for defining roadmap, PRs, release notes, and future improvements.