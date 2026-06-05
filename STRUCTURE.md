# Repository structure

DSALT (**Dynamic Sparse Attention with Landmark Tokens**), a PyTorch library
implementing the sparse attention of the paper. The training path runs on GPU via
hand-written Triton kernels; a dense SDPA fallback covers CPU / no-Triton
environments and serves as the correctness reference.

```
└── 📁dsalt_pytorch
    └── 📁dsalt                       # the installable package
        └── 📁kernels                 # low-level compute: Triton kernels + pure-PyTorch helpers
        │    ├── __init__.py              # Re-exports the public low-level symbols (norm, window, scoring, autotune)
        │    ├── RMSENorm.py              # Root-Mean-Square LayerNorm (pre-norm blocks + final norm)
        │    ├── window_utils.py          # Sliding-window utilities: window sizing, masks, RoPE cache, rotary apply
        │    ├── landmark_tokens_ker.py   # Hybrid-energy landmark scoring (§4.3), single source of the score formula
        │    ├── selectors.py             # Pure-PyTorch selectors (window cont., landmark top-k, packed metadata, KV gather); Triton-free so the dense path can import them
        │    ├── sparse_attn.py           # SDPA-based dense sparse attention (CPU / no-Triton fallback path)
        │    ├── dsalt_triton_attn.py     # Triton inference kernel (forward) + its autograd Function
        │    ├── dsalt_triton_bwd.py      # Triton backward kernel for the inference Function
        │    ├── dsalt_triton_train.py    # Triton training kernel: split FlashAttention-2 fwd + bwd (DSALTTrainFunction)
        │    ├── autotune.py              # One-shot block-size autotuning per (head_dim, compute capability); main-process guard for DDP
        │    ├── cross_entropy.py         # Fused linear cross-entropy (adapted from LinkedIn Liger-Kernel, Apache-2.0)
        │    └── loss_autotune.py         # One-shot loss-fn picker for loss_fn="auto" (chunked vs liger), per GPU
        │
        ├── 📁model                   # user-facing model API
        │    ├── __init__.py              # Exposes DSALTConfig and DSALTLMHeadModel
        │    ├── config.py                # DSALTConfig dataclass + validation
        │    └── dsalt_lm.py              # Causal LM: embeddings, block stack, tied LM head, loss (chunked/liger/auto)
        │
        ├── 📁modules                 # Transformer building blocks
        │    ├── __init__.py              # Exposes DSALTAttention, DSALTTransformerBlock, SwiGLUFFN
        │    ├── dsalt_attention.py       # DSALT attention: window ∪ landmark fusion, kernel + dense paths, shared selectors
        │    └── dsalt_transformer.py     # Pre-norm Transformer block + SwiGLU FFN
        │
        ├── 📁training                # training loop and infrastructure
        │    ├── __init__.py              # Exposes DSALTTrainer
        │    ├── gpu_auto.py              # Hardware detection, DDP setup, device/VRAM helpers
        │    ├── logging_config.py        # ANSI step formatter, file logging, StepTimer (it/s, tok/s)
        │    └── trainer.py               # Training loop, AMP/scaler, DDP+torch.compile wrap, checkpointing, LR schedule, metrics
        │
        ├── __init__.py               # Library version and top-level re-exports
        └── py.typed                  # PEP 561 marker: the package ships inline type hints
    │
    ├── 📁 (project root)
    ├── .gitignore
    ├── .pre-commit-config.yaml       # Pre-commit hooks (black/isort/flake8/mypy), mirrors pyproject tooling
    ├── README.md                     # Project README
    ├── FEATURE.md                    # Feature overview / roadmap
    ├── DESIGN_NOTES.md               # Engineering design rationale (differentiable approximations, DDP+compile, key-parallel backward, profiling)
    ├── STRUCTURE.md                  # This file, repository layout and intra-package usage map
    ├── CONTRIBUTING.md               # Contribution guidelines
    ├── LICENSE                       # License text
    ├── MANIFEST.in                   # sdist data-file inclusion rules
    ├── pyproject.toml                # Build system + tooling config
    ├── setup.py                      # Packaging entry point
    ├── requirements.txt              # Runtime dependencies
    ├── requirements-dev.txt          # Dev/test dependencies
    └── release.ps1                   # PowerShell release/bump script
```


## Intra-package usage map (who imports whom)

```
kernels/RMSENorm.py            -> model/dsalt_lm.py, modules/dsalt_transformer.py, kernels/__init__.py, dsalt/__init__.py
kernels/window_utils.py        -> modules/dsalt_attention.py, kernels/__init__.py
kernels/landmark_tokens_ker.py -> modules/dsalt_attention.py, kernels/selectors.py (re-exported)
kernels/selectors.py           -> modules/dsalt_attention.py, kernels/dsalt_triton_attn.py
kernels/sparse_attn.py         -> modules/dsalt_attention.py, kernels/__init__.py
kernels/dsalt_triton_attn.py   -> modules/dsalt_attention.py (inference path)
kernels/dsalt_triton_bwd.py    -> kernels/dsalt_triton_attn.py
kernels/dsalt_triton_train.py  -> modules/dsalt_attention.py (training path)
kernels/autotune.py            -> kernels/dsalt_triton_attn.py, kernels/dsalt_triton_bwd.py, kernels/dsalt_triton_train.py, kernels/__init__.py
kernels/cross_entropy.py       -> model/dsalt_lm.py, kernels/__init__.py   (the fused liger CE Function)
kernels/loss_autotune.py       -> model/dsalt_lm.py                        (picks chunked vs liger when loss_fn="auto")

model/config.py                -> model/dsalt_lm.py, model/__init__.py
model/dsalt_lm.py              -> model/__init__.py, dsalt/__init__.py
modules/dsalt_attention.py     -> modules/dsalt_transformer.py, modules/__init__.py
modules/dsalt_transformer.py   -> model/dsalt_lm.py, modules/__init__.py
training/trainer.py            -> training/__init__.py, dsalt/__init__.py
```

Notes:
- `loss_autotune.py` deliberately **does not** import `autotune.py` or
  `cross_entropy.py` at module load; it re-implements the tiny bench/print helpers
  inline to keep its import graph minimal (so it stays importable without Triton).
- `selectors.py` and `landmark_tokens_ker.py` are Triton-free on purpose, so the
  dense fallback path imports them without a GPU+Triton install.
- The inference kernel (`dsalt_triton_attn.py` + `dsalt_triton_bwd.py`) and the
  training kernel (`dsalt_triton_train.py`, fwd+bwd in one file) are separate code
  paths; both go through `dsalt_attention.py`.


## Engineering design rationale

See **[DESIGN_NOTES.md](DESIGN_NOTES.md)** for the formal design write-up:
differentiable approximations for hardware-efficient attention, the
DDP + `torch.compile` graph-integrity strategy, the asymmetric key-parallel
backward, and the profiling evidence behind the paper's performance claims.
