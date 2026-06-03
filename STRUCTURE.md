```
└── 📁dsalt_pytorch
    └── 📁dsalt
        ├── 📁kernels
        │    ├── __init__.py          # Exposes the low-level functions (scoring, masking, kernels)
        │    ├── landmark_tokens_ker.py # Landmark selection logic (hybrid-energy scoring + Top-K)
        │    ├── RMSENorm.py          # Root Mean Square Layer Normalization (stability)
        │    ├── sparse_attn.py       # SDPA-based sparse attention (CPU/GPU fallback path)
        │    ├── window_utils.py      # Sliding-window utilities (masking, RoPE cache, relative indices)
        │    ├── dsalt_triton_attn.py # Triton forward kernel + autograd Function
        │    ├── dsalt_triton_bwd.py  # Triton backward kernel
        │    ├── autotune.py            # One-shot block-size autotuning (per head_dim, GPU)  
        │    ├── dsalt_triton_train.py  # train triton  
        │    ├── selectors.py           # Pure-PyTorch selector 
        │    ├── cross_entropy.py      # Fused cross-entropy, adapted from https://github.com/linkedin/Liger-Kernel
        │
        ├── 📁model
        │    ├── __init__.py          # Model namespace initialization
        │    ├── config.py            # DSALTConfig + from_config
        │    ├── dsalt_lm.py          # Language Model class (causal LM head + wrapper)
        │
        ├── 📁modules
        │    ├── __init__.py          # Exposes the Transformer blocks
        │    ├── dsalt_attention.py   # DSALT attention implementation (window + landmark fusion)
        │    ├── dsalt_transformer.py # Transformer block and SwiGLU FFN
        │
        ├── 📁training
        │    ├── __init__.py          # Exposes the training loop entry point
        │    ├── gpu_auto.py          # Hardware detection, DDP setup, VRAM stats
        │    ├── logging_config.py    # ANSI step formatter, file logging, StepTimer (it/s, tok/s)
        │    ├── trainer.py           # Training loop, backprop, checkpointing, LR scheduling, metrics
        │
        ├── __init__.py               # Library version and top-level imports
        └── py.typed                  # Marker telling mypy the package ships type hints
    ├── .gitignore
    ├── CONTRIBUTING.md
    ├── FEATURE.md
    ├── LICENSE
    ├── MANIFEST.in
    ├── pyproject.toml
    ├── README.md
    ├── release.ps1
    ├── requirements-dev.txt
    ├── requirements.txt
    ├── setup.py
    └── STRUCTURE.md
```



HOW ARE THE FILES IN kernels/ USED?  
window_utils.py        -> used in dsalt_attention.py  
sparse_attn.py         -> used in dsalt_attention.py  
landmark_tokens_ker.py -> used in dsalt_triton_attn.py and dsalt_attention.py  
cross_entropy.py       -> used in dsalt_lm.py  
dsalt_triton_attn.py   -> used in dsalt_attention.py  
dsalt_triton_bwd.py    -> used in dsalt_triton_attn.py  
autotune.py            -> used in dsalt_triton_attn.py and dsalt_triton_bwd.py  
RMSENorm.py            -> used in dsalt_transformer.py and dsalt_lm.py  
dsalt_triton_train.py  -> used in dsalt_attention.py  
selectors.py           -> used in dsalt_attention.py and dsalt_triton_attn.py