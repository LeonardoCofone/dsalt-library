```
└── 📁dsalt_pytorch
        ├── .gitignore
        ├── README.md
    └── 📁dsalt
        ├── 📁kernels
        │    ├── __init__.py          # Espone le funzioni di basso livello (scoring e masking)
        │    ├── landmark_tokens_ker.py # Logica di selezione dei Landmark (Energy-based scoring + Top-K)
        │    ├── RMSENorm.py          # Implementazione della Root Mean Square Layer Normalization (stabilità)
        │    ├── sparse_attn.py       # Funzioni atomiche per il calcolo dell'attenzione sparsa (logica core)
        │    ├── window_utils.py      # Utility per gestire la sliding window (masking, padding, indici relativi)  
        │    ├── dsalt_triton_attn.py      #triton  
        │    ├── dsalt_triton_bwd.py      #triton backend (non usato)  
        │    ├── cross_entropy.py      #cross entropy fused da: # https://github.com/linkedin/Liger-Kernel
        │
        ├── 📁model
        │    ├── __init__.py          # Inizializzazione del namespace del modello
        │    ├── dsalt_lm.py          # Definizione della classe Language Model (Causal LM head + Wrapper)
        │
        ├── 📁modules
        │    ├── __init__.py          # Espone i blocchi del Transformer
        │    ├── dsalt_attention.py   # Implementazione dell'attenzione DSALT (Window + Landmark fusion)
        │    ├── dsalt_transformer.py # Definizione del Transformer Block e della struttura a strati (Encoder/Decoder)
        │
        ├── 📁training
        │    ├── __init__.py          # Utility per l'avvio del loop di training
        │    ├── gpu_auto.py          # Rilevamento hardware, setup DDP (DistributedDataParallel) e allocazione VRAM
        │    ├── logging_config.py    # Configurazione di WandB/Tensorboard per monitorare Loss e Rank Collapse
        │    ├── trainer.py           # Loop di allenamento, backpropagation, checkpointing e scheduling LR
        │
        ├── __init__.py               # Versione della libreria e import principali
        └── py.typed                  # Marcatore per indicare a mypy che il pacchetto supporta il type hinting
    ├── .env
    ├── .gitignore
    ├── CHANGELOG.md
    ├── CONTRIBUTING.md
    ├── FEATURE.md
    ├── LICENSE
    ├── Makefile
    ├── MANIFEST.in
    ├── pyproject.toml
    ├── README.md
    ├── requirements-dev.txt
    ├── requirements.txt
    ├── setup.py
    └── STRUCTURE.md
```



COME SONO USATI I FILES IN KERNELS?:
window_utils.py -> usato in dsalt_attention.py
sparse_attn.py -> usato in dsalt_attention.py
landmark_tokens_ker.py -> NON USATO (boh)
cross_entropy.py -> dsalt_lm.py
dsalt_triton_attn.py -> dsalt_attention.py
dsalt_triton_bwd.py -> NON USATO AL MOMENTO
RMSENorm.py -> dsalt_transformer.py e dsalt_lm.py