# DSALT Feature Catalog

Questo file elenca tutte le funzionalità del progetto DSALT, sia quelle attualmente implementate sia le estensioni naturali/possibili per la libreria. L'obiettivo è avere un unico punto di riferimento per tutte le capacità, i moduli e le estensioni future.

---

## 1. Core Project

- `dsalt` package distribuito come libreria Python su PyPI (`pip install dsalt`).
- Supporto per installazione da sorgente con `pip install -e .`.
- Package metadata moderni basati su `pyproject.toml`.
- Compatibilità Python 3.8+.
- Licenza Apache 2.0.
- Documentazione base in `README.md`.
- File di sviluppo: `requirements.txt`, `requirements-dev.txt`, `Makefile`, `MANIFEST.in`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`.

## 2. API Esportate

- `dsalt.DSALTAttention`
- `dsalt.DSALTTransformer`
- `dsalt.dsalt_attention`
- `dsalt.model.DSALTLMHeadModel`
- `dsalt.training.DSALTTrainer`
- `dsalt.kernels.compute_hybrid_energy_scores`
- `dsalt.kernels.select_landmarks`

## 3. Kernels e Computazione Sparsa

### 3.1. Sparse Attention

- Kernel Triton per DSALT sparse causal attention.
- Pipeline di attenzione che combina:
  - finestra locale causale variabile, token-by-token
  - punti di riferimento (landmark tokens) globali
- Supporto GPU con Triton quando disponibile.
- Fallback CPU via PyTorch quando Triton non è disponibile.
- Supporto mixed precision per FP16/BF16/FP32.
- Test di corrispondenza CPU vs Triton per accuratezza numerica.
- Test di backward gradients per coerenza del gradiente.

### 3.2. Hybrid Energy

- Calcolo punteggio ibrido per landmark selection.
- Normale z-score dei valori energetici.
- Selezione Top-k landmark globali.
- Esclusione dei token già coperti dalla finestra locale.
- Supporto GPU/Triton e fallback CPU.

### 3.3. Window Utils

- `WindowSizePredictor`: modulo per predire finestre di attenzione continue.
- Calcolo delle dimensioni di finestra adattive per ogni token.
- Output di window size continuo per regolarizzazione.

## 4. Modello Transformer

- `DSALTTransformer` come stack di blocchi decoder-only.
- `DSALTBlock` con:
  - pre-norm RMSNorm
  - DSALTAttention multitasca
  - SwiGLU feed-forward
  - dropout e residual connections
- `DSALTLMHeadModel` wrapper LM completo con:
  - embeddings token + positional
  - LM head condiviso con embedding opzionale
  - supporto labels e loss cross-entropy
  - ritorno di `windows` per regolarizzazione

## 5. Training e Addestramento

- `DSALTTrainer` con:
  - ottimizzatore AdamW
  - schedule cosine con warmup lineare
  - clipping dei gradienti
  - mixed precision (`torch.autocast`) per BF16/FP16
  - supporto DDP (DistributedDataParallel)
  - gestione device automatica CPU/GPU
  - checkpointing periodico e caricamento resume
  - validazione con calcolo perplexity
  - regularizzazione di window entropy
- Supporto per dataset PyTorch standard e batching.
- Esempi di training single-GPU e multi-GPU.

## 6. Testing

- Suite di test unitari in `tests/`.
- Test per:
  - attenzione sparsa CPU/Triton
  - coerenza forward/backward
  - hybrid energy
  - wrapper LM
  - allenamento smoke test
- `pyproject.toml` configurato con Pytest.
- Coverage report HTML disponibile.

## 7. Packaging e Distribuzione

- pacchetto PyPI `dsalt` prodotto con `python -m build`.
- file `.whl` e `.tar.gz` generati.
- upload automatico su PyPI con `twine`.
- `setup.py` compatibile per installazione legacy.
- `pyproject.toml` come configurazione principale.
- `Makefile` con comandi:
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

## 8. Sviluppo e Qualità del Codice

- Style formatting con Black.
- Import sorting con isort.
- Linting con Flake8.
- Type checking con Mypy.
- Pre-commit hook segnalato nei dev requirements.
- Gestione `.gitignore` personalizzata per Python, build artifacts e checkpoint.

## 9. Documentazione e Supporto

- README con istruzioni di installazione, quick start, API, testing e citazione.
- CONTRIBUTING per il contributo al progetto.
- CHANGELOG per tenere traccia delle modifiche.
- LICENSE Apache 2.0.

## 10. Funzionalità Implementate Ma Da Rifinire

- `dsalt.training.trainer`:
  - gestione più robusta del checkpointing con `resume_from`
  - supporto DDP e rank-0 logging
  - warning di deprecazione `torch.cuda.amp.GradScaler` da aggiornare
- `tests/test.py`: file `tests/test.py` funzionante ma non allineato al pattern standard `test_*.py`.

## 11. Funzionalità Extra e Possibili Estensioni

### 11.1. Estensioni di modelli

- `DSALTEncoder` per encoder-only o encoder-decoder architetture.
- Implementazione GPT-style completa con tokenizer e config.
- Modello `DSALTForSequenceClassification` / `DSALTForQuestionAnswering`.

### 11.2. Kernel e Prestazioni

- supporto attento per raggruppamento token / cluster attention
- optimizzazioni per sequenze ultra-lunghe (> 8k)
- fallback automatico a FlashAttention o kernel CUDA nativo
- uso avanzato di Triton per layout 2D/3D e kernel mixed-precision

### 11.3. Addestramento avanzato

- learning rate schedule multipli (AdamW, Adam, Adafactor)
- warmup/decay configurabile con lambda scheduler, cosine, linear, step
- gradient checkpointing per risparmiare memoria
- quantizzazione FP8 / int8 during training
- supporto per pipeline parallelism e model parallelism
- logging con Weights & Biases / TensorBoard

### 11.4. Dataset e Data Loading

- classi dataset per testo tokenizzato e autoregressivo
- data collator con masking e padding dinamico
- supporto per dataset HuggingFace
- generazione campioni in inferenza con beam search, top-k, top-p

### 11.5. Documentazione & UX

- documentazione Sphinx / ReadTheDocs
- esempi `examples/` per training e inferenza
- tutorial su how-to per addestramento DSALT
- notebook demo

### 11.6. DevOps e CI/CD

- GitHub Actions per test, lint, build e publish
- packaging per release automatiche su PyPI
- badge di status build/test/coverage in README

### 11.7. UX package

- CLI `dsalt` per:
  - training
  - valutazione
  - inference
  - gestione checkpoint
- configurazione YAML / JSON per esperimenti

## 12. Funzionalità Documentate nel Codebase

- supporto `return_windows` nel modello per regolarizzazione
- selezione landmark globale con broadcasting `landmark_idx`
- gestione window size continua per regolarizzazione
- fallback CPU per tutti i kernel principali
- `py.typed` per segnalare type hints nel package

## 13. Panoramica del Repository

- `dsalt/`
  - `kernels/`: Triton kernels e CPU fallback
  - `modules/`: attenzione, transformer e blocco DSALT
  - `model/`: wrapper LM
  - `training/`: trainer e scheduler
  - `utils/`: spazio riservato per utilità future
- `tests/`: suite di test automatizzati
- `dist/`: release buildate
- `pyproject.toml`: configurazione principale
- `setup.py`: compatibilità legacy
- `README.md`: documentazione principale
- `CONTRIBUTING.md`: linee guida per contribuzione
- `CHANGELOG.md`: cronologia delle release
- `MANIFEST.in`: inclusione file di package

---

## 14. Priorità per il futuro

1. rendere `tests/` completamente compatibile con `pytest` automatico;
2. aggiornare la configurazione `pyproject.toml` per l’extras `all`;
3. aggiungere un entrypoint `dsalt` CLI;
4. completare le funzionalità in `utils/`;
5. estendere la documentazione con esempi e tutorial.

---

Questo documento è pensato per essere la mappa completa delle funzionalità DSALT, utile per definire roadmap, PR, release notes e miglioramenti futuri.