"""Configurazione serializzabile del modello DSALT."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path


@dataclass
class DSALTConfig:
    """Iperparametri di :class:`~dsalt.model.dsalt_lm.DSALTLMHeadModel`.

    Raccoglie in un solo oggetto serializzabile tutti gli argomenti del modello,
    così da poter salvare/ricaricare la configurazione di un esperimento e
    istanziare il modello con ``DSALTLMHeadModel.from_config(cfg)``.

    Esempio::

        cfg = DSALTConfig(vocab_size=50257, d_model=512, n_layers=6,
                          n_heads=8, n_min=64, n_max=256, k_lmk=16,
                          max_seq_len=1024)
        model = DSALTLMHeadModel.from_config(cfg)
        cfg.save("config.json")
        cfg2 = DSALTConfig.load("config.json")
    """

    # --- obbligatori ---
    vocab_size:  int
    d_model:     int
    n_layers:    int
    n_heads:     int
    n_min:       int
    n_max:       int
    k_lmk:       int
    max_seq_len: int

    # --- opzionali (default allineati al costruttore del modello) ---
    d_ff:               int | None = None
    dropout:            float      = 0.0
    yarn_scale:         float      = 1.0
    tie_weights:        bool       = True
    padding_idx:        int | None = None
    lm_head_chunk_size: int        = 2048
    loss_fn:            str        = "chunked"
    aux_loss_weight:    float      = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) deve essere divisibile per n_heads ({self.n_heads})"
            )
        if not (0 <= self.n_min <= self.n_max):
            raise ValueError(f"richiesto 0 <= n_min <= n_max, ho n_min={self.n_min} n_max={self.n_max}")
        if self.k_lmk < 0:
            raise ValueError(f"k_lmk deve essere >= 0, ho {self.k_lmk}")
        if self.loss_fn not in ("chunked", "liger"):
            raise ValueError(f"loss_fn deve essere 'chunked' o 'liger', ho {self.loss_fn!r}")

    # --- serializzazione ---
    def to_dict(self) -> dict:
        """Restituisce la config come dict (JSON-serializzabile)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DSALTConfig":
        """Costruisce una config da dict, ignorando chiavi sconosciute."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        """Salva la config come JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DSALTConfig":
        """Carica una config da file JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
