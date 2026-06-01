"""Serializable configuration of the DSALT model."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path


@dataclass
class DSALTConfig:
    """Hyperparameters of :class:`~dsalt.model.dsalt_lm.DSALTLMHeadModel`.

    Collects in a single serializable object all the model's arguments, so an
    experiment's configuration can be saved/reloaded and the model instantiated
    with ``DSALTLMHeadModel.from_config(cfg)``.

    Example::

        cfg = DSALTConfig(vocab_size=50257, d_model=512, n_layers=6,
                          n_heads=8, n_min=64, n_max=256, k_lmk=16,
                          max_seq_len=1024)
        model = DSALTLMHeadModel.from_config(cfg)
        cfg.save("config.json")
        cfg2 = DSALTConfig.load("config.json")
    """

    # --- required ---
    vocab_size:  int
    d_model:     int
    n_layers:    int
    n_heads:     int
    n_min:       int
    n_max:       int
    k_lmk:       int
    max_seq_len: int

    # --- optional (defaults aligned with the model constructor) ---
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
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not (0 <= self.n_min <= self.n_max):
            raise ValueError(f"required 0 <= n_min <= n_max, got n_min={self.n_min} n_max={self.n_max}")
        if self.k_lmk < 0:
            raise ValueError(f"k_lmk must be >= 0, got {self.k_lmk}")
        if self.loss_fn not in ("chunked", "liger"):
            raise ValueError(f"loss_fn must be 'chunked' or 'liger', got {self.loss_fn!r}")

    # --- serialization ---
    def to_dict(self) -> dict:
        """Return the config as a dict (JSON-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DSALTConfig":
        """Build a config from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        """Save the config as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DSALTConfig":
        """Load a config from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
