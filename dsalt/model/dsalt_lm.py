"""
dsalt/model/dsalt_lm.py
-----------------------
Language model wrapper exposing DSALTTransformer as an LM head model.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsalt.modules.dsalt_transformer import DSALTTransformer


class DSALTLMHeadModel(nn.Module):
    """Language model wrapper around DSALTTransformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_min: int = 32,
        n_max: int = 256,
        k_lmk: int = 16,
        alpha: float = 0.6,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_fa2: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
        assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0, \
            "Head dim must be a power of 2 and >= 16"

        self.transformer = DSALTTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_min=n_min,
            n_max=n_max,
            k_lmk=k_lmk,
            alpha=alpha,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_fa2=use_fa2,
            tie_weights=tie_weights,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.tok_emb

    def get_output_embeddings(self) -> nn.Linear:
        return self.transformer.lm_head

    def tie_weights(self) -> None:
        self.transformer.lm_head.weight = self.transformer.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_windows: bool = False,
    ) -> Dict[str, Any]:
        logits, windows = self.transformer(
            input_ids,
            return_windows=return_windows or labels is not None,
        )

        output = {"logits": logits}
        if return_windows:
            output["windows"] = windows

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output
