from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsalt.modules.dsalt_transformer import DSALTTransformer


class DSALTLMHeadModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_min=32,
                 n_max=256, k_lmk=16, alpha=0.6, d_ff=None,
                 max_seq_len=2048, dropout=0.0, use_fa2=True, tie_weights=True):
        super().__init__()
        assert d_model % n_heads == 0
        head_dim = d_model // n_heads
        assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0, \
            "Head dim must be power of 2 and >= 16"
        self.transformer = DSALTTransformer(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, n_min=n_min, n_max=n_max, k_lmk=k_lmk,
            alpha=alpha, d_ff=d_ff, max_seq_len=max_seq_len,
            dropout=dropout, use_fa2=use_fa2, tie_weights=tie_weights,
        )

    def get_input_embeddings(self):
        return self.transformer.tok_emb

    def get_output_embeddings(self):
        return self.transformer.lm_head

    def tie_weights(self):
        self.transformer.lm_head.weight = self.transformer.tok_emb.weight

    def forward(self, input_ids, labels=None, return_windows=False):
        logits, windows = self.transformer(
            input_ids, return_windows=return_windows or labels is not None,
        )
        output = {"logits": logits}
        if return_windows:
            output["windows"] = windows
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1), ignore_index=-100,
            )
            output["loss"] = loss
        return output