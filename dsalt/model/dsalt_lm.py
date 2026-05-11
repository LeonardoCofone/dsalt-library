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
        head_dim = d_model // n_heads
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

    def forward(self, input_ids, labels=None, return_window=False):
        logits, windows = self.transformer(
            input_ids, return_window=return_window or labels is not None,
        )
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1), ignore_index=-100,
            )
            return logits, windows, loss
        return logits, windows

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=1.0, top_k=50,
                 device=None, tokenizer=None):
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        max_seq_len = self.transformer.pos_emb.weight.shape[0]
        ids = input_ids.to(device)

        for _ in range(max_new_tokens):
            ids_cond = ids[:, -max_seq_len:]
            logits, _ = self(ids_cond)
            logits = logits[:, -1, :] / (temperature + 1e-9)

            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        self.train()

        if tokenizer is not None:
            return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
        return ids