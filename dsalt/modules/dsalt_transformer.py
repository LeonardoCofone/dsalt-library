import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from dsalt.modules.dsalt_attention import DSALTAttention


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / norm * self.weight).to(x.dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class DSALTBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_min=32, n_max=256, k_lmk=16,
                 alpha=0.6, d_ff=None, dropout=0.0, use_fa2=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm  = RMSNorm(d_model)
        self.attn = DSALTAttention(d_model, n_heads, n_min, n_max, k_lmk,
                                   alpha, dropout, use_fa2)
        self.ffn  = SwiGLUFFN(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_prev=None, return_window=False):
        print(f"[BLOCK] forward start x.shape={x.shape} device={x.device}", flush=True)
        residual = x
        attn_out, cont_w = self.attn(self.attn_norm(x), x_prev=x_prev,
                                      return_window=return_window)
        print(f"[BLOCK] attn done attn_out.shape={attn_out.shape}", flush=True)
        x = residual + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        print(f"[BLOCK] ffn done", flush=True)
        return x, cont_w


class DSALTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_min=32,
                 n_max=256, k_lmk=16, alpha=0.6, d_ff=None,
                 max_seq_len=2048, dropout=0.0, use_fa2=True, tie_weights=True):
        super().__init__()
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DSALTBlock(d_model, n_heads, n_min, n_max, k_lmk, alpha,
                       d_ff, dropout, use_fa2)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids, return_window=False):
        B, N = input_ids.shape
        assert N <= self.max_seq_len
        print(f"[TRANS] forward start B={B} N={N} device={input_ids.device}", flush=True)
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x   = self.emb_drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        print(f"[TRANS] embeddings done x.shape={x.shape}", flush=True)
        cont_windows = [] if return_window else None
        x_prev = None
        for i, block in enumerate(self.blocks):
            print(f"[TRANS] block {i} start", flush=True)
            x, cont_w = block(x, x_prev=x_prev, return_window=return_window)
            x_prev    = x
            if return_window and cont_w is not None:
                cont_windows.append(cont_w)
            print(f"[TRANS] block {i} done", flush=True)
        x      = self.final_norm(x)
        logits = self.lm_head(x)
        print(f"[TRANS] final done logits.shape={logits.shape}", flush=True)
        return logits, cont_windows

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self):
        return (f"d_model={self.d_model}, n_layers={self.n_layers}, "
                f"params={self.count_parameters() / 1e6:.1f}M")