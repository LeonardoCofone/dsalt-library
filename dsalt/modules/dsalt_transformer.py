"""
dsalt/modules/dsalt_transformer.py
------------------------------------
Decoder-only Transformer block and stack using DSALT attention.

Architecture per block:
  x = x + DSALT_Attention( RMSNorm(x) )
  x = x + FFN( RMSNorm(x) )

FFN: SwiGLU (used in LLaMA / GPT-NeoX) for better expressiveness.
Norm: RMSNorm (no bias, efficient).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from dsalt.modules.dsalt_attention import DSALTAttention


# ═════════════════════════════════════════════════════════════════════════════
# RMS Norm
# ═════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / norm * self.weight).to(x.dtype)


# ═════════════════════════════════════════════════════════════════════════════
# SwiGLU FFN
# ═════════════════════════════════════════════════════════════════════════════

class SwiGLUFFN(nn.Module):
    """
    FFN with SwiGLU activation:
      FFN(x) = (W1(x) ⊙ SiLU(W2(x))) W3

    Intermediate dimension defaults to ≈ 8/3 * d_model (LLaMA convention)
    rounded to next multiple of 256 for hardware efficiency.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 255) // 256) * 256  # round up to multiple of 256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


# ═════════════════════════════════════════════════════════════════════════════
# DSALT Transformer Block
# ═════════════════════════════════════════════════════════════════════════════

class DSALTBlock(nn.Module):
    """
    Single decoder-only Transformer block with DSALT attention.

    Applies pre-norm (LLaMA-style) for training stability.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_min:    int   = 32,
        n_max:    int   = 256,
        k_lmk:   int   = 16,
        alpha:    float = 0.6,
        d_ff:     Optional[int] = None,
        dropout:  float = 0.0,
        use_fa2:  bool  = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm  = RMSNorm(d_model)

        self.attn = DSALTAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_min=n_min,
            n_max=n_max,
            k_lmk=k_lmk,
            alpha=alpha,
            dropout=dropout,
            use_fa2=use_fa2,
        )
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:      torch.Tensor,              # [B, N, D]
        x_prev: Optional[torch.Tensor],    # [B, N, D] previous layer hidden state
        return_window: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # ── Attention sub-layer ─────────────────────────────────────────────
        residual = x
        x_normed = self.attn_norm(x)
        attn_out, cont_w = self.attn(
            x_normed, x_prev=x_prev, return_window=return_window
        )
        x = residual + self.drop(attn_out)

        # ── FFN sub-layer ───────────────────────────────────────────────────
        x = x + self.drop(self.ffn(self.ffn_norm(x)))

        return x, cont_w


# ═════════════════════════════════════════════════════════════════════════════
# Full DSALT Transformer Stack
# ═════════════════════════════════════════════════════════════════════════════

class DSALTTransformer(nn.Module):
    """
    Decoder-only Transformer stack using DSALT attention.

    Passes x^{l-1} to each block for window and landmark prediction,
    consistent with the paper's formulation.

    Parameters
    ----------
    vocab_size : vocabulary size (for embedding + LM head)
    d_model    : hidden dimension
    n_layers   : number of Transformer blocks
    n_heads    : number of attention heads
    n_min      : minimum adaptive window size
    n_max      : maximum adaptive window size
    k_lmk     : number of landmark tokens
    alpha      : hybrid energy mixing coefficient
    d_ff       : FFN intermediate size (None = auto SwiGLU sizing)
    max_seq_len: max sequence length (for positional encoding)
    dropout    : dropout probability
    use_fa2    : use Flash Attention 2 when applicable
    tie_weights: tie embedding and LM head weights
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int,
        n_layers:    int,
        n_heads:     int,
        n_min:       int   = 32,
        n_max:       int   = 256,
        k_lmk:      int   = 16,
        alpha:       float = 0.6,
        d_ff:        Optional[int] = None,
        max_seq_len: int   = 2048,
        dropout:     float = 0.0,
        use_fa2:     bool  = True,
        tie_weights: bool  = True,
    ):
        super().__init__()
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.max_seq_len = max_seq_len

        # Token + positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DSALTBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_min=n_min,
                n_max=n_max,
                k_lmk=k_lmk,
                alpha=alpha,
                d_ff=d_ff,
                dropout=dropout,
                use_fa2=use_fa2,
            )
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

    def forward(
        self,
        input_ids:      torch.Tensor,          # [B, N]
        return_windows: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Returns
        -------
        logits  : [B, N, vocab_size]
        windows : list of [B, N] continuous window sizes per layer
                  (only if return_windows=True, useful for regularisation)
        """
        B, N = input_ids.shape
        assert N <= self.max_seq_len, \
            f"Sequence length {N} exceeds max_seq_len {self.max_seq_len}"

        # Embeddings
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)  # [1, N]
        x   = self.emb_drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        cont_windows = [] if return_windows else None
        x_prev = None   # layer 0: no previous hidden state

        for block in self.blocks:
            x, cont_w = block(x, x_prev=x_prev, return_window=return_windows)
            x_prev    = x  # pass current output as next block's x_prev
            if return_windows and cont_w is not None:
                cont_windows.append(cont_w)

        x      = self.final_norm(x)
        logits = self.lm_head(x)    # [B, N, vocab_size]

        return logits, cont_windows

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"params={self.count_parameters() / 1e6:.1f}M"
        )