import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..kernels.RMSENorm import RMSENorm
from .dsalt_attention import DSALTAttention


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


class DSALTTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_min: int,
        n_max: int,
        k_lmk: int,
        max_seq_len: int,
        d_ff: int,
        dropout: float = 0.0,
        yarn_scale: float = 1.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSENorm(d_model)
        self.ffn_norm  = RMSENorm(d_model)
        self.attn = DSALTAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_min=n_min,
            n_max=n_max,
            k_lmk=k_lmk,
            max_seq_len=max_seq_len,
            dropout=dropout,
            yarn_scale=yarn_scale,
            layer_idx=layer_idx,
        )
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:

        if gradient_checkpointing and self.training:
            x = x + torch.utils.checkpoint.checkpoint(
                lambda h: self.attn(self.attn_norm(h), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen),
                x,
                use_reentrant=False,
            )
        else:
            x = x + self.attn(self.attn_norm(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        if gradient_checkpointing and self.training:
            x = x + torch.utils.checkpoint.checkpoint(
                lambda h: self.ffn(self.ffn_norm(h)),
                x,
                use_reentrant=False,
            )
        else:
            x = x + self.ffn(self.ffn_norm(x))

        return x