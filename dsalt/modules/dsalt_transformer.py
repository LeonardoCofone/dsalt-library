import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..kernels.RMSENorm  import RMSENorm
from .dsalt_attention    import DSALTAttention


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop      = nn.Dropout(dropout)
        print(f"--- [SwiGLUFFN] init | d_model={d_model} d_ff={d_ff} dropout={dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        print(f"--- [SwiGLUFFN] forward | x={tuple(x.shape)}")
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        print(f"--- [SwiGLUFFN] gate={tuple(gate.shape)} up={tuple(up.shape)} | gate_norm={gate.norm().item():.4f} up_norm={up.norm().item():.4f}")
        out  = self.drop(self.down_proj(gate * up))
        print(f"--- [SwiGLUFFN] forward DONE | out={tuple(out.shape)} | t={time.perf_counter()-t0:.4f}s")
        return out


class DSALTTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        n_min:       int,
        n_max:       int,
        k_lmk:       int,
        max_seq_len: int,
        d_ff:        int,
        dropout:     float = 0.0,
        yarn_scale:  float = 1.0,
        layer_idx:   int   = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSENorm(d_model)
        self.ffn_norm  = RMSENorm(d_model)
        self.attn = DSALTAttention(
            d_model=d_model, n_heads=n_heads, n_min=n_min, n_max=n_max,
            k_lmk=k_lmk, max_seq_len=max_seq_len, dropout=dropout,
            yarn_scale=yarn_scale, layer_idx=layer_idx,
        )
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        print(f"--- [DSALTTransformerBlock] init | layer={layer_idx} d_model={d_model} n_heads={n_heads} d_ff={d_ff}")

    def forward(
        self,
        x:                      torch.Tensor,
        cu_seqlens:             torch.Tensor | None = None,
        max_seqlen:             int | None          = None,
        gradient_checkpointing: bool                = False,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} forward START | x={tuple(x.shape)} gc={gradient_checkpointing}")

        x_norm_pre_attn = x.norm().item()

        if gradient_checkpointing and self.training:
            print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} usando gradient_checkpointing per attn")
            t1 = time.perf_counter()
            x = x + torch.utils.checkpoint.checkpoint(
                lambda h: self.attn(self.attn_norm(h), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen),
                x, use_reentrant=False,
            )
            print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} attn (gc) DONE | t={time.perf_counter()-t1:.4f}s | out_norm={x.norm().item():.4f}")

            print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} usando gradient_checkpointing per ffn")
            t2 = time.perf_counter()
            x = x + torch.utils.checkpoint.checkpoint(
                lambda h: self.ffn(self.ffn_norm(h)),
                x, use_reentrant=False,
            )
            print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} ffn (gc) DONE | t={time.perf_counter()-t2:.4f}s | out_norm={x.norm().item():.4f}")
        else:
            t1 = time.perf_counter()
            x_attn = self.attn(self.attn_norm(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            x      = x + x_attn
            print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} attn residual DONE | out_norm={x.norm().item():.4f} | t={time.perf_counter()-t1:.4f}s")

            t2 = time.perf_counter()
            x_ffn = self.ffn(self.ffn_norm(x))
            x     = x + x_ffn
            print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} ffn residual DONE | out_norm={x.norm().item():.4f} | t={time.perf_counter()-t2:.4f}s")

        residual_growth = x.norm().item() / (x_norm_pre_attn + 1e-9)
        if residual_growth > 10.0:
            print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} WARNING: residual cresciuto x{residual_growth:.2f} (potenziale instabilità)")

        print(f"--- [DSALTTransformerBlock] layer={self.layer_idx} forward DONE | t_total={time.perf_counter()-t0:.4f}s")
        return x