import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..kernels.flash_engine.moba_engine import parallel_moba

def _yarn_freqs(d_head, max_seq_len, base=10000.0, scale=1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t / scale, inv_freq)
    return freqs.cos(), freqs.sin()

class DSALTAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_min: int,
        n_max: int,
        k_lmk: int,
        max_seq_len: int,
        dropout: float = 0.0,
        yarn_scale: float = 1.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_min = n_min
        self.n_max = n_max
        self.k_lmk = k_lmk
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.window_proj = nn.Linear(d_model, 1, bias=True)
        self.alpha_raw = nn.Parameter(torch.full((n_heads,), math.log(0.6 / 0.4)))

        cos, sin = _yarn_freqs(self.d_head, max_seq_len, scale=yarn_scale)
        self.register_buffer("yarn_cos", cos)
        self.register_buffer("yarn_sin", sin)

    def _apply_rope(self, x):
        T, H, D = x.shape
        cos = self.yarn_cos[:T, :D // 2].unsqueeze(1)
        sin = self.yarn_sin[:T, :D // 2].unsqueeze(1)
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        
        q = self.q_proj(hidden_states).view(-1, self.n_heads, self.d_head)
        k = self.k_proj(hidden_states).view(-1, self.n_heads, self.d_head)
        v = self.v_proj(hidden_states).view(-1, self.n_heads, self.d_head)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        attn_out = parallel_moba(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            chunk_size=self.n_max,
            topk=self.k_lmk
        )

        attn_out = attn_out.squeeze(0).view(-1, self.d_model)
        output = self.out_proj(attn_out)
        
        if self.dropout_p > 0.0 and self.training:
            output = F.dropout(output, p=self.dropout_p)

        return output.to(dtype)