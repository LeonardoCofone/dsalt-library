import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.sparse_attn import sparse_attention_forward


def _yarn_freqs(d_head: int, max_seq_len: int, base: float = 10000.0, scale: float = 1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t        = torch.arange(max_seq_len).float()
    freqs    = torch.outer(t / scale, inv_freq)
    return freqs.cos(), freqs.sin()


class DSALTAttention(nn.Module):
    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        n_min:       int,
        n_max:       int,
        k_lmk:       int,
        max_seq_len: int,
        dropout:     float = 0.0,
        yarn_scale:  float = 1.0,
        layer_idx:   int   = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.n_min      = n_min
        self.n_max      = n_max
        self.k_lmk      = k_lmk
        self.max_seq_len = max_seq_len
        self.dropout_p  = dropout
        self.layer_idx  = layer_idx

        self.q_proj      = nn.Linear(d_model, d_model, bias=False)
        self.k_proj      = nn.Linear(d_model, d_model, bias=False)
        self.v_proj      = nn.Linear(d_model, d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.window_proj = nn.Linear(d_model, 1, bias=True)
        self.alpha_raw   = nn.Parameter(torch.full((n_heads,), math.log(0.6 / 0.4)))

        cos, sin = _yarn_freqs(self.d_head, max_seq_len, scale=yarn_scale)
        self.register_buffer("yarn_cos", cos)
        self.register_buffer("yarn_sin", sin)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        cos = self.yarn_cos[:N, :D // 2].unsqueeze(0).unsqueeze(0)
        sin = self.yarn_sin[:N, :D // 2].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _compute_lmk_indices(
        self,
        x_norm: torch.Tensor,
        w_int:  torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x_norm.shape
        H       = self.n_heads
        device  = x_norm.device

        x_f = x_norm.float()
        xv  = x_f @ self.v_proj.weight.float().T           # [B, N, d_model]

        nx  = x_f.norm(dim=-1)                             # [B, N]
        nv  = xv.norm(dim=-1)                              # [B, N]

        z_x = (nx - nx.mean(-1, keepdim=True)) / nx.std(-1, keepdim=True).clamp(min=1e-8)
        z_v = (nv - nv.mean(-1, keepdim=True)) / nv.std(-1, keepdim=True).clamp(min=1e-8)

        alpha  = torch.sigmoid(self.alpha_raw).view(1, H, 1)   # [1, H, 1]
        scores = alpha       * z_v.unsqueeze(1).expand(B, H, N) \
               + (1 - alpha) * z_x.unsqueeze(1).expand(B, H, N)  # [B, H, N]

        # mask tokens inside each token's own window without building [N,N]
        # w_int: [B, N] → for landmark selection we use the per-token max window
        # a token j is "in window of any i" if at least one i has w_int[i] > i-j
        # we approximate with the global max window to stay O(N) not O(N²)
        w_max   = int(w_int.max().item())
        j_idx   = torch.arange(N, device=device)
        # causal: only tokens before position N-1 are valid landmarks anyway;
        # we just mask the last w_max tokens since they're too recent to be global
        # more precisely: for a strict "outside any window" mask we'd need [N,N],
        # but scoring them -inf via causal + w check in sparse_attn is enough.
        # Here we only mask positions that are NEVER outside any window (j >= N-w_max).
        always_in_win = j_idx >= (N - w_max)
        scores = scores.masked_fill(always_in_win.view(1, 1, N), float("-inf"))

        _, lmk_indices = torch.topk(scores, self.k_lmk, dim=-1, sorted=False)
        return lmk_indices                                 # [B, H, k_lmk]

    def forward(
        self,
        x:        torch.Tensor,
        x_norm:   torch.Tensor,
        use_triton: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        dtype_in = x.dtype

        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        w_logits = self.window_proj(x_norm).squeeze(-1)                   # [B, N]
        w_cont   = self.n_min + torch.sigmoid(w_logits) * (self.n_max - self.n_min)
        w_int    = w_cont.round().long().clamp(self.n_min, self.n_max)

        lmk_indices = self._compute_lmk_indices(x_norm, w_int)           # [B, H, k_lmk]
        attn_out    = sparse_attention_forward(q, k, v, w_int, lmk_indices)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        if self.dropout_p > 0.0 and self.training:
            attn_out = F.dropout(attn_out, p=self.dropout_p)
        return self.out_proj(attn_out.to(dtype_in))