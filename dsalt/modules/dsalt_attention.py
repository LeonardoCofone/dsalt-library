import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.sparse_attn import sparse_attention_forward


def _yarn_freqs(d_head: int, max_seq_len: int, base: float = 10000.0, scale: float = 1.0):
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
        assert d_model % n_heads == 0

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

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        cos = self.yarn_cos[:N, :D // 2].unsqueeze(0).unsqueeze(0)
        sin = self.yarn_sin[:N, :D // 2].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _compute_lmk_indices(
        self,
        x_norm: torch.Tensor,
        w_int: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x_norm.shape
        H = self.n_heads
        device = x_norm.device

        x_f = x_norm.float()
        xv = x_f @ self.v_proj.weight.float().T

        nx = x_f.norm(dim=-1)
        nv = xv.norm(dim=-1)

        mu_x = nx.mean(dim=-1, keepdim=True)
        sg_x = nx.std(dim=-1, keepdim=True).clamp(min=1e-8)
        mu_v = nv.mean(dim=-1, keepdim=True)
        sg_v = nv.std(dim=-1, keepdim=True).clamp(min=1e-8)

        z_x = ((nx - mu_x) / sg_x).unsqueeze(1).expand(B, H, N)
        z_v = ((nv - mu_v) / sg_v).unsqueeze(1).expand(B, H, N)

        alpha = torch.sigmoid(self.alpha_raw).view(1, H, 1)
        scores = alpha * z_v + (1.0 - alpha) * z_x

        i_idx = torch.arange(N, device=device).view(1, 1, N)
        col_off = torch.arange(int(w_int.max().item()), device=device).view(1, 1, 1, -1)
        j_win = (i_idx.unsqueeze(-1) - col_off).clamp(min=0)
        w_exp = w_int.unsqueeze(1).unsqueeze(-1)
        in_win_mask = (col_off < w_exp) & (col_off <= i_idx.unsqueeze(-1))

        in_window = torch.zeros(B, N, dtype=torch.bool, device=device)
        in_window.scatter_(1, j_win.squeeze(1).view(B, -1).clamp(0, N - 1),
                           in_win_mask.squeeze(1).view(B, -1))

        scores = scores.masked_fill(in_window.unsqueeze(1).expand(B, H, N), float("-inf"))

        _, lmk_indices = torch.topk(scores, self.k_lmk, dim=-1, sorted=False)
        return lmk_indices

    def forward(
        self,
        x: torch.Tensor,
        x_norm: torch.Tensor,
        use_triton: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        dtype_in = x.dtype

        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        w_logits = self.window_proj(x_norm).squeeze(-1)
        w_cont = self.n_min + torch.sigmoid(w_logits) * (self.n_max - self.n_min)
        w_int = w_cont.round().long().clamp(self.n_min, self.n_max)

        lmk_indices = self._compute_lmk_indices(x_norm, w_int)

        attn_out = sparse_attention_forward(q, k, v, w_int, lmk_indices)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)

        if self.dropout_p > 0.0 and self.training:
            attn_out = F.dropout(attn_out, p=self.dropout_p)

        return self.out_proj(attn_out.to(dtype_in))