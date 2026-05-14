import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import (
    compute_window_sizes_triton,
    build_window_mask_triton,
    compute_hybrid_energy_triton,
    apply_yarn_rope_triton,
    select_landmarks,
    sparse_attention_triton,
    sparse_attention_pytorch_fallback,
)


def _yarn_freqs(d_head: int, max_seq_len: int, base: float = 10000.0, scale: float = 1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t        = torch.arange(max_seq_len).float()
    freqs    = torch.outer(t / scale, inv_freq)
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

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads
        self.n_min       = n_min
        self.n_max       = n_max
        self.k_lmk       = k_lmk
        self.max_seq_len = max_seq_len
        self.dropout_p   = dropout
        self.layer_idx   = layer_idx

        self.q_proj      = nn.Linear(d_model, d_model, bias=False)
        self.k_proj      = nn.Linear(d_model, d_model, bias=False)
        self.v_proj      = nn.Linear(d_model, d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.window_proj = nn.Linear(d_model, 1, bias=True)

        self.alpha_raw = nn.Parameter(torch.full((n_heads,), math.log(0.6 / 0.4)))

        cos, sin = _yarn_freqs(self.d_head, max_seq_len, scale=yarn_scale)
        self.register_buffer("yarn_cos", cos)
        self.register_buffer("yarn_sin", sin)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        cos = self.yarn_cos[:N, :D // 2].to(x.device).unsqueeze(0).unsqueeze(0)
        sin = self.yarn_sin[:N, :D // 2].to(x.device).unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _apply_rope_to_landmarks(
        self,
        k: torch.Tensor,
        lmk_indices: torch.Tensor,
    ) -> torch.Tensor:
        B, H, N, D = k.shape
        k_lmk      = lmk_indices.shape[-1]

        flat_idx = lmk_indices.reshape(-1)
        cos_lmk  = self.yarn_cos[flat_idx, :D // 2].to(k.device).reshape(B, H, k_lmk, D // 2)
        sin_lmk  = self.yarn_sin[flat_idx, :D // 2].to(k.device).reshape(B, H, k_lmk, D // 2)

        b_idx = torch.arange(B, device=k.device).view(B, 1, 1).expand(B, H, k_lmk)
        h_idx = torch.arange(H, device=k.device).view(1, H, 1).expand(B, H, k_lmk)

        k_lmk_vecs = k[b_idx, h_idx, lmk_indices]
        x1 = k_lmk_vecs[..., :D // 2]
        x2 = k_lmk_vecs[..., D // 2:]
        k_lmk_rot  = torch.cat([x1 * cos_lmk - x2 * sin_lmk,
                                 x2 * cos_lmk + x1 * sin_lmk], dim=-1)

        k_out = k.clone()
        k_out[b_idx, h_idx, lmk_indices] = k_lmk_rot
        return k_out

    def _compute_lmk_indices(
        self,
        x_norm: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x_norm.shape
        H       = self.n_heads
        device  = x_norm.device

        alpha_vals = torch.sigmoid(self.alpha_raw)

        xv  = (x_norm.float() @ self.v_proj.weight.T)
        x_f = x_norm.float()

        nx   = x_f.norm(dim=-1)
        nv   = xv.norm(dim=-1)
        mu_x = nx.mean(dim=-1, keepdim=True)
        sg_x = nx.std(dim=-1, keepdim=True).clamp(min=1e-8)
        mu_v = nv.mean(dim=-1, keepdim=True)
        sg_v = nv.std(dim=-1, keepdim=True).clamp(min=1e-8)

        z_x = (nx - mu_x) / sg_x
        z_v = (nv - mu_v) / sg_v

        alpha_exp = alpha_vals.view(1, H, 1)
        z_x_exp   = z_x.unsqueeze(1).expand(B, H, N)
        z_v_exp   = z_v.unsqueeze(1).expand(B, H, N)
        scores    = alpha_exp * z_v_exp + (1.0 - alpha_exp) * z_x_exp

        in_any_window = window_mask.any(dim=1)
        outside_exp   = (~in_any_window).unsqueeze(1).expand(B, H, N)
        scores        = torch.where(outside_exp, scores, torch.full_like(scores, float("-inf")))

        k_eff = min(self.k_lmk, int((~in_any_window).sum(dim=-1).min().item()))
        k_eff = max(k_eff, 1)

        _, lmk_indices = torch.topk(scores, k_eff, dim=-1)

        if k_eff < self.k_lmk:
            lmk_indices = F.pad(lmk_indices, (0, self.k_lmk - k_eff), value=0)

        return lmk_indices

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
        w_cont   = compute_window_sizes_triton(w_logits, self.n_min, self.n_max)
        w_int    = w_cont.round().long().clamp(self.n_min, self.n_max)

        window_mask_list = [build_window_mask_triton(w_int[b], N) for b in range(B)]
        window_mask      = torch.stack(window_mask_list, dim=0)

        lmk_indices = self._compute_lmk_indices(x_norm, window_mask)

        k = self._apply_rope_to_landmarks(k, lmk_indices)

        if use_triton and x.is_cuda:
            attn_out = sparse_attention_triton(q, k, v, window_mask, lmk_indices)
        else:
            attn_out = sparse_attention_pytorch_fallback(q, k, v, window_mask, lmk_indices)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)

        if self.dropout_p > 0.0 and self.training:
            attn_out = F.dropout(attn_out, p=self.dropout_p)

        return self.out_proj(attn_out.to(dtype_in))