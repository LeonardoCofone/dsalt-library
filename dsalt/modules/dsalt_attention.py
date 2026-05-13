import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import (
    TritonRMSNorm,
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
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t / scale, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


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

        self.alpha_raw = nn.Parameter(
            torch.full((n_heads,), math.log(0.6 / 0.4))
        )

        self.norm = TritonRMSNorm(d_model)

        cos, sin = _yarn_freqs(self.d_head, max_seq_len, scale=yarn_scale)

        self.register_buffer("yarn_cos", cos)
        self.register_buffer("yarn_sin", sin)

        self.yarn_scale = yarn_scale

        self.dropout = nn.Dropout(dropout)

    def _get_alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def _apply_local_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        B, H, N, D = x.shape

        cos = self.yarn_cos[:N, : D // 2].to(x.device)
        sin = self.yarn_sin[:N, : D // 2].to(x.device)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]

        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)

    def _apply_yarn_to_landmarks(
        self,
        k: torch.Tensor,
        lmk_indices: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        B, H, N, D = k.shape
        k_lmk_count = lmk_indices.shape[-1]
        k_out = k.clone()
        for b in range(B):
            for h in range(H):
                for ki in range(k_lmk_count):
                    pos = lmk_indices[b, h, ki].item()
                    if pos < 0 or pos >= N:
                        continue
                    tok = k[b, h, pos]
                    cos_pos = self.yarn_cos[pos].to(k.device)
                    sin_pos = self.yarn_sin[pos].to(k.device)
                    rotated = apply_yarn_rope_triton(
                        tok.unsqueeze(0),
                        cos_pos.unsqueeze(0),
                        sin_pos.unsqueeze(0),
                    )
                    k_out[b, h, pos] = rotated.squeeze(0)
        return k_out

    def forward(
        self,
        x: torch.Tensor,
        use_triton: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        dtype_in = x.dtype

        x_norm = self.norm(x)

        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        q = self._apply_local_rope(q, N)
        k = self._apply_local_rope(k, N)

        w_logits = self.window_proj(x_norm.detach()).squeeze(-1)
        w_cont = compute_window_sizes_triton(w_logits, self.n_min, self.n_max)
        w_int = w_cont.round().long().clamp(self.n_min, self.n_max)

        window_mask_list = []
        for b in range(B):
            wm = build_window_mask_triton(w_int[b], N)
            window_mask_list.append(wm)
        window_mask = torch.stack(window_mask_list, dim=0)

        alpha_vals = self._get_alpha()

        lmk_indices_list = []
        for b in range(B):
            head_lmks = []
            xv_b = (x_norm[b] @ self.v_proj.weight.T).float()
            x_b = x_norm[b].float()
            for h in range(self.n_heads):
                alpha_h = alpha_vals[h].item()
                scores = compute_hybrid_energy_triton(x_b, xv_b, alpha_h)
                wm_b = window_mask[b]
                causal_wm = wm_b.any(dim=0)
                lmk_idx = select_landmarks(scores, self.k_lmk, causal_wm)
                pad = self.k_lmk - lmk_idx.shape[0]
                if pad > 0:
                    lmk_idx = F.pad(lmk_idx, (0, pad), value=0)
                head_lmks.append(lmk_idx)
            lmk_indices_list.append(torch.stack(head_lmks, dim=0))
        lmk_indices = torch.stack(lmk_indices_list, dim=0)

        k = self._apply_yarn_to_landmarks(k, lmk_indices, N)

        if use_triton and x.is_cuda:
            attn_out = sparse_attention_triton(q, k, v, window_mask, lmk_indices)
        else:
            attn_out = sparse_attention_pytorch_fallback(q, k, v, window_mask, lmk_indices)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)

        if self.dropout_p > 0.0 and self.training:
            attn_out = F.dropout(attn_out, p=self.dropout_p)

        out = self.out_proj(attn_out.to(dtype_in))
        return out