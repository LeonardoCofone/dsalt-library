import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm import RMSENorm
from ..kernels.window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_rope_cache,
    apply_rotary_emb,
)
from ..kernels.landmark_tokens_ker import (
    compute_hybrid_scores,
    select_landmarks,
    build_landmark_mask,
)
from ..kernels.sparse_attn import (
    sparse_attention_forward,
    sparse_attention_forward_packed,
    merge_window_landmark_mask,
)

try:
    from ..kernels.dsalt_triton_attn import dsalt_triton_attention
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


def _compute_attn_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores   = torch.matmul(q, k.transpose(-2, -1)) / scale
    additive = torch.zeros_like(scores).masked_fill(~attn_mask.unsqueeze(0).expand_as(scores), float("-inf"))
    return torch.softmax(scores + additive, dim=-1)


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
        self.head_dim    = d_model // n_heads
        self.n_min       = n_min
        self.n_max       = n_max
        self.k_lmk       = k_lmk
        self.max_seq_len = max_seq_len
        self.dropout     = dropout
        self.yarn_scale  = yarn_scale
        self.layer_idx   = layer_idx
        self.scale       = math.sqrt(self.head_dim)

        self.q_proj      = nn.Linear(d_model, d_model, bias=False)
        self.k_proj      = nn.Linear(d_model, d_model, bias=False)
        self.v_proj      = nn.Linear(d_model, d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.window_proj = nn.Linear(d_model, 1, bias=True)

        self.alpha_w = nn.Parameter(
            torch.full((n_heads,), fill_value=math.log(0.6 / 0.4))
        )

        self.attn_dropout = nn.Dropout(dropout)

        self._last_P:     torch.Tensor | None = None
        self._window_aux: torch.Tensor | None = None

        cos, sin = build_rope_cache(
            seq_len=max_seq_len,
            head_dim=self.head_dim,
            device=torch.device("cpu"),
            scale=yarn_scale,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _get_rope(self, seq_len: int, device: torch.device):
        if seq_len > self.rope_cos.shape[0]:
            return build_rope_cache(seq_len=seq_len, head_dim=self.head_dim, device=device, scale=self.yarn_scale)
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def _window_alpha_aux(self, w_sizes: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        return attn_out + w_sizes.mean() * 0.0 + torch.sigmoid(self.alpha_w).mean() * 0.0

    def _build_full_attn_mask(self, x: torch.Tensor, w_sizes: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        window_mask = build_local_window_mask(seq_len=seq_len, window_sizes=w_sizes, device=device, causal=True)

        dh    = self.head_dim
        W_V   = self.v_proj.weight.detach()
        alpha = torch.sigmoid(self.alpha_w.detach())
        x_2d  = x.view(-1, self.d_model) if x.dim() == 3 else x

        scores = sum(
            compute_hybrid_scores(x_2d, W_V[h * dh:(h + 1) * dh, :], alpha[h])
            for h in range(self.n_heads)
        ) / self.n_heads

        landmarks = select_landmarks(scores, k=self.k_lmk, exclude_mask=window_mask.any(dim=0))
        lmk_mask  = build_landmark_mask(seq_len=seq_len, landmark_indices=landmarks, device=device)
        return merge_window_landmark_mask(window_mask, lmk_mask)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        device    = x.device
        is_packed = cu_seqlens is not None

        if is_packed:
            return self._forward_packed(x, cu_seqlens, x.shape[0], device)
        else:
            B, T, _ = x.shape
            return self._forward_batched(x, B, T, device)

    def _forward_batched(self, x: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(T, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        with torch.no_grad():
            w_sizes   = compute_window_sizes(x.mean(dim=0), self.window_proj, self.n_min, self.n_max)
            attn_mask = self._build_full_attn_mask(x.mean(dim=0), w_sizes, T, device)

        out = sparse_attention_forward(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, training=self.training)

        if not self.training:
            with torch.no_grad():
                self._last_P = _compute_attn_weights(q[0].detach(), k[0].detach(), attn_mask, self.scale)
        else:
            self._last_P = None

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self._window_alpha_aux(w_sizes, self.out_proj(out))

    def _forward_packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor, total_len: int, device: torch.device) -> torch.Tensor:
        q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_heads, self.head_dim)

        cos, sin = self._get_rope(total_len, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        with torch.no_grad():
            w_sizes = compute_window_sizes(x, self.window_proj, self.n_min, self.n_max)

        if _TRITON_OK and self.training and device.type == "cuda":
            W_V   = self.v_proj.weight.detach()
            alpha = torch.sigmoid(self.alpha_w.detach())
            out   = dsalt_triton_attention(
                q, k, v, x, W_V, alpha, w_sizes, cu_seqlens, self.k_lmk
            )
        else:
            with torch.no_grad():
                attn_mask = self._build_packed_attn_mask(x, w_sizes, cu_seqlens, total_len, device)
            out = sparse_attention_forward_packed(
                q, k, v,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=self.max_seq_len,
                dropout_p=self.dropout,
                training=self.training,
            )

        if not self.training:
            with torch.no_grad():
                s    = int(cu_seqlens[0])
                e    = int(cu_seqlens[1])
                attn_mask_0 = self._build_packed_attn_mask(x, w_sizes, cu_seqlens, total_len, device)
                self._last_P = _compute_attn_weights(
                    q[s:e].transpose(0, 1).detach(),
                    k[s:e].transpose(0, 1).detach(),
                    attn_mask_0[s:e, s:e],
                    self.scale,
                )
        else:
            self._last_P = None

        out = self.out_proj(out.contiguous().view(total_len, self.d_model))
        return self._window_alpha_aux(w_sizes, out)

    def _build_packed_attn_mask(self, x, w_sizes, cu_seqlens, total_len, device):
        from ..kernels.window_utils import build_local_window_mask_packed
        window_mask = build_local_window_mask_packed(
            cu_seqlens=cu_seqlens, window_sizes=w_sizes, total_len=total_len, device=device,
        )
        dh    = self.head_dim
        W_V   = self.v_proj.weight.detach()
        alpha = torch.sigmoid(self.alpha_w.detach())

        scores = sum(
            compute_hybrid_scores(x, W_V[h * dh:(h + 1) * dh, :], alpha[h])
            for h in range(self.n_heads)
        ) / self.n_heads

        for b in range(len(cu_seqlens) - 1):
            s    = int(cu_seqlens[b])
            e    = int(cu_seqlens[b + 1])
            sc   = scores[s:e].clone()
            in_w = window_mask[s:e, s:e].any(dim=0)
            sc[in_w] = float("-inf")
            k_act = min(self.k_lmk, int((sc != float("-inf")).sum()))
            if k_act > 0:
                _, idx = torch.topk(sc, k_act, sorted=False)
                window_mask[s:e, idx + s] = True

        return window_mask