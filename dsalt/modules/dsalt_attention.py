import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm import RMSENorm
from ..kernels.window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    apply_rotary_emb,
    build_rope_cache,
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


def _compute_attn_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    scale    = math.sqrt(q.shape[-1])
    scores   = torch.matmul(q, k.transpose(-2, -1)) / scale
    additive = torch.zeros_like(scores)
    additive = additive.masked_fill(~attn_mask.unsqueeze(0).expand_as(scores), float("-inf"))
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

    def _get_rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.rope_cos.shape[0]:
            return build_rope_cache(seq_len=seq_len, head_dim=self.head_dim, device=device, scale=self.yarn_scale)
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def _compute_window_sizes_for_input(self, x_prev: torch.Tensor) -> torch.Tensor:
        flat = x_prev.view(-1, self.d_model) if x_prev.dim() == 3 else x_prev
        return compute_window_sizes(flat, self.window_proj, self.n_min, self.n_max)

    def _window_alpha_aux(self, w_sizes: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        w_mean = w_sizes.mean()
        a_mean = torch.sigmoid(self.alpha_w).mean()
        return attn_out + w_mean * 0.0 + a_mean * 0.0

    def _build_full_attn_mask(
        self,
        x: torch.Tensor,
        w_sizes: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        window_mask = build_local_window_mask(
            seq_len=seq_len,
            window_sizes=w_sizes,
            device=device,
            causal=True,
        )

        dh    = self.head_dim
        W_V   = self.v_proj.weight
        alpha = torch.sigmoid(self.alpha_w)

        all_head_scores = torch.zeros(seq_len, device=device, dtype=x.dtype)
        for h in range(self.n_heads):
            W_V_h = W_V[h * dh : (h + 1) * dh, :]
            all_head_scores = all_head_scores + compute_hybrid_scores(x, W_V_h, alpha[h].detach())
        all_head_scores = all_head_scores / self.n_heads

        in_window_any = window_mask.any(dim=0)
        landmarks     = select_landmarks(all_head_scores, k=self.k_lmk, exclude_mask=in_window_any)
        lmk_mask      = build_landmark_mask(seq_len=seq_len, landmark_indices=landmarks, device=device)

        return merge_window_landmark_mask(window_mask, lmk_mask)

    def _build_packed_attn_mask(
        self,
        x: torch.Tensor,
        w_sizes: torch.Tensor,
        cu_seqlens: torch.Tensor,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        window_mask = build_local_window_mask_packed(
            cu_seqlens=cu_seqlens,
            window_sizes=w_sizes,
            total_len=total_len,
            device=device,
        )

        dh    = self.head_dim
        W_V   = self.v_proj.weight
        alpha = torch.sigmoid(self.alpha_w)

        all_head_scores = torch.zeros(total_len, device=device, dtype=x.dtype)
        for h in range(self.n_heads):
            W_V_h = W_V[h * dh : (h + 1) * dh, :]
            all_head_scores = all_head_scores + compute_hybrid_scores(x, W_V_h, alpha[h].detach())
        all_head_scores = all_head_scores / self.n_heads

        for b in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[b])
            end   = int(cu_seqlens[b + 1])

            in_window_any = window_mask[start:end, start:end].any(dim=0)
            local_scores  = all_head_scores[start:end].masked_fill(in_window_any, float("-inf"))
            k_actual      = min(self.k_lmk, int((local_scores != float("-inf")).sum()))
            if k_actual > 0:
                _, lmk_local = torch.topk(local_scores, k=k_actual, sorted=False)
                window_mask[start:end, lmk_local + start] = True

        return window_mask

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
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        x_first = x[0]
        w_sizes  = self._compute_window_sizes_for_input(x_first)

        with torch.no_grad():
            attn_mask = self._build_full_attn_mask(x_first, w_sizes, T, device)

        out = sparse_attention_forward(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )

        if not self.training:
            with torch.no_grad():
                self._last_P = _compute_attn_weights(q[0].detach(), k[0].detach(), attn_mask)
        else:
            self._last_P = None

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return self._window_alpha_aux(w_sizes, out)

    def _forward_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_heads, self.head_dim)

        cos, sin = self._get_rope(total_len, device)
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        w_sizes = self._compute_window_sizes_for_input(x)

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
                start = int(cu_seqlens[0])
                end   = int(cu_seqlens[1])
                q0    = q[start:end].transpose(0, 1).detach()
                k0    = k[start:end].transpose(0, 1).detach()
                self._last_P = _compute_attn_weights(q0, k0, attn_mask[start:end, start:end])
        else:
            self._last_P = None

        out = self.out_proj(out.contiguous().view(total_len, self.d_model))
        return self._window_alpha_aux(w_sizes, out)