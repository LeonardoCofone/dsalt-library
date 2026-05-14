import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm import RMSENorm
from ..kernels.window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_soft_window_bias,
    build_local_window_mask_packed,
    build_soft_window_bias_packed,
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
    scale  = math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    additive = torch.zeros_like(attn_mask, dtype=q.dtype)
    additive = additive.masked_fill(~attn_mask, float("-inf"))
    scores = scores + additive.unsqueeze(0)
    return torch.softmax(scores, dim=-1)


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

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

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

        self.alpha_w = nn.Parameter(torch.zeros(n_heads))

        self.attn_dropout = nn.Dropout(dropout)

        self._last_P: torch.Tensor | None = None

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
            cos, sin = build_rope_cache(
                seq_len=seq_len,
                head_dim=self.head_dim,
                device=device,
                scale=self.yarn_scale,
            )
            return cos, sin
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def _compute_window_sizes_for_input(self, x_prev: torch.Tensor) -> torch.Tensor:
        flat = x_prev.view(-1, self.d_model) if x_prev.dim() == 3 else x_prev
        return compute_window_sizes(flat, self.window_proj, self.n_min, self.n_max)

    def _build_masks_and_bias(
        self,
        x: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_sizes     = self._compute_window_sizes_for_input(x)
        window_mask = build_local_window_mask(seq_len=seq_len, window_sizes=w_sizes, device=device, causal=True)
        soft_bias   = build_soft_window_bias(seq_len=seq_len, window_sizes=w_sizes, device=device, causal=True)

        x_2d            = x if x.dim() == 2 else x.view(-1, self.d_model)
        W_V             = self.v_proj.weight
        all_head_scores = torch.zeros(seq_len, device=device, dtype=x.dtype)
        for h in range(self.n_heads):
            all_head_scores = all_head_scores + compute_hybrid_scores(x_2d, W_V, self.alpha_w[h].detach())
        all_head_scores = all_head_scores / self.n_heads

        in_window_any = window_mask.any(dim=0)
        landmarks     = select_landmarks(all_head_scores, k=self.k_lmk, exclude_mask=in_window_any)
        lmk_mask      = build_landmark_mask(seq_len=seq_len, landmark_indices=landmarks, device=device)
        full_mask     = merge_window_landmark_mask(window_mask, lmk_mask)

        return full_mask, soft_bias

    def _build_masks_and_bias_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        total_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_sizes     = self._compute_window_sizes_for_input(x)
        window_mask = build_local_window_mask_packed(cu_seqlens=cu_seqlens, window_sizes=w_sizes, total_len=total_len, device=device)
        soft_bias   = build_soft_window_bias_packed(cu_seqlens=cu_seqlens, window_sizes=w_sizes, total_len=total_len, device=device)

        W_V             = self.v_proj.weight
        all_head_scores = torch.zeros(total_len, device=device, dtype=x.dtype)
        for h in range(self.n_heads):
            all_head_scores = all_head_scores + compute_hybrid_scores(x, W_V, self.alpha_w[h].detach())
        all_head_scores = all_head_scores / self.n_heads

        full_mask = window_mask.clone()
        for b in range(len(cu_seqlens) - 1):
            start         = cu_seqlens[b].item()
            end           = cu_seqlens[b + 1].item()
            in_window_any = window_mask[start:end, start:end].any(dim=0)
            local_scores  = all_head_scores[start:end].masked_fill(in_window_any, float("-inf"))
            k_actual      = min(self.k_lmk, (local_scores != float("-inf")).sum().item())
            if k_actual > 0:
                _, lmk_local = torch.topk(local_scores, k=k_actual, sorted=False)
                full_mask[start:end, lmk_local + start] = True

        return full_mask, soft_bias

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

    def _forward_batched(
        self,
        x: torch.Tensor,
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(T, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        x_2d                 = x.view(B * T, self.d_model)
        full_mask, soft_bias = self._build_masks_and_bias(x_2d, T, device)

        alpha       = torch.sigmoid(self.alpha_w).to(dtype=x.dtype)
        soft_bias_t = soft_bias.to(dtype=x.dtype)
        dead_bias   = torch.zeros(T, T, dtype=x.dtype, device=device).masked_fill(~full_mask, float("-inf"))

        outputs = []
        for b in range(B):
            head_outs = []
            for h in range(self.n_heads):
                qh = q[b, h].unsqueeze(0)
                kh = k[b, h].unsqueeze(0)
                vh = v[b, h].unsqueeze(0)

                scores = torch.matmul(qh, kh.transpose(-2, -1)) / self.scale
                scores = scores + dead_bias.unsqueeze(0) + alpha[h] * soft_bias_t.unsqueeze(0)

                attn = torch.softmax(scores, dim=-1)
                attn = self.attn_dropout(attn) if self.training else attn
                head_outs.append(torch.matmul(attn, vh).squeeze(0))

            outputs.append(torch.stack(head_outs, dim=0))

        if not self.training:
            with torch.no_grad():
                self._last_P = _compute_attn_weights(q[0].detach(), k[0].detach(), full_mask)
        else:
            self._last_P = None

        out = torch.stack(outputs, dim=0).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

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
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        full_mask, soft_bias = self._build_masks_and_bias_packed(x, cu_seqlens, total_len, device)

        alpha       = torch.sigmoid(self.alpha_w).to(dtype=x.dtype)
        soft_bias_t = soft_bias.to(dtype=x.dtype)
        dead_bias   = torch.zeros(total_len, total_len, dtype=x.dtype, device=device).masked_fill(~full_mask, float("-inf"))

        head_outputs = []
        for h in range(self.n_heads):
            seq_outs = []
            for b in range(len(cu_seqlens) - 1):
                start = cu_seqlens[b].item()
                end   = cu_seqlens[b + 1].item()

                qb = q[start:end, h].unsqueeze(0)
                kb = k[start:end, h].unsqueeze(0)
                vb = v[start:end, h].unsqueeze(0)

                db = dead_bias[start:end, start:end].unsqueeze(0)
                sb = soft_bias_t[start:end, start:end].unsqueeze(0)

                scores = torch.matmul(qb, kb.transpose(-2, -1)) / self.scale
                scores = scores + db + alpha[h] * sb

                attn = torch.softmax(scores, dim=-1)
                attn = self.attn_dropout(attn) if self.training else attn
                seq_outs.append(torch.matmul(attn, vb).squeeze(0))

            head_outputs.append(torch.cat(seq_outs, dim=0))

        out = torch.stack(head_outputs, dim=1).contiguous().view(total_len, self.d_model)

        if not self.training:
            with torch.no_grad():
                start = cu_seqlens[0].item()
                end   = cu_seqlens[1].item()
                q0    = q[start:end].permute(1, 0, 2).detach()
                k0    = k[start:end].permute(1, 0, 2).detach()
                self._last_P = _compute_attn_weights(q0, k0, full_mask[start:end, start:end])
        else:
            self._last_P = None

        return self.out_proj(out)