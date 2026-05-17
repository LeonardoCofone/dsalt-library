import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm import RMSENorm
from ..kernels.window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    build_rope_cache,
    apply_rotary_emb,
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
    q:         torch.Tensor,
    k:         torch.Tensor,
    attn_mask: torch.Tensor,
    scale:     float,
) -> torch.Tensor:
    scores   = torch.matmul(q, k.transpose(-2, -1)) / scale
    additive = torch.zeros_like(scores).masked_fill(~attn_mask.unsqueeze(0).expand_as(scores), float("-inf"))
    return torch.softmax(scores + additive, dim=-1)


def _hybrid_scores_all_heads(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
    dh:    int,
) -> torch.Tensor:
    n_heads = alpha.shape[0]
    x_norm  = x.norm(dim=-1)
    mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
    z_x = (x_norm - mu_x) / std_x

    W_V_heads = W_V.view(n_heads, dh, -1)
    xwv_norms = (x @ W_V_heads.transpose(1, 2)).norm(dim=-1)
    mu_v  = xwv_norms.mean(0, keepdim=True)
    std_v = xwv_norms.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv_norms - mu_v) / std_v

    a      = alpha.unsqueeze(0)
    scores = (a * z_v + (1.0 - a) * z_x.unsqueeze(1)).mean(dim=1)
    return scores


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

        cos, sin = build_rope_cache(
            seq_len=max_seq_len, head_dim=self.head_dim,
            device=torch.device("cpu"), scale=yarn_scale,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _get_rope(self, seq_len: int, device: torch.device):
        if seq_len > self.rope_cos.shape[0]:
            return build_rope_cache(seq_len=seq_len, head_dim=self.head_dim, device=device, scale=self.yarn_scale)
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def _alpha_sigmoid(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_w)

    def _get_W_V_detached(self) -> torch.Tensor:
        return self.v_proj.weight.detach()

    def _build_full_attn_mask(
        self,
        x:       torch.Tensor,
        w_sizes: torch.Tensor,
        seq_len: int,
        device:  torch.device,
    ) -> torch.Tensor:
        window_mask = build_local_window_mask(seq_len=seq_len, window_sizes=w_sizes, device=device, causal=True)
        W_V         = self._get_W_V_detached()
        alpha       = self._alpha_sigmoid().detach()
        scores      = _hybrid_scores_all_heads(x, W_V, alpha, self.head_dim)
        in_window   = window_mask.any(dim=0)
        scores      = scores.masked_fill(in_window, float("-inf"))
        k_act       = min(self.k_lmk, int((scores != float("-inf")).sum()))
        lmk_mask    = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        if k_act > 0:
            _, idx           = torch.topk(scores, k_act, sorted=False)
            lmk_mask[:, idx] = True
        return window_mask | lmk_mask

    def _build_packed_attn_mask(
        self,
        x:          torch.Tensor,
        w_sizes:    torch.Tensor,
        cu_seqlens: torch.Tensor,
        total_len:  int,
        device:     torch.device,
    ) -> torch.Tensor:
        num_seqs    = cu_seqlens.shape[0] - 1
        lens        = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
        window_mask = build_local_window_mask_packed(
            cu_seqlens=cu_seqlens, window_sizes=w_sizes, total_len=total_len, device=device,
        )

        W_V    = self._get_W_V_detached()
        alpha  = self._alpha_sigmoid().detach()
        scores = _hybrid_scores_all_heads(x, W_V, alpha, self.head_dim)

        max_len  = int(lens.max())
        seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
        seq_off  = torch.arange(total_len, device=device) - cu_seqlens[:-1].to(device)[seq_ids]
        starts   = cu_seqlens[:-1].to(device)

        # exclude tokens already covered by their own window
        in_window = window_mask.any(dim=0)  # [total_len]
        scores    = scores.masked_fill(in_window, float("-inf"))

        score_pad = torch.full((num_seqs, max_len), float("-inf"), device=device)
        score_pad[seq_ids, seq_off] = scores

        k_eff      = min(self.k_lmk, max_len)
        _, top_loc = torch.topk(score_pad, k_eff, dim=1, sorted=False)

        # valid only where score != -inf
        valid_lmk = score_pad.gather(1, top_loc) != float("-inf")  # [num_seqs, k_eff]

        # absolute column indices for landmarks
        abs_lmk = (starts.unsqueeze(1) + top_loc).clamp(max=total_len - 1)  # [num_seqs, k_eff]

        # vectorised: for every row in [s:e], set landmark columns True
        # build row ranges per sequence
        row_starts = starts                          # [num_seqs]
        row_ends   = cu_seqlens[1:].to(device)       # [num_seqs]

        # expand landmark cols: [total_len] row -> which seq
        # Use scatter on the flat [total, total] mask
        for b in range(num_seqs):
            vl = valid_lmk[b]
            if not vl.any():
                continue
            s   = int(row_starts[b])
            e   = int(row_ends[b])
            idx = abs_lmk[b][vl]               # [n_valid]
            window_mask[s:e, idx] = True

        return window_mask

    def forward(
        self,
        x:          torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        device    = x.device
        is_packed = cu_seqlens is not None
        if is_packed:
            return self._forward_packed(x, cu_seqlens, x.shape[0], device)
        B, T, _ = x.shape
        return self._forward_batched(x, B, T, device)

    def _forward_batched(
        self,
        x:      torch.Tensor,
        B:      int,
        T:      int,
        device: torch.device,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(T, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        # Use per-token window sizes, not mean over batch (which was wrong)
        with torch.no_grad():
            # x is [B, T, d] — use first batch item or mean over B, NOT mean over T
            x_2d      = x.view(B * T, self.d_model)
            w_sizes   = compute_window_sizes(x_2d, self.window_proj, self.n_min, self.n_max)[:T]
            attn_mask = self._build_full_attn_mask(x_2d[:T], w_sizes, T, device)

        out = sparse_attention_forward(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, training=self.training)

        if not self.training:
            with torch.no_grad():
                self._last_P = _compute_attn_weights(q[0].detach(), k[0].detach(), attn_mask, self.scale)
        else:
            self._last_P = None

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        # keep gradient path for alpha and w_sizes via a zero-multiply trick
        aux = self._alpha_sigmoid().mean() * 0.0 + w_sizes.mean() * 0.0
        return self.out_proj(out) + aux

    def _forward_packed(
        self,
        x:          torch.Tensor,
        cu_seqlens: torch.Tensor,
        total_len:  int,
        device:     torch.device,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_heads, self.head_dim)

        cos, sin = self._get_rope(total_len, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        with torch.no_grad():
            w_sizes = compute_window_sizes(x, self.window_proj, self.n_min, self.n_max)

        attn_mask = None

        if _TRITON_OK and self.training and device.type == "cuda":
            W_V   = self._get_W_V_detached()
            alpha = self._alpha_sigmoid().detach()
            out   = dsalt_triton_attention(q, k, v, x, W_V, alpha, w_sizes, cu_seqlens, self.k_lmk)
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
                if attn_mask is None:
                    attn_mask = self._build_packed_attn_mask(x, w_sizes, cu_seqlens, total_len, device)
                s = int(cu_seqlens[0])
                e = int(cu_seqlens[1])
                self._last_P = _compute_attn_weights(
                    q[s:e].transpose(0, 1).detach(),
                    k[s:e].transpose(0, 1).detach(),
                    attn_mask[s:e, s:e],
                    self.scale,
                )
        else:
            self._last_P = None

        aux = self._alpha_sigmoid().mean() * 0.0 + w_sizes.mean() * 0.0
        return self.out_proj(out.contiguous().view(total_len, self.d_model)) + aux