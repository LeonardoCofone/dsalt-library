import math
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

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
)

try:
    from ..kernels.dsalt_triton_attn import dsalt_triton_attention
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


@torch.no_grad()
def _hybrid_scores(
    x:     torch.Tensor,   # [T, d_model]
    W_V:   torch.Tensor,   # [d_model, d_model]  (nn.Linear weight convention: [out, in])
    alpha: torch.Tensor,   # [n_heads]
    dh:    int,
) -> torch.Tensor:
    T, d    = x.shape
    n_heads = alpha.shape[0]

    x_norm = x.norm(dim=-1)                                  # [T]
    z_x    = (x_norm - x_norm.mean()) / x_norm.std().clamp(min=1e-6)

    # W_V layout from nn.Linear: [d_out, d_in] = [d_model, d_model]
    # reshape to [n_heads, dh, d_model] so einsum contracts over d_model
    W_V_h = W_V.view(n_heads, dh, d)                        # [H, dh, d_model]
    xwv   = torch.einsum("td,hkd->thk", x, W_V_h).norm(dim=-1)  # [T, H]

    mu_v  = xwv.mean(0, keepdim=True)
    std_v = xwv.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv - mu_v) / std_v                            # [T, H]

    # weighted combination across heads, then mean → [T]
    scores = (alpha.unsqueeze(0) * z_v + (1.0 - alpha.unsqueeze(0)) * z_x.unsqueeze(1)).mean(dim=1)
    return scores


@torch.no_grad()
def _build_mask_batched(
    x:           torch.Tensor,
    window_proj: nn.Linear,
    v_weight:    torch.Tensor,
    alpha:       torch.Tensor,
    n_min:       int,
    n_max:       int,
    k_lmk:       int,
    T:           int,
    device:      torch.device,
) -> torch.Tensor:
    w_sizes     = compute_window_sizes(x, window_proj, n_min, n_max)
    window_mask = build_local_window_mask(T, w_sizes, device)

    dh            = v_weight.shape[0] // alpha.shape[0]
    scores        = _hybrid_scores(x, v_weight, alpha, dh)
    in_window     = window_mask.any(dim=0)
    scores_fil    = scores.masked_fill(in_window, float("-inf"))
    k_act         = min(k_lmk, int((scores_fil > float("-inf")).sum()))

    if k_act > 0:
        _, idx = torch.topk(scores_fil, k_act, sorted=False)
        window_mask[:, idx] = True
    else:
        print("[DSALT] WARNING: no landmark added (k_act=0)")

    return window_mask


@torch.no_grad()
def _build_mask_packed(
    x:           torch.Tensor,
    w_sizes:     torch.Tensor,
    window_proj: nn.Linear,
    v_weight:    torch.Tensor,
    alpha:       torch.Tensor,
    cu_seqlens:  torch.Tensor,
    total_len:   int,
    k_lmk:       int,
    dh:          int,
    n_min:       int,
    device:      torch.device,
) -> torch.Tensor:
    mask     = build_local_window_mask_packed(cu_seqlens, w_sizes, total_len, device)
    scores   = _hybrid_scores(x, v_weight, alpha, dh)

    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total_len, device=device) - starts[seq_ids]
    max_len  = int(lens.max())

    last_off      = lens - 1
    threshold     = (last_off[seq_ids] - n_min + 1).clamp(min=0)
    in_win_global = seq_off >= threshold

    s_masked  = scores.masked_fill(in_win_global, float("-inf"))
    score_pad = torch.full((num_seqs, max_len), float("-inf"), device=device)
    score_pad[seq_ids, seq_off] = s_masked

    k_eff        = min(k_lmk, max_len)
    vals, top    = torch.topk(score_pad, k_eff, dim=1, sorted=False)
    valid        = vals > float("-inf")
    abs_lmk      = (starts.unsqueeze(1) + top).clamp(max=total_len - 1)

    # vectorized landmark injection: scatter True into mask
    for b in range(num_seqs):
        if not valid[b].any():
            continue
        s   = int(cu_seqlens[b])
        e   = int(cu_seqlens[b + 1])
        idx = abs_lmk[b][valid[b]]
        mask[s:e, idx] = True

    return mask


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

        self.q_proj      = nn.Linear(d_model, d_model, bias=False)
        self.k_proj      = nn.Linear(d_model, d_model, bias=False)
        self.v_proj      = nn.Linear(d_model, d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.window_proj = nn.Linear(d_model, 1, bias=True)

        self.alpha_w = nn.Parameter(torch.full((n_heads,), math.log(0.6 / 0.4)))
        self._last_P: torch.Tensor | None = None

        cos, sin = build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"), scale=yarn_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_w)

    @torch.no_grad()
    def _rope(self, n: int, device: torch.device):
        max_n = self.rope_cos.shape[0]
        if n <= max_n:
            return self.rope_cos[:n].to(device), self.rope_sin[:n].to(device)
        return self.rope_cos.to(device), self.rope_sin.to(device)

    @torch.no_grad()
    def warmup(self, device: torch.device) -> None:
        dummy_T = max(self.n_min * 2, 16)
        x_w     = torch.randn(dummy_T, self.d_model, device=device)
        cu_w    = torch.tensor([0, dummy_T], dtype=torch.int32, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = self._packed(x_w, cu_w, dummy_T, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(
        self,
        x:          torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None          = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            return self._packed(x, cu_seqlens, x.shape[0], x.device)
        B, T, _ = x.shape
        return self._batched(x, B, T, x.device)

    def _batched(self, x: torch.Tensor, B: int, T: int, device: torch.device) -> torch.Tensor:
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(T, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        x_1b      = x[0] if B > 1 else x.squeeze(0)
        attn_mask = _build_mask_batched(
            x_1b, self.window_proj, self.v_proj.weight.detach(),
            self._alpha().detach(), self.n_min, self.n_max, self.k_lmk, T, device,
        )

        out = sparse_attention_forward(q, k, v, attn_mask, self.dropout, self.training)

        if not self.training:
            scale    = math.sqrt(self.head_dim)
            sc       = torch.matmul(q[0], k[0].transpose(-2, -1)) / scale
            additive = sc.new_zeros(sc.shape).masked_fill_(~attn_mask.unsqueeze(0).expand_as(sc), float("-inf"))
            self._last_P = torch.softmax(sc + additive, dim=-1).detach()
        else:
            self._last_P = None

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, self.d_model))

    def _packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor, total_len: int, device: torch.device) -> torch.Tensor:
        q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_heads, self.head_dim)

        num_seqs = cu_seqlens.shape[0] - 1
        lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
        starts   = cu_seqlens[:-1].to(device)
        seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
        pos_ids  = torch.arange(total_len, device=device) - starts[seq_ids]

        cos = self.rope_cos[pos_ids].to(device)
        sin = self.rope_sin[pos_ids].to(device)
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        w_sizes = compute_window_sizes(x, self.window_proj, self.n_min, self.n_max).detach()

        if _TRITON_OK and self.training and device.type == "cuda":
            out = dsalt_triton_attention(
                q, k, v, x,
                self.v_proj.weight.detach(),
                self._alpha().detach(),
                w_sizes, cu_seqlens, self.k_lmk, self.n_min,
            )
            self._last_P = None
            return self.out_proj(out.contiguous().view(total_len, self.d_model))

        t0        = time.perf_counter()
        attn_mask = _build_mask_packed(
            x, w_sizes, self.window_proj, self.v_proj.weight.detach(),
            self._alpha().detach(), cu_seqlens, total_len,
            self.k_lmk, self.head_dim, self.n_min, device,
        )
        if self.layer_idx == 0 and self.training:
            print(f"[DSALT] mask_build layer={self.layer_idx} t={time.perf_counter()-t0:.3f}s "
                  f"sparsity={1-attn_mask.float().mean().item():.3f}")

        out = sparse_attention_forward_packed(q, k, v, attn_mask, self.dropout, self.training)

        if not self.training:
            scale = math.sqrt(self.head_dim)
            with torch.no_grad():
                s  = int(cu_seqlens[0]); e = int(cu_seqlens[1])
                sc = torch.matmul(q[s:e].transpose(0, 1), k[s:e].transpose(0, 1).transpose(-2, -1)) / scale
                ad = sc.new_zeros(sc.shape).masked_fill_(~attn_mask[s:e, s:e].unsqueeze(0).expand_as(sc), float("-inf"))
                self._last_P = torch.softmax(sc + ad, dim=-1).detach()
        else:
            self._last_P = None

        return self.out_proj(out.contiguous().view(total_len, self.d_model))