import math
import warnings
import torch
import torch.nn as nn

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
    from ..kernels.dsalt_triton_attn import dsalt_triton_attention, _compute_landmark_indices
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


@torch.no_grad()
def _hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
    dh:    int,
) -> torch.Tensor:
    T, d    = x.shape
    n_heads = alpha.shape[0]

    x_norm = x.norm(dim=-1)
    z_x    = (x_norm - x_norm.mean()) / x_norm.std().clamp(min=1e-6)

    xwv   = (x @ W_V.T).view(T, n_heads, dh).norm(dim=-1)
    mu_v  = xwv.mean(0, keepdim=True)
    std_v = xwv.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv - mu_v) / std_v

    return alpha * z_v + (1 - alpha) * z_x.unsqueeze(1)

def _landmark_bias_grad_alpha(
    alpha:       torch.Tensor,
    z_x:         torch.Tensor,
    z_v:         torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
) -> torch.Tensor:
    T       = z_v.shape[0]
    n_heads = alpha.shape[0]
    device  = z_v.device

    scores_live = alpha * z_v + (1.0 - alpha) * z_x.unsqueeze(1)

    starts  = cu_seqlens[:-1].to(device)
    abs_idx = (starts[None, :, None] + lmk_indices.clamp(min=0)).clamp(max=T - 1)
    h_idx   = torch.arange(n_heads, device=device)[:, None, None].expand_as(abs_idx)
    gathered = scores_live[abs_idx, h_idx]

    bias = torch.nn.functional.logsigmoid(gathered)
    return torch.where(lmk_indices >= 0, bias, torch.zeros_like(bias))


@torch.no_grad()
def _build_mask_batched(
    x:        torch.Tensor,
    w_sizes:  torch.Tensor,
    v_weight: torch.Tensor,
    alpha:    torch.Tensor,
    k_lmk:   int,
    T:        int,
    device:   torch.device,
) -> torch.Tensor:
    n_heads = alpha.shape[0]
    dh      = v_weight.shape[0] // n_heads

    window_mask   = build_local_window_mask(T, w_sizes, device)
    scores        = _hybrid_scores(x, v_weight, alpha, dh)
    in_window     = window_mask.any(dim=0)
    scores_fil    = scores.masked_fill(in_window.unsqueeze(1), float("-inf"))
    k_act         = min(k_lmk, int((scores_fil > float("-inf")).sum(dim=0).min().item()))
    landmark_mask = window_mask.unsqueeze(0).expand(n_heads, -1, -1).clone()

    if k_act > 0:
        _, idx  = torch.topk(scores_fil, k_act, dim=0, sorted=False)
        idx_exp = idx.T.unsqueeze(1).expand(n_heads, T, k_act)
        landmark_mask.scatter_(2, idx_exp, True)

    return landmark_mask


@torch.no_grad()
def _build_mask_packed(
    x:          torch.Tensor,
    w_sizes:    torch.Tensor,
    v_weight:   torch.Tensor,
    alpha:      torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_len:  int,
    k_lmk:     int,
    dh:         int,
    device:     torch.device,
) -> torch.Tensor:
    mask     = build_local_window_mask_packed(cu_seqlens, w_sizes, total_len, device)
    scores   = _hybrid_scores(x, v_weight, alpha, dh)

    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total_len, device=device) - starts[seq_ids]
    max_len  = int(lens.max())

    in_win_global = (seq_off < w_sizes.long()).unsqueeze(1)
    s_masked      = scores.masked_fill(in_win_global, float("-inf"))

    score_pad = torch.full((num_seqs, max_len, alpha.shape[0]), float("-inf"), device=device)
    score_pad[seq_ids, seq_off] = s_masked

    k_eff     = min(k_lmk, max_len)
    vals, top = torch.topk(score_pad, k_eff, dim=1, sorted=False)
    valid     = vals > float("-inf")

    abs_lmk = (starts[:, None, None] + top).clamp(max=total_len - 1)

    n_heads = alpha.shape[0]
    rows_per_seq = lens.long()
    row_starts   = cu_seqlens[:-1].to(device).long()

    for b in range(num_seqs):
        if not valid[b].any():
            continue
        s   = int(cu_seqlens[b])
        e   = int(cu_seqlens[b + 1])
        idx = abs_lmk[b][valid[b]].reshape(-1).unique()

        row_idx = torch.arange(s, e, device=device).unsqueeze(1).expand(-1, idx.shape[0])
        col_idx = idx.unsqueeze(0).expand(e - s, -1)
        mask[row_idx, col_idx] = True

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

    def _window_aux(self, w_sizes_soft: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha()
        return (w_sizes_soft.mean() * alpha.mean())

    @torch.no_grad()
    def warmup(self, device: torch.device) -> None:
        was_training = self.training
        self.eval()
        dummy_T = max(self.n_min * 2, 16)
        x_w     = torch.randn(dummy_T, self.d_model, device=device)
        cu_w    = torch.tensor([0, dummy_T], dtype=torch.int32, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = self._packed(x_w, cu_w, dummy_T, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train(was_training)

    def forward(
        self,
        x:          torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None          = None,
        rope_cs:    tuple | None         = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None:
            return self._packed(x, cu_seqlens, x.shape[0], x.device, rope_cs)
        B, T, _ = x.shape
        return self._batched(x, B, T, x.device)

    def _batched(self, x: torch.Tensor, B: int, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(T, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        x_1b         = x[0] if B > 1 else x.squeeze(0)
        w_sizes_soft = compute_window_sizes(x_1b, self.window_proj, self.n_min, self.n_max)
        alpha        = self._alpha()
        aux          = self._window_aux(w_sizes_soft)

        attn_mask = _build_mask_batched(
            x_1b, w_sizes_soft.detach(), self.v_proj.weight.detach(),
            alpha.detach(), self.k_lmk, T, device,
        )

        out = sparse_attention_forward(q, k, v, attn_mask, self.dropout, self.training)

        if not self.training:
            scale    = 1.0 / math.sqrt(self.head_dim)
            sc       = torch.matmul(q[0], k[0].transpose(-2, -1)) * scale
            additive = sc.new_full(sc.shape, float("-inf")).masked_fill_(attn_mask[0], 0.0)
            self._last_P = torch.softmax(sc + additive, dim=-1).detach()
        else:
            self._last_P = None

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, self.d_model)), aux

    def _packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor, total_len: int, device: torch.device, rope_cs: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_heads, self.head_dim)

        if rope_cs is None:
            num_seqs = cu_seqlens.shape[0] - 1
            lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
            starts   = cu_seqlens[:-1].to(device)
            seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
            pos_ids  = torch.arange(total_len, device=device) - starts[seq_ids]
            cos = self.rope_cos[pos_ids]
            sin = self.rope_sin[pos_ids]
        else:
            cos, sin = rope_cs
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        w_sizes_soft = compute_window_sizes(x, self.window_proj, self.n_min, self.n_max)
        alpha        = self._alpha()
        aux          = self._window_aux(w_sizes_soft)

        if _TRITON_OK:
            lmk_indices, z_x, z_v = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(),
                alpha.detach().float(), w_sizes_soft.detach(),
                cu_seqlens, self.k_lmk, self.n_min, total_len,
            )
            lmk_bias = _landmark_bias_grad_alpha(
                alpha, z_x, z_v, lmk_indices, cu_seqlens,
            )
            out = dsalt_triton_attention(
                q, k, v, lmk_indices, lmk_bias, w_sizes_soft.detach(), cu_seqlens,
            )
        else:
            attn_mask = _build_mask_packed(
                x, w_sizes_soft.detach(), self.v_proj.weight.detach(),
                alpha.detach(), cu_seqlens, total_len,
                self.k_lmk, self.head_dim, device,
            )
            out = sparse_attention_forward_packed(
                q, k, v, attn_mask, self.dropout, self.training,
            )

        if not self.training:
            s0    = int(cu_seqlens[0])
            e0    = int(cu_seqlens[1])
            q0    = q[s0:e0].transpose(0, 1)
            k0    = k[s0:e0].transpose(0, 1)
            v0    = v[s0:e0].transpose(0, 1)
            scale = 1.0 / math.sqrt(self.head_dim)
            T0    = e0 - s0

            cu0    = torch.tensor([0, T0], dtype=torch.int32, device=device)
            w0     = w_sizes_soft[s0:e0].detach()
            alpha0 = alpha.detach()

            lmk_idx, _, _ = _compute_landmark_indices(
                x[s0:e0].float().detach(),
                self.v_proj.weight.float().detach(),
                alpha0.float(),
                w0.float(),
                cu0,
                self.k_lmk,
                self.n_min,
                T0,
            )

            rows = torch.arange(T0, device=device)
            cols = torch.arange(T0, device=device)
            w0l  = w0.long()
            in_win = (
                (cols.unsqueeze(0) >= (rows.unsqueeze(1) - w0l.unsqueeze(1) + 1)) &
                (cols.unsqueeze(0) <= rows.unsqueeze(1))
            )

            lmk_abs   = lmk_idx[:, 0, :]
            in_lmk    = torch.zeros(self.n_heads, T0, T0, dtype=torch.bool, device=device)
            h_range   = torch.arange(self.n_heads, device=device)
            for h in range(self.n_heads):
                valid_pos = lmk_abs[h]
                causal    = valid_pos.unsqueeze(0) <= rows.unsqueeze(1)
                in_lmk[h].scatter_(1, valid_pos.unsqueeze(0).expand(T0, -1), causal)

            full_mask = in_win.unsqueeze(0) | in_lmk
            sc        = torch.matmul(q0, k0.transpose(-2, -1)) * scale
            additive  = sc.new_full((self.n_heads, T0, T0), float("-inf")).masked_fill_(full_mask, 0.0)
            self._last_P = torch.softmax(sc + additive, dim=-1).detach()
        else:
            self._last_P = None

        return self.out_proj(out.view(total_len, self.d_model)), aux