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
from ..kernels.landmark_tokens_ker import hybrid_scores_per_head

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
    # Single source of the formula: see kernels.landmark_tokens_ker
    n_heads      = alpha.shape[0]
    scores, _, _ = hybrid_scores_per_head(x, W_V, alpha, n_heads, dh)
    return scores

def _landmark_ste_term(
    alpha:       torch.Tensor,
    z_x:         torch.Tensor,
    z_v:         torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
) -> torch.Tensor:
    """Straight-through gradient term for the per-head ``alpha`` (§4.3).

    The landmark set is selected by a hard top-k over the hybrid energy score
    ``s = alpha * z_v + (1 - alpha) * z_x`` (eq. 30), which is non-differentiable
    in ``alpha`` (it is an ``argmax``). The attention itself stays exactly the
    paper's ``A(i) = W(i) ∪ L(i)`` (eq. 32): the score does **not** bias the
    attention logits.

    To keep ``alpha`` learnable as the paper requires, we route a gradient
    through a straight-through estimator. We return a scalar that depends on
    ``alpha`` differentiably and equals zero in the forward pass (so it does not
    perturb the output numerically): the mean ``softmax(s)`` mass that falls on
    the landmarks actually selected, minus its own detached value. Maximising it
    pushes each head's ``alpha`` toward the trade-off (output sensitivity vs.
    representational persistence) that makes the chosen landmarks stand out among
    the candidates, which is exactly the per-head adaptivity of §4.3.

    Returns a scalar tensor to be added (as ``term - term.detach()``) to the loss
    path so the forward value is unchanged.
    """
    T       = z_v.shape[0]
    n_heads = alpha.shape[0]
    device  = z_v.device

    # Live (differentiable) hybrid score for every token and head: [T, H].
    scores_live = alpha * z_v + (1.0 - alpha) * z_x.unsqueeze(1)

    # Per-sequence softmax over candidate tokens, then gather the selected ones.
    num_seqs = cu_seqlens.shape[0] - 1
    starts   = cu_seqlens[:-1].to(device)
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(T, device=device) - starts[seq_ids]
    max_len  = int(lens.max())

    # Pad scores to [H, num_seqs, max_len] so the softmax stays within a sequence.
    score_pad = torch.full((n_heads, num_seqs, max_len), float("-inf"), device=device)
    score_pad[:, seq_ids, seq_off] = scores_live.T
    weights = torch.softmax(score_pad, dim=2)  # [H, num_seqs, max_len]

    # lmk_indices: [H, num_seqs, k_lmk], sequence-local positions (-1 = padding).
    valid    = lmk_indices >= 0
    safe_idx = lmk_indices.clamp(min=0, max=max_len - 1)
    sel_w    = torch.gather(weights, 2, safe_idx)        # [H, num_seqs, k_lmk]
    sel_w    = torch.where(valid, sel_w, torch.zeros_like(sel_w))

    denom = valid.sum().clamp(min=1)
    return sel_w.sum() / denom


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
    """DSALT sparse attention: adaptive local window + global landmark tokens (§4).

    Each query's attention set is ``A(i) = W(i) ∪ L(i)``: a causal local window of
    **fixed** size ``(n_min+n_max)//2`` (§4.2) joined with a small set of landmarks
    selected by hybrid energy (learnable per-head ``alpha_w``, §4.3).

    Two paths:
      * **packed** (``cu_seqlens`` provided) → Triton kernel ``dsalt_triton_attention``
        in training, with a masked SDPA fallback if Triton is unavailable;
      * **batched** (``[B, T, d]``) → SDPA over a dense mask, used at inference.

    Implementation note: the window is frozen to a constant value (no learnable
    parameter); the demonstrated adaptivity is that of the per-head ``alpha``.

    In ``eval`` it stores in ``_last_P`` the dense attention matrix of the first
    sequence, used by the trainer's rank/entropy/attention-sink metrics.
    """

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

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

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

    def _aux_zero(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Frozen window: no auxiliary loss on the window. We keep the (out, aux)
        # signature by returning an inert zero term.
        return torch.zeros((), device=device, dtype=dtype)

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

        x_1b    = x[0] if B > 1 else x.squeeze(0)
        w_sizes = compute_window_sizes(x_1b, self.n_min, self.n_max)
        alpha   = self._alpha()
        aux     = self._aux_zero(x.device, x.dtype)

        attn_mask = _build_mask_batched(
            x_1b, w_sizes, self.v_proj.weight.detach(),
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

        w_sizes = compute_window_sizes(x, self.n_min, self.n_max)
        alpha   = self._alpha()
        aux     = self._aux_zero(x.device, x.dtype)

        if _TRITON_OK:
            lmk_indices, z_x, z_v = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(),
                alpha.detach().float(), w_sizes,
                cu_seqlens, self.k_lmk, self.n_min, total_len,
            )
            # The attention matrix is the pure A(i) = W(i) ∪ L(i) of the paper
            # (eq. 32): the hybrid score only *selects* the landmarks, it does not
            # bias the logits. We pass a zero bias to the kernel.
            lmk_bias = torch.zeros_like(lmk_indices, dtype=torch.float32)
            out = dsalt_triton_attention(
                q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens,
            )
            # Straight-through term: keeps alpha learnable (§4.3) without
            # perturbing the forward output (term - term.detach() == 0 in value).
            if self.training and alpha.requires_grad:
                ste = _landmark_ste_term(alpha, z_x, z_v, lmk_indices, cu_seqlens)
                out = out + (ste - ste.detach())
        else:
            attn_mask = _build_mask_packed(
                x, w_sizes, self.v_proj.weight.detach(),
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
            w0     = w_sizes[s0:e0]
            alpha0 = alpha.detach()

            if _TRITON_OK:
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
                for h in range(self.n_heads):
                    valid_pos = lmk_abs[h]
                    causal    = valid_pos.unsqueeze(0) <= rows.unsqueeze(1)
                    in_lmk[h].scatter_(1, valid_pos.unsqueeze(0).expand(T0, -1), causal)

                full_mask = in_win.unsqueeze(0) | in_lmk
            else:
                # Fallback without Triton: dense mask shared across heads [T0, T0]
                full_mask = _build_mask_packed(
                    x[s0:e0], w0, self.v_proj.weight.detach(),
                    alpha0, cu0, T0, self.k_lmk, self.head_dim, device,
                ).unsqueeze(0).expand(self.n_heads, T0, T0)

            sc        = torch.matmul(q0, k0.transpose(-2, -1)) * scale
            additive  = sc.new_full((self.n_heads, T0, T0), float("-inf")).masked_fill_(full_mask, 0.0)
            self._last_P = torch.softmax(sc + additive, dim=-1).detach()
        else:
            self._last_P = None

        return self.out_proj(out.view(total_len, self.d_model)), aux