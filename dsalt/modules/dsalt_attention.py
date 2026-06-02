import math
import warnings
import torch
import torch.nn as nn

from ..kernels.window_utils import (
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

try:
    from ..kernels.dsalt_triton_train import dsalt_triton_train_attention
    _TRITON_TRAIN_OK = True
except Exception:
    _TRITON_TRAIN_OK = False


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
    per-token size ``w(i) = n_min + σ(f(x_i))·(n_max-n_min)`` (§4.2), predicted by
    ``win_gate`` ``f`` from the previous-layer hidden state, joined with a small set
    of landmarks selected by the hybrid-energy score (per-head ``alpha_w``, §4.3).

    ``win_gate`` (§4.2) and ``alpha_w`` (§4.3) are **trained**. The selection itself
    (which keys are local / which tokens are landmarks) stays hard and
    non-differentiable, but the gradient reaches the two predictors through soft,
    differentiable *weights*:
      * a **soft window edge**: the window core is hard, but the last ``win_edge``
        boundary keys get weight ``σ((w̃(i)-d)/τ)`` (continuous in ``w̃`` → trains
        ``win_gate``);
      * a **soft landmark weight**: the hard top-k picks the landmarks, but each
        selected landmark's logit is biased by ``log σ(s_j(α)/τ)`` (continuous in
        ``α`` → trains ``alpha_w``).
    The union ``A(i)=W(i)∪L(i)`` (eq. 32) is the elementwise max of the two
    log-biases. ``q·k`` itself is never biased by the score.

    Two regimes:
      * **training** → a differentiable dense path (per sequence) carrying the soft
        edges, both for batched and packed inputs;
      * **inference** → the fast Triton kernel ``dsalt_triton_attention`` (packed)
        or a dense hard mask (batched), using the discretised selectors.

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

        # §4.2 adaptive local window: learned linear projection f: R^d -> R that
        # predicts, per token, w(i) = n_min + σ(f(x_i))·(n_max-n_min) from the
        # previous-layer hidden state (the block input → no circular dependency).
        # f IS trained: the gradient reaches it through the *soft window edge* (see
        # _soft_window_mask), which makes the boundary of the mask differentiable
        # in the continuous w̃(i) while the core stays hard. nn.Linear default init.
        self.win_gate = nn.Linear(d_model, 1, bias=True)

        # §4.3 per-head balance α^(l,h) = σ(alpha_w), init σ⁻¹(0.6). It IS trained:
        # the top-k selection stays hard (non-diff), but each selected landmark's
        # contribution is re-weighted by σ(s_j(α)/τ) inside the softmax, so the
        # gradient flows to α through the *weights*, not through which tokens are
        # picked (see _soft_landmark_logbias).
        self.alpha_w = nn.Parameter(torch.full((n_heads,), math.log(0.6 / 0.4)))

        # Soft-edge temperatures. tau_win: sharpness of the differentiable window
        # border (σ transition width); tau_lmk: sharpness of the α-dependent
        # landmark reweight. win_edge: width (in tokens) of the boundary band that
        # carries a soft, differentiable weight; deeper inside the window the bias
        # is forced to exactly 0 (hard core), so the forward ≈ the hard A(i)=W∪L and
        # the w̃ gradient is concentrated on the boundary.
        self.tau_win  = 1.0
        self.tau_lmk  = 2.0 
        self.win_edge = 4

        self._last_P: torch.Tensor | None = None

        cos, sin = build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"), scale=yarn_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_w)

    def _window_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous window size w̃(i) (§4.2), differentiable in ``win_gate``.

        ``w̃(i) = n_min + σ(f(x_i))·(n_max - n_min)`` from the block input (layer
        ``l-1`` hidden state → no circular dependency). No floor/round here: this
        is the real-valued window used by the *soft edge* so gradients flow to
        ``f``. Returns ``[T]`` float in ``[n_min, n_max]``.
        """
        win_logits = self.win_gate(x).squeeze(-1)                                  # [T]
        return self.n_min + torch.sigmoid(win_logits) * (self.n_max - self.n_min)

    @torch.no_grad()
    def _window_sizes(self, x: torch.Tensor, floor: bool) -> torch.Tensor:
        """Discrete window size w(i) for hard masks (inference / Triton path).

        Same predictor as :meth:`_window_continuous` but discretised: floor at
        inference (paper eq. 29), round otherwise. ``no_grad`` because the integer
        window feeds a hard mask. Returns an int-valued float tensor ``[T]`` (≥1).
        """
        w_cont = self._window_continuous(x)
        w_disc = w_cont.floor() if floor else w_cont.round()
        return w_disc.clamp(min=1)

    def _soft_window_logbias(self, w_cont: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
        """Differentiable log-bias for the local window, hard core + soft edge.

        For a causal pair (query ``i``, key ``j``) let ``d = i - j ≥ 0``. With the
        continuous window ``w̃ = w_cont[i]`` and edge width ``B = win_edge``:

          * ``d < w̃ - B``           → inside core: bias 0  (g ≈ 1, hard)
          * ``w̃ - B ≤ d ≤ w̃ + ...`` → soft edge:  bias log σ((w̃ - d)/τ)
          * far outside              → bias → -inf via the σ tail

        The bias is added to the attention logits; ``∂/∂w̃`` is non-zero only on the
        boundary tokens, so ``win_gate`` is trained through the edge while the bulk
        of the window stays a hard mask (forward ≈ the paper's A(i)=W∪L). Non-causal
        pairs (``d < 0``) get ``-inf``.

        Returns ``[T, T]`` additive log-bias (float32 for numerical headroom).
        """
        idx  = torch.arange(T, device=device)
        d    = (idx.unsqueeze(1) - idx.unsqueeze(0)).float()      # [T,T] = i - j
        w_i  = w_cont.unsqueeze(1)                                # [T,1] broadcast over j
        # σ((w̃ - d)/τ): →1 well inside, →0 well outside, smooth on the border.
        g    = torch.sigmoid((w_i - d) / self.tau_win)           # [T,T] in (0,1)
        logb = torch.log(g.clamp(min=1e-20))
        # Hard core: well inside the window (d ≤ w̃ - win_edge) force bias exactly 0,
        # so the gradient on w̃ is concentrated on the boundary band and the forward
        # there is identical to the hard mask. ``+ 0·logb`` keeps autograd connected
        # outside the core without touching the core values.
        core = d <= (w_i - self.win_edge)
        logb = torch.where(core, torch.zeros_like(logb), logb)
        # Causal hard cut: j > i (d < 0) is never attended.
        logb = logb.masked_fill(d < 0, float("-inf"))
        return logb

    def _soft_landmark_logbias(
        self,
        scores:    torch.Tensor,
        lmk_mask:  torch.Tensor,
        T:         int,
    ) -> torch.Tensor:
        """Differentiable log-bias for landmarks, hard selection + soft α-weight.

        ``scores`` ``[T, H]`` is the hybrid score s_j(α) (§4.3, eq. 30), continuous
        in ``α``. ``lmk_mask`` ``[H, T, T]`` is the *hard* boolean top-k selection
        (detached — which tokens are landmarks does not depend on the gradient).
        For every selected landmark key ``j`` of head ``h`` we add to the logits
        ``log σ(s_j / τ)``, so the landmark's weight inside the softmax depends on
        ``α`` and the gradient reaches ``alpha_w`` through the weight (not through
        the selection). At non-selected entries returns ``-inf`` (no landmark there;
        the caller combines this with the window branch via elementwise max).

        Returns ``[H, T, T]`` additive log-bias.
        """
        # s_j broadcast over the query dimension: weight depends only on the key j.
        w_lmk = torch.log(torch.sigmoid(scores / self.tau_lmk).clamp(min=1e-20))  # [T,H]
        w_lmk = w_lmk.transpose(0, 1).unsqueeze(1).expand(self.n_heads, T, T)     # [H,T,T] over key j
        neg_inf = torch.full_like(w_lmk, float("-inf"))
        return torch.where(lmk_mask, w_lmk, neg_inf)

    @torch.no_grad()
    def _rope(self, n: int, device: torch.device):
        max_n = self.rope_cos.shape[0]
        if n <= max_n:
            return self.rope_cos[:n].to(device), self.rope_sin[:n].to(device)
        return self.rope_cos.to(device), self.rope_sin.to(device)

    def _aux_zero(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # No explicit auxiliary loss: win_gate and alpha_w are trained directly
        # through the soft window edge / soft landmark weight in the main forward.
        # We keep the (out, aux) signature by returning an inert zero.
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
        cu_list:    list | None          = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None:
            return self._packed(x, cu_seqlens, x.shape[0], x.device, rope_cs, cu_list)
        B, T, _ = x.shape
        return self._batched(x, B, T, x.device)

    def _batched(self, x: torch.Tensor, B: int, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(T, device)
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        x_1b = x[0] if B > 1 else x.squeeze(0)
        scale = 1.0 / math.sqrt(self.head_dim)

        # §4.2/§4.3 differentiable sparse mask: hard core + soft, learnable edges.
        # win_logbias [T,T] trains win_gate via the soft window border; lmk_logbias
        # [H,T,T] trains alpha via the σ(s/τ) re-weight of the (hard-selected)
        # landmarks. The union A(i)=W∪L is the elementwise max of the two log-biases
        # (a key is attended if it is in the window OR a landmark, taking whichever
        # gives the larger — i.e. less negative — bias).
        win_logbias, lmk_logbias = self._diff_logbias(x_1b, T, device)             # [T,T],[H,T,T]
        logbias = torch.maximum(win_logbias.unsqueeze(0), lmk_logbias)            # [H,T,T]

        sc  = torch.matmul(q[0], k[0].transpose(-2, -1)).float() * scale          # [H,T,T]
        P   = torch.softmax(sc + logbias, dim=-1)                                  # [H,T,T]
        if self.training and self.dropout > 0:
            P = torch.dropout(P, self.dropout, train=True)
        out = torch.matmul(P.to(v.dtype), v[0])                                    # [H,T,D]
        out = out.unsqueeze(0)                                                     # [1,H,T,D]

        aux = self._aux_zero(x.device, x.dtype)
        self._last_P = P.detach() if not self.training else None

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, self.d_model)), aux

    def _diff_logbias(self, x: torch.Tensor, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the two differentiable log-biases (window, landmarks) for [T] tokens.

        Returns ``(win_logbias [T,T], lmk_logbias [H,T,T])``. The window bias trains
        ``win_gate``; the landmark bias trains ``alpha_w``. The landmark *selection*
        (top-k, excluding the local window) is computed under ``no_grad`` and
        detached — only the per-landmark weight σ(s/τ) carries the α gradient.
        """
        w_cont = self._window_continuous(x)                                       # [T] differentiable
        alpha  = self._alpha()                                                     # [H] differentiable
        scores, _, _ = hybrid_scores_per_head(                                     # [T,H] diff. in α
            x, self.v_proj.weight, alpha, self.n_heads, self.head_dim,
        )

        win_logbias = self._soft_window_logbias(w_cont, T, device)               # [T,T]

        # Hard landmark selection (which tokens), detached. The exclusion j∉W(i) is
        # PER-QUERY, not global (a key local to a near query can still be a landmark
        # for a far query), so we select the global top-k by score and apply the
        # window exclusion per (i,j) when building the mask — exactly like the Triton
        # kernel's `causal_lmk = lmk_pos < win_lo`.
        with torch.no_grad():
            w_disc = w_cont.floor().clamp(min=1)                                  # [T]
            idx    = torch.arange(T, device=device)
            d      = idx.unsqueeze(1) - idx.unsqueeze(0)                          # [T,T] = i - j
            k_act  = min(self.k_lmk, T)
            # is-landmark-key per head [H,T], fully vectorised (no python head loop)
            sel_hk = torch.zeros(self.n_heads, T, dtype=torch.bool, device=device)
            if k_act > 0:
                _, top = torch.topk(scores.detach(), k_act, dim=0, sorted=False)  # [k,H] key indices
                sel_hk.scatter_(1, top.transpose(0, 1), True)                     # [H,T]
            # per-query validity (causal + outside the window) is head-independent
            valid_ij = (d >= 0) & (d >= w_disc.unsqueeze(1))                      # [T,T]
            lmk_mask = sel_hk.unsqueeze(1) & valid_ij.unsqueeze(0)               # [H,T,T]

        # -inf where there is no landmark, σ(s/τ) weight where selected (diff. in α).
        lmk_logbias = self._soft_landmark_logbias(scores, lmk_mask, T)           # [H,T,T]
        return win_logbias, lmk_logbias

    def _packed_train_triton(self, q, k, v, x, cu_seqlens, total_len, device, cu_list=None):
        """Differentiable sparse training via the Triton kernel (GPU only).

        Prepares the three tensors the kernel needs and lets autograd route their
        gradients to the predictors:
          * ``w_cont`` ``[total_len]`` continuous window (diff in ``win_gate``);
          * ``lmk_pos`` ``[H,num_seqs,k]`` hard landmark indices (selection only,
            detached, computed exactly like the inference kernel);
          * ``lmk_logw`` ``[H,num_seqs,k]`` = log σ(s_j(α)/τ_lmk) at the selected
            landmarks (diff in ``alpha`` via the hybrid score).
        ``cu_list`` (host copy of cu_seqlens) is threaded through to avoid D2H syncs.
        """
        from ..kernels.dsalt_triton_attn import _compute_landmark_indices

        w_cont = self._window_continuous(x)                                       # [total_len] diff
        alpha  = self._alpha()                                                     # [H] diff
        # hard landmark selection (detached): reuse the inference selector, which
        # excludes per-query in-window candidates and returns [H,num_seqs,k].
        with torch.no_grad():
            w_disc = w_cont.floor().clamp(min=1)
            lmk_pos, _, _ = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(), alpha.detach().float(),
                w_disc, cu_seqlens, self.k_lmk, self.n_min, total_len, cu_list,
            )                                                                     # [H,num_seqs,k]
        # differentiable score at the selected landmarks → log σ(s/τ) weight.
        scores, _, _ = hybrid_scores_per_head(                                     # [total_len,H] diff in α
            x, self.v_proj.weight, alpha, self.n_heads, self.head_dim,
        )
        num_seqs = cu_seqlens.shape[0] - 1
        starts   = cu_seqlens[:-1].to(device)
        safe_pos = lmk_pos.clamp(min=0)                                            # [H,S,k]
        abs_pos  = starts[None, :, None] + safe_pos                               # [H,S,k] token index
        head_ix  = torch.arange(self.n_heads, device=device)[:, None, None]
        s_sel    = scores[abs_pos, head_ix]                                        # [H,S,k] diff in α
        lmk_logw = torch.log(torch.sigmoid(s_sel / self.tau_lmk).clamp(min=1e-20))
        lmk_logw = torch.where(lmk_pos >= 0, lmk_logw, torch.full_like(lmk_logw, float("-inf")))

        return dsalt_triton_train_attention(
            q, k, v, lmk_pos, lmk_logw, w_cont, cu_seqlens,
            tau_win=self.tau_win, win_edge=float(self.win_edge), cu_list=cu_list,
        )

    def _packed_train(self, q, k, v, x, cu, lens, total_len, device):
        """Differentiable training attention over packed sequences, sync-free.

        When all sequences have the same length (the common PG-19 case, fixed
        ``seq_len``) the whole batch is reshaped to ``[N, L, …]`` and attended in a
        SINGLE batched SDPA call — no python loop, no per-sequence kernel launches.
        Otherwise it falls back to a python loop driven by the host-side ``lens``
        (still one D2H sync total, none inside the loop). Returns ``[total_len,H,D]``.
        """
        H, D = self.n_heads, self.head_dim
        uniform = len(lens) > 0 and all(l == lens[0] for l in lens)

        if uniform:
            N, L = len(lens), lens[0]
            qb = q.view(N, L, H, D).transpose(1, 2)                                # [N,H,L,D]
            kb = k.view(N, L, H, D).transpose(1, 2)
            vb = v.view(N, L, H, D).transpose(1, 2)
            xb = x.view(N, L, self.d_model)                                        # [N,L,d]
            logbias = self._diff_logbias_batched(xb, L, device)                   # [N,H,L,L]
            ob = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, attn_mask=logbias.to(qb.dtype),
                dropout_p=self.dropout if self.dropout > 0 else 0.0,
            )                                                                     # [N,H,L,D]
            return ob.transpose(1, 2).reshape(total_len, H, D)

        out = torch.empty_like(v)                                                  # [total_len,H,D]
        for b in range(len(lens)):
            s, Tb = cu[b], lens[b]
            if Tb == 0:
                continue
            e  = s + Tb
            qb = q[s:e].transpose(0, 1).unsqueeze(0)                               # [1,H,Tb,D]
            kb = k[s:e].transpose(0, 1).unsqueeze(0)
            vb = v[s:e].transpose(0, 1).unsqueeze(0)
            logbias = self._diff_logbias_batched(x[s:e].unsqueeze(0), Tb, device)  # [1,H,Tb,Tb]
            ob = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, attn_mask=logbias.to(qb.dtype),
                dropout_p=self.dropout if self.dropout > 0 else 0.0,
            )
            out[s:e] = ob.squeeze(0).transpose(0, 1)
        return out

    def _diff_logbias_batched(self, x: torch.Tensor, L: int, device: torch.device) -> torch.Tensor:
        """Batched union log-bias ``[N,H,L,L]`` for ``x`` ``[N,L,d]`` (no head loop).

        Same math as :meth:`_diff_logbias` (soft window edge + α-reweighted hard
        landmarks, combined by elementwise max) but fully vectorised over the batch
        and head axes, so the whole packed minibatch is one set of kernels.
        """
        N = x.shape[0]
        H, dh = self.n_heads, self.head_dim
        w_cont = self._window_continuous(x)                                        # [N,L] diff in win_gate
        alpha  = self._alpha()                                                     # [H]   diff in alpha
        # §4.3 hybrid score, standardised PER SEQUENCE (z-score over the L tokens of
        # each sequence, not globally — must match the per-sequence semantics of
        # hybrid_scores_per_head). Vectorised over the batch axis N.
        x_norm = x.norm(dim=-1).float()                                            # [N,L]
        z_x    = (x_norm - x_norm.mean(1, keepdim=True)) / x_norm.std(1, keepdim=True).clamp(min=1e-6)
        xwv    = (x @ self.v_proj.weight.T).view(N, L, H, dh).norm(dim=-1).float() # [N,L,H]
        z_v    = (xwv - xwv.mean(1, keepdim=True)) / xwv.std(1, keepdim=True).clamp(min=1e-6)
        scores = alpha * z_v + (1.0 - alpha) * z_x.unsqueeze(-1)                   # [N,L,H] diff in α

        idx = torch.arange(L, device=device)
        d   = (idx.unsqueeze(1) - idx.unsqueeze(0)).float()                        # [L,L] = i - j

        # --- soft window edge (trains win_gate), batched over N ---
        w_i = w_cont.unsqueeze(-1)                                                 # [N,L,1] over key j
        g   = torch.sigmoid((w_i - d.unsqueeze(0)) / self.tau_win)                # [N,L,L]
        win_lb = torch.log(g.clamp(min=1e-20))
        core   = d.unsqueeze(0) <= (w_i - self.win_edge)
        win_lb = torch.where(core, torch.zeros_like(win_lb), win_lb)
        win_lb = win_lb.masked_fill((d < 0).unsqueeze(0), float("-inf"))          # [N,L,L]

        # --- hard landmark selection + soft α weight (trains alpha_w), batched ---
        with torch.no_grad():
            w_disc = w_cont.floor().clamp(min=1)                                   # [N,L]
            k_act  = min(self.k_lmk, L)
            sel    = torch.zeros(N, self.n_heads, L, dtype=torch.bool, device=device)
            if k_act > 0:
                # top-k over the key axis (dim=1) per (batch, head)
                _, top = torch.topk(scores, k_act, dim=1, sorted=False)           # [N,k,H]
                sel.scatter_(2, top.transpose(1, 2), True)                         # [N,H,L]
            causal_outwin = (d >= 0).unsqueeze(0) & (d >= w_disc.unsqueeze(-1))    # [N,L,L]
            lmk_mask = sel.unsqueeze(2) & causal_outwin.unsqueeze(1)              # [N,H,L,L]

        w_lmk = torch.log(torch.sigmoid(scores / self.tau_lmk).clamp(min=1e-20))  # [N,L,H]
        w_lmk = w_lmk.permute(0, 2, 1).unsqueeze(2).expand(N, self.n_heads, L, L)  # [N,H,L,L] over key j
        lmk_lb = torch.where(lmk_mask, w_lmk, torch.full_like(w_lmk, float("-inf")))

        return torch.maximum(win_lb.unsqueeze(1), lmk_lb)                          # [N,H,L,L]

    def _packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor, total_len: int, device: torch.device, rope_cs: tuple | None = None, cu_list: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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

        aux = self._aux_zero(x.device, x.dtype)

        # TRAINING: differentiable dense path, per sequence. This is the ONLY way the
        # soft window edge (§4.2) and the α-reweight of landmarks (§4.3) can train
        # win_gate / alpha_w — the Triton kernel's hand-written backward does not
        # carry those gradients. Slower than Triton but exact and gradcheck-verified.
        # INFERENCE: fall through to the fast Triton (or SDPA) selector path below.
        if self.training:
            self._last_P = None
            # Fast path: differentiable sparse Triton kernel (only w(i)+k tokens per
            # query, no dense [N,H,L,L]). Carries the gradients to win_gate (soft
            # window edge) and alpha (soft landmark weight) directly in the kernel.
            if _TRITON_TRAIN_OK and device.type == "cuda":
                out = self._packed_train_triton(q, k, v, x, cu_seqlens, total_len, device, cu_list)
                return self.out_proj(out.view(total_len, self.d_model)), aux
            # Fallback (CPU / no Triton): dense SDPA path, same math, gradcheck-tested.
            cu   = cu_list if cu_list is not None else cu_seqlens.detach().to("cpu").tolist()
            lens = [cu[b + 1] - cu[b] for b in range(len(cu) - 1)]
            out  = self._packed_train(q, k, v, x, cu, lens, total_len, device)
            return self.out_proj(out.view(total_len, self.d_model)), aux

        # ---- inference (selectors, no gradient) ----
        w_sizes = self._window_sizes(x, floor=True)
        alpha   = self._alpha()

        if _TRITON_OK:
            lmk_indices, _, _ = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(),
                alpha.detach().float(), w_sizes,
                cu_seqlens, self.k_lmk, self.n_min, total_len,
            )
            # Pure A(i)=W∪L (eq. 32): the score only selects, zero logit bias.
            lmk_bias = torch.zeros_like(lmk_indices, dtype=torch.float32)
            out = dsalt_triton_attention(
                q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens,
            )
        else:
            attn_mask = _build_mask_packed(
                x, w_sizes, self.v_proj.weight.detach(),
                alpha.detach(), cu_seqlens, total_len,
                self.k_lmk, self.head_dim, device,
            )
            out = sparse_attention_forward_packed(
                q, k, v, attn_mask, self.dropout, self.training,
            )

        if True:
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