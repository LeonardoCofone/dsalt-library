"""DSALT sparse attention module: adaptive local window ∪ global landmarks (§4).

Wires together the differentiable selectors (soft window edge trains ``win_gate``,
soft landmark re-weight trains per-head ``alpha``) and dispatches to the Triton
training/inference kernels on CUDA or a masked-SDPA fallback elsewhere. The selector
math lives once in ``selectors`` / ``landmark_tokens_ker`` and is shared by every path.
"""

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
# Selectors are triton-free → always importable (CPU fallback / verification).
from ..kernels.selectors import _compute_landmark_indices

try:
    from ..kernels.dsalt_triton_attn import dsalt_triton_attention
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

try:
    from ..kernels.dsalt_triton_train import dsalt_triton_train_attention
    _TRITON_TRAIN_OK = True
except Exception:
    _TRITON_TRAIN_OK = False


def _dynamo_opaque(fn):
    """Make ``fn`` opaque to torch.compile/Dynamo (graph-break, run eager).

    Same helper as in ``dsalt_triton_train``. Autograd is unaffected (gradients
    flow normally), only the *tracing* is skipped. Degrades to identity on torch
    builds without ``_dynamo``.
    """
    disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(fn) if disable is not None else fn


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
        (detached, which tokens are landmarks does not depend on the gradient).
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
        # gives the larger, i.e. less negative, bias).
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
        detached, only the per-landmark weight σ(s/τ) carries the α gradient.
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
        # window exclusion per (i,j) when building the mask, exactly like the Triton
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
        w_cont, lmk_pos, lmk_logw = self._train_selectors(x, cu_seqlens, total_len, device, cu_list)
        return dsalt_triton_train_attention(
            q, k, v, lmk_pos, lmk_logw, w_cont, cu_seqlens,
            tau_win=self.tau_win, win_edge=float(self.win_edge),
            n_max=self.n_max, cu_list=cu_list,
        )

    @_dynamo_opaque
    def _train_selectors(self, x, cu_seqlens, total_len, device, cu_list=None):
        """Shared selector prelude for BOTH the Triton kernel and the dense fallback.

        Marked ``@_dynamo_opaque``: this prelude branches on data-dependent PYTHON
        values derived from ``cu_list`` (per-step ``total_len`` / ``num_seqs`` /
        ``max_len`` via ``int()``/``max()``/``range()`` in ``selectors``). Under
        torch.compile those become guards on concrete python ints that change every
        step (varlen packing), forcing a full re-trace per step — which dominated
        wall-clock (~82s/step) while the in-step timer still read ~17s of compute.
        Running it eager (it is pure-torch, ~5ms/call) keeps the graph stable; the
        ``w_cont`` / ``lmk_logw`` gradients to ``win_gate`` / ``alpha`` are
        unaffected (Dynamo opacity skips tracing, not autograd), exactly like the
        already-opaque ``dsalt_triton_train_attention`` it feeds.

        Returns ``(w_cont [total_len], lmk_pos [H,S,k], lmk_logw [H,S,k])``, the
        SAME tensors both paths must consume, so the dense verification reference
        and the kernel select identical landmarks (the previous mismatch came from
        the fallback using its own top-k instead of these).

        Note: ``hybrid_scores_per_head`` (the ``x @ W_V`` GEMM over all tokens) runs
        twice here, once detached for the hard indices, once differentiable for the
        landmark logits. Profiled at ~5 ms/call eager (~32 ms/step over 6 layers),
        but torch.compile fuses it into the surrounding graph, so a manual fusion has
        low ROI in the real (compiled) training path. Left as-is by design.
        """
        w_cont = self._window_continuous(x)                                       # [total_len] diff
        alpha  = self._alpha()                                                     # [H] diff
        with torch.no_grad():
            w_disc = w_cont.floor().clamp(min=1)
            lmk_pos, _, _ = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(), alpha.detach().float(),
                w_disc, cu_seqlens, self.k_lmk, self.n_min, total_len, cu_list,
            )                                                                     # [H,num_seqs,k]
        scores, _, _ = hybrid_scores_per_head(                                     # [total_len,H] diff in α
            x, self.v_proj.weight, alpha, self.n_heads, self.head_dim,
        )
        starts   = cu_seqlens[:-1].to(device)
        safe_pos = lmk_pos.clamp(min=0)                                            # [H,S,k]
        abs_pos  = starts[None, :, None] + safe_pos                               # [H,S,k]
        head_ix  = torch.arange(self.n_heads, device=device)[:, None, None]
        s_sel    = scores[abs_pos, head_ix]                                        # [H,S,k] diff in α
        lmk_logw = torch.log(torch.sigmoid(s_sel / self.tau_lmk).clamp(min=1e-20))
        lmk_logw = torch.where(lmk_pos >= 0, lmk_logw, torch.full_like(lmk_logw, float("-inf")))
        return w_cont, lmk_pos, lmk_logw

    def _packed_train(self, q, k, v, x, cu, lens, total_len, device, cu_seqlens=None):
        """Dense differentiable training attention, reference for the Triton kernel.

        Builds the SAME log-bias the Triton kernel computes (hard band ``d≤w̃`` +
        soft edge; landmarks at the kernel-selected ``lmk_pos`` weighted by
        ``lmk_logw``; union = elementwise max) and runs it through SDPA. Consuming
        the kernel's exact selectors (via :meth:`_train_selectors`) is what makes
        this a faithful reference, uniform-length only (the verify/PG-19 case).
        """
        H, D = self.n_heads, self.head_dim
        uniform = len(lens) > 0 and all(l == lens[0] for l in lens)
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
        if not uniform:
            raise NotImplementedError(
                "_packed_train dense fallback supports uniform-length packing only "
                "(used for CPU tests / kernel verification)."
            )

        N, L = len(lens), lens[0]
        # Same selectors the kernel uses → identical landmarks & weights.
        w_cont, lmk_pos, lmk_logw = self._train_selectors(x, cu_seqlens, total_len, device, cu)
        logbias = self._dense_logbias_from_kernel(w_cont, lmk_pos, lmk_logw, N, L, device)  # [N,H,L,L]

        qb = q.view(N, L, H, D).transpose(1, 2)                                    # [N,H,L,D]
        kb = k.view(N, L, H, D).transpose(1, 2)
        vb = v.view(N, L, H, D).transpose(1, 2)
        ob = torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=logbias.to(qb.dtype),
            dropout_p=self.dropout if self.dropout > 0 else 0.0,
        )                                                                         # [N,H,L,D]
        return ob.transpose(1, 2).reshape(total_len, H, D)

    def _dense_logbias_from_kernel(self, w_cont, lmk_pos, lmk_logw, N, L, device):
        """Dense ``[N,H,L,L]`` log-bias replicating the Triton kernel math exactly.

        * window: hard band ``0 ≤ d ≤ w̃`` with soft edge ``log σ((w̃−d)/τ_win)``
          (0 in the core ``d ≤ w̃−win_edge``);
        * landmarks: at the (per-head, per-seq) ``lmk_pos`` add ``lmk_logw``, valid
          only where ``lmk_pos ≤ i`` and ``lmk_pos < i − w̃`` (outside the band);
        * union = elementwise max of the two.
        """
        H = self.n_heads
        w_seq = w_cont.view(N, L)                                                  # [N,L] per query i
        idx   = torch.arange(L, device=device)
        d     = (idx.unsqueeze(1) - idx.unsqueeze(0)).float()                      # [L,L] = i - j

        # --- window band (trains win_gate) ---
        w_i    = w_seq.unsqueeze(-1)                                               # [N,L,1]
        z      = (w_i - d.unsqueeze(0)) / self.tau_win
        win_lb = torch.log(torch.sigmoid(z).clamp(min=1e-20))
        core   = d.unsqueeze(0) <= (w_i - self.win_edge)
        win_lb = torch.where(core, torch.zeros_like(win_lb), win_lb)
        in_band = (d.unsqueeze(0) >= 0) & (d.unsqueeze(0) <= w_i)
        win_lb = win_lb.masked_fill(~in_band, float("-inf"))                       # [N,L,L]

        # --- landmarks (trains alpha) ---
        # For each query i and selected landmark key p: bias = lmk_logw if the
        # landmark is causal (p ≤ i) and outside the band (p < i − w̃(i)), else -inf.
        # Built without in-place ops (autograd-safe): each landmark kk produces a
        # [N,H,L,L] term placed at its column j==p, combined by elementwise max.
        lmk_lb = torch.full((N, H, L, L), float("-inf"), device=device)
        k_lmk  = lmk_pos.shape[-1]
        i_idx  = idx.view(1, 1, L).float()                                         # [1,1,L] query i
        wl     = w_seq.view(N, 1, L)                                               # [N,1,L] w̃(i)
        for kk in range(k_lmk):
            pos  = lmk_pos[:, :, kk].transpose(0, 1)                               # [N,H]
            logw = lmk_logw[:, :, kk].transpose(0, 1)                              # [N,H]
            pos_c = pos.clamp(min=0)
            posf  = pos_c.float().view(N, H, 1)                                    # [N,H,1]
            ok = (pos >= 0).view(N, H, 1) & (posf <= i_idx) & (posf < (i_idx - wl))  # [N,H,L] over query i
            col  = pos_c.view(N, H, 1, 1).expand(N, H, L, 1)                       # key column j==p
            val  = torch.where(ok, logw.view(N, H, 1), torch.full((1, 1, 1), float("-inf"), device=device))
            term = torch.full((N, H, L, L), float("-inf"), device=device).scatter(
                3, col, val.unsqueeze(-1)
            )                                                                     # out-of-place
            lmk_lb = torch.maximum(lmk_lb, term)

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
        # The RoPE cache is stored in fp32, but q/k come out of their projections
        # in the autocast compute dtype (bf16 on Ampere+, fp16 on T4). Multiplying
        # by an fp32 cos/sin promotes q/k back to fp32, so the training kernel then
        # re-casts them down (a real [total_len,H,D] copy per tensor) AND, because
        # the post-RoPE dtype is fp32, its ``in_dtype`` guard falls through to fp16
        # even on a bf16 run. Cast cos/sin to q's dtype here so RoPE stays in the
        # compute dtype: q/k reach the kernel already in bf16 (kernel runs in the
        # right precision, and its .to(in_dtype) becomes a no-op). dtype-agnostic.
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        aux = self._aux_zero(x.device, x.dtype)

        # TRAINING: differentiable dense path, per sequence. This is the ONLY way the
        # soft window edge (§4.2) and the α-reweight of landmarks (§4.3) can train
        # win_gate / alpha_w, the Triton kernel's hand-written backward does not
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
            out  = self._packed_train(q, k, v, x, cu, lens, total_len, device, cu_seqlens)
            return self.out_proj(out.reshape(total_len, self.d_model)), aux

        # ---- inference (selectors, no gradient) ----
        w_sizes = self._window_sizes(x, floor=True)
        alpha   = self._alpha()

        if _TRITON_OK:
            lmk_indices, _, _ = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(),
                alpha.detach().float(), w_sizes,
                cu_seqlens, self.k_lmk, self.n_min, total_len, cu_list,
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

        s0    = int(cu_seqlens[0])
        e0    = int(cu_seqlens[1])
        q0    = q[s0:e0].transpose(0, 1)
        k0    = k[s0:e0].transpose(0, 1)
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
                [0, T0],   # host-side cu_list → no D2H sync / graph break
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

        return self.out_proj(out.view(total_len, self.d_model)), aux