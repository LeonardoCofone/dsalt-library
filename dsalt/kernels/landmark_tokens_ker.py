"""Hybrid Energy Landmark scoring, single source of the DSALT formula (§4.3).

This module defines *the only* implementation of the hybrid-energy score used for
landmark selection. Both the attention path (``dsalt_attention``) and the Triton
kernel (``dsalt_triton_attn``) call :func:`hybrid_scores_per_head`, so that the
paper's formula lives in a single place and cannot diverge between execution
paths.

Formula (for token ``j``, layer ``l``, head ``h``)::

    s = alpha * z(||x_j W_V||_2) + (1 - alpha) * z(||x_j||_2)

where ``z(·)`` is standardisation (empirical mean/std over the candidate tokens),
``alpha = sigmoid(alpha_raw)`` is the per-head/per-layer balancing parameter, and
``||x_j W_V||_2`` is computed **per head** (output-sensitivity), while ``||x_j||_2``
is the representational persistence (shared across heads).
"""

import torch
import torch.nn as nn


def hybrid_scores_per_head(
    x:       torch.Tensor,
    W_V:     torch.Tensor,
    alpha:   torch.Tensor,
    n_heads: int,
    dh:      int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-head hybrid score and the raw normalised signals.

    Args:
        x:       Hidden states ``[T, d]``.
        W_V:     Value projection matrix ``[d, d]`` (``v_proj.weight``).
        alpha:   Per-head balancing, ``[n_heads]``, in ``(0, 1)``.
        n_heads: Number of heads.
        dh:      Per-head dimension (``d // n_heads``).

    Returns:
        Tuple ``(scores, z_x, z_v)`` where:
          * ``scores`` ``[T, n_heads]`` is the hybrid score ``s``;
          * ``z_x`` ``[T]`` is the standardised representational persistence;
          * ``z_v`` ``[T, n_heads]`` is the per-head standardised output-sensitivity.

    ``z_x`` and ``z_v`` are returned separately because the backward over ``alpha``
    (the landmark gate in the Triton kernel) needs them to reconstruct the
    differentiable score without recomputing the norms.
    """
    T = x.shape[0]

    x_norm = x.norm(dim=-1).float()
    z_x    = (x_norm - x_norm.mean()) / x_norm.std().clamp(min=1e-6)

    xwv   = (x @ W_V.T).view(T, n_heads, dh).norm(dim=-1).float()
    mu_v  = xwv.mean(0, keepdim=True)
    std_v = xwv.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv - mu_v) / std_v

    scores = alpha * z_v + (1.0 - alpha) * z_x.unsqueeze(1)
    return scores, z_x, z_v


def compute_hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Hybrid score, backward-compatible wrapper of the public API.

    Keeps the historical signature ``(x, W_V, alpha)``. ``alpha`` can be:

    * **scalar** → legacy *query-agnostic* behaviour: the value norm is computed
      over the whole vector (single head), returns ``[T]``;
    * **vector** ``[n_heads]`` → delegates to :func:`hybrid_scores_per_head`,
      returning the per-head scores ``[T, n_heads]``.

    For the DSALT runtime path use :func:`hybrid_scores_per_head` directly.
    """
    if alpha.ndim == 0:
        x_norm      = x.norm(dim=-1)
        xwv         = (x @ W_V.T).norm(dim=-1)
        mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
        mu_v, std_v = xwv.mean(),    xwv.std().clamp(min=1e-6)
        return alpha * (xwv - mu_v) / std_v + (1.0 - alpha) * (x_norm - mu_x) / std_x

    n_heads = alpha.shape[0]
    dh      = W_V.shape[0] // n_heads
    scores, _, _ = hybrid_scores_per_head(x, W_V, alpha, n_heads, dh)
    return scores


def select_landmarks(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k landmarks from the scores, optionally excluding the local window."""
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))

    n_valid  = int((scores != float("-inf")).sum())
    k_actual = min(k, n_valid)

    if k_actual == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    _, indices = torch.topk(scores, k=k_actual, dim=-1, sorted=False)
    return indices


def soft_landmark_weights(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
    temperature:  float = 1.0,
) -> torch.Tensor:
    """Soft weights (temperature-scaled softmax) over candidates, auxiliary utility."""
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))
    weights = torch.softmax(scores / temperature, dim=-1)
    return weights


class HybridEnergyLandmarkSelector(nn.Module):
    """Landmark selector with a learnable per-layer/per-head ``alpha``.

    An ``nn.Module`` wrapper around the shared formula. Keeps
    ``alpha`` ``[n_layers, n_heads]`` initialised to ``sigmoid^{-1}(0.6)``,
    consistent with :class:`~dsalt.modules.dsalt_attention.DSALTAttention`.
    """

    def __init__(self, n_layers: int, n_heads: int):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((n_layers, n_heads), torch.logit(torch.tensor(0.6)).item())
        )

    def forward(
        self,
        x:           torch.Tensor,
        W_V:         torch.Tensor,
        layer_idx:   int,
        head_idx:    int,
        k:           int,
        window_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha    = torch.sigmoid(self.alpha[layer_idx, head_idx])
        scores   = compute_hybrid_scores(x, W_V, alpha)
        in_win   = window_mask.any(dim=0)

        indices  = select_landmarks(scores, k=k, exclude_mask=in_win)
        weights  = soft_landmark_weights(scores, k=k, exclude_mask=in_win)

        return indices, weights
