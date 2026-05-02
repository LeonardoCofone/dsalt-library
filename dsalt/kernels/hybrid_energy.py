"""
dsalt/kernels/hybrid_energy.py
-------------------------------
Hybrid Energy scoring and landmark selection for DSALT.

The score for token j (candidate landmark) is:

    s_j = α * z(‖x_j W_V‖₂)  +  (1-α) * z(‖x_j‖₂)

where z(·) is standard-normalization across candidates at the current layer:

    z(x) = (x - mean(x)) / std(x)

Two Triton kernels:
  1. _compute_norms_kernel  — compute ‖x_j‖₂ and ‖x_j W_V‖₂ for all j in parallel
  2. Top-k selection        — via PyTorch topk (already O(n log k), GPU-native)

The landmark selection is **global** (shared across all query tokens in the
sequence), consistent with DSALT's design where each token attends to the
same top-k globally informative tokens.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# Triton kernel: compute norms in a single fused pass
# ═════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    @triton.jit
    def _hybrid_energy_kernel(
        X_ptr,          # [N, D]  input hidden states (one head, one batch)
        WV_ptr,         # [D, D]  value projection matrix
        XNorm_ptr,      # [N]     output: ‖x_j‖₂
        XVNorm_ptr,     # [N]     output: ‖x_j W_V‖₂
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,   # tokens per program
        BLOCK_D: tl.constexpr,   # must be >= D, power-of-2
    ):
        """
        Grid: (cdiv(N, BLOCK_N),)
        Each program handles BLOCK_N tokens, computing both norms.
        """
        pid   = tl.program_id(0)
        start = pid * BLOCK_N
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load X tile  [BLOCK_N, BLOCK_D]
        x = tl.load(
            X_ptr + offs_n[:, None] * D + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )   # [BLOCK_N, BLOCK_D]

        # ‖x_j‖₂²
        x_norm_sq = tl.sum(x * x, axis=1)   # [BLOCK_N]

        # x_j W_V :  [BLOCK_N, BLOCK_D] @ [BLOCK_D, BLOCK_D]
        # We load WV in column blocks to compute the product tile by tile
        xv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        # Full product in one dot if BLOCK_D == D (usually true for D <= 256)
        wv = tl.load(
            WV_ptr + offs_d[:, None] * D + offs_d[None, :],
            mask=mask_d[:, None] & mask_d[None, :],
            other=0.0,
        )   # [BLOCK_D, BLOCK_D]
        xv = tl.dot(x, wv)   # [BLOCK_N, BLOCK_D]

        xv_norm_sq = tl.sum(xv * xv, axis=1)   # [BLOCK_N]

        # Store sqrt of norms
        x_norm  = tl.sqrt(x_norm_sq)
        xv_norm = tl.sqrt(xv_norm_sq)

        tl.store(XNorm_ptr  + offs_n, x_norm,  mask=mask_n)
        tl.store(XVNorm_ptr + offs_n, xv_norm, mask=mask_n)


# ═════════════════════════════════════════════════════════════════════════════
# CPU fallback
# ═════════════════════════════════════════════════════════════════════════════

def _cpu_compute_norms(
    X:  torch.Tensor,    # [N, D_model]
    WV: torch.Tensor,    # [D_model, D_head]  (already oriented for right-multiply)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (x_norm [N], xv_norm [N])."""
    x_norm  = X.float().norm(dim=-1)                    # [N]
    xv_norm = (X.float() @ WV.float()).norm(dim=-1)     # [N, D_head] → norm → [N]
    return x_norm, xv_norm


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def compute_hybrid_energy_scores(
    X:     torch.Tensor,   # [B, H, N, D]  hidden states
    WV:    torch.Tensor,   # [H, D, D]     per-head value projections
    alpha: float = 0.6,
) -> torch.Tensor:
    """
    Computes the Hybrid Energy score for every token in the sequence.

        s_j = α * z(‖x_j W_V‖₂)  +  (1-α) * z(‖x_j‖₂)

    where z(·) is standard-normalisation over j.

    Returns scores of shape [B, H, N]  (higher = more likely landmark).
    """
    B, H, N, D = X.shape
    x_norms  = torch.empty(B, H, N, dtype=torch.float32, device=X.device)
    xv_norms = torch.empty(B, H, N, dtype=torch.float32, device=X.device)

    if X.is_cuda and _TRITON_AVAILABLE:
        BLOCK_N = 64
        BLOCK_D = triton.next_power_of_2(D)
        grid    = (triton.cdiv(N, BLOCK_N),)

        for b in range(B):
            for h in range(H):
                _hybrid_energy_kernel[grid](
                    X[b, h].contiguous(),
                    WV[h].contiguous(),
                    x_norms[b, h],
                    xv_norms[b, h],
                    N=N, D=D,
                    BLOCK_N=BLOCK_N,
                    BLOCK_D=BLOCK_D,
                )
    else:
        for b in range(B):
            for h in range(H):
                xn, xvn = _cpu_compute_norms(X[b, h], WV[h])
                x_norms[b, h]  = xn
                xv_norms[b, h] = xvn

    # Z-normalise each (b, h) independently over the N dimension
    def _znorm(t: torch.Tensor) -> torch.Tensor:
        mu  = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (t - mu) / std

    scores = alpha * _znorm(xv_norms) + (1.0 - alpha) * _znorm(x_norms)
    return scores   # [B, H, N]


def select_landmarks(
    scores:       torch.Tensor,    # [B, H, N]
    k:            int,             # number of landmarks
    window_sizes: torch.Tensor,    # [B, H, N]   int  — exclude in-window tokens
    exclude_last: int = 0,         # never select the last `exclude_last` tokens
                                   # (avoids selecting current token as own landmark)
) -> torch.Tensor:
    """
    Selects k landmark token indices per (batch, head) via top-k on Hybrid Energy.

    Tokens inside the maximum window are excluded from landmark candidacy
    (they are already covered by the local window pass).

    Returns landmark_idx of shape [B, H, N, k]  (int32).
    Each query token i gets the SAME global top-k landmarks (the standard
    DSALT design); the per-query mask is applied inside the attention kernel.
    """
    B, H, N = scores.shape
    device  = scores.device

    # Build candidate mask: exclude tokens too close to the end of the sequence
    # and tokens that are trivially in the window for most queries.
    max_w = window_sizes.max()   # conservative: use global max window

    # Mask out the last max_w positions (they're in everyone's window)
    cand_scores = scores.clone()
    if max_w > 0 and N > max_w:
        # Soft exclusion: set in-window region to -inf so they're never picked
        # as "global" landmarks (they're covered locally).
        cand_scores[..., N - max_w:] = float("-inf")
    if exclude_last > 0:
        cand_scores[..., N - exclude_last:] = float("-inf")

    # Top-k selection  [B, H, k]
    k_safe = min(k, N)
    _, top_idx = torch.topk(cand_scores, k=k_safe, dim=-1)  # [B, H, k]

    # Sort indices for more cache-friendly access in the attention kernel
    top_idx, _ = top_idx.sort(dim=-1)

    # Broadcast to [B, H, N, k]  — same landmarks for every query token
    landmark_idx = top_idx.unsqueeze(2).expand(B, H, N, k_safe)

    return landmark_idx.to(torch.int32)


def compute_landmark_idx(
    X:            torch.Tensor,    # [B, H, N, D]
    WV:           torch.Tensor,    # [H, D, D]
    window_sizes: torch.Tensor,    # [B, H, N]
    k:            int,
    alpha:        float = 0.6,
) -> torch.Tensor:
    """
    Convenience function: score + select in one call.
    Returns landmark_idx [B, H, N, k] int32.
    """
    scores = compute_hybrid_energy_scores(X, WV, alpha)
    return select_landmarks(scores, k, window_sizes)