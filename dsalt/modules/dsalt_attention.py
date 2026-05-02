"""
dsalt/modules/dsalt_attention.py
---------------------------------
nn.Module implementing the full DSALT multi-head attention layer.

Pipeline per forward pass:
  1. Project X → Q, K, V                              (standard linear)
  2. WindowSizePredictor: x^{l-1} → w(i) per token   (learned, differentiable)
  3. HybridEnergy: x^{l-1}, W_V → landmark_idx        (top-k, no gradient)
  4. DSALTAttentionFunction: sparse causal attention   (Triton / CPU fallback)
  5. Output projection                                  (standard linear)

Flash Attention 2 integration:
  When flash_attn is available AND the local window is large enough that the
  full sequence fits in the window (i.e., n <= n_max), we fall back to FA2
  for correctness testing. Otherwise our Triton kernel is used.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsalt.kernels.sparse_attn import dsalt_attention
from dsalt.kernels.hybrid_energy import compute_landmark_idx
from dsalt.kernels.window_utils import WindowSizePredictor

try:
    from flash_attn import flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


class DSALTAttention(nn.Module):
    """
    Dynamic Sparse Attention with Landmark Tokens.

    Parameters
    ----------
    d_model  : total model dimension
    n_heads  : number of attention heads (d_model must be divisible by n_heads)
    n_min    : minimum adaptive window size
    n_max    : maximum adaptive window size
    k_lmk   : number of landmark tokens per query
    alpha    : hybrid energy mixing coefficient (0 → pure norm, 1 → pure value norm)
    dropout  : attention dropout probability (applied in training, CPU fallback only)
    use_fa2  : try to use Flash Attention 2 for short sequences / dense fallback
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_min:    int   = 32,
        n_max:    int   = 256,
        k_lmk:   int   = 16,
        alpha:    float = 0.6,
        dropout:  float = 0.0,
        use_fa2:  bool  = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.k_lmk   = k_lmk
        self.alpha    = alpha
        self.dropout  = dropout
        self.use_fa2  = use_fa2 and _FLASH_AVAILABLE

        # Standard QKV projection (single fused matrix for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

        # Adaptive window predictor
        self.window_pred = WindowSizePredictor(
            d_model=d_model,
            n_heads=n_heads,
            n_min=n_min,
            n_max=n_max,
        )

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.d_head)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _get_wv_per_head(self) -> torch.Tensor:
        """
        Extract the per-head value projection matrices from qkv_proj.
        qkv_proj.weight shape: [3*d_model, d_model]
        V portion: rows [2*d_model : 3*d_model]
        Reshape to [n_heads, d_head, d_model] then transpose to [n_heads, d_model, d_head].

        For hybrid energy we need W_V as [H, D_model, D_head].
        We approximate with the full d_model projection collapsed per head.
        """
        W = self.qkv_proj.weight    # [3*d_model, d_model]
        d = self.d_model
        h = self.n_heads
        dh = self.d_head

        # V rows
        Wv = W[2 * d: 3 * d, :]    # [d_model, d_model]
        # Reshape to [H, d_head, d_model] and return [H, d_model, d_head]
        Wv_heads = Wv.view(h, dh, d)   # [H, d_head, d_model]
        # For energy scoring we want [H, d_model, d_head] (right-multiply x @ Wv)
        # x: [N, d_model], Wv_heads[h]: [d_model, d_head]
        return Wv_heads.permute(0, 2, 1).contiguous()   # [H, d_model, d_head]

    def forward(
        self,
        x:            torch.Tensor,              # [B, N, D]
        x_prev:       Optional[torch.Tensor] = None,  # [B, N, D] from prev layer
        return_window: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x       : current layer input [B, N, D]
        x_prev  : previous layer hidden states for window + landmark prediction.
                  If None, x is used (layer 0 behaviour).
        return_window : if True, also return the continuous window sizes.

        Returns
        -------
        out          : [B, N, D]
        cont_w       : [B, N] continuous window sizes (only if return_window=True)
        """
        B, N, D = x.shape
        H  = self.n_heads
        Dh = self.d_head

        # Use x_prev for prediction; fall back to x at layer 0
        x_pred = x_prev if x_prev is not None else x

        # ── 1. Predict adaptive window sizes ────────────────────────────────
        window_sizes, cont_w = self.window_pred(x_pred, training=self.training)
        # window_sizes: [B, H, N]  int32
        # cont_w      : [B, N]     float (for gradient / regularisation)

        # ── 2. Compute QKV ──────────────────────────────────────────────────
        qkv = self.qkv_proj(x)                           # [B, N, 3*D]
        qkv = qkv.view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)  # [3, B, H, N, Dh]
        Q, K, V = qkv[0], qkv[1], qkv[2]                # each [B, H, N, Dh]

        # ── 3. Select landmark tokens via Hybrid Energy ──────────────────────
        # We need [B, H, N, Dh] hidden states per head for energy scoring.
        # We project x_pred to per-head space using the V projection.
        # Shape: x_pred [B, N, D] → expand to [B, H, N, D] per head
        with torch.no_grad():
            x_pred_heads = x_pred.unsqueeze(1).expand(B, H, N, D)  # [B, H, N, D]
            Wv = self._get_wv_per_head()                             # [H, D, Dh]
            # For efficiency pass x_pred_heads as [B, H, N, D] and Wv as [H, D, Dh]
            # hybrid_energy expects [B, H, N, D] and [H, D, D] (square) — we adapt:
            landmark_idx = compute_landmark_idx(
                X=x_pred_heads,
                WV=Wv,
                window_sizes=window_sizes,
                k=self.k_lmk,
                alpha=self.alpha,
            )   # [B, H, N, k_lmk]  int32

        # ── 4. DSALT sparse attention ────────────────────────────────────────
        if self.use_fa2 and N <= window_sizes.max():
            # Sequence fits in window → use Flash Attention 2 as exact fallback
            # FA2 expects [B, N, H, Dh]
            out = flash_attn_func(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).transpose(1, 2)   # back to [B, H, N, Dh]
        else:
            out = dsalt_attention(Q, K, V, window_sizes, landmark_idx)
            # out: [B, H, N, Dh]

        # ── 5. Merge heads and project ──────────────────────────────────────
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)   # [B, N, D]
        out = self.out_proj(out)

        if return_window:
            return out, cont_w
        return out, None

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, k_lmk={self.k_lmk}, alpha={self.alpha}"
        )