"""
dsalt/kernels/window_utils.py
------------------------------
Adaptive window size computation for DSALT.

From the paper (Eq. 29):
    w(i) = N_min + floor( σ(f(x_i^{l-1})) * (N_max - N_min) )

where f: R^d → R is a learned linear projection shared across layers.
During training we use the continuous relaxation (no floor) for
differentiability; at inference we apply the floor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowSizePredictor(nn.Module):
    """
    Predicts a per-token adaptive window size from the previous-layer hidden state.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads. The projection is shared across heads
        but the window is broadcast to all heads (DSALT design).
    n_min : int
        Minimum window size.
    n_max : int
        Maximum window size.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_min:   int = 32,
        n_max:   int = 256,
    ):
        super().__init__()
        self.n_min   = n_min
        self.n_max   = n_max
        self.n_heads = n_heads

        # Shared linear projection  f: R^d → R
        # No bias improves stability (output is sigmoid-squashed anyway)
        self.proj = nn.Linear(d_model, 1, bias=False)

        # Initialise to predict mid-range window by default
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        x: torch.Tensor,   # [B, N, D]  previous-layer hidden states
        training: bool = True,
    ) -> torch.Tensor:
        """
        Returns window_sizes of shape [B, H, N] as int32.

        During training, uses continuous relaxation (float) internally
        for gradient flow, then converts to int32 for the attention mask.
        During inference, uses floor directly.
        """
        B, N, D = x.shape
        H = self.n_heads

        # f(x): [B, N, 1] → [B, N]
        logits = self.proj(x).squeeze(-1)          # [B, N]  float
        cont_w = self.n_min + torch.sigmoid(logits) * (self.n_max - self.n_min)
        # cont_w: [B, N]  continuous window sizes in [n_min, n_max]

        if training:
            # Store continuous version for gradient flow through the predictor
            # (the kernel receives int32, but we keep cont_w for the loss / reg)
            w_int = cont_w.detach().floor().long().clamp(self.n_min, self.n_max)
        else:
            w_int = cont_w.floor().long().clamp(self.n_min, self.n_max)

        # Broadcast to all heads: [B, N] → [B, H, N]
        w_int = w_int.unsqueeze(1).expand(B, H, N).contiguous()

        return w_int.to(torch.int32), cont_w   # (discrete for kernel, continuous for loss)

    def window_entropy_reg(self, cont_w: torch.Tensor) -> torch.Tensor:
        """
        Optional entropy regularisation to prevent window collapse.
        Penalises solutions where all tokens use the same window size.

        Loss = -H( p(w) )  where p is the empirical distribution over [n_min, n_max].
        We approximate this with the variance of the predicted windows.

        In practice a simple L2 towards a target window can work just as well;
        this is provided as an optional add-on.
        """
        # Variance across token dimension; maximise variance → diverse windows
        var = cont_w.var(dim=-1).mean()  # scalar
        return -var   # minimise → maximise variance