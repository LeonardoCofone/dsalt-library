import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSENorm(nn.Module):
    """Root-Mean-Square LayerNorm (RMSNorm) over the last dimension.

    A learnable per-channel scale with no mean subtraction and no bias, computed
    via the fused ``F.rms_norm``. Used for the pre-norm blocks and the final norm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.normalized_shape = (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)