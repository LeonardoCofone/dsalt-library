import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSENorm(nn.Module):
    """Root-Mean-Square LayerNorm (RMSNorm) over the last dimension.

    A learnable per-channel scale with no mean subtraction and no bias, computed
    via the fused ``F.rms_norm``. Used for the pre-norm blocks and the final norm.

    ``F.rms_norm`` is not autocast-aware: with a bf16/fp16 input ``x`` and the
    fp32 ``weight`` it promotes the output to fp32. We cast the result back to the
    input dtype so the residual stream and the following projections stay in the
    autocast dtype — otherwise an fp32 activation hits a bf16 weight in the next
    matmul and (under torch.compile, which emits dtype-strict ``addmm``) raises
    "self and mat2 must have the same dtype". GPU-portable: a no-op in fp32.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.normalized_shape = (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return out.to(x.dtype)