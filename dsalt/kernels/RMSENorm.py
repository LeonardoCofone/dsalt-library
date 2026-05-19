import time
import torch
import torch.nn as nn


class RMSENorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        #print(f"--- [RMSENorm] init | d_model={d_model} eps={eps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0  = time.perf_counter()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        x_in_norm  = x.norm().item()
        out        = x * rms * self.weight
        x_out_norm = out.norm().item()

        if x_out_norm > 1e4 or x_out_norm < 1e-4:
            print(f"--- [RMSENorm] WARNING: high norm | in={x_in_norm:.4f} out={x_out_norm:.4f} | shape={tuple(x.shape)}")

        return out