import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowSizePredictor(nn.Module):
    def __init__(self, d_model, n_heads, n_min=32, n_max=256):
        super().__init__()
        self.n_min   = n_min
        self.n_max   = n_max
        self.n_heads = n_heads
        self.proj    = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, training=True):
        B, N, D = x.shape
        H = self.n_heads
        logits = self.proj(x).squeeze(-1)
        cont_w = self.n_min + torch.sigmoid(logits) * (self.n_max - self.n_min)
        if training:
            w_int = cont_w.detach().floor().long().clamp(self.n_min, self.n_max)
        else:
            w_int = cont_w.floor().long().clamp(self.n_min, self.n_max)
        w_int = w_int.unsqueeze(1).expand(B, H, N).contiguous()
        return w_int.to(torch.int32), cont_w

    def window_entropy_reg(self, cont_w):
        return -cont_w.var(dim=-1).mean()