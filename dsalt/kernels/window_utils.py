"""
Questo modulo implementa il predittore della dimensione della finestra, fornendo la classe
WindowSizePredictor per calcolare la larghezza della finestra in base al modello e al numero
di teste di attenzione.
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WindowSizePredictor(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_min: int = 32, n_max: int = 256):
        super().__init__()
        assert n_min < n_max, f"n_min={n_min} must be < n_max={n_max}"
        assert n_min > 0,     f"n_min={n_min} must be positive"

        self.n_min   = n_min
        self.n_max   = n_max
        self.n_heads = n_heads
        self.proj    = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

        logger.debug(
            "WindowSizePredictor | d_model=%d n_heads=%d n_min=%d n_max=%d",
            d_model, n_heads, n_min, n_max,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        H       = self.n_heads

        logits = self.proj(x).squeeze(-1)      
        logits = logits.clamp(-10.0, 10.0)                        
        cont_w = self.n_min + torch.sigmoid(logits) * (self.n_max - self.n_min)

        w_int = cont_w.detach().floor().long().clamp(self.n_min, self.n_max)
        w_int = w_int.unsqueeze(1).expand(B, H, N).contiguous()            

        return w_int.to(torch.int32), cont_w

    def window_entropy_reg(self, cont_w: torch.Tensor) -> torch.Tensor:
        range_span = float(self.n_max - self.n_min)
        normalized = (cont_w - self.n_min) / range_span 
        return -normalized.var(dim=-1).mean()