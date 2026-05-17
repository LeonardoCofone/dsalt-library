import torch
import torch.nn as nn


def compute_hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    x_norm       = x.norm(dim=-1)
    xwv          = (x @ W_V.T).norm(dim=-1)
    mu_x, std_x  = x_norm.mean(), x_norm.std().clamp(min=1e-6)
    mu_v, std_v  = xwv.mean(),    xwv.std().clamp(min=1e-6)
    return alpha * (xwv - mu_v) / std_v + (1.0 - alpha) * (x_norm - mu_x) / std_x


def select_landmarks(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))
    k_actual   = min(k, int((scores != float("-inf")).sum()))
    _, indices = torch.topk(scores, k=k_actual, dim=-1, sorted=False)
    return indices


class HybridEnergyLandmarkSelector(nn.Module):
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
    ) -> torch.Tensor:
        alpha    = torch.sigmoid(self.alpha[layer_idx, head_idx])
        scores   = compute_hybrid_scores(x, W_V, alpha)
        in_window = window_mask.any(dim=0)
        return select_landmarks(scores, k=k, exclude_mask=in_window)