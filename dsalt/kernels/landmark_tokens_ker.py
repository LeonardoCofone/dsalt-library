import time
import torch
import torch.nn as nn


def compute_hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    t0          = time.perf_counter()
    x_norm      = x.norm(dim=-1)
    xwv         = (x @ W_V.T).norm(dim=-1)
    mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
    mu_v, std_v = xwv.mean(),    xwv.std().clamp(min=1e-6)
    scores      = alpha * (xwv - mu_v) / std_v + (1.0 - alpha) * (x_norm - mu_x) / std_x
    print(f"--- [landmark] compute_hybrid_scores | scores={tuple(scores.shape)} mean={scores.mean().item():.4f} | t={time.perf_counter()-t0:.4f}s")
    return scores


def select_landmarks(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    t0 = time.perf_counter()
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))

    n_valid  = int((scores != float("-inf")).sum())
    k_actual = min(k, n_valid)

    if k_actual == 0:
        print(f"--- [landmark] WARNING: k_actual=0, nessun landmark selezionato!")
        return torch.empty(0, dtype=torch.long, device=scores.device)

    _, indices = torch.topk(scores, k=k_actual, dim=-1, sorted=False)
    print(f"--- [landmark] select_landmarks | n_valid={n_valid} k_actual={k_actual} | t={time.perf_counter()-t0:.4f}s")
    return indices


class HybridEnergyLandmarkSelector(nn.Module):
    def __init__(self, n_layers: int, n_heads: int):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((n_layers, n_heads), torch.logit(torch.tensor(0.6)).item())
        )
        print(f"--- [landmark] HybridEnergyLandmarkSelector init | n_layers={n_layers} n_heads={n_heads}")

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
        in_win   = window_mask.any(dim=0)
        return select_landmarks(scores, k=k, exclude_mask=in_win)