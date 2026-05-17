import time
import torch
import torch.nn as nn


def compute_hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    t0 = time.perf_counter()
    print(f"--- [landmark] compute_hybrid_scores START | x={tuple(x.shape)} W_V={tuple(W_V.shape)} alpha={alpha.item():.4f} | device={x.device}")

    x_norm      = x.norm(dim=-1)
    xwv         = (x @ W_V.T).norm(dim=-1)
    mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
    mu_v, std_v = xwv.mean(),    xwv.std().clamp(min=1e-6)

    print(f"--- [landmark] norm stats | mu_x={mu_x.item():.4f} std_x={std_x.item():.4f} | mu_v={mu_v.item():.4f} std_v={std_v.item():.4f}")

    scores = alpha * (xwv - mu_v) / std_v + (1.0 - alpha) * (x_norm - mu_x) / std_x
    print(f"--- [landmark] compute_hybrid_scores DONE | scores shape={tuple(scores.shape)} mean={scores.mean().item():.4f} | t={time.perf_counter()-t0:.4f}s")
    return scores


def select_landmarks(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    t0 = time.perf_counter()
    print(f"--- [landmark] select_landmarks START | scores={tuple(scores.shape)} k={k} | has_exclude={exclude_mask is not None}")

    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))
        n_excluded = exclude_mask.sum().item()
        print(f"--- [landmark] excluded {n_excluded} tokens dalla selezione")

    n_valid    = int((scores != float("-inf")).sum())
    k_actual   = min(k, n_valid)
    print(f"--- [landmark] n_valid={n_valid} k_actual={k_actual}")

    if k_actual == 0:
        print(f"--- [landmark] WARNING: k_actual=0, nessun landmark selezionato!")
        return torch.empty(0, dtype=torch.long, device=scores.device)

    _, indices = torch.topk(scores, k=k_actual, dim=-1, sorted=False)
    print(f"--- [landmark] select_landmarks DONE | indices={tuple(indices.shape)} | t={time.perf_counter()-t0:.4f}s")
    return indices


class HybridEnergyLandmarkSelector(nn.Module):
    def __init__(self, n_layers: int, n_heads: int):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((n_layers, n_heads), torch.logit(torch.tensor(0.6)).item())
        )
        print(f"--- [landmark] HybridEnergyLandmarkSelector init | n_layers={n_layers} n_heads={n_heads} | alpha_shape={tuple(self.alpha.shape)}")

    def forward(
        self,
        x:           torch.Tensor,
        W_V:         torch.Tensor,
        layer_idx:   int,
        head_idx:    int,
        k:           int,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        print(f"--- [landmark] HybridEnergyLandmarkSelector.forward | layer={layer_idx} head={head_idx} k={k} | x={tuple(x.shape)}")

        alpha     = torch.sigmoid(self.alpha[layer_idx, head_idx])
        print(f"--- [landmark] alpha[{layer_idx},{head_idx}]={alpha.item():.4f}")

        scores    = compute_hybrid_scores(x, W_V, alpha)
        in_window = window_mask.any(dim=0)
        print(f"--- [landmark] tokens in window: {in_window.sum().item()} / {in_window.shape[0]}")

        result = select_landmarks(scores, k=k, exclude_mask=in_window)
        print(f"--- [landmark] HybridEnergyLandmarkSelector.forward DONE | t={time.perf_counter()-t0:.4f}s")
        return result