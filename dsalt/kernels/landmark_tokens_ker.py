"""Hybrid Energy Landmark scoring — fonte unica della formula DSALT (§4.3).

Questo modulo definisce *l'unica* implementazione del punteggio energetico ibrido
usato per la selezione dei landmark. Sia il path di attenzione (``dsalt_attention``)
sia il kernel Triton (``dsalt_triton_attn``) richiamano :func:`hybrid_scores_per_head`,
così che la formula del paper esista in un solo posto e non possa divergere tra i
percorsi di esecuzione.

Formula (per token ``j``, layer ``l``, head ``h``)::

    s = alpha * z(||x_j W_V||_2) + (1 - alpha) * z(||x_j||_2)

dove ``z(·)`` è la standardizzazione (media/dev. std empiriche sui token candidati),
``alpha = sigmoid(alpha_raw)`` è il parametro di bilanciamento per-head/per-layer, e
``||x_j W_V||_2`` è calcolato **per head** (output-sensitivity), mentre ``||x_j||_2``
è la persistenza rappresentazionale (condivisa tra le head).
"""

import torch
import torch.nn as nn


def hybrid_scores_per_head(
    x:       torch.Tensor,
    W_V:     torch.Tensor,
    alpha:   torch.Tensor,
    n_heads: int,
    dh:      int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcola il punteggio ibrido per-head e i segnali normalizzati grezzi.

    Args:
        x:       Stati nascosti ``[T, d]``.
        W_V:     Matrice di proiezione dei value ``[d, d]`` (``v_proj.weight``).
        alpha:   Bilanciamento per head, ``[n_heads]``, in ``(0, 1)``.
        n_heads: Numero di head.
        dh:      Dimensione per head (``d // n_heads``).

    Returns:
        Tupla ``(scores, z_x, z_v)`` dove:
          * ``scores`` ``[T, n_heads]`` è il punteggio ibrido ``s``;
          * ``z_x`` ``[T]`` è la persistenza rappresentazionale standardizzata;
          * ``z_v`` ``[T, n_heads]`` è la output-sensitivity standardizzata per head.

    ``z_x`` e ``z_v`` sono restituiti separatamente perché il backward su ``alpha``
    (gate dei landmark nel kernel Triton) li richiede per ricostruire lo score
    differenziabile senza ricalcolare le norme.
    """
    T = x.shape[0]

    x_norm = x.norm(dim=-1).float()
    z_x    = (x_norm - x_norm.mean()) / x_norm.std().clamp(min=1e-6)

    xwv   = (x @ W_V.T).view(T, n_heads, dh).norm(dim=-1).float()
    mu_v  = xwv.mean(0, keepdim=True)
    std_v = xwv.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv - mu_v) / std_v

    scores = alpha * z_v + (1.0 - alpha) * z_x.unsqueeze(1)
    return scores, z_x, z_v


def compute_hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Punteggio ibrido — wrapper retro-compatibile dell'API pubblica.

    Mantiene la firma storica ``(x, W_V, alpha)``. ``alpha`` può essere:

    * **scalare** → comportamento legacy *query-agnostic*: la norma value è
      calcolata sull'intero vettore (head singola), restituisce ``[T]``;
    * **vettore** ``[n_heads]`` → delega a :func:`hybrid_scores_per_head`,
      restituendo i punteggi per head ``[T, n_heads]``.

    Per il path runtime DSALT usare direttamente :func:`hybrid_scores_per_head`.
    """
    if alpha.ndim == 0:
        x_norm      = x.norm(dim=-1)
        xwv         = (x @ W_V.T).norm(dim=-1)
        mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
        mu_v, std_v = xwv.mean(),    xwv.std().clamp(min=1e-6)
        return alpha * (xwv - mu_v) / std_v + (1.0 - alpha) * (x_norm - mu_x) / std_x

    n_heads = alpha.shape[0]
    dh      = W_V.shape[0] // n_heads
    scores, _, _ = hybrid_scores_per_head(x, W_V, alpha, n_heads, dh)
    return scores


def select_landmarks(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k landmark dai punteggi, escludendo opzionalmente la finestra locale."""
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))

    n_valid  = int((scores != float("-inf")).sum())
    k_actual = min(k, n_valid)

    if k_actual == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    _, indices = torch.topk(scores, k=k_actual, dim=-1, sorted=False)
    return indices


def soft_landmark_weights(
    scores:       torch.Tensor,
    k:            int,
    exclude_mask: torch.Tensor | None = None,
    temperature:  float = 1.0,
) -> torch.Tensor:
    """Pesi soft (softmax temperato) sui candidati — utility ausiliaria."""
    if exclude_mask is not None:
        scores = scores.masked_fill(exclude_mask, float("-inf"))
    weights = torch.softmax(scores / temperature, dim=-1)
    return weights


class HybridEnergyLandmarkSelector(nn.Module):
    """Selettore landmark con ``alpha`` per-layer/per-head apprendibile.

    Wrapper ``nn.Module`` attorno alla formula condivisa. Mantiene
    ``alpha`` ``[n_layers, n_heads]`` inizializzato a ``sigmoid^{-1}(0.6)``,
    coerente con :class:`~dsalt.modules.dsalt_attention.DSALTAttention`.
    """

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha    = torch.sigmoid(self.alpha[layer_idx, head_idx])
        scores   = compute_hybrid_scores(x, W_V, alpha)
        in_win   = window_mask.any(dim=0)

        indices  = select_landmarks(scores, k=k, exclude_mask=in_win)
        weights  = soft_landmark_weights(scores, k=k, exclude_mask=in_win)

        return indices, weights
