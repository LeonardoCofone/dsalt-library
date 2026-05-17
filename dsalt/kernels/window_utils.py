import torch
import torch.nn as nn


def compute_window_sizes(
    x_prev: torch.Tensor,
    proj:   nn.Linear,
    n_min:  int,
    n_max:  int,
) -> torch.Tensor:
    return n_min + torch.sigmoid(proj(x_prev).squeeze(-1)) * (n_max - n_min)


def build_local_window_mask(
    seq_len:      int,
    window_sizes: torch.Tensor,
    device:       torch.device,
    causal:       bool = True,
) -> torch.Tensor:
    # Costruisce la mask per sequenza singola in O(n * w_max) invece di O(n²).
    # Per ogni token i, gli indici validi sono [i - w[i] + 1, i].
    w    = window_sizes.clamp(min=1, max=seq_len).long()
    rows = torch.arange(seq_len, device=device)

    # Massimo numero di posizioni coperte: serve solo per allocare.
    # Costruiamo la mask riga per riga usando scatter, senza espandere [n, n].
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # lo[i] = max(0, i - w[i] + 1), hi[i] = i  (causal)
    lo = (rows - w + 1).clamp(min=0)
    hi = rows

    # Invece di fare diff/broadcast n×n, usiamo un segmented fill:
    # per ogni i settiamo mask[i, lo[i]:hi[i]+1] = True.
    # Trick vettorizzato con cumsum: costruiamo un vettore di delta e scansioniamo.
    # delta[i, lo[i]] += 1, delta[i, hi[i]+1] -= 1  → cumsum lungo dim=1 → mask.
    delta = torch.zeros(seq_len, seq_len + 1, dtype=torch.int8, device=device)
    idx_i = rows.unsqueeze(1)
    delta.scatter_add_(1, lo.unsqueeze(1),  torch.ones(seq_len, 1, dtype=torch.int8, device=device))
    delta.scatter_add_(1, (hi + 1).unsqueeze(1), -torch.ones(seq_len, 1, dtype=torch.int8, device=device))
    mask = delta[:, :seq_len].cumsum(dim=1) > 0

    if not causal:
        mask = mask | mask.T

    return mask


def build_local_window_mask_packed(
    cu_seqlens:   torch.Tensor,
    window_sizes: torch.Tensor,
    total_len:    int,
    device:       torch.device,
) -> torch.Tensor:
    # Costruisce la mask packed processando ogni sequenza indipendentemente,
    # cosi' ogni sottochiamata e' O(L_b * w_max_b) invece di O(total^2).
    num_seqs = cu_seqlens.shape[0] - 1
    mask     = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    for b in range(num_seqs):
        s = int(cu_seqlens[b]); e = int(cu_seqlens[b + 1])
        L = e - s

        w_b  = window_sizes[s:e].clamp(min=1).long()
        rows = torch.arange(L, device=device)
        lo   = (rows - w_b + 1).clamp(min=0)
        hi   = rows

        delta = torch.zeros(L, L + 1, dtype=torch.int8, device=device)
        delta.scatter_add_(1, lo.unsqueeze(1), torch.ones(L, 1, dtype=torch.int8, device=device))
        delta.scatter_add_(1, (hi + 1).unsqueeze(1), -torch.ones(L, 1, dtype=torch.int8, device=device))
        sub_mask = delta[:, :L].cumsum(dim=1) > 0

        mask[s:e, s:e] = sub_mask

    return mask


def apply_rotary_emb(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = q.shape[-1] // 2
    def _rot(x):
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return q * cos + _rot(q) * sin, k * cos + _rot(k) * sin


def build_rope_cache(
    seq_len:  int,
    head_dim: int,
    device:   torch.device,
    dtype:    torch.dtype = torch.float32,
    base:     float = 10000.0,
    scale:    float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta     = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype) / scale
    emb       = torch.cat([torch.outer(positions, theta)] * 2, dim=-1)
    return emb.cos(), emb.sin()