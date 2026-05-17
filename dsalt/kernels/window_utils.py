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
    rows = torch.arange(seq_len, device=device)
    w    = window_sizes.clamp(min=1, max=seq_len).long()
    lo   = (rows - w + 1).clamp(min=0)
    hi   = rows

    delta = torch.zeros(seq_len, seq_len + 1, dtype=torch.int8, device=device)
    ones  = torch.ones(seq_len, 1, dtype=torch.int8, device=device)
    delta.scatter_add_(1, lo.unsqueeze(1), ones)
    delta.scatter_add_(1, (hi + 1).unsqueeze(1), -ones)
    mask = delta[:, :seq_len].cumsum(dim=1).bool()

    if not causal:
        mask = mask | mask.T
    return mask


def build_local_window_mask_packed(
    cu_seqlens:   torch.Tensor,
    window_sizes: torch.Tensor,
    total_len:    int,
    device:       torch.device,
) -> torch.Tensor:
    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.cat([torch.arange(int(l), device=device) for l in lens.tolist()])

    w   = window_sizes.clamp(min=1).long()
    lo  = (seq_off - w + 1).clamp(min=0)
    hi  = seq_off

    starts = cu_seqlens[:-1].to(device)

    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    max_len = int(lens.max())
    delta   = torch.zeros(num_seqs, max_len, max_len + 1, dtype=torch.int8, device=device)
    ones    = torch.ones(total_len, 1, dtype=torch.int8, device=device)

    delta[seq_ids, seq_off, lo].add_(1)
    delta[seq_ids, seq_off, hi + 1].add_(-1)

    sub = delta[:, :, :max_len].cumsum(dim=2).bool()

    for b in range(num_seqs):
        s = int(cu_seqlens[b]); e = int(cu_seqlens[b + 1])
        L = e - s
        mask[s:e, s:e] = sub[b, :L, :L]

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