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
) -> torch.Tensor:
    rows  = torch.arange(seq_len, device=device)
    w     = window_sizes.clamp(min=1, max=seq_len).long()
    lo    = (rows - w + 1).clamp(min=0)
    ones  = torch.ones(seq_len, 1, dtype=torch.int8, device=device)
    delta = torch.zeros(seq_len, seq_len + 1, dtype=torch.int8, device=device)
    delta.scatter_add_(1, lo.unsqueeze(1), ones)
    delta.scatter_add_(1, (rows + 1).unsqueeze(1), -ones)
    return delta[:, :seq_len].cumsum(dim=1).bool()


def build_local_window_mask_packed(
    cu_seqlens:   torch.Tensor,
    window_sizes: torch.Tensor,
    total_len:    int,
    device:       torch.device,
) -> torch.Tensor:
    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total_len, device=device) - starts[seq_ids]
    w        = window_sizes.clamp(min=1).long()
    abs_lo   = (starts[seq_ids] + (seq_off - w + 1).clamp(min=0)).unsqueeze(1)
    abs_hi   = (torch.arange(total_len, device=device) + 1).unsqueeze(1)
    j        = torch.arange(total_len, device=device).unsqueeze(0)
    same_seq = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0))
    return (j >= abs_lo) & (j < abs_hi) & same_seq


def apply_rotary_emb(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = q.shape[-1] // 2
    def _rot(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return q * cos + _rot(q) * sin, k * cos + _rot(k) * sin


def build_rope_cache(
    seq_len:  int,
    head_dim: int,
    device:   torch.device,
    dtype:    torch.dtype = torch.float32,
    base:     float       = 10000.0,
    scale:    float       = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta     = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype) / scale
    emb       = torch.cat([torch.outer(positions, theta)] * 2, dim=-1)
    return emb.cos(), emb.sin()