import torch
import torch.nn.functional as F


def compute_window_sizes(
    x_prev: torch.Tensor,
    proj: torch.nn.Linear,
    n_min: int,
    n_max: int,
) -> torch.Tensor:
    return n_min + torch.sigmoid(proj(x_prev).squeeze(-1)) * (n_max - n_min)


def build_local_window_mask(
    seq_len: int,
    window_sizes: torch.Tensor,
    device: torch.device,
    causal: bool = True,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    diff      = positions.unsqueeze(1) - positions.unsqueeze(0)
    w         = window_sizes.clamp(min=1, max=seq_len).long().unsqueeze(1)
    mask      = (diff >= 0) & (diff < w)
    if not causal:
        mask = mask | mask.T
    return mask


def build_local_window_mask_packed(
    cu_seqlens: torch.Tensor,
    window_sizes: torch.Tensor,
    total_len: int,
    device: torch.device,
) -> torch.Tensor:
    row_idx = torch.arange(total_len, device=device)
    col_idx = torch.arange(total_len, device=device)

    seq_ids  = torch.zeros(total_len, dtype=torch.long, device=device)
    seq_off  = torch.zeros(total_len, dtype=torch.long, device=device)
    num_seqs = cu_seqlens.shape[0] - 1

    for b in range(num_seqs):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        seq_ids[s:e]  = b
        seq_off[s:e]  = torch.arange(e - s, device=device)

    w    = window_sizes.clamp(min=1).long()
    diff = seq_off.unsqueeze(1) - seq_off.unsqueeze(0)
    same = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)
    mask = same & (diff >= 0) & (diff < w.unsqueeze(1))
    return mask


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    half  = q.shape[-1] // 2
    def _rot(x):
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return q * cos + _rot(q) * sin, k * cos + _rot(k) * sin


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta     = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype) / scale
    emb       = torch.cat([torch.outer(positions, theta)] * 2, dim=-1)
    return emb.cos(), emb.sin()