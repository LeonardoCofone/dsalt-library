import torch


def compute_window_sizes(
    x_prev: torch.Tensor,
    n_min:  int,
    n_max:  int,
) -> torch.Tensor:
    """Per-token local window size.

    In this version the window is **frozen** to the characteristic value
    ``(n_min + n_max) // 2`` (§4.2): no learnable parameter, no dependence on a
    per-token adaptivity degree. The model's demonstrated adaptivity is that of
    the per-head ``alpha`` in landmark selection (§4.3).

    Returns a constant tensor ``[T]`` aligned with ``x_prev``.
    """
    T = x_prev.shape[0]
    w = (n_min + n_max) // 2
    return torch.full((T,), float(w), device=x_prev.device, dtype=torch.float32)


def build_local_window_mask(
    seq_len:      int,
    window_sizes: torch.Tensor,
    device:       torch.device,
) -> torch.Tensor:
    rows = torch.arange(seq_len, device=device)
    w    = window_sizes.clamp(min=1, max=seq_len).long()
    lo   = (rows - w + 1).clamp(min=0)

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

    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off = torch.arange(total_len, device=device) - starts[seq_ids]

    w      = window_sizes.clamp(min=1).long()
    lo_rel = (seq_off - w + 1).clamp(min=0)
    hi_rel = seq_off

    abs_lo = (starts[seq_ids] + lo_rel).clamp(min=0)
    abs_hi = (starts[seq_ids] + hi_rel).clamp(max=total_len - 1)

    # difference array trick, zero Python loops, fully vectorized
    mask_1d = torch.zeros(total_len + 1, dtype=torch.int32, device=device)
    mask_1d.scatter_add_(0, abs_lo,       torch.ones(total_len, dtype=torch.int32, device=device))
    mask_1d.scatter_add_(0, abs_hi + 1,  -torch.ones(total_len, dtype=torch.int32, device=device))

    # build full [T, T] bool mask via row-broadcast
    row_idx = torch.arange(total_len, device=device)
    col_idx = torch.arange(total_len, device=device)

    # causal + window: col in [abs_lo[row], abs_hi[row]] AND same sequence
    lo2d = abs_lo.unsqueeze(1)   # [T, 1]
    hi2d = abs_hi.unsqueeze(1)   # [T, 1]
    c    = col_idx.unsqueeze(0)  # [1, T]

    same_seq = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)  # [T, T]
    in_win   = (c >= lo2d) & (c <= hi2d)                     # [T, T]
    return (same_seq & in_win)


@torch.compile(dynamic=False)
def apply_rotary_emb(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    half  = q.shape[-1] // 2
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