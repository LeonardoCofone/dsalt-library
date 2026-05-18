import time
import torch
import torch.nn as nn


def compute_window_sizes(
    x_prev: torch.Tensor,
    proj:   nn.Linear,
    n_min:  int,
    n_max:  int,
) -> torch.Tensor:
    t0 = time.perf_counter()
    out = n_min + torch.sigmoid(proj(x_prev).squeeze(-1)) * (n_max - n_min)
    print(f"--- [window_utils] compute_window_sizes | shape_in={tuple(x_prev.shape)} | n_min={n_min} n_max={n_max} | w_mean={out.mean().item():.2f} w_std={out.std().item():.2f} | t={time.perf_counter()-t0:.4f}s")
    return out


def build_local_window_mask(
    seq_len:      int,
    window_sizes: torch.Tensor,
    device:       torch.device,
) -> torch.Tensor:
    t0 = time.perf_counter()
    print(f"--- [window_utils] build_local_window_mask START | seq_len={seq_len} | device={device}")

    rows = torch.arange(seq_len, device=device)
    w    = window_sizes.clamp(min=1, max=seq_len).long()
    lo   = (rows - w + 1).clamp(min=0)

    ones  = torch.ones(seq_len, 1, dtype=torch.int8, device=device)
    delta = torch.zeros(seq_len, seq_len + 1, dtype=torch.int8, device=device)
    delta.scatter_add_(1, lo.unsqueeze(1), ones)
    delta.scatter_add_(1, (rows + 1).unsqueeze(1), -ones)
    mask = delta[:, :seq_len].cumsum(dim=1).bool()

    mem_mb = mask.numel() * mask.element_size() / 1e6
    print(f"--- [window_utils] build_local_window_mask DONE | mask={tuple(mask.shape)} | mem={mem_mb:.2f}MB | nonzero_frac={mask.float().mean().item():.4f} | t={time.perf_counter()-t0:.4f}s")
    return mask


def build_local_window_mask_packed(
    cu_seqlens:   torch.Tensor,
    window_sizes: torch.Tensor,
    total_len:    int,
    device:       torch.device,
) -> torch.Tensor:
    t0 = time.perf_counter()

    num_seqs = cu_seqlens.shape[0] - 1

    lens   = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts = cu_seqlens[:-1].to(device)

    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off = torch.arange(total_len, device=device) - starts[seq_ids]

    w = window_sizes.clamp(min=1).long()

    lo = (seq_off - w + 1).clamp(min=0)
    hi = seq_off

    abs_lo = starts[seq_ids] + lo
    abs_hi = starts[seq_ids] + hi

    idx = torch.arange(total_len, device=device)

    mask = torch.zeros((total_len, total_len), device=device, dtype=torch.bool)

    for i in range(total_len):
        s = seq_ids[i].item()
        start = starts[s].item()
        end = start + lens[s].item()

        li = abs_lo[i].item()
        hi_i = abs_hi[i].item()

        li = max(li, start)
        hi_i = min(hi_i, end - 1)

        if li <= hi_i:
            mask[i, li:hi_i + 1] = True

    print(f"--- [window_utils] packed_window DONE | t={time.perf_counter()-t0:.4f}s")

    return mask


def apply_rotary_emb(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t0 = time.perf_counter()
    half = q.shape[-1] // 2
    def _rot(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    q_out = q * cos + _rot(q) * sin
    k_out = k * cos + _rot(k) * sin
    print(f"--- [window_utils] apply_rotary_emb | q={tuple(q.shape)} | t={time.perf_counter()-t0:.4f}s")
    return q_out, k_out


def build_rope_cache(
    seq_len:  int,
    head_dim: int,
    device:   torch.device,
    dtype:    torch.dtype = torch.float32,
    base:     float       = 10000.0,
    scale:    float       = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    t0 = time.perf_counter()
    theta     = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype) / scale
    emb       = torch.cat([torch.outer(positions, theta)] * 2, dim=-1)
    cos, sin  = emb.cos(), emb.sin()
    print(f"--- [window_utils] build_rope_cache | seq_len={seq_len} head_dim={head_dim} scale={scale} | t={time.perf_counter()-t0:.4f}s")
    return cos, sin