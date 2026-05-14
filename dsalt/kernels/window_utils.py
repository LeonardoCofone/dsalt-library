import torch
import torch.nn.functional as F


def compute_window_sizes(
    x_prev: torch.Tensor,
    proj: torch.nn.Linear,
    n_min: int,
    n_max: int,
) -> torch.Tensor:
    logits = proj(x_prev).squeeze(-1)
    w_cont = n_min + torch.sigmoid(logits) * (n_max - n_min)
    return w_cont


def build_local_window_mask(
    seq_len: int,
    window_sizes: torch.Tensor,
    device: torch.device,
    causal: bool = True,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    i_idx = positions.unsqueeze(1)
    j_idx = positions.unsqueeze(0)

    w = window_sizes.long().clamp(min=1, max=seq_len)
    w = w.unsqueeze(1)

    local_mask = (j_idx >= (i_idx - w + 1)) & (j_idx <= i_idx)

    if not causal:
        local_mask = local_mask | local_mask.T

    return local_mask


def build_local_window_mask_packed(
    cu_seqlens: torch.Tensor,
    window_sizes: torch.Tensor,
    total_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    for b in range(len(cu_seqlens) - 1):
        start = cu_seqlens[b].item()
        end = cu_seqlens[b + 1].item()
        seq_len = end - start

        local_w = window_sizes[start:end].long().clamp(min=1, max=seq_len)

        positions = torch.arange(seq_len, device=device)
        i_idx = positions.unsqueeze(1)
        j_idx = positions.unsqueeze(0)
        w = local_w.unsqueeze(1)

        local = (j_idx >= (i_idx - w + 1)) & (j_idx <= i_idx)
        mask[start:end, start:end] = local

    return mask


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype) / scale
    freqs = torch.outer(positions, theta)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()