from turtle import pos

import torch
from typing import Tuple

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _hybrid_energy_kernel(
        X_ptr,
        WV_ptr,
        XSq_ptr,
        XVSq_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        D_head: tl.constexpr,
        BLOCK_N: tl.constexpr,
        stride_xb,
        stride_xn,
        stride_wh,
        stride_wd,
        stride_ob,
        stride_oh,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        offs_dh = tl.arange(0, D_head)
        mask_n = offs_n < N

        x = tl.load(
            X_ptr + pid_b * stride_xb + offs_n[:, None] * stride_xn + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float32)

        wv = tl.load(
            WV_ptr + pid_h * stride_wh + offs_d[:, None] * stride_wd + offs_dh[None, :],
        ).to(tl.float32)

        x_sq = tl.sum(x * x, axis=1)
        xv = tl.dot(x, wv)
        xv_sq = tl.sum(xv * xv, axis=1)

        out_base = XSq_ptr + pid_b * stride_ob + pid_h * stride_oh
        tl.store(out_base + offs_n, x_sq, mask=mask_n)

        out_base_xv = XVSq_ptr + pid_b * stride_ob + pid_h * stride_oh
        tl.store(out_base_xv + offs_n, xv_sq, mask=mask_n)


def _cpu_compute_norms(
    X: torch.Tensor,
    WV: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, D = X.shape
    H = WV.shape[0]
    Xf = X.float()
    x_sq = (Xf * Xf).sum(dim=-1)
    x_sq_bhn = x_sq.unsqueeze(1).expand(B, H, N).contiguous()
    xv_sq_list = []
    for h in range(H):
        xv_h = Xf @ WV[h].float()
        xv_sq_list.append((xv_h * xv_h).sum(dim=-1))
    xv_sq_bhn = torch.stack(xv_sq_list, dim=1)
    return x_sq_bhn, xv_sq_bhn


def _znorm(t: torch.Tensor) -> torch.Tensor:
    mu = t.mean(dim=-1, keepdim=True)
    std = t.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (t - mu) / std


def compute_hybrid_energy_scores(
    X: torch.Tensor,
    WV: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    B, N, D = X.shape
    H, _, D_head = WV.shape

    assert D == triton.next_power_of_2(D) if _TRITON_AVAILABLE else True, \
        "D deve essere potenza di 2 per il kernel Triton"
    assert D_head == triton.next_power_of_2(D_head) if _TRITON_AVAILABLE else True, \
        "D_head deve essere potenza di 2 per il kernel Triton"

    if _TRITON_AVAILABLE and X.is_cuda:
        X_c = X.contiguous()
        WV_c = WV.contiguous()
        x_sq = torch.empty((B, H, N), device=X.device, dtype=torch.float32)
        xv_sq = torch.empty((B, H, N), device=X.device, dtype=torch.float32)
        BLOCK_N = min(128, triton.next_power_of_2(N))
        grid = (triton.cdiv(N, BLOCK_N), H, B)
        _hybrid_energy_kernel[grid](
            X_c, WV_c, x_sq, xv_sq,
            N=N, D=D, D_head=D_head, BLOCK_N=BLOCK_N,
            stride_xb=X_c.stride(0), stride_xn=X_c.stride(1),
            stride_wh=WV_c.stride(0), stride_wd=WV_c.stride(1),
            stride_ob=x_sq.stride(0), stride_oh=x_sq.stride(1),
        )
    else:
        x_sq, xv_sq = _cpu_compute_norms(X, WV)

    a = alpha.view(1, H, 1)
    return a * _znorm(xv_sq) + (1.0 - a) * _znorm(x_sq)


def select_landmarks(
    scores: torch.Tensor,
    k: int,
    window_sizes: torch.Tensor,
) -> torch.Tensor:
    B, H, N = scores.shape
    pos = torch.arange(N, device=scores.device).view(1, 1, N)
    max_w = window_sizes.max(dim=-1, keepdim=True).values
    last_pos = N - 1
    in_window = pos >= (last_pos - max_w + 1)
    masked_scores = scores.masked_fill(in_window, float("-inf"))

    k_safe = min(k, N)
    top_idx = torch.topk(masked_scores, k=k_safe, dim=-1, largest=True, sorted=False).indices
    top_idx = top_idx.sort(dim=-1).values.to(torch.int32)

    if k_safe < k:
        pad = torch.full(
            (B, H, k - k_safe), fill_value=0,
            device=scores.device, dtype=torch.int32
        )
        top_idx = torch.cat([top_idx, pad], dim=-1)

    return top_idx


def compute_landmark_idx(
    X: torch.Tensor,
    WV: torch.Tensor,
    window_sizes: torch.Tensor,
    k: int,
    alpha: torch.Tensor,
) -> torch.Tensor:
    scores = compute_hybrid_energy_scores(X, WV, alpha)
    return select_landmarks(scores, k, window_sizes)