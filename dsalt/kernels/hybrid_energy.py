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
        N,
        D,
        D_head: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        stride_xb,
        stride_xn,
        stride_xd,
        stride_wh,
        stride_wd,
        stride_wdh,
        stride_ob,
        stride_oh,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_n  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_dh = tl.arange(0, D_head)
        mask_n  = offs_n < N

        acc_x_sq = tl.zeros([BLOCK_N],         dtype=tl.float32)
        acc_xv   = tl.zeros([BLOCK_N, D_head], dtype=tl.float32)

        x_base  = X_ptr  + pid_b * stride_xb
        wv_base = WV_ptr + pid_h * stride_wh

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            x_ptrs  = x_base  + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd
            wv_ptrs = wv_base + offs_d[:, None] * stride_wd  + offs_dh[None, :] * stride_wdh

            x_tile  = tl.load(x_ptrs,  mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            wv_tile = tl.load(wv_ptrs, mask=mask_d[:, None],                   other=0.0).to(tl.float32)

            acc_xv   += tl.dot(x_tile, wv_tile)
            acc_x_sq += tl.sum(x_tile * x_tile, axis=1)

        xv_sq = tl.sum(acc_xv * acc_xv, axis=1)

        xsq_ptrs  = XSq_ptr  + pid_b * stride_ob + pid_h * stride_oh + offs_n
        xvsq_ptrs = XVSq_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_n

        tl.store(xsq_ptrs,  acc_x_sq, mask=mask_n)
        tl.store(xvsq_ptrs, xv_sq,    mask=mask_n)


def _cpu_compute_norms(
    X: torch.Tensor,
    WV: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, D = X.shape
    H = WV.shape[0]
    Xf = X.float()
    x_sq = (Xf * Xf).sum(dim=-1)
    x_sq_bhn = x_sq.unsqueeze(1).expand(B, H, N).contiguous()
    xv = torch.einsum('bnd,hdo->bhno', Xf, WV.float())
    xv_sq_bhn = (xv * xv).sum(dim=-1)
    return x_sq_bhn, xv_sq_bhn


def _znorm(t: torch.Tensor) -> torch.Tensor:
    mu  = t.mean(dim=-1, keepdim=True)
    std = t.std(dim=-1,  keepdim=True).clamp(min=1e-6)
    return (t - mu) / std


def compute_hybrid_energy_scores(
    X: torch.Tensor,
    WV: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    B, N, D      = X.shape
    H, _, D_head = WV.shape

    if _TRITON_AVAILABLE and X.is_cuda:
        X_c  = X.detach().contiguous()
        WV_c = WV.detach().contiguous()

        x_sq  = torch.empty((B, H, N), device=X.device, dtype=torch.float32)
        xv_sq = torch.empty((B, H, N), device=X.device, dtype=torch.float32)

        BLOCK_N  = 64
        BLOCK_D  = triton.next_power_of_2(min(D, 64))
        D_head_p = triton.next_power_of_2(D_head)

        grid = (triton.cdiv(N, BLOCK_N), H, B)

        _hybrid_energy_kernel[grid](
            X_c, WV_c, x_sq, xv_sq,
            N=N,
            D=D,
            D_head=D_head_p,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            stride_xb=X_c.stride(0),
            stride_xn=X_c.stride(1),
            stride_xd=X_c.stride(2),
            stride_wh=WV_c.stride(0),
            stride_wd=WV_c.stride(1),
            stride_wdh=WV_c.stride(2),
            stride_ob=x_sq.stride(0),
            stride_oh=x_sq.stride(1),
        )
    else:
        x_sq, xv_sq = _cpu_compute_norms(X.detach(), WV.detach())

    a = alpha.detach().float().view(1, H, 1)
    return a * _znorm(xv_sq) + (1.0 - a) * _znorm(x_sq)


def select_landmarks(
    scores: torch.Tensor,
    k: int,
    window_sizes: torch.Tensor,
) -> torch.Tensor:
    B, H, N = scores.shape

    pos   = torch.arange(N, device=scores.device).view(1, 1, N)
    max_w = window_sizes.amax(dim=-1).view(B, 1, 1)
    in_window     = pos >= (N - max_w)
    masked_scores = scores.masked_fill(in_window, float("-inf"))

    k_safe  = min(k, N)
    top_idx = torch.topk(masked_scores, k=k_safe, dim=-1, largest=True, sorted=False).indices
    top_idx = top_idx.sort(dim=-1).values.to(torch.int32)

    if k_safe < k:
        pad     = torch.zeros((B, H, k - k_safe), device=scores.device, dtype=torch.int32)
        top_idx = torch.cat([top_idx, pad], dim=-1)

    return top_idx


def compute_landmark_idx(
    X: torch.Tensor,
    WV: torch.Tensor,
    window_sizes: torch.Tensor,
    k: int,
    alpha: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        scores = compute_hybrid_energy_scores(X, WV, alpha)
        return select_landmarks(scores, k, window_sizes)