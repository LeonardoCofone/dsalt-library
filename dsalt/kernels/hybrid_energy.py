import torch
import torch.nn.functional as F
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
        X_ptr, WV_ptr, XNorm_ptr, XVNorm_ptr,
        N: tl.constexpr, D: tl.constexpr, D_head: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_Dh: tl.constexpr,
        stride_xb: tl.constexpr, stride_xh: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr,
    ):
        pid   = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        start  = pid * BLOCK_N
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_dh = tl.arange(0, BLOCK_Dh)

        acc_x_norm  = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_xv_norm = tl.zeros([BLOCK_N, BLOCK_Dh], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x_chunk = tl.load(
                X_ptr + pid_b * stride_xb + pid_h * stride_xh + offs_n[:, None] * D + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            acc_x_norm += tl.sum(x_chunk * x_chunk, axis=1)
            wv_chunk = tl.load(
                WV_ptr + pid_h * stride_wh + offs_d[:, None] * D_head + offs_dh[None, :],
                mask=mask_d[:, None] & (offs_dh[None, :] < D_head), other=0.0,
            )
            acc_xv_norm += tl.dot(x_chunk, wv_chunk)

        xv_norm_sq = tl.sum(acc_xv_norm * acc_xv_norm, axis=1)
        tl.store(XNorm_ptr  + pid_b * stride_ob + pid_h * stride_oh + offs_n, tl.sqrt(acc_x_norm),  mask=mask_n)
        tl.store(XVNorm_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_n, tl.sqrt(xv_norm_sq), mask=mask_n)


def _cpu_compute_norms(X, WV):
    x_norm  = X.float().norm(dim=-1)
    xv_norm = (X.float() @ WV.float()).norm(dim=-1)
    return x_norm, xv_norm


def compute_hybrid_energy_scores(X, WV, alpha=0.6):
    B, H, N, D = X.shape
    _, _, D_head = WV.shape
    x_norms  = torch.empty(B, H, N, dtype=torch.float32, device=X.device)
    xv_norms = torch.empty(B, H, N, dtype=torch.float32, device=X.device)

    if X.is_cuda and _TRITON_AVAILABLE:
        BLOCK_N  = 64
        BLOCK_D  = 64
        BLOCK_Dh = triton.next_power_of_2(D_head)
        grid = (triton.cdiv(N, BLOCK_N), H, B)
        _hybrid_energy_kernel[grid](
            X.contiguous(), WV.contiguous(), x_norms, xv_norms,
            N=N, D=D, D_head=D_head,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_Dh=BLOCK_Dh,
            stride_xb=X.stride(0), stride_xh=X.stride(1),
            stride_wh=WV.stride(0),
            stride_ob=x_norms.stride(0), stride_oh=x_norms.stride(1),
        )
    else:
        for b in range(B):
            for h in range(H):
                xn, xvn = _cpu_compute_norms(X[b, h], WV[h])
                x_norms[b, h]  = xn
                xv_norms[b, h] = xvn

    def _znorm(t):
        mu  = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (t - mu) / std

    a = alpha if isinstance(alpha, float) else alpha.view(1, -1, 1)
    return a * _znorm(xv_norms) + (1.0 - a) * _znorm(x_norms)


def select_landmarks(scores, k, window_sizes, exclude_last=0):
    B, H, N = scores.shape
    cand_scores = scores.clone()
    max_w = int(window_sizes.max().item())
    if max_w > 0 and N > max_w:
        cand_scores[..., N - max_w:] = float("-inf")
    if exclude_last > 0:
        cand_scores[..., N - exclude_last:] = float("-inf")
    k_safe = min(k, N)
    _, top_idx = torch.topk(cand_scores, k=k_safe, dim=-1)
    top_idx, _ = top_idx.sort(dim=-1)
    top_idx = top_idx.to(torch.int32)
    if k_safe < k:
        pad = torch.zeros(B, H, k - k_safe, dtype=torch.int32, device=scores.device)
        top_idx = torch.cat([top_idx, pad], dim=-1)
    return top_idx


def compute_landmark_idx(X, WV, window_sizes, k, alpha=0.6):
    scores = compute_hybrid_energy_scores(X, WV, alpha)
    landmark_idx = select_landmarks(scores, k, window_sizes)
    B, H, K = landmark_idx.shape
    N = window_sizes.shape[-1]
    return landmark_idx.unsqueeze(2).expand(B, H, N, K).contiguous()