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
        X_ptr,        # [B, N, D]
        WV_ptr,       # [H, D, D_head]
        XNorm_ptr,    # [B, H, N]  output ‖x‖
        XVNorm_ptr,   # [B, H, N]  output ‖xWv‖
        N: tl.constexpr,
        D: tl.constexpr,
        D_head: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_Dh: tl.constexpr,
        # strides X [B, N, D]
        stride_xb,
        stride_xn,
        # stride WV [H, D, D_head]
        stride_wh,
        stride_wd,
        # strides output [B, H, N]
        stride_ob,
        stride_oh,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        start  = pid_n * BLOCK_N
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        X_base  = X_ptr  + pid_b * stride_xb
        WV_base = WV_ptr + pid_h * stride_wh
        O_base_x  = XNorm_ptr  + pid_b * stride_ob + pid_h * stride_oh
        O_base_xv = XVNorm_ptr + pid_b * stride_ob + pid_h * stride_oh

        acc_x_sq  = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_xv    = tl.zeros([BLOCK_N, BLOCK_Dh], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            x_chunk = tl.load(
                X_base + offs_n[:, None] * stride_xn + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            acc_x_sq += tl.sum(x_chunk * x_chunk, axis=1)

            offs_dh = tl.arange(0, BLOCK_Dh)
            wv_chunk = tl.load(
                WV_base + offs_d[:, None] * stride_wd + offs_dh[None, :],
                mask=mask_d[:, None] & (offs_dh[None, :] < D_head),
                other=0.0,
            ).to(tl.float32)
            acc_xv = acc_xv + tl.dot(x_chunk, wv_chunk)
        xv_norm_sq = tl.sum(acc_xv * acc_xv, axis=1)

        tl.store(
            O_base_x  + offs_n, tl.sqrt(acc_x_sq),  mask=mask_n,
        )
        tl.store(
            O_base_xv + offs_n, tl.sqrt(xv_norm_sq), mask=mask_n,
        )


def _cpu_compute_norms(
    X: torch.Tensor,   # [B, N, D]
    WV: torch.Tensor,  # [H, D, D_head]
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, D = X.shape
    H = WV.shape[0]
    X_f = X.float()
    x_norm_bh = X_f.norm(dim=-1)
    x_norm = x_norm_bh.unsqueeze(1).expand(B, H, N)
    xv_list = []
    for h in range(H):
        xv_h = (X_f @ WV[h].float()).norm(dim=-1)
        xv_list.append(xv_h)
    xv_norm = torch.stack(xv_list, dim=1)
    return x_norm.contiguous(), xv_norm.contiguous()


def compute_hybrid_energy_scores(X, WV, alpha):
    B, N, D = X.shape
    H, _, D_head = WV.shape
    if _TRITON_AVAILABLE and X.is_cuda:
        x_norms = torch.empty((B, H, N), device=X.device, dtype=torch.float32)
        xv_norms = torch.empty((B, H, N), device=X.device, dtype=torch.float32)
        BLOCK_N, BLOCK_D = 128, 64
        BLOCK_Dh = triton.next_power_of_2(D_head)
        grid = (triton.cdiv(N, BLOCK_N), H, B)
        _hybrid_energy_kernel[grid](
            X, WV, x_norms, xv_norms,
            N=N, D=D, D_head=D_head, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_Dh=BLOCK_Dh,
            stride_xb=X.stride(0), stride_xn=X.stride(1),
            stride_wh=WV.stride(0), stride_wd=WV.stride(1),
            stride_ob=x_norms.stride(0), stride_oh=x_norms.stride(1)
        )
    else:
        x_norms, xv_norms = _cpu_compute_norms(X, WV)
        
    def _znorm(t: torch.Tensor) -> torch.Tensor:
        mu = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True).clamp(min=1e-5)
        return (t - mu) / std

    if isinstance(alpha, (float, int)):
        a = alpha
    else:
        a = alpha.view(1, -1, 1)

    return a * _znorm(xv_norms) + (1.0 - a) * _znorm(x_norms)


def select_landmarks(
    scores: torch.Tensor,        # [B, H, N]
    k: int,
    window_sizes: torch.Tensor,  # [B, H, N]  int32 — esclude la finestra locale
    exclude_last: int = 0,
) -> torch.Tensor:
    B, H, N = scores.shape
    cand_scores = scores.clone()
    max_w = int(window_sizes[..., -1].max().item())
    if max_w > 0 and N > max_w:
        cand_scores[..., N - max_w:] = float("-inf")
    if exclude_last > 0:
        cand_scores[..., N - exclude_last:] = float("-inf")
    k_safe = min(k, N)
    _, top_idx = torch.topk(cand_scores, k=k_safe, dim=-1, sorted=True)
    top_idx, _ = top_idx.sort(dim=-1)
    top_idx = top_idx.to(torch.int32)
    if k_safe < k:
        pad = torch.zeros(B, H, k - k_safe, dtype=torch.int32, device=scores.device)
        top_idx = torch.cat([top_idx, pad], dim=-1)
    return top_idx


def compute_landmark_idx(
    X: torch.Tensor,             # [B, N, D]       — hidden states
    WV: torch.Tensor,            # [H, D, D_head]  — value weights
    window_sizes: torch.Tensor,  # [B, H, N]       — finestre adattive
    k: int,
    alpha: torch.Tensor,         # [H] sigmoid già applicato
) -> torch.Tensor:
    scores = compute_hybrid_energy_scores(X, WV, alpha)
    result = select_landmarks(scores, k, window_sizes)
    return result