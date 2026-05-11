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
    def _energy_topk_kernel(
        X_ptr, WV_ptr, alpha_ptr,
        landmark_idx_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        D_head: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_Dh: tl.constexpr,
        stride_xb, stride_xn,
        stride_wh, stride_wd,
        stride_out_b, stride_out_h, stride_out_k,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        start_n = pid_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        X_base = X_ptr + pid_b * stride_xb
        WV_base = WV_ptr + pid_h * stride_wh
        OUT_base = landmark_idx_ptr + pid_b * stride_out_b + pid_h * stride_out_h

        alpha_w = tl.load(alpha_ptr + pid_h)

        energy = tl.zeros([BLOCK_N], dtype=tl.float32)
        x_norm_sq = tl.zeros([BLOCK_N], dtype=tl.float32)
        xv_norm_sq = tl.zeros([BLOCK_N], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            x_chunk = tl.load(
                X_base + offs_n[:, None] * stride_xn + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            x_norm_sq += tl.sum(x_chunk * x_chunk, axis=1)

            offs_dh = tl.arange(0, BLOCK_Dh)
            wv_chunk = tl.load(
                WV_base + offs_d[:, None] * stride_wd + offs_dh[None, :],
                mask=mask_d[:, None] & (offs_dh[None, :] < D_head),
                other=0.0,
            ).to(tl.float32)

            acc_xv = tl.dot(x_chunk, wv_chunk).to(tl.float32)
            xv_norm_sq += tl.sum(acc_xv * acc_xv, axis=1)

        x_norm = tl.sqrt(x_norm_sq)
        xv_norm = tl.sqrt(xv_norm_sq)
        energy = alpha_w * xv_norm + (1.0 - alpha_w) * x_norm

        neg_energy = -energy

        top_indices = tl.zeros([K], dtype=tl.int32)
        top_energies = tl.full([K], float("inf"), dtype=tl.float32)

        for i in range(BLOCK_N):
            token_idx = start_n + i
            token_energy = neg_energy[i]
            mask_valid = token_idx < N

            worst_slot = K - 1
            if token_energy < top_energies[worst_slot]:
                insert_pos = K - 1
                for j in range(K - 1):
                    if token_energy < top_energies[j]:
                        insert_pos = j
                        break

                if insert_pos < K - 1:
                    for j in range(K - 1, insert_pos, -1):
                        top_indices[j] = top_indices[j - 1]
                        top_energies[j] = top_energies[j - 1]

                top_indices[insert_pos] = tl.cast(token_idx, tl.int32)
                top_energies[insert_pos] = token_energy

        for k in range(K):
            tl.store(OUT_base + k * stride_out_k, top_indices[k])


def _cpu_energy_topk(
    X: torch.Tensor,             # [B, N, D]
    WV: torch.Tensor,            # [H, D, D_head]
    k: int,
    alpha_w: torch.Tensor,       # [H] già sigmoidizzato
) -> torch.Tensor:
    B, N, D = X.shape
    H = WV.shape[0]
    X_f = X.float()

    x_norm = X_f.norm(dim=-1)

    xv_norms = []
    for h in range(H):
        xv_h = (X_f @ WV[h].float()).norm(dim=-1)
        xv_norms.append(xv_h)
    xv_norm = torch.stack(xv_norms, dim=1)

    def znorm(t):
        mu = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (t - mu) / std

    x_norm_zn = znorm(x_norm.unsqueeze(1).expand(B, H, N))
    xv_norm_zn = znorm(xv_norm)

    alpha = alpha_w.view(1, -1, 1)
    scores = alpha * xv_norm_zn + (1.0 - alpha) * x_norm_zn

    k_safe = min(k, N)
    _, top_idx = torch.topk(scores, k=k_safe, dim=-1, sorted=True)
    top_idx = top_idx.sort(dim=-1)[0]

    if k_safe < k:
        pad = torch.zeros(B, H, k - k_safe, dtype=torch.int32, device=scores.device)
        top_idx = torch.cat([top_idx, pad], dim=-1)

    return top_idx.to(torch.int32)


def energy_topk_fused(X, WV, k, alpha_w):
    if not _TRITON_AVAILABLE or not X.is_cuda:
        return _cpu_energy_topk(X, WV, k, alpha_w)

    B, N, D = X.shape
    H, _, D_head = WV.shape
    landmark_idx = torch.empty((B, H, k), dtype=torch.int32, device=X.device)
    
    BLOCK_N = 128
    BLOCK_D = 64
    BLOCK_Dh = triton.next_power_of_2(D_head)
    grid = (triton.cdiv(N, BLOCK_N), H, B)
    
    _energy_topk_kernel[grid](
        X, WV, alpha_w,
        landmark_idx,
        N=N, D=D, D_head=D_head, K=k,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_Dh=BLOCK_Dh,
        stride_xb=X.stride(0), stride_xn=X.stride(1),
        stride_wh=WV.stride(0), stride_wd=WV.stride(1),
        stride_out_b=landmark_idx.stride(0), stride_out_h=landmark_idx.stride(1), stride_out_k=landmark_idx.stride(2),
    )
    
    landmark_idx, _ = landmark_idx.sort(dim=-1)
    return landmark_idx