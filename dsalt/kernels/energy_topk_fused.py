import logging
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

if _TRITON_AVAILABLE:

    @triton.jit
    def _energy_kernel(
        X_ptr,
        WV_ptr,
        alpha_ptr,
        out_ptr,
        N,
        D,
        D_HEAD: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
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

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        offs_dh = tl.arange(0, D_HEAD)

        acc_x  = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_xv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)

        x_base  = X_ptr  + pid_b * stride_xb
        wv_base = WV_ptr + pid_h * stride_wh

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            x_ptrs  = x_base  + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd
            wv_ptrs = wv_base + offs_d[:, None] * stride_wd  + offs_dh[None, :] * stride_wdh

            x_tile  = tl.load(x_ptrs,  mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            wv_tile = tl.load(wv_ptrs, mask=mask_d[:, None],                   other=0.0).to(tl.float32)

            acc_xv += tl.dot(x_tile, wv_tile)
            acc_x  += tl.sum(x_tile * x_tile, axis=1)

        alpha_raw = tl.load(alpha_ptr + pid_h).to(tl.float32)
        alpha     = 1.0 / (1.0 + tl.exp(-alpha_raw))

        acc_xv_norm = tl.sum(acc_xv * acc_xv, axis=1)
        energy = alpha * acc_xv_norm + (1.0 - alpha) * acc_x

        out_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_n
        tl.store(out_ptrs, energy, mask=mask_n)


def compute_energy(X, WV, alpha):
    assert _TRITON_AVAILABLE
    assert X.is_cuda and WV.is_cuda

    B, N, D    = X.shape
    H, _, D_HEAD = WV.shape

    X     = X.contiguous()
    WV    = WV.contiguous()
    alpha = alpha.detach().float().contiguous()

    out = torch.empty((B, H, N), device=X.device, dtype=torch.float32)

    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(min(D, 64))

    grid = (triton.cdiv(N, BLOCK_N), H, B)

    _energy_kernel[grid](
        X, WV, alpha, out,
        N=N,
        D=D,
        D_HEAD=triton.next_power_of_2(D_HEAD),
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        stride_xb=X.stride(0),
        stride_xn=X.stride(1),
        stride_xd=X.stride(2),
        stride_wh=WV.stride(0),
        stride_wd=WV.stride(1),
        stride_wdh=WV.stride(2),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
    )

    return out


def compute_energy_and_topk(X, WV, alpha, k):
    with torch.no_grad():
        scores = compute_energy(X, WV, alpha)

    B, H, N = scores.shape
    k_safe  = min(k, N)

    idx = torch.topk(scores, k=k_safe, dim=-1, sorted=False).indices
    idx = idx.sort(dim=-1).values.to(torch.int32)

    if k_safe < k:
        pad = torch.zeros((B, H, k - k_safe), device=scores.device, dtype=torch.int32)
        idx = torch.cat([idx, pad], dim=-1)

    return idx