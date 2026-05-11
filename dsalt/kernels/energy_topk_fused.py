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
    def _energy_topk_fused_kernel(
        X_ptr,
        WV_ptr,
        alpha_ptr,
        out_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        D_HEAD: tl.constexpr,
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
        offs_d  = tl.arange(0, D)
        offs_dh = tl.arange(0, D_HEAD)
        mask_n  = offs_n < N
        alpha = tl.load(alpha_ptr + pid_h).to(tl.float32)

        x = tl.load(
            X_ptr + pid_b * stride_xb
            + offs_n[:, None] * stride_xn
            + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float32)

        wv = tl.load(
            WV_ptr + pid_h * stride_wh
            + offs_d[:, None] * stride_wd
            + offs_dh[None, :],
        ).to(tl.float32)

        x_sq  = tl.sum(x * x, axis=1)
        xv    = tl.dot(x, wv)
        xv_sq = tl.sum(xv * xv, axis=1)

        n_valid = tl.sum(mask_n.to(tl.float32), axis=0)

        x_sq_mu  = tl.sum(x_sq  * mask_n.to(tl.float32), axis=0) / n_valid
        xv_sq_mu = tl.sum(xv_sq * mask_n.to(tl.float32), axis=0) / n_valid

        x_sq_var  = tl.sum((x_sq  - x_sq_mu)  * (x_sq  - x_sq_mu)  * mask_n.to(tl.float32), axis=0) / tl.maximum(n_valid - 1.0, 1.0)
        xv_sq_var = tl.sum((xv_sq - xv_sq_mu) * (xv_sq - xv_sq_mu) * mask_n.to(tl.float32), axis=0) / tl.maximum(n_valid - 1.0, 1.0)

        x_sq_z  = (x_sq  - x_sq_mu)  / (tl.sqrt(x_sq_var)  + 1e-6)
        xv_sq_z = (xv_sq - xv_sq_mu) / (tl.sqrt(xv_sq_var) + 1e-6)

        energy = alpha * xv_sq_z + (1.0 - alpha) * x_sq_z

        tl.store(
            out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_n,
            energy,
            mask=mask_n,
        )


def compute_energy(
    X: torch.Tensor,
    WV: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    assert X.is_cuda and WV.is_cuda, "X and WV must be on CUDA"
    assert _TRITON_AVAILABLE, "Triton is not available"

    B, N, D     = X.shape
    H, D2, D_HEAD = WV.shape

    assert D == D2, f"Dimension mismatch: X.D={D}, WV.D={D2}"
    assert alpha.shape == (H,), f"alpha must be shape [H={H}], got {alpha.shape}"
    assert D      == triton.next_power_of_2(D),      f"D={D} must be a power of 2 for tl.dot"
    assert D_HEAD == triton.next_power_of_2(D_HEAD), f"D_HEAD={D_HEAD} must be a power of 2 for tl.dot"

    X_c     = X.contiguous()
    WV_c    = WV.contiguous()

    alpha_c = torch.sigmoid(alpha).contiguous().float()

    out = torch.empty((B, H, N), device=X.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid    = (triton.cdiv(N, BLOCK_N), H, B)

    logger.debug(
        "compute_energy | B=%d H=%d N=%d D=%d D_HEAD=%d BLOCK_N=%d grid=%s",
        B, H, N, D, D_HEAD, BLOCK_N, grid,
    )

    _energy_topk_fused_kernel[grid](
        X_c, WV_c, alpha_c, out,
        N=N, D=D, D_HEAD=D_HEAD, BLOCK_N=BLOCK_N,
        stride_xb=X_c.stride(0), stride_xn=X_c.stride(1),
        stride_wh=WV_c.stride(0), stride_wd=WV_c.stride(1),
        stride_ob=out.stride(0),  stride_oh=out.stride(1),
    )

    return out


def compute_energy_and_topk(
    X: torch.Tensor,
    WV: torch.Tensor,
    alpha: torch.Tensor,
    k: int,
    window_end: torch.Tensor,
) -> torch.Tensor:
    scores = compute_energy(X, WV, alpha)

    B, H, N = scores.shape

    pos       = torch.arange(N, device=scores.device).view(1, 1, N)
    in_window = pos >= window_end.unsqueeze(-1)
    scores    = scores.masked_fill(in_window, float("-inf"))

    k_safe  = min(k, N)
    top_idx = torch.topk(scores, k=k_safe, dim=-1, largest=True, sorted=False).indices
    top_idx = top_idx.sort(dim=-1).values.to(torch.int32)

    if k_safe < k:
        logger.warning(
            "compute_energy_and_topk | N=%d < k=%d, padding landmark indices with zeros",
            N, k,
        )
        pad     = torch.zeros(B, H, k - k_safe, device=scores.device, dtype=torch.int32)
        top_idx = torch.cat([top_idx, pad], dim=-1)

    logger.debug(
        "compute_energy_and_topk | selected %d landmarks out of %d candidates | B=%d H=%d",
        k_safe, N, B, H,
    )

    return top_idx