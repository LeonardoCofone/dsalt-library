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
        mask_n = offs_n < N

        X_base = X_ptr + pid_b * stride_xb
        WV_base = WV_ptr + pid_h * stride_wh
        O_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh

        alpha = tl.load(alpha_ptr + pid_h).to(tl.float32)

        acc_x = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_xv = tl.zeros([BLOCK_N], dtype=tl.float32)

        offs_d = tl.arange(0, D)

        for d in range(0, D):

            x = tl.load(
                X_base + offs_n[:, None] * stride_xn + offs_d[None, d],
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float32)

            wv = tl.load(
                WV_base + offs_d[d] * stride_wd,
                mask=True,
                other=0.0,
            ).to(tl.float32)

            acc_x += x * x
            acc_xv += (x * wv) * (x * wv)

        energy = alpha * acc_xv + (1.0 - alpha) * acc_x

        tl.store(O_base + offs_n, energy, mask=mask_n)

def compute_energy(X, WV, alpha):
    assert X.is_cuda and WV.is_cuda
    assert _TRITON_AVAILABLE

    B, N, D = X.shape
    H, _, D_HEAD = WV.shape

    X = X.contiguous()
    WV = WV.contiguous()
    alpha = torch.sigmoid(alpha).contiguous().float()

    out = torch.empty((B, H, N), device=X.device, dtype=torch.float32)

    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N), H, B)

    _energy_topk_fused_kernel[grid](
        X, WV, alpha, out,
        N=N, D=D, D_HEAD=D_HEAD,
        BLOCK_N=BLOCK_N,
        stride_xb=X.stride(0),
        stride_xn=X.stride(1),
        stride_wh=WV.stride(0),
        stride_wd=WV.stride(1),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
    )

    return out

def compute_energy_and_topk(X, WV, alpha, k, window_end=None):
    scores = compute_energy(X, WV, alpha)

    B, H, N = scores.shape

    if window_end is not None:
        pos = torch.arange(N, device=scores.device).view(1, 1, N)
        scores = scores.masked_fill(
            pos >= window_end.unsqueeze(-1),
            float("-inf")
        )

    k_safe = min(k, N)

    idx = torch.topk(scores, k=k_safe, dim=-1, sorted=False).indices
    idx = idx.sort(dim=-1).values.to(torch.int32)

    if k_safe < k:
        pad = torch.zeros(
            (B, H, k - k_safe),
            device=scores.device,
            dtype=torch.int32
        )
        idx = torch.cat([idx, pad], dim=-1)

    return idx