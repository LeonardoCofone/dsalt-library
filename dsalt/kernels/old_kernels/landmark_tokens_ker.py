import torch
import triton
import triton.language as tl


@triton.jit
def _hybrid_energy_kernel(
    x_ptr,
    xv_ptr,
    scores_ptr,
    alpha,
    mu_x,
    sigma_x,
    mu_v,
    sigma_v,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    x_base = x_ptr + pid * D
    xv_base = xv_ptr + pid * D

    acc_x = 0.0
    acc_v = 0.0

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        xval = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        vval = tl.load(xv_base + cols, mask=mask, other=0.0).to(tl.float32)
        acc_x += tl.sum(xval * xval)
        acc_v += tl.sum(vval * vval)

    norm_x = tl.sqrt(acc_x)
    norm_v = tl.sqrt(acc_v)

    z_x = (norm_x - mu_x) / (sigma_x + 1e-8)
    z_v = (norm_v - mu_v) / (sigma_v + 1e-8)

    score = alpha * z_v + (1.0 - alpha) * z_x
    tl.store(scores_ptr + pid, score)


def compute_hybrid_energy_triton(
    x: torch.Tensor,
    xv: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    N, D = x.shape
    scores = torch.empty(N, dtype=torch.float32, device=x.device)

    norm_x = x.float().norm(dim=-1)
    norm_v = xv.float().norm(dim=-1)
    mu_x = norm_x.mean().item()
    sigma_x = norm_x.std().item()
    mu_v = norm_v.mean().item()
    sigma_v = norm_v.std().item()

    BLOCK_D = min(256, triton.next_power_of_2(D))
    _hybrid_energy_kernel[(N,)](
        x.float().contiguous(),
        xv.float().contiguous(),
        scores,
        alpha,
        mu_x, sigma_x,
        mu_v, sigma_v,
        N, D,
        BLOCK_D=BLOCK_D,
    )
    return scores


@triton.jit
def _yarn_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    x_base = x_ptr + pid * D
    out_base = out_ptr + pid * D
    half = D // 2

    for off in range(0, half, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < half

        x0 = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_base + cols + half, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + pid * half + cols, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + pid * half + cols, mask=mask, other=0.0).to(tl.float32)

        out0 = x0 * c - x1 * s
        out1 = x0 * s + x1 * c

        tl.store(out_base + cols, out0, mask=mask)
        tl.store(out_base + cols + half, out1, mask=mask)


def apply_yarn_rope_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    N, D = x.shape
    assert D % 2 == 0
    out = torch.empty_like(x, dtype=torch.float32)
    BLOCK_D = min(128, triton.next_power_of_2(D // 2))
    _yarn_rope_kernel[(N,)](
        x.float().contiguous(),
        cos.float().contiguous(),
        sin.float().contiguous(),
        out,
        N, D,
        BLOCK_D=BLOCK_D,
    )
    return out.to(x.dtype)


def select_landmarks(scores, k, window_mask):
    outside = ~window_mask
    scores = torch.where(outside, scores, torch.tensor(float("-inf"), device=scores.device))
    k = min(k, int(outside.sum().detach().cpu()))
    _, indices = torch.topk(scores, k)
    return indices