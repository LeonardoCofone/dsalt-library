import torch
import triton
import triton.language as tl


@triton.jit
def _window_size_kernel(
    logits_ptr,
    out_ptr,
    n_min,
    n_max,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    logit = tl.load(logits_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-logit))
    w_cont = n_min + sig * (n_max - n_min)
    tl.store(out_ptr + offs, w_cont, mask=mask)


@triton.jit
def _build_window_mask_kernel(
    w_int_ptr,
    mask_ptr,
    seq_len,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < seq_len

    w_vals = tl.load(w_int_ptr + q_offs, mask=q_mask, other=0).to(tl.int32)

    for kb in range(0, seq_len, BLOCK_K):
        k_offs = kb + tl.arange(0, BLOCK_K)
        k_mask = k_offs < seq_len

        causal = k_offs[None, :] <= q_offs[:, None]
        in_window = (q_offs[:, None] - k_offs[None, :]) < w_vals[:, None]
        valid = causal & in_window & q_mask[:, None] & k_mask[None, :]

        out_offs = q_offs[:, None] * seq_len + k_offs[None, :]
        tl.store(mask_ptr + out_offs, valid.to(tl.int8), mask=q_mask[:, None] & k_mask[None, :])


def compute_window_sizes_triton(
    logits: torch.Tensor,
    n_min: int,
    n_max: int,
) -> torch.Tensor:
    N = logits.numel()
    out = torch.empty(N, dtype=torch.float32, device=logits.device)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _window_size_kernel[grid](
        logits.reshape(-1).float(),
        out,
        float(n_min),
        float(n_max),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(logits.shape)


def build_window_mask_triton(
    w_int: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, dtype=torch.int8, device=w_int.device)
    BLOCK_Q = 64
    BLOCK_K = 64
    grid = (triton.cdiv(seq_len, BLOCK_Q),)
    _build_window_mask_kernel[grid](
        w_int.reshape(-1).int(),
        mask,
        seq_len,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )
    return mask.bool()