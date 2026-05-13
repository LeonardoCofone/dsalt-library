import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    window_mask_ptr,
    lmk_idx_ptr,
    Out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mn,
    stride_lb, stride_lh, stride_lk,
    B, H, N, D,
    k_lmk,
    scale,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    q_base   = Q_ptr   + pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn
    out_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_n * stride_on
    mask_base = window_mask_ptr + pid_b * stride_mb + pid_n * stride_mn
    lmk_base  = lmk_idx_ptr    + pid_b * stride_lb + pid_h * stride_lh

    q = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask_d = cols < D
        q += tl.load(q_base + cols, mask=mask_d, other=0.0).to(tl.float32)

    m_i  = -1e9
    l_i  = 0.0
    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k_pos in range(0, N):
        use = tl.load(mask_base + k_pos).to(tl.int1)
        if use:
            k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh + k_pos * stride_kn
            v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh + k_pos * stride_vn

            dot = 0.0
            for off in range(0, D, BLOCK_D):
                cols = off + tl.arange(0, BLOCK_D)
                mask_d = cols < D
                k_val = tl.load(k_base + cols, mask=mask_d, other=0.0).to(tl.float32)
                dot = dot + tl.sum(q * tl.where(mask_d, k_val, 0.0), axis=0)
            dot = dot * scale

            m_new    = tl.maximum(m_i, dot)
            rescale  = tl.exp(m_i - m_new)
            exp_dot  = tl.exp(dot - m_new)
            l_i      = l_i * rescale + exp_dot
            acc      = acc * rescale

            for off in range(0, D, BLOCK_D):
                cols  = off + tl.arange(0, BLOCK_D)
                mask_d = cols < D
                v_val = tl.load(v_base + cols, mask=mask_d, other=0.0).to(tl.float32)
                acc   += tl.where(mask_d, exp_dot * v_val, 0.0)

            m_i = m_new

    for ki in range(0, k_lmk):
        lmk_pos  = tl.load(lmk_base + ki).to(tl.int32)
        is_window = tl.load(mask_base + lmk_pos).to(tl.int1)
        if not is_window:
            k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh + lmk_pos * stride_kn
            v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh + lmk_pos * stride_vn

            dot = 0.0
            for off in range(0, D, BLOCK_D):
                cols  = off + tl.arange(0, BLOCK_D)
                mask_d = cols < D
                k_val = tl.load(k_base + cols, mask=mask_d, other=0.0).to(tl.float32)
                dot = dot + tl.sum(q * tl.where(mask_d, k_val, 0.0), axis=0)
            dot = dot * scale

            m_new   = tl.maximum(m_i, dot)
            rescale = tl.exp(m_i - m_new)
            exp_dot = tl.exp(dot - m_new)
            l_i     = l_i * rescale + exp_dot
            acc     = acc * rescale

            for off in range(0, D, BLOCK_D):
                cols  = off + tl.arange(0, BLOCK_D)
                mask_d = cols < D
                v_val = tl.load(v_base + cols, mask=mask_d, other=0.0).to(tl.float32)
                acc  += tl.where(mask_d, exp_dot * v_val, 0.0)

            m_i = m_new

    inv_l = 1.0 / (l_i + 1e-10)
    for off in range(0, D, BLOCK_D):
        cols  = off + tl.arange(0, BLOCK_D)
        mask_d = cols < D
        tl.store(out_base + cols, (acc * inv_l).to(tl.float32), mask=mask_d)


def sparse_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_mask: torch.Tensor,
    lmk_indices: torch.Tensor,
) -> torch.Tensor:
    B, H, N, D = q.shape
    scale  = float(D) ** -0.5
    out    = torch.zeros(B, H, N, D, dtype=torch.float32, device=q.device)
    BLOCK_D = min(64, triton.next_power_of_2(D))

    _sparse_attn_fwd_kernel[(B, H, N)](
        q.float().contiguous(),
        k.float().contiguous(),
        v.float().contiguous(),
        window_mask.to(torch.int8).contiguous(),
        lmk_indices.int().contiguous(),
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        window_mask.stride(0), window_mask.stride(1),
        lmk_indices.stride(0), lmk_indices.stride(1), lmk_indices.stride(2),
        B, H, N, D,
        lmk_indices.shape[-1],
        scale,
        BLOCK_D=BLOCK_D,
    )
    return out.to(q.dtype)


def sparse_attention_pytorch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_mask: torch.Tensor,
    lmk_indices: torch.Tensor,
) -> torch.Tensor:
    B, H, N, D  = q.shape
    scale       = float(D) ** -0.5
    k_lmk       = lmk_indices.shape[-1]
    device      = q.device

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    combined_mask = window_mask.unsqueeze(1).expand(B, H, N, N).clone().float()

    b_idx  = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, k_lmk)
    h_idx  = torch.arange(H, device=device).view(1, H, 1).expand(B, H, k_lmk)
    q_idx  = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, H, N, k_lmk)
    lmk_exp = lmk_indices.unsqueeze(2).expand(B, H, N, k_lmk)

    causal_ok = lmk_exp < q_idx
    b_exp = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, N, k_lmk)
    h_exp = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, N, k_lmk)
    n_exp = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, H, N, k_lmk)

    combined_mask[
        b_exp[causal_ok], h_exp[causal_ok], n_exp[causal_ok], lmk_exp[causal_ok]
    ] = 1.0

    scores = scores.masked_fill(combined_mask == 0, float("-inf"))
    attn   = torch.softmax(scores, dim=-1)
    attn   = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v.float()).to(q.dtype)