import torch
import triton
import triton.language as tl


@triton.jit
def _dsalt_bwd_preprocess(
    Out, DO,
    Delta,
    stride_ot, stride_oh, stride_od,
    stride_dot, stride_doh, stride_dod,
    stride_dt, stride_dh,
    total_len: tl.constexpr,
    HEAD_DIM:  tl.constexpr,
    BLOCK_M:   tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    valid  = offs_m < total_len

    o = tl.load(
        Out + offs_m[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :] * stride_od,
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + offs_m[:, None] * stride_dot + pid_h * stride_doh + offs_d[None, :] * stride_dod,
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(
        Delta + offs_m * stride_dt + pid_h * stride_dh,
        delta,
        mask=valid,
    )


@triton.jit
def _dsalt_bwd_kernel(
    Q, K, V, Out, DO,
    DQ, DK, DV,
    W_sizes, Lmk_K, Lmk_V, Lmk_pos,
    Delta, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_dot, stride_doh, stride_dod,
    stride_dqt, stride_dqh, stride_dqd,
    stride_dkt, stride_dkh, stride_dkd,
    stride_dvt, stride_dvh, stride_dvd,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    stride_lph, stride_lpb, stride_lpk,
    stride_dt,  stride_dh,
    scale:    tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_LMK:   tl.constexpr,
):
    pid_bm = tl.program_id(0)
    pid_h  = tl.program_id(1)

    seq_id    = tl.load(Seq_block_map + pid_bm * 2).to(tl.int32)
    block_off = tl.load(Seq_block_map + pid_bm * 2 + 1).to(tl.int32)
    seq_start = tl.load(Cu_seqlens + seq_id).to(tl.int32)
    seq_end   = tl.load(Cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len   = seq_end - seq_start
    m_start   = block_off * BLOCK_M

    offs_m  = tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, HEAD_DIM)
    offs_n  = tl.arange(0, BLOCK_N)
    offs_lk = tl.arange(0, K_LMK)
    valid_m = (m_start + offs_m) < seq_len
    i_abs   = m_start + offs_m

    q_base  = seq_start + m_start + offs_m[:, None]
    q = tl.load(
        Q + q_base * stride_qt + pid_h * stride_qh + offs_d[None, :] * stride_qd,
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)

    do = tl.load(
        DO + q_base * stride_dot + pid_h * stride_doh + offs_d[None, :] * stride_dod,
        mask=valid_m[:, None], other=0.0,
    ).to(tl.float32)

    delta = tl.load(
        Delta + (seq_start + m_start + offs_m) * stride_dt + pid_h * stride_dh,
        mask=valid_m, other=0.0,
    ).to(tl.float32)

    w_sizes     = tl.load(W_sizes + seq_start + m_start + offs_m, mask=valid_m, other=1).to(tl.int32)
    w_sizes     = tl.maximum(w_sizes, 1)
    w_max_block = tl.max(w_sizes, axis=0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    lk = tl.load(
        Lmk_K
        + pid_h * stride_lkh
        + seq_id * stride_lkb
        + offs_lk[None, :] * stride_lks
        + offs_d[:, None] * stride_lkd,
    ).to(tl.float32)
    lv = tl.load(
        Lmk_V
        + pid_h * stride_lvh
        + seq_id * stride_lvb
        + offs_lk[None, :] * stride_lvs
        + offs_d[:, None] * stride_lvd,
    ).to(tl.float32)
    lmk_pos = tl.load(
        Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + offs_lk * stride_lpk,
    ).to(tl.int32)

    causal_lmk = lmk_pos[None, :] <= (m_start + offs_m[:, None])
    valid_lmk  = causal_lmk & valid_m[:, None]

    qk_lmk = tl.dot(q, lk) * scale
    qk_lmk = tl.where(valid_lmk, qk_lmk, float("-inf"))

    window_start = tl.maximum(0, m_start - w_max_block + 1)
    window_end   = m_start + BLOCK_M
    n_start_win  = window_start - (window_start % BLOCK_N)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    n_iter = (window_end - n_start_win + BLOCK_N - 1) // BLOCK_N
    n_tmp  = n_start_win
    for _ in range(0, n_iter):
        valid_n = ((offs_n + n_tmp) < seq_len) & (n_tmp < window_end)
        j_abs   = n_tmp + offs_n
        in_win  = (
            (j_abs[None, :] >= i_abs[:, None] - w_sizes[:, None] + 1) &
            (j_abs[None, :] <= i_abs[:, None])
        )
        final = in_win & valid_n[None, :] & valid_m[:, None]
        k_blk = tl.load(
            K + (seq_start + n_tmp + offs_n[None, :]) * stride_kt
              + pid_h * stride_kh + offs_d[:, None] * stride_kd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)
        qk    = tl.where(final, tl.dot(q, k_blk) * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p     = tl.where(final, tl.exp(qk - m_new[:, None]), 0.0)
        l_i   = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)
        m_i   = m_new
        n_tmp += BLOCK_N

    m_lmk = tl.maximum(m_i, tl.max(qk_lmk, axis=1))
    p_lmk = tl.where(valid_lmk, tl.exp(qk_lmk - m_lmk[:, None]), 0.0)
    l_i   = l_i * tl.exp(m_i - m_lmk) + tl.sum(p_lmk, axis=1)
    l_i   = tl.where(l_i > 1e-9, l_i, 1.0)

    p_lmk_norm = p_lmk / l_i[:, None]
    dp_lmk     = tl.dot(do, lv)
    ds_lmk     = p_lmk_norm * (dp_lmk - delta[:, None])
    ds_lmk     = tl.where(valid_lmk, ds_lmk * scale, 0.0)

    dq += tl.dot(ds_lmk.to(tl.float16), tl.trans(lk).to(tl.float16)).to(tl.float32)

    dlk = tl.dot(tl.trans(ds_lmk).to(tl.float16), q.to(tl.float16)).to(tl.float32)
    dlv = tl.dot(tl.trans(p_lmk_norm).to(tl.float16), do.to(tl.float16)).to(tl.float32)

    for lk_idx in range(0, K_LMK):
        lpos = tl.load(Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + lk_idx * stride_lpk).to(tl.int32)
        abs_tok = seq_start + lpos

        dk_ptr = DK + abs_tok * stride_dkt + pid_h * stride_dkh + offs_d * stride_dkd
        dv_ptr = DV + abs_tok * stride_dvt + pid_h * stride_dvh + offs_d * stride_dvd

        tl.atomic_add(dk_ptr, dlk[lk_idx, :])
        tl.atomic_add(dv_ptr, dlv[lk_idx, :])

    n_start = n_start_win
    for _ in range(0, n_iter):
        valid_n = ((offs_n + n_start) < seq_len) & (n_start < window_end)
        j_abs   = n_start + offs_n
        in_win  = (
            (j_abs[None, :] >= i_abs[:, None] - w_sizes[:, None] + 1) &
            (j_abs[None, :] <= i_abs[:, None])
        )
        final = in_win & valid_n[None, :] & valid_m[:, None]

        k_blk = tl.load(
            K + (seq_start + n_start + offs_n[None, :]) * stride_kt
              + pid_h * stride_kh + offs_d[:, None] * stride_kd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)
        v_blk = tl.load(
            V + (seq_start + n_start + offs_n[None, :]) * stride_vt
              + pid_h * stride_vh + offs_d[:, None] * stride_vd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)

        qk     = tl.where(final, tl.dot(q, k_blk) * scale, float("-inf"))
        p      = tl.where(final, tl.exp(qk - m_lmk[:, None]), 0.0)
        p_norm = p / l_i[:, None]

        dp   = tl.dot(do, v_blk)
        ds   = tl.where(final, p_norm * (dp - delta[:, None]) * scale, 0.0)

        dq  += tl.dot(ds.to(tl.float16), tl.trans(k_blk).to(tl.float16)).to(tl.float32)

        for n_idx in range(0, BLOCK_N):
            n_tok   = n_start + n_idx
            valid_j = (n_tok < seq_len) & (n_tok < window_end)
            if valid_j:
                abs_tok = seq_start + n_tok
                ds_col  = ds[:, n_idx]
                p_col   = p_norm[:, n_idx]

                dk_ptr = DK + abs_tok * stride_dkt + pid_h * stride_dkh + offs_d * stride_dkd
                dv_ptr = DV + abs_tok * stride_dvt + pid_h * stride_dvh + offs_d * stride_dvd

                dk_contrib = tl.sum(ds_col[:, None] * q, axis=0)
                dv_contrib = tl.sum(p_col[:, None] * do, axis=0)

                tl.atomic_add(dk_ptr, dk_contrib)
                tl.atomic_add(dv_ptr, dv_contrib)

        n_start += BLOCK_N

    tl.store(
        DQ + (seq_start + m_start + offs_m[:, None]) * stride_dqt
           + pid_h * stride_dqh + offs_d[None, :] * stride_dqd,
        dq.to(DQ.dtype.element_ty),
        mask=valid_m[:, None],
    )


def dsalt_triton_backward(
    grad_out:    torch.Tensor,
    q:           torch.Tensor,
    k:           torch.Tensor,
    v:           torch.Tensor,
    out:         torch.Tensor,
    lmk_K:      torch.Tensor,
    lmk_V:      torch.Tensor,
    lmk_pos:    torch.Tensor,
    w_sizes:    torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale:      float,
    seq_block_map: torch.Tensor,
    total_blk:  int,
    BLOCK_M:    int,
    BLOCK_N:    int,
    HEAD_DIM_C: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_len, n_heads, head_dim = q.shape
    device = q.device

    q_h   = q.contiguous().to(torch.float16)
    k_h   = k.contiguous().to(torch.float16)
    v_h   = v.contiguous().to(torch.float16)
    out_h = out.contiguous().to(torch.float16)
    do_h  = grad_out.contiguous().to(torch.float16)

    dq = torch.zeros_like(q_h)
    dk = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)
    dv = torch.zeros(total_len, n_heads, head_dim, device=device, dtype=torch.float32)

    delta = torch.empty(total_len, n_heads, device=device, dtype=torch.float32)

    grid_pre = (triton.cdiv(total_len, BLOCK_M), n_heads)
    _dsalt_bwd_preprocess[grid_pre](
        out_h, do_h, delta,
        out_h.stride(0), out_h.stride(1), out_h.stride(2),
        do_h.stride(0),  do_h.stride(1),  do_h.stride(2),
        delta.stride(0), delta.stride(1),
        total_len=total_len,
        HEAD_DIM=HEAD_DIM_C,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    num_warps = 4 if head_dim <= 64 else 2
    cu_int    = cu_seqlens.to(torch.int32).contiguous()

    _dsalt_bwd_kernel[(total_blk, n_heads)](
        q_h, k_h, v_h, out_h, do_h,
        dq, dk, dv,
        w_sizes.clamp(min=1).long().contiguous(),
        lmk_K.to(torch.float16).contiguous(),
        lmk_V.to(torch.float16).contiguous(),
        lmk_pos.to(torch.int32).contiguous(),
        delta, cu_int, seq_block_map,
        q_h.stride(0),  q_h.stride(1),  q_h.stride(2),
        k_h.stride(0),  k_h.stride(1),  k_h.stride(2),
        v_h.stride(0),  v_h.stride(1),  v_h.stride(2),
        out_h.stride(0), out_h.stride(1), out_h.stride(2),
        do_h.stride(0),  do_h.stride(1),  do_h.stride(2),
        dq.stride(0),   dq.stride(1),   dq.stride(2),
        dk.stride(0),   dk.stride(1),   dk.stride(2),
        dv.stride(0),   dv.stride(1),   dv.stride(2),
        lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
        lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
        lmk_pos.stride(0), lmk_pos.stride(1), lmk_pos.stride(2),
        delta.stride(0), delta.stride(1),
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_C,
        K_LMK=lmk_K.shape[2],
        num_warps=num_warps,
        num_stages=1,
    )

    return dq.float(), dk.float(), dv.float()