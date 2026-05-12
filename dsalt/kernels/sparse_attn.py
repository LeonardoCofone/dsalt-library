"""
Questo modulo implementa l'attenzione sparsa per DSALT, fornendo la funzione dsalt_attention e la classe DSALTAttentionFunction.
"""
import logging
import math

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_gpu_config(device: int = 0):
    if not torch.cuda.is_available():
        return 64, 32, 4, 2
    sm = torch.cuda.get_device_capability(device)
    sm = sm[0] * 10 + sm[1]
    if sm >= 80:
        return 128, 64, 8, 3
    elif sm >= 75:
        return 64, 32, 4, 2
    return 32, 32, 4, 1


if _TRITON_AVAILABLE:

    @triton.jit
    def _dsalt_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        LM_ptr,
        Out_ptr, LSE_ptr,
        Win_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_lmb, stride_lmh, stride_lmk,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_wb, stride_wh, stride_wn,
        stride_lseb, stride_lseh,
        N: tl.constexpr,
        D: tl.constexpr,
        K_LMK: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        mask_m = offs_m < N

        Q_bh  = Q_ptr  + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr  + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr  + pid_b * stride_vb + pid_h * stride_vh
        LM_bh = LM_ptr + pid_b * stride_lmb + pid_h * stride_lmh
        O_bh  = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh

        q   = tl.load(
            Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
            mask=mask_m[:, None], other=0.0,
        ).to(tl.float32)
        w_i = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        q_start   = pid_m * BLOCK_M
        max_w     = tl.max(w_i, axis=0)
        win_start = tl.maximum(0, q_start - max_w)
        k_blk     = (win_start // BLOCK_N) * BLOCK_N

        while k_blk < q_start + BLOCK_M:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_tile = tl.load(
                K_bh + offs_n[None, :] * stride_kn + offs_d[:, None],
                mask=mask_n[None, :], other=0.0,
            ).to(tl.float32)
            s = tl.dot(q, k_tile) * SCALE

            causal = offs_n[None, :] <= offs_m[:, None]
            in_win = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            valid  = causal & in_win & mask_n[None, :] & mask_m[:, None]
            s      = tl.where(valid, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p     = tl.where(valid, tl.exp(s - m_new[:, None]), 0.0)

            v_tile = tl.load(
                V_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=mask_n[:, None], other=0.0,
            ).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v_tile.to(tl.float16)).to(tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
            k_blk += BLOCK_N

        lmk_offs = tl.arange(0, K_LMK)
        idx_k    = tl.load(
            LM_bh + lmk_offs * stride_lmk,
            mask=lmk_offs < K_LMK, other=0,
        ).to(tl.int32)
        idx_k = tl.minimum(tl.maximum(idx_k, 0), N - 1)

        kl = tl.load(
            K_bh + idx_k[:, None] * stride_kn + offs_d[None, :],
            mask=(lmk_offs[:, None] < K_LMK), other=0.0,
        ).to(tl.float32)
        s_lmk = tl.dot(q, tl.trans(kl)) * SCALE

        lmk_ok = (lmk_offs[None, :] < K_LMK) & mask_m[:, None]
        s_lmk  = tl.where(lmk_ok, s_lmk, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(s_lmk, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p_lmk  = tl.where(lmk_ok, tl.exp(s_lmk - m_new[:, None]), 0.0)

        vl = tl.load(
            V_bh + idx_k[:, None] * stride_vn + offs_d[None, :],
            mask=(lmk_offs[:, None] < K_LMK), other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p_lmk.to(tl.float16), vl.to(tl.float16)).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p_lmk, axis=1)
        m_i = m_new

        l_safe = tl.maximum(l_i, 1e-6)
        lse    = m_i + tl.log(l_safe)
        out_f  = acc / l_safe[:, None]

        tl.store(
            LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh + offs_m,
            lse, mask=mask_m,
        )
        tl.store(
            O_bh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
            out_f.to(Q_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )

    @triton.jit
    def _dsalt_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr, LSE_ptr, DQ_ptr,
        LM_ptr, Win_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_lmb, stride_lmh, stride_lmk,
        stride_wb, stride_wh, stride_wn,
        stride_lseb, stride_lseh,
        N: tl.constexpr,
        D: tl.constexpr,
        K_LMK: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        mask_m = offs_m < N

        Q_bh  = Q_ptr  + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr  + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr  + pid_b * stride_vb + pid_h * stride_vh
        DO_bh = DO_ptr + pid_b * stride_qb + pid_h * stride_qh
        DQ_bh = DQ_ptr + pid_b * stride_qb + pid_h * stride_qh
        LM_bh = LM_ptr + pid_b * stride_lmb + pid_h * stride_lmh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh

        q   = tl.load(Q_bh  + offs_m[:, None] * stride_qn + offs_d[None, :],
                      mask=mask_m[:, None], other=0.0).to(tl.float32)
        do  = tl.load(DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                      mask=mask_m[:, None], other=0.0).to(tl.float32)
        lse = tl.load(LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh + offs_m,
                      mask=mask_m, other=0.0)
        w_i = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)

        dq        = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        q_start   = pid_m * BLOCK_M
        max_w     = tl.max(w_i, axis=0)
        win_start = tl.maximum(0, q_start - max_w)
        k_blk     = (win_start // BLOCK_N) * BLOCK_N

        while k_blk < q_start + BLOCK_M:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_tile = tl.load(K_bh + offs_n[:, None] * stride_kn + offs_d[None, :],
                             mask=mask_n[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(V_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
                             mask=mask_n[:, None], other=0.0).to(tl.float32)

            s      = tl.dot(q, tl.trans(k_tile)) * SCALE
            causal = offs_n[None, :] <= offs_m[:, None]
            in_win = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            valid  = causal & in_win & mask_n[None, :] & mask_m[:, None]
            s      = tl.where(valid, s, float("-inf"))
            p      = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)

            dp     = tl.dot(do, tl.trans(v_tile))
            rowsum = tl.sum(p * dp, axis=1)
            ds     = tl.where(valid, p * (dp - rowsum[:, None]) * SCALE, 0.0)
            dq    += tl.dot(ds.to(tl.float16), k_tile.to(tl.float16)).to(tl.float32)
            k_blk += BLOCK_N

        lmk_offs = tl.arange(0, K_LMK)
        idx_k    = tl.load(LM_bh + lmk_offs * stride_lmk,
                           mask=lmk_offs < K_LMK, other=0).to(tl.int32)
        idx_k    = tl.minimum(tl.maximum(idx_k, 0), N - 1)

        kl = tl.load(K_bh + idx_k[:, None] * stride_kn + offs_d[None, :],
                     mask=(lmk_offs[:, None] < K_LMK), other=0.0).to(tl.float32)
        vl = tl.load(V_bh + idx_k[:, None] * stride_vn + offs_d[None, :],
                     mask=(lmk_offs[:, None] < K_LMK), other=0.0).to(tl.float32)

        s_lmk  = tl.dot(q, tl.trans(kl)) * SCALE
        lmk_ok = (lmk_offs[None, :] < K_LMK) & mask_m[:, None]
        s_lmk  = tl.where(lmk_ok, s_lmk, float("-inf"))
        p_lmk  = tl.where(lmk_ok, tl.exp(s_lmk - lse[:, None]), 0.0)

        dp_l  = tl.dot(do, tl.trans(vl))
        rs_l  = tl.sum(p_lmk * dp_l, axis=1)
        ds_l  = tl.where(lmk_ok, p_lmk * (dp_l - rs_l[:, None]) * SCALE, 0.0)
        dq   += tl.dot(ds_l.to(tl.float16), kl.to(tl.float16)).to(tl.float32)

        tl.store(DQ_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                 dq.to(Q_ptr.dtype.element_ty), mask=mask_m[:, None])

    @triton.jit
    def _dsalt_bwd_dkdv_kernel(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr, LSE_ptr,
        DK_ptr, DV_ptr,
        Win_ptr,
        LM_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_wb, stride_wh, stride_wn,
        stride_lseb, stride_lseh,
        stride_lmb, stride_lmh, stride_lmk,
        N: tl.constexpr,
        D: tl.constexpr,
        K_LMK: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MAX_WIN: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        mask_n = offs_n < N

        Q_bh  = Q_ptr  + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr  + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr  + pid_b * stride_vb + pid_h * stride_vh
        DO_bh = DO_ptr + pid_b * stride_qb + pid_h * stride_qh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh
        DK_bh = DK_ptr + pid_b * stride_kb + pid_h * stride_kh
        DV_bh = DV_ptr + pid_b * stride_vb + pid_h * stride_vh
        LM_bh = LM_ptr + pid_b * stride_lmb + pid_h * stride_lmh

        k_tile = tl.load(K_bh + offs_n[:, None] * stride_kn + offs_d[None, :],
                         mask=mask_n[:, None], other=0.0).to(tl.float32)
        v_tile = tl.load(V_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
                         mask=mask_n[:, None], other=0.0).to(tl.float32)

        lmk_offs = tl.arange(0, K_LMK)
        lm_idx   = tl.load(LM_bh + lmk_offs * stride_lmk,
                           mask=lmk_offs < K_LMK, other=-1).to(tl.int32)

        is_lmk_n = tl.any(lm_idx[:, None] == offs_n[None, :], axis=0)

        dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

        k_start = pid_n * BLOCK_N
        q_lo    = tl.maximum(0, k_start)
        q_hi    = tl.minimum(N, k_start + BLOCK_N + MAX_WIN)
        q_blk   = (q_lo // BLOCK_M) * BLOCK_M

        while q_blk < q_hi:
            offs_m = q_blk + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N
            w_i    = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)

            q_t   = tl.load(Q_bh  + offs_m[:, None] * stride_qn + offs_d[None, :],
                            mask=mask_m[:, None], other=0.0).to(tl.float32)
            do_t  = tl.load(DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                            mask=mask_m[:, None], other=0.0).to(tl.float32)
            lse_t = tl.load(LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh + offs_m,
                            mask=mask_m, other=0.0)

            s      = tl.dot(q_t, tl.trans(k_tile)) * SCALE
            causal = offs_n[None, :] <= offs_m[:, None]
            in_win = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            valid  = causal & in_win & mask_n[None, :] & mask_m[:, None]
            s      = tl.where(valid, s, float("-inf"))
            p      = tl.where(valid, tl.exp(s - lse_t[:, None]), 0.0)

            dv    += tl.dot(tl.trans(p).to(tl.float16), do_t.to(tl.float16)).to(tl.float32)
            dp     = tl.dot(do_t, tl.trans(v_tile))
            rs     = tl.sum(p * dp, axis=1)
            ds     = tl.where(valid, p * (dp - rs[:, None]) * SCALE, 0.0)
            dk    += tl.dot(tl.trans(ds).to(tl.float16), q_t.to(tl.float16)).to(tl.float32)
            q_blk += BLOCK_M

        q_blk2 = 0
        while q_blk2 < N:
            offs_m2 = q_blk2 + tl.arange(0, BLOCK_M)
            mask_m2 = offs_m2 < N
            w_i2    = tl.load(W_bh + offs_m2 * stride_wn, mask=mask_m2, other=0)

            q_t2   = tl.load(Q_bh  + offs_m2[:, None] * stride_qn + offs_d[None, :],
                             mask=mask_m2[:, None], other=0.0).to(tl.float32)
            do_t2  = tl.load(DO_bh + offs_m2[:, None] * stride_qn + offs_d[None, :],
                             mask=mask_m2[:, None], other=0.0).to(tl.float32)
            lse_t2 = tl.load(LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh + offs_m2,
                             mask=mask_m2, other=0.0)

            causal2         = offs_n[None, :] <= offs_m2[:, None]
            in_win2         = offs_n[None, :] >= (offs_m2[:, None] - w_i2[:, None])
            already_covered = causal2 & in_win2
            valid2          = is_lmk_n[None, :] & (~already_covered) & mask_m2[:, None] & mask_n[None, :]

            s2 = tl.dot(q_t2, tl.trans(k_tile)) * SCALE
            s2 = tl.where(valid2, s2, float("-inf"))
            p2 = tl.where(valid2, tl.exp(s2 - lse_t2[:, None]), 0.0)

            dv    += tl.dot(tl.trans(p2).to(tl.float16), do_t2.to(tl.float16)).to(tl.float32)
            dp2    = tl.dot(do_t2, tl.trans(v_tile))
            rs2    = tl.sum(p2 * dp2, axis=1)
            ds2    = tl.where(valid2, p2 * (dp2 - rs2[:, None]) * SCALE, 0.0)
            dk    += tl.dot(tl.trans(ds2).to(tl.float16), q_t2.to(tl.float16)).to(tl.float32)
            q_blk2 += BLOCK_M

        tl.store(DK_bh + offs_n[:, None] * stride_kn + offs_d[None, :],
                 dk.to(K_ptr.dtype.element_ty), mask=mask_n[:, None])
        tl.store(DV_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
                 dv.to(V_ptr.dtype.element_ty), mask=mask_n[:, None])


def _cpu_ref_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_sizes: torch.Tensor,
    landmark_idx: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = Q.shape
    K_lmk = landmark_idx.shape[-1]
    device = Q.device

    pos = torch.arange(N, device=device)
    i   = pos.view(1, 1, N, 1)
    j   = pos.view(1, 1, 1, N)

    w = window_sizes.unsqueeze(-1).long()
    window_mask = (j <= i) & (j >= (i - w))

    lmk_flat = landmark_idx.clamp(0, N - 1).long()
    lmk_mask = torch.zeros((B, H, N, N), dtype=torch.bool, device=device)
    q_idx    = pos.view(1, 1, N, 1).expand(B, H, N, K_lmk)
    b_idx    = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, N, K_lmk)
    h_idx    = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, N, K_lmk)
    k_idx    = lmk_flat.unsqueeze(2).expand(B, H, N, K_lmk)
    lmk_mask[b_idx, h_idx, q_idx, k_idx] = True
    lmk_mask = lmk_mask & (j <= i)

    mask = window_mask | lmk_mask

    s   = torch.einsum("bhid,bhjd->bhij", Q.float(), K.float()) * scale
    s   = s.masked_fill(~mask, float("-inf"))
    a   = torch.softmax(s, dim=-1).nan_to_num(0.0)
    out = torch.einsum("bhij,bhjd->bhid", a, V.float())
    lse = torch.logsumexp(s, dim=-1)
    lse = torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)
    return out.to(Q.dtype), lse


def _cpu_ref_bwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dOut: torch.Tensor,
    window_sizes: torch.Tensor,
    landmark_idx: torch.Tensor,
    LSE: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, D = Q.shape
    K_lmk = landmark_idx.shape[-1]
    device = Q.device

    pos = torch.arange(N, device=device)
    i   = pos.view(1, 1, N, 1)
    j   = pos.view(1, 1, 1, N)

    w = window_sizes.unsqueeze(-1).long()
    window_mask = (j <= i) & (j >= (i - w))

    lmk_flat = landmark_idx.clamp(0, N - 1).long()
    lmk_mask = torch.zeros((B, H, N, N), dtype=torch.bool, device=device)
    q_idx    = pos.view(1, 1, N, 1).expand(B, H, N, K_lmk)
    b_idx    = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, N, K_lmk)
    h_idx    = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, N, K_lmk)
    k_idx    = lmk_flat.unsqueeze(2).expand(B, H, N, K_lmk)
    lmk_mask[b_idx, h_idx, q_idx, k_idx] = True
    lmk_mask = lmk_mask & (j <= i)

    mask = window_mask | lmk_mask

    s   = torch.einsum("bhid,bhjd->bhij", Q.float(), K.float()) * scale
    s   = s.masked_fill(~mask, float("-inf"))
    a   = torch.softmax(s, dim=-1).nan_to_num(0.0)

    dV  = torch.einsum("bhij,bhid->bhjd", a, dOut.float())
    dp  = torch.einsum("bhid,bhjd->bhij", dOut.float(), V.float())
    ds  = a * (dp - (a * dp).sum(dim=-1, keepdim=True)) * scale
    ds  = ds.masked_fill(~mask, 0.0)
    dQ  = torch.einsum("bhij,bhjd->bhid", ds, K.float())
    dK  = torch.einsum("bhij,bhid->bhjd", ds, Q.float())
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


class DSALTAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, window_sizes, landmark_idx):
        B, H, N, D = Q.shape
        K_lmk = landmark_idx.shape[-1]
        scale = 1.0 / math.sqrt(D)

        Out = torch.empty_like(Q)
        LSE = torch.empty((B, H, N), dtype=torch.float32, device=Q.device)

        use_triton = Q.is_cuda and _TRITON_AVAILABLE

        if use_triton:
            BM, BN, WARPS, STAGES = _get_gpu_config(Q.get_device())
            BD  = triton.next_power_of_2(D)
            Q_c = Q.contiguous()
            K_c = K.contiguous()
            V_c = V.contiguous()
            ws  = window_sizes.contiguous().to(torch.int32)
            lm  = landmark_idx.contiguous().to(torch.int32)

            grid = (triton.cdiv(N, BM), H, B)
            _dsalt_fwd_kernel[grid](
                Q_c, K_c, V_c, lm, Out, LSE, ws,
                Q_c.stride(0), Q_c.stride(1), Q_c.stride(2), Q_c.stride(3),
                K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
                V_c.stride(0), V_c.stride(1), V_c.stride(2), V_c.stride(3),
                lm.stride(0), lm.stride(1), lm.stride(2),
                Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
                ws.stride(0), ws.stride(1), ws.stride(2),
                LSE.stride(0), LSE.stride(1),
                N=N, D=BD, K_LMK=K_lmk, SCALE=scale,
                BLOCK_M=BM, BLOCK_N=BN,
                num_warps=WARPS, num_stages=STAGES,
            )
        else:
            Out, LSE = _cpu_ref_fwd(Q, K, V, window_sizes, landmark_idx, scale)
            BM, BN, WARPS, STAGES = 64, 32, 4, 2
            BD = D

        ctx.save_for_backward(Q, K, V, window_sizes, landmark_idx, LSE)
        ctx.scale    = scale
        ctx.max_win  = int(window_sizes.max().item())
        ctx.use_triton = use_triton
        ctx.BM       = BM
        ctx.BN       = BN
        ctx.BD       = BD
        ctx.WARPS    = WARPS
        ctx.STAGES   = STAGES
        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, window_sizes, landmark_idx, LSE = ctx.saved_tensors
        B, H, N, D = Q.shape
        K_lmk = landmark_idx.shape[-1]
        dOut  = dOut.contiguous()

        if ctx.use_triton:
            BM      = ctx.BM
            BN      = ctx.BN
            BD      = ctx.BD
            WARPS   = ctx.WARPS
            STAGES  = ctx.STAGES
            scale   = ctx.scale
            max_win = ctx.max_win

            dQ = torch.zeros_like(Q)
            dK = torch.zeros_like(K)
            dV = torch.zeros_like(V)

            grid_dkdv = (triton.cdiv(N, BN), H, B)
            _dsalt_bwd_dkdv_kernel[grid_dkdv](
                Q, K, V, dOut, LSE, dK, dV, window_sizes, landmark_idx,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                LSE.stride(0), LSE.stride(1),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                N=N, D=BD, K_LMK=K_lmk, SCALE=scale,
                BLOCK_M=BM, BLOCK_N=BN, MAX_WIN=max_win,
                num_warps=WARPS, num_stages=STAGES,
            )

            grid_dq = (triton.cdiv(N, BM), H, B)
            _dsalt_bwd_dq_kernel[grid_dq](
                Q, K, V, dOut, LSE, dQ, landmark_idx, window_sizes,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                LSE.stride(0), LSE.stride(1),
                N=N, D=BD, K_LMK=K_lmk, SCALE=scale,
                BLOCK_M=BM, BLOCK_N=BN,
                num_warps=WARPS, num_stages=STAGES,
            )
        else:
            dQ, dK, dV = _cpu_ref_bwd(
                Q, K, V, dOut, window_sizes, landmark_idx, LSE, ctx.scale
            )

        return dQ, dK, dV, None, None


def dsalt_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_sizes: torch.Tensor,
    landmark_idx: torch.Tensor,
) -> torch.Tensor:
    assert Q.shape == K.shape == V.shape, \
        f"Q/K/V shape mismatch: {Q.shape} {K.shape} {V.shape}"
    assert window_sizes.shape[:2] == (Q.shape[0], Q.shape[1]), \
        f"window_sizes batch/head mismatch"
    assert landmark_idx.shape[:2] == (Q.shape[0], Q.shape[1]), \
        f"landmark_idx batch/head mismatch"

    return DSALTAttentionFunction.apply(Q, K, V, window_sizes, landmark_idx)