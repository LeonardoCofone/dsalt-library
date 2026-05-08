import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _gather_landmark_kv(
    K: torch.Tensor,
    V: torch.Tensor,
    landmark_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = K.shape
    K_lmk = landmark_idx.shape[-1]
    idx = landmark_idx.long().clamp(0, N - 1)
    idx_exp = idx.unsqueeze(-1).expand(B, H, K_lmk, D)
    K_gathered = K.gather(2, idx_exp).contiguous()
    V_gathered = V.gather(2, idx_exp).contiguous()
    return K_gathered, V_gathered


if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            # Small blocks: num_warps=8 per nascondere latenza landmark loading
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=3),
            # Medium blocks: balanced warps
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
            # Large blocks: 8 warps
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        ],
        key=["N", "D", "K"],
    )
    @triton.jit
    def _dsalt_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        landmark_idx_ptr,
        Out_ptr, LSE_ptr,
        Win_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_lmb, stride_lmh, stride_lmk,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_wb, stride_wh, stride_wn,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        q_start = pid_m * BLOCK_M
        offs_m  = q_start + tl.arange(0, BLOCK_M)
        mask_m  = offs_m < N
        offs_d  = tl.arange(0, BLOCK_D)
        mask_d  = offs_d < D

        Q_bh  = Q_ptr   + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr   + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr   + pid_b * stride_vb + pid_h * stride_vh
        LM_bh = landmark_idx_ptr + pid_b * stride_lmb + pid_h * stride_lmh
        O_bh  = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh

        q = tl.load(
            Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        )

        w_i   = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)
        max_w = tl.max(w_i, axis=0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        k_win_start = tl.maximum(0, q_start - max_w)
        k_win_end   = q_start + BLOCK_M
        k_blk = (k_win_start // BLOCK_N) * BLOCK_N

        while k_blk < k_win_end:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_tile = tl.load(
                K_bh + offs_n[None, :] * stride_kn + offs_d[:, None],
                mask=mask_n[None, :] & mask_d[:, None], other=0.0,
            )
            s = tl.dot(q, k_tile, out_dtype=tl.float32) * SCALE

            causal = offs_n[None, :] <= offs_m[:, None]
            window = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            comb   = causal & window & mask_n[None, :] & mask_m[:, None]
            s      = tl.where(comb, s, float("-inf"))

            m_new  = tl.maximum(m_i, tl.max(s, axis=1))
            alpha_ = tl.exp(m_i - m_new)
            p      = tl.exp(s - m_new[:, None])
            p      = tl.where(comb, p, 0.0)

            v_tile = tl.load(
                V_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            acc = acc * alpha_[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            l_i = l_i * alpha_ + tl.sum(p, axis=1)
            m_i = m_new
            k_blk += BLOCK_N

        lmk_blk = 0
        while lmk_blk < K:
            offs_k  = lmk_blk + tl.arange(0, BLOCK_N)
            valid_k = offs_k < K

            idx_k = tl.load(
                LM_bh + offs_k * stride_lmk,
                mask=valid_k,
                other=0,
            ).to(tl.int32)
            idx_k = tl.minimum(idx_k, N - 1)

            kl_tile = tl.load(
                K_bh + idx_k[:, None] * stride_kn + offs_d[None, :],
                mask=valid_k[:, None] & mask_d[None, :], other=0.0,
            )
            s_lmk = tl.dot(q, kl_tile, out_dtype=tl.float32) * SCALE
            lmk_ok = valid_k[None, :] & mask_m[:, None]
            s_lmk  = tl.where(lmk_ok, s_lmk, float("-inf"))

            m_new  = tl.maximum(m_i, tl.max(s_lmk, axis=1))
            alpha_ = tl.exp(m_i - m_new)
            p_lmk  = tl.exp(s_lmk - m_new[:, None])
            p_lmk  = tl.where(lmk_ok, p_lmk, 0.0)

            vl_tile = tl.load(
                V_bh + idx_k[:, None] * stride_vn + offs_d[None, :],
                mask=valid_k[:, None] & mask_d[None, :], other=0.0,
            )
            acc = acc * alpha_[:, None] + tl.dot(p_lmk.to(vl_tile.dtype), vl_tile)
            l_i = l_i * alpha_ + tl.sum(p_lmk, axis=1)
            m_i = m_new
            lmk_blk += BLOCK_N

        l_safe = tl.where(l_i > 0, l_i, 1.0)
        out_f  = acc / l_safe[:, None]
        lse    = m_i + tl.log(l_safe)

        tl.store(
            LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
            lse, mask=mask_m,
        )
        tl.store(
            O_bh + offs_m[:, None] * stride_on + offs_d[None, :],
            out_f.to(Q_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_d[None, :],
        )

    @triton.autotune(
        configs=[
            # Backward dK/dV: iteriamo su N (query), per ogni K
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "D", "K"],
    )
    @triton.jit
    def _dsalt_bwd_kernel_dkdv(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr, LSE_ptr,
        DK_ptr, DV_ptr,
        landmark_idx_ptr,
        DKL_ptr, DVL_ptr,
        Win_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_lmb, stride_lmh, stride_lmk,
        stride_wb, stride_wh, stride_wn,
        stride_dklb, stride_dklh, stride_dklk, stride_dkld,
        stride_dvlb, stride_dvlh, stride_dvlk, stride_dvld,
        B: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
        D: tl.constexpr, K: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        k_start = pid_n * BLOCK_N
        offs_n  = k_start + tl.arange(0, BLOCK_N)
        mask_n  = offs_n < N
        offs_d  = tl.arange(0, BLOCK_D)
        mask_d  = offs_d < D

        Q_bh  = Q_ptr  + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr  + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr  + pid_b * stride_vb + pid_h * stride_vh
        DO_bh = DO_ptr + pid_b * stride_qb + pid_h * stride_qh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh
        DK_bh = DK_ptr + pid_b * stride_kb + pid_h * stride_kh
        DV_bh = DV_ptr + pid_b * stride_vb + pid_h * stride_vh

        k_tile = tl.load(
            K_bh + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :], other=0.0,
        )
        v_tile = tl.load(
            V_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :], other=0.0,
        )

        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        q_blk = k_start
        while q_blk < N:
            offs_m = q_blk + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N
            w_i    = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)
            max_w  = tl.max(w_i, axis=0)

            if q_blk <= (k_start + BLOCK_N - 1 + max_w):
                q_t  = tl.load(
                    Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0,
                )
                do_t = tl.load(
                    DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0,
                )
                lse_t = tl.load(
                    LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
                    mask=mask_m, other=0.0,
                )

                s      = tl.dot(q_t, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                causal = offs_n[None, :] <= offs_m[:, None]
                window = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
                comb   = causal & window & mask_n[None, :] & mask_m[:, None]
                s      = tl.where(comb, s, float("-inf"))
                p      = tl.exp(s - lse_t[:, None])
                p      = tl.where(comb, p, 0.0)

                dv    += tl.dot(tl.trans(p).to(do_t.dtype), do_t)
                dp     = tl.dot(do_t, tl.trans(v_tile))
                rowsum = tl.sum(p * dp, axis=1)
                ds     = p * (dp - rowsum[:, None]) * SCALE
                ds     = tl.where(comb, ds, 0.0)
                dk    += tl.dot(tl.trans(ds).to(q_t.dtype), q_t)

            q_blk += BLOCK_M

        tl.store(
            DK_bh + offs_n[:, None] * stride_kn + offs_d[None, :],
            dk.to(K_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_d[None, :],
        )
        tl.store(
            DV_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
            dv.to(V_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_d[None, :],
        )

        offs_k = tl.arange(0, BLOCK_N)
        valid_k = offs_k < K
        LM_bh  = landmark_idx_ptr + pid_b * stride_lmb + pid_h * stride_lmh
        DKL_bh = DKL_ptr + pid_b * stride_dklb + pid_h * stride_dklh
        DVL_bh = DVL_ptr + pid_b * stride_dvlb + pid_h * stride_dvlh

        idx_k = tl.load(
            LM_bh + offs_k * stride_lmk,
            mask=valid_k,
            other=0,
        ).to(tl.int32)
        idx_k = tl.minimum(idx_k, N - 1)

        kl_tile = tl.load(
            K_bh + idx_k[:, None] * stride_kn + offs_d[None, :],
            mask=valid_k[:, None] & mask_d[None, :], other=0.0,
        )
        vl_tile = tl.load(
            V_bh + idx_k[:, None] * stride_vn + offs_d[None, :],
            mask=valid_k[:, None] & mask_d[None, :], other=0.0,
        )

        dkl = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dvl = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        q_blk = 0
        while q_blk < N:
            offs_m = q_blk + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N
            q_t  = tl.load(
                Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :], other=0.0,
            )
            do_t = tl.load(
                DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :], other=0.0,
            )
            lse_t = tl.load(
                LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
                mask=mask_m, other=0.0,
            )

            s_lmk  = tl.dot(q_t, tl.trans(kl_tile), out_dtype=tl.float32) * SCALE
            lmk_ok = valid_k[None, :] & mask_m[:, None]
            s_lmk  = tl.where(lmk_ok, s_lmk, float("-inf"))
            p_lmk  = tl.exp(s_lmk - lse_t[:, None])
            p_lmk  = tl.where(lmk_ok, p_lmk, 0.0)

            dvl   += tl.dot(tl.trans(p_lmk).to(do_t.dtype), do_t)
            dp_l   = tl.dot(do_t, tl.trans(vl_tile))
            rs_l   = tl.sum(p_lmk * dp_l, axis=1)
            ds_l   = p_lmk * (dp_l - rs_l[:, None]) * SCALE
            ds_l   = tl.where(lmk_ok, ds_l, 0.0)
            dkl   += tl.dot(tl.trans(ds_l).to(q_t.dtype), q_t)

            q_blk += BLOCK_M

        tl.store(
            DKL_bh + offs_k[:, None] * stride_dklk + offs_d[None, :],
            dkl.to(K_ptr.dtype.element_ty),
            mask=valid_k[:, None] & mask_d[None, :],
        )
        tl.store(
            DVL_bh + offs_k[:, None] * stride_dvlk + offs_d[None, :],
            dvl.to(V_ptr.dtype.element_ty),
            mask=valid_k[:, None] & mask_d[None, :],
        )

    @triton.autotune(
        configs=[
            # Backward dQ: iteriamo su M (query), per ogni K landmark
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "D", "K"],
    )
    @triton.jit
    def _dsalt_bwd_kernel_dq(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr, LSE_ptr, DQ_ptr,
        landmark_idx_ptr,
        Win_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_lmb, stride_lmh, stride_lmk,
        stride_wb, stride_wh, stride_wn,
        B: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
        D: tl.constexpr, K: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        q_start = pid_m * BLOCK_M
        offs_m  = q_start + tl.arange(0, BLOCK_M)
        mask_m  = offs_m < N
        offs_d  = tl.arange(0, BLOCK_D)
        mask_d  = offs_d < D

        Q_bh  = Q_ptr  + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr  + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr  + pid_b * stride_vb + pid_h * stride_vh
        DO_bh = DO_ptr + pid_b * stride_qb + pid_h * stride_qh
        DQ_bh = DQ_ptr + pid_b * stride_qb + pid_h * stride_qh
        LM_bh = landmark_idx_ptr + pid_b * stride_lmb + pid_h * stride_lmh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh

        q   = tl.load(Q_bh  + offs_m[:, None] * stride_qn + offs_d[None, :],
                      mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do  = tl.load(DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
                      mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        lse = tl.load(LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
                      mask=mask_m, other=0.0)
        w_i   = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)
        max_w = tl.max(w_i, axis=0)

        dq    = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        k_end = q_start + BLOCK_M
        k_blk = (tl.maximum(0, q_start - max_w) // BLOCK_N) * BLOCK_N

        while k_blk < k_end:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_tile = tl.load(K_bh + offs_n[:, None] * stride_kn + offs_d[None, :],
                             mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v_tile = tl.load(V_bh + offs_n[:, None] * stride_vn + offs_d[None, :],
                             mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            s      = tl.dot(q, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
            causal = offs_n[None, :] <= offs_m[:, None]
            window = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            comb   = causal & window & mask_n[None, :] & mask_m[:, None]
            s      = tl.where(comb, s, float("-inf"))
            p      = tl.exp(s - lse[:, None])
            p      = tl.where(comb, p, 0.0)

            dp     = tl.dot(do, tl.trans(v_tile))
            rowsum = tl.sum(p * dp, axis=1)
            ds     = p * (dp - rowsum[:, None]) * SCALE
            ds     = tl.where(comb, ds, 0.0)
            dq    += tl.dot(ds.to(k_tile.dtype), k_tile)
            k_blk += BLOCK_N

        lmk_blk = 0
        while lmk_blk < K:
            offs_k  = lmk_blk + tl.arange(0, BLOCK_N)
            valid_k = offs_k < K

            idx_k = tl.load(
                LM_bh + offs_k * stride_lmk,
                mask=valid_k,
                other=0,
            ).to(tl.int32)
            idx_k = tl.minimum(idx_k, N - 1)

            kl_tile = tl.load(
                K_bh + idx_k[:, None] * stride_kn + offs_d[None, :],
                mask=valid_k[:, None] & mask_d[None, :], other=0.0,
            )
            vl_tile = tl.load(
                V_bh + idx_k[:, None] * stride_vn + offs_d[None, :],
                mask=valid_k[:, None] & mask_d[None, :], other=0.0,
            )

            s_lmk  = tl.dot(q, tl.trans(kl_tile), out_dtype=tl.float32) * SCALE
            lmk_ok = valid_k[None, :] & mask_m[:, None]
            s_lmk  = tl.where(lmk_ok, s_lmk, float("-inf"))
            p_lmk  = tl.exp(s_lmk - lse[:, None])
            p_lmk  = tl.where(lmk_ok, p_lmk, 0.0)

            dp_l   = tl.dot(do, tl.trans(vl_tile))
            rs_l   = tl.sum(p_lmk * dp_l, axis=1)
            ds_l   = p_lmk * (dp_l - rs_l[:, None]) * SCALE
            ds_l   = tl.where(lmk_ok, ds_l, 0.0)
            dq    += tl.dot(ds_l.to(kl_tile.dtype), kl_tile)
            lmk_blk += BLOCK_N

        tl.store(
            DQ_bh + offs_m[:, None] * stride_qn + offs_d[None, :],
            dq.to(Q_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_d[None, :],
        )


class DSALTAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, window_sizes, landmark_idx):
        B, H, N, D = Q.shape
        K_lmk = landmark_idx.shape[-1]

        assert landmark_idx.ndim == 3
        assert Q.shape == K.shape == V.shape
        assert window_sizes.shape == (B, H, N)
        assert landmark_idx.shape == (B, H, K_lmk)
        assert D & (D - 1) == 0 and D >= 16

        BLOCK_D = triton.next_power_of_2(D) if _TRITON_AVAILABLE else D
        scale   = 1.0 / math.sqrt(D)

        Out = torch.empty_like(Q)
        LSE = torch.empty(B, H, N, dtype=torch.float32, device=Q.device)

        if Q.is_cuda and _TRITON_AVAILABLE:
            Q  = Q.contiguous()
            K  = K.contiguous()
            V  = V.contiguous()
            window_sizes = window_sizes.contiguous().to(torch.int32)
            landmark_idx = landmark_idx.contiguous().to(torch.int32)

            BLOCK_D = triton.next_power_of_2(D)
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), H, B)
            _dsalt_fwd_kernel[grid](
                Q, K, V,
                landmark_idx,
                Out, LSE,
                window_sizes,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                B=B, H=H, N=N, D=D, K=K_lmk, SCALE=scale, BLOCK_D=BLOCK_D,
            )
        else:
            Out, LSE = _cpu_reference_forward(Q, K, V, window_sizes, landmark_idx, scale)

        ctx.save_for_backward(Q, K, V, window_sizes, landmark_idx, LSE)
        ctx.BLOCK_D = BLOCK_D
        ctx.scale   = scale
        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, window_sizes, landmark_idx, LSE = ctx.saved_tensors
        B, H, N, D = Q.shape
        K_lmk = landmark_idx.shape[-1]

        dOut = dOut.contiguous()
        dQ   = torch.zeros_like(Q)
        dK   = torch.zeros_like(K)
        dV   = torch.zeros_like(V)
        dKL  = torch.zeros(B, H, K_lmk, D, dtype=K.dtype, device=K.device)
        dVL  = torch.zeros(B, H, K_lmk, D, dtype=V.dtype, device=V.device)

        if Q.is_cuda and _TRITON_AVAILABLE:
            BD = ctx.BLOCK_D

            grid_n = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), H, B)
            _dsalt_bwd_kernel_dkdv[grid_n](
                Q, K, V, dOut, LSE, dK, dV,
                landmark_idx, dKL, dVL,
                window_sizes,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                dKL.stride(0), dKL.stride(1), dKL.stride(2), dKL.stride(3),
                dVL.stride(0), dVL.stride(1), dVL.stride(2), dVL.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                B=B, H=H, N=N, D=D, K=K_lmk, SCALE=ctx.scale, BLOCK_D=BD,
            )

            grid_m = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), H, B)
            _dsalt_bwd_kernel_dq[grid_m](
                Q, K, V, dOut, LSE, dQ,
                landmark_idx,
                window_sizes,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                B=B, H=H, N=N, D=D, K=K_lmk, SCALE=ctx.scale, BLOCK_D=BD,
            )

            idx = landmark_idx.long().clamp(0, N - 1)
            idx_exp = idx.unsqueeze(-1).expand_as(dKL)
            dK.scatter_add_(2, idx_exp, dKL)
            dV.scatter_add_(2, idx_exp, dVL)
        else:
            dQ, dK, dV = _cpu_reference_backward(
                Q, K, V, dOut, window_sizes, landmark_idx, LSE, ctx.scale
            )

        return dQ, dK, dV, None, None


def _build_sparse_mask(N, window_sizes, landmark_idx):
    B, H, _ = window_sizes.shape
    K_lmk   = landmark_idx.shape[-1]
    device  = window_sizes.device

    i_idx = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, H, N, N)
    j_idx = torch.arange(N, device=device).view(1, 1, 1, N).expand(B, H, N, N)

    w      = window_sizes.unsqueeze(-1)
    causal = j_idx <= i_idx
    window = j_idx >= (i_idx - w)
    mask   = causal & window

    lmk_mask = torch.zeros(B, H, N, N, dtype=torch.bool, device=device)
    lmk_flat = landmark_idx.unsqueeze(2).expand(B, H, N, K_lmk).clamp(0, N - 1).long()
    lmk_mask.scatter_(-1, lmk_flat, True)
    lmk_mask = lmk_mask & causal

    return mask | lmk_mask


def _cpu_reference_forward(Q, K, V, window_sizes, landmark_idx, scale):
    N    = Q.shape[2]
    mask = _build_sparse_mask(N, window_sizes, landmark_idx)
    s    = torch.einsum("bhid,bhjd->bhij", Q.float(), K.float()) * scale
    s    = s.masked_fill(~mask, float("-inf"))
    a    = torch.softmax(s, dim=-1)
    a    = torch.nan_to_num(a, nan=0.0)
    out  = torch.einsum("bhij,bhjd->bhid", a, V.float())
    lse  = torch.logsumexp(s, dim=-1)
    lse  = torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)
    return out.to(Q.dtype), lse


def _cpu_reference_backward(Q, K, V, dOut, window_sizes, landmark_idx, LSE, scale):
    N    = Q.shape[2]
    mask = _build_sparse_mask(N, window_sizes, landmark_idx)
    s    = torch.einsum("bhid,bhjd->bhij", Q.float(), K.float()) * scale
    s    = s.masked_fill(~mask, float("-inf"))
    a    = torch.softmax(s, dim=-1)
    a    = torch.nan_to_num(a, nan=0.0)

    dV  = torch.einsum("bhij,bhid->bhjd", a, dOut.float())
    dp  = torch.einsum("bhid,bhjd->bhij", dOut.float(), V.float())
    ds  = a * (dp - (a * dp).sum(dim=-1, keepdim=True)) * scale
    ds  = ds.masked_fill(~mask, 0.0)
    dQ  = torch.einsum("bhij,bhjd->bhid", ds, K.float())
    dK  = torch.einsum("bhij,bhid->bhjd", ds, Q.float())
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


def dsalt_attention(Q, K, V, window_sizes, landmark_idx):
    return DSALTAttentionFunction.apply(Q, K, V, window_sizes, landmark_idx)