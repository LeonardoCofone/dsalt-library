"""
dsalt/kernels/sparse_attn.py
-----------------------------
Triton kernels per DSALT sparse attention.

Cambiamenti rispetto alla versione precedente:
  - landmark_idx accettato come [B, H, K] — NON più [B, H, N, K].
    Ogni query token dello stesso head usa gli stessi landmark globali.
    Questo è corretto semanticamente (landmark = top-k globali per head)
    e risparmia N× memoria.
  - Bug fix backward: _dsalt_bwd_kernel_dkdv aveva dQ_ptr in firma ma
    non era usato; rimosso. dQ è calcolato in un kernel separato.
  - Bug fix _dsalt_bwd_kernel_dq: rimosso il dead code con W_bh usato
    come placeholder per lmk_base.
  - BLOCK_M/BLOCK_N determinati una sola volta per tutto il modulo
    (non per ogni chiamata forward) tramite una funzione helper.
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tile sizes in base alla GPU
# ─────────────────────────────────────────────────────────────────────────────

def _get_block_sizes(D: int) -> Tuple[int, int]:
    """Ritorna (BLOCK_M, BLOCK_N) ottimali per la GPU corrente."""
    if not torch.cuda.is_available():
        return 32, 32
    mem_gb = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory / 1e9
    if mem_gb < 16:        # T4, RTX 3070/3080
        base = 16
    elif mem_gb < 32:      # RTX 4090, A6000
        base = 32
    else:                  # A100, H100
        base = 64
    # Riduci ulteriormente per head dim piccoli
    if D <= 64:
        base = min(base, 32)
    return base, base


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernel
# ─────────────────────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.jit
    def _dsalt_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        Out_ptr, LSE_ptr,
        Win_ptr,   # [B, H, N]  int32  window sizes
        Lmk_ptr,   # [B, H, K]  int32  landmark indices — NON [B,H,N,K]
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_wb, stride_wh, stride_wn,
        stride_lb, stride_lh, stride_lk,  # ← solo 3 strides (no stride_ln)
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,   # num landmarks
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Grid: (cdiv(N, BLOCK_M), H, B)
        landmark_idx ha shape [B, H, K] — stessi landmark per tutti i query dello stesso head.
        """
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        q_start = pid_m * BLOCK_M
        offs_m  = q_start + tl.arange(0, BLOCK_M)
        mask_m  = offs_m < N

        Q_bh = Q_ptr   + pid_b * stride_qb + pid_h * stride_qh
        K_bh = K_ptr   + pid_b * stride_kb + pid_h * stride_kh
        V_bh = V_ptr   + pid_b * stride_vb + pid_h * stride_vh
        O_bh = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        W_bh = Win_ptr + pid_b * stride_wb + pid_h * stride_wh
        # Landmark base: [B, H, K] → puntatore all'head corrente
        L_bh = Lmk_ptr + pid_b * stride_lb + pid_h * stride_lh

        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(
            Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0,
        )

        m_i  = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        w_i   = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)
        max_w = tl.max(w_i, axis=0)

        # ── Pass 1: finestra locale ───────────────────────────────────────
        k_win_start = tl.maximum(0, q_start - max_w)
        k_win_end   = q_start + BLOCK_M

        k_blk = (k_win_start // BLOCK_N) * BLOCK_N
        while k_blk < k_win_end:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_tile = tl.load(
                K_bh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=mask_n[None, :] & (offs_d[:, None] < D), other=0.0,
            )
            s = tl.dot(q, k_tile) * SCALE

            causal = offs_n[None, :] <= offs_m[:, None]
            window = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            comb   = causal & window & mask_n[None, :] & mask_m[:, None]
            s      = tl.where(comb, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha_ = tl.exp(m_i - m_new)
            p      = tl.exp(s - m_new[:, None])
            p      = tl.where(comb, p, 0.0)

            v_tile = tl.load(
                V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0,
            )
            acc = acc * alpha_[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            l_i = l_i * alpha_ + tl.sum(p, axis=1)
            m_i = m_new
            k_blk += BLOCK_N

        # ── Pass 2: landmark tokens ───────────────────────────────────────
        # L_bh punta a [K] indici int32 per questo (b, h)
        lmk_blk = 0
        while lmk_blk < K:
            offs_k  = lmk_blk + tl.arange(0, BLOCK_N)
            valid_k = offs_k < K

            # Carica indici landmark [BLOCK_N]
            lmk_idx = tl.load(
                L_bh + offs_k * stride_lk,
                mask=valid_k, other=0,
            )

            # Salta i landmark che cadono già nella finestra locale
            # (usa il range conservativo per l'intero tile)
            already = (lmk_idx >= k_win_start) & (lmk_idx < k_win_end)

            k_lmk_tile = tl.load(
                K_bh + lmk_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=(valid_k & ~already)[None, :] & (offs_d[:, None] < D),
                other=0.0,
            )
            s_lmk = tl.dot(q, k_lmk_tile) * SCALE

            lmk_causal = lmk_idx[None, :] <= offs_m[:, None]
            lmk_ok     = valid_k[None, :] & lmk_causal & (~already[None, :]) & mask_m[:, None]
            s_lmk      = tl.where(lmk_ok, s_lmk, float("-inf"))

            m_new  = tl.maximum(m_i, tl.max(s_lmk, axis=1))
            alpha_ = tl.exp(m_i - m_new)
            p_lmk  = tl.exp(s_lmk - m_new[:, None])
            p_lmk  = tl.where(lmk_ok, p_lmk, 0.0)

            v_lmk = tl.load(
                V_bh + lmk_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=(valid_k & ~already)[:, None] & (offs_d[None, :] < D),
                other=0.0,
            )
            acc = acc * alpha_[:, None] + tl.dot(p_lmk.to(v_lmk.dtype), v_lmk)
            l_i = l_i * alpha_ + tl.sum(p_lmk, axis=1)
            m_i = m_new
            lmk_blk += BLOCK_N

        # ── Normalizza e scrivi output ────────────────────────────────────
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        out_f  = acc / l_safe[:, None]
        lse    = m_i + tl.log(l_safe)

        tl.store(
            LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
            lse, mask=mask_m,
        )
        tl.store(
            O_bh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
            out_f.to(Q_ptr.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Backward: dK, dV
    # ─────────────────────────────────────────────────────────────────────

    @triton.jit
    def _dsalt_bwd_kernel_dkdv(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr, LSE_ptr,
        DK_ptr, DV_ptr,   # ← rimosso dQ_ptr dalla firma
        Win_ptr, Lmk_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_wb, stride_wh, stride_wn,
        stride_lb, stride_lh, stride_lk,
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

        Q_bh  = Q_ptr  + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr  + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr  + pid_b * stride_vb + pid_h * stride_vh
        DO_bh = DO_ptr + pid_b * stride_qb + pid_h * stride_qh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh
        DK_bh = DK_ptr + pid_b * stride_kb + pid_h * stride_kh
        DV_bh = DV_ptr + pid_b * stride_vb + pid_h * stride_vh

        k_tile = tl.load(
            K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0,
        )
        v_tile = tl.load(
            V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0,
        )

        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        # Itera sui Q tiles che possono attendere a questo K tile
        q_blk = k_start
        while q_blk < N:
            offs_m = q_blk + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N

            w_i   = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)
            max_w = tl.max(w_i, axis=0)

            # Condizione: k_start è raggiungibile da almeno un q in questo tile
            q_can_attend = q_blk <= (k_start + BLOCK_N - 1 + max_w)
            if q_can_attend:
                q_t  = tl.load(
                    Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0,
                )
                do_t = tl.load(
                    DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0,
                )
                lse_t = tl.load(
                    LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
                    mask=mask_m, other=0.0,
                )

                s = tl.dot(q_t, tl.trans(k_tile)) * SCALE
                causal = offs_n[None, :] <= offs_m[:, None]
                window = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
                comb   = causal & window & mask_n[None, :] & mask_m[:, None]
                s      = tl.where(comb, s, float("-inf"))
                p      = tl.exp(s - lse_t[:, None])
                p      = tl.where(comb, p, 0.0)

                dv += tl.dot(tl.trans(p).to(do_t.dtype), do_t)

                dp     = tl.dot(do_t, tl.trans(v_tile))
                rowsum = tl.sum(p * dp, axis=1)
                ds     = p * (dp - rowsum[:, None]) * SCALE
                ds     = tl.where(comb, ds, 0.0)
                dk    += tl.dot(tl.trans(ds).to(q_t.dtype), q_t)

            q_blk += BLOCK_M

        tl.store(
            DK_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            dk.to(K_ptr.dtype.element_ty),
            mask=mask_n[:, None] & (offs_d[None, :] < D),
        )
        tl.store(
            DV_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            dv.to(V_ptr.dtype.element_ty),
            mask=mask_n[:, None] & (offs_d[None, :] < D),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Backward: dQ
    # ─────────────────────────────────────────────────────────────────────

    @triton.jit
    def _dsalt_bwd_kernel_dq(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr, LSE_ptr, DQ_ptr,
        Win_ptr, Lmk_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_wb, stride_wh, stride_wn,
        stride_lb, stride_lh, stride_lk,
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

        Q_bh  = Q_ptr  + pid_b * stride_qb + pid_h * stride_qh
        K_bh  = K_ptr  + pid_b * stride_kb + pid_h * stride_kh
        V_bh  = V_ptr  + pid_b * stride_vb + pid_h * stride_vh
        DO_bh = DO_ptr + pid_b * stride_qb + pid_h * stride_qh
        DQ_bh = DQ_ptr + pid_b * stride_qb + pid_h * stride_qh
        W_bh  = Win_ptr + pid_b * stride_wb + pid_h * stride_wh
        L_bh  = Lmk_ptr + pid_b * stride_lb + pid_h * stride_lh

        q   = tl.load(Q_bh  + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                      mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        do  = tl.load(DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                      mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        lse = tl.load(LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
                      mask=mask_m, other=0.0)
        w_i   = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)
        max_w = tl.max(w_i, axis=0)

        dq      = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        k_end   = q_start + BLOCK_M
        k_blk   = (tl.maximum(0, q_start - max_w) // BLOCK_N) * BLOCK_N

        # Finestra locale
        while k_blk < k_end:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_tile = tl.load(K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                             mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)
            v_tile = tl.load(V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                             mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)

            s      = tl.dot(q, tl.trans(k_tile)) * SCALE
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

        # Landmark
        k_win_start = tl.maximum(0, q_start - max_w)
        lmk_blk = 0
        while lmk_blk < K:
            offs_k  = lmk_blk + tl.arange(0, BLOCK_N)
            valid_k = offs_k < K
            lmk_idx = tl.load(L_bh + offs_k * stride_lk, mask=valid_k, other=0)

            already  = (lmk_idx >= k_win_start) & (lmk_idx < k_end)
            lmk_ok_v = valid_k & ~already

            k_lmk = tl.load(
                K_bh + lmk_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=lmk_ok_v[:, None] & (offs_d[None, :] < D), other=0.0,
            )
            v_lmk = tl.load(
                V_bh + lmk_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=lmk_ok_v[:, None] & (offs_d[None, :] < D), other=0.0,
            )

            s_lmk    = tl.dot(q, tl.trans(k_lmk)) * SCALE
            lmk_causal = lmk_idx[None, :] <= offs_m[:, None]
            lmk_mask   = lmk_ok_v[:, None] & lmk_causal & mask_m[:, None]
            s_lmk      = tl.where(lmk_mask, s_lmk, float("-inf"))
            p_lmk      = tl.exp(s_lmk - lse[:, None])
            p_lmk      = tl.where(lmk_mask, p_lmk, 0.0)

            dp_l    = tl.dot(do, tl.trans(v_lmk))
            rowsum_l = tl.sum(p_lmk * dp_l, axis=1)
            ds_l    = p_lmk * (dp_l - rowsum_l[:, None]) * SCALE
            ds_l    = tl.where(lmk_mask, ds_l, 0.0)
            dq     += tl.dot(ds_l.to(k_lmk.dtype), k_lmk)
            lmk_blk += BLOCK_N

        tl.store(
            DQ_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            dq.to(Q_ptr.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Function
# ─────────────────────────────────────────────────────────────────────────────

class DSALTAttentionFunction(torch.autograd.Function):
    """
    Forward/backward DSALT.

    Signature:
      Q, K, V      : [B, H, N, D]  fp16/bf16/fp32
      window_sizes : [B, H, N]     int32
      landmark_idx : [B, H, K]     int32   ← NON più [B,H,N,K]
    """

    @staticmethod
    def forward(ctx, Q, K, V, window_sizes, landmark_idx):
        B, H, N, D = Q.shape
        K_lmk = landmark_idx.shape[-1]

        assert landmark_idx.ndim == 3, (
            f"landmark_idx deve essere [B,H,K], ricevuto shape {landmark_idx.shape}"
        )
        assert Q.shape == K.shape == V.shape
        assert window_sizes.shape == (B, H, N)
        assert landmark_idx.shape == (B, H, K_lmk)
        assert D & (D - 1) == 0 and D >= 16

        BLOCK_M, BLOCK_N = _get_block_sizes(D)
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

            grid = (triton.cdiv(N, BLOCK_M), H, B)
            _dsalt_fwd_kernel[grid](
                Q, K, V, Out, LSE,
                window_sizes, landmark_idx,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                B, H, N, D, K_lmk, scale, BLOCK_M, BLOCK_N, BLOCK_D,
            )
        else:
            Out, LSE = _cpu_reference_forward(Q, K, V, window_sizes, landmark_idx, scale)

        ctx.save_for_backward(Q, K, V, window_sizes, landmark_idx, LSE)
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
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

        if Q.is_cuda and _TRITON_AVAILABLE:
            BM = ctx.BLOCK_M
            BN = ctx.BLOCK_N
            BD = ctx.BLOCK_D

            # dK, dV
            grid_n = (triton.cdiv(N, BN), H, B)
            _dsalt_bwd_kernel_dkdv[grid_n](
                Q, K, V, dOut, LSE, dK, dV,
                window_sizes, landmark_idx,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                B, H, N, D, K_lmk, ctx.scale, BM, BN, BD,
            )

            # dQ
            grid_m = (triton.cdiv(N, BM), H, B)
            _dsalt_bwd_kernel_dq[grid_m](
                Q, K, V, dOut, LSE, dQ,
                window_sizes, landmark_idx,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                landmark_idx.stride(0), landmark_idx.stride(1), landmark_idx.stride(2),
                B, H, N, D, K_lmk, ctx.scale, BM, BN, BD,
            )
        else:
            dQ, dK, dV = _cpu_reference_backward(
                Q, K, V, dOut, window_sizes, landmark_idx, LSE, ctx.scale
            )

        return dQ, dK, dV, None, None


# ─────────────────────────────────────────────────────────────────────────────
# CPU reference  (esatta, usata per test)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sparse_mask(
    N: int,
    window_sizes: torch.Tensor,   # [B, H, N]
    landmark_idx: torch.Tensor,   # [B, H, K]  ← NON [B,H,N,K]
) -> torch.Tensor:
    B, H, _ = window_sizes.shape
    K_lmk   = landmark_idx.shape[-1]
    device  = window_sizes.device

    i_idx = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, H, N, N)
    j_idx = torch.arange(N, device=device).view(1, 1, 1, N).expand(B, H, N, N)

    w      = window_sizes.unsqueeze(-1)
    causal = j_idx <= i_idx
    window = j_idx >= (i_idx - w)
    mask   = causal & window

    # Landmark: [B, H, K] → espandiamo solo per la maschera (non per l'attenzione)
    # mask_lmk[b, h, i, lmk_j] = True per ogni i
    lmk_mask = torch.zeros(B, H, N, N, dtype=torch.bool, device=device)
    # landmark_idx: [B, H, K] → unsqueeze a [B, H, 1, K] e scatter su dim 3
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


# ─────────────────────────────────────────────────────────────────────────────
# API pubblica
# ─────────────────────────────────────────────────────────────────────────────

def dsalt_attention(
    Q: torch.Tensor,              # [B, H, N, D]
    K: torch.Tensor,
    V: torch.Tensor,
    window_sizes: torch.Tensor,   # [B, H, N]  int32
    landmark_idx: torch.Tensor,   # [B, H, K]  int32  ← NON più [B,H,N,K]
) -> torch.Tensor:
    """
    DSALT sparse causal self-attention.

    landmark_idx deve avere shape [B, H, K] — gli stessi landmark
    vengono usati per tutti i query token dello stesso head.
    """
    return DSALTAttentionFunction.apply(Q, K, V, window_sizes, landmark_idx)