"""
dsalt/kernels/sparse_attn.py
----------------------------
Triton kernels for DSALT sparse attention.

Architecture:
  Each query token i attends to:
    - A causal local window  W(i) = [i-w(i), ..., i]    (size w, adaptive)
    - A fixed set of k landmark tokens L(i)              (indices pre-computed)

Forward kernel: dsalt_attn_fwd
  - Tiled over Q rows (BLOCK_M) and K/V cols (BLOCK_N)
  - Skips tiles entirely outside W(i) ∪ L(i)  → sparse FLOPs
  - Online softmax (safe, numerically stable)  → no O(n) buffer

Backward kernel: dsalt_attn_bwd
  - Recomputes attention weights from saved Q,K (Flash-style recompute)
  - Accumulates dQ, dK, dV with sparse tile mask

Design constraints:
  - BLOCK_M, BLOCK_N must be powers of 2 and ≥ 16
  - Landmark indices must be pre-sorted and deduplicated
  - Supports fp16 / bf16 / fp32 (autocast-aware)
  - CPU fallback via PyTorch (for testing without CUDA)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ── Triton import with graceful CPU fallback ──────────────────────────────────
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# Triton forward kernel
# ═════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    @triton.jit
    def _dsalt_fwd_kernel(
        # ── Pointers ──
        Q_ptr, K_ptr, V_ptr,
        Out_ptr,
        LSE_ptr,            # log-sum-exp per row  (for bwd recompute)
        Win_ptr,            # int32[B, H, N]  window size w(i) per token
        Lmk_ptr,            # int32[B, H, N, K]  landmark indices per token
        # ── Strides for Q/K/V/Out  (B, H, N, D) ──
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        # ── Strides for Win (B, H, N) ──
        stride_wb, stride_wh, stride_wn,
        # ── Strides for Lmk (B, H, N, K) ──
        stride_lb, stride_lh, stride_ln, stride_lk,
        # ── Scalar params ──
        B: tl.constexpr,   # batch size
        H: tl.constexpr,   # num heads
        N: tl.constexpr,   # sequence length
        D: tl.constexpr,   # head dimension
        K: tl.constexpr,   # num landmarks per token
        SCALE: tl.constexpr,       # 1/sqrt(D)
        BLOCK_M: tl.constexpr,     # tile size along Q dim
        BLOCK_N: tl.constexpr,     # tile size along K/V dim
        BLOCK_D: tl.constexpr,     # tile size along head-dim (== D, power-of-2)
    ):
        """
        Grid: (cdiv(N, BLOCK_M), H, B)
        Each program handles BLOCK_M consecutive query tokens for one (batch, head).
        """
        # ── Program IDs ──────────────────────────────────────────────────────
        pid_m = tl.program_id(0)   # which tile of Q rows
        pid_h = tl.program_id(1)   # head index
        pid_b = tl.program_id(2)   # batch index

        # ── Row range for this tile ──────────────────────────────────────────
        q_start = pid_m * BLOCK_M
        offs_m  = q_start + tl.arange(0, BLOCK_M)          # [BLOCK_M]
        mask_m  = offs_m < N

        # ── Base pointers for this (b, h) ────────────────────────────────────
        Q_bh = Q_ptr   + pid_b * stride_qb + pid_h * stride_qh
        K_bh = K_ptr   + pid_b * stride_kb + pid_h * stride_kh
        V_bh = V_ptr   + pid_b * stride_vb + pid_h * stride_vh
        O_bh = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        W_bh = Win_ptr + pid_b * stride_wb + pid_h * stride_wh
        L_bh = Lmk_ptr + pid_b * stride_lb + pid_h * stride_lh

        # ── Load Q tile  [BLOCK_M, BLOCK_D] ─────────────────────────────────
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(
            Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )   # [BLOCK_M, BLOCK_D]

        # ── Per-row running stats for online softmax ─────────────────────────
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # row max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # row sum
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)        # output acc

        # ── Load per-query window sizes ──────────────────────────────────────
        w_i = tl.load(
            W_bh + offs_m * stride_wn,
            mask=mask_m, other=0,
        )   # [BLOCK_M]  int32

        # ════════════════════════════════════════════════════════════════════
        # Pass 1: local window tiles
        #   Iterate over K-tiles that might overlap [i-w(i), i] for ANY i in
        #   this Q-tile.  We use a conservative range:
        #     k_start_min = max(0, q_start - max_w)
        #     k_end_max   = q_start + BLOCK_M   (causal)
        # ════════════════════════════════════════════════════════════════════
        max_w = tl.max(w_i, axis=0)          # scalar: widest window in tile
        k_win_start = tl.maximum(0, q_start - max_w)
        k_win_end   = q_start + BLOCK_M       # exclusive (causal)

        k_blk = k_win_start // BLOCK_N * BLOCK_N   # align to tile boundary
        while k_blk < k_win_end:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            # Load K tile  [BLOCK_D, BLOCK_N]
            k = tl.load(
                K_bh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=mask_n[None, :] & (offs_d[:, None] < D),
                other=0.0,
            )   # [BLOCK_D, BLOCK_N]

            # QK^T  [BLOCK_M, BLOCK_N]
            s = tl.dot(q, k) * SCALE  # [BLOCK_M, BLOCK_N]

            # Causal mask: j <= i
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            # Window mask: j >= i - w(i)
            window_mask = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            combined = causal_mask & window_mask & mask_n[None, :] & mask_m[:, None]
            s = tl.where(combined, s, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha  = tl.exp(m_i - m_new)
            p      = tl.exp(s - m_new[:, None])
            p      = tl.where(combined, p, 0.0)

            # Load V tile  [BLOCK_N, BLOCK_D]
            v = tl.load(
                V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None] & (offs_d[None, :] < D),
                other=0.0,
            )   # [BLOCK_N, BLOCK_D]

            acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i   = l_i * alpha + tl.sum(p, axis=1)
            m_i   = m_new

            k_blk += BLOCK_N

        # ════════════════════════════════════════════════════════════════════
        # Pass 2: landmark tiles
        #   K landmark indices are loaded per-query.  We process them by
        #   grouping into virtual tiles of BLOCK_N.  Indices outside the
        #   window (already handled above) are NOT re-added.
        # ════════════════════════════════════════════════════════════════════
        # We load all K landmark indices for the first query in the tile
        # as a representative set (shared landmark assumption).
        # For per-query landmarks, this loop would need per-row gating.
        # Current design: landmarks are shared within the BLOCK_M tile
        # (conservative union) — correct for the paper's global-k scheme.

        lmk_base = L_bh + (q_start) * stride_ln   # base for first q in tile
        # process K landmarks in chunks of BLOCK_N
        lmk_blk = 0
        while lmk_blk < K:
            offs_k = lmk_blk + tl.arange(0, BLOCK_N)
            valid_k = offs_k < K

            # Load landmark indices  [BLOCK_N]
            lmk_idx = tl.load(
                lmk_base + offs_k * stride_lk,
                mask=valid_k, other=0,
            )   # int32 [BLOCK_N]

            # Skip if already in window (conservative: check against minimum
            # window start across the tile — avoids double-counting)
            min_win_start = tl.minimum(0, q_start - max_w)
            already_in_window = (lmk_idx >= min_win_start) & (lmk_idx < k_win_end)
            lmk_mask = valid_k & ~already_in_window & mask_m[:, None]  # broadcast issue — handled below

            # Load K values at landmark positions  [BLOCK_D, BLOCK_N]
            k_lmk = tl.load(
                K_bh + lmk_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=valid_k[None, :] & (offs_d[:, None] < D),
                other=0.0,
            )

            s_lmk = tl.dot(q, k_lmk) * SCALE   # [BLOCK_M, BLOCK_N]

            # Causal mask for landmarks (j <= i)
            lmk_causal = lmk_idx[None, :] <= offs_m[:, None]
            lmk_valid  = valid_k[None, :] & lmk_causal & (~already_in_window[None, :]) & mask_m[:, None]
            s_lmk = tl.where(lmk_valid, s_lmk, float("-inf"))

            # Online softmax update
            m_new  = tl.maximum(m_i, tl.max(s_lmk, axis=1))
            alpha  = tl.exp(m_i - m_new)
            p_lmk  = tl.exp(s_lmk - m_new[:, None])
            p_lmk  = tl.where(lmk_valid, p_lmk, 0.0)

            # Load V at landmark positions  [BLOCK_N, BLOCK_D]
            v_lmk = tl.load(
                V_bh + lmk_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=valid_k[:, None] & (offs_d[None, :] < D),
                other=0.0,
            )

            acc   = acc * alpha[:, None] + tl.dot(p_lmk.to(v_lmk.dtype), v_lmk)
            l_i   = l_i * alpha + tl.sum(p_lmk, axis=1)
            m_i   = m_new

            lmk_blk += BLOCK_N

        # ── Normalise accumulator ────────────────────────────────────────────
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        out    = acc / l_safe[:, None]

        # ── Store log-sum-exp (needed for backward) ──────────────────────────
        lse = m_i + tl.log(l_safe)
        tl.store(
            LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
            lse, mask=mask_m,
        )

        # ── Store output ─────────────────────────────────────────────────────
        tl.store(
            O_bh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
            out.to(Q_ptr.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D),
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Triton backward kernel  (dV, dK, dQ)
    # ═════════════════════════════════════════════════════════════════════════

    @triton.jit
    def _dsalt_bwd_kernel_dkdv(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr,             # gradient of output  [B, H, N, D]
        LSE_ptr,            # saved log-sum-exp   [B, H, N]
        DK_ptr, DV_ptr,
        Win_ptr, Lmk_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_wb, stride_wh, stride_wn,
        stride_lb, stride_lh, stride_ln, stride_lk,
        B: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
        D: tl.constexpr, K: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """
        Grid: (cdiv(N, BLOCK_N), H, B)
        Each program computes dK, dV for BLOCK_N key tokens.
        Iterates over all Q rows that could attend to these K tokens.
        """
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
        L_bh  = Lmk_ptr + pid_b * stride_lb + pid_h * stride_lh
        DK_bh = DK_ptr + pid_b * stride_kb + pid_h * stride_kh
        DV_bh = DV_ptr + pid_b * stride_vb + pid_h * stride_vh

        # Load K, V tiles
        k = tl.load(
            K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0,
        )   # [BLOCK_N, BLOCK_D]
        v = tl.load(
            V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0,
        )

        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        # Iterate over Q tiles that could attend to this K tile
        q_blk = k_start  # causal: only q >= k can attend to k
        while q_blk < N:
            offs_m = q_blk + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N

            w_i = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)

            # Check if any q in tile could attend to any k in tile
            max_w = tl.max(w_i, axis=0)
            # q attends to k if k >= q - w(q) and k <= q  (causal)
            # i.e., q >= k  (already guaranteed) and q - w(q) <= k
            # => q <= k + w(q)  =>  at least q_start <= k_end + max_w
            if q_blk <= k_start + BLOCK_N - 1 + max_w:
                q_tile = tl.load(
                    Q_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0,
                )   # [BLOCK_M, BLOCK_D]
                do_tile = tl.load(
                    DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0,
                )
                lse_tile = tl.load(
                    LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
                    mask=mask_m, other=0.0,
                )

                # Recompute attention weights
                s = tl.dot(q_tile, tl.trans(k)) * SCALE  # [BLOCK_M, BLOCK_N]
                causal_mask = offs_n[None, :] <= offs_m[:, None]
                window_mask = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
                combined = causal_mask & window_mask & mask_n[None, :] & mask_m[:, None]
                s = tl.where(combined, s, float("-inf"))
                p = tl.exp(s - lse_tile[:, None])   # [BLOCK_M, BLOCK_N]
                p = tl.where(combined, p, 0.0)

                # dV += p^T @ do
                dv += tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)

                # dp = do @ V^T
                dp = tl.dot(do_tile, tl.trans(v))   # [BLOCK_M, BLOCK_N]
                # ds = p * (dp - rowsum(p*dp))
                rowsum = tl.sum(p * dp, axis=1)
                ds = p * (dp - rowsum[:, None]) * SCALE
                ds = tl.where(combined, ds, 0.0)

                # dK += ds^T @ q
                dk += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile)

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

    @triton.jit
    def _dsalt_bwd_kernel_dq(
        Q_ptr, K_ptr, V_ptr,
        DO_ptr, LSE_ptr, DQ_ptr,
        Win_ptr, Lmk_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_wb, stride_wh, stride_wn,
        stride_lb, stride_lh, stride_ln, stride_lk,
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

        q   = tl.load(Q_bh  + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        do  = tl.load(DO_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        lse = tl.load(LSE_ptr + pid_b * H * N + pid_h * N + offs_m,
                    mask=mask_m, other=0.0)
        w_i = tl.load(W_bh + offs_m * stride_wn, mask=mask_m, other=0)

        dq    = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        max_w = tl.max(w_i, axis=0)
        k_end = q_start + BLOCK_M
        k_blk = tl.maximum(0, q_start - max_w) // BLOCK_N * BLOCK_N

        # window tiles
        while k_blk < k_end:
            offs_n = k_blk + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_tile = tl.load(
                K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0,
            )
            v_tile = tl.load(
                V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0,
            )

            s = tl.dot(q, tl.trans(k_tile)) * SCALE
            causal  = offs_n[None, :] <= offs_m[:, None]
            window  = offs_n[None, :] >= (offs_m[:, None] - w_i[:, None])
            combined = causal & window & mask_n[None, :] & mask_m[:, None]
            s = tl.where(combined, s, float("-inf"))
            p = tl.exp(s - lse[:, None])
            p = tl.where(combined, p, 0.0)

            # dp = do @ V^T  [BLOCK_M, BLOCK_N]
            dp     = tl.dot(do, tl.trans(v_tile))
            rowsum = tl.sum(p * dp, axis=1)
            ds     = p * (dp - rowsum[:, None]) * SCALE
            ds     = tl.where(combined, ds, 0.0)

            dq += tl.dot(ds.to(k_tile.dtype), k_tile)
            k_blk += BLOCK_N

        # landmark tiles
        lmk_base = W_bh + q_start * stride_ln  # riuso W_bh come placeholder — usa L_bh
        # corretto:
        L_bh     = Lmk_ptr + pid_b * stride_lb + pid_h * stride_lh
        lmk_base = L_bh + q_start * stride_ln
        lmk_blk  = 0
        while lmk_blk < K:
            offs_k  = lmk_blk + tl.arange(0, BLOCK_N)
            valid_k = offs_k < K
            lmk_idx = tl.load(lmk_base + offs_k * stride_lk, mask=valid_k, other=0)

            min_win_start = tl.maximum(0, q_start - max_w)
            already_in_window = (lmk_idx >= min_win_start) & (lmk_idx < k_end)
            lmk_valid = valid_k & ~already_in_window

            k_lmk = tl.load(
                K_bh + lmk_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=lmk_valid[None, :] & (offs_d[:, None] < D), other=0.0,
            )
            v_lmk = tl.load(
                V_bh + lmk_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=lmk_valid[:, None] & (offs_d[None, :] < D), other=0.0,
            )

            s_lmk    = tl.dot(q, k_lmk) * SCALE
            lmk_causal = lmk_idx[None, :] <= offs_m[:, None]
            lmk_mask   = lmk_valid[None, :] & lmk_causal & (~already_in_window[None, :]) & mask_m[:, None]
            s_lmk = tl.where(lmk_mask, s_lmk, float("-inf"))
            p_lmk = tl.exp(s_lmk - lse[:, None])
            p_lmk = tl.where(lmk_mask, p_lmk, 0.0)

            dp_lmk  = tl.dot(do, tl.trans(v_lmk))
            rowsum_l = tl.sum(p_lmk * dp_lmk, axis=1)
            ds_lmk  = p_lmk * (dp_lmk - rowsum_l[:, None]) * SCALE
            ds_lmk  = tl.where(lmk_mask, ds_lmk, 0.0)

            dq += tl.dot(ds_lmk.to(k_lmk.dtype), k_lmk)
            lmk_blk += BLOCK_N

        tl.store(
            DQ_bh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            dq.to(Q_ptr.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D),
        )


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch autograd Function  (dispatches to Triton or CPU fallback)
# ═════════════════════════════════════════════════════════════════════════════

class DSALTAttentionFunction(torch.autograd.Function):
    """
    Autograd-compatible wrapper around the Triton DSALT kernels.

    forward(ctx, Q, K, V, window_sizes, landmark_idx)
      Q, K, V        : [B, H, N, D]  fp16/bf16/fp32
      window_sizes   : [B, H, N]     int32   per-token window size w(i)
      landmark_idx   : [B, H, K] or [B, H, N, K]  int32   landmark indices (pre-sorted)

    Returns:
      output         : [B, H, N, D]  same dtype as Q
    """

    @staticmethod
    def forward(ctx, Q, K, V, window_sizes, landmark_idx):
        B, H, N, D = Q.shape
        K_lmk = landmark_idx.shape[-1]

        # ── Normalize landmark tensor shapes ─────────────────────────────
        if landmark_idx.ndim == 3:
            landmark_idx = landmark_idx.unsqueeze(2).expand(B, H, N, K_lmk).contiguous()
        elif landmark_idx.ndim != 4:
            raise AssertionError("landmark_idx must be shape [B,H,K] or [B,H,N,K]")

        # ── Validate ──────────────────────────────────────────────────────
        assert Q.shape == K.shape == V.shape, "Q/K/V must have same shape"
        assert window_sizes.shape  == (B, H, N), f"window_sizes shape mismatch"
        assert landmark_idx.shape  == (B, H, N, K_lmk), "landmark_idx shape mismatch"
        assert D & (D - 1) == 0 and D >= 16, "Head dim D must be power-of-2 and ≥ 16"

        # ── Tile sizes (tuned for A100/H100; adjust for smaller GPUs) ────
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = triton.next_power_of_2(D) if _TRITON_AVAILABLE else D

        scale  = 1.0 / math.sqrt(D)
        Out    = torch.empty_like(Q)
        LSE    = torch.empty(B, H, N, dtype=torch.float32, device=Q.device)

        if Q.is_cuda and _TRITON_AVAILABLE:
            # Ensure contiguous layout
            Q  = Q.contiguous();  K  = K.contiguous();  V = V.contiguous()
            window_sizes  = window_sizes.contiguous().to(torch.int32)
            landmark_idx  = landmark_idx.contiguous().to(torch.int32)

            grid = (triton.cdiv(N, BLOCK_M), H, B)
            _dsalt_fwd_kernel[grid](
                Q, K, V, Out, LSE,
                window_sizes, landmark_idx,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                landmark_idx.stride(0), landmark_idx.stride(1),
                landmark_idx.stride(2), landmark_idx.stride(3),
                B=B, H=H, N=N, D=D, K=K_lmk,
                SCALE=scale,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            )
        else:
            # ── CPU reference fallback (exact, no tiling) ─────────────────
            Out, LSE = _cpu_reference_forward(Q, K, V, window_sizes, landmark_idx, scale)

        ctx.save_for_backward(Q, K, V, window_sizes, landmark_idx, LSE)
        ctx.scale    = scale
        ctx.BLOCK_M  = BLOCK_M
        ctx.BLOCK_N  = BLOCK_N
        ctx.BLOCK_D  = BLOCK_D
        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, window_sizes, landmark_idx, LSE = ctx.saved_tensors
        B, H, N, D = Q.shape
        K_lmk      = landmark_idx.shape[-1]

        dOut = dOut.contiguous()
        dQ   = torch.zeros_like(Q)
        dK   = torch.zeros_like(K)
        dV   = torch.zeros_like(V)

        if Q.is_cuda and _TRITON_AVAILABLE:
            BLOCK_M = ctx.BLOCK_M
            BLOCK_N = ctx.BLOCK_N
            BLOCK_D = ctx.BLOCK_D

            grid_n = (triton.cdiv(N, BLOCK_N), H, B)
            _dsalt_bwd_kernel_dkdv[grid_n](
                Q, K, V, dOut, LSE, dK, dV, dQ,
                window_sizes, landmark_idx,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                window_sizes.stride(0), window_sizes.stride(1), window_sizes.stride(2),
                landmark_idx.stride(0), landmark_idx.stride(1),
                landmark_idx.stride(2), landmark_idx.stride(3),
                B=B, H=H, N=N, D=D, K=K_lmk,
                SCALE=ctx.scale,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            )
        else:
            dQ, dK, dV = _cpu_reference_backward(
                Q, K, V, dOut, window_sizes, landmark_idx, LSE, ctx.scale
            )

        return dQ, dK, dV, None, None   # no grad for window_sizes, landmark_idx


# ═════════════════════════════════════════════════════════════════════════════
# CPU reference implementation  (exact, readable, used for testing)
# ═════════════════════════════════════════════════════════════════════════════

def _build_sparse_mask(
    N: int,
    window_sizes: torch.Tensor,   # [B, H, N]  int
    landmark_idx: torch.Tensor,   # [B, H, N, K]  int
) -> torch.Tensor:
    """
    Returns a boolean mask [B, H, N, N] where mask[b,h,i,j] = True iff
    token j is in the attention set of token i.
    """
    B, H, _ = window_sizes.shape
    K_lmk   = landmark_idx.shape[-1]
    device  = window_sizes.device

    i_idx = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, H, N, N)
    j_idx = torch.arange(N, device=device).view(1, 1, 1, N).expand(B, H, N, N)

    w = window_sizes.unsqueeze(-1)              # [B, H, N, 1]
    causal = j_idx <= i_idx
    window = j_idx >= (i_idx - w)
    mask   = causal & window                    # [B, H, N, N]

    # Add landmark positions
    # landmark_idx: [B, H, N, K] → scatter into [B, H, N, N]
    lmk_mask = torch.zeros(B, H, N, N, dtype=torch.bool, device=device)
    lmk_flat = landmark_idx.clamp(0, N - 1)    # safety clamp
    lmk_mask.scatter_(-1, lmk_flat.long(), True)
    # Landmarks are still causal (j <= i)
    lmk_mask = lmk_mask & causal

    return mask | lmk_mask


def _cpu_reference_forward(
    Q: torch.Tensor,               # [B, H, N, D]
    K: torch.Tensor,
    V: torch.Tensor,
    window_sizes: torch.Tensor,    # [B, H, N]
    landmark_idx: torch.Tensor,    # [B, H, N, K]
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask  = _build_sparse_mask(Q.shape[2], window_sizes, landmark_idx)
    score = torch.einsum("bhid,bhjd->bhij", Q, K) * scale  # [B,H,N,N]
    score = score.masked_fill(~mask, float("-inf"))
    attn  = torch.softmax(score.float(), dim=-1)            # [B,H,N,N]
    # tokens with all-inf rows → nan after softmax; replace with 0
    attn  = torch.nan_to_num(attn, nan=0.0)
    out   = torch.einsum("bhij,bhjd->bhid", attn, V.float())

    # log-sum-exp for backward
    lse = torch.logsumexp(score.float(), dim=-1)  # [B, H, N]
    lse = torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)
    return out.to(Q.dtype), lse


def _cpu_reference_backward(
    Q, K, V, dOut, window_sizes, landmark_idx, LSE, scale
):
    mask  = _build_sparse_mask(Q.shape[2], window_sizes, landmark_idx)
    score = torch.einsum("bhid,bhjd->bhij", Q, K) * scale
    score = score.masked_fill(~mask, float("-inf"))
    attn  = torch.softmax(score.float(), dim=-1)
    attn  = torch.nan_to_num(attn, nan=0.0)

    dV  = torch.einsum("bhij,bhid->bhjd", attn, dOut.float())
    dp  = torch.einsum("bhid,bhjd->bhij", dOut.float(), V.float())
    ds  = attn * (dp - (attn * dp).sum(dim=-1, keepdim=True)) * scale
    ds  = ds.masked_fill(~mask, 0.0)
    dQ  = torch.einsum("bhij,bhjd->bhid", ds, K.float())
    dK  = torch.einsum("bhij,bhid->bhjd", ds, Q.float())
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def dsalt_attention(
    Q:             torch.Tensor,       # [B, H, N, D]
    K:             torch.Tensor,
    V:             torch.Tensor,
    window_sizes:  torch.Tensor,       # [B, H, N]   int32
    landmark_idx:  torch.Tensor,       # [B, H, K] or [B, H, N, K] int32
) -> torch.Tensor:
    """
    DSALT sparse causal self-attention.

    Each token i attends only to:
      - tokens j in [i - window_sizes[i], i]   (local causal window)
      - tokens in landmark_idx[i]               (global landmarks)

    Returns output of shape [B, H, N, D].
    """
    return DSALTAttentionFunction.apply(Q, K, V, window_sizes, landmark_idx)