import math
import torch
import triton
import triton.language as tl

from .dsalt_triton_bwd import dsalt_triton_backward


_SEQ_BLOCK_MAP_CACHE: dict = {}

_SEQ_META_CACHE: dict = {}


def _seq_meta(cu_seqlens: torch.Tensor, total: int, device: torch.device):
    key = (int(cu_seqlens[-1]), cu_seqlens.shape[0] - 1)
    if key in _SEQ_META_CACHE:
        return _SEQ_META_CACHE[key]
    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total, device=device) - starts[seq_ids]
    max_len  = int(lens.max())
    val = (num_seqs, lens, starts, seq_ids, seq_off, max_len)
    _SEQ_META_CACHE[key] = val
    return val


@triton.jit
def _dsalt_fwd_kernel(
    Q, K, V, Out, LSE,
    W_sizes, Lmk_K, Lmk_V, Lmk_pos, Lmk_bias, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lset, stride_lseh,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    stride_lph, stride_lpb, stride_lpk,
    stride_lbh, stride_lbb, stride_lbk,
    scale:    tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_LMK:    tl.constexpr,
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
    valid_m = (m_start + offs_m) < seq_len

    q_ptrs = (
        Q
        + (seq_start + m_start + offs_m[:, None]) * stride_qt
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    w_sizes     = tl.load(W_sizes + seq_start + m_start + offs_m, mask=valid_m, other=1).to(tl.int32)
    w_sizes     = tl.maximum(w_sizes, 1)
    w_max_block = tl.max(w_sizes, axis=0)
    i_abs       = m_start + offs_m

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    window_start = tl.maximum(0, m_start - w_max_block + 1)
    window_end   = m_start + BLOCK_M
    n_start      = window_start - (window_start % BLOCK_N)

    for _ in range(0, (window_end - n_start + BLOCK_N - 1) // BLOCK_N):
        n_end_blk    = n_start + BLOCK_N
        blk_in_range = (n_end_blk > window_start) & (n_start < window_end) & (n_start < seq_len)
        valid_n      = ((offs_n + n_start) < seq_len) & blk_in_range

        k_blk = tl.load(
            K + (seq_start + n_start + offs_n[None, :]) * stride_kt
              + pid_h * stride_kh
              + offs_d[:, None] * stride_kd,
            mask=valid_n[None, :], other=0.0,
        )
        v_blk = tl.load(
            V + (seq_start + n_start + offs_n[None, :]) * stride_vt
              + pid_h * stride_vh
              + offs_d[:, None] * stride_vd,
            mask=valid_n[None, :], other=0.0,
        )

        qk    = tl.dot(q, k_blk) * scale
        j_abs = n_start + offs_n
        in_win = (
            (j_abs[None, :] >= i_abs[:, None] - w_sizes[:, None] + 1) &
            (j_abs[None, :] <= i_abs[:, None])
        )
        final  = in_win & valid_n[None, :] & valid_m[:, None]
        qk     = tl.where(final, qk, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
        p      = tl.where(final, tl.exp(qk - m_new[:, None]), 0.0)
        l_corr = tl.exp(m_i - m_new)
        l_i    = l_i * l_corr + tl.sum(p, axis=1)
        acc    = acc * l_corr[:, None] + tl.dot(p.to(k_blk.dtype), tl.trans(v_blk))
        m_i     = m_new
        n_start += BLOCK_N

    offs_lk = tl.arange(0, K_LMK)
    lk_k = tl.load(
        Lmk_K + pid_h * stride_lkh + seq_id * stride_lkb
              + offs_lk[None, :] * stride_lks + offs_d[:, None] * stride_lkd
    )
    lk_v = tl.load(
        Lmk_V + pid_h * stride_lvh + seq_id * stride_lvb
              + offs_lk[None, :] * stride_lvs + offs_d[:, None] * stride_lvd
    )
    lmk_pos = tl.load(
        Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + offs_lk * stride_lpk
    ).to(tl.int32)

    lmk_valid  = (lmk_pos >= 0) & (lmk_pos < seq_len)
    win_lo     = i_abs[:, None] - w_sizes[:, None] + 1
    causal_lmk = (lmk_pos[None, :] <= i_abs[:, None]) & (lmk_pos[None, :] < win_lo)
    valid_lmk  = causal_lmk & valid_m[:, None] & lmk_valid[None, :]

    lmk_bias = tl.load(
        Lmk_bias + pid_h * stride_lbh + seq_id * stride_lbb + offs_lk * stride_lbk
    ).to(tl.float32)
    qk_lmk = tl.dot(q, lk_k) * scale + lmk_bias[None, :]
    qk_lmk = tl.where(valid_lmk, qk_lmk, float("-inf"))

    m_new   = tl.maximum(m_i, tl.max(qk_lmk, axis=1))
    p_lmk   = tl.where(valid_lmk, tl.exp(qk_lmk - m_new[:, None]), 0.0)
    l_corr  = tl.exp(m_i - m_new)
    l_i     = l_i * l_corr + tl.sum(p_lmk, axis=1)
    acc     = acc * l_corr[:, None] + tl.dot(p_lmk.to(lk_v.dtype), tl.trans(lk_v))
    m_i = m_new

    out_val = tl.where(
        l_i[:, None] > 1e-9,
        acc / l_i[:, None],
        tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
    )
    tl.store(
        Out
        + (seq_start + m_start + offs_m[:, None]) * stride_ot
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od,
        out_val.to(Out.dtype.element_ty),
        mask=valid_m[:, None],
    )

    lse_val = tl.where(l_i > 1e-9, m_i + tl.log(l_i), float("-inf"))
    tl.store(
        LSE + (seq_start + m_start + offs_m) * stride_lset + pid_h * stride_lseh,
        lse_val,
        mask=valid_m,
    )


def _pick_block_m(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 32
    return 16


def _pick_block_n(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    elif head_dim <= 128:
        return 32
    elif head_dim <= 256:
        return 16
    else:
        return 8


def _build_seq_block_map(
    cu_seqlens: torch.Tensor,
    block_m:    int,
    device:     torch.device,
) -> tuple[torch.Tensor, int]:
    num_seqs  = cu_seqlens.shape[0] - 1
    total_len = int(cu_seqlens[-1])
    key = (total_len, num_seqs, block_m)
    if key in _SEQ_BLOCK_MAP_CACHE:
        return _SEQ_BLOCK_MAP_CACHE[key]

    cu_cpu     = cu_seqlens.detach().to("cpu")
    lens       = cu_cpu[1:] - cu_cpu[:-1]
    blocks_per = (lens + block_m - 1) // block_m
    total_blks = int(blocks_per.sum())
    seq_col    = torch.repeat_interleave(torch.arange(lens.shape[0], dtype=torch.int32), blocks_per)
    blk_col    = (
        torch.arange(total_blks, dtype=torch.int32)
        - torch.repeat_interleave(blocks_per.cumsum(0) - blocks_per, blocks_per).int()
    )
    result = torch.stack([seq_col, blk_col], dim=1).contiguous().to(device)
    _SEQ_BLOCK_MAP_CACHE[key] = (result, total_blks)
    return result, total_blks

def _score_block(x: torch.Tensor, W_V: torch.Tensor, alpha: torch.Tensor, n_heads: int, dh: int):
    x_norm = x.norm(dim=-1).float()
    z_x    = (x_norm - x_norm.mean()) / x_norm.std().clamp(min=1e-6)
    xwv_h  = (x @ W_V.T).view(x.shape[0], n_heads, dh).norm(dim=-1).float()
    mu_v   = xwv_h.mean(0, keepdim=True)
    std_v  = xwv_h.std(0, keepdim=True).clamp(min=1e-6)
    z_v    = (xwv_h - mu_v) / std_v
    scores = alpha * z_v + (1 - alpha) * z_x.unsqueeze(1)
    return scores, z_x, z_v


def _compute_landmark_indices(
    x:          torch.Tensor,
    W_V:        torch.Tensor,
    alpha:      torch.Tensor,
    w_sizes:    torch.Tensor,
    cu_seqlens: torch.Tensor,
    k_lmk:      int,
    n_min:      int,
    total_len:  int,
) -> torch.Tensor:
    device   = x.device
    total    = x.shape[0]
    num_seqs, lens, starts, seq_ids, seq_off, max_len = _seq_meta(cu_seqlens, total, device)

    n_heads = alpha.shape[0]
    dh      = W_V.shape[0] // n_heads

    scores, z_x, z_v = _score_block(x, W_V, alpha, n_heads, dh)

    covered    = seq_off < w_sizes.long()
    covered_h  = covered.unsqueeze(1).expand(-1, n_heads)
    scores_fil = scores.masked_fill(covered_h, float("-inf"))

    uniform = bool((lens == lens[0]).all())
    if uniform:
        L = int(lens[0])
        sp = scores_fil.T.view(n_heads, num_seqs, L)
        k_eff = min(k_lmk, L)
        top_val, top_lc = torch.topk(sp, k_eff, dim=2, sorted=False)
        out = torch.full((n_heads, num_seqs, k_lmk), -1, dtype=torch.long, device=device)
        valid = torch.isfinite(top_val)
        out[:, :, :k_eff] = torch.where(valid, top_lc, torch.full_like(top_lc, -1))
        return out, z_x, z_v

    score_pad = torch.full((n_heads, num_seqs, max_len), float("-inf"), device=device)
    score_pad[:, seq_ids, seq_off] = scores_fil.T
    k_eff           = min(k_lmk, max_len)
    top_val, top_lc = torch.topk(score_pad, k_eff, dim=2, sorted=False)
    out = torch.full((n_heads, num_seqs, k_lmk), -1, dtype=torch.long, device=device)
    valid = torch.isfinite(top_val)
    top_lc = torch.where(valid, top_lc, torch.full_like(top_lc, -1))
    out[:, :, :k_eff] = top_lc
    return out, z_x, z_v


def _build_landmark_kv(
    K:           torch.Tensor,
    V:           torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
    k_lmk:       int,
    n_heads:     int,
    head_dim:    int,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts   = cu_seqlens[:-1].to(K.device)
    safe_idx = lmk_indices.clamp(min=0)
    abs_idx  = starts[None, :, None] + safe_idx
    h_idx    = torch.arange(n_heads, device=K.device)[:, None, None]
    lmk_K    = K[abs_idx, h_idx, :]
    lmk_V    = V[abs_idx, h_idx, :]
    invalid  = (lmk_indices < 0).unsqueeze(-1)
    lmk_K    = lmk_K.masked_fill(invalid, 0.0)
    lmk_V    = lmk_V.masked_fill(invalid, 0.0)
    return lmk_K.contiguous(), lmk_V.contiguous()


class DSALTAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens):
        total_len, n_heads, head_dim = q.shape
        device     = q.device
        scale      = 1.0 / math.sqrt(head_dim)
        HEAD_DIM_C = triton.next_power_of_2(head_dim)
        BLOCK_M    = _pick_block_m(head_dim)
        BLOCK_N    = _pick_block_n(head_dim)
        num_warps  = 4 if head_dim <= 64 else 2
        k_lmk      = lmk_indices.shape[-1]

        in_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q_c   = q.contiguous().to(in_dtype)
        k_c   = k.contiguous().to(in_dtype)
        v_c   = v.contiguous().to(in_dtype)
        out   = torch.zeros_like(q_c)
        lse   = torch.empty((total_len, n_heads), device=device, dtype=torch.float32)
        w_int = w_sizes.clamp(min=1).long().contiguous()

        with torch.no_grad():
            lmk_K, lmk_V = _build_landmark_kv(
                k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim,
            )
            seq_block_map, total_blk = _build_seq_block_map(cu_seqlens, BLOCK_M, device)

        cu_int      = cu_seqlens.to(torch.int32).contiguous()
        lmk_pos     = lmk_indices.to(torch.int32).contiguous()
        lmk_bias_c  = lmk_bias.detach().to(torch.float32).contiguous()

        _dsalt_fwd_kernel[(total_blk, n_heads)](
            q_c, k_c, v_c, out, lse,
            w_int, lmk_K, lmk_V, lmk_pos, lmk_bias_c, cu_int, seq_block_map,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
            lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
            lmk_pos.stride(0), lmk_pos.stride(1), lmk_pos.stride(2),
            lmk_bias_c.stride(0), lmk_bias_c.stride(1), lmk_bias_c.stride(2),
            scale=scale,
            HEAD_DIM=HEAD_DIM_C,
            K_LMK=k_lmk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=2,
        )

        ctx.save_for_backward(q_c, k_c, v_c, out, lse, lmk_K, lmk_V, lmk_pos, lmk_bias_c)
        ctx.w_sizes       = w_sizes
        ctx.cu_seqlens    = cu_seqlens
        ctx.seq_block_map = seq_block_map
        ctx.total_blk     = total_blk
        ctx.scale         = scale
        ctx.BLOCK_M       = BLOCK_M
        ctx.BLOCK_N       = BLOCK_N
        ctx.HEAD_DIM_C    = HEAD_DIM_C
        ctx.out_dtype     = q.dtype

        return out.float()

    @staticmethod
    def backward(ctx, grad_out):
        q_c, k_c, v_c, out, lse, lmk_K, lmk_V, lmk_pos, lmk_bias_c = ctx.saved_tensors

        dq, dk, dv, d_bias = dsalt_triton_backward(
            grad_out, q_c, k_c, v_c, out, lse,
            lmk_K, lmk_V, lmk_pos, lmk_bias_c,
            ctx.w_sizes, ctx.cu_seqlens, ctx.scale,
            ctx.seq_block_map, ctx.total_blk,
            ctx.BLOCK_M, ctx.BLOCK_N, ctx.HEAD_DIM_C,
        )

        return (
            dq.to(ctx.out_dtype), dk.to(ctx.out_dtype), dv.to(ctx.out_dtype),
            None, d_bias.to(ctx.out_dtype), None, None,
        )

def dsalt_triton_attention(
    q:           torch.Tensor,
    k:           torch.Tensor,
    v:           torch.Tensor,
    lmk_indices: torch.Tensor,
    lmk_bias:    torch.Tensor,
    w_sizes:     torch.Tensor,
    cu_seqlens:  torch.Tensor,
) -> torch.Tensor:
    return DSALTAttentionFunction.apply(
        q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens,
    )