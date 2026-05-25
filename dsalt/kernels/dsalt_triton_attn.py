import torch
import triton
import triton.language as tl
import math
from collections import OrderedDict

_SEQ_BLOCK_MAP_CACHE: OrderedDict = OrderedDict()
_SEQ_BLOCK_MAP_CACHE_MAXSIZE = 256


@triton.jit
def _dsalt_fwd_kernel(
    Q, K, V, Out,
    W_sizes, Lmk_K, Lmk_V, Lmk_pos, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    stride_lph, stride_lpb, stride_lpk,
    scale:    tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_LMK:   tl.constexpr,
    MAX_WINS: tl.constexpr,
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
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0).to(tl.float32)

    w_sizes = tl.load(W_sizes + seq_start + m_start + offs_m, mask=valid_m, other=1).to(tl.int32)
    w_sizes = tl.maximum(w_sizes, 1)
    i_abs   = m_start + offs_m

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    window_end   = m_start + BLOCK_M
    window_start = tl.maximum(0, m_start - MAX_WINS + 1)
    n_start      = window_start - (window_start % BLOCK_N)
    num_iters    = (window_end - n_start + BLOCK_N - 1) // BLOCK_N

    for iter_idx in range(num_iters):
        cur_n   = n_start + iter_idx * BLOCK_N
        valid_n = ((offs_n + cur_n) < seq_len) & (cur_n < window_end) & (cur_n + BLOCK_N > window_start)

        k_blk = tl.load(
            K + (seq_start + cur_n + offs_n[None, :]) * stride_kt
              + pid_h * stride_kh
              + offs_d[:, None] * stride_kd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)
        v_blk = tl.load(
            V + (seq_start + cur_n + offs_n[None, :]) * stride_vt
              + pid_h * stride_vh
              + offs_d[:, None] * stride_vd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)

        qk    = tl.dot(q, k_blk) * scale
        j_abs = cur_n + offs_n
        in_win = (
            (j_abs[None, :] >= i_abs[:, None] - w_sizes[:, None] + 1) &
            (j_abs[None, :] <= i_abs[:, None])
        )
        final = in_win & valid_n[None, :] & valid_m[:, None]
        qk    = tl.where(final, qk, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
        p      = tl.where(final, tl.exp(qk - m_new[:, None]), 0.0)
        l_corr = tl.exp(m_i - m_new)
        l_i    = l_i * l_corr + tl.sum(p, axis=1)
        acc    = acc * l_corr[:, None] + tl.dot(p, v_blk)
        m_i    = m_new

    offs_lk = tl.arange(0, K_LMK)
    lk_k = tl.load(
        Lmk_K
        + pid_h * stride_lkh
        + seq_id * stride_lkb
        + offs_lk[None, :] * stride_lks
        + offs_d[:, None] * stride_lkd,
    ).to(tl.float32)
    lk_v = tl.load(
        Lmk_V
        + pid_h * stride_lvh
        + seq_id * stride_lvb
        + offs_lk[None, :] * stride_lvs
        + offs_d[:, None] * stride_lvd,
    ).to(tl.float32)

    lmk_pos    = tl.load(Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + offs_lk * stride_lpk)
    causal_lmk = lmk_pos[None, :] <= (m_start + offs_m[:, None])

    qk_lmk = tl.dot(q, lk_k) * scale
    qk_lmk = tl.where(causal_lmk & valid_m[:, None], qk_lmk, float("-inf"))

    m_new  = tl.maximum(m_i, tl.max(qk_lmk, axis=1))
    p_lmk  = tl.where(causal_lmk & valid_m[:, None], tl.exp(qk_lmk - m_new[:, None]), 0.0)
    l_corr = tl.exp(m_i - m_new)
    l_i    = l_i * l_corr + tl.sum(p_lmk, axis=1)
    acc    = acc * l_corr[:, None] + tl.dot(p_lmk, tl.trans(lk_v))

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


def _pick_block_m(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 32
    return 16


def _pick_block_n(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 32
    return 16


def _build_seq_block_map(
    cu_seqlens_cpu: torch.Tensor,
    block_m:        int,
    device:         torch.device,
) -> tuple[torch.Tensor, int]:
    key = (tuple(cu_seqlens_cpu.tolist()), block_m)
    if key in _SEQ_BLOCK_MAP_CACHE:
        cached_map, total_blks = _SEQ_BLOCK_MAP_CACHE[key]
        _SEQ_BLOCK_MAP_CACHE.move_to_end(key)
        return cached_map.to(device), total_blks

    lens       = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    blocks_per = (lens + block_m - 1) // block_m
    total_blks = int(blocks_per.sum())
    seq_col    = torch.repeat_interleave(torch.arange(lens.shape[0], dtype=torch.int32), blocks_per)
    blk_col    = (
        torch.arange(total_blks, dtype=torch.int32)
        - torch.repeat_interleave(blocks_per.cumsum(0) - blocks_per, blocks_per).int()
    )
    result = torch.stack([seq_col, blk_col], dim=1).contiguous()

    if len(_SEQ_BLOCK_MAP_CACHE) >= _SEQ_BLOCK_MAP_CACHE_MAXSIZE:
        _SEQ_BLOCK_MAP_CACHE.popitem(last=False)
    _SEQ_BLOCK_MAP_CACHE[key] = (result, total_blks)
    return result.to(device), total_blks


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
    num_seqs = cu_seqlens.shape[0] - 1
    n_heads  = alpha.shape[0]
    dh       = W_V.shape[0] // n_heads

    lens    = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts  = cu_seqlens[:-1].to(device)
    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off = torch.arange(total, device=device) - starts[seq_ids]
    max_len = int(lens.max())

    x_norm      = x.norm(dim=-1)
    mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
    z_x         = (x_norm - mu_x) / std_x

    xwv_h = (x @ W_V.T).view(total, n_heads, dh).norm(dim=-1)
    mu_v  = xwv_h.mean(0, keepdim=True)
    std_v = xwv_h.std(0, keepdim=True).clamp(min=1e-6)
    z_v   = (xwv_h - mu_v) / std_v

    scores = alpha * z_v + (1 - alpha) * z_x.unsqueeze(1)

    covered    = seq_off < w_sizes.long()
    scores_fil = scores.masked_fill(covered.unsqueeze(1).expand(-1, n_heads), float("-inf"))

    score_pad = torch.full((n_heads, num_seqs, max_len), float("-inf"), device=device)
    score_pad[:, seq_ids, seq_off] = scores_fil.T

    k_eff        = min(k_lmk, max_len)
    _, top_local = torch.topk(score_pad, k_eff, dim=2, sorted=False)
    return top_local


def _build_landmark_kv(
    K:           torch.Tensor,
    V:           torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
    k_lmk:       int,
    n_heads:     int,
    head_dim:    int,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts  = cu_seqlens[:-1].to(K.device)
    abs_idx = starts[None, :, None] + lmk_indices
    h_idx   = torch.arange(n_heads, device=K.device)[:, None, None].expand_as(abs_idx)
    lmk_K   = K[abs_idx, h_idx, :]
    lmk_V   = V[abs_idx, h_idx, :]
    return lmk_K.contiguous(), lmk_V.contiguous()


def _sparse_attn_backward(
    grad_out:   torch.Tensor,
    q:          torch.Tensor,
    k:          torch.Tensor,
    v:          torch.Tensor,
    lmk_K:     torch.Tensor,
    lmk_V:     torch.Tensor,
    lmk_pos:   torch.Tensor,
    w_sizes:   torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale:      float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_len, n_heads, head_dim = q.shape
    device   = q.device
    num_seqs = cu_seqlens.shape[0] - 1

    lens    = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    max_len = int(lens.max())

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    starts = cu_seqlens[:-1].to(device)
    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off = torch.arange(total_len, device=device) - starts[seq_ids]

    q_pad  = torch.zeros(num_seqs, max_len, n_heads, head_dim, device=device, dtype=q.dtype)
    k_pad  = torch.zeros_like(q_pad)
    v_pad  = torch.zeros_like(q_pad)
    go_pad = torch.zeros_like(q_pad)
    ws_pad = torch.zeros(num_seqs, max_len, device=device, dtype=torch.long)

    q_pad[seq_ids, seq_off]  = q
    k_pad[seq_ids, seq_off]  = k
    v_pad[seq_ids, seq_off]  = v
    go_pad[seq_ids, seq_off] = grad_out
    ws_pad[seq_ids, seq_off] = w_sizes.long()

    q_bh  = q_pad.permute(0, 2, 1, 3).reshape(num_seqs * n_heads, max_len, head_dim)
    k_bh  = k_pad.permute(0, 2, 1, 3).reshape(num_seqs * n_heads, max_len, head_dim)
    v_bh  = v_pad.permute(0, 2, 1, 3).reshape(num_seqs * n_heads, max_len, head_dim)
    go_bh = go_pad.permute(0, 2, 1, 3).reshape(num_seqs * n_heads, max_len, head_dim)

    i_idx = torch.arange(max_len, device=device)
    j_idx = torch.arange(max_len, device=device)

    ws_bh = ws_pad.unsqueeze(1).expand(-1, n_heads, -1).reshape(num_seqs * n_heads, max_len)

    win_mask = (
        (j_idx[None, None, :] >= (i_idx[None, :, None] - ws_bh[:, :, None] + 1)) &
        (j_idx[None, None, :] <= i_idx[None, :, None])
    )

    len_mask = (
        (i_idx[None, :] < lens.to(device).long().unsqueeze(1)).unsqueeze(1).expand(-1, n_heads, -1)
        .reshape(num_seqs * n_heads, max_len)
    )
    col_mask = (
        (j_idx[None, :] < lens.to(device).long().unsqueeze(1)).unsqueeze(1).expand(-1, n_heads, -1)
        .reshape(num_seqs * n_heads, max_len)
    )
    win_mask = win_mask & len_mask.unsqueeze(2) & col_mask.unsqueeze(1)

    lmk_K_bh = lmk_K.permute(1, 0, 2, 3).reshape(num_seqs * n_heads, lmk_K.shape[2], head_dim)
    lmk_V_bh = lmk_V.permute(1, 0, 2, 3).reshape(num_seqs * n_heads, lmk_V.shape[2], head_dim)
    lmk_pos_bh = lmk_pos.permute(1, 0, 2).reshape(num_seqs * n_heads, lmk_pos.shape[2])

    causal_lmk = lmk_pos_bh[:, None, :] <= i_idx[None, :, None]
    causal_lmk = causal_lmk & len_mask.unsqueeze(2)

    qk_win = torch.bmm(q_bh, k_bh.transpose(1, 2)) * scale
    qk_win = qk_win.masked_fill(~win_mask, float("-inf"))

    qk_lmk = torch.bmm(q_bh, lmk_K_bh.transpose(1, 2)) * scale
    qk_lmk = qk_lmk.masked_fill(~causal_lmk, float("-inf"))

    qk_all = torch.cat([qk_win, qk_lmk], dim=2)
    p_all  = torch.softmax(qk_all, dim=2)
    p_win  = p_all[:, :, :max_len]
    p_lmk  = p_all[:, :, max_len:]

    dv_bh   = p_win.transpose(1, 2).bmm(go_bh)
    dp_win  = go_bh.bmm(v_bh.transpose(1, 2))
    dp_lmk  = go_bh.bmm(lmk_V_bh.transpose(1, 2))
    dp_all  = torch.cat([dp_win, dp_lmk], dim=2)
    ds_all  = p_all * (dp_all - (dp_all * p_all).sum(dim=2, keepdim=True))
    ds_win  = ds_all[:, :, :max_len] * scale
    ds_lmk  = ds_all[:, :, max_len:] * scale

    dq_bh  = torch.bmm(ds_win, k_bh) + torch.bmm(ds_lmk, lmk_K_bh)
    dk_bh  = ds_win.transpose(1, 2).bmm(q_bh)

    dq_pad = dq_bh.reshape(num_seqs, n_heads, max_len, head_dim).permute(0, 2, 1, 3)
    dk_pad = dk_bh.reshape(num_seqs, n_heads, max_len, head_dim).permute(0, 2, 1, 3)
    dv_pad = dv_bh.reshape(num_seqs, n_heads, max_len, head_dim).permute(0, 2, 1, 3)

    dq.index_add_(0, torch.where(seq_ids >= 0)[0],
                  dq_pad[seq_ids, seq_off] - dq_pad[seq_ids, seq_off] + dq_pad[seq_ids, seq_off])

    dq[...] = dq_pad[seq_ids, seq_off]
    dk[...] = dk_pad[seq_ids, seq_off]
    dv[...] = dv_pad[seq_ids, seq_off]

    abs_lmk = (starts[None, :].expand(n_heads, -1).reshape(-1)[:, None] + lmk_pos_bh.long()).clamp(0, total_len - 1)
    dv_lmk  = p_lmk.transpose(1, 2).bmm(go_bh)
    idx_exp = abs_lmk.unsqueeze(-1).expand(-1, -1, head_dim)
    dv_lmk_flat = dv_lmk.reshape(num_seqs * n_heads, -1, head_dim)

    dv_full = dv.reshape(total_len, n_heads, head_dim)
    for bh in range(num_seqs * n_heads):
        b = bh // n_heads
        h = bh % n_heads
        s = int(cu_seqlens[b])
        e = int(cu_seqlens[b + 1])
        dv_full[s:e, h, :] += dv_bh[bh, :e - s, :]
        dv_full[abs_lmk[bh], h, :].scatter_add_(
            0,
            torch.zeros(abs_lmk.shape[1], dtype=torch.long, device=device).unsqueeze(1).expand(-1, head_dim),
            dv_lmk_flat[bh],
        )

    return dq, dk, dv_full.view(total_len, n_heads, head_dim)


class DSALTAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v, x, W_V, alpha, w_sizes, cu_seqlens, k_lmk, n_min, n_max,
    ):
        total_len, n_heads, head_dim = q.shape
        device    = q.device
        scale     = 1.0 / math.sqrt(head_dim)
        HEAD_DIM  = triton.next_power_of_2(head_dim)
        BLOCK_M   = _pick_block_m(head_dim)
        BLOCK_N   = _pick_block_n(head_dim)
        MAX_WINS  = n_max
        num_warps = 4 if head_dim <= 64 else 2

        q_c   = q.contiguous().to(torch.float16)
        k_c   = k.contiguous().to(torch.float16)
        v_c   = v.contiguous().to(torch.float16)
        out   = torch.zeros(total_len, n_heads, HEAD_DIM, device=device, dtype=torch.float16)
        w_int = w_sizes.clamp(min=1).long().contiguous()

        cu_seqlens_cpu = cu_seqlens.cpu()

        with torch.no_grad():
            lmk_indices   = _compute_landmark_indices(
                x.float(), W_V.float(), alpha.float(),
                w_sizes.float(), cu_seqlens, k_lmk, n_min, total_len,
            )
            lmk_K, lmk_V = _build_landmark_kv(
                k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim,
            )
            seq_block_map, total_blk = _build_seq_block_map(cu_seqlens_cpu, BLOCK_M, device)

        cu_int         = cu_seqlens.to(torch.int32).contiguous()
        lmk_pos_tensor = lmk_indices.to(torch.int32).contiguous()

        _dsalt_fwd_kernel[(total_blk, n_heads)](
            q_c, k_c, v_c, out,
            w_int, lmk_K, lmk_V, lmk_pos_tensor, cu_int, seq_block_map,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
            lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
            lmk_pos_tensor.stride(0), lmk_pos_tensor.stride(1), lmk_pos_tensor.stride(2),
            scale=scale,
            HEAD_DIM=HEAD_DIM,
            K_LMK=k_lmk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            MAX_WINS=MAX_WINS,
            num_warps=num_warps,
            num_stages=2,
        )

        result = out[:, :, :head_dim].float()

        ctx.save_for_backward(
            q.float(), k.float(), v.float(),
            lmk_K.float(), lmk_V.float(),
            lmk_pos_tensor, w_sizes, cu_seqlens,
        )
        ctx.scale = scale

        return result

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, lmk_K, lmk_V, lmk_pos, w_sizes, cu_seqlens = ctx.saved_tensors
        scale = ctx.scale

        dq, dk, dv = _sparse_attn_backward(
            grad_out.float(), q, k, v,
            lmk_K, lmk_V, lmk_pos,
            w_sizes, cu_seqlens, scale,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None


def dsalt_triton_attention(
    q:          torch.Tensor,
    k:          torch.Tensor,
    v:          torch.Tensor,
    x:          torch.Tensor,
    W_V:        torch.Tensor,
    alpha:      torch.Tensor,
    w_sizes:    torch.Tensor,
    cu_seqlens: torch.Tensor,
    k_lmk:      int,
    n_min:      int,
    n_max:      int,
) -> torch.Tensor:
    return DSALTAttentionFunction.apply(
        q, k, v, x, W_V, alpha, w_sizes, cu_seqlens, k_lmk, n_min, n_max,
    )