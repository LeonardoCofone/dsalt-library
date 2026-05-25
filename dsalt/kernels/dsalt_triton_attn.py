import math
import torch
import triton
import triton.language as tl


_SEQ_BLOCK_MAP_CACHE: dict = {}


@triton.jit
def _dsalt_fwd_kernel(
    Q, K, V, Out, LSE,
    W_sizes, Lmk_K, Lmk_V, Lmk_pos, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lset, stride_lseh,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    stride_lph, stride_lpb, stride_lpk,
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
    valid_m = (m_start + offs_m) < seq_len

    q_ptrs = (
        Q
        + (seq_start + m_start + offs_m[:, None]) * stride_qt
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0).to(tl.float32)

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
        ).to(tl.float32)
        v_blk = tl.load(
            V + (seq_start + n_start + offs_n[None, :]) * stride_vt
              + pid_h * stride_vh
              + offs_d[:, None] * stride_vd,
            mask=valid_n[None, :], other=0.0,
        ).to(tl.float32)

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
        acc    = acc * l_corr[:, None] + tl.dot(
            p.to(tl.float16), tl.trans(v_blk).to(tl.float16)
        ).to(tl.float32)
        m_i     = m_new
        n_start += BLOCK_N

    offs_lk = tl.arange(0, K_LMK)
    lk_k = tl.load(
        Lmk_K
        + pid_h * stride_lkh
        + seq_id * stride_lkb
        + offs_lk[None, :] * stride_lks
        + offs_d[:, None] * stride_lkd
    ).to(tl.float32)
    lk_v = tl.load(
        Lmk_V
        + pid_h * stride_lvh
        + seq_id * stride_lvb
        + offs_lk[None, :] * stride_lvs
        + offs_d[:, None] * stride_lvd
    ).to(tl.float32)

    lmk_pos    = tl.load(Lmk_pos + pid_h * stride_lph + seq_id * stride_lpb + offs_lk * stride_lpk)
    causal_lmk = lmk_pos[None, :] <= (m_start + offs_m[:, None])

    qk_lmk = tl.dot(q, lk_k) * scale
    qk_lmk = tl.where(causal_lmk & valid_m[:, None], qk_lmk, float("-inf"))

    m_new   = tl.maximum(m_i, tl.max(qk_lmk, axis=1))
    p_lmk   = tl.where(causal_lmk & valid_m[:, None], tl.exp(qk_lmk - m_new[:, None]), 0.0)
    l_corr  = tl.exp(m_i - m_new)
    l_i     = l_i * l_corr + tl.sum(p_lmk, axis=1)
    acc     = acc * l_corr[:, None] + tl.dot(
        p_lmk.to(tl.float16), tl.trans(lk_v).to(tl.float16)
    ).to(tl.float32)
    m_i = m_new

    lse_val = tl.where(valid_m, m_i + tl.log(tl.where(l_i > 0.0, l_i, 1e-9)), float("-inf"))
    tl.store(
        LSE
        + (seq_start + m_start + offs_m) * stride_lset
        + pid_h * stride_lseh,
        lse_val,
        mask=valid_m,
    )

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
    if head_dim <= 96:
        return 32
    return 16


def _pick_block_n(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    elif head_dim <= 96:
        return 32
    elif head_dim <= 128:
        return 16
    else:
        return 8


def _build_seq_block_map(
    cu_seqlens_cpu: torch.Tensor,
    block_m:        int,
    device:         torch.device,
) -> tuple[torch.Tensor, int]:
    key = (tuple(cu_seqlens_cpu.tolist()), block_m)
    if key in _SEQ_BLOCK_MAP_CACHE:
        cached_map, total_blks = _SEQ_BLOCK_MAP_CACHE[key]
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
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total, device=device) - starts[seq_ids]
    max_len  = int(lens.max())

    n_heads = alpha.shape[0]
    dh      = W_V.shape[0] // n_heads

    x_norm      = x.norm(dim=-1)
    mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
    z_x         = (x_norm - mu_x) / std_x

    xwv_h    = (x @ W_V.T).view(total, n_heads, dh).norm(dim=-1)
    mu_v     = xwv_h.mean(0, keepdim=True)
    std_v    = xwv_h.std(0, keepdim=True).clamp(min=1e-6)
    z_v      = (xwv_h - mu_v) / std_v

    scores = alpha * z_v + (1 - alpha) * z_x.unsqueeze(1)

    covered    = seq_off < w_sizes.long()
    covered_h  = covered.unsqueeze(1).expand(-1, n_heads)
    scores_fil = scores.masked_fill(covered_h, float("-inf"))

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


def _sparse_attn_backward_lse(
    grad_out:    torch.Tensor,
    q:           torch.Tensor,
    k:           torch.Tensor,
    v:           torch.Tensor,
    out:         torch.Tensor,
    lse:         torch.Tensor,
    w_sizes:     torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
    scale:       float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_len, n_heads, head_dim = q.shape
    device   = q.device
    num_seqs = cu_seqlens.shape[0] - 1

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    starts  = cu_seqlens[:-1].to(device)
    abs_lmk = (starts[None, :, None] + lmk_indices.long()).clamp(max=total_len - 1)

    for s in range(num_seqs):
        s0   = int(cu_seqlens[s])
        s1   = int(cu_seqlens[s + 1])
        slen = s1 - s0
        if slen == 0:
            continue

        q_s   = q[s0:s1].transpose(0, 1).contiguous()
        k_s   = k[s0:s1].transpose(0, 1).contiguous()
        v_s   = v[s0:s1].transpose(0, 1).contiguous()
        go_s  = grad_out[s0:s1].transpose(0, 1).contiguous()
        o_s   = out[s0:s1].transpose(0, 1).contiguous()
        lse_s = lse[s0:s1].transpose(0, 1).contiguous()

        rows_s = torch.arange(slen, device=device)
        w_s    = w_sizes[s0:s1].clamp(min=1).long()
        lo_rel = (rows_s - w_s + 1).clamp(min=0)
        win_m  = (rows_s.unsqueeze(0) >= lo_rel.unsqueeze(1)) & (rows_s.unsqueeze(0) <= rows_s.unsqueeze(1))

        for h in range(n_heads):
            lp_abs = abs_lmk[h, s]
            lp_rel = (lp_abs - s0).clamp(min=0, max=slen - 1)
            lp_u   = lp_rel.unique()

            lk_h   = k_s[h, lp_u, :]
            lv_h   = v_s[h, lp_u, :]
            caus_m = lp_u.unsqueeze(0) <= rows_s.unsqueeze(1)

            qk_w   = (q_s[h] @ k_s[h].T) * scale
            qk_w   = qk_w.masked_fill(~win_m, float("-inf"))
            qk_l   = (q_s[h] @ lk_h.T) * scale
            qk_l   = qk_l.masked_fill(~caus_m, float("-inf"))

            qk_cat  = torch.cat([qk_w, qk_l], dim=1)
            lse_h   = lse_s[h].unsqueeze(1)
            p_all   = torch.exp(qk_cat - lse_h)

            p_w     = p_all[:, :slen] * win_m.float()
            p_l     = p_all[:, slen:] * caus_m.float()

            D_h     = (go_s[h] * o_s[h]).sum(dim=1, keepdim=True)

            dv_w    = p_w.T @ go_s[h]
            dv[s0:s1, h, :] += dv_w

            dv_l    = p_l.T @ go_s[h]
            dv[s0:s1, h, :].scatter_add_(0, lp_u.unsqueeze(1).expand_as(dv_l), dv_l)

            dp_w    = go_s[h] @ v_s[h].T
            dp_l    = go_s[h] @ lv_h.T

            ds_w    = p_w * (dp_w * win_m.float() - D_h) * scale
            ds_l    = p_l * (dp_l * caus_m.float() - D_h) * scale

            dq[s0:s1, h, :] += ds_w @ k_s[h] + ds_l @ lk_h
            dk[s0:s1, h, :] += ds_w.T @ q_s[h]
            dk_l = ds_l.T @ q_s[h]
            dk[s0:s1, h, :].scatter_add_(0, lp_u.unsqueeze(1).expand_as(dk_l), dk_l)

    return dq, dk, dv


class DSALTAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v, x, W_V, alpha, w_sizes, cu_seqlens, k_lmk, n_min,
    ):
        total_len, n_heads, head_dim = q.shape
        device     = q.device
        scale      = 1.0 / math.sqrt(head_dim)
        HEAD_DIM_C = triton.next_power_of_2(head_dim)
        BLOCK_M    = _pick_block_m(head_dim)
        BLOCK_N    = _pick_block_n(head_dim)
        num_warps  = 4 if head_dim <= 64 else 2

        q_c   = q.contiguous().to(torch.float16)
        k_c   = k.contiguous().to(torch.float16)
        v_c   = v.contiguous().to(torch.float16)
        out   = torch.zeros_like(q_c)
        lse   = torch.full((total_len, n_heads), float("-inf"), dtype=torch.float32, device=device)
        w_int = w_sizes.clamp(min=1).long().contiguous()

        cu_seqlens_cpu = cu_seqlens.cpu()

        with torch.no_grad():
            lmk_indices   = _compute_landmark_indices(
                x.float(), W_V.float(), alpha.detach().float(),
                w_sizes.float(), cu_seqlens, k_lmk, n_min, total_len,
            )
            lmk_K, lmk_V = _build_landmark_kv(
                k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim,
            )
            seq_block_map, total_blk = _build_seq_block_map(cu_seqlens_cpu, BLOCK_M, device)

        cu_int         = cu_seqlens.to(torch.int32).contiguous()
        lmk_pos_tensor = lmk_indices.to(torch.int32).contiguous()

        _dsalt_fwd_kernel[(total_blk, n_heads)](
            q_c, k_c, v_c, out, lse,
            w_int, lmk_K, lmk_V, lmk_pos_tensor, cu_int, seq_block_map,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
            lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
            lmk_pos_tensor.stride(0), lmk_pos_tensor.stride(1), lmk_pos_tensor.stride(2),
            scale=scale,
            HEAD_DIM=HEAD_DIM_C,
            K_LMK=k_lmk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=2,
        )

        out_f = out.float()

        ctx.save_for_backward(
            q.float(), k.float(), v.float(),
            out_f, lse,
            lmk_indices, w_int, cu_seqlens,
        )
        ctx.scale = scale

        return out_f

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, lmk_indices, w_sizes, cu_seqlens = ctx.saved_tensors

        dq, dk, dv = _sparse_attn_backward_lse(
            grad_out.float(), q, k, v, out, lse,
            w_sizes, lmk_indices, cu_seqlens, ctx.scale,
        )

        return dq, dk, dv, None, None, None, None, None, None, None


def dsalt_triton_attention(
    q:          torch.Tensor,
    k:          torch.Tensor,
    v:          torch.Tensor,
    x:          torch.Tensor,
    W_V:        torch.Tensor,
    alpha:      torch.Tensor,
    w_sizes:    torch.Tensor,
    cu_seqlens: torch.Tensor,
    k_lmk:     int,
    n_min:      int,
) -> torch.Tensor:
    return DSALTAttentionFunction.apply(
        q, k, v, x, W_V, alpha, w_sizes, cu_seqlens, k_lmk, n_min,
    )