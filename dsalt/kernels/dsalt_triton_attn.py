import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dsalt_fwd_kernel(
    Q, K, V, Out,
    W_sizes,
    Lmk_K, Lmk_V,
    Cu_seqlens,
    Seq_block_map,
    stride_qh, stride_qt, stride_qd,
    stride_kh, stride_kt, stride_kd,
    stride_vh, stride_vt, stride_vd,
    stride_oh, stride_ot, stride_od,
    stride_lkh, stride_lks, stride_lkd,
    stride_lvh, stride_lvs, stride_lvd,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_LMK: tl.constexpr,
):
    pid_bm = tl.program_id(0)
    pid_h  = tl.program_id(1)

    seq_id    = tl.load(Seq_block_map + pid_bm * 2)
    block_off = tl.load(Seq_block_map + pid_bm * 2 + 1)

    seq_start = tl.load(Cu_seqlens + seq_id)
    seq_end   = tl.load(Cu_seqlens + seq_id + 1)
    seq_len   = seq_end - seq_start

    m_start = block_off * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    valid_m = (m_start + offs_m) < seq_len

    q_ptrs = (
        Q
        + pid_h * stride_qh
        + (seq_start + m_start + offs_m[:, None]) * stride_qt
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0).to(tl.float32)

    w_ptrs  = W_sizes + seq_start + m_start + offs_m
    w_sizes = tl.load(w_ptrs, mask=valid_m, other=1)
    w_sizes = tl.maximum(w_sizes, 1)

    w_max_block = tl.max(w_sizes, axis=0)

    i_abs = m_start + offs_m

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    window_start = tl.maximum(0, m_start - w_max_block + 1)
    window_end   = m_start + BLOCK_M

    n_start = window_start - (window_start % BLOCK_N)

    while n_start < window_end:
        n_end_blk = n_start + BLOCK_N

        blk_max_j = n_end_blk - 1
        blk_min_j = n_start
        skip       = (blk_max_j < window_start) or (blk_min_j >= window_end)

        if not skip:
            valid_n = (offs_n + n_start) < seq_len

            k_ptrs = (
                K
                + pid_h * stride_kh
                + (seq_start + n_start + offs_n[None, :]) * stride_kt
                + offs_d[:, None] * stride_kd
            )
            v_ptrs = (
                V
                + pid_h * stride_vh
                + (seq_start + n_start + offs_n[None, :]) * stride_vt
                + offs_d[:, None] * stride_vd
            )

            k_blk = tl.load(k_ptrs, mask=valid_n[None, :], other=0.0).to(tl.float32)
            v_blk = tl.load(v_ptrs, mask=valid_n[None, :], other=0.0).to(tl.float32)

            qk = tl.dot(q, k_blk) * scale

            j_abs    = n_start + offs_n
            causal   = j_abs[None, :] <= i_abs[:, None]
            win_lo   = i_abs[:, None] - w_sizes[:, None] + 1
            in_win   = (j_abs[None, :] >= win_lo) & causal
            pad_mask = valid_n[None, :] & valid_m[:, None]
            final    = in_win & pad_mask

            qk = tl.where(final, qk, float("-inf"))

            m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
            p      = tl.exp(qk - m_new[:, None])
            p      = tl.where(final, p, 0.0)
            l_corr = tl.exp(m_i - m_new)
            l_i    = l_i * l_corr + tl.sum(p, axis=1)
            acc    = acc * l_corr[:, None] + tl.dot(p.to(k_blk.dtype), tl.trans(v_blk).to(k_blk.dtype)).to(tl.float32)
            m_i    = m_new

        n_start += BLOCK_N

    for lk in range(0, K_LMK):
        lk_k_base = pid_h * stride_lkh + seq_id * stride_lks + lk * HEAD_DIM
        lk_v_base = pid_h * stride_lvh + seq_id * stride_lvs + lk * HEAD_DIM

        lk_k = tl.load(Lmk_K + lk_k_base + offs_d).to(tl.float32)
        lk_v = tl.load(Lmk_V + lk_v_base + offs_d).to(tl.float32)

        qk_lk  = tl.sum(q * lk_k[None, :], axis=1) * scale
        qk_lk  = tl.where(valid_m, qk_lk, float("-inf"))

        m_new  = tl.maximum(m_i, qk_lk)
        p_lk   = tl.where(valid_m, tl.exp(qk_lk - m_new), 0.0)
        l_corr = tl.exp(m_i - m_new)
        l_i    = l_i * l_corr + p_lk
        acc    = acc * l_corr[:, None] + p_lk[:, None] * lk_v[None, :]
        m_i    = m_new

    out_val = acc / tl.maximum(l_i[:, None], 1e-9)

    out_ptrs = (
        Out
        + pid_h * stride_oh
        + (seq_start + m_start + offs_m[:, None]) * stride_ot
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_val.to(Out.dtype.element_ty), mask=valid_m[:, None])


def _compute_landmark_indices(
    x:          torch.Tensor,
    W_V:        torch.Tensor,
    alpha:      torch.Tensor,
    w_sizes:    torch.Tensor,
    cu_seqlens: torch.Tensor,
    k_lmk:      int,
) -> torch.Tensor:
    device   = x.device
    num_seqs = cu_seqlens.shape[0] - 1

    a_mean  = alpha.mean()
    x_norm  = x.norm(dim=-1)
    xwv     = (x @ W_V.T).norm(dim=-1)

    mu_x = x_norm.mean();  std_x = x_norm.std().clamp(min=1e-6)
    mu_v = xwv.mean();     std_v = xwv.std().clamp(min=1e-6)
    z_x  = (x_norm - mu_x) / std_x
    z_v  = (xwv   - mu_v)  / std_v

    scores = a_mean * z_v + (1.0 - a_mean) * z_x

    lmk_indices = torch.zeros(num_seqs, k_lmk, dtype=torch.long, device=device)

    for b in range(num_seqs):
        s  = int(cu_seqlens[b])
        e  = int(cu_seqlens[b + 1])
        sl = e - s
        sc = scores[s:e].clone()

        w_b = w_sizes[s:e].long().clamp(min=1, max=sl)
        pos = torch.arange(sl, device=device)
        lo  = (pos - w_b + 1).clamp(min=0)

        in_window = torch.zeros(sl, dtype=torch.bool, device=device)
        for i in range(sl):
            in_window[int(lo[i]):i + 1] = True

        sc[in_window] = float("-inf")
        k_act = min(k_lmk, int((sc != float("-inf")).sum()))

        if k_act > 0:
            _, idx = torch.topk(sc, k_act, sorted=False)
            lmk_indices[b, :k_act] = idx
            if k_act < k_lmk:
                lmk_indices[b, k_act:] = idx[0]

    return lmk_indices


def _build_landmark_kv(
    K:           torch.Tensor,
    V:           torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
    k_lmk:       int,
    n_heads:     int,
    head_dim:    int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_seqs = cu_seqlens.shape[0] - 1
    device   = K.device

    lmk_K = torch.empty(n_heads, num_seqs, k_lmk, head_dim, device=device, dtype=K.dtype)
    lmk_V = torch.empty(n_heads, num_seqs, k_lmk, head_dim, device=device, dtype=V.dtype)

    for b in range(num_seqs):
        s       = int(cu_seqlens[b])
        abs_idx = s + lmk_indices[b]
        lmk_K[:, b, :, :] = K[abs_idx].transpose(0, 1)
        lmk_V[:, b, :, :] = V[abs_idx].transpose(0, 1)

    return lmk_K.contiguous(), lmk_V.contiguous()


def _build_seq_block_map(
    cu_seqlens: torch.Tensor,
    BLOCK_M:    int,
    device:     torch.device,
) -> tuple[torch.Tensor, int]:
    num_seqs = cu_seqlens.shape[0] - 1
    entries  = []

    for b in range(num_seqs):
        s  = int(cu_seqlens[b])
        e  = int(cu_seqlens[b + 1])
        sl = e - s
        for blk in range(math.ceil(sl / BLOCK_M)):
            entries.append((b, blk))

    map_t = torch.tensor(entries, dtype=torch.int32, device=device).contiguous()
    return map_t, len(entries)


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
    BLOCK_M:    int = 64,
    BLOCK_N:    int = 64,
) -> torch.Tensor:
    total_len, n_heads, head_dim = q.shape
    device   = q.device
    scale    = 1.0 / math.sqrt(head_dim)

    HEAD_DIM_C = triton.next_power_of_2(head_dim)

    q_c = q.contiguous().to(torch.float16)
    k_c = k.contiguous().to(torch.float16)
    v_c = v.contiguous().to(torch.float16)
    out = torch.empty_like(q_c)

    w_int = w_sizes.clamp(min=1).long().contiguous()

    with torch.no_grad():
        lmk_indices = _compute_landmark_indices(
            x.float(), W_V.float(), alpha.float(),
            w_sizes.float(), cu_seqlens, k_lmk,
        )
        lmk_K, lmk_V = _build_landmark_kv(
            k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim,
        )
        seq_block_map, total_blocks = _build_seq_block_map(cu_seqlens, BLOCK_M, device)

    cu_int = cu_seqlens.to(torch.int32).contiguous()

    grid = (total_blocks, n_heads)

    _dsalt_fwd_kernel[grid](
        q_c, k_c, v_c, out,
        w_int,
        lmk_K, lmk_V,
        cu_int,
        seq_block_map,
        q_c.stride(1), q_c.stride(0), q_c.stride(2),
        k_c.stride(1), k_c.stride(0), k_c.stride(2),
        v_c.stride(1), v_c.stride(0), v_c.stride(2),
        out.stride(1),  out.stride(0),  out.stride(2),
        lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2),
        lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2),
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_C,
        K_LMK=k_lmk,
    )

    return out.float()