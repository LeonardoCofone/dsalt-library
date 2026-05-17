import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["HEAD_DIM", "K_LMK"],
)
@triton.jit
def _dsalt_fwd_kernel(
    Q, K, V, Out,
    W_sizes,
    Lmk_K, Lmk_V,
    Cu_seqlens,
    Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_LMK: tl.constexpr,
):
    pid_bm = tl.program_id(0)
    pid_h  = tl.program_id(1)

    seq_id    = tl.load(Seq_block_map + pid_bm * 2).to(tl.int32)
    block_off = tl.load(Seq_block_map + pid_bm * 2 + 1).to(tl.int32)

    seq_start = tl.load(Cu_seqlens + seq_id).to(tl.int32)
    seq_end   = tl.load(Cu_seqlens + seq_id + 1).to(tl.int32)
    seq_len   = seq_end - seq_start

    m_start = block_off * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    valid_m = (m_start + offs_m) < seq_len

    q_ptrs = (
        Q
        + (seq_start + m_start + offs_m[:, None]) * stride_qt
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0).to(tl.float32)

    w_ptrs      = W_sizes + seq_start + m_start + offs_m
    w_sizes     = tl.load(w_ptrs, mask=valid_m, other=1).to(tl.int32)
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

        k_ptrs = (
            K
            + (seq_start + n_start + offs_n[None, :]) * stride_kt
            + pid_h * stride_kh
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + (seq_start + n_start + offs_n[None, :]) * stride_vt
            + pid_h * stride_vh
            + offs_d[:, None] * stride_vd
        )

        k_blk  = tl.load(k_ptrs, mask=valid_n[None, :], other=0.0).to(tl.float32)
        v_blk  = tl.load(v_ptrs, mask=valid_n[None, :], other=0.0).to(tl.float32)
        qk     = tl.dot(q, k_blk) * scale
        j_abs  = n_start + offs_n
        causal = j_abs[None, :] <= i_abs[:, None]
        win_lo = i_abs[:, None] - w_sizes[:, None] + 1
        in_win = (j_abs[None, :] >= win_lo) & causal
        final  = in_win & valid_n[None, :] & valid_m[:, None]

        qk     = tl.where(final, qk, float("-inf"))
        m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
        p      = tl.exp(qk - m_new[:, None])
        p      = tl.where(final, p, 0.0)
        l_corr = tl.exp(m_i - m_new)
        l_i    = l_i * l_corr + tl.sum(p, axis=1)
        acc    = acc * l_corr[:, None] + tl.dot(p.to(tl.float16), tl.trans(v_blk).to(tl.float16)).to(tl.float32)
        m_i    = m_new
        n_start += BLOCK_N

    for lk in range(0, K_LMK):
        lk_k_ptr = (
            Lmk_K
            + pid_h * stride_lkh
            + seq_id * stride_lkb
            + lk * stride_lks
            + offs_d * stride_lkd
        )
        lk_v_ptr = (
            Lmk_V
            + pid_h * stride_lvh
            + seq_id * stride_lvb
            + lk * stride_lvs
            + offs_d * stride_lvd
        )

        lk_k   = tl.load(lk_k_ptr).to(tl.float32)
        lk_v   = tl.load(lk_v_ptr).to(tl.float32)
        qk_lk  = tl.sum(q * lk_k[None, :], axis=1) * scale
        qk_lk  = tl.where(valid_m, qk_lk, float("-inf"))
        m_new  = tl.maximum(m_i, qk_lk)
        p_lk   = tl.where(valid_m, tl.exp(qk_lk - m_new), 0.0)
        l_corr = tl.exp(m_i - m_new)
        l_i    = l_i * l_corr + p_lk
        acc    = acc * l_corr[:, None] + p_lk[:, None] * lk_v[None, :]
        m_i    = m_new

    out_val = tl.where(
        l_i[:, None] > 1e-9,
        acc / l_i[:, None],
        tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
    )

    out_ptrs = (
        Out
        + (seq_start + m_start + offs_m[:, None]) * stride_ot
        + pid_h * stride_oh
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
    total    = x.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1

    x_norm = x.norm(dim=-1)
    xwv    = (x @ W_V.T).norm(dim=-1)
    mu_x   = x_norm.mean();  std_x = x_norm.std().clamp(min=1e-6)
    mu_v   = xwv.mean();     std_v = xwv.std().clamp(min=1e-6)
    a      = alpha.mean()
    scores = a * (xwv - mu_v) / std_v + (1.0 - a) * (x_norm - mu_x) / std_x

    lens    = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off = torch.cat([torch.arange(int(l), device=device) for l in lens.tolist()])
    w_int   = w_sizes.long().clamp(min=1)
    lo      = (seq_off - w_int + 1).clamp(min=0)

    starts  = cu_seqlens[:-1].to(device)
    ends    = cu_seqlens[1:].to(device)

    # indice invertito dentro la sequenza
    inv_pos = ends[seq_ids] - 1 - (starts[seq_ids] + seq_off)

    # scatter lo nel buffer invertito
    lo_inv  = torch.empty(total, dtype=torch.long, device=device)
    lo_inv[starts[seq_ids] + inv_pos] = lo

    # cummin per sequenza — le sequenze sono contigue, lo facciamo in batch
    # costruendo un tensore padded [num_seqs, max_len] e poi cummin
    max_len    = int(lens.max())
    lo_pad     = torch.full((num_seqs, max_len), torch.iinfo(torch.long).max, device=device)
    row_idx    = seq_ids
    col_idx    = inv_pos
    lo_pad[row_idx, col_idx] = lo_inv[starts[seq_ids] + inv_pos]

    min_lo_inv_pad = lo_pad.cummin(dim=1).values

    # raccogliamo min_lo_suffix per ogni token
    min_lo_suffix = min_lo_inv_pad[seq_ids, inv_pos]

    in_window_any = min_lo_suffix <= seq_off
    masked_scores = scores.masked_fill(in_window_any, float("-inf"))

    # top-k per sequenza in batch [num_seqs, max_len]
    score_pad = torch.full((num_seqs, max_len), float("-inf"), device=device)
    score_pad[seq_ids, seq_off] = masked_scores

    k_eff        = min(k_lmk, max_len)
    _, top_local = torch.topk(score_pad, k_eff, dim=1, sorted=False)

    lmk_indices = top_local[:, :k_lmk]
    if k_eff < k_lmk:
        fill = top_local[:, :1].expand(num_seqs, k_lmk - k_eff)
        lmk_indices = torch.cat([top_local, fill], dim=1)

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
    device   = K.device
    num_seqs = cu_seqlens.shape[0] - 1
    starts   = cu_seqlens[:-1].to(device)
    abs_idx  = (starts.unsqueeze(1) + lmk_indices).view(-1)

    lmk_K = K[abs_idx].view(num_seqs, k_lmk, n_heads, head_dim).permute(2, 0, 1, 3).contiguous()
    lmk_V = V[abs_idx].view(num_seqs, k_lmk, n_heads, head_dim).permute(2, 0, 1, 3).contiguous()

    return lmk_K, lmk_V


def _build_seq_block_map(
    cu_seqlens: torch.Tensor,
    BLOCK_M:    int,
    device:     torch.device,
) -> tuple[torch.Tensor, int]:
    lens       = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    blocks_per = ((lens + BLOCK_M - 1) // BLOCK_M)
    total_blks = int(blocks_per.sum())

    seq_col = torch.repeat_interleave(
        torch.arange(lens.shape[0], dtype=torch.int32),
        blocks_per,
    )
    blk_col = torch.cat([torch.arange(int(b), dtype=torch.int32) for b in blocks_per.tolist()])
    entries = torch.stack([seq_col, blk_col], dim=1)

    return entries.to(device).contiguous(), total_blks


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
) -> torch.Tensor:
    total_len, n_heads, head_dim = q.shape
    device     = q.device
    scale      = 1.0 / math.sqrt(head_dim)
    HEAD_DIM_C = triton.next_power_of_2(head_dim)

    q_c = q.contiguous().to(torch.float16)
    k_c = k.contiguous().to(torch.float16)
    v_c = v.contiguous().to(torch.float16)
    out = torch.zeros_like(q_c)
    w_int = w_sizes.clamp(min=1).long().contiguous()

    with torch.no_grad():
        lmk_indices = _compute_landmark_indices(
            x.float(), W_V.float(), alpha.float(),
            w_sizes.float(), cu_seqlens, k_lmk,
        )
        lmk_K, lmk_V = _build_landmark_kv(
            k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim,
        )
        seq_block_map, total_blocks = _build_seq_block_map(cu_seqlens, 64, device)

    cu_int = cu_seqlens.to(torch.int32).contiguous()
    grid   = lambda meta: (total_blocks, n_heads)

    _dsalt_fwd_kernel[grid](
        q_c, k_c, v_c, out,
        w_int, lmk_K, lmk_V, cu_int, seq_block_map,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
        lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
        scale=scale,
        HEAD_DIM=HEAD_DIM_C,
        K_LMK=k_lmk,
    )

    return out.float()