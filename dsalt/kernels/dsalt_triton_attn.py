import math
import time
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
    W_sizes, Lmk_K, Lmk_V, Cu_seqlens, Seq_block_map,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lkh, stride_lkb, stride_lks, stride_lkd,
    stride_lvh, stride_lvb, stride_lvs, stride_lvd,
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

    for lk in range(0, K_LMK):
        lk_k = tl.load(
            Lmk_K + pid_h * stride_lkh + seq_id * stride_lkb + lk * stride_lks + offs_d * stride_lkd
        ).to(tl.float32)
        lk_v = tl.load(
            Lmk_V + pid_h * stride_lvh + seq_id * stride_lvb + lk * stride_lvs + offs_d * stride_lvd
        ).to(tl.float32)

        qk_lk  = tl.where(valid_m, tl.sum(q * lk_k[None, :], axis=1) * scale, float("-inf"))
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
    tl.store(
        Out
        + (seq_start + m_start + offs_m[:, None]) * stride_ot
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od,
        out_val.to(Out.dtype.element_ty),
        mask=valid_m[:, None],
    )


def _build_seq_block_map(
    cu_seqlens: torch.Tensor,
    block_m:    int,
    device:     torch.device,
) -> tuple[torch.Tensor, int]:
    t0         = time.perf_counter()
    lens       = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    blocks_per = (lens + block_m - 1) // block_m
    total_blks = int(blocks_per.sum())
    seq_col    = torch.repeat_interleave(torch.arange(lens.shape[0], dtype=torch.int32), blocks_per)
    blk_col    = (
        torch.arange(total_blks, dtype=torch.int32)
        - torch.repeat_interleave(blocks_per.cumsum(0) - blocks_per, blocks_per).int()
    )
    result = torch.stack([seq_col, blk_col], dim=1).to(device).contiguous(), total_blks
    print(f"--- [triton] _build_seq_block_map | total_blks={total_blks} block_m={block_m} | t={time.perf_counter()-t0:.4f}s")
    return result


def _compute_landmark_indices(
    x:          torch.Tensor,
    W_V:        torch.Tensor,
    alpha:      torch.Tensor,
    w_sizes:    torch.Tensor,
    cu_seqlens: torch.Tensor,
    k_lmk:      int,
) -> torch.Tensor:
    t0       = time.perf_counter()
    device   = x.device
    total    = x.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total, device=device) - starts[seq_ids]
    max_len  = int(lens.max())

    print(f"--- [triton] _compute_landmark_indices START | total={total} num_seqs={num_seqs} k_lmk={k_lmk} max_len={max_len}")

    t1          = time.perf_counter()
    x_norm      = x.norm(dim=-1)
    xwv         = (x @ W_V.T).norm(dim=-1)
    mu_x, std_x = x_norm.mean(), x_norm.std().clamp(min=1e-6)
    mu_v, std_v = xwv.mean(),    xwv.std().clamp(min=1e-6)
    a           = alpha.mean()
    scores      = a * (xwv - mu_v) / std_v + (1.0 - a) * (x_norm - mu_x) / std_x
    print(f"--- [triton] scores calcolati | mu_x={mu_x.item():.4f} std_x={std_x.item():.4f} mu_v={mu_v.item():.4f} alpha={a.item():.4f} | t={time.perf_counter()-t1:.4f}s")

    w_int = w_sizes.long().clamp(min=1)
    lo    = seq_off - w_int + 1             

    lo_pad = torch.full((num_seqs, max_len), max_len, device=device, dtype=torch.long)
    lo_pad[seq_ids, seq_off] = lo

    suffix_min = lo_pad.flip(1).cummin(dim=1).values.flip(1) 

    pos_range  = torch.arange(max_len, device=device).unsqueeze(0).expand(num_seqs, -1)
    covered_2d = suffix_min <= pos_range                 

    covered = covered_2d[seq_ids, seq_off]

    n_covered = covered.sum().item()
    print(f"--- [triton] covered (in-window) tokens={n_covered}/{total} | non-covered disponibili per landmark={total - n_covered}")

    score_pad = torch.full((num_seqs, max_len), float("-inf"), device=device)
    score_pad[seq_ids, seq_off] = scores.masked_fill(covered, float("-inf"))

    k_eff        = min(k_lmk, max_len)
    _, top_local = torch.topk(score_pad, k_eff, dim=1, sorted=False)
    print(f"--- [triton] topk DONE | k_eff={k_eff}")

    if k_eff >= k_lmk:
        print(f"--- [triton] _compute_landmark_indices DONE | shape={tuple(top_local.shape)} | t={time.perf_counter()-t0:.4f}s")
        return top_local

    fill   = top_local[:, :1].expand(num_seqs, k_lmk - k_eff)
    result = torch.cat([top_local, fill], dim=1)
    print(f"--- [triton] _compute_landmark_indices DONE (con fill) | shape={tuple(result.shape)} | t={time.perf_counter()-t0:.4f}s")
    return result


def _build_landmark_kv(
    K:           torch.Tensor,
    V:           torch.Tensor,
    lmk_indices: torch.Tensor,
    cu_seqlens:  torch.Tensor,
    k_lmk:       int,
    n_heads:     int,
    head_dim:    int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t0       = time.perf_counter()
    starts   = cu_seqlens[:-1].to(K.device)
    num_seqs = cu_seqlens.shape[0] - 1
    abs_idx  = (starts.unsqueeze(1) + lmk_indices).reshape(-1)

    print(f"--- [triton] _build_landmark_kv START | num_seqs={num_seqs} k_lmk={k_lmk} n_heads={n_heads} head_dim={head_dim}")
    print(f"--- [triton] abs_idx range=[{abs_idx.min().item()}, {abs_idx.max().item()}] shape={tuple(abs_idx.shape)}")

    lmk_K = K[abs_idx].view(num_seqs, k_lmk, n_heads, head_dim).permute(2, 0, 1, 3).contiguous()
    lmk_V = V[abs_idx].view(num_seqs, k_lmk, n_heads, head_dim).permute(2, 0, 1, 3).contiguous()

    lmk_mem_mb = (lmk_K.numel() + lmk_V.numel()) * lmk_K.element_size() / 1e6
    print(f"--- [triton] _build_landmark_kv DONE | lmk_K={tuple(lmk_K.shape)} lmk_V={tuple(lmk_V.shape)} | mem={lmk_mem_mb:.2f}MB | t={time.perf_counter()-t0:.4f}s")
    return lmk_K, lmk_V


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
    t0                      = time.perf_counter()
    total_len, n_heads, head_dim = q.shape
    device                  = q.device
    scale                   = 1.0 / math.sqrt(head_dim)
    HEAD_DIM_C              = triton.next_power_of_2(head_dim)

    print(f"--- [triton] dsalt_triton_attention START | total_len={total_len} n_heads={n_heads} head_dim={head_dim} HEAD_DIM_C={HEAD_DIM_C} k_lmk={k_lmk}")
    print(f"--- [triton] scale={scale:.6f} | device={device} | q.dtype={q.dtype}")

    q_c = q.contiguous().to(torch.float16)
    k_c = k.contiguous().to(torch.float16)
    v_c = v.contiguous().to(torch.float16)
    out = torch.zeros_like(q_c)

    w_int = w_sizes.clamp(min=1).long().contiguous()
    print(f"--- [triton] w_int stats | min={w_int.min().item()} max={w_int.max().item()} mean={w_int.float().mean().item():.1f}")

    with torch.no_grad():
        t1 = time.perf_counter()
        lmk_indices = _compute_landmark_indices(
            x.float(), W_V.float(), alpha.float(), w_sizes.float(), cu_seqlens, k_lmk
        )
        print(f"--- [triton] lmk_indices shape={tuple(lmk_indices.shape)} | t={time.perf_counter()-t1:.4f}s")

        t2 = time.perf_counter()
        lmk_K, lmk_V = _build_landmark_kv(k_c, v_c, lmk_indices, cu_seqlens, k_lmk, n_heads, head_dim)
        print(f"--- [triton] landmark KV costruiti | t={time.perf_counter()-t2:.4f}s")

        t3 = time.perf_counter()
        seq_block_map, total_blk = _build_seq_block_map(cu_seqlens, 64, device)
        print(f"--- [triton] seq_block_map={tuple(seq_block_map.shape)} total_blk={total_blk} | t={time.perf_counter()-t3:.4f}s")

    cu_int = cu_seqlens.to(torch.int32).contiguous()

    if torch.cuda.is_available():
        mem_pre = torch.cuda.memory_allocated(device) / 1e9
        print(f"--- [triton] GPU mem PRE kernel: {mem_pre:.3f}GB")

    t4 = time.perf_counter()
    print(f"--- [triton] lancio _dsalt_fwd_kernel | grid=({total_blk}, {n_heads})")
    _dsalt_fwd_kernel[lambda meta: (total_blk, n_heads)](
        q_c, k_c, v_c, out,
        w_int, lmk_K, lmk_V, cu_int, seq_block_map,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lmk_K.stride(0), lmk_K.stride(1), lmk_K.stride(2), lmk_K.stride(3),
        lmk_V.stride(0), lmk_V.stride(1), lmk_V.stride(2), lmk_V.stride(3),
        scale=scale, HEAD_DIM=HEAD_DIM_C, K_LMK=k_lmk,
    )
    torch.cuda.synchronize()
    print(f"--- [triton] _dsalt_fwd_kernel DONE | t_kernel={time.perf_counter()-t4:.4f}s")

    if torch.cuda.is_available():
        mem_post = torch.cuda.memory_allocated(device) / 1e9
        print(f"--- [triton] GPU mem POST kernel: {mem_post:.3f}GB | delta={mem_post-mem_pre:.3f}GB")

    out_f    = out.float()
    out_norm = out_f.norm().item()
    print(f"--- [triton] output | out_norm={out_norm:.4f} | t_total={time.perf_counter()-t0:.4f}s")

    if not math.isfinite(out_norm):
        print(f"--- [triton] CRITICAL: output contiene NaN/Inf! out_norm={out_norm}")

    return out_f