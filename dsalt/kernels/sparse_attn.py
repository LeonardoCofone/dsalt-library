import time
import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q:         torch.Tensor,
    k:         torch.Tensor,
    v:         torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training:  bool  = False,
) -> torch.Tensor:
    t0 = time.perf_counter()
    print(f"--- [sparse_attn] sparse_attention_forward START | q={tuple(q.shape)} | mask={tuple(attn_mask.shape)} | device={q.device}")

    additive = torch.zeros(attn_mask.shape, dtype=q.dtype, device=q.device)
    additive.masked_fill_(~attn_mask, float("-inf"))
    mask_mem_mb = additive.numel() * additive.element_size() / 1e6
    print(f"--- [sparse_attn] additive mask allocata | mem={mask_mem_mb:.2f}MB | nonzero_frac={attn_mask.float().mean().item():.4f}")

    while additive.dim() < 4:
        additive = additive.unsqueeze(0)

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=additive,
        dropout_p=dropout_p if training else 0.0,
    )
    del additive
    print(f"--- [sparse_attn] sparse_attention_forward DONE | out={tuple(out.shape)} | t={time.perf_counter()-t0:.4f}s")
    return out


def sparse_attention_forward_packed(
    q:          torch.Tensor,
    k:          torch.Tensor,
    v:          torch.Tensor,
    attn_mask:  torch.Tensor,
    dropout_p:  float = 0.0,
    training:   bool  = False,
) -> torch.Tensor:
    t0 = time.perf_counter()
    total_len, n_heads, head_dim = q.shape
    print(f"--- [sparse_attn] sparse_attention_forward_packed START | total_len={total_len} n_heads={n_heads} head_dim={head_dim} | device={q.device}")

    mask_mem_mb = attn_mask.numel() * attn_mask.element_size() / 1e6
    print(f"--- [sparse_attn] attn_mask ricevuta | shape={tuple(attn_mask.shape)} | mem={mask_mem_mb:.2f}MB | nonzero_frac={attn_mask.float().mean().item():.6f}")

    if total_len > 1024:
        print(f"--- [sparse_attn] WARNING: total_len={total_len} > 1024, la mask T×T peserà {total_len*total_len*2/1e6:.1f}MB in fp16 — considera il kernel Triton!")

    additive = torch.zeros(total_len, total_len, dtype=q.dtype, device=q.device)
    additive_mem_mb = additive.numel() * additive.element_size() / 1e6
    print(f"--- [sparse_attn] additive T×T allocata | mem={additive_mem_mb:.2f}MB")

    additive.masked_fill_(~attn_mask, float("-inf"))

    q_ = q.transpose(0, 1).unsqueeze(0)
    k_ = k.transpose(0, 1).unsqueeze(0)
    v_ = v.transpose(0, 1).unsqueeze(0)
    additive_4d = additive.unsqueeze(0).unsqueeze(0)

    print(f"--- [sparse_attn] lancio scaled_dot_product_attention | q_={tuple(q_.shape)} k_={tuple(k_.shape)} mask_4d={tuple(additive_4d.shape)}")
    t1 = time.perf_counter()
    out = F.scaled_dot_product_attention(
        q_, k_, v_,
        attn_mask=additive_4d,
        dropout_p=dropout_p if training else 0.0,
    )
    print(f"--- [sparse_attn] scaled_dot_product_attention DONE | t_sdpa={time.perf_counter()-t1:.4f}s")

    del additive, additive_4d
    result = out.squeeze(0).transpose(0, 1).contiguous()
    print(f"--- [sparse_attn] sparse_attention_forward_packed DONE | out={tuple(result.shape)} | t_total={time.perf_counter()-t0:.4f}s")
    return result