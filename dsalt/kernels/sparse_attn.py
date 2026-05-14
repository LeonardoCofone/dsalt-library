import torch
import torch.nn.functional as F
import math

try:
    from flash_attn import flash_attn_varlen_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:

    additive_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
    additive_mask = additive_mask.masked_fill(~attn_mask, float("-inf"))
    additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=additive_mask,
        dropout_p=dropout_p if training else 0.0,
    )
    return out


def sparse_attention_forward_packed(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if _FLASH_AVAILABLE and q.is_cuda:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_p if training else 0.0,
            causal=True,
        )
    
    outputs = []
    num_seqs = cu_seqlens.shape[0] - 1

    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end   = cu_seqlens[i + 1].item()
        seq_len = end - start

        qi = q[start:end].permute(1, 0, 2).unsqueeze(0)
        ki = k[start:end].permute(1, 0, 2).unsqueeze(0)
        vi = v[start:end].permute(1, 0, 2).unsqueeze(0)

        mask_i = attn_mask[start:end, start:end]
        additive_i = torch.zeros(seq_len, seq_len, dtype=qi.dtype, device=qi.device)
        additive_i = additive_i.masked_fill(~mask_i, float("-inf"))
        additive_i = additive_i.unsqueeze(0).unsqueeze(0)

        out_i = F.scaled_dot_product_attention(
            qi, ki, vi,
            attn_mask=additive_i,
            dropout_p=dropout_p if training else 0.0,
        )
        outputs.append(out_i.squeeze(0).permute(1, 0, 2))

    return torch.cat(outputs, dim=0)


def merge_window_landmark_mask(
    window_mask: torch.Tensor,
    landmark_mask: torch.Tensor,
) -> torch.Tensor:
    return window_mask | landmark_mask