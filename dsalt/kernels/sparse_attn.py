import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q:         torch.Tensor,
    k:         torch.Tensor,
    v:         torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training:  bool = False,
) -> torch.Tensor:
    additive = attn_mask.to(dtype=q.dtype).masked_fill(~attn_mask, float("-inf"))
    while additive.dim() < 4:
        additive = additive.unsqueeze(0)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=additive,
        dropout_p=dropout_p if training else 0.0,
    )


def sparse_attention_forward_packed(
    q:          torch.Tensor,
    k:          torch.Tensor,
    v:          torch.Tensor,
    attn_mask:  torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p:  float = 0.0,
    training:   bool = False,
) -> torch.Tensor:
    # q/k/v: [total, n_heads, head_dim]
    # attn_mask: [total, total] bool
    # We do a single batched SDPA over the full packed sequence.
    # The mask already encodes both causal + cross-sequence blocking,
    # so no loop is needed.
    total_len, n_heads, head_dim = q.shape
    dp = dropout_p if training else 0.0

    # [1, n_heads, total, head_dim]
    q_ = q.transpose(0, 1).unsqueeze(0)
    k_ = k.transpose(0, 1).unsqueeze(0)
    v_ = v.transpose(0, 1).unsqueeze(0)

    # additive mask: [1, 1, total, total]
    additive = attn_mask.to(dtype=q.dtype).masked_fill(~attn_mask, float("-inf"))
    additive = additive.unsqueeze(0).unsqueeze(0)

    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=additive, dropout_p=dp)
    # [total, n_heads, head_dim]
    return out.squeeze(0).transpose(0, 1)


def merge_window_landmark_mask(
    window_mask:   torch.Tensor,
    landmark_mask: torch.Tensor,
) -> torch.Tensor:
    return window_mask | landmark_mask