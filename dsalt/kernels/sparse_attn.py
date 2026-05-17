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
    additive = torch.zeros(attn_mask.shape, dtype=q.dtype, device=q.device)
    additive.masked_fill_(~attn_mask, float("-inf"))
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
    dropout_p:  float = 0.0,
    training:   bool  = False,
) -> torch.Tensor:
    total_len, n_heads, head_dim = q.shape
    additive = torch.zeros(total_len, total_len, dtype=q.dtype, device=q.device)
    additive.masked_fill_(~attn_mask, float("-inf"))
    q_ = q.transpose(0, 1).unsqueeze(0)
    k_ = k.transpose(0, 1).unsqueeze(0)
    v_ = v.transpose(0, 1).unsqueeze(0)
    additive = additive.unsqueeze(0).unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q_, k_, v_,
        attn_mask=additive,
        dropout_p=dropout_p if training else 0.0,
    )
    return out.squeeze(0).transpose(0, 1).contiguous()