import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    additive = torch.zeros_like(attn_mask.to(dtype=q.dtype)).masked_fill(~attn_mask, float("-inf"))
    while additive.dim() < 4:
        additive = additive.unsqueeze(0)

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=additive,
        dropout_p=dropout_p if training else 0.0,
    )


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
    outputs  = []
    num_seqs = cu_seqlens.shape[0] - 1

    for i in range(num_seqs):
        start   = int(cu_seqlens[i])
        end     = int(cu_seqlens[i + 1])
        seq_len = end - start

        qi = q[start:end].transpose(0, 1).unsqueeze(0)
        ki = k[start:end].transpose(0, 1).unsqueeze(0)
        vi = v[start:end].transpose(0, 1).unsqueeze(0)

        mask_i     = attn_mask[start:end, start:end]
        additive_i = torch.zeros(seq_len, seq_len, dtype=qi.dtype, device=qi.device)
        additive_i = additive_i.masked_fill(~mask_i, float("-inf"))
        additive_i = additive_i.unsqueeze(0).unsqueeze(0)

        out_i = F.scaled_dot_product_attention(
            qi, ki, vi,
            attn_mask=additive_i,
            dropout_p=dropout_p if training else 0.0,
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)


def merge_window_landmark_mask(
    window_mask: torch.Tensor,
    landmark_mask: torch.Tensor,
) -> torch.Tensor:
    return window_mask | landmark_mask