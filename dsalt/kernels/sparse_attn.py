import torch
import torch.nn.functional as F
import math


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    scale = math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if torch.isnan(attn_weights).any():
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, v)


def sparse_attention_forward_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    scale = math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if torch.isnan(attn_weights).any():
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, v)


def merge_window_landmark_mask(
    window_mask: torch.Tensor,
    landmark_mask: torch.Tensor,
) -> torch.Tensor:
    return window_mask | landmark_mask