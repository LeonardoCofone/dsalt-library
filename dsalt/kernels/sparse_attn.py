import torch
import torch.nn.functional as F


def build_window_mask_batched(w_int: torch.Tensor, N: int) -> torch.Tensor:
    B = w_int.shape[0]
    device = w_int.device
    i_idx = torch.arange(N, device=device).unsqueeze(1)
    j_idx = torch.arange(N, device=device).unsqueeze(0)
    dist = i_idx - j_idx
    w_exp = w_int.unsqueeze(-1).unsqueeze(-1)
    window = (dist >= 0) & (dist < w_exp)
    causal = j_idx <= i_idx
    return window & causal


def sparse_attention_pytorch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_mask: torch.Tensor,
    lmk_indices: torch.Tensor,
) -> torch.Tensor:
    B, H, N, D = q.shape
    k_lmk = lmk_indices.shape[-1]
    device = q.device
    dtype = q.dtype

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    wm = window_mask.unsqueeze(1).expand(B, H, N, N)

    lmk_col = lmk_indices.unsqueeze(3).expand(B, H, N, k_lmk)
    row_idx = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, H, N, k_lmk)
    lmk_valid = lmk_col < row_idx

    lmk_mask = torch.zeros(B, H, N, N, dtype=torch.bool, device=device)
    lmk_mask.scatter_(3, lmk_col.clamp(min=0), lmk_valid)

    combined = wm | lmk_mask

    bias = torch.zeros(B, H, N, N, device=device, dtype=torch.float32)
    bias.masked_fill_(~combined, float("-inf"))

    scale = D ** -0.5
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale + bias
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v_f).to(dtype)


sparse_attention_triton = sparse_attention_pytorch_fallback