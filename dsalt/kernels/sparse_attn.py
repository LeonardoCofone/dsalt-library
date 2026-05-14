import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_int: torch.Tensor,
    lmk_indices: torch.Tensor,
) -> torch.Tensor:
    B, H, N, D = q.shape
    k_lmk = lmk_indices.shape[-1]
    device = q.device
    dtype = q.dtype
    scale = D ** -0.5

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    w_max = int(w_int.max().item())

    i_idx = torch.arange(N, device=device)
    col_off = torch.arange(w_max, device=device)
    j_win = (i_idx.unsqueeze(1) - col_off.unsqueeze(0)).clamp(min=0)

    valid_win = col_off.unsqueeze(0) <= i_idx.unsqueeze(1)
    w_int_exp = w_int.unsqueeze(-1)
    within_w = col_off.unsqueeze(0) < w_int_exp
    win_mask = (within_w & valid_win).unsqueeze(1).expand(B, H, N, w_max)

    k_win = k_f[:, :, j_win, :]
    v_win = v_f[:, :, j_win, :]

    logits_win = (q_f.unsqueeze(3) * k_win).sum(-1) * scale
    logits_win = logits_win.masked_fill(~win_mask, float("-inf"))

    lmk_exp = lmk_indices.unsqueeze(3).expand(B, H, N, k_lmk)
    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, N, k_lmk)
    h_idx = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, N, k_lmk)

    k_lmk_vecs = k_f[b_idx, h_idx, lmk_exp]
    v_lmk_vecs = v_f[b_idx, h_idx, lmk_exp]

    logits_lmk = (q_f.unsqueeze(3) * k_lmk_vecs).sum(-1) * scale

    row_idx = i_idx.view(1, 1, N, 1).expand(B, H, N, k_lmk)
    lmk_causal = lmk_exp < row_idx

    dist_to_lmk = (row_idx - lmk_exp).clamp(min=0)
    lmk_in_win = dist_to_lmk < w_int_exp.unsqueeze(1)
    lmk_valid = lmk_causal & ~lmk_in_win
    logits_lmk = logits_lmk.masked_fill(~lmk_valid, float("-inf"))

    logits_all = torch.cat([logits_win, logits_lmk], dim=-1)
    attn = torch.softmax(logits_all, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    attn_win = attn[..., :w_max]
    attn_lmk = attn[..., w_max:]

    out = (attn_win.unsqueeze(-1) * v_win).sum(-2) + (attn_lmk.unsqueeze(-1) * v_lmk_vecs).sum(-2)
    return out.to(dtype)