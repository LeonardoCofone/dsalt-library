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

    # ── window branch ────────────────────────────────────────────────────────
    col_off  = torch.arange(w_max, device=device)
    j_win    = (i_idx.unsqueeze(1) - col_off.unsqueeze(0)).clamp(min=0)   # [N, w_max]

    valid_win  = col_off.unsqueeze(0) <= i_idx.unsqueeze(1)               # [N, w_max]
    within_w   = col_off.unsqueeze(0) <  w_int.unsqueeze(-1)              # [B, N, w_max]
    win_mask   = (within_w & valid_win.unsqueeze(0))                       # [B, N, w_max]
    win_mask   = win_mask.unsqueeze(1).expand(B, H, N, w_max)

    k_win = k_f[:, :, j_win, :]                                           # [B, H, N, w_max, D]
    v_win = v_f[:, :, j_win, :]

    logits_win = (q_f.unsqueeze(3) * k_win).sum(-1) * scale               # [B, H, N, w_max]
    logits_win = logits_win.masked_fill(~win_mask, float("-inf"))

    # ── landmark branch ───────────────────────────────────────────────────────
    # lmk_indices: [B, H, k_lmk]  →  expand to [B, H, N, k_lmk]
    lmk_exp = lmk_indices.unsqueeze(2).expand(B, H, N, k_lmk)            # FIX: dim 2 not 3

    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, N, k_lmk)
    h_idx = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, N, k_lmk)

    k_lmk_vecs = k_f[b_idx, h_idx, lmk_exp]                              # [B, H, N, k_lmk, D]
    v_lmk_vecs = v_f[b_idx, h_idx, lmk_exp]

    logits_lmk = (q_f.unsqueeze(3) * k_lmk_vecs).sum(-1) * scale         # [B, H, N, k_lmk]

    row_idx    = i_idx.view(1, 1, N, 1).expand(B, H, N, k_lmk)
    lmk_causal = lmk_exp < row_idx
    dist       = (row_idx - lmk_exp).clamp(min=0)
    lmk_in_win = dist < w_int.unsqueeze(1).unsqueeze(-1)                  # [B, H, N, k_lmk]
    lmk_valid  = lmk_causal & ~lmk_in_win

    logits_lmk = logits_lmk.masked_fill(~lmk_valid, float("-inf"))

    # ── joint softmax ─────────────────────────────────────────────────────────
    logits_all = torch.cat([logits_win, logits_lmk], dim=-1)
    attn       = torch.softmax(logits_all, dim=-1)
    attn       = torch.nan_to_num(attn, nan=0.0)

    attn_win = attn[..., :w_max]
    attn_lmk = attn[..., w_max:]

    out = (attn_win.unsqueeze(-1) * v_win).sum(-2) \
        + (attn_lmk.unsqueeze(-1) * v_lmk_vecs).sum(-2)

    return out.to(dtype)