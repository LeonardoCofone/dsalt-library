import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_int: torch.Tensor,
    lmk_indices: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    B, H, N, D = q.shape
    k_lmk  = lmk_indices.shape[-1]
    device = q.device
    dtype  = q.dtype
    scale  = D ** -0.5

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    w_max = int(w_int.max().item())
    out   = torch.zeros(B, H, N, D, device=device, dtype=torch.float32)

    # pre-gather landmark k/v once — [B, H, k_lmk, D]
    b_lmk = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, k_lmk)
    h_lmk = torch.arange(H, device=device).view(1, H, 1).expand(B, H, k_lmk)
    k_lmk_g = k_f[b_lmk, h_lmk, lmk_indices]
    v_lmk_g = v_f[b_lmk, h_lmk, lmk_indices]

    i_idx = torch.arange(N, device=device)
    col_off = torch.arange(w_max, device=device)

    for start in range(0, N, chunk_size):
        end  = min(start + chunk_size, N)
        C    = end - start

        i_c = i_idx[start:end]                                    # [C]
        q_c = q_f[:, :, start:end, :]                             # [B, H, C, D]
        w_c = w_int[:, start:end]                                  # [B, C]

        # window branch — peak alloc: [B, H, C, w_max, D]
        j_win     = (i_c.unsqueeze(1) - col_off.unsqueeze(0)).clamp(min=0)   # [C, w_max]
        valid_win = col_off.unsqueeze(0) <= i_c.unsqueeze(1)                 # [C, w_max]
        within_w  = col_off.unsqueeze(0) < w_c.unsqueeze(-1)                 # [B, C, w_max]
        win_mask  = (within_w & valid_win.unsqueeze(0)).unsqueeze(1).expand(B, H, C, w_max)

        k_win = k_f[:, :, j_win, :]                               # [B, H, C, w_max, D]
        v_win = v_f[:, :, j_win, :]

        logits_win = (q_c.unsqueeze(3) * k_win).sum(-1) * scale   # [B, H, C, w_max]
        logits_win = logits_win.masked_fill(~win_mask, float("-inf"))

        # landmark branch
        logits_lmk = torch.einsum("bhcd,bhkd->bhck", q_c, k_lmk_g) * scale  # [B, H, C, k_lmk]

        lmk_exp = lmk_indices.unsqueeze(2).expand(B, H, C, k_lmk)
        row_exp = i_c.view(1, 1, C, 1).expand(B, H, C, k_lmk)
        dist    = (row_exp - lmk_exp).clamp(min=0)
        lmk_mask = (lmk_exp < row_exp) & (dist >= w_c.unsqueeze(1).unsqueeze(-1))

        logits_lmk = logits_lmk.masked_fill(~lmk_mask, float("-inf"))

        # joint softmax
        attn = torch.softmax(torch.cat([logits_win, logits_lmk], dim=-1), dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out[:, :, start:end] = (
            (attn[..., :w_max].unsqueeze(-1) * v_win).sum(-2)
            + torch.einsum("bhck,bhkd->bhcd", attn[..., w_max:], v_lmk_g)
        )

    return out.to(dtype)