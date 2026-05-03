"""
dsalt/kernels/hybrid_energy.py
-------------------------------
Hybrid Energy scoring per landmark selection.

Cambiamenti rispetto alla versione precedente:
  - X ha shape [B, N, D] invece di [B, H, N, D]: elimina la replica per head
    che causava x4-x8 uso di memoria inutile nel forward.
  - compute_landmark_idx NON espande a [B, H, N, K]: il kernel sparse_attn
    usa landmark_idx con shape [B, H, K] leggendo gli stessi indici per ogni
    query token nello stesso head — coerente con la semantica "landmark globali".
  - Il Triton kernel ora itera su B (batch) e H (heads) con X condiviso per head,
    riducendo la memoria da O(B*H*N*D) a O(B*N*D).
"""

import torch
import torch.nn.functional as F
from typing import Tuple

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: calcola ‖x_j‖₂ e ‖x_j W_V^(h)‖₂ per ogni token j, head h
#
# X:   [B, N, D]      — hidden states (non replicati per head)
# WV:  [H, D, D_head] — value projections per head
# out: [B, H, N]      — scores per token
# ─────────────────────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:
    @triton.jit
    def _hybrid_energy_kernel(
        X_ptr,        # [B, N, D]
        WV_ptr,       # [H, D, D_head]
        XNorm_ptr,    # [B, H, N]  output ‖x‖
        XVNorm_ptr,   # [B, H, N]  output ‖xWv‖
        N: tl.constexpr,
        D: tl.constexpr,
        D_head: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_Dh: tl.constexpr,
        # strides X [B, N, D]
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        # stride WV [H, D, D_head]
        stride_wh: tl.constexpr,
        stride_wd: tl.constexpr,
        # strides output [B, H, N]
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
    ):
        """
        Grid: (cdiv(N, BLOCK_N), H, B)
        Ogni program: BLOCK_N token consecutivi, un head, un batch.
        X è condiviso tra tutti gli head → nessuna replica.
        """
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        start  = pid_n * BLOCK_N
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Base pointers
        X_base  = X_ptr  + pid_b * stride_xb
        WV_base = WV_ptr + pid_h * stride_wh
        O_base_x  = XNorm_ptr  + pid_b * stride_ob + pid_h * stride_oh
        O_base_xv = XVNorm_ptr + pid_b * stride_ob + pid_h * stride_oh

        acc_x_sq  = tl.zeros([BLOCK_N], dtype=tl.float32)
        # xV accumulator: [BLOCK_N, BLOCK_Dh]
        acc_xv    = tl.zeros([BLOCK_N, BLOCK_Dh], dtype=tl.float32)

        # Scorre D in blocchi di BLOCK_D
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Carica x_chunk: [BLOCK_N, BLOCK_D]
            x_chunk = tl.load(
                X_base + offs_n[:, None] * stride_xn + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            # ‖x‖² parziale
            acc_x_sq += tl.sum(x_chunk * x_chunk, axis=1)

            # Carica WV chunk: [BLOCK_D, BLOCK_Dh]
            offs_dh = tl.arange(0, BLOCK_Dh)
            wv_chunk = tl.load(
                WV_base + offs_d[:, None] * stride_wd + offs_dh[None, :],
                mask=mask_d[:, None] & (offs_dh[None, :] < D_head),
                other=0.0,
            ).to(tl.float32)

            # x @ WV parziale: [BLOCK_N, BLOCK_Dh]
            acc_xv = acc_xv + tl.dot(x_chunk, wv_chunk)

        # ‖xWv‖² = sum(acc_xv²)
        xv_norm_sq = tl.sum(acc_xv * acc_xv, axis=1)

        tl.store(
            O_base_x  + offs_n, tl.sqrt(acc_x_sq),  mask=mask_n,
        )
        tl.store(
            O_base_xv + offs_n, tl.sqrt(xv_norm_sq), mask=mask_n,
        )


def _cpu_compute_norms(
    X: torch.Tensor,   # [B, N, D]
    WV: torch.Tensor,  # [H, D, D_head]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Referenza CPU.
    Ritorna x_norm [B, H, N] e xv_norm [B, H, N].
    X NON viene replicato: lo proiettiamo per ogni head separatamente.
    """
    B, N, D = X.shape
    H = WV.shape[0]
    X_f = X.float()                        # [B, N, D]
    x_norm_bh = X_f.norm(dim=-1)           # [B, N]
    x_norm = x_norm_bh.unsqueeze(1).expand(B, H, N)  # broadcast, no alloc

    # xV: [B, H, N, D_head]  — calcolato head per head per risparmiare memoria
    xv_list = []
    for h in range(H):
        xv_h = (X_f @ WV[h].float()).norm(dim=-1)  # [B, N]
        xv_list.append(xv_h)
    xv_norm = torch.stack(xv_list, dim=1)  # [B, H, N]

    return x_norm.contiguous(), xv_norm.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# API pubblica
# ─────────────────────────────────────────────────────────────────────────────

def compute_hybrid_energy_scores(
    X: torch.Tensor,      # [B, N, D]       — NON [B, H, N, D]
    WV: torch.Tensor,     # [H, D, D_head]
    alpha: torch.Tensor,  # [H] o scalar
) -> torch.Tensor:
    """
    Calcola gli score ibridi s_j = α‖x_j W_V‖ + (1-α)‖x_j‖ per ogni token j.

    Input:
      X   : [B, N, D]       hidden states (non replicati per head)
      WV  : [H, D, D_head]  value projection weights
      alpha: [H] float in [0,1]

    Output:
      scores: [B, H, N]  z-normalizzati per (b, h)
    """
    B, N, D = X.shape
    H, _, D_head = WV.shape

    x_norms  = torch.empty(B, H, N, dtype=torch.float32, device=X.device)
    xv_norms = torch.empty(B, H, N, dtype=torch.float32, device=X.device)

    if X.is_cuda and _TRITON_AVAILABLE:
        BLOCK_N  = 64
        BLOCK_D  = min(64, triton.next_power_of_2(D))
        BLOCK_Dh = triton.next_power_of_2(D_head)

        X_c  = X.contiguous()
        WV_c = WV.contiguous()

        grid = (triton.cdiv(N, BLOCK_N), H, B)
        _hybrid_energy_kernel[grid](
            X_c, WV_c, x_norms, xv_norms,
            N=N, D=D, D_head=D_head,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_Dh=BLOCK_Dh,
            stride_xb=X_c.stride(0),
            stride_xn=X_c.stride(1),
            stride_wh=WV_c.stride(0),
            stride_wd=WV_c.stride(1),
            stride_ob=x_norms.stride(0),
            stride_oh=x_norms.stride(1),
        )
    else:
        x_norms, xv_norms = _cpu_compute_norms(X, WV)

    def _znorm(t: torch.Tensor) -> torch.Tensor:
        mu  = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (t - mu) / std

    # alpha: [H] → [1, H, 1] per broadcast su [B, H, N]
    if isinstance(alpha, float):
        a = alpha
    else:
        a = alpha.view(1, -1, 1)

    return a * _znorm(xv_norms) + (1.0 - a) * _znorm(x_norms)


def select_landmarks(
    scores: torch.Tensor,        # [B, H, N]
    k: int,
    window_sizes: torch.Tensor,  # [B, H, N]  int32 — esclude la finestra locale
    exclude_last: int = 0,
) -> torch.Tensor:
    """
    Seleziona i top-k token per ogni (batch, head) escludendo la finestra locale.

    Ritorna landmark_idx: [B, H, K]  int32
    NON espanso a [B, H, N, K] — risparmia N× memoria.
    Il kernel sparse_attn usa direttamente [B, H, K].
    """
    B, H, N = scores.shape
    cand_scores = scores.clone()

    # Escludi la finestra locale dell'ultimo token (posizione N-1)
    # window_sizes[:, :, -1] = dimensione finestra dell'ultimo token
    max_w = int(window_sizes[..., -1].max().item())
    if max_w > 0 and N > max_w:
        cand_scores[..., N - max_w:] = float("-inf")
    if exclude_last > 0:
        cand_scores[..., N - exclude_last:] = float("-inf")

    k_safe = min(k, N)
    _, top_idx = torch.topk(cand_scores, k=k_safe, dim=-1, sorted=True)
    top_idx, _ = top_idx.sort(dim=-1)   # ordina per posizione, aiuta la coerenza cache
    top_idx = top_idx.to(torch.int32)

    if k_safe < k:
        pad = torch.zeros(B, H, k - k_safe, dtype=torch.int32, device=scores.device)
        top_idx = torch.cat([top_idx, pad], dim=-1)

    return top_idx   # [B, H, K]  — NON espanso


def compute_landmark_idx(
    X: torch.Tensor,             # [B, N, D]       — hidden states
    WV: torch.Tensor,            # [H, D, D_head]  — value weights
    window_sizes: torch.Tensor,  # [B, H, N]       — finestre adattive
    k: int,
    alpha: torch.Tensor,         # [H] sigmoid già applicato
) -> torch.Tensor:
    """
    Calcola gli indici landmark.

    Ritorna: [B, H, K]  int32
    (non più [B, H, N, K] — eliminata l'espansione costosa)
    """
    scores = compute_hybrid_energy_scores(X, WV, alpha)  # [B, H, N]
    return select_landmarks(scores, k, window_sizes)      # [B, H, K]