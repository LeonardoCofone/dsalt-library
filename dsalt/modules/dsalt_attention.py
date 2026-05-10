"""
dsalt/modules/dsalt_attention.py
---------------------------------
DSALTAttention — Multi-head sparse attention con finestre adattive e landmark.

Cambiamenti rispetto alla versione precedente:
  - x_pred NON viene espanso a [B,H,N,D]: compute_landmark_idx ora accetta [B,N,D]
  - _get_wv_per_head ritorna [H, D, D_head] (invariato)
  - landmark_idx ha shape [B,H,K] invece di [B,H,N,K] — risparmia N× memoria
  - gradient_checkpointing funziona correttamente: fa checkpoint del blocco
    attention completo, non di un lambda su un tensore già calcolato
  - alpha_w: sigmoid applicato una volta sola al momento del calcolo
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils

from dsalt.kernels.sparse_attn import dsalt_attention
from dsalt.kernels.hybrid_energy import compute_landmark_idx
from dsalt.kernels.window_utils import WindowSizePredictor

try:
    from flash_attn import flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


class DSALTAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_min: int   = 32,
        n_max: int   = 256,
        k_lmk: int   = 16,
        alpha: float = 0.6,
        dropout: float = 0.0,
        use_fa2: bool  = True,
        gradient_checkpointing: bool = False,
        compile_attention: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.k_lmk    = k_lmk
        self.alpha_init = alpha
        self.dropout  = dropout
        self.use_fa2  = use_fa2 and _FLASH_AVAILABLE
        self.gradient_checkpointing = gradient_checkpointing

        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model,     bias=False)
        # alpha per head: sigmoid(alpha_w) in [0,1]
        self.alpha_w   = nn.Parameter(torch.full((n_heads,), alpha))
        self.window_pred = WindowSizePredictor(d_model, n_heads, n_min, n_max)
        self.scale = 1.0 / math.sqrt(self.d_head)
        self._init_weights()

        # torch.compile opzionale sul solo blocco di attenzione
        self._attn_fn = (
            torch.compile(self._compute_attention, mode="reduce-overhead")
            if compile_attention else self._compute_attention
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _get_wv_weights(self) -> torch.Tensor:
        """
        Estrae le proiezioni Value dalla matrice QKV.

        qkv_proj.weight: [3*D, D]
        Righe V:         [2D : 3D]   → shape [D, D]
        Per head h:      righe [h*Dh : (h+1)*Dh]

        Ritorna: [H, D, D_head]   (usato da compute_landmark_idx)
        """
        D  = self.d_model
        H  = self.n_heads
        Dh = self.d_head
        Wv = self.qkv_proj.weight[2 * D : 3 * D, :]   # [D, D]
        # Wv[h*Dh:(h+1)*Dh, :] è la proiezione dell'head h
        # Vogliamo [H, D, Dh] = per ogni head, la matrice che proietta D → Dh
        # La matrice del kernel è W tale che output = input @ W.T
        # qkv_proj: y = x @ weight.T  →  V_h = x @ Wv[h].T
        # Wv[h]: [Dh, D] — righe = output dims
        # Per calcolare ‖x @ Wv[h].T‖ = ‖x @ Wv_h‖ con Wv_h = Wv[h].T: [D, Dh]
        Wv_h = Wv.view(H, Dh, D)          # [H, Dh, D]
        return Wv_h.permute(0, 2, 1).contiguous()  # [H, D, Dh]

    def _compute_attention(
        self,
        Q: torch.Tensor,              # [B, H, N, Dh]
        K: torch.Tensor,
        V: torch.Tensor,
        window_sizes: torch.Tensor,   # [B, H, N]  int32
        landmark_idx: torch.Tensor,   # [B, H, K]  int32
    ) -> torch.Tensor:
        """Dispatching FlashAttention2 o kernel DSALT."""
        N = Q.shape[2]
        max_w = int(window_sizes.max().item())

        if self.use_fa2 and N <= max_w:
            # Se tutti i token cadono nella finestra locale, FA2 è equivalente
            return flash_attn_func(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).transpose(1, 2)
        else:
            return dsalt_attention(Q, K, V, window_sizes, landmark_idx)

    def forward(
        self,
        x: torch.Tensor,                      # [B, N, D]
        x_prev: Optional[torch.Tensor] = None, # [B, N, D]  hidden state prev layer
        return_window: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        # Usiamo x_prev per predire le finestre (layer l-1), fallback a x
        x_pred = x_prev if x_prev is not None else x

        # ── Window size prediction ────────────────────────────────────────
        window_sizes, cont_w = self.window_pred(x_pred, training=self.training)
        # window_sizes: [B, H, N]  int32
        # cont_w:       [B, N]     float — per regularizzazione

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Wv = self._get_wv_weights().detach()
        with torch.no_grad():     
            alpha = torch.sigmoid(self.alpha_w)           # [H]
            landmark_idx = compute_landmark_idx(
                X=x_pred,                  # [B, N, D]
                WV=Wv,                     # [H, D, Dh]
                window_sizes=window_sizes, # [B, H, N]
                k=self.k_lmk,
                alpha=alpha,               # [H]
            )
            # landmark_idx: [B, H, K]  int32

        # ── Attention ────────────────────────────────────────────────────
        if self.gradient_checkpointing and self.training:
            # Gradient checkpointing reale: ricalcola il forward durante il backward
            # Risparmia la memoria di Q,K,V activations nel graph
            out = ckpt_utils.checkpoint(
                self._attn_fn,
                Q, K, V, window_sizes, landmark_idx,
                use_reentrant=False,
            )
        else:
            out = self._attn_fn(Q, K, V, window_sizes, landmark_idx)

        # ── Output projection ────────────────────────────────────────────
        # out: [B, H, N, Dh] → [B, N, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)

        if return_window:
            return out, cont_w
        return out, None

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, k_lmk={self.k_lmk}, "
            f"alpha_init={self.alpha_init:.2f}"
        )