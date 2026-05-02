import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsalt.kernels.sparse_attn import dsalt_attention
from dsalt.kernels.hybrid_energy import compute_landmark_idx
from dsalt.kernels.window_utils import WindowSizePredictor

try:
    from flash_attn import flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


class DSALTAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_min=32, n_max=256, k_lmk=16,
                 alpha=0.6, dropout=0.0, use_fa2=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.k_lmk   = k_lmk
        self.alpha_init = alpha
        self.alpha_w = nn.Parameter(torch.full((n_heads,), alpha))
        self.dropout  = dropout
        self.use_fa2  = use_fa2 and _FLASH_AVAILABLE
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.window_pred = WindowSizePredictor(d_model, n_heads, n_min, n_max)
        self.scale = 1.0 / math.sqrt(self.d_head)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _get_wv_per_head(self):
        # qkv_proj.weight: [3*d_model, d_model]
        # Righe V: [2*d_model : 3*d_model] — shape [d_model, d_model]
        # PyTorch usa layout head-contiguous:
        #   V[h*d_head:(h+1)*d_head, :] sono le righe dell'head h
        # Quindi Wv[h]: righe [h*d_head:(h+1)*d_head] → shape [d_head, d_model]
        # Per energy scoring vogliamo [H, d_model, d_head] (x @ Wv[h])
        W  = self.qkv_proj.weight           # [3*d, d]
        d  = self.d_model
        h  = self.n_heads
        dh = self.d_head
        Wv = W[2 * d: 3 * d, :]            # [d_model, d_model]
        # Wv ha righe ordinate come [head0_dim0..head0_dimDh, head1_dim0..]
        Wv_heads = Wv.view(h, dh, d)       # [H, d_head, d_model]
        return Wv_heads.permute(0, 2, 1).contiguous()   # [H, d_model, d_head]

    def forward(self, x, x_prev=None, return_window=False):
        B, N, D = x.shape
        H  = self.n_heads
        Dh = self.d_head
        x_pred = x_prev if x_prev is not None else x

        window_sizes, cont_w = self.window_pred(x_pred, training=self.training)

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        with torch.no_grad():
            x_pred_heads = x_pred.unsqueeze(1).expand(B, H, N, D)
            Wv = self._get_wv_per_head()
            landmark_idx = compute_landmark_idx(
                X=x_pred_heads, WV=Wv,
                window_sizes=window_sizes,
                k=self.k_lmk, alpha = torch.sigmoid(self.alpha_w),
            )

        if self.use_fa2 and N <= int(window_sizes.max().item()):
            out = flash_attn_func(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).transpose(1, 2)
        else:
            out = dsalt_attention(Q, K, V, window_sizes, landmark_idx)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)

        if return_window:
            return out, cont_w
        return out, None

    def extra_repr(self):
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"d_head={self.d_head}, k_lmk={self.k_lmk}, alpha={self.alpha}")