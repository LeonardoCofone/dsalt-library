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
        self.alpha_w   = nn.Parameter(torch.full((n_heads,), alpha))
        self.window_pred = WindowSizePredictor(d_model, n_heads, n_min, n_max)
        self.scale = 1.0 / math.sqrt(self.d_head)
        self._init_weights()

        self._attn_fn = (
            torch.compile(self._compute_attention, mode="reduce-overhead")
            if compile_attention else self._compute_attention
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _get_wv_weights(self) -> torch.Tensor:
        D  = self.d_model
        H  = self.n_heads
        Dh = self.d_head
        Wv = self.qkv_proj.weight[2 * D : 3 * D, :]  
        Wv_h = Wv.view(H, Dh, D)     
        return Wv_h.permute(0, 2, 1).contiguous() 

    def _compute_attention(
        self,
        Q: torch.Tensor,         
        K: torch.Tensor,
        V: torch.Tensor,
        window_sizes: torch.Tensor,  
        landmark_idx: torch.Tensor,  
    ) -> torch.Tensor:
        N = Q.shape[2]
        max_w = int(window_sizes.max().item())

        if self.use_fa2 and N <= max_w:
            return flash_attn_func(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).transpose(1, 2)
        else:
            return dsalt_attention(Q, K, V, window_sizes, landmark_idx)

    def forward(self, x, x_prev=None, return_window=False):
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head
        dev = x.device

        print(f"[ATT] forward start dev={dev} B={B} N={N}", flush=True)
        
        x_pred = x_prev if x_prev is not None else x

        print(f"[ATT] window_pred start dev={dev}", flush=True)
        window_sizes, cont_w = self.window_pred(x_pred, training=self.training)
        print(f"[ATT] window_pred done dev={dev}", flush=True)

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        print(f"[ATT] qkv done dev={dev}", flush=True)

        Wv = self._get_wv_weights().detach()
        print(f"[ATT] wv done dev={dev}", flush=True)

        with torch.no_grad():
            alpha = torch.sigmoid(self.alpha_w)
            print(f"[ATT] landmark start dev={dev}", flush=True)
            landmark_idx = compute_landmark_idx(
                X=x_pred, WV=Wv, window_sizes=window_sizes,
                k=self.k_lmk, alpha=alpha,
            )
        print(f"[ATT] landmark done dev={dev}", flush=True)

        print(f"[ATT] attn_fn start dev={dev}", flush=True)
        out = self._attn_fn(Q, K, V, window_sizes, landmark_idx)
        print(f"[ATT] attn_fn done dev={dev}", flush=True)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)
        print(f"[ATT] out_proj done dev={dev}", flush=True)

        if return_window:
            return out, cont_w
        return out, None

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, k_lmk={self.k_lmk}, "
            f"alpha_init={self.alpha_init:.2f}"
        )