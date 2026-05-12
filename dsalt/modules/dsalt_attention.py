"""
Questo modulo definisce l'attenzione DSALT, basata su implementazioni sparse e ottimizzazioni Triton/FlashAttention.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsalt.kernels.sparse_attn import dsalt_attention
from dsalt.kernels.energy_topk_fused import compute_energy_and_topk
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
        n_min: int    = 32,
        n_max: int    = 256,
        k_lmk: int    = 16,
        alpha: float  = 0.6,
        dropout: float = 0.0,
        use_fa2: bool  = True,
        gradient_checkpointing: bool = False,
        compile_attention: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.k_lmk     = k_lmk
        self.alpha_init = alpha
        self.dropout   = dropout
        self.use_fa2   = use_fa2 and _FLASH_AVAILABLE
        self.gradient_checkpointing = gradient_checkpointing
        self.scale     = 1.0 / math.sqrt(self.d_head)

        self.qkv_proj    = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model,     bias=False)

        alpha_init_logit = math.log(alpha / (1.0 - alpha))
        self.alpha_w     = nn.Parameter(torch.full((n_heads,), alpha_init_logit))

        self.window_pred = WindowSizePredictor(d_model, n_heads, n_min, n_max)

        self._init_weights()

        self._attn_fn = (
            torch.compile(self._sparse_attn_forward, mode="reduce-overhead")
            if compile_attention else self._sparse_attn_forward
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _get_wv_weights(self) -> torch.Tensor:
        D  = self.d_model
        H  = self.n_heads
        Dh = self.d_head
        Wv   = self.qkv_proj.weight[2 * D : 3 * D, :]   
        Wv_h = Wv.reshape(H, Dh, D)                      
        return Wv_h.permute(0, 2, 1).contiguous()         

    def _sparse_attn_forward(
        self,
        Q: torch.Tensor,        
        K: torch.Tensor,
        V: torch.Tensor,
        window_sizes: torch.Tensor, 
        landmark_idx: torch.Tensor, 
    ) -> torch.Tensor:       
        return dsalt_attention(Q, K, V, window_sizes, landmark_idx)

    def forward(
        self,
        x: torch.Tensor,                  
        x_prev: Optional[torch.Tensor] = None,  
        return_window: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, N, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        x_pred = x_prev if x_prev is not None else x

        window_sizes, cont_w = self.window_pred(x_pred) 

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  

        Wv = self._get_wv_weights().detach() 

        alpha = torch.sigmoid(self.alpha_w)   

        landmark_idx = compute_energy_and_topk(
            X=x_pred,
            WV=Wv,
            k=self.k_lmk,
            alpha=alpha,
        )  

        if self.gradient_checkpointing and self.training:
            out = torch.utils.checkpoint.checkpoint(
                self._attn_fn, Q, K, V, window_sizes, landmark_idx,
                use_reentrant=False,
            )
        else:
            out = self._attn_fn(Q, K, V, window_sizes, landmark_idx)


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