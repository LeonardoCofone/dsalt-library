from .RMSENorm import RMSENorm
from .window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    apply_rotary_emb,
    build_rope_cache,
)
from .landmark_tokens_ker import (
    compute_hybrid_scores,
    select_landmarks,
    HybridEnergyLandmarkSelector,
)
from .sparse_attn import (
    sparse_attention_forward,
    sparse_attention_forward_packed,
)

from .dsalt_triton_bwd import dsalt_triton_backward

from .cross_entropy import LigerFusedLinearCrossEntropyFunction

try:
    from .dsalt_triton_attn import dsalt_triton_attention
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

__all__ = [
    "dsalt_triton_backward",
    "LigerFusedLinearCrossEntropyFunction",
    "dsalt_triton_attention",
    "RMSENorm",
    "compute_window_sizes",
    "build_local_window_mask",
    "build_local_window_mask_packed",
    "apply_rotary_emb",
    "build_rope_cache",
    "compute_hybrid_scores",
    "select_landmarks",
    "HybridEnergyLandmarkSelector",
    "sparse_attention_forward",
    "sparse_attention_forward_packed",
]