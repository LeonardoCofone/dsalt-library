from .RMSENorm import RMSENorm
from .window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    apply_rotary_emb,
    build_rope_cache,
)
from .landmark_tokens_ker import (
    hybrid_scores_per_head,
    compute_hybrid_scores,
    select_landmarks,
    soft_landmark_weights,
    HybridEnergyLandmarkSelector,
)
from .sparse_attn import (
    sparse_attention_forward,
    sparse_attention_forward_packed,
)

# I kernel Triton sono opzionali: senza Triton/CUDA la libreria resta
# importabile e usa il fallback SDPA (sparse_attn). Gli oggetti Triton
# valgono ``None`` quando il backend non è disponibile.
try:
    from .dsalt_triton_bwd import dsalt_triton_backward
    from .dsalt_triton_attn import dsalt_triton_attention
    from .cross_entropy import LigerFusedLinearCrossEntropyFunction
    _TRITON_OK = True
except Exception:
    dsalt_triton_backward                 = None
    dsalt_triton_attention                = None
    LigerFusedLinearCrossEntropyFunction  = None
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
    "hybrid_scores_per_head",
    "compute_hybrid_scores",
    "select_landmarks",
    "soft_landmark_weights",
    "HybridEnergyLandmarkSelector",
    "sparse_attention_forward",
    "sparse_attention_forward_packed",
]