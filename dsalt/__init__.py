"""
Questo pacchetto aggrega i componenti principali di DSALT, includendo:
- moduli per l'attenzione e il trasformatore,
- il modello di linguaggio DSALTLM,
- kernel per calcoli energetici e predizione della dimensione della finestra,
- il trainer per l'addestramento.
"""

from dsalt.kernels import (
    dsalt_triton_attention,
    RMSENorm,
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    apply_rotary_emb,
    build_rope_cache,
    compute_hybrid_scores,
    select_landmarks,
    HybridEnergyLandmarkSelector,
    sparse_attention_forward,
    sparse_attention_forward_packed,
    LigerFusedLinearCrossEntropyFunction,
)

from dsalt.modules import (
    DSALTAttention,
    DSALTTransformerBlock,
    SwiGLUFFN,
)

from dsalt.model.dsalt_lm import DSALTLMHeadModel
from dsalt.training.trainer import DSALTTrainer

__all__ = [
    "LigerFusedLinearCrossEntropyFunction",
    "DSALTTrainer",
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
    "DSALTAttention",
    "DSALTTransformerBlock",
    "SwiGLUFFN",
    "DSALTLMHeadModel",
]