"""
Questo pacchetto aggrega i componenti principali di DSALT, includendo:
- moduli per l'attenzione e il trasformatore,
- il modello di linguaggio DSALTLM,
- kernel per calcoli energetici e predizione della dimensione della finestra,
- il trainer per l'addestramento.
"""

from dsalt.modules.dsalt_attention import DSALTAttention
from dsalt.modules.dsalt_transformer import DSALTTransformerBlock, SwiGLUFFN
from dsalt.model.dsalt_lm import DSALTLMHeadModel
from dsalt.kernels.RMSENorm import TritonRMSNorm
from dsalt.kernels.window_utils import compute_window_sizes_triton, build_window_mask_triton
from dsalt.kernels.landmark_tokens_ker import compute_hybrid_energy_triton, apply_yarn_rope_triton, select_landmarks
from dsalt.kernels.sparse_attn import sparse_attention_forward
from dsalt.training.trainer import DSALTTrainer

__all__ = [
    "DSALTAttention",
    "DSALTTransformerBlock",
    "SwiGLUFFN",
    "DSALTLMHeadModel",
    "TritonRMSNorm",
    "compute_window_sizes_triton",
    "build_window_mask_triton",
    "compute_hybrid_energy_triton",
    "apply_yarn_rope_triton",
    "select_landmarks", 
    "sparse_attention_forward",
    "DSALTTrainer",
]