"""
Questo pacchetto aggrega i componenti principali di DSALT, includendo:
- moduli per l'attenzione e il trasformatore,
- il modello di linguaggio DSALTLM,
- kernel per calcoli energetici e predizione della dimensione della finestra,
- il trainer per l'addestramento.
"""

from dsalt.modules.dsalt_attention import DSALTAttention
from dsalt.modules.dsalt_transformer import DSALTTransformer, DSALTBlock, RMSNorm, SwiGLUFFN
from dsalt.model.dsalt_lm import DSALTLMHeadModel
from dsalt.kernels.sparse_attn import dsalt_attention
from dsalt.kernels.RMSENorm import compute_hybrid_energy_scores, select_landmarks, compute_landmark_idx
from dsalt.kernels.window_utils import WindowSizePredictor
from dsalt.training.trainer import DSALTTrainer, get_cosine_schedule_with_warmup

__all__ = [
    "DSALTAttention",
    "DSALTTransformer",
    "DSALTBlock",
    "RMSNorm",
    "SwiGLUFFN",
    "DSALTLMHeadModel",
    "dsalt_attention",
    "compute_hybrid_energy_scores",
    "select_landmarks",
    "compute_landmark_idx",
    "WindowSizePredictor",
    "DSALTTrainer",
    "get_cosine_schedule_with_warmup",
]