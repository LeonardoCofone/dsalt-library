"""DSALT — Dynamic Sparse Attention with Landmark Tokens.

Questo pacchetto aggrega i componenti principali della libreria:
- :class:`DSALTConfig` / :class:`DSALTLMHeadModel` — config e modello di linguaggio;
- :class:`DSALTAttention`, :class:`DSALTTransformerBlock`, :class:`SwiGLUFFN` — i blocchi;
- :func:`hybrid_scores_per_head` — fonte unica del punteggio energetico ibrido (§4.3),
  condivisa tra il path SDPA e il kernel Triton;
- utility per finestra adattiva e RoPE;
- :class:`DSALTTrainer` — loop di addestramento con DDP e checkpointing.

I kernel Triton (attenzione sparsa e cross-entropy fuso) sono opzionali: senza
Triton la libreria resta importabile e usa il fallback SDPA.

Esempio::

    from dsalt import DSALTConfig, DSALTLMHeadModel
    cfg   = DSALTConfig(vocab_size=50257, d_model=512, n_layers=6, n_heads=8,
                        n_min=64, n_max=256, k_lmk=16, max_seq_len=1024)
    model = DSALTLMHeadModel.from_config(cfg)
"""

from dsalt.kernels import (
    dsalt_triton_attention,
    RMSENorm,
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    apply_rotary_emb,
    build_rope_cache,
    hybrid_scores_per_head,
    compute_hybrid_scores,
    select_landmarks,
    soft_landmark_weights,
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

from dsalt.model import DSALTConfig, DSALTLMHeadModel
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
    "hybrid_scores_per_head",
    "compute_hybrid_scores",
    "select_landmarks",
    "soft_landmark_weights",
    "HybridEnergyLandmarkSelector",
    "sparse_attention_forward",
    "sparse_attention_forward_packed",
    "DSALTAttention",
    "DSALTTransformerBlock",
    "SwiGLUFFN",
    "DSALTConfig",
    "DSALTLMHeadModel",
]