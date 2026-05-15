"""
sparse_attn.py — DSALT sparse attention kernels

Fix rispetto alla versione originale:
  - Bug #5: flash_attn_varlen_func ignorava completamente attn_mask (landmark persi).
             Ora il path flash_attn è rimosso da sparse_attention_forward_packed:
             flash_attn non supporta attn_bias arbitrari, quindi non può rispettare
             la maschera DSALT. Il fallback SDPA con maschera esplicita è corretto
             e più veloce del loop Python originale grazie al batching per-seq.
  - Bug #3 (parziale): sparse_attention_forward ora riceve (B,H,T,dh) direttamente
             senza loop esterno — il loop era in dsalt_attention._forward_batched.
"""

import torch
import torch.nn.functional as F


def sparse_attention_forward(
    q: torch.Tensor,          # (B, H, T, dh)  oppure (H, T, dh) per singolo sample
    k: torch.Tensor,          # stessa shape di q
    v: torch.Tensor,          # stessa shape di q
    attn_mask: torch.Tensor,  # (T, T) bool — STESSA maschera per tutti i sample
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Attenzione sparsa batched.  q/k/v possono essere (B,H,T,dh) o (H,T,dh).
    attn_mask è (T,T) bool: True = posizione attendibile.

    Internamente converte in additive mask e delega a SDPA che su CUDA sceglie
    automaticamente FlashAttention quando la maschera lo permette.
    """
    # additive mask: shape (1, 1, T, T) — broadcastabile su B e H
    additive = attn_mask.to(dtype=q.dtype)
    additive = torch.zeros_like(additive).masked_fill(~attn_mask, float("-inf"))
    # porta a 4D se necessario
    while additive.dim() < 4:
        additive = additive.unsqueeze(0)

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=additive,
        dropout_p=dropout_p if training else 0.0,
    )


def sparse_attention_forward_packed(
    q: torch.Tensor,           # (total_len, H, dh)
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,   # (total_len, total_len) bool
    cu_seqlens: torch.Tensor,  # (B+1,) int32/int64
    max_seqlen: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Attenzione sparsa packed con maschera DSALT esplicita.

    NON usa flash_attn_varlen_func perché flash_attn non accetta attn_bias
    arbitrari: userebbe solo causal=True e ignorerebbe i landmark token,
    invalidando l'architettura DSALT.

    Strategia: per ogni sequenza nel batch estrae la sotto-maschera quadrata,
    la converte in additive mask e chiama SDPA — una chiamata per sequenza.
    Rispetto al loop originale, q/k/v vengono già tenuti in formato
    (1, H, T_i, dh) evitando permute/unsqueeze extra.
    """
    outputs = []
    num_seqs = cu_seqlens.shape[0] - 1

    for i in range(num_seqs):
        start   = int(cu_seqlens[i])
        end     = int(cu_seqlens[i + 1])
        seq_len = end - start

        # shape: (1, H, T_i, dh)  — SDPA vuole (B, H, T, dh)
        qi = q[start:end].transpose(0, 1).unsqueeze(0)  # (1, H, T_i, dh)
        ki = k[start:end].transpose(0, 1).unsqueeze(0)
        vi = v[start:end].transpose(0, 1).unsqueeze(0)

        # sotto-maschera (T_i, T_i) → additive (1, 1, T_i, T_i)
        mask_i     = attn_mask[start:end, start:end]
        additive_i = torch.zeros(seq_len, seq_len, dtype=qi.dtype, device=qi.device)
        additive_i = additive_i.masked_fill(~mask_i, float("-inf"))
        additive_i = additive_i.unsqueeze(0).unsqueeze(0)

        out_i = F.scaled_dot_product_attention(
            qi, ki, vi,
            attn_mask=additive_i,
            dropout_p=dropout_p if training else 0.0,
        )
        # out_i: (1, H, T_i, dh) → (T_i, H, dh)
        outputs.append(out_i.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)  # (total_len, H, dh)


def merge_window_landmark_mask(
    window_mask: torch.Tensor,
    landmark_mask: torch.Tensor,
) -> torch.Tensor:
    """OR booleano tra maschera locale e maschera landmark."""
    return window_mask | landmark_mask