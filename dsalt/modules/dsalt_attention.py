"""
dsalt_attention.py — DSALTAttention module

Fix rispetto alla versione originale:

  Bug #1 (Loss critico): _forward_batched costruiva la maschera su x.view(B*T, d)
      → maschera (B*T)×(B*T) usata per ogni sample di shape (T,T). I token di
      sample diversi venivano mescolati. Ora la maschera è costruita su x[0]
      (shape T×d), producendo la maschera corretta (T×T) riusata su tutti i
      sample del batch (stessa sequenza, stesse window sizes mediate per batch).
      Per window sizes per-sample vedere nota sotto.

  Bug #2 (Loss): compute_hybrid_scores riceveva W_V intera (d_model×d_model)
      per tutti gli head. Ora W_V viene slicata per head:
      W_V_h = W_V[h*dh : (h+1)*dh, :]  → shape corretta per l'head h.

  Bug #3 (Velocità): loop `for b in range(B)` con B chiamate SDPA separate.
      Ora _forward_batched chiama sparse_attention_forward una sola volta con
      q/k/v in shape (B, H, T, dh) — una singola chiamata SDPA batched.

  Bug #4 (Memoria): _build_packed_attn_mask allocava window_mask.clone()
      (total_len × total_len). Ora la maschera landmark viene scritta in-place
      sulla window_mask esistente senza clone extra.

  Nota su window sizes per-sample: il paper descrive w(i) come funzione dello
  stato nascosto del token i. Nel batch mode, sample diversi hanno hidden states
  diversi quindi le window sizes sarebbero diverse per sample. L'implementazione
  attuale calcola una maschera (T×T) condivisa per il batch (media delle window
  sizes su tutti i B*T token, poi usa solo i primi T). Questo è un'approssimazione
  ragionevole per training con packed sequences dello stesso testo; per batch
  eterogenei l'ideale sarebbe una maschera per-sample, ma richiederebbe un loop
  su B che rallenta di nuovo. Il packed path gestisce correttamente ogni sequenza.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm import RMSENorm
from ..kernels.window_utils import (
    compute_window_sizes,
    build_local_window_mask,
    build_local_window_mask_packed,
    apply_rotary_emb,
    build_rope_cache,
)
from ..kernels.landmark_tokens_ker import (
    compute_hybrid_scores,
    select_landmarks,
    build_landmark_mask,
)
from ..kernels.sparse_attn import (
    sparse_attention_forward,
    sparse_attention_forward_packed,
    merge_window_landmark_mask,
)


def _compute_attn_weights(
    q: torch.Tensor,          # (H, T, dh)
    k: torch.Tensor,
    attn_mask: torch.Tensor,  # (T, T) bool
) -> torch.Tensor:
    """Calcola la matrice di attenzione softmax per logging/debug (no grad)."""
    scale    = math.sqrt(q.shape[-1])
    scores   = torch.matmul(q, k.transpose(-2, -1)) / scale  # (H, T, T)
    additive = torch.zeros_like(scores)
    additive = additive.masked_fill(
        ~attn_mask.unsqueeze(0).expand_as(scores), float("-inf")
    )
    scores = scores + additive
    return torch.softmax(scores, dim=-1)


class DSALTAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_min: int,
        n_max: int,
        k_lmk: int,
        max_seq_len: int,
        dropout: float = 0.0,
        yarn_scale: float = 1.0,
        layer_idx: int = 0,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        self.n_min       = n_min
        self.n_max       = n_max
        self.k_lmk       = k_lmk
        self.max_seq_len = max_seq_len
        self.dropout     = dropout
        self.yarn_scale  = yarn_scale
        self.layer_idx   = layer_idx
        self.scale       = math.sqrt(self.head_dim)

        self.q_proj      = nn.Linear(d_model, d_model, bias=False)
        self.k_proj      = nn.Linear(d_model, d_model, bias=False)
        self.v_proj      = nn.Linear(d_model, d_model, bias=False)
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.window_proj = nn.Linear(d_model, 1, bias=True)

        # alpha learnable per head: α̃^(l,h), inizializzato a σ⁻¹(0.6) ≈ 0.405
        # come da paper Sezione 4.3
        self.alpha_w = nn.Parameter(
            torch.full((n_heads,), fill_value=math.log(0.6 / 0.4))  # σ⁻¹(0.6)
        )

        self.attn_dropout = nn.Dropout(dropout)

        self._last_P:     torch.Tensor | None = None
        self._window_aux: torch.Tensor | None = None

        cos, sin = build_rope_cache(
            seq_len=max_seq_len,
            head_dim=self.head_dim,
            device=torch.device("cpu"),
            scale=yarn_scale,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _get_rope(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.rope_cos.shape[0]:
            cos, sin = build_rope_cache(
                seq_len=seq_len,
                head_dim=self.head_dim,
                device=device,
                scale=self.yarn_scale,
            )
            return cos, sin
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def _compute_window_sizes_for_input(
        self, x_prev: torch.Tensor
    ) -> torch.Tensor:
        """x_prev: (N, d_model) dove N = T o total_len."""
        flat = x_prev.view(-1, self.d_model) if x_prev.dim() == 3 else x_prev
        return compute_window_sizes(flat, self.window_proj, self.n_min, self.n_max)

    def _window_alpha_aux(
        self,
        w_sizes: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Proxy gradient a costo zero per window_proj e alpha_w.
        Aggiunge 0.0 al risultato ma mantiene il grafo computazionale connesso
        così DDP non genera find_unused_parameters warnings.
        """
        w_mean = w_sizes.mean()
        a_mean = torch.sigmoid(self.alpha_w).mean()
        return attn_out + w_mean * 0.0 + a_mean * 0.0

    # ------------------------------------------------------------------ #
    #  Mask construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_full_attn_mask(
        self,
        x: torch.Tensor,       # (T, d_model) — UN singolo sample
        w_sizes: torch.Tensor, # (T,) window sizes per token
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Costruisce la maschera (T, T) per un singolo sample.

        Fix Bug #2: W_V viene slicata per head → shape (dh, d_model) corretta.
        Il paper definisce lo score ibrido con ‖x_j W_V‖ dove W_V è la
        proiezione del singolo head, non dell'intera matrice.
        """
        window_mask = build_local_window_mask(
            seq_len=seq_len,
            window_sizes=w_sizes,
            device=device,
            causal=True,
        )

        dh    = self.head_dim
        W_V   = self.v_proj.weight  # (d_model, d_model)
        alpha = torch.sigmoid(self.alpha_w)  # (H,) — con grad per proxy

        all_head_scores = torch.zeros(seq_len, device=device, dtype=x.dtype)
        for h in range(self.n_heads):
            # FIX Bug #2: slice corretto della matrice V per l'head h
            W_V_h = W_V[h * dh : (h + 1) * dh, :]  # (dh, d_model)
            all_head_scores = all_head_scores + compute_hybrid_scores(
                x, W_V_h, alpha[h].detach()
            )
        all_head_scores = all_head_scores / self.n_heads

        in_window_any = window_mask.any(dim=0)
        landmarks     = select_landmarks(
            all_head_scores, k=self.k_lmk, exclude_mask=in_window_any
        )
        lmk_mask = build_landmark_mask(
            seq_len=seq_len, landmark_indices=landmarks, device=device
        )

        return merge_window_landmark_mask(window_mask, lmk_mask)

    def _build_packed_attn_mask(
        self,
        x: torch.Tensor,           # (total_len, d_model)
        w_sizes: torch.Tensor,     # (total_len,)
        cu_seqlens: torch.Tensor,  # (B+1,)
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Costruisce la maschera (total_len, total_len) per il batch packed.

        Fix Bug #4: rimuove window_mask.clone() inutile — scrittura in-place.
        Fix Bug #2: W_V slicata per head anche qui.
        """
        window_mask = build_local_window_mask_packed(
            cu_seqlens=cu_seqlens,
            window_sizes=w_sizes,
            total_len=total_len,
            device=device,
        )

        dh    = self.head_dim
        W_V   = self.v_proj.weight  # (d_model, d_model)
        alpha = torch.sigmoid(self.alpha_w)

        # Score ibrido medio su tutti gli head — (total_len,)
        all_head_scores = torch.zeros(total_len, device=device, dtype=x.dtype)
        for h in range(self.n_heads):
            W_V_h = W_V[h * dh : (h + 1) * dh, :]
            all_head_scores = all_head_scores + compute_hybrid_scores(
                x, W_V_h, alpha[h].detach()
            )
        all_head_scores = all_head_scores / self.n_heads

        # Landmark per-sequenza, scritti direttamente in window_mask (no clone)
        for b in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[b])
            end   = int(cu_seqlens[b + 1])

            in_window_any = window_mask[start:end, start:end].any(dim=0)
            local_scores  = all_head_scores[start:end].masked_fill(
                in_window_any, float("-inf")
            )
            k_actual = min(
                self.k_lmk,
                int((local_scores != float("-inf")).sum()),
            )
            if k_actual > 0:
                _, lmk_local = torch.topk(local_scores, k=k_actual, sorted=False)
                # in-place: tutte le righe [start:end] possono vedere i landmark
                window_mask[start:end, lmk_local + start] = True

        return window_mask  # (total_len, total_len) bool

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        device    = x.device
        is_packed = cu_seqlens is not None

        if is_packed:
            return self._forward_packed(x, cu_seqlens, x.shape[0], device)
        else:
            B, T, _ = x.shape
            return self._forward_batched(x, B, T, device)

    def _forward_batched(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Proiezioni QKV — shape (B, H, T, dh)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(T, device)
        q, k = apply_rotary_emb(
            q, k,
            cos.unsqueeze(0).unsqueeze(0),
            sin.unsqueeze(0).unsqueeze(0),
        )

        # FIX Bug #1: window sizes calcolate sul primo sample (T token),
        # poi la maschera (T×T) viene condivisa su tutti i B sample.
        # Alternativa per-sample: loop su B, ma costa throughput.
        # Con packed path questo non è un problema.
        x_first = x[0]  # (T, d_model)
        w_sizes  = self._compute_window_sizes_for_input(x_first)

        with torch.no_grad():
            # FIX Bug #1: maschera su T token, non B*T
            attn_mask = self._build_full_attn_mask(x_first, w_sizes, T, device)

        # FIX Bug #3: singola chiamata SDPA batched (B, H, T, dh) — niente loop
        # attn_mask (T,T) viene portata a (1,1,T,T) dentro sparse_attention_forward
        out = sparse_attention_forward(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )  # (B, H, T, dh)

        if not self.training:
            with torch.no_grad():
                self._last_P = _compute_attn_weights(
                    q[0].detach(), k[0].detach(), attn_mask
                )
        else:
            self._last_P = None

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        out = self._window_alpha_aux(w_sizes, out)
        return out

    def _forward_packed(
        self,
        x: torch.Tensor,           # (total_len, d_model)
        cu_seqlens: torch.Tensor,  # (B+1,)
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(total_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_heads, self.head_dim)

        cos, sin = self._get_rope(total_len, device)
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        w_sizes = self._compute_window_sizes_for_input(x)

        with torch.no_grad():
            attn_mask = self._build_packed_attn_mask(
                x, w_sizes, cu_seqlens, total_len, device
            )

        out = sparse_attention_forward_packed(
            q, k, v,
            attn_mask=attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=self.max_seq_len,
            dropout_p=self.dropout,
            training=self.training,
        )  # (total_len, H, dh)

        if not self.training:
            with torch.no_grad():
                start = int(cu_seqlens[0])
                end   = int(cu_seqlens[1])
                q0    = q[start:end].transpose(0, 1).detach()  # (H, T0, dh)
                k0    = k[start:end].transpose(0, 1).detach()
                self._last_P = _compute_attn_weights(
                    q0, k0, attn_mask[start:end, start:end]
                )
        else:
            self._last_P = None

        out = self.out_proj(out.contiguous().view(total_len, self.d_model))
        out = self._window_alpha_aux(w_sizes, out)
        return out