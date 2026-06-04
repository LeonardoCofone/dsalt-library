import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm          import RMSENorm
from ..modules.dsalt_transformer import DSALTTransformerBlock
from .config                     import DSALTConfig

# Liger (fused cross-entropy) requires Triton: optional, imported at runtime
# only when loss_fn="liger" is actually requested.
try:
    from ..kernels.cross_entropy import LigerFusedLinearCrossEntropyFunction
    _LIGER_OK = True
except Exception:
    LigerFusedLinearCrossEntropyFunction = None
    _LIGER_OK = False


def _chunked_cross_entropy(
    x:          torch.Tensor,
    weight:     torch.Tensor,
    labels:     torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    total     = x.shape[0]
    loss_acc  = torch.zeros((), device=x.device, dtype=torch.float32)
    valid_acc = torch.zeros((), device=x.device, dtype=torch.float32)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        y_c = labels[start:end]
        # No explicit ``.float()``: F.cross_entropy already upcasts the logits to
        # fp32 internally for the log_softmax, so a manual fp32 copy of the
        # ``[chunk, vocab]`` logits was pure overhead (profiled at ~24.6ms / step on
        # T4, the dominant aten::copy_). Passing the fp16 logits straight in keeps
        # the same numerics (fp32 softmax) without materialising the fp32 tensor.
        logits = F.linear(x[start:end], weight)
        loss = F.cross_entropy(logits, y_c, ignore_index=-100, reduction="sum")
        loss_acc  += loss
        valid_acc += (y_c != -100).sum()

    return loss_acc / valid_acc.clamp(min=1.0)


def _liger_cross_entropy(
    x:      torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    loss, *_ = LigerFusedLinearCrossEntropyFunction.apply(
        x.contiguous(),
        weight,
        labels.contiguous(),
    )
    return loss


# Valid loss_fn values. "auto" is resolved per-(device, vocab) at first forward
# via loss_autotune (chunked on T4, liger on A100+), then cached.
_LOSS_FN = {
    "auto":    None,
    "liger":   _liger_cross_entropy,
    "chunked": _chunked_cross_entropy,
}


class DSALTLMHeadModel(nn.Module):
    """Causal Language Model based on DSALT (Dynamic Sparse Attention with Landmark Tokens).

    A stack of :class:`~dsalt.modules.dsalt_transformer.DSALTTransformerBlock` with
    token embeddings, a final RMSNorm, and an LM head (optionally tied to the embedding).

    The input is expected in **packed** format (concatenated sequences + ``cu_seqlens``),
    the path optimised for training; the forward also accepts ``[B, T]`` tensors for
    inference.

    Preferably instantiate via :meth:`from_config` with a
    :class:`~dsalt.model.config.DSALTConfig`.

    Args:
        vocab_size:         Vocabulary size.
        d_model:            Model dimension (must be divisible by ``n_heads``).
        n_layers:           Number of Transformer blocks.
        n_heads:            Number of attention heads.
        n_min, n_max:       Bounds of the adaptive local window (§4.2).
        k_lmk:              Number of landmark tokens per query (§4.3).
        max_seq_len:        Max length for the RoPE cache.
        d_ff:               FFN dimension; if ``None`` uses ~8/3·d_model rounded.
        dropout:            Dropout probability.
        yarn_scale:         RoPE/YaRN positional scaling factor.
        tie_weights:        Tie the LM head weights to the embedding.
        padding_idx:        Padding index for the embedding.
        lm_head_chunk_size: Chunk for the "chunked" cross-entropy (memory).
        loss_fn:            ``"chunked"`` (default, memory-frugal), ``"liger"``
                            (fused Triton, wins on A100+), or ``"auto"`` (measure
                            per-GPU). NOTE: ``"chunked"`` with a large chunk is
                            fastest on T4 but materialises ``[chunk, vocab]`` fp32
                            logits — a big memory peak. ``"auto"`` picks by speed
                            only; prefer it on big-VRAM GPUs where ``liger`` (no
                            logits materialisation) wins, not on T4.
        aux_loss_weight:    Weight of the auxiliary term (inert: frozen window,
                            kept for signature compatibility).
    """

    def __init__(
        self,
        vocab_size:         int,
        d_model:            int,
        n_layers:           int,
        n_heads:            int,
        n_min:              int,
        n_max:              int,
        k_lmk:              int,
        max_seq_len:        int,
        d_ff:               int | None = None,
        dropout:            float      = 0.0,
        yarn_scale:         float      = 1.0,
        tie_weights:        bool       = True,
        padding_idx:        int | None = None,
        lm_head_chunk_size: int        = 2048,
        loss_fn:            str        = "chunked",
        aux_loss_weight:    float      = 0.0,
    ):
        super().__init__()
        assert loss_fn in _LOSS_FN, f"loss_fn must be one of {list(_LOSS_FN)}"
        if loss_fn == "liger" and not _LIGER_OK:
            raise RuntimeError(
                "loss_fn='liger' requires the Triton kernel (fused cross_entropy), "
                "not available in this environment. Use loss_fn='chunked'."
            )

        self.d_model            = d_model
        self.n_layers           = n_layers
        self.vocab_size         = vocab_size
        self.lm_head_chunk_size = lm_head_chunk_size
        self.loss_fn            = loss_fn
        self.aux_loss_weight    = aux_loss_weight

        if d_ff is None:
            hidden_dim  = int(8 / 3 * d_model)
            multiple_of = 128
            d_ff        = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of

        self.embed_tokens  = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DSALTTransformerBlock(
                d_model=d_model, n_heads=n_heads, n_min=n_min, n_max=n_max,
                k_lmk=k_lmk, max_seq_len=max_seq_len, d_ff=d_ff,
                dropout=dropout, yarn_scale=yarn_scale, layer_idx=i,
            )
            for i in range(n_layers)
        ])

        self.final_norm = RMSENorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    @classmethod
    def from_config(cls, config: "DSALTConfig") -> "DSALTLMHeadModel":
        """Instantiate the model from a :class:`~dsalt.model.config.DSALTConfig`."""
        return cls(**config.to_dict())

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = residual_std if any(k in name for k in ("out_proj", "down_proj")) else 0.02
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _resolve_loss_fn(self, flat_x: torch.Tensor, flat_labels: torch.Tensor) -> tuple[str, int]:
        """Resolve ``loss_fn="auto"`` to a concrete (loss_fn, chunk_size) per GPU.

        Measured once per ``(device, vocab)`` then cached — same one-shot pattern
        as the kernel block-size autotune. For explicit ``"chunked"``/``"liger"``
        this is a no-op passthrough. Never hard-codes a device: ``"auto"`` picks
        whatever wins on the card actually running (chunked on T4, liger on A100+).
        """
        if self.loss_fn != "auto":
            return self.loss_fn, self.lm_head_chunk_size
        from ..kernels.loss_autotune import autotune_loss
        choice = autotune_loss(
            flat_x, self.lm_head.weight, flat_labels,
            vocab=self.vocab_size,
            chunked_fn=_chunked_cross_entropy,
            liger_fn=_liger_cross_entropy if _LIGER_OK else None,
            liger_ok=_LIGER_OK,
            default_chunk=self.lm_head_chunk_size,
        )
        return choice["loss_fn"], (choice["chunk_size"] or self.lm_head_chunk_size)

    def _compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_x      = x.view(-1, self.d_model)
        flat_labels = labels.view(-1)
        loss_fn, chunk = self._resolve_loss_fn(flat_x, flat_labels)
        if loss_fn == "liger":
            return _liger_cross_entropy(flat_x, self.lm_head.weight, flat_labels)
        return _chunked_cross_entropy(flat_x, self.lm_head.weight, flat_labels, chunk)

    def forward(
        self,
        input_ids:              torch.Tensor,
        cu_seqlens:             torch.Tensor | None     = None,
        max_seqlen:             int | None              = None,
        labels:                 torch.Tensor | None     = None,
        gradient_checkpointing: bool                    = False,
        padding_mask:           torch.BoolTensor | None = None,
    ) -> dict:
        if max_seqlen is None:
            max_seqlen = input_ids.shape[-1]

        x        = self.embed_dropout(self.embed_tokens(input_ids))
        aux_loss = torch.zeros((), device=input_ids.device, dtype=x.dtype)

        rope_cs = None
        cu_list = None
        if cu_seqlens is not None:
            device   = input_ids.device
            attn0    = self.layers[0].attn
            num_seqs = cu_seqlens.shape[0] - 1
            lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
            starts   = cu_seqlens[:-1].to(device)
            seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
            total_len = input_ids.shape[0]
            pos_ids  = torch.arange(total_len, device=device) - starts[seq_ids]
            rope_cs  = (attn0.rope_cos[pos_ids], attn0.rope_sin[pos_ids])
            # Single host copy of cu_seqlens for the whole step (all layers share it).
            # Done here, early, so the one unavoidable D2H sync happens while the GPU
            # queue is still shallow — instead of once per layer mid-stream (which the
            # profiler showed costing ~160ms total). Plain python ints downstream → no
            # per-layer .item()/.to('cpu').
            cu_list = cu_seqlens.detach().to("cpu").tolist()

        for layer in self.layers:
            x, layer_aux = layer(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gradient_checkpointing=gradient_checkpointing,
                rope_cs=rope_cs,
                cu_list=cu_list,
            )
            aux_loss = aux_loss + layer_aux

        x = self.final_norm(x)

        if labels is not None:
            main_loss = self._compute_loss(x, labels)
            loss      = main_loss + self.aux_loss_weight * aux_loss
            return {"loss": loss, "logits": None, "aux_loss": aux_loss.detach()}

        return {"loss": None, "logits": self.lm_head(x), "aux_loss": None}

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())