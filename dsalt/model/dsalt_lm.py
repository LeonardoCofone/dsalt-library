import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm         import RMSENorm
from ..modules.dsalt_transformer import DSALTTransformerBlock


def _chunked_cross_entropy(
    x:          torch.Tensor,
    weight:     torch.Tensor,
    labels:     torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    t0 = time.perf_counter()
    total = x.shape[0]
    n_chunks = math.ceil(total / chunk_size)
    print(f"--- [dsalt_lm] _chunked_cross_entropy START | total={total} chunk_size={chunk_size} n_chunks={n_chunks} | x={tuple(x.shape)} labels={tuple(labels.shape)}")
    print(f"--- [dsalt_lm] x dtype={x.dtype} device={x.device} | weight={tuple(weight.shape)} dtype={weight.dtype}")

    n_valid_total = (labels != -100).sum().item()
    print(f"--- [dsalt_lm] token validi={n_valid_total}/{total} ({100*n_valid_total/max(total,1):.1f}%)")

    if n_valid_total == 0:
        print(f"--- [dsalt_lm] WARNING: nessun token valido! Restituisco loss=0")
        return torch.zeros((), device=x.device, dtype=torch.float32)

    loss_acc  = torch.zeros((), device=x.device, dtype=torch.float32)
    valid_acc = torch.zeros((), device=x.device, dtype=torch.long)

    for chunk_idx, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size, total)
        y_c = labels[start:end]

        valid = (y_c != -100)
        n_valid_chunk = valid.sum().item()
        print(f"--- [dsalt_lm] chunk {chunk_idx}/{n_chunks} | [{start}:{end}] | valid={n_valid_chunk}/{end-start}")

        if not valid.any():
            print(f"--- [dsalt_lm] chunk {chunk_idx}: skip (nessun token valido)")
            continue

        t1 = time.perf_counter()
        x_chunk = x[start:end].float()
        logits  = F.linear(x_chunk, weight.float())
        print(f"--- [dsalt_lm] chunk {chunk_idx}: logits={tuple(logits.shape)} dtype={logits.dtype} | t_proj={time.perf_counter()-t1:.4f}s")

        logit_max = logits.max().item()
        logit_min = logits.min().item()
        print(f"--- [dsalt_lm] chunk {chunk_idx}: logits range=[{logit_min:.2f}, {logit_max:.2f}]")

        if not math.isfinite(logit_max) or not math.isfinite(logit_min):
            print(f"--- [dsalt_lm] CRITICAL: logits contengono NaN/Inf! chunk={chunk_idx}")

        t2 = time.perf_counter()
        loss = F.cross_entropy(logits, y_c, ignore_index=-100, reduction="sum")
        print(f"--- [dsalt_lm] chunk {chunk_idx}: loss={loss.item():.4f} | t_ce={time.perf_counter()-t2:.4f}s")

        loss_acc  = loss_acc  + loss.float()
        valid_acc = valid_acc + valid.sum()

        del logits, loss, x_chunk

    final_loss = loss_acc / valid_acc.clamp(min=1).float()
    print(f"--- [dsalt_lm] _chunked_cross_entropy DONE | loss={final_loss.item():.4f} | valid_total={valid_acc.item()} | t={time.perf_counter()-t0:.4f}s")
    return final_loss


class DSALTLMHeadModel(nn.Module):
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
        lm_head_chunk_size: int        = 512,
    ):
        super().__init__()
        self.d_model            = d_model
        self.n_layers           = n_layers
        self.vocab_size         = vocab_size
        self.lm_head_chunk_size = lm_head_chunk_size

        if d_ff is None:
            hidden_dim  = int(8 / 3 * d_model)
            multiple_of = 128
            d_ff        = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of

        print(f"--- [DSALTLMHeadModel] init | vocab={vocab_size} d_model={d_model} n_layers={n_layers} n_heads={n_heads} d_ff={d_ff}")
        print(f"--- [DSALTLMHeadModel] init | n_min={n_min} n_max={n_max} k_lmk={k_lmk} max_seq={max_seq_len} dropout={dropout} tie={tie_weights}")

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
            print(f"--- [DSALTLMHeadModel] tie_weights=True: lm_head.weight == embed_tokens.weight")

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"--- [DSALTLMHeadModel] init DONE | parametri totali={n_params:,} | d_ff calcolato={d_ff}")

    def _init_weights(self):
        print(f"--- [DSALTLMHeadModel] _init_weights START")
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        print(f"--- [DSALTLMHeadModel] residual_std={residual_std:.6f} (n_layers={self.n_layers})")
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = residual_std if any(k in name for k in ("out_proj", "down_proj")) else 0.02
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        print(f"--- [DSALTLMHeadModel] _init_weights DONE")

    def forward(
        self,
        input_ids:              torch.Tensor,
        cu_seqlens:             torch.Tensor | None     = None,
        max_seqlen:             int | None              = None,
        labels:                 torch.Tensor | None     = None,
        gradient_checkpointing: bool                    = False,
        padding_mask:           torch.BoolTensor | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        print(f"--- [DSALTLMHeadModel] forward START | input_ids={tuple(input_ids.shape)} | packed={cu_seqlens is not None} | has_labels={labels is not None} | gc={gradient_checkpointing}")

        if max_seqlen is None:
            max_seqlen = input_ids.shape[-1]
        print(f"--- [DSALTLMHeadModel] max_seqlen={max_seqlen}")

        t1 = time.perf_counter()
        x = self.embed_dropout(self.embed_tokens(input_ids))
        print(f"--- [DSALTLMHeadModel] embedding | x={tuple(x.shape)} dtype={x.dtype} norm={x.norm().item():.4f} | t={time.perf_counter()-t1:.4f}s")

        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated(input_ids.device) / 1e9
            print(f"--- [DSALTLMHeadModel] GPU mem PRIMA dei layer: {mem_before:.3f}GB")

        for i, layer in enumerate(self.layers):
            t_layer = time.perf_counter()
            print(f"--- [DSALTLMHeadModel] === LAYER {i} START === | x_norm={x.norm().item():.4f}")
            x = layer(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gradient_checkpointing=gradient_checkpointing,
            )
            print(f"--- [DSALTLMHeadModel] === LAYER {i} DONE === | x_norm={x.norm().item():.4f} | t={time.perf_counter()-t_layer:.4f}s")

            if torch.cuda.is_available():
                mem_now = torch.cuda.memory_allocated(input_ids.device) / 1e9
                print(f"--- [DSALTLMHeadModel] GPU mem dopo layer {i}: {mem_now:.3f}GB")

        t2 = time.perf_counter()
        x = self.final_norm(x)
        print(f"--- [DSALTLMHeadModel] final_norm | x={tuple(x.shape)} norm={x.norm().item():.4f} | t={time.perf_counter()-t2:.4f}s")

        loss   = None
        logits = None

        if labels is not None:
            print(f"--- [DSALTLMHeadModel] calcolo loss su labels={tuple(labels.shape)} | chunk_size={self.lm_head_chunk_size}")
            if torch.cuda.is_available():
                mem_pre_loss = torch.cuda.memory_allocated(input_ids.device) / 1e9
                print(f"--- [DSALTLMHeadModel] GPU mem PRE loss: {mem_pre_loss:.3f}GB")
            t3 = time.perf_counter()
            with torch.autocast(x.device.type, enabled=False):
                loss = _chunked_cross_entropy(
                    x.view(-1, self.d_model),
                    self.lm_head.weight,
                    labels.view(-1),
                    self.lm_head_chunk_size,
                )
            print(f"--- [DSALTLMHeadModel] loss={loss.item():.4f} | t_loss={time.perf_counter()-t3:.4f}s")

            if torch.cuda.is_available():
                mem_post_loss = torch.cuda.memory_allocated(input_ids.device) / 1e9
                print(f"--- [DSALTLMHeadModel] GPU mem POST loss: {mem_post_loss:.3f}GB")
        else:
            print(f"--- [DSALTLMHeadModel] nessuna label, calcolo logits completi")
            logits = self.lm_head(x)
            print(f"--- [DSALTLMHeadModel] logits={tuple(logits.shape)}")

        print(f"--- [DSALTLMHeadModel] forward DONE | t_total={time.perf_counter()-t0:.4f}s")
        return {"loss": loss, "logits": logits}

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            n = sum(p.numel() for p in self.parameters())
        print(f"--- [DSALTLMHeadModel] num_parameters(trainable_only={trainable_only})={n:,}")
        return n