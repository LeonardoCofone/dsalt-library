import math
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
    total     = x.shape[0]
    loss_acc  = torch.zeros((), device=x.device, dtype=torch.float32)
    valid_acc = torch.zeros((), device=x.device, dtype=torch.long)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        y_c = labels[start:end]

        valid = (y_c != -100)
        if not valid.any():
            continue

        logits = F.linear(x[start:end], weight)
        loss   = F.cross_entropy(logits, y_c, ignore_index=-100, reduction="sum")

        loss_acc  = loss_acc  + loss.float()
        valid_acc = valid_acc + valid.sum()

        del logits, loss

    return loss_acc / valid_acc.clamp(min=1).float()


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

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = residual_std if any(k in name for k in ("out_proj", "down_proj")) else 0.02
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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

        x = self.embed_dropout(self.embed_tokens(input_ids))

        for layer in self.layers:
            x = layer(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gradient_checkpointing=gradient_checkpointing,
            )

        x = self.final_norm(x)

        loss   = None
        logits = None

        if labels is not None:
            with torch.autocast(x.device.type, enabled=False):
                loss = _chunked_cross_entropy(
                    x.view(-1, self.d_model),
                    self.lm_head.weight,
                    labels.view(-1),
                    self.lm_head_chunk_size,
                )
        else:
            logits = self.lm_head(x)

        return {"loss": loss, "logits": logits}

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())