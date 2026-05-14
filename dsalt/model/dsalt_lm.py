import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.RMSENorm import RMSENorm
from ..modules.dsalt_transformer import DSALTTransformerBlock

class DSALTLMHeadModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_min: int,
        n_max: int,
        k_lmk: int,
        max_seq_len: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        yarn_scale: float = 1.0,
        tie_weights: bool = True,
        use_moba: bool = True,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        d_ff = d_ff if d_ff is not None else 4 * d_model

        self.embed_tokens = nn.Embedding(
            vocab_size, d_model,
            padding_idx=padding_idx if padding_idx is not None else -1,
        )
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DSALTTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_min=n_min,
                n_max=n_max,
                k_lmk=k_lmk,
                max_seq_len=max_seq_len,
                d_ff=d_ff,
                dropout=dropout,
                yarn_scale=yarn_scale,
                layer_idx=i,
                use_moba=use_moba,
            ) for i in range(n_layers)
        ])

        self.final_norm = RMSENorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        labels: torch.Tensor | None = None,
        gradient_checkpointing: bool = False,
        padding_mask: torch.BoolTensor | None = None,
    ) -> dict:
        if max_seqlen is None:
            max_seqlen = input_ids.shape[-1]

        x = self.embed_tokens(input_ids)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gradient_checkpointing=gradient_checkpointing,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())