from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock, TransformerConfig, init_weights


class BaselineLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        layer_config = TransformerConfig(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            dropout=config.get("dropout", 0.0),
            tie_embeddings=config.get("tie_embeddings", True),
        )
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.drop = nn.Dropout(config.get("dropout", 0.0))
        self.layers = nn.ModuleList([TransformerBlock(layer_config) for _ in range(config["num_layers"])])
        self.norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if config.get("tie_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(init_weights)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        _, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        x = self.drop(self.embed_tokens(input_ids))
        for block in self.layers:
            x = block(x, position_ids)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        out = {"logits": logits, "hidden": hidden}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return out
