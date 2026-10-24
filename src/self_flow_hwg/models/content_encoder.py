from __future__ import annotations

import torch
from torch import nn


class ContentEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, max_length: int = 32, depth: int = 2, heads: int = 4):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, norm_first=False, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.shape[1] > self.max_length:
            raise ValueError(f"Text has {ids.shape[1]} tokens, maximum is {self.max_length}")
        x = self.embedding(ids) + self.position[:, : ids.shape[1]]
        x = self.transformer(x, src_key_padding_mask=~valid_mask)
        return self.norm(x), valid_mask

