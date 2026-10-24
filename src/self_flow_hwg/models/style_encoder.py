from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    """Patch-token writer encoder inspired by WriteViT's Writer module."""
    def __init__(self, dim: int = 256, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, dim, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn(reference)
        pooled = F.adaptive_avg_pool2d(features, (1, self.num_tokens)).flatten(2).transpose(1, 2)
        tokens = self.norm(pooled)
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)

