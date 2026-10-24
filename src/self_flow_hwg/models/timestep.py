from __future__ import annotations

import math

import torch
from torch import nn


class PerTokenTimeEmbedding(nn.Module):
    def __init__(self, dim: int, fourier_dim: int = 64):
        super().__init__()
        if fourier_dim % 2:
            raise ValueError("fourier_dim must be even")
        frequencies = torch.exp(torch.linspace(0, math.log(10000), fourier_dim // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(fourier_dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        angles = tau[..., None] * self.frequencies * (2 * math.pi)
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))

