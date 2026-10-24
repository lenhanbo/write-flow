from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .timestep import PerTokenTimeEmbedding


def patchify(latent: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    b, c, h, w = latent.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"Latent {(h, w)} is not divisible by patch_size={patch_size}")
    return latent.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * patch_size**2)


def unpatchify(tokens: torch.Tensor, channels: int, height: int, width: int, patch_size: int = 2) -> torch.Tensor:
    b, n, d = tokens.shape
    hp, wp = height // patch_size, width // patch_size
    if n != hp * wp or d != channels * patch_size**2:
        raise ValueError(f"Cannot unpatchify {tuple(tokens.shape)} into {(channels, height, width)}")
    return tokens.reshape(b, hp, wp, channels, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape(b, channels, height, width)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, condition: torch.Tensor, condition_mask: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        x = x + self.self_attn(q, q, q, need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), condition, condition, key_padding_mask=~condition_mask, need_weights=False)[0]
        return x + self.ffn(self.norm3(x))


class ConditionalDiT(nn.Module):
    def __init__(
        self, latent_channels: int = 8, latent_size: tuple[int, int] = (16, 64), patch_size: int = 2,
        dim: int = 256, depth: int = 8, heads: int = 4, activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.latent_channels, self.latent_size, self.patch_size = latent_channels, latent_size, patch_size
        self.grid = (latent_size[0] // patch_size, latent_size[1] // patch_size)
        self.activation_checkpointing = activation_checkpointing
        self.num_tokens = self.grid[0] * self.grid[1]
        raw_dim = latent_channels * patch_size**2
        self.input = nn.Linear(raw_dim, dim)
        self.output = nn.Linear(dim, raw_dim)
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, dim) * 0.02)
        self.time = PerTokenTimeEmbedding(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, tokens: torch.Tensor, tau: torch.Tensor, content: torch.Tensor, content_mask: torch.Tensor,
        style: torch.Tensor, style_mask: torch.Tensor, capture_layers: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        if tokens.shape[1] != self.num_tokens or tau.shape != tokens.shape[:2]:
            raise ValueError(f"Expected tokens/tau with N={self.num_tokens}, got {tokens.shape}/{tau.shape}")
        x = self.input(tokens) + self.position + self.time(tau)
        condition = torch.cat((content, style), dim=1)
        condition_mask = torch.cat((content_mask, style_mask), dim=1)
        hidden = {}
        for index, block in enumerate(self.blocks):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, condition, condition_mask, use_reentrant=False)
            else:
                x = block(x, condition, condition_mask)
            if index in capture_layers:
                hidden[index] = x
        return self.output(self.norm(x)), hidden

