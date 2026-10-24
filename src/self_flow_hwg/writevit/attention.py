"""Minimal PyTorch-only adaptation of WriteViT attention blocks."""

from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        return self.proj((attention @ value).transpose(1, 2).reshape(batch, tokens, dim))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    """Parameter-compatible adaptation of WriteViT CrossAttention."""
    def __init__(self, dim: int, heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.que_dim = dim
        self.key_dim = dim
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_transform = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_transform = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_transform = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.noise_strength_1 = nn.Parameter(torch.zeros(()))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, query_tokens, dim = query.shape
        context_tokens = context.shape[1]
        q = self.q_transform(query).reshape(batch, query_tokens, self.num_heads, dim // self.num_heads).transpose(1, 2)
        k = self.k_transform(context).reshape(batch, context_tokens, self.num_heads, dim // self.num_heads).transpose(1, 2)
        v = self.v_transform(context).reshape(batch, context_tokens, self.num_heads, dim // self.num_heads).transpose(1, 2)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        return query + self.proj((attention @ v).transpose(1, 2).reshape(batch, query_tokens, dim))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CrossConditionBlock(nn.Module):
    """Query self-attention followed by query-to-style cross-attention."""
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attention = CrossAttention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, query: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        query = query + self.attn(self.norm1(query))
        query = self.norm2(query)
        query = query + self.cross_attention(query, style)
        return query + self.mlp(self.norm3(query))


class PosCNN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim))

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return (feature + self.proj(feature)).flatten(2).transpose(1, 2)

