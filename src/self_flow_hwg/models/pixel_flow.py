from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from self_flow_hwg.writevit import WriteViTConditioner, load_writevit_weights


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, dim: int, fourier_dim: int = 64):
        super().__init__()
        frequencies = torch.exp(torch.linspace(0, math.log(10000), fourier_dim // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(fourier_dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = timestep[:, None] * self.frequencies[None] * (2 * math.pi)
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))


class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.modulation = nn.Linear(condition_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(image) + self.conv2(F.silu(hidden))


class BottleneckCrossAttention(nn.Module):
    def __init__(self, channels: int, condition_dim: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.condition_projection = nn.Linear(condition_dim, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor, condition_mask: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        context = self.condition_projection(condition)
        tokens = tokens + self.attention(normalized, context, context, key_padding_mask=~condition_mask, need_weights=False)[0]
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class PixelFlowUNet(nn.Module):
    """Compact conditional pixel-space U-Net for original-style Flow Matching."""
    def __init__(self, base_channels: int = 32, condition_dim: int = 256, heads: int = 4, merge_conditioning: bool = False):
        super().__init__()
        self.merge_conditioning = merge_conditioning
        self.time = ScalarTimeEmbedding(condition_dim)
        self.input = nn.Conv2d(3 if merge_conditioning else 1, base_channels, 3, padding=1)
        self.down_block1 = ConditionalResBlock(base_channels, base_channels, condition_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down_block2 = ConditionalResBlock(base_channels * 2, base_channels * 2, condition_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.middle1 = ConditionalResBlock(base_channels * 4, base_channels * 4, condition_dim)
        self.middle_attention = BottleneckCrossAttention(base_channels * 4, condition_dim, heads)
        self.middle2 = ConditionalResBlock(base_channels * 4, base_channels * 4, condition_dim)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up_block1 = ConditionalResBlock(base_channels * 4, base_channels * 2, condition_dim)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up_block2 = ConditionalResBlock(base_channels * 2, base_channels, condition_dim)
        self.output_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.output = nn.Conv2d(base_channels, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(tokens.dtype)[..., None]
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def forward(
        self, image: torch.Tensor, timestep: torch.Tensor, content: torch.Tensor,
        content_mask: torch.Tensor, style: torch.Tensor, style_mask: torch.Tensor,
        context_image: torch.Tensor | None = None, cut_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition_tokens = torch.cat((content, style), dim=1)
        condition_mask = torch.cat((content_mask, style_mask), dim=1)
        global_condition = self.time(timestep) + self._masked_mean(condition_tokens, condition_mask)
        if self.merge_conditioning:
            if context_image is None or cut_mask is None:
                raise ValueError("Merge-conditioned PixelFlowUNet requires context_image and cut_mask")
            image_input = torch.cat((image, context_image, cut_mask), dim=1)
        else:
            if context_image is not None or cut_mask is not None:
                raise ValueError("Context was provided to a non-merge PixelFlowUNet")
            image_input = image
        first = self.down_block1(self.input(image_input), global_condition)
        second = self.down_block2(self.downsample1(first), global_condition)
        middle = self.middle1(self.downsample2(second), global_condition)
        middle = self.middle_attention(middle, condition_tokens, condition_mask)
        middle = self.middle2(middle, global_condition)
        up = self.upsample1(middle)
        up = self.up_block1(torch.cat((up, second), dim=1), global_condition)
        up = self.upsample2(up)
        up = self.up_block2(torch.cat((up, first), dim=1), global_condition)
        return self.output(F.silu(self.output_norm(up)))


class WriteViTPixelFlow(nn.Module):
    def __init__(self, vocabulary: list[str], config: dict):
        super().__init__()
        condition_dim = int(config.get("condition_dim", 256))
        self.conditioning = WriteViTConditioner(
            vocabulary, condition_dim, int(config.get("writevit_dim", 256)),
            int(config.get("writer_depth", 3)), int(config.get("fusion_depth", 5)),
            int(config.get("writevit_heads", 4)), int(config.get("style_tokens", 8)),
            int(config.get("max_text_length", 32)), config.get("unifont_path"), config.get("use_unifont"),
        )
        self.conditioning_load_report = None
        if config.get("writevit_checkpoint"):
            self.conditioning_load_report = load_writevit_weights(self.conditioning, config["writevit_checkpoint"])
        if bool(config.get("freeze_conditioning", False)):
            self.conditioning.requires_grad_(False)
        self.merge_conditioning = bool(config.get("merge_conditioning", False))
        self.flow = PixelFlowUNet(int(config.get("base_channels", 32)), condition_dim, int(config.get("flow_heads", 4)), self.merge_conditioning)

    def forward(
        self, image: torch.Tensor, timestep: torch.Tensor, text_ids: torch.Tensor,
        text_mask: torch.Tensor, reference: torch.Tensor,
        context_image: torch.Tensor | None = None, cut_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditions = self.conditioning(text_ids, text_mask, reference)
        return self.flow(image, timestep, *conditions, context_image, cut_mask)

