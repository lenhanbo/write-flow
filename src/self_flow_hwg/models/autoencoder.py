from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ContinuousAutoencoder(nn.Module):
    """Deterministic x4 convolutional autoencoder; deliberately not a VQ/KL VAE."""
    def __init__(self, latent_channels: int = 8, base_channels: int = 32):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 5, padding=2), ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1), ResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, latent_channels, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 2, 3, padding=1), ResidualBlock(base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1), ResidualBlock(base_channels),
            nn.ConvTranspose2d(base_channels, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(image))


@dataclass
class LatentStats:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.mean.to(latent)) / self.std.to(latent).clamp_min(1e-6)

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.std.to(latent).clamp_min(1e-6) + self.mean.to(latent)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "LatentStats":
        return cls(state["mean"], state["std"])


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    true_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, true_dx) + F.l1_loss(pred_dy, true_dy)


def reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, foreground: torch.Tensor,
    ink_weight: float = 2.0, edge_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(prediction, target)
    ink = ((prediction - target).abs() * foreground).sum() / foreground.sum().clamp_min(1.0)
    edge = edge_loss(prediction, target)
    total = l1 + ink_weight * ink + edge_weight * edge
    return total, {"l1": l1.detach(), "ink_l1": ink.detach(), "edge_l1": edge.detach()}
