from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowBatch:
    student: torch.Tensor
    target_velocity: torch.Tensor
    tau: torch.Tensor
    noise: torch.Tensor
    teacher: torch.Tensor | None = None
    teacher_tau: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    t: torch.Tensor | None = None
    s: torch.Tensor | None = None


def interpolate(clean: torch.Tensor, noise: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return (1 - tau[..., None]) * clean + tau[..., None] * noise


def vanilla_batch(clean: torch.Tensor, generator: torch.Generator | None = None) -> FlowBatch:
    b, n, _ = clean.shape
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    t = torch.rand((b, 1), device=clean.device, generator=generator)
    tau = t.expand(-1, n)
    return FlowBatch(interpolate(clean, noise, tau), noise - clean, tau, noise, t=t)


def dual_timestep_batch(clean: torch.Tensor, mask: torch.Tensor, generator: torch.Generator | None = None) -> FlowBatch:
    b, n, _ = clean.shape
    if mask.shape != (b, n):
        raise ValueError(f"Expected mask {(b, n)}, got {tuple(mask.shape)}")
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    t = torch.rand((b, 1), device=clean.device, generator=generator)
    s = torch.rand((b, 1), device=clean.device, generator=generator)
    tau = torch.where(mask, s.expand(-1, n), t.expand(-1, n))
    teacher_tau = torch.minimum(t, s).expand(-1, n)
    return FlowBatch(interpolate(clean, noise, tau), noise - clean, tau, noise, interpolate(clean, noise, teacher_tau), teacher_tau, mask, t, s)

