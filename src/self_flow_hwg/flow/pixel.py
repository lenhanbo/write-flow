from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch


@dataclass
class PixelFlowBatch:
    state: torch.Tensor
    velocity: torch.Tensor
    timestep: torch.Tensor
    noise: torch.Tensor


def pixel_ot_batch(data: torch.Tensor, sigma_min: float = 0.001, generator: torch.Generator | None = None) -> PixelFlowBatch:
    """OT conditional path from noise at t=0 to pixel data at t=1.

    This follows Eq. 22/23 of Lipman et al. rather than the reversed
    clean-at-zero convention used by Self-Flow.
    """
    if not 0 <= sigma_min < 1:
        raise ValueError(f"sigma_min must be in [0,1), got {sigma_min}")
    noise = torch.randn(data.shape, device=data.device, dtype=data.dtype, generator=generator)
    timestep = torch.rand(data.shape[0], device=data.device, generator=generator)
    weight = timestep[:, None, None, None]
    state = (1 - (1 - sigma_min) * weight) * noise + weight * data
    velocity = data - (1 - sigma_min) * noise
    return PixelFlowBatch(state, velocity, timestep, noise)


@torch.no_grad()
def pixel_ode_sample(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], initial_noise: torch.Tensor,
    steps: int = 30, solver: str = "euler",
) -> torch.Tensor:
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError(f"Invalid ODE settings: steps={steps}, solver={solver}")
    state = initial_noise
    times = torch.linspace(0.0, 1.0, steps + 1, device=state.device)
    for current, nxt in zip(times[:-1], times[1:]):
        timestep = torch.full((state.shape[0],), current, device=state.device, dtype=state.dtype)
        dt = nxt - current
        current_velocity = velocity(state, timestep)
        if solver == "heun" and nxt < 1:
            provisional = state + dt * current_velocity
            next_timestep = torch.full_like(timestep, nxt)
            state = state + dt * 0.5 * (current_velocity + velocity(provisional, next_timestep))
        else:
            state = state + dt * current_velocity
    return state
