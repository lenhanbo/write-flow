from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def ode_sample(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], initial_noise: torch.Tensor,
    steps: int = 30, solver: str = "euler",
) -> torch.Tensor:
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError(f"Invalid ODE settings: steps={steps}, solver={solver}")
    x = initial_noise
    times = torch.linspace(1.0, 0.0, steps + 1, device=x.device)
    for current, nxt in zip(times[:-1], times[1:]):
        tau = torch.full(x.shape[:2], current, device=x.device, dtype=x.dtype)
        dt = nxt - current
        v = velocity(x, tau)
        if solver == "heun" and nxt > 0:
            provisional = x + dt * v
            next_tau = torch.full_like(tau, nxt)
            x = x + dt * 0.5 * (v + velocity(provisional, next_tau))
        else:
            x = x + dt * v
    return x
