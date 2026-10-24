from __future__ import annotations

import torch
import torch.nn.functional as F


def random_mask(batch: int, tokens: int, ratio: float, device: torch.device) -> torch.Tensor:
    if not 0 <= ratio <= 0.5:
        raise ValueError(f"mask ratio must be in [0, .5], got {ratio}")
    count = round(tokens * ratio)
    scores = torch.rand(batch, tokens, device=device)
    result = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
    if count:
        result.scatter_(1, scores.topk(count, dim=1).indices, True)
    return result


def token_ink_occupancy(foreground: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(foreground.float(), grid).flatten(1)


def ink_aware_mask(occupancy: torch.Tensor, ratio: float, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    batch, tokens = occupancy.shape
    count = round(tokens * ratio)
    result = torch.zeros_like(occupancy, dtype=torch.bool)
    if not count:
        return result
    weights = alpha + beta * occupancy.clamp(0, 1)
    empty = occupancy.sum(dim=1) <= 1e-8
    weights[empty] = 1.0
    indices = torch.multinomial(weights, count, replacement=False)
    return result.scatter(1, indices, True)

