from __future__ import annotations

import copy

import torch
from torch import nn


def make_ema_teacher(model: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    if not 0 <= decay <= 1:
        raise ValueError(f"EMA decay must be in [0,1], got {decay}")
    for target, source in zip(teacher.parameters(), student.parameters(), strict=True):
        target.lerp_(source.detach(), 1.0 - decay)
    for target, source in zip(teacher.buffers(), student.buffers(), strict=True):
        target.copy_(source)

