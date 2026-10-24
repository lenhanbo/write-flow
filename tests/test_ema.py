import torch
from torch import nn

from self_flow_hwg.models.ema import make_ema_teacher, update_ema


def test_ema_is_frozen_and_matches_weighted_average():
    student = nn.Linear(2, 2, bias=False); teacher = make_ema_teacher(student)
    old = teacher.weight.detach().clone()
    with torch.no_grad(): student.weight.add_(2)
    update_ema(teacher, student, .75)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert torch.allclose(teacher.weight, old * .75 + student.weight * .25)

