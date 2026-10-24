import torch

from self_flow_hwg.flow.losses import SelfFlowObjective
from self_flow_hwg.models.dit import ConditionalDiT
from self_flow_hwg.models.ema import make_ema_teacher


def test_all_variants_finite_and_teacher_frozen():
    student = ConditionalDiT(2, (4, 8), 2, 16, 2, 4)
    teacher = make_ema_teacher(student); objective = SelfFlowObjective(16, 0, 1)
    clean = torch.randn(2, 8, 8); content = torch.randn(2, 3, 16); mask = torch.ones(2, 3, dtype=torch.bool); style = torch.randn(2, 1, 16); style_mask = torch.ones(2, 1, dtype=torch.bool); fg = torch.rand(2, 1, 16, 32)
    projector_received_gradient = False
    for variant in sorted(objective.VARIANTS):
        student.zero_grad(); objective.zero_grad()
        loss, terms = objective(variant, student, teacher if "self_flow" in variant else None, clean, content, mask, style, style_mask, fg)
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
        loss.backward()
        if variant in {"self_flow", "ink_self_flow"}:
            projector_received_gradient |= any(p.grad is not None for p in objective.projector.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert projector_received_gradient
