from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from self_flow_hwg.models.dit import ConditionalDiT
from .mask_samplers import ink_aware_mask, random_mask, token_ink_occupancy
from .noising import dual_timestep_batch, vanilla_batch


class SelfFlowObjective(nn.Module):
    VARIANTS = {"vanilla_fm", "dual_fm", "self_flow", "ink_self_flow"}

    def __init__(self, dim: int, student_layer: int, teacher_layer: int):
        super().__init__()
        self.student_layer, self.teacher_layer = student_layer, teacher_layer
        self.projector = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(
        self, variant: str, student: ConditionalDiT, teacher: ConditionalDiT | None,
        clean: torch.Tensor, content: torch.Tensor, content_mask: torch.Tensor,
        style: torch.Tensor, style_mask: torch.Tensor, foreground: torch.Tensor | None = None,
        mask_ratio: float = 0.25, gamma: float = 0.8, ink_alpha: float = 0.3, ink_beta: float = 0.7,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(self.VARIANTS)}")
        mask_ink = clean.new_zeros(())
        if variant == "vanilla_fm":
            batch = vanilla_batch(clean)
        else:
            if variant == "ink_self_flow":
                if foreground is None:
                    raise ValueError("ink_self_flow requires a training-only foreground mask")
                occupancy = token_ink_occupancy(foreground, student.grid)
                mask = ink_aware_mask(occupancy, mask_ratio, ink_alpha, ink_beta)
                mask_ink = occupancy[mask].mean() if mask.any() else clean.new_zeros(())
            else:
                mask = random_mask(clean.shape[0], clean.shape[1], mask_ratio, clean.device)
            batch = dual_timestep_batch(clean, mask)
        capture = (self.student_layer,) if variant in {"self_flow", "ink_self_flow"} else ()
        prediction, hidden = student(batch.student, batch.tau, content, content_mask, style, style_mask, capture)
        generation = F.mse_loss(prediction, batch.target_velocity)
        representation = generation.new_zeros(())
        student_variance = generation.new_zeros(())
        teacher_variance = generation.new_zeros(())
        if variant in {"self_flow", "ink_self_flow"}:
            if teacher is None:
                raise ValueError(f"{variant} requires an EMA teacher")
            with torch.no_grad():
                _, target_hidden = teacher(batch.teacher, batch.teacher_tau, content.detach(), content_mask, style.detach(), style_mask, (self.teacher_layer,))
            projected = F.normalize(self.projector(hidden[self.student_layer]), dim=-1)
            target = F.normalize(target_hidden[self.teacher_layer].detach(), dim=-1)
            representation = 1.0 - (projected * target).sum(dim=-1).mean()
            student_variance = hidden[self.student_layer].float().var(dim=(0, 1)).mean()
            teacher_variance = target_hidden[self.teacher_layer].float().var(dim=(0, 1)).mean()
        total = generation + gamma * representation
        return total, {"loss": total.detach(), "generation": generation.detach(), "representation": representation.detach(), "student_feature_variance": student_variance.detach(), "teacher_feature_variance": teacher_variance.detach(), "selected_ink_occupancy": mask_ink.detach()}

