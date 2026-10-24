from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms.functional import gaussian_blur


@dataclass
class MergeCut:
    context: torch.Tensor
    hard_mask: torch.Tensor
    soft_mask: torch.Tensor
    center: torch.Tensor
    width: torch.Tensor


def sample_center_cut(
    image: torch.Tensor,
    *,
    center_std: float = 0.10,
    width_mean: float = 0.18,
    width_std: float = 0.03,
    min_context: float = 0.20,
    blur_sigma: float = 1.0,
    generator: torch.Generator | None = None,
) -> MergeCut:
    """Cut a vertical middle band while preserving useful context on both sides.

    Centers and widths are sampled from normal distributions. Values are then
    clamped so each side retains at least ``min_context`` of the image width.
    Only the mask boundary is lightly feathered; the removed pixels never enter
    the model context.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(image.shape)}")
    if not 0 <= center_std <= .5 or not 0 < width_mean < 1 or width_std < 0:
        raise ValueError("Invalid normal cut distribution parameters")
    if not 0 <= min_context < .5 or width_mean > 1 - 2 * min_context:
        raise ValueError("Cut width/min_context leave no information on both sides")
    batch, _, _, image_width = image.shape
    device, dtype = image.device, image.dtype
    center = torch.randn(batch, device=device, dtype=dtype, generator=generator) * center_std + .5
    width = torch.randn(batch, device=device, dtype=dtype, generator=generator) * width_std + width_mean
    width = width.clamp(1 / image_width, 1 - 2 * min_context)
    half = width / 2
    center = torch.maximum(center, min_context + half)
    center = torch.minimum(center, 1 - min_context - half)
    coordinates = (torch.arange(image_width, device=device, dtype=dtype) + .5) / image_width
    hard = ((coordinates[None] >= (center - half)[:, None]) & (coordinates[None] <= (center + half)[:, None]))
    hard_mask = hard[:, None, None, :].expand(batch, 1, image.shape[-2], image_width).to(dtype)
    if blur_sigma > 0:
        radius = max(1, round(2 * blur_sigma))
        kernel_width = radius * 2 + 1
        soft_mask = gaussian_blur(hard_mask, [1, kernel_width], [1e-3, blur_sigma]).clamp(0, 1)
    else:
        soft_mask = hard_mask
    context = image * (1 - hard_mask)
    return MergeCut(context, hard_mask, soft_mask, center, width)


def composite_merge(generated: torch.Tensor, original: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
    if generated.shape != original.shape or soft_mask.shape[0] != generated.shape[0]:
        raise ValueError("Generated/original/mask shapes are incompatible")
    return generated * soft_mask + original * (1 - soft_mask)
