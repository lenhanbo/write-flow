from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def fit_word_to_canvas(
    image: Image.Image,
    size: tuple[int, int] = (64, 256),
    margin: int = 4,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Fit a grayscale word onto a white canvas without changing its aspect ratio."""
    if margin < 0 or size[0] <= 2 * margin or size[1] <= 2 * margin:
        raise ValueError(f"Invalid canvas/margin combination: size={size}, margin={margin}")
    image = image.convert("L")
    source = pil_to_tensor(image).float().div_(255.0).unsqueeze(0)
    _, _, source_h, source_w = source.shape
    scale = min((size[0] - 2 * margin) / source_h, (size[1] - 2 * margin) / source_w)
    new_h = max(1, round(source_h * scale))
    new_w = max(1, round(source_w * scale))
    resized = F.interpolate(source, size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    canvas = torch.ones(1, *size, dtype=torch.float32)
    top = (size[0] - new_h) // 2
    left = margin
    canvas[:, top : top + new_h, left : left + new_w] = resized
    return canvas, {
        "scale": float(scale),
        "downscaled": scale < 1.0,
        "clipped": False,
        "content_width": float(new_w),
    }


def load_word_image(path: str | Path, **kwargs: object) -> tuple[torch.Tensor, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IAM word image not found: {path}")
    with Image.open(path) as image:
        return fit_word_to_canvas(image, **kwargs)


def compute_foreground(gray: torch.Tensor, threshold: float = 0.8, soft: bool = True) -> torch.Tensor:
    if gray.ndim not in (3, 4):
        raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(gray.shape)}")
    if soft:
        return ((1.0 - gray) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
    return (gray < threshold).to(gray.dtype)

