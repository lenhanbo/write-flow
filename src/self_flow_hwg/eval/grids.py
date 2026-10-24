from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image


def save_contact_sheet(images: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); save_image(images.clamp(0, 1), path, nrow=nrow, padding=2, pad_value=1)
