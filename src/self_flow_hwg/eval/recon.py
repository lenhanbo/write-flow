from __future__ import annotations

import torch


def reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor, foreground: torch.Tensor) -> dict[str, float]:
    error = (prediction - target).abs()
    fg = foreground > 0.5
    return {"mae": float(error.mean()), "foreground_mae": float(error[fg].mean()) if fg.any() else 0.0, "background_mae": float(error[~fg].mean()) if (~fg).any() else 0.0}

