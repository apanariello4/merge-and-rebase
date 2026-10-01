"""Small vision-model helpers shared across rebase methods and eval entrypoints."""

from __future__ import annotations

import torch


def _encode_image(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image") and callable(model.encode_image):
        return model.encode_image(images)
    if hasattr(model, "visual") and callable(model.visual):
        return model.visual(images)
    return model(images)
