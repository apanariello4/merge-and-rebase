"""Stored, already preprocessed calibration images (task-data-free sets), shared by the Ariadne / anchors fits.

``<dir>/<Task>.pt`` holds a float tensor ``[N, 3, H, W]`` (or a dict with key ``images``) in the models' preprocessed
(normalised) space. Ported from the rebuttal branch (exp/ariadne-calibsrc, 78ea091, eval/vision_rebase/
precomputed_calibration.py) without its context / stage hooks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import Dataset


def tensor_sha256(images: torch.Tensor) -> str:
    """sha256 of the contiguous float32 CPU bytes."""
    arr = images.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


class PrecomputedImageDataset(Dataset):
    """Preprocessed tensors served as ``(image, 0)`` in stored order (``calibration_stored_order`` makes
    ``paired_calibration`` keep that order); ``sample_ids`` give a process-independent identity."""

    calibration_stored_order = True

    def __init__(self, images: torch.Tensor, sha256: str) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"precomputed calibration images must be [N, 3, H, W], got {tuple(images.shape)}")
        if not torch.is_floating_point(images):
            raise ValueError(f"precomputed calibration images must be a float tensor, got {images.dtype}")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("precomputed calibration images contain non-finite values")
        self.images = images
        self.sha256 = sha256
        self.sample_ids = [f"{sha256}:{i}" for i in range(int(images.shape[0]))]

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], 0


def load_precomputed_images(images_dir: str | Path, task: str) -> tuple[torch.Tensor, str, Path]:
    path = Path(images_dir) / f"{task}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"calibration_source='precomputed': {path} does not exist")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, Mapping):
        obj = obj.get("images")
    if not isinstance(obj, torch.Tensor):
        raise ValueError(f"{path} must contain a float tensor [N, 3, H, W] (or a dict with key 'images')")
    return obj, tensor_sha256(obj), path
