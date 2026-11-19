"""Opt-in augmented calibration views for the paired Ariadne vision fit (``calibration_augment``).

With ``calibration_augment='openclip_train'`` the calibration images keep the same dataset indices and order as
the default draw, but are transformed with the OpenCLIP *train* transform instead of the eval transform. The
random augmentation of every example is a pure function of ``(calibration seed, dataset index)``, so the source
and the target view of one example receive the identical crop (their train transforms must be equal), and the
result does not depend on loader order, worker count or the global RNG (only the CPU generator is seeded, inside
``fork_rng``, and restored; CUDA generators are untouched).
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .vision_loaders import EMNISTFixTransform, HFVisionDataset

CALIBRATION_AUGMENT_CHOICES = ("none", "openclip_train")

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def transform_repr(transform: Any) -> str:
    """``repr(transform)`` without object addresses (stable across processes)."""
    return _ADDRESS.sub("", repr(transform))


def example_seed(seed: int, index: int) -> int:
    """63-bit seed derived from ``(calibration seed, dataset index)``."""
    digest = hashlib.sha256(f"calibration_augment:{int(seed)}:{int(index)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


class SeededAugmentDataset(Dataset):
    """``base`` (an `HFVisionDataset`) with ``transform`` applied under an RNG seeded per (seed, index)."""

    def __init__(self, base: HFVisionDataset, transform: Callable[[Any], torch.Tensor], seed: int):
        if not isinstance(base, HFVisionDataset):
            raise TypeError(f"calibration_augment needs an HFVisionDataset, got {type(base).__name__}")
        raw_transform = base.transform
        if isinstance(raw_transform, EMNISTFixTransform):
            transform = EMNISTFixTransform(transform)  # same orientation fix as the eval path
        self.base = base
        self.transform = transform
        self.seed = int(seed)
        self._raw = copy.copy(base)
        self._raw.transform = None  # yields (PIL image, label)

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        # Identity attributes (split fingerprint, sample_ids, classes, ...) come from the wrapped dataset.
        if name in {"base", "_raw"}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, idx: int):
        img, y = self._raw[int(idx)]
        with torch.random.fork_rng(devices=[]):  # restores the global CPU RNG on exit
            # CPU generator only (torch.manual_seed would also reseed every CUDA generator, outside fork_rng)
            torch.random.default_generator.manual_seed(example_seed(self.seed, int(idx)))
            out = self.transform(img)
        return out, y


def _rebuild_loader(loader: DataLoader, dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=False,
    )


def _first_batch_sha256(dataset: Dataset, *, seed: int, batch_size: int) -> str:
    """sha256 of the first calibration batch images, in the order ``paired_calibration`` draws them."""
    n = len(dataset)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()[:batch_size]
    images = torch.stack([dataset[i][0] for i in order])
    return hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()


def build_augmented_calibration_pair(
    source_loader: DataLoader,
    target_loader: DataLoader,
    *,
    source_transform: Callable[[Any], torch.Tensor],
    target_transform: Callable[[Any], torch.Tensor],
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Augmented, identically-seeded source/target calibration loaders plus their provenance metadata."""
    src_repr, tgt_repr = transform_repr(source_transform), transform_repr(target_transform)
    if src_repr != tgt_repr:
        raise ValueError(
            "calibration_augment requires equal source and target train transforms (identical augmentation "
            f"in both views); got source={src_repr!r} target={tgt_repr!r}"
        )
    src_ds = SeededAugmentDataset(source_loader.dataset, source_transform, seed)
    tgt_ds = SeededAugmentDataset(target_loader.dataset, target_transform, seed)
    meta = {
        "transform_repr": src_repr,
        "augment_seed": int(seed),
        "first_batch_sha256": {
            "source": _first_batch_sha256(src_ds, seed=seed, batch_size=int(source_loader.batch_size)),
            "target": _first_batch_sha256(tgt_ds, seed=seed, batch_size=int(target_loader.batch_size)),
        },
    }
    return _rebuild_loader(source_loader, src_ds), _rebuild_loader(target_loader, tgt_ds), meta
