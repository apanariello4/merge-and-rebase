"""Opt-in task-data-free calibration images for the Ariadne vision fit (``calibration_data: "precomputed"``).

Per task, ``<calibration_images_dir>/<Task>.pt`` holds a float tensor ``[N, 3, H, W]`` already in the models'
preprocessed (normalised) space. The fit uses the first ``num_batches * batch_size`` images IN STORED ORDER, the
very same tensors for the source and the target model (no resize or renormalisation: the source and target
OpenCLIP preprocessors must agree, which ``assert_same_preprocessing`` checks on the built classifiers).
The sha256 of every tensor is recorded in the run summary (``direct_residual.calibration``).

Default runs (``calibration_data: "task_local"``) never touch this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .context import _CalibrationLoaders, _TaskContext

PRECOMPUTED = "precomputed"
IMAGES_DIR_KEY = "calibration_images_dir"


def tensor_sha256(images: torch.Tensor) -> str:
    """sha256 of the contiguous float32 CPU bytes (dtype is part of the hash input through the cast)."""
    arr = images.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


class PrecomputedImageDataset(Dataset):
    """Preprocessed image tensors served as ``(image, 0)`` in stored order.

    ``sample_ids`` (``<sha256>:<row>``) gives `_stable_dataset_identity` a process-independent identity;
    ``calibration_stored_order`` makes ``paired_calibration`` keep the stored order instead of drawing a seeded
    permutation (the builders already shuffled the images into their batches).
    """

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
        raise FileNotFoundError(f"calibration_data='precomputed': {path} does not exist")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, Mapping):  # builders save {"images": tensor, ...}; a bare tensor is accepted too
        obj = obj.get("images")
    if not isinstance(obj, torch.Tensor):
        raise ValueError(f"{path} must contain a float tensor [N, 3, H, W] (or a dict with key 'images')")
    return obj, tensor_sha256(obj), path


def build_precomputed_context(
    task: str,
    *,
    cfg: Mapping[str, Any],
    num_batches: int,
    batch_size: int,
    template: _TaskContext,
) -> tuple[_TaskContext, dict[str, Any]]:
    """Paired source/target context over the first ``num_batches * batch_size`` stored images of ``task``.

    ``template`` is the task's own context: classnames and build configs are kept (only the calibration images
    change; alpha search and evaluation keep the task's own splits).
    """
    images_dir = cfg.get(IMAGES_DIR_KEY)
    if not images_dir:
        raise ValueError(f"calibration_data='precomputed' requires the top-level config key {IMAGES_DIR_KEY!r}")
    images, digest, path = load_precomputed_images(images_dir, task)
    need = int(num_batches) * int(batch_size)
    if images.shape[0] < need:
        raise ValueError(f"{path}: {images.shape[0]} images but num_batches*batch_size = {need} are required")
    used = images[:need]
    dataset = PrecomputedImageDataset(used, tensor_sha256(used))
    kwargs = {"batch_size": int(batch_size), "shuffle": False, "num_workers": 0, "drop_last": False}
    # One dataset object for both loaders: source and target see identical tensors in identical order.
    loader_source = DataLoader(dataset, **kwargs)
    loader_target = DataLoader(dataset, **kwargs)
    context = _TaskContext(
        loaders=_CalibrationLoaders(train=loader_target),
        source_loaders=_CalibrationLoaders(train=loader_source),
        classnames=template.classnames,
        build_cfg_task=template.build_cfg_task,
        source_build_cfg_task=template.source_build_cfg_task,
        target_text_features=template.target_text_features,
        source_text_features=template.source_text_features,
    )
    meta = {
        "path": str(path),
        "file_sha256": digest,
        "n_stored": int(images.shape[0]),
        "n_used": need,
        "used_sha256": dataset.sha256,
        "shape": list(used.shape),
    }
    return context, meta


def assert_same_preprocessing(clf_source: Any, clf_target: Any) -> dict[str, Any]:
    """The tensors are fed to both models unchanged: refuse unless both normalise to the same space."""

    def describe(clf: Any) -> dict[str, Any]:
        visual = clf.model.visual
        return {
            "image_size": tuple(getattr(visual, "image_size", ())) if hasattr(visual, "image_size") else None,
            "mean": tuple(getattr(visual, "image_mean", ()) or ()),
            "std": tuple(getattr(visual, "image_std", ()) or ()),
        }

    src, tgt = describe(clf_source), describe(clf_target)
    if src != tgt:
        raise ValueError(
            f"calibration_data='precomputed' feeds identical tensors to both models, but their preprocessing "
            f"differs: source={src} target={tgt}"
        )
    return src


def check_precomputed_supported(cfg: Mapping[str, Any], resolved: Any) -> None:
    """Fail before any model is built if the precomputed-calibration options are inconsistent."""
    is_precomputed = bool(resolved.direct_fit) and resolved.ariadne_cfg.calibration_data == PRECOMPUTED
    if cfg.get(IMAGES_DIR_KEY) and not is_precomputed:
        raise ValueError(f"{IMAGES_DIR_KEY} is only used with the Ariadne direct fit and calibration_data='precomputed'")
    if not is_precomputed:
        return
    if not cfg.get(IMAGES_DIR_KEY):
        raise ValueError(f"calibration_data='precomputed' requires the top-level config key {IMAGES_DIR_KEY!r}")
    if resolved.ariadne_cfg.merge_mode == "merge_in_source_then_fit":
        raise ValueError("calibration_data='precomputed' is per task and does not support merge_in_source_then_fit")
    root = Path(str(cfg[IMAGES_DIR_KEY]))
    missing = [t for t in resolved.tasks if not (root / f"{t}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"{IMAGES_DIR_KEY}={root}: missing {[f'{t}.pt' for t in missing]}")
