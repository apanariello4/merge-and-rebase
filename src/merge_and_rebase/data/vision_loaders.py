from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from datasets import ClassLabel, DatasetDict, Features
from datasets import Dataset as HFDataset
from datasets import load_dataset as hf_load_dataset
from PIL import ExifTags, Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _ensure_pillow_exif_compat() -> None:
    # datasets may access PIL.Image.ExifTags.Base.Orientation, which is missing
    # in older Pillow versions. Provide a minimal compatibility shim.
    if not hasattr(Image, "ExifTags"):
        Image.ExifTags = ExifTags
    if not hasattr(Image.ExifTags, "Base"):
        orientation = 274
        for tag, name in getattr(ExifTags, "TAGS", {}).items():
            if name == "Orientation":
                orientation = int(tag)
                break
        Image.ExifTags.Base = SimpleNamespace(Orientation=orientation)


_ensure_pillow_exif_compat()

LabelRemap = dict[int, int] | Sequence[int] | np.ndarray | Callable[[int], int] | None

Transform = Callable[[Any], torch.Tensor] | None


@dataclass(frozen=True)
class VisionLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    classnames: Sequence[str]
    ft_epochs: int
    sizes: dict[str, int]
    meta: dict[str, Any]


def _to_pil_if_needed(x: Any):
    """
    HF images can come as PIL.Image, numpy arrays, or dataset-specific types.
    We only convert numpy arrays -> PIL, everything else is returned as-is.
    """
    if isinstance(x, np.ndarray):
        # local import to avoid torchvision hard dependency at import time
        import torchvision.transforms.functional as F

        return F.to_pil_image(x)
    return x


def _apply_emnist_orientation_fix(img: Any) -> Any:
    import torchvision.transforms.functional as F

    img = F.rotate(img, -90)
    img = F.hflip(img)
    return img


@dataclass(frozen=True)
class EMNISTFixTransform:
    base: Callable[[Any], Any]

    def __call__(self, img: Any) -> Any:
        return _apply_emnist_orientation_fix(self.base(img))


def emnist_fix_transform(base: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    EMNIST samples often need rotation + flip to match expected orientation.
    """
    return EMNISTFixTransform(base)


KMNIST_CLASSNAMES = [
    "hiragana o",
    "hiragana ki",
    "hiragana su",
    "hiragana tsu",
    "hiragana na",
    "hiragana ha",
    "hiragana ma",
    "hiragana ya",
    "hiragana re",
    "hiragana wo",
]


def batch_to_dict(batch, x_key: str = "x", y_key: str = "y") -> dict[str, Any]:
    """
    Convert (x, y) or (x, y, meta) tuples to dicts.
    If already dict, return unchanged.
    """
    if isinstance(batch, dict):
        return batch
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Expected tuple/list or dict batch, got {type(batch)}")

    if len(batch) == 2:
        return {x_key: batch[0], y_key: batch[1]}
    if len(batch) == 3:
        return {x_key: batch[0], y_key: batch[1], "meta": batch[2]}
    raise ValueError(f"Unexpected batch arity: {len(batch)}")


def _normalize_name(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def compute_label_remap_by_names(
    current_names: Sequence[str],
    desired_names: Sequence[str],
    normalize: Callable[[str], str] = _normalize_name,
) -> np.ndarray:
    """
    Returns an array map old_label -> new_label by matching class names (normalized).
    """
    cur = [normalize(n) for n in current_names]
    des = [normalize(n) for n in desired_names]

    if len(set(cur)) != len(cur):
        raise ValueError("Duplicates in current_names after normalization.")
    if len(set(des)) != len(des):
        raise ValueError("Duplicates in desired_names after normalization.")

    if set(cur) != set(des):
        only_cur = sorted(set(cur) - set(des))
        only_des = sorted(set(des) - set(cur))
        raise ValueError(f"Class mismatch. only_current={only_cur}, only_desired={only_des}")

    des_index = {n: i for i, n in enumerate(des)}
    return np.array([des_index[n] for n in cur], dtype=np.int64)


# ---------------------------
# Torch dataset wrapper over HF
# ---------------------------


class HFVisionDataset(Dataset):
    """
    Thin wrapper around a HuggingFace Dataset split that yields (image_tensor, label).
    Supports:
      - image conversion (numpy -> PIL)
      - label remap
      - arbitrary transform (open_clip preprocess, torchvision transforms, etc.)
    """

    def __init__(
        self,
        split: HFDataset,
        *,
        transform: Transform,
        image_key: str = "image",
        label_key: str = "label",
        label_remap: LabelRemap = None,
    ):
        self.split = split
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.label_remap = label_remap

        # Optional attribute expected by some codebases
        self.classes: Sequence[str] | None = None

    def __len__(self) -> int:
        return int(self.split.num_rows)

    def _map_label(self, y: Any) -> int:
        if isinstance(y, bool):
            y = int(y)
        y = int(y)

        m = self.label_remap
        if m is None:
            return y
        if callable(m):
            return int(m(y))
        return int(m[y])

    def __getitem__(self, idx: int):
        ex = self.split[int(idx)]
        img = _to_pil_if_needed(ex[self.image_key])
        y = self._map_label(ex[self.label_key])

        if self.transform is not None:
            img = self.transform(img)

        return img, y


# ---------------------------
# HF loading helpers
# ---------------------------

_ALLOWED_SPLITS = ("train", "test", "val", "validation")


def _fix_special_dataset_columns(path: str, splits: dict[str, HFDataset]) -> dict[str, HFDataset]:
    if path != "clip-benchmark/wds_fer2013":
        return splits

    # FER2013 webdataset wrapper used in some CLIP benchmarks.
    # Produces keys: image, label
    fixed_out = {}
    for k, v in splits.items():
        v = v.remove_columns([c for c in ["__key__", "__url__"] if c in v.column_names])
        v = v.rename_columns({"jpg": "image", "cls": "label"})
        fixed_out[k] = v
    return fixed_out


def load_hf_splits(
    path: str,
    *,
    config: str | None = None,
    requested_splits: Iterable[str] | None = None,
    allowed_splits: Iterable[str] = _ALLOWED_SPLITS,
    **kwargs,
) -> DatasetDict:
    """
    Loads only the requested HF splits.

    When ``requested_splits`` is omitted, the loader falls back to the common
    vision split names in ``allowed_splits``. This avoids resolving unrelated
    dataset splits up front, which can be noisy and slower on datasets with many
    named shards.
    """
    allowed = set(allowed_splits)
    wanted = list(dict.fromkeys(requested_splits if requested_splits is not None else allowed_splits))
    if not wanted:
        raise ValueError("No dataset splits requested.")

    invalid = [split for split in wanted if split not in allowed]
    if invalid:
        raise ValueError(f"Unsupported requested splits: {invalid}. allowed={sorted(allowed)}")

    out: dict[str, HFDataset] = {}
    errors: dict[str, str] = {}
    for split in wanted:
        try:
            if config is None:
                out[split] = hf_load_dataset(path, split=split, **kwargs)
            else:
                out[split] = hf_load_dataset(path, config, split=split, **kwargs)
        except Exception as exc:
            errors[split] = f"{type(exc).__name__}: {exc}"

    if len(out) == len(wanted):
        return DatasetDict(_fix_special_dataset_columns(path, out))

    # Fallback: load everything only if direct per-split loading did not satisfy
    # the requested keys. This preserves compatibility with unusual dataset
    # builders while keeping the common path quiet and minimal.
    ds = hf_load_dataset(path, config, **kwargs) if config else hf_load_dataset(path, **kwargs)
    if isinstance(ds, DatasetDict):
        filtered = {k: v for k, v in ds.items() if k in wanted}
        if len(filtered) == len(wanted):
            return DatasetDict(_fix_special_dataset_columns(path, filtered))
        missing = [split for split in wanted if split not in filtered]
        raise ValueError(
            f"Missing requested splits after fallback. requested={wanted}, "
            f"available={list(ds.keys())}, missing={missing}, direct_errors={errors}"
        )
    if isinstance(ds, HFDataset):
        if wanted == ["train"]:
            return DatasetDict({"train": ds})
        raise ValueError(f"Dataset exposes a single unnamed split, but requested={wanted}. direct_errors={errors}")
    raise ValueError(f"Unexpected HF dataset type: {type(ds)}") from None


def extract_classnames(
    hf_ds: DatasetDict,
    *,
    label_key: str = "label",
    override: Sequence[str] | None = None,
    strict: bool = True,
) -> Sequence[str]:
    """
    Extracts classnames from ClassLabel if present, else either raises (strict)
    or returns stringified indices.
    """
    if override is not None:
        return list(override)

    for split in hf_ds.values():
        feats: Features = split.features
        if label_key in feats and isinstance(feats[label_key], ClassLabel):
            return list(feats[label_key].names)

    if strict:
        raise AssertionError(
            f"Could not find ClassLabel for '{label_key}'. "
            "Provide classnames override or use a dataset with ClassLabel."
        )

    # best-effort fallback: infer class count from observed labels on a small prefix
    # (kept small to avoid scanning huge datasets)
    sample_split = next(iter(hf_ds.values()))
    mx = 0
    for i in range(min(10_000, len(sample_split))):
        mx = max(mx, int(sample_split[i][label_key]))
    return [str(i) for i in range(mx + 1)]


# ---------------------------
# Main entrypoint: build loaders
# ---------------------------


def build_vision_loaders(
    hf_ds: DatasetDict,
    hf_path: str | None = None,
    *,
    preprocess: Callable[[Any], torch.Tensor],
    ft_epochs: int,
    split_map: dict[str, str] | None = None,
    batch_size: int = 128,
    num_workers: int = 6,
    pin_memory: bool = True,
    val_fraction: float = 0.1,
    seed: int = 42,
    image_key: str = "image",
    label_key: str = "label",
    label_remap: LabelRemap = None,
    classnames_override: Sequence[str] | None = None,
    strict_classnames: bool = True,
    drop_last_train: bool = False,
) -> VisionLoaders:
    """
    Build train/val/test DataLoaders for HF image classification datasets.

    Rules:
      - requires train + test (or split_map mapping those to existing keys)
      - val is a reproducible random slice of test (val_fraction)
      - no contamination: val/test are disjoint subsets of the original test split
    """
    # resolve split keys
    if split_map is None:
        train_key = "train"
        test_key = "test"
    else:
        train_key = split_map["train"]
        test_key = split_map["test"]

    if train_key not in hf_ds or test_key not in hf_ds:
        raise KeyError(f"Missing required splits. have={list(hf_ds.keys())}, need={[train_key, test_key]}")

    # Create val/test from test using HF-native split (fast, reproducible)
    test_full = hf_ds[test_key]
    # test_size here means fraction of *kept* test, so keep 1 - val_fraction
    split = test_full.train_test_split(test_size=(1.0 - float(val_fraction)), seed=int(seed), shuffle=True)
    val_split = split["train"]
    test_split = split["test"]

    if hf_path == "tanganke/emnist_mnist":
        preprocess = emnist_fix_transform(preprocess)

    # Wrap with torch datasets
    train_ds = HFVisionDataset(
        hf_ds[train_key],
        transform=preprocess,
        image_key=image_key,
        label_key=label_key,
        label_remap=label_remap,
    )
    val_ds = HFVisionDataset(
        val_split,
        transform=preprocess,
        image_key=image_key,
        label_key=label_key,
        label_remap=label_remap,
    )
    test_ds = HFVisionDataset(
        test_split,
        transform=preprocess,
        image_key=image_key,
        label_key=label_key,
        label_remap=label_remap,
    )
    if classnames_override is not None:
        classnames_overrides = classnames_override
    else:
        if hf_path == "1aurent/PatchCamelyon":
            classnames_overrides = ["lymph node", "lymph node containing metastatic tumor tissue"]
        elif hf_path == "clip-benchmark/wds_fer2013":
            classnames_overrides = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
        # elif hf_path == "tanganke/kmnist":
        #     classnames_overrides = KMNIST_CLASSNAMES
        else:
            classnames_overrides = None

    classnames = extract_classnames(
        hf_ds,
        label_key=label_key,
        override=classnames_overrides,
        strict=strict_classnames,
    )

    # mirror torchvision convention
    train_ds.classes = classnames
    val_ds.classes = classnames
    test_ds.classes = classnames

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    sizes = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    meta = {
        "split_map": {"train": train_key, "test": test_key},
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "image_key": image_key,
        "label_key": label_key,
    }

    return VisionLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        classnames=classnames,
        ft_epochs=int(ft_epochs),
        sizes=sizes,
        meta=meta,
    )
