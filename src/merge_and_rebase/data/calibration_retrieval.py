"""Task-data-free calibration source 1: retrieval of generic images by the task's class names.

No task image and no label is read. Every image of a generic pool (locally cached HF datasets OUTSIDE Vision8) is
scored with the TARGET base CLIP: cosine between its image embedding and the target's zero-shot class text
embeddings (the task's own templates, i.e. the eval head). ``select_class_balanced`` then picks ``n_images``
images round-robin over the classes, each class taking its next-best image not yet used.

Everything here is pure tensor logic except ``load_pool_images`` / ``embed_pool`` (HF datasets, GPU).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Pool datasets: (name, HF path, split, label/class source). None overlaps Vision8
#: (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN); ``assert_pool_disjoint_from`` enforces it.
POOL_SPECS: tuple[dict[str, str], ...] = (
    {"name": "tiny_imagenet", "path": "zh-plus/tiny-imagenet", "split": "valid"},
    {"name": "cifar100", "path": "tanganke/cifar100", "split": "test"},
    {"name": "stl10", "path": "tanganke/stl10", "split": "test"},
    {"name": "food101", "path": "ethz/food101", "split": "validation"},
    {"name": "flowers102", "path": "dpdl-benchmark/oxford_flowers102", "split": "test"},
    {"name": "oxford_pets", "path": "timm/oxford-iiit-pet", "split": "test"},
    {"name": "imagenet_r", "path": "axiong/imagenet-r", "split": "test"},
)

IMAGE_KEY = "image"


def parse_extra_pool(items: Sequence[str]) -> list[dict[str, str]]:
    """Parse ``name=hf_path:split[@cache_dir]`` strings (opt-in extra pool datasets) into pool specs."""
    out = []
    for item in items:
        name, _, rest = item.partition("=")
        rest, _, cache_dir = rest.partition("@")
        path, _, split = rest.rpartition(":")
        if not (name and path and split):
            raise ValueError(f"extra pool spec must be name=hf_path:split[@cache_dir], got {item!r}")
        spec = {"name": name, "path": path, "split": split}
        if cache_dir:
            spec["cache_dir"] = cache_dir
        out.append(spec)
    return out


def assert_pool_disjoint_from(pool: Sequence[Mapping[str, str]], forbidden_paths: Sequence[str]) -> None:
    """Refuse a pool that contains any of the ``forbidden_paths`` (the HF paths of the evaluated suite)."""
    banned = {str(p).lower() for p in forbidden_paths}
    clash = [spec["path"] for spec in pool if str(spec["path"]).lower() in banned]
    if clash:
        raise ValueError(f"retrieval pool overlaps the evaluated suite: {clash}")


def _seeded_perm(n: int, seed: int) -> list[int]:
    return torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()


def select_classes(best_score: torch.Tensor, n_images: int, seed: int) -> list[int]:
    """All classes when ``C <= n_images``; else the ``n_images`` classes with the highest best-score.

    ``best_score[c]`` = max over pool images of the cosine to class ``c``. Ties are broken by a seeded permutation.
    """
    n_classes = int(best_score.shape[0])
    if n_classes <= n_images:
        return list(range(n_classes))
    tie = torch.tensor(_seeded_perm(n_classes, seed), dtype=torch.float64)
    keys = sorted(range(n_classes), key=lambda c: (-float(best_score[c]), float(tie[c])))
    return sorted(keys[:n_images])


@dataclass
class Selection:
    """Picked pool rows in stored (batch) order plus how each was obtained."""

    pool_rows: list[int]  # row into the pool, one per image, in stored order
    class_ids: list[int]  # the class each image was retrieved for
    scores: list[float]  # cosine(image, class text) at pick time
    round_index: list[int]  # round-robin round (0 = class's best image)
    classes_used: list[int]


def select_class_balanced(
    scores: torch.Tensor,
    *,
    n_images: int,
    batch_size: int,
    seed: int,
) -> Selection:
    """Round-robin class-balanced selection from ``scores`` ``[N_pool, C]`` (cosine image-vs-class-text).

    Classes (after ``select_classes``) are visited in a seed-permuted order; in every round each class takes its
    highest-scoring pool image not yet used by any class, until ``n_images`` are picked (the last round may be
    partial, so class counts differ by at most one). Score ties are resolved by a seeded per-image key. The picked
    images are finally shuffled with the same seed and cut into batches of ``batch_size`` by the caller.
    """
    n_pool = int(scores.shape[0])
    if n_images > n_pool:
        raise ValueError(f"pool has {n_pool} images, {n_images} requested")
    classes = select_classes(scores.max(dim=0).values, n_images, seed)
    visit = [classes[i] for i in _seeded_perm(len(classes), seed)]
    tie_key = torch.tensor(_seeded_perm(n_pool, seed), dtype=torch.float64)
    ranked: dict[int, list[int]] = {}

    def ranking(c: int) -> list[int]:
        if c not in ranked:
            col = scores[:, c].to(torch.float64)
            # descending score, ties by the seeded key (lexsort: last key is primary)
            ranked[c] = np.lexsort((tie_key.numpy(), -col.numpy())).tolist()
        return ranked[c]

    used: set[int] = set()
    cursor = {c: 0 for c in visit}
    picks: list[tuple[int, int, float, int]] = []
    rnd = 0
    while len(picks) < n_images:
        progressed = False
        for c in visit:
            if len(picks) >= n_images:
                break
            order = ranking(c)
            i = cursor[c]
            while i < len(order) and order[i] in used:
                i += 1
            if i >= len(order):
                continue
            row = order[i]
            cursor[c] = i + 1
            used.add(row)
            picks.append((row, c, float(scores[row, c]), rnd))
            progressed = True
        if not progressed:
            raise RuntimeError("class-balanced selection ran out of pool images")
        rnd += 1
    shuffle = _seeded_perm(len(picks), seed)
    picks = [picks[i] for i in shuffle]
    return Selection(
        pool_rows=[p[0] for p in picks],
        class_ids=[p[1] for p in picks],
        scores=[p[2] for p in picks],
        round_index=[p[3] for p in picks],
        classes_used=classes,
    )


# --------------------------------------------------------------------------------------------------------------
# pool access (HF datasets, offline) and embedding cache
# --------------------------------------------------------------------------------------------------------------


def pool_fingerprint(pool: Sequence[Mapping[str, str]], sizes: Sequence[int], model: str, pretrained: str) -> str:
    payload = json.dumps({"pool": list(pool), "sizes": list(sizes), "model": model, "pretrained": pretrained}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_pool_datasets(pool: Sequence[Mapping[str, str]]) -> list[Any]:
    """Load each pool split from the local HF cache (never from the network)."""
    import os

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    # optional per-spec "cache_dir" (extra pool datasets may live outside the default HF cache)
    return [
        load_dataset(spec["path"], split=spec["split"], **({"cache_dir": spec["cache_dir"]} if spec.get("cache_dir") else {}))
        for spec in pool
    ]


class _PoolImages(torch.utils.data.Dataset):
    def __init__(self, ds: Any, preprocess: Callable) -> None:
        self.ds, self.preprocess = ds, preprocess

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.preprocess(self.ds[i][IMAGE_KEY].convert("RGB"))


@torch.no_grad()
def embed_pool(
    datasets_: Sequence[Any],
    encode_image: Callable[[torch.Tensor], torch.Tensor],
    preprocess: Callable,
    *,
    device: str,
    batch_size: int = 256,
    num_workers: int = 8,
    amp_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    """L2-normalised image embeddings ``[N_pool, D]`` (float32, pool order = datasets concatenated)."""
    out: list[torch.Tensor] = []
    for ds in datasets_:
        loader = torch.utils.data.DataLoader(
            _PoolImages(ds, preprocess), batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        for x in loader:
            x = x.to(device, non_blocking=True)
            if amp_dtype is not None and str(device).startswith("cuda"):
                with torch.autocast("cuda", dtype=amp_dtype):
                    f = encode_image(x)
            else:
                f = encode_image(x)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out, dim=0)


def load_or_build_pool_embeddings(
    cache_path: Path,
    pool: Sequence[Mapping[str, str]],
    datasets_: Sequence[Any],
    *,
    model: str,
    pretrained: str,
    build: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    """One cache file per target model; rebuilt when the pool, split sizes or model differ."""
    fp = pool_fingerprint(pool, [len(d) for d in datasets_], model, pretrained)
    if cache_path.is_file():
        obj = torch.load(cache_path, map_location="cpu", weights_only=True)
        if obj.get("fingerprint") == fp:
            return obj["embeddings"], True
    emb = build()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    torch.save({"embeddings": emb, "fingerprint": fp, "pool": list(pool), "model": model, "pretrained": pretrained}, tmp)
    tmp.replace(cache_path)
    return emb, False


def locate_pool_row(row: int, sizes: Sequence[int]) -> tuple[int, int]:
    """(pool dataset index, index inside it) of a concatenated pool row."""
    for k, size in enumerate(sizes):
        if row < size:
            return k, row
        row -= size
    raise IndexError("pool row out of range")
