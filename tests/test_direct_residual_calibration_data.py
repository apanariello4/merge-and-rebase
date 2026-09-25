"""DirectResidualConfig.calibration_data: task-independent calibration images.

The default ("task_local") path is pinned by the existing golden-hash suites
(component coverage, endpoint target, merge-in-source, dispatch, streaming);
this module covers the parser contract and the two task-independent loader
sources: the balanced Vision8 plan and the direct HF (Tiny-ImageNet) split.
Both must satisfy ``paired_calibration``'s source/target identity check.
"""

from __future__ import annotations

import collections
import os

import pytest
import torch
from torch.utils.data import TensorDataset

from merge_and_rebase.data.balanced_calibration import (
    Vision8TaskContext,
    build_balanced_vision8_calibration_loaders,
)
from merge_and_rebase.eval.direct_residual import DirectResidualConfig, parse_direct_residual_config
from merge_and_rebase.eval.target_informed_runtime import _dataset_identity, paired_calibration

# ---- parser -----------------------------------------------------------------


def test_default_is_task_local():
    assert DirectResidualConfig().calibration_data == "task_local"
    assert parse_direct_residual_config({}).calibration_data == "task_local"


@pytest.mark.parametrize("value", ["task_local", "tiny_imagenet", "vision8_mix"])
def test_accepts_known_values(value):
    assert parse_direct_residual_config({"calibration_data": value}).calibration_data == value


def test_rejects_unknown_value():
    with pytest.raises(ValueError, match="calibration_data must be"):
        parse_direct_residual_config({"calibration_data": "imagenet"})


@pytest.mark.parametrize("value", ["tiny_imagenet", "vision8_mix"])
def test_task_independent_calibration_requires_activation_procrustes(value):
    with pytest.raises(ValueError, match="requires procrustes_source='activation'"):
        parse_direct_residual_config({"calibration_data": value, "procrustes_source": "gradient"})


def test_task_local_still_allows_gradient_procrustes():
    cfg = parse_direct_residual_config({"calibration_data": "task_local", "procrustes_source": "gradient"})
    assert cfg.procrustes_source == "gradient"


# ---- balanced Vision8 plan --------------------------------------------------


def _task_datasets(n_tasks=8, size=40):
    """Per task: a source and a target view of the SAME images (distinct objects)."""
    contexts = {}
    for t in range(n_tasks):
        images = torch.arange(size, dtype=torch.float32).reshape(-1, 1) + 1000 * t
        labels = torch.arange(size) % 4
        contexts[f"task{t}"] = Vision8TaskContext(
            source_dataset=TensorDataset(images, labels),
            target_dataset=TensorDataset(images.clone(), labels.clone()),
            classnames=[f"c{k}" for k in range(4)],
        )
    return contexts


def _balanced(seed, n_batches=3, batch_size=16):
    return build_balanced_vision8_calibration_loaders(
        _task_datasets(), n_batches=n_batches, batch_size=batch_size, seed=seed
    )


def test_balanced_views_share_identity_and_pass_paired_calibration():
    built = _balanced(seed=33)
    src, tgt = built.source_loaders, built.target_loaders
    assert _dataset_identity(src.dataset) == _dataset_identity(tgt.dataset)
    # Direct Residual draws num_batches * batch_size = the whole balanced set.
    source_batches, target_batches, meta = paired_calibration(src, tgt, num_batches=3, seed=33)
    assert meta["actual_batches"] == 3 and len(source_batches) == len(target_batches) == 3
    for (xs, ys), (xt, yt) in zip(source_batches, target_batches, strict=True):
        assert torch.equal(xs, xt) and torch.equal(ys, yt)


def test_balanced_plan_is_exactly_balanced_and_seed_deterministic():
    a, b, c = _balanced(seed=33), _balanced(seed=33), _balanced(seed=54)
    ids_a = a.source_loaders.dataset.sample_ids
    assert ids_a == b.source_loaders.dataset.sample_ids == b.target_loaders.dataset.sample_ids
    assert ids_a != c.source_loaders.dataset.sample_ids
    per_task = collections.Counter(sid.split(":")[0] for sid in ids_a)
    # 3 batches x 16 images / 8 tasks = 6 images per task, no duplicates.
    assert set(per_task.values()) == {6}
    assert len(set(ids_a)) == len(ids_a) == 48


def test_identity_check_still_rejects_different_plans():
    a, c = _balanced(seed=33), _balanced(seed=54)
    with pytest.raises(ValueError, match="identities or ordering do not match"):
        paired_calibration(a.source_loaders, c.target_loaders, num_batches=3, seed=33)


# ---- direct HF split (Tiny-ImageNet) ----------------------------------------

_HF_CACHE = "/leonardo_scratch/large/userexternal/frinaldi/.hf_cache/datasets"


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_HF_CACHE, "zh-plus___tiny-imagenet")),
    reason="Tiny-ImageNet not in the offline HF cache",
)
def test_tiny_imagenet_views_share_identity_and_are_seed_deterministic(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", _HF_CACHE)
    from torchvision import transforms

    from merge_and_rebase.data.vision_loaders import build_vision_calibration_loader
    from merge_and_rebase.eval.vision_rebase import DIRECT_RESIDUAL_TINY_IMAGENET_SPEC

    def loader(size):
        return build_vision_calibration_loader(
            DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
            resolver=lambda name: (_ for _ in ()).throw(KeyError(name)),
            preprocess=transforms.Compose([transforms.Resize(size), transforms.ToTensor()]),
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )

    # Two independent loads under different preprocessors, as for the two models.
    src, tgt = loader(32), loader(24)
    assert len(src.dataset) == 10000
    assert _dataset_identity(src.dataset) == _dataset_identity(tgt.dataset)
    first_src, first_tgt, first_meta = paired_calibration(src, tgt, num_batches=2, seed=33)
    again_src, _again_tgt, again_meta = paired_calibration(src, tgt, num_batches=2, seed=33)
    assert first_meta["indices"] == again_meta["indices"] and len(first_meta["indices"]) == 8
    assert first_meta["actual_batches"] == 2
    for (xa, ya), (xb, yb), (_xt, yt) in zip(first_src, again_src, first_tgt, strict=True):
        assert torch.equal(xa, xb) and torch.equal(ya, yb) and torch.equal(ya, yt)
