"""calibration_augment: opt-in OpenCLIP-train views of the Ariadne calibration images (CPU, tiny fakes)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from merge_and_rebase.data.calibration_augment import (
    SeededAugmentDataset,
    build_augmented_calibration_pair,
    example_seed,
    transform_repr,
)
from merge_and_rebase.data.vision_loaders import HFVisionDataset
from merge_and_rebase.eval.vision_rebase.method_stages import DirectFitStage
from merge_and_rebase.rebase.methods._ariadne.capture import paired_calibration
from merge_and_rebase.rebase.methods._ariadne.config import (
    DirectResidualConfig,
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)


class _FakeSplit:
    _fingerprint = "fake-split"

    def __init__(self, n=40):
        rng = np.random.default_rng(0)
        self.rows = [
            {"image": Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)), "label": i % 5}
            for i in range(n)
        ]
        self.num_rows = n

    def __getitem__(self, i):
        return self.rows[i]


NORM = T.Normalize((0.5,) * 3, (0.5,) * 3)


def _eval_tf():
    return T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor(), NORM])


def _train_tf():
    return T.Compose([T.RandomResizedCrop(32, scale=(0.5, 1.0)), T.ToTensor(), NORM])


def _loader(split, tf):
    return DataLoader(HFVisionDataset(split, transform=tf), batch_size=4, shuffle=False, num_workers=0)


def _pair(seed=54, train_tf_target=None):
    split = _FakeSplit()
    src, tgt = _loader(split, _eval_tf()), _loader(split, _eval_tf())
    return split, build_augmented_calibration_pair(
        src,
        tgt,
        source_transform=_train_tf(),
        target_transform=train_tf_target if train_tf_target is not None else _train_tf(),
        seed=seed,
    ), (src, tgt)


def _batches(loader, seed=54, n=3):
    # a fresh DataLoader pair is drawn exactly as the fit does
    return paired_calibration(loader, loader, num_batches=n, seed=seed)


def test_views_identical_across_source_and_target_and_differ_from_eval():
    _, (src, tgt, meta), (src0, _) = _pair()
    a = _batches(src)
    b = _batches(tgt)
    for (xa, ya), (xb, yb) in zip(a[0], b[0], strict=True):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    eval_first = next(iter(DataLoader(src0.dataset, batch_size=4)))[0]
    assert not torch.equal(a[0][0][0], eval_first)
    assert meta["first_batch_sha256"]["source"] == meta["first_batch_sha256"]["target"]
    assert "RandomResizedCrop" in meta["transform_repr"] and " at 0x" not in meta["transform_repr"]


def test_same_indices_and_order_as_default_draw():
    split, (src, _, _), (src0, tgt0) = _pair()
    default = paired_calibration(src0, tgt0, num_batches=3, seed=54)
    aug = paired_calibration(src, src, num_batches=3, seed=54)
    assert default[2]["indices"] == aug[2]["indices"]
    for (_, y0), (_, y1) in zip(default[0], aug[0], strict=True):
        assert torch.equal(y0, y1)


def test_deterministic_across_builds_and_seed_sensitive():
    _, (_, _, m1), _ = _pair(54)
    _, (_, _, m2), _ = _pair(54)
    _, (_, _, m3), _ = _pair(55)
    assert m1["first_batch_sha256"] == m2["first_batch_sha256"]
    assert m1["first_batch_sha256"] != m3["first_batch_sha256"]


def test_index_dependent_and_global_rng_untouched():
    ds = SeededAugmentDataset(HFVisionDataset(_FakeSplit(), transform=_eval_tf()), _train_tf(), 54)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    x0, x0b, x1 = ds[0][0], ds[0][0], ds[1][0]
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(x0, x0b) and not torch.equal(x0, x1)
    assert example_seed(54, 3) != example_seed(54, 4) != example_seed(55, 4)


def test_identity_attributes_forwarded():
    split, (src, tgt, _), _ = _pair()
    from merge_and_rebase.rebase.methods._ariadne.capture import _dataset_identity

    assert _dataset_identity(src.dataset) == _dataset_identity(tgt.dataset) == ("fake-split", 40)


def test_unequal_train_transforms_raise():
    other = T.Compose([T.RandomResizedCrop(32, scale=(0.9, 1.0)), T.ToTensor(), NORM])
    with pytest.raises(ValueError, match="equal source and target train transforms"):
        _pair(train_tf_target=other)


def test_transform_repr_strips_addresses():
    assert transform_repr(lambda x: x).startswith("<function")
    assert " at 0x" not in transform_repr(lambda x: x)


def test_config_validation_and_serialization():
    assert DirectResidualConfig().calibration_augment == "none"
    assert parse_direct_residual_config({}).calibration_augment == "none"
    cfg = parse_direct_residual_config({"preset": "ariadne", "calibration_augment": "openclip_train"})
    assert cfg.calibration_augment == "openclip_train"
    assert direct_residual_config_dict(cfg)["calibration_augment"] == "openclip_train"
    # default runs serialize exactly as before (key omitted)
    assert "calibration_augment" not in direct_residual_config_dict(DirectResidualConfig())
    with pytest.raises(ValueError, match="calibration_augment must be"):
        parse_direct_residual_config({"calibration_augment": "randaug"})
    with pytest.raises(ValueError, match="requires calibration_data='task_local'"):
        parse_direct_residual_config({"calibration_augment": "openclip_train", "calibration_data": "tiny_imagenet"})
    with pytest.raises(ValueError):
        resolve_ariadne_decoder_config({"calibration_augment": "openclip_train"}, SimpleNamespace(mlp_down_name=None))


def test_stage_default_returns_loaders_untouched_and_records_nothing():
    split = _FakeSplit()
    s, t = _loader(split, _eval_tf()), _loader(split, _eval_tf())
    stage = SimpleNamespace(cfg=DirectResidualConfig(), record=SimpleNamespace(calibration_meta={"calibration_data": "task_local"}))
    env = SimpleNamespace(clf_source=SimpleNamespace(train_preprocess=_train_tf()), clf_target=SimpleNamespace(train_preprocess=_train_tf()))
    out = DirectFitStage._augmented_calibration_loaders(stage, env, s, t, "Cars")
    assert out[0] is s and out[1] is t
    assert stage.record.calibration_meta == {"calibration_data": "task_local"}


def test_stage_openclip_train_records_meta():
    split = _FakeSplit()
    s, t = _loader(split, _eval_tf()), _loader(split, _eval_tf())
    cfg = parse_direct_residual_config({"calibration_augment": "openclip_train", "seed": 54})
    stage = SimpleNamespace(cfg=cfg, record=SimpleNamespace(calibration_meta={"calibration_data": "task_local"}))
    env = SimpleNamespace(clf_source=SimpleNamespace(train_preprocess=_train_tf()), clf_target=SimpleNamespace(train_preprocess=_train_tf()))
    s2, t2 = DirectFitStage._augmented_calibration_loaders(stage, env, s, t, "Cars")
    assert s2 is not s and t2 is not t
    meta = stage.record.calibration_meta
    assert meta["calibration_augment"] == "openclip_train"
    h = meta["first_batch_sha256_by_task"]["Cars"]
    assert h["source"] == h["target"] and len(h["source"]) == 64
