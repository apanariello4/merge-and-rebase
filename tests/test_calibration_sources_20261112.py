"""Task-data-free calibration sources (precomputed images, class-name retrieval, model inversion). CPU only."""

from __future__ import annotations

import types

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.data.calibration_inversion import class_targets, invert, teacher_report
from merge_and_rebase.data.calibration_retrieval import (
    POOL_SPECS,
    assert_pool_disjoint_from,
    locate_pool_row,
    select_class_balanced,
)
from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.vision_rebase.context import _TaskContext
from merge_and_rebase.eval.vision_rebase.precomputed_calibration import (
    PrecomputedImageDataset,
    assert_same_preprocessing,
    build_precomputed_context,
    check_precomputed_supported,
    tensor_sha256,
)
from merge_and_rebase.rebase.methods._ariadne.capture import _stable_dataset_identity, paired_calibration
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig, parse_direct_residual_config


def _template() -> _TaskContext:
    return _TaskContext(loaders=None, source_loaders=None, classnames=["a", "b"], build_cfg_task=None, source_build_cfg_task=None)


def _write(tmp_path, task="DTD", n=12):
    images = torch.randn(n, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    torch.save({"images": images}, tmp_path / f"{task}.pt")
    return images


# ----------------------------------------------------------------------------------------------- precomputed path


def test_precomputed_loads_in_stored_order_and_feeds_identical_batches(tmp_path):
    images = _write(tmp_path)
    ctx, meta = build_precomputed_context(
        "DTD", cfg={"calibration_images_dir": str(tmp_path)}, num_batches=2, batch_size=4, template=_template()
    )
    # first num_batches*batch_size = 8 images, even though the seeded draw would permute them
    src, tgt, md = paired_calibration(ctx.source_loaders.train, ctx.loaders.train, num_batches=2, seed=54)
    assert md["indices"] == list(range(8))
    assert len(src) == len(tgt) == 2
    for k in range(2):
        assert torch.equal(src[k][0], images[4 * k : 4 * k + 4])
        assert torch.equal(src[k][0], tgt[k][0])
    assert meta["n_used"] == 8 and meta["n_stored"] == 12
    assert ctx.classnames == ["a", "b"]


def test_sha256_recorded_and_sensitive(tmp_path):
    images = _write(tmp_path)
    _, meta = build_precomputed_context(
        "DTD", cfg={"calibration_images_dir": str(tmp_path)}, num_batches=2, batch_size=4, template=_template()
    )
    assert meta["used_sha256"] == tensor_sha256(images[:8])
    assert meta["file_sha256"] == tensor_sha256(images)
    ds1, ds2 = PrecomputedImageDataset(images[:8], "x"), PrecomputedImageDataset(images[:8], "y")
    assert _stable_dataset_identity(ds1) != _stable_dataset_identity(ds2)
    changed = images.clone()
    changed[0, 0, 0, 0] += 1
    assert tensor_sha256(changed) != tensor_sha256(images)


def test_too_few_images_and_missing_file_are_refused(tmp_path):
    _write(tmp_path, n=5)
    with pytest.raises(ValueError, match="required"):
        build_precomputed_context(
            "DTD", cfg={"calibration_images_dir": str(tmp_path)}, num_batches=2, batch_size=4, template=_template()
        )
    with pytest.raises(FileNotFoundError):
        build_precomputed_context(
            "SVHN", cfg={"calibration_images_dir": str(tmp_path)}, num_batches=1, batch_size=4, template=_template()
        )


def test_default_paired_calibration_unchanged():
    """A dataset without the opt-in flag keeps the seeded permutation (the pre-existing behaviour)."""
    ds = TensorDataset(torch.arange(20).float().view(20, 1), torch.zeros(20, dtype=torch.long))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    _, _, md = paired_calibration(loader, loader, num_batches=2, seed=54)
    expected = torch.randperm(20, generator=torch.Generator().manual_seed(54)).tolist()[:8]
    assert md["indices"] == expected != list(range(8))


def test_config_option_default_and_validation():
    assert DirectResidualConfig().calibration_data == "task_local"
    assert parse_direct_residual_config({"calibration_data": "precomputed"}).calibration_data == "precomputed"
    with pytest.raises(ValueError, match="procrustes_source='activation'"):
        parse_direct_residual_config({"calibration_data": "precomputed", "procrustes_source": "gradient"})


def _resolved(calibration_data="precomputed", direct_fit=True, merge_mode="per_task", tasks=("DTD",)):
    ariadne_cfg = types.SimpleNamespace(calibration_data=calibration_data, merge_mode=merge_mode)
    return types.SimpleNamespace(direct_fit=direct_fit, ariadne_cfg=ariadne_cfg, tasks=list(tasks))


def test_check_precomputed_supported(tmp_path):
    _write(tmp_path)
    check_precomputed_supported({"calibration_images_dir": str(tmp_path)}, _resolved())
    check_precomputed_supported({}, _resolved("task_local"))  # default run: nothing required, nothing checked
    with pytest.raises(ValueError, match="requires"):
        check_precomputed_supported({}, _resolved())
    with pytest.raises(ValueError, match="only used with the Ariadne"):
        check_precomputed_supported({"calibration_images_dir": str(tmp_path)}, _resolved(direct_fit=False))
    with pytest.raises(ValueError, match="merge_in_source_then_fit"):
        check_precomputed_supported(
            {"calibration_images_dir": str(tmp_path)}, _resolved(merge_mode="merge_in_source_then_fit")
        )
    with pytest.raises(FileNotFoundError):
        check_precomputed_supported({"calibration_images_dir": str(tmp_path)}, _resolved(tasks=("DTD", "SVHN")))


def test_assert_same_preprocessing():
    def clf(mean):
        visual = types.SimpleNamespace(image_size=(224, 224), image_mean=mean, image_std=(1.0, 1.0, 1.0))
        return types.SimpleNamespace(model=types.SimpleNamespace(visual=visual))

    assert_same_preprocessing(clf((0.1, 0.2, 0.3)), clf((0.1, 0.2, 0.3)))
    with pytest.raises(ValueError, match="preprocessing differs"):
        assert_same_preprocessing(clf((0.1, 0.2, 0.3)), clf((0.5, 0.2, 0.3)))


# ----------------------------------------------------------------------------------------------- retrieval


def test_pool_is_disjoint_from_vision8():
    paths = [SUITES["vision8"].resolver(t)[0] for t in SUITES["vision8"].tasks]
    assert_pool_disjoint_from(POOL_SPECS, paths)
    with pytest.raises(ValueError, match="overlaps"):
        assert_pool_disjoint_from([{"name": "dtd", "path": "tanganke/dtd", "split": "test"}], paths)


def _scores(n_pool=200, n_classes=7, seed=1):
    return torch.randn(n_pool, n_classes, generator=torch.Generator().manual_seed(seed))


def test_selection_is_class_balanced_unique_and_deterministic():
    s = _scores()
    a = select_class_balanced(s, n_images=30, batch_size=10, seed=54)
    b = select_class_balanced(s, n_images=30, batch_size=10, seed=54)
    assert a == b
    assert len(set(a.pool_rows)) == 30
    counts = torch.bincount(torch.tensor(a.class_ids), minlength=7)
    assert int(counts.max() - counts.min()) <= 1 and int(counts.sum()) == 30
    c = select_class_balanced(s, n_images=30, batch_size=10, seed=7)
    assert c.pool_rows != a.pool_rows  # the seed drives order / batch mixing
    # a class's picks are its best images among those not taken by another class
    best_first = [(r, k) for r, k, rnd in zip(a.pool_rows, a.class_ids, a.round_index, strict=True) if rnd == 0]
    assert best_first
    for row, cls in best_first[:3]:
        assert float(s[row, cls]) >= float(s[:, cls].topk(8).values[-1])


def test_selection_takes_top_classes_when_more_classes_than_images():
    s = _scores(n_pool=100, n_classes=12)
    s[:, 3] += 10.0  # classes 3 and 9 have the highest best-scores
    s[:, 9] += 5.0
    sel = select_class_balanced(s, n_images=4, batch_size=2, seed=54)
    assert 3 in sel.classes_used and 9 in sel.classes_used and len(sel.classes_used) == 4
    assert sorted(sel.class_ids) == sorted(sel.classes_used)


def test_locate_pool_row():
    assert locate_pool_row(0, [3, 4]) == (0, 0)
    assert locate_pool_row(3, [3, 4]) == (1, 0)
    assert locate_pool_row(6, [3, 4]) == (1, 3)


# ----------------------------------------------------------------------------------------------- inversion


def test_class_targets_balanced_and_seeded():
    t, classes = class_targets(47, 320, 54)
    counts = torch.bincount(torch.tensor(t), minlength=47)
    assert int(counts.max() - counts.min()) <= 1 and len(t) == 320
    assert class_targets(47, 320, 54)[0] == t
    t2, classes2 = class_targets(397, 320, 54)
    assert len(set(t2)) == 320 == len(classes2)


def test_inversion_reduces_ce_on_toy_model():
    torch.manual_seed(0)
    head = torch.nn.Linear(3 * 8 * 8, 5)
    for p in head.parameters():
        p.requires_grad_(False)

    def logits_fn(x):
        return head(x.flatten(1))

    targets = [0, 1, 2, 3, 4, 0]
    images, trace = invert(
        logits_fn, targets, iters=60, seed=54, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), image_size=8,
        lr=0.1, jitter=0, warmup=5, micro_batch=4,
    )
    assert trace[-1] < 0.2 * trace[0]
    # valid pixel range after de-normalisation
    pix = images * 0.25 + 0.5
    assert float(pix.min()) >= -1e-6 and float(pix.max()) <= 1 + 1e-6
    rep = teacher_report(logits_fn, images, targets)
    assert rep["top1_rate"] == 1.0 and rep["mean_target_prob"] > 0.5
    again, _ = invert(
        logits_fn, targets, iters=60, seed=54, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), image_size=8,
        lr=0.1, jitter=0, warmup=5, micro_batch=4,
    )
    assert torch.equal(again, images)
