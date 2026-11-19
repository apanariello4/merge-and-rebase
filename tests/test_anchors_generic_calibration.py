"""calibration_source='generic' / 'precomputed' of the anchors method (CPU, toy models of test_anchors_method)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from test_anchors_method import _cfg, _loader, _setup

from merge_and_rebase.rebase.methods._ariadne.capture import paired_calibration
from merge_and_rebase.rebase.methods.anchors import AnchorsRebase, parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.method import (
    generic_calibration_loaders,
    precomputed_calibration_loaders,
)

BIG = _loader(24, seed=202)  # 12 batches of 2: frame draw 4 batches, calibration 3, spare 5


def _run(cfg, setup, *, task_data=None, task_name="T"):
    source_base, source_ft, target_base, data, pairing, target_sd = setup
    data = task_data if task_data is not None else data
    return AnchorsRebase().prepare(
        source_base_model=source_base, source_ft_model=source_ft, target_model=target_base, target_base_sd=target_sd,
        source_loader=data, target_loader=data, pairing=pairing, config=cfg, device="cpu",
        source_anchor_loader=BIG, target_anchor_loader=BIG, source_model_id="A", target_model_id="B",
        task_name=task_name,
    )


def _same(a, b):
    assert set(a) == set(b)
    return all(torch.equal(a[k], b[k]) for k in a)


def test_default_is_task_and_config_dict_unchanged():
    cfg = _cfg()
    assert cfg.calibration_source == "task" and cfg.calibration_images is None
    assert "calibration_source" not in cfg.to_dict() and "calibration_images" not in cfg.to_dict()
    setup = _setup()
    a = _run(cfg, setup)
    b = _run(_cfg(calibration_source="task"), setup)
    assert _same(a.unit_task_vector, b.unit_task_vector)
    assert "calibration" not in a.extra["anchors"]
    assert a.extra["calibration"]["split"] == "val"


def test_generic_ignores_task_images_and_records_identity():
    setup = _setup()
    cfg = _cfg(calibration_source="generic", anchor_pool="tiny_imagenet")
    g1 = _run(cfg, setup)
    g2 = _run(cfg, setup, task_data=_loader(6, seed=999))  # other task images: no effect
    assert _same(g1.unit_task_vector, g2.unit_task_vector)
    t = _run(_cfg(), setup)
    assert not _same(g1.unit_task_vector, t.unit_task_vector)
    info = g1.extra["anchors"]["calibration"]
    assert info["calibration_source"] == "generic" and info["n_images"] == 3 * 2 and info["disjoint_from_frame_draw"]
    cal = g1.extra["calibration"]
    assert cal["split"] == "anchor_pool" and cal["calibration_indices_sha256"] == info["calibration_indices_sha256"]
    assert cal["dataset_identity"].startswith("Subset(")
    assert g1.extra["anchors"]["config"]["calibration_source"] == "generic"


def test_generic_calibration_images_are_disjoint_from_frame_preserve_and_refine_rows():
    cfg = _cfg(calibration_source="generic", anchor_pool="tiny_imagenet", seed=5, pool_seed=3)
    src, tgt, info = generic_calibration_loaders(cfg, BIG, BIG)
    cal = list(src.dataset.indices)
    assert cal == list(tgt.dataset.indices) and len(cal) == len(set(cal)) == 6
    # the frame's draw (registration + anchors, ridge_preserve G = its first anchor_batches, refine keep rows and
    # the refine holdout = its batches) is paired_calibration(num_batches=anchor+registration, seed=pool_seed)
    _, _, meta = paired_calibration(BIG, BIG, num_batches=cfg.anchor_batches + cfg.registration_batches, seed=3)
    assert len(meta["indices"]) == 8 and not set(cal) & set(meta["indices"])
    # the images themselves differ (rows of the dataset tensors)
    pool_imgs = BIG.dataset.tensors[0]
    for i in cal:
        for j in meta["indices"]:
            assert not torch.equal(pool_imgs[i], pool_imgs[j])
    # deterministic in the calibration seed, and varies with it
    again = generic_calibration_loaders(cfg, BIG, BIG)[2]["calibration_indices_sha256"]
    other = generic_calibration_loaders(dataclasses.replace(cfg, seed=6), BIG, BIG)[2]["calibration_indices_sha256"]
    assert again == info["calibration_indices_sha256"] != other
    # the batches the capture sees are the pool images, not task images
    s_b, _, _ = paired_calibration(src, tgt, num_batches=3, seed=cfg.seed)
    assert {tuple(x.flatten().tolist()) for b in s_b for x in b[0]} == {tuple(pool_imgs[i].flatten().tolist()) for i in cal}


def test_generic_works_with_tracking_o2():
    setup = _setup()
    out = _run(_cfg(calibration_source="generic", anchor_pool="tiny_imagenet", depth_fit="tracking",
                    synthesis="ridge_preserve", rho_g=0.1, preserve_inputs="updated", preserve_tokens=16), setup)
    assert all(torch.isfinite(v).all() for v in out.unit_task_vector.values())


def test_generic_pool_too_small_is_an_error():
    cfg = _cfg(calibration_source="generic", anchor_pool="tiny_imagenet", num_batches=9)
    with pytest.raises(ValueError, match="calibration needs"):
        generic_calibration_loaders(cfg, BIG, BIG)


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"calibration_source": "bogus"}, "calibration_source must be"),
        ({"calibration_source": "generic", "map_rows": "task"}, "map_rows='frame'"),
        ({"calibration_source": "generic", "anchor_pool": "vision8_mix"}, "tiny_imagenet"),
        ({"calibration_source": "generic", "anchor_pool": "wikitext103"}, "vision-only"),
        ({"calibration_source": "generic", "calibration_images": "/x"}, "calibration_images"),
        ({"calibration_source": "precomputed"}, "calibration_images"),
        ({"calibration_images": "/x"}, "calibration_images"),
    ],
)
def test_parse_rejects_unsupported_combinations(kw, msg):
    base = {"anchor_pool": "tiny_imagenet", "synthesis": "ridge"}
    with pytest.raises(ValueError, match=msg):
        parse_anchors_config({**base, **kw})


def test_parse_rejects_operator_library_and_accepts_good_configs():
    with pytest.raises(ValueError, match="operator"):
        parse_anchors_config({"anchor_pool": "tiny_imagenet", "calibration_source": "generic", "library": "operator",
                              "frame_map": "gpa"})
    assert parse_anchors_config({"anchor_pool": "tiny_imagenet", "calibration_source": "generic"}).calibration_source == "generic"
    cfg = parse_anchors_config({"anchor_pool": "tiny_imagenet", "calibration_source": "precomputed",
                                "calibration_images": "/x"})
    assert cfg.to_dict()["calibration_images"] == "/x"


def test_precomputed_loaders_serve_stored_order_per_task(tmp_path):
    imgs = torch.randn(10, 3, 2, 2)
    torch.save({"images": imgs, "source": "retrieval"}, tmp_path / "DTD.pt")
    cfg = _cfg(calibration_source="precomputed", calibration_images=str(tmp_path), num_batches=3)
    src, tgt, info = precomputed_calibration_loaders(cfg, "DTD", 2)
    assert info["n_used"] == 6 and info["task"] == "DTD" and len(info["used_sha256"]) == 64
    s_b, t_b, _ = paired_calibration(src, tgt, num_batches=3, seed=cfg.seed)  # seed given, stored order kept
    assert torch.equal(torch.cat([b[0] for b in s_b]), imgs[:6])
    assert torch.equal(torch.cat([b[0] for b in t_b]), imgs[:6])
    with pytest.raises(FileNotFoundError):
        precomputed_calibration_loaders(cfg, "GTSRB", 2)
    with pytest.raises(ValueError, match="are required"):
        precomputed_calibration_loaders(dataclasses.replace(cfg, num_batches=6), "DTD", 2)
    with pytest.raises(ValueError, match="task name"):
        precomputed_calibration_loaders(cfg, None, 2)
