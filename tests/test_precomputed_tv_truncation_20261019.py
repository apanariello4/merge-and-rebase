"""Eval-only path: saved transported vectors (optionally rank-truncated) evaluated without refitting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from golden.test_main_golden import _ARIADNE, _EXT, _SEARCH, World, _base_cfg, run_main  # noqa: E402

from merge_and_rebase.eval.vision_rebase import method_stages  # noqa: E402
from merge_and_rebase.eval.vision_rebase.artifacts import truncate_tv_rank  # noqa: E402

_ARI = {"method": "ariadne", "method_params": {}, "ariadne_params": _ARIADNE}


def _tv(depth, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for i in range(depth):
        out[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"] = torch.randn(d, 2 * d, generator=g)
        out[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"] = torch.randn(d, generator=g)
    return out


# ---- (a) truncation unit tests ------------------------------------------------------------------------------


def test_energy_one_is_bit_identical():
    tv = _tv(4)
    out, rec = truncate_tv_rank(tv, energy=1.0, blocks="all", depth=4)
    for k, v in tv.items():
        assert torch.equal(out[k], v) and out[k].data_ptr() == v.data_ptr()
    assert all(r["k"] == r["min_dim"] and r["kept_energy_fraction"] == 1.0 for r in rec["keys"].values())
    assert len(rec["keys"]) == 4


def test_known_spectrum_k_rank_and_bias_untouched():
    s = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05], dtype=torch.float64)
    u, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(16, 8, dtype=torch.float64))
    w = (u * s) @ v.T
    key = "visual.transformer.resblocks.0.mlp.c_proj.weight"
    bias = torch.randn(8)
    tv = {key: w.float(), key.replace("weight", "bias"): bias}
    total = float((s**2).sum())
    # first two components hold (16 + 9) / total ~ 0.9 of the energy
    out, rec = truncate_tv_rank(tv, energy=25.0 / total, blocks="all", depth=1)
    assert rec["keys"][key]["k"] == 2
    assert torch.linalg.matrix_rank(out[key].double(), atol=1e-4).item() == 2
    assert out[key].dtype == torch.float32
    assert torch.equal(out[key.replace("weight", "bias")], bias)
    assert rec["keys"][key]["frobenius_after"] < rec["keys"][key]["frobenius_before"]
    assert abs(rec["keys"][key]["kept_energy_fraction"] - 25.0 / total) < 1e-6
    assert len(rec["keys"][key]["sha256"]) == 64


@pytest.mark.parametrize(("depth", "selected"), [(24, range(16, 24)), (12, range(8, 12))])
def test_last_third_blocks(depth, selected):
    tv = _tv(depth, d=4)
    out, rec = truncate_tv_rank(tv, energy=0.5, blocks="last_third", depth=depth)
    got = {int(k.split(".")[3]) for k in rec["keys"]}
    assert got == set(selected)
    for k, v in tv.items():
        if k.endswith(".bias") or int(k.split(".")[3]) not in selected:
            assert torch.equal(out[k], v)


def test_truncate_rejects_bad_energy():
    with pytest.raises(ValueError):
        truncate_tv_rank(_tv(1), energy=0.0, blocks="all", depth=1)


# ---- (b) end to end on the toy model --------------------------------------------------------------------------


def _fit_cfg(world, root):
    return _base_cfg(
        world, root, **_ARI, **_SEARCH, alpha_selection="per_task", save_transported_tvs_dir=str(root / "tvs")
    )


def test_precomputed_run_equals_fit_run(tmp_path, monkeypatch):
    world = World(**_EXT)
    fit = run_main(_fit_cfg(world, tmp_path / "fit"), tmp_path / "fit", monkeypatch, world=world)
    calls = []
    real = method_stages._run_direct_residual_fit
    monkeypatch.setattr(method_stages, "_run_direct_residual_fit", lambda *a, **k: calls.append(1) or real(*a, **k))
    cfg = _base_cfg(
        world,
        tmp_path / "pre",
        **_ARI,
        **_SEARCH,
        alpha_selection="per_task",
        precomputed_tv_dir=str(tmp_path / "fit" / "tvs"),
        tv_rank_truncation={"energy": 1.0, "blocks": "all"},
    )
    pre = run_main(cfg, tmp_path / "pre", monkeypatch, world=world)
    assert calls == []  # fitting is fully skipped
    for key in ("selected_alpha_by_task", "per_task_absolute_accuracy", "per_task_baseline_accuracy"):
        a, b = _find(fit.summary, key), _find(pre.summary, key)
        assert a is not None and a == b, key
    assert _find(fit.summary, "selected_alpha_by_task")
    prov = pre.summary["precomputed_tvs"]
    assert set(prov["files_by_task"]) == set(world.tasks)
    assert all(len(v["file_sha256"]) == 64 for v in prov["files_by_task"].values())
    assert pre.summary["tv_rank_truncation"]["energy"] == 1.0
    for t in world.tasks:
        saved = torch.load(next((tmp_path / "fit" / "tvs").glob(f"{t}_*_transported_native.pt")), weights_only=True)
        from merge_and_rebase.eval.vision_rebase.artifacts import _state_dict_sha256

        assert prov["transported_delta_sha256_by_task"][t] == _state_dict_sha256(saved)
    assert "precomputed_tvs" not in fit.summary


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def test_truncated_run_records_ranks(tmp_path, monkeypatch):
    world = World(**_EXT)
    run_main(_fit_cfg(world, tmp_path / "fit"), tmp_path / "fit", monkeypatch, world=world)
    cfg = _base_cfg(
        world,
        tmp_path / "pre",
        **_ARI,
        alpha=1.0,
        precomputed_tv_dir=str(tmp_path / "fit" / "tvs"),
        tv_rank_truncation={"energy": 0.5, "blocks": "last_third"},
    )
    pre = run_main(cfg, tmp_path / "pre", monkeypatch, world=world)
    by_task = pre.summary["tv_rank_truncation"]["by_task"]
    assert set(by_task) == set(world.tasks)
    for keys in by_task.values():
        assert keys and all(r["k"] <= r["min_dim"] for r in keys.values())
        assert json.dumps(keys)


# ---- (c) config rejections -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"precomputed_tv_dir": "/x", "load_direct_residual_tvs_dir": "/y"}, "cannot be combined with load_direct"),
        ({"precomputed_tv_dir": "/x", "save_transported_tvs_dir": "/y"}, "cannot save transported artifacts"),
        ({"tv_rank_truncation": {"energy": 0.5}}, "require precomputed_tv_dir"),
        ({"precomputed_tv_dir": "/x", "tv_rank_truncation": {"energy": 1.5}}, r"energy must be in \(0, 1\]"),
        ({"precomputed_tv_dir": "/x", "tv_rank_truncation": {"energy": 0.5, "blocks": "first"}}, "blocks must be"),
        ({"precomputed_tv_dir": "/x", "method": "theseus"}, "require the Ariadne"),
    ],
)
def test_config_rejections(tmp_path, monkeypatch, extra, message):
    world = World(**_EXT)
    cfg = _base_cfg(world, tmp_path, **{**_ARI, **extra})
    with pytest.raises(ValueError, match=message):
        run_main(cfg, tmp_path, monkeypatch, world=world)
