"""LLM rebase: saving the unscaled transported task vector (``save_transported_tvs_dir``)."""

import json

import pytest
import torch

from merge_and_rebase.eval.llm_rebase.artifacts import save_merged_state, save_transported_task_vector
from merge_and_rebase.eval.llm_rebase.summary import assemble_nli_summary


class _Planner:
    def search_summary(self):
        return {}


def _inputs():
    g = torch.Generator().manual_seed(0)
    base = {"a": torch.randn(4, 3, generator=g), "b": torch.randn(5, generator=g)}
    tau = {"a": torch.randn(4, 3, generator=g), "b": torch.randn(5, generator=g)}
    return base, tau


def _save(cfg, tau, alpha=0.7):
    return save_transported_task_vector(
        cfg, tau, method_name="m", best_alpha=alpha, alpha_curve=[{"alpha": alpha, "score": 0.5}]
    )


def test_off_by_default_writes_nothing(tmp_path):
    _, tau = _inputs()
    assert _save({}, tau) is None
    assert list(tmp_path.iterdir()) == []


def test_artifacts_without_dir_raises():
    _, tau = _inputs()
    with pytest.raises(ValueError):
        _save({"save_transported_artifacts": True}, tau)


def test_round_trip_sidecar_and_hash_stable(tmp_path):
    _, tau = _inputs()
    recs = [_save({"save_transported_tvs_dir": str(tmp_path / d), "source_base_ckpt": "s"}, tau) for d in "xy"]
    assert recs[0]["sha256"] == recs[1]["sha256"]
    loaded = torch.load(recs[0]["path"])
    assert loaded.keys() == tau.keys()
    assert all(torch.equal(loaded[k], tau[k]) for k in tau)
    meta = json.loads(open(recs[0]["sidecar"]).read())
    assert meta["vector_sha256"] == recs[0]["sha256"]
    assert meta["best_alpha"] == 0.7 and meta["alpha_curve"][0]["score"] == 0.5
    assert meta["refs"]["source_base_ckpt"] == "s" and "git_commit" in meta
    with pytest.raises(FileExistsError):
        _save({"save_transported_tvs_dir": str(tmp_path / "x")}, tau)


def test_base_plus_alpha_tau_equals_merged_state(tmp_path):
    base, tau = _inputs()
    rec = _save({"save_transported_tvs_dir": str(tmp_path)}, tau, alpha=0.3)
    save_merged_state(tmp_path / "m.pt", tau, 0.3, base, message="x")
    merged = torch.load(tmp_path / "m.pt")
    loaded = torch.load(rec["path"])
    for k in base:
        assert torch.equal(base[k] + 0.3 * loaded[k], merged[k])


def test_summary_key_additive():
    kw = dict(
        ignored_block_extension_fields=None,
        calibration_provenance=None,
        method_name="m",
        best_alpha=1.0,
        task_data=[],
        delta_stats={},
        task_vector_report={},
        search_planner=_Planner(),
        search_results=[],
        best_vals=[],
        saved_merged_path=None,
    )
    assert "transported_tv" not in assemble_nli_summary(**kw)
    assert assemble_nli_summary(**kw, transported_tv={"path": "p"})["transported_tv"] == {"path": "p"}
