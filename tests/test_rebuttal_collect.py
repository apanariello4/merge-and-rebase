"""Tests for scripts/rebuttal_cproj/collect_results.py (synthetic fixture + optional real seed-54 summary)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebuttal_cproj" / "collect_results.py"
SEED54 = Path("/data/merge_and_rebase_results/seqalpha2_20261102/runs/indepa2_D_shrink_seed54/indepa2_D_shrink_seed54.json")
spec = importlib.util.spec_from_file_location("collect_results", SCRIPT)
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
T = cr.TASKS


def summary(acc, base, *, alpha=None, fit=False, status="success", commit="abc"):
    d = {
        "baseline_label": "target_zeroshot",
        "test_results": {
            "per_task_rebased": {t: acc for t in T},
            "per_task_baseline": {t: base for t in T},
        },
        "selected_alpha_by_task": {t: alpha for t in T} if alpha is not None else {t: 1.0 for t in T},
        "run_logging": {
            "status": status,
            "metadata": {
                "code_fingerprint": {"git_commit": commit, "git_dirty": False},
                "resolved_config": {
                    "val_fraction": 0.1, "batch_size": 32, "alpha_min": 0.0, "alpha_max": 1.0, "alpha_step": 0.05,
                    "ariadne_params": {"num_batches": 10, "seed": 89},
                    "tuned_ckpts": {"Cars": "/ck/ViT-B-16/dc/cars/best.pt"},
                },
            },
        },
    }
    if fit:
        d["alignment_calibration_timings"] = {t: {"alignment_calibration_seconds": 10.0} for t in T}
        d["correction_fit_timings"] = {t: {"correction_fit_seconds": 90.0} for t in T}
    return d


def put(root, name, d):
    p = root / "runs" / name / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def test_partial_and_full_fixture(tmp_path):
    root = tmp_path / "R"
    put(root, "ariadne_extend_seed89", summary(0.8, 0.5, alpha=0.25, fit=True))
    put(root, "eval_ariadne_extend_alpha1_seed89", summary(0.7, 0.5))
    put(root, "eval_cproj_budget_extend_seed89", summary(0.6, 0.5, status="failed"))  # incomplete -> missing
    for t in T:
        name = f"ceiling_native_extend_{t}_seed89"
        p = root / "runs" / "ceiling_native_extend" / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary(0.9, 0.5)))
    ft = root / "cproj_ft" / "extend_budget_lr1e-5"
    ft.mkdir(parents=True)
    (ft / "DTD_cproj_calib.json").write_text(json.dumps({
        "config": {"time_budget_s": 100.0}, "timed_region": {"wall_seconds": 100.4}, "steps": 77, "git_commit": "ft1"}))
    rc = cr.main(["--root", str(root)])
    assert rc == 0
    res = json.loads((root / "RESULTS.json").read_text())["results"]["extend"]
    assert res["rows"]["base"]["mean"] == pytest.approx(50.0)
    assert res["rows"]["ariadne_val"]["mean"] == pytest.approx(80.0)
    assert res["rows"]["ariadne_val"]["alpha"]["Cars"] == 0.25
    assert res["rows"]["ariadne_val"]["fit_seconds_total"] == pytest.approx(800.0)
    assert res["gap_closed"]["ariadne_val"]["mean"] == pytest.approx(0.75)
    assert res["gap_closed"]["ariadne_alpha1"]["Cars"] == pytest.approx(0.5)
    assert res["rows"]["cproj_budget"]["mean"] is None and res["gap_closed"]["cproj_budget"]["mean"] is None
    assert res["ft_logs"]["budget"]["DTD"]["steps"] == 77
    md = (root / "RESULTS.md").read_text()
    assert "## Sources" in md and "## Missing files" in md
    assert "status='failed'" in md and "ViT-L-14" in md and cr.DASH in md
    # shrink direction entirely missing -> still renders
    assert json.loads((root / "RESULTS.json").read_text())["results"]["shrink"]["rows"]["base"]["mean"] is None


@pytest.mark.skipif(not SEED54.is_file(), reason="seed-54 summary not available")
def test_real_seed54_override(tmp_path):
    rc = cr.main(["--root", str(tmp_path / "R"), "--override", f"shrink:ariadne_val={SEED54}"])
    assert rc == 0
    res = json.loads((tmp_path / "R" / "RESULTS.json").read_text())["results"]["shrink"]
    d = json.loads(SEED54.read_text())
    want = 100 * d["test_results"]["avg_rebased"]
    assert res["rows"]["ariadne_val"]["mean"] == pytest.approx(want)
    assert res["rows"]["base"]["mean"] is not None
    assert res["rows"]["ariadne_val"]["fit_seconds_total"] > 0
    assert res["rows"]["ariadne_val"]["alpha"]["DTD"] == pytest.approx(0.2)
