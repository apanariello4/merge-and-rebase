from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _load("paper_pilot_analysis", "analyze_paper_merge_pilot.py")
locked = _load("locked_alpha_eval", "generate_locked_alpha_eval.py")


def test_analysis_tables_are_complete_and_mark_split_roles() -> None:
    tables = analysis.build_tables(
        repo_root=ROOT,
        manifest_path=Path("configs/final/paper_merge_pilot_lambda100_20260829/full/manifest.json"),
        aggregate_dir=Path("results/final/paper_merge_pilot_lambda100_20260829/aggregate"),
    )
    assert len(tables["run"]) == 24
    assert len(tables["comparison"]) == 12
    assert len(tables["task"]) == 192
    assert len(tables["alpha"]) == 216  # 8 task alphas + one final global alpha per run.
    assert {row["alpha_selection_split"] for row in tables["run"]} == {"val"}
    assert {row["reported_evaluation_split"] for row in tables["run"]} == {"test"}
    assert all(row["validation_selected"] and row["test_evaluated_at_locked_selection"] for row in tables["task"])


def test_locked_config_encodes_premerge_alphas_as_fixed_task_weights() -> None:
    manifest = json.loads(
        (ROOT / "configs/final/paper_merge_pilot_lambda100_20260829/full/manifest.json").read_text()
    )
    row = manifest["runs"][0]
    summary_path = ROOT / row["result_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    config = locked.build_locked_config(
        repo_root=ROOT,
        row=row,
        summary=summary,
        summary_path=summary_path,
        variant="full",
        output_result_dir="results/final/paper_merge_pilot_lambda100_20260829/analysis/locked_alpha_eval/full/test",
    )
    expected = [float(summary["per_task_premerge_alphas"][task]) for task in locked.TASKS]
    assert config["tasks"] == ",".join(locked.TASKS)
    assert config["merge_mode"] == "none"
    assert config["weights"] == expected
    assert config["alpha_search"] is False
    assert config["global_alpha_search"] is False
    assert config["alpha"] == 1.0
    assert config["locked_alpha_eval"]["task_alpha_selection_split"] == "val"
    assert config["locked_alpha_eval"]["evaluation_split"] == "test"


def test_locked_smoke_reduces_batches_but_keeps_all_tasks() -> None:
    manifest = json.loads(
        (ROOT / "configs/final/paper_merge_pilot_lambda100_20260829/full/manifest.json").read_text()
    )
    row = manifest["runs"][0]
    summary_path = ROOT / row["result_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    config = locked.build_locked_config(
        repo_root=ROOT,
        row=row,
        summary=summary,
        summary_path=summary_path,
        variant="smoke",
        output_result_dir="results/final/paper_merge_pilot_lambda100_20260829/analysis/locked_alpha_eval/smoke/test",
    )
    assert config["tasks"] == ",".join(locked.TASKS)
    assert config["method_params"]["num_batches"] == 1
    assert config["block_extension_params"]["n_batches_act"] == 1
    assert config["locked_alpha_eval"]["smoke_only"] is True


def test_generated_locked_alpha_manifests_are_unsubmitted_and_complete() -> None:
    for variant, expected_count in (("full", 24), ("smoke", 1)):
        manifest_path = (
            ROOT
            / "results/final/paper_merge_pilot_lambda100_20260829/analysis/locked_alpha_eval"
            / variant
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["submitted"] is False
        assert len(manifest["runs"]) == expected_count
        for row in manifest["runs"]:
            config = json.loads((ROOT / row["config_path"]).read_text())
            assert config["tasks"] == ",".join(locked.TASKS)
            assert len(config["weights"]) == len(locked.TASKS)
            assert config["merge_mode"] == "none"
            assert config["alpha_search"] is False
            assert config["global_alpha_search"] is False
            assert config["locked_alpha_eval"]["no_alpha_search"] is True
