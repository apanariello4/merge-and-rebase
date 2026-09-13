from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_brace_rebase_focused_center500_20260905.py"
AGGREGATOR = ROOT / "scripts/aggregate_brace_rebase_focused_center500_20260905.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_focused_matrix_counts_and_axes() -> None:
    campaign = _load(GENERATOR, "focused_center500_campaign")
    combined, controls = campaign.validate_spec()
    rows = [row for direction_rows in combined.values() for row in direction_rows]

    assert len(rows) == 48
    assert sum(row["reused"] for row in rows) == 12
    assert sum(not row["reused"] for row in rows) == 36
    assert len(controls) == 4
    assert {row["n_batches_act"] for row in rows} == {1, 2, 5}
    assert {row["ridge_identity"] for row in rows} == {200.0, 500.0}
    assert {row["ridge_weight"] for row in rows} == {1e-6}
    assert {row["center_acts"] for row in rows} == {False, True}
    assert {row["lmc_mode"] for row in rows} == {"shared"}
    assert {row["strategy"] for row in rows if row["direction"] == "extend"} == {"duplicate_per_weight"}
    assert {row["strategy"] for row in rows if row["direction"] == "shrink"} == {"interpolate_per_weight"}


def test_new_configs_use_target_zero_shot_contract_and_no_artifacts() -> None:
    campaign = _load(GENERATOR, "focused_center500_configs")
    combined, _ = campaign.validate_spec()
    new_rows = [row for rows in combined.values() for row in rows if not row["reused"]]

    for row in new_rows:
        config = campaign._config_for(row)
        assert config["method_params"]["center_acts"] is row["center_acts"]
        assert config["block_extension_params"]["ridge_identity"] == row["ridge_identity"]
        assert config["block_extension_params"]["n_batches_act"] in {1, 2, 5}
        assert config["campaign_metadata"]["baseline_contract"] == "native_target_zeroshot"
        assert config["alpha_selection"] == "per_task"
        assert config["alpha_search_split"] == "val"
        assert config["strict_load"] is True
        assert config["save_transported_artifacts"] is False
        assert "save_transported_tvs_dir" not in config


def _summary(label: str = "target_zeroshot") -> dict:
    return {
        "baseline_label": label,
        "tasks": ["A", "B"],
        "selected_baseline_alpha_by_task": {"A": 0.0, "B": 0.0},
        "run_logging": {"status": "success"},
        "test_results": {
            "per_task_baseline_accuracy": {"A": 0.5, "B": 0.25},
            "per_task_absolute_accuracy": {"A": 0.6, "B": 0.5},
            "per_task_normalized_accuracy_ratio": {"A": 1.2, "B": 2.0},
        },
    }


def test_aggregator_requires_native_target_zero_shot() -> None:
    aggregate = _load(AGGREGATOR, "focused_center500_aggregate")
    metrics = aggregate.validate_summary(_summary(), {"A", "B"})
    assert metrics["baseline"] == {"A": 0.5, "B": 0.25}

    with pytest.raises(ValueError, match="target_zeroshot"):
        aggregate.validate_summary(_summary("untransported"), {"A", "B"})


def test_aggregator_rejects_nonzero_baseline_alpha_and_bad_ratio() -> None:
    aggregate = _load(AGGREGATOR, "focused_center500_aggregate_validation")
    summary = _summary()
    summary["selected_baseline_alpha_by_task"]["A"] = 0.1
    with pytest.raises(ValueError, match="baseline alpha"):
        aggregate.validate_summary(summary, {"A", "B"})

    summary = _summary()
    summary["test_results"]["per_task_normalized_accuracy_ratio"]["A"] = 1.0
    with pytest.raises(ValueError, match="ratio mismatch"):
        aggregate.validate_summary(summary, {"A", "B"})


def test_aggregator_uses_union_of_corrected_and_control_fields() -> None:
    aggregate = _load(AGGREGATOR, "focused_center500_aggregate_fields")
    records = [{"run_id": "corrected", "ridge_identity": 500.0}, {"run_id": "control", "skip_correction": True}]
    assert aggregate._fieldnames(records) == ["run_id", "ridge_identity", "skip_correction"]
