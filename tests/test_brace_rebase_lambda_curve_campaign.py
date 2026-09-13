from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_brace_rebase_lambda_curve_20260905.py"
AGGREGATOR = ROOT / "scripts/aggregate_brace_rebase_lambda_curve_20260905.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lambda_curve_matrix_counts_and_fixed_axes() -> None:
    campaign = _load(GENERATOR, "lambda_curve_campaign")
    rows, controls = campaign.validate_spec()

    assert len(rows) == 24
    assert sum(row["reused"] for row in rows) == 8
    assert sum(not row["reused"] for row in rows) == 16
    assert len(controls) == 4
    assert {row["ridge_identity"] for row in rows} == {50.0, 100.0, 200.0, 300.0, 500.0, 1000.0}
    assert {row["ridge_weight"] for row in rows} == {1e-6}
    assert {row["n_batches_act"] for row in rows} == {5}
    assert {row["transport_batches"] for row in rows} == {10}
    assert {row["center_acts"] for row in rows} == {True}
    assert {row["lmc_mode"] for row in rows} == {"shared"}
    assert all(len(campaign.new_rows(direction)) == 8 for direction in campaign.DIRECTIONS)


def test_new_configs_preserve_selected_recipe_and_baseline_contract() -> None:
    campaign = _load(GENERATOR, "lambda_curve_configs")
    rows, _ = campaign.validate_spec()

    for row in (row for row in rows if not row["reused"]):
        config = campaign._config(row)
        assert config["method"] == row["method"]
        assert config["method_params"]["num_batches"] == 10
        assert config["method_params"]["center_acts"] is True
        assert config["block_extension_params"]["n_batches_act"] == 5
        assert config["block_extension_params"]["ridge_identity"] == row["ridge_identity"]
        assert config["block_extension_params"]["ridge_weight"] == 1e-6
        assert config["block_extension_params"]["extension_strategy"] == row["strategy"]
        assert config["block_extension_params"]["lmc_mode"] == "shared"
        assert config["alpha_min"] == 0.0
        assert config["alpha_max"] == 10.0
        assert config["alpha_step"] == 0.1
        assert config["alpha_selection"] == "per_task"
        assert config["alpha_search_split"] == "val"
        assert config["strict_load"] is True
        assert config["campaign_metadata"]["baseline_contract"] == "native_target_zeroshot"
        assert config["save_transported_artifacts"] is False
        assert "save_transported_tvs_dir" not in config


def _summary(label: str = "target_zeroshot") -> dict:
    tasks = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
    return {
        "baseline_label": label,
        "tasks": tasks,
        "selected_baseline_alpha_by_task": dict.fromkeys(tasks, 0.0),
        "run_logging": {"status": "success"},
        "test_results": {
            "per_task_baseline_accuracy": dict.fromkeys(tasks, 0.5),
            "per_task_absolute_accuracy": dict.fromkeys(tasks, 0.6),
            "per_task_normalized_accuracy_ratio": dict.fromkeys(tasks, 1.2),
        },
    }


def test_aggregator_enforces_target_zero_shot_contract() -> None:
    aggregate = _load(AGGREGATOR, "lambda_curve_aggregate")
    assert aggregate.validate_summary(_summary())["absolute"]["Cars"] == 0.6

    with pytest.raises(ValueError, match="target_zeroshot"):
        aggregate.validate_summary(_summary("untransported"))

    summary = _summary()
    summary["selected_baseline_alpha_by_task"]["Cars"] = 0.1
    with pytest.raises(ValueError, match="baseline alpha"):
        aggregate.validate_summary(summary)


def test_lambda_selection_uses_smaller_value_only_inside_tolerance() -> None:
    aggregate = _load(AGGREGATOR, "lambda_curve_selection")

    assert aggregate.elect_lambda({100.0: 0.8000, 200.0: 0.8015, 300.0: 0.8020}) == 200.0
    assert aggregate.elect_lambda({100.0: 0.8000, 200.0: 0.8011, 300.0: 0.8022}) == 300.0
