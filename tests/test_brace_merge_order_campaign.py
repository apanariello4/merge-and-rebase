from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_brace_merge_order_lambda100_20260906.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("brace_merge_order_campaign", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_has_exact_388_rows_and_pipeline_counts() -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    assert len(rows) == 388
    assert campaign._expected_counts(rows) == {
        "brace_transport_then_merge": 208,
        "brace_merge_then_transport": 96,
        "merge_then_brace_then_transport": 80,
        "native_target_merge_control": 4,
    }
    assert [row["array_index"] for row in rows] == list(range(388))


def test_manifest_has_unique_runs_and_result_paths() -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    assert len({row["run_id"] for row in rows}) == 388
    assert len({row["result_dir"] for row in rows}) == 388
    assert {row["direction"] for row in rows} == {"extend", "shrink"}
    assert {row["transport"] for row in rows if row["pipeline_mode"] != "native_target_merge_control"} == {
        "theseus",
        "bico",
    }
    assert {row["merger"] for row in rows} == {"task_arithmetic", "tsv_merge"}


def test_corrected_calibration_and_skip_axes_are_explicit() -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    corrected = [row for row in rows if row["correction_mode"] in {"independent", "shared"}]
    assert {row["calibration"] for row in corrected} == set(campaign.CORRECTED_CALIBRATIONS)
    assert {row["alpha_policy"] for row in rows if row["pipeline_mode"] == "brace_transport_then_merge"} == {
        "shared",
        "hierarchical",
    }
    assert all(
        row["alpha_policy"] == "shared"
        for row in rows
        if row["pipeline_mode"] != "brace_transport_then_merge"
    )
    assert not any(
        row["correction_mode"] == "skip" and row["calibration"] in {"tiny_40", "vision8_mix_40"}
        for row in rows
    )
    assert {row["calibration"] for row in rows if row["pipeline_mode"] == "brace_transport_then_merge" and row["correction_mode"] == "skip"} == {
        "task_local_5",
        "tiny_5",
        "vision8_mix_5",
    }


@pytest.mark.parametrize(
    "pipeline, expected_correction_modes",
    [
        ("brace_transport_then_merge", {"independent", "shared", "skip"}),
        ("brace_merge_then_transport", {"independent", "shared", "skip"}),
        ("merge_then_brace_then_transport", {"independent", "shared", "skip"}),
    ],
)
def test_pipeline_correction_modes_match_approved_scope(pipeline: str, expected_correction_modes: set[str]) -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    assert {row["correction_mode"] for row in rows if row["pipeline_mode"] == pipeline} == expected_correction_modes


def test_resolved_configs_inherit_lambda100_recipe_and_disable_heavy_artifacts() -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    for row in rows:
        config = campaign.build_config(row)
        assert config["pipeline_mode"] == row["pipeline_mode"]
        assert config["merge_mode"] == (
            "rebase_then_merge"
            if row["pipeline_mode"] == "native_target_merge_control"
            else row["pipeline_mode"]
        )
        assert config["merge_method"] == row["merger"]
        assert config["strict_load"] is True
        assert config["alpha_min"] == 0.0
        assert config["alpha_max"] == 10.0
        assert config["alpha_step"] == 0.1
        assert config["alpha_patience"] == 5
        assert config["save_transported_artifacts"] is False
        assert "save_transported_tvs_dir" not in config
        assert config["logging"]["project"] == campaign.WANDB_PROJECT
        assert config["logging"]["mode"] == "offline"
        if row["pipeline_mode"] == "native_target_merge_control":
            assert config["block_extension_enabled"] is False
            assert config["method"] == "theseus"
            assert config["native_target_tasks"] == list(campaign.TASKS)
        else:
            assert config["method"] == row["transport"]
            assert config["block_extension_params"]["ridge_identity"] == 100.0
            assert config["block_extension_params"]["ridge_weight"] == 1e-6
            assert config["block_extension_params"]["extension_strategy"] == row["extension_strategy"]
            assert config["method_params"]["num_batches"] == 10
            assert config["block_extension_params"]["n_batches_act"] == row["brace_batches"]
            assert "calibration_protocol" in config["block_extension_params"]
            calibration_dataset = config["block_extension_params"].get("calibration_dataset")
            if row["correction_mode"] == "skip" or not row["calibration"].startswith("tiny_"):
                assert calibration_dataset is None
            else:
                assert calibration_dataset == {
                    "path": "zh-plus/tiny-imagenet",
                    "split": "valid",
                    "max_samples": 2048,
                }
            if row["pipeline_mode"] == "brace_transport_then_merge":
                assert "transport_calibration_protocol" not in config
            elif row["calibration"].startswith("tiny_"):
                assert config["transport_calibration_protocol"] == "tiny_imagenet_10"
            else:
                assert config["transport_calibration_protocol"] == "vision8_mix_10"


def test_schema_uses_explicit_merge_and_calibration_fields() -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    for row in rows:
        config = campaign.build_config(row)
        assert config["merge_mode"] in {
            "brace_transport_then_merge",
            "brace_merge_then_transport",
            "merge_then_brace_then_transport",
            "rebase_then_merge",
        }
        if row["pipeline_mode"] != "native_target_merge_control":
            assert config["block_extension_params"]["n_batches_act"] == row["brace_batches"]
            expected_protocol = (
                "skip_correction"
                if row["correction_mode"] == "skip"
                else {
                    "task_local_5": "task_local_5",
                    "tiny_5": "tiny_imagenet_5",
                    "tiny_40": "tiny_imagenet_40",
                    "vision8_mix_5": "vision8_mix_5",
                    "vision8_mix_40": "vision8_mix_40",
                }[row["calibration"]]
            )
            assert config["block_extension_params"]["calibration_protocol"] == expected_protocol
        if row["pipeline_mode"] == "brace_transport_then_merge":
            assert "transport_calibration_protocol" not in config


def test_skip_configs_record_inactive_correction_factors() -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    for row in rows:
        config = campaign.build_config(row)
        metadata = config["campaign_metadata"]
        if row["correction_mode"] == "skip":
            assert config["block_extension_params"]["skip_correction"] is True
            assert metadata["inactive_factors"] == ["brace_correction", "brace_calibration"]
        elif row["pipeline_mode"] != "native_target_merge_control":
            assert config["block_extension_params"]["skip_correction"] is False
            assert metadata["inactive_factors"] == []


def test_tiny_and_vision8_calibration_specs_are_reproducible() -> None:
    campaign = _load_generator()
    tiny = campaign._calibration_spec("tiny_40", "brace_merge_then_transport")
    assert tiny["source"] == "tiny_imagenet"
    assert tiny["brace_batches"] == 40
    assert tiny["transport_batches"] == 10
    assert tiny["dataset"] == {"path": "zh-plus/tiny-imagenet", "split": "valid", "max_samples": 2048}
    mix = campaign._calibration_spec("vision8_mix_5", "merge_then_brace_then_transport")
    assert mix["source"] == "vision8_mix"
    assert mix["dataset"]["tasks"] == list(campaign.TASKS)
    assert mix["brace_batches"] == 5


def test_write_campaign_creates_manifest_and_configs_without_overwrite(tmp_path: Path) -> None:
    campaign = _load_generator()
    manifest_path = campaign.write_campaign(tmp_path / "generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["total"] == 388
    assert len(manifest["runs"]) == 388
    assert (tmp_path / "generated" / "brace_transport_then_merge" / "configs").is_dir()
    assert len(list((tmp_path / "generated").rglob("*.json"))) == 389
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        campaign.write_campaign(tmp_path / "generated")


def test_source_template_is_not_mutated_by_config_build() -> None:
    campaign = _load_generator()
    rows = campaign.validate_rows()
    row = next(row for row in rows if row["pipeline_mode"] == "brace_transport_then_merge")
    source_path = ROOT / row["source_config_path"]
    before = source_path.read_text(encoding="utf-8")
    config = campaign.build_config(row)
    config["block_extension_params"]["ridge_identity"] = 999.0
    config["logging"]["tags"].append("test-only")
    assert source_path.read_text(encoding="utf-8") == before
