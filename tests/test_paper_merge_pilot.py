from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_paper_merge_pilot.py"
_SPEC = importlib.util.spec_from_file_location("paper_merge_pilot", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def test_full_manifest_has_complete_24_cell_factorial() -> None:
    rows = pilot.build_rows("full")
    assert len(rows) == 24
    assert len({row["run_id"] for row in rows}) == 24
    assert {row["merge_method"] for row in rows} == {"task_arithmetic", "tsv_merge", "isoc_merge"}
    assert {row["direction"] for row in rows} == {"extend", "shrink"}
    assert {row["lmc_mode"] for row in rows} == {"independent", "shared"}
    assert {row["transport"] for row in rows} == {"theseus", "bico"}


def test_full_configs_use_target_base_hierarchical_alpha_without_averaging() -> None:
    for row in pilot.build_rows("full"):
        config = pilot.build_config(row)
        assert "base_construction" not in config
        assert config["merge_mode"] == "rebase_then_merge"
        assert config["alpha_selection"] == "per_task"
        assert config["global_alpha_search"] is True
        assert config["block_extension_params"]["ridge_identity"] == 100.0
        assert config["block_extension_params"]["n_batches_act"] == 10
        assert config["method_params"]["num_batches"] == 10
        assert config["tasks"] == ",".join(pilot.TASKS)


def test_smoke_configs_are_one_task_fixed_alpha() -> None:
    for row in pilot.build_rows("smoke"):
        config = pilot.build_config(row)
        assert config["tasks"] == "Cars"
        assert config["alpha_search"] is False
        assert config["alpha"] == 0.3
        assert config["method_params"]["num_batches"] == 1
        assert config["block_extension_params"]["n_batches_act"] == 1
