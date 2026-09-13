from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_brace_rebase_strategy_centered_20260907.py"


def load():
    spec = importlib.util.spec_from_file_location("strategy_centered", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_centered_confirmation_is_exactly_matched_eight_cell_matrix():
    campaign = load()
    rows = campaign.validate_spec()
    assert len(rows) == 8
    assert {(row["direction"], row["method"], row["strategy"]) for row in rows} == {
        (direction, method, strategy)
        for direction in campaign.DIRECTIONS for method in campaign.METHODS for strategy in campaign.STRATEGIES
    }
    assert {row["lmc_mode"] for row in rows} == {"shared"}
    assert {row["center_acts"] for row in rows} == {True}
    assert {row["n_batches_act"] for row in rows} == {5}
    assert {row["transport_batches"] for row in rows} == {10}
    assert {row["ridge_identity"] for row in rows} == {100.0}
    assert {row["ridge_weight"] for row in rows} == {1e-6}


def test_confirmation_config_changes_only_structural_strategy_from_canonical_recipe():
    campaign = load()
    for row in campaign.rows():
        cfg = campaign.config(row)
        assert cfg["method"] == row["method"]
        assert cfg["method_params"]["num_batches"] == 10
        assert cfg["method_params"]["center_acts"] is True
        assert cfg["block_extension_params"]["extension_strategy"] == row["strategy"]
        assert cfg["block_extension_params"]["lmc_mode"] == "shared"
        assert cfg["block_extension_params"]["n_batches_act"] == 5
        assert cfg["block_extension_params"]["ridge_identity"] == 100.0
        assert cfg["block_extension_params"]["ridge_weight"] == 1e-6
        assert cfg["alpha_selection"] == "per_task"
        assert cfg["alpha_search_split"] == "val"
        assert cfg["strict_load"] is True


def test_smoke_is_representative_noncanonical_extension_strategy():
    campaign = load()
    smoke = campaign.smoke_config()
    assert smoke["tasks"] == "EuroSAT"
    assert smoke["method"] == "bico"
    assert smoke["method_params"]["num_batches"] == 1
    assert smoke["method_params"]["center_acts"] is True
    assert smoke["block_extension_params"]["extension_strategy"] == "interpolate_per_weight"
    assert smoke["block_extension_params"]["ridge_identity"] == 100.0
