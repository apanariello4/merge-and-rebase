from __future__ import annotations

import importlib.util
from pathlib import Path

from merge_and_rebase.eval.block_extension import BlockExtender

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_brace_rebase_extension_spread_mod_20260909.py"


def load():
    spec = importlib.util.spec_from_file_location("spread_mod_campaign", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spread_mod_extension_schedule_matches_declared_behavior():
    assert BlockExtender._build_duplication_schedule(12, 12, "bottom-top", "spread_mod") == [
        *range(11),
        0,
    ]
    assert BlockExtender._build_duplication_schedule(12, 12, "bottom-top", "spread") == list(range(12))


def test_campaign_is_exactly_the_twelve_cell_matrix():
    campaign = load()
    rows = campaign.validate_spec()
    assert len(rows) == 12
    assert {(row["method"], row["correction"], row["strategy"]) for row in rows} == {
        (method, correction, strategy)
        for method in campaign.METHODS
        for correction in campaign.CORRECTIONS
        for strategy in campaign.STRATEGIES
    }
    assert {row["extension_density"] for row in rows} == {"spread_mod"}


def test_configs_change_only_declared_factors_from_main_extension_recipe():
    campaign = load()
    for row in campaign.rows():
        cfg = campaign.config(row)
        params = cfg["block_extension_params"]
        assert cfg["source_clip_model"] == "ViT-B-16"
        assert cfg["target_clip_model"] == "ViT-L-14"
        assert cfg["method"] == row["method"]
        assert cfg["method_params"]["num_batches"] == 10
        assert cfg["method_params"]["seq_align"] == "interpolate2d"
        assert cfg["method_params"]["split_qkv"] is True
        assert cfg["method_params"]["center_acts"] is True
        assert params["extension_density"] == "spread_mod"
        assert params["extension_strategy"] == row["strategy"]
        assert params["insertion_order"] == "bottom-top"
        assert params["n_batches_act"] == 5
        assert params["ridge_identity"] == 100.0
        assert params["ridge_weight"] == 1e-6
        assert params["skip_correction"] is (row["correction"] == "skip")
        assert params["lmc_mode"] == row["lmc_mode"]
        assert cfg["alpha_selection"] == "per_task"
        assert cfg["alpha_min"] == 0.0
        assert cfg["alpha_max"] == 10.0
        assert cfg["alpha_step"] == 0.1
        assert cfg["alpha_patience"] == 5
        assert cfg["alpha_search_split"] == "val"
        assert cfg["seed"] == 89
        assert cfg["strict_load"] is True


def test_smoke_is_eurosat_bico_and_representative_interpolate_run():
    campaign = load()
    smoke = campaign.smoke_config()
    assert smoke["tasks"] == "EuroSAT"
    assert smoke["method"] == "bico"
    assert smoke["method_params"]["num_batches"] == 1
    assert smoke["block_extension_params"]["n_batches_act"] == 1
    assert smoke["block_extension_params"]["extension_density"] == "spread_mod"
    assert smoke["block_extension_params"]["extension_strategy"] == "interpolate_per_weight"
