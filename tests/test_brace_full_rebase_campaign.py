from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_brace_full_rebase_vision8_20260905.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("brace_full_rebase_campaign", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_rebase_matrix_is_complete_and_low_storage() -> None:
    campaign = _load_generator()
    by_direction, controls = campaign.validate_spec()

    assert campaign.CORRECTED_TOTAL == 384
    assert campaign.CORRECTED_PER_DIRECTION == 192
    assert len(controls) == 4
    assert {len(rows) for rows in by_direction.values()} == {192}

    corrected = [item for rows in by_direction.values() for item in rows]
    configs = [config for _, config in corrected]
    assert {config["block_extension_params"]["ridge_identity"] for config in configs} == {
        0.1,
        10.0,
        200.0,
    }
    assert {config["block_extension_params"]["ridge_weight"] for config in configs} == {
        1e-6,
        1e-3,
    }
    assert all(config["save_transported_artifacts"] is False for config in configs)
    assert all("save_transported_tvs_dir" not in config for config in configs)


def test_skip_controls_mark_correction_factors_inactive() -> None:
    campaign = _load_generator()
    _, controls = campaign.validate_spec()

    expected = {
        "lmc_mode",
        "extension_strategy",
        "n_batches_act",
        "ridge_identity",
        "ridge_weight",
    }
    for row, config in controls:
        assert row["skip_correction"] is True
        assert set(config["campaign_metadata"]["inactive_factors"]) == expected
        assert config["block_extension_params"]["skip_correction"] is True
