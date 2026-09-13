from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_brace_shared_ft_core_20260908.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("shared_ft_core_campaign", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_is_exact_and_reuses_the_completed_best_cell() -> None:
    campaign = load_generator()
    rows = campaign.validate_rows()
    assert len(rows) == 8
    assert {(r["direction"], r["transport"], r["merger"]) for r in rows} == {
        (direction, transport, merger)
        for direction in campaign.DIRECTIONS
        for transport in campaign.TRANSPORTS
        for merger in campaign.MERGERS
    }
    external = [r for r in rows if r["execution"] == "external_complete"]
    assert len(external) == 1
    assert (external[0]["direction"], external[0]["transport"], external[0]["merger"]) == campaign.EXTERNAL_CELL


def test_resolved_configs_change_only_the_map_anchor() -> None:
    campaign = load_generator()
    for row in campaign.validate_rows():
        config = campaign.build_config(row)
        assert config["block_extension_params"]["lmc_mode"] == "shared_ft"
        assert config["merge_mode"] == "brace_transport_then_merge"
        assert config["alpha_selection"] == "per_task"
        assert config["global_alpha_search"] is True
        assert config["method_params"]["center_acts"] is True
        assert config["block_extension_params"]["n_batches_act"] == 5


def test_write_is_additive_and_skips_external_config(tmp_path: Path) -> None:
    campaign = load_generator()
    manifest_path = campaign.write_campaign(tmp_path / "campaign")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["counts"] == {"logical_total": 8, "external_complete": 1, "to_submit": 7}
    assert len(list((tmp_path / "campaign" / "configs").glob("*.json"))) == 7
    with pytest.raises(FileExistsError):
        campaign.write_campaign(tmp_path / "campaign")
