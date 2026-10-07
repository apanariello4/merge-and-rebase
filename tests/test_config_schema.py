"""Canonical config schema (``rebase/config_schema.py``): a config written with the canonical names resolves exactly
like its legacy spelling, legacy names keep working, and mixing both spellings of one key is an error."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import pytest
from test_resolved_config_snapshot import DEPTHS, _configs, _modality, _normalize  # noqa: E402

from merge_and_rebase.eval.llm_rebase.run_config import resolve_llm_method, resolve_llm_run_config
from merge_and_rebase.rebase.block_extension import config as be_config
from merge_and_rebase.rebase.config_schema import (
    canonicalize,
    legacy_keys_used,
    load_run_config,
    to_canonical,
)
from merge_and_rebase.rebase.run_config import resolve_run_config
from merge_and_rebase.utils.helpers import load_json

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "configs" / "examples"


def _resolve(cfg: dict, modality: str) -> dict:
    cfg = dict(canonicalize(cfg))
    cfg.setdefault("block_extension_enabled", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if modality == "vision":
            resolved = resolve_run_config(cfg)
        else:
            method_name, method_obj, method_params = resolve_llm_method(cfg)
            enabled, be_cfg = be_config.resolve_block_extension_config(cfg)
            resolved = resolve_llm_run_config(
                cfg,
                method=method_obj,
                method_name=method_name,
                method_params=method_params,
                block_extension_enabled=enabled,
                block_extension_cfg=be_cfg,
                device="cpu",
                eval_before_rebase_only=False,
            )
        out = _normalize(resolved)
        out["plans"] = {f"{s}->{t}": _normalize(resolved.bind(s, t)) for s, t in DEPTHS[modality]}
    out.pop("cfg")  # the raw input differs by construction
    # Which legacy alias the rule came from is provenance of the spelling, not of the run.
    for record in (out["depth_rule"], out["depth_rule_resolved"]):
        if record.get("source") == "alias:depth_alignment":
            record["source"] = "config"
    return json.loads(json.dumps(out))


@pytest.mark.parametrize("rel", _configs())
def test_canonical_spelling_resolves_like_the_legacy_one(rel):
    canonical = dict(load_json(EXAMPLES / rel))  # the example configs are written with the canonical names
    modality = _modality(rel, canonical)
    assert not legacy_keys_used(canonical), legacy_keys_used(canonical)
    legacy = dict(canonicalize(canonical))
    assert legacy_keys_used(legacy)  # the flat spelling really is the legacy one
    assert _resolve(canonical, modality) == _resolve(legacy, modality)


@pytest.mark.parametrize("rel", _configs())
def test_round_trip_is_exact(rel):
    canonical = dict(load_json(EXAMPLES / rel))
    assert to_canonical(dict(canonicalize(canonical))) == canonical


def test_flat_config_is_returned_unchanged():
    cfg = {"method": "theseus", "method_params": {"num_batches": 2}, "alpha_search": True}
    assert canonicalize(cfg) is cfg


def test_load_run_config_warns_about_legacy_keys_only():
    with pytest.warns(FutureWarning, match=r"legacy key names \['method', 'method_params'\]"):
        _, legacy = load_run_config({"method": "theseus", "method_params": {}})
    assert legacy == ["method", "method_params"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg, legacy = load_run_config({"method": {"name": "theseus", "params": {"num_batches": 2}}})
    assert legacy == [] and cfg["method"] == "theseus" and cfg["method_params"] == {"num_batches": 2}


def test_translation_details():
    cfg = canonicalize(
        {
            "method": {"name": "ariadne", "params": {"preset": "ariadne"}},
            "depth_alignment": {
                "rule": "brace",
                "brace": {"correction_endpoint": "base", "data": {"num_batches": 4, "split": "test"}},
            },
            "alpha": {"search": True, "selection": "per_task"},
            "humanize_classnames": True,
        }
    )
    assert cfg["ariadne_params"] == {"preset": "ariadne"} and "method_params" not in cfg
    assert cfg["block_extension_params"] == {
        "lmc_mode": "shared",
        "n_batches_act": 4,
        "calibration_split": "test",
        "depth_rule": "brace",
    }
    assert cfg["alpha_search"] is True and cfg["alpha_selection"] == "per_task" and cfg["no_humanize"] is False


@pytest.mark.parametrize(
    "cfg,match",
    [
        ({"merge": {"mode": "none"}, "merge_mode": "none"}, "both 'merge.mode' and its legacy spelling 'merge_mode'"),
        ({"humanize_classnames": True, "no_humanize": True}, "both 'humanize_classnames' and its legacy inverse"),
        ({"depth_alignment": {"brace": {"correction_endpoint": "x"}}}, "correction_endpoint must be one of"),
        ({"alpha": {"bogus": 1}}, r"unknown canonical config keys: \['alpha.bogus'\]"),
    ],
)
def test_invalid_spellings_raise(cfg, match):
    with pytest.raises(ValueError, match=match):
        canonicalize(copy.deepcopy(cfg))


def test_method_data_and_gradient_blocks_leave_no_trace_in_method_params():
    cfg = canonicalize(
        {
            "method": {
                "name": "gradfix",
                "params": {"gradient": {"images_per_class": 1}, "data": {"protocol": "task_local"}, "vote": "max"},
            }
        }
    )
    assert cfg["method_params"] == {"vote": "max"}
    assert cfg["grad_imgs_per_class"] == 1 and cfg["transport_calibration_protocol"] == "task_local"
