"""``RunPlan.depth_alignment`` reproduces the four prestep booleans ``bind`` used to store (pinned formula)."""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from pathlib import Path

import pytest

from merge_and_rebase.rebase.config_schema import canonicalize
from merge_and_rebase.rebase.merge_modes import _VALID_MERGE_MODES
from merge_and_rebase.rebase.run_config import DepthRule, resolve_run_config
from merge_and_rebase.utils.helpers import load_json

ROOT = Path(__file__).resolve().parents[1]


def _legacy_flags(depth_prestep_method, block_extension_enabled, rule, merge_mode, source_depth, target_depth):
    """The pre-DepthAlignment ``bind`` body, verbatim in substance."""
    run_be = bool(depth_prestep_method and block_extension_enabled and rule == "brace" and source_depth != target_depth)
    run_di = bool(depth_prestep_method and rule == "discrete_index_match" and source_depth != target_depth)
    if merge_mode == "merge_then_rebase" and run_be:
        return "NotImplementedError"
    per_task = merge_mode != "merge_then_brace_then_transport"
    return (run_be, run_di, bool(run_be and per_task), bool(run_di and per_task))


@pytest.fixture(scope="module")
def base():
    cfg = dict(canonicalize(load_json(ROOT / "configs" / "examples" / "vision8_theseus_b16_to_l14.json")))
    cfg["block_extension_enabled"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return resolve_run_config(cfg)


def test_depth_alignment_matches_legacy_booleans(base):
    combos = itertools.product(
        ("theseus", "gradfix"),
        (True, False),
        ("none", "brace", "discrete_index_match"),
        sorted(_VALID_MERGE_MODES),
        ((12, 12), (12, 24), (24, 12)),
    )
    for method_name, enabled, rule, merge_mode, (src, tgt) in combos:
        resolved = dataclasses.replace(
            base,
            method_name=method_name,
            block_extension_enabled=enabled,
            depth_rule=DepthRule(kind=rule),
            depth_guard=None,
            merge=dataclasses.replace(base.merge, mode=merge_mode),
        )
        expected = _legacy_flags(resolved.depth_prestep_method, enabled, rule, merge_mode, src, tgt)
        try:
            plan = resolved.bind(src, tgt)
        except NotImplementedError:
            assert expected == "NotImplementedError", (method_name, enabled, rule, merge_mode, src, tgt)
            continue
        actual = (
            plan.run_block_extension_prestep,
            plan.run_discrete_layer_match_prestep,
            plan.task_block_extension_prestep,
            plan.task_discrete_layer_match_prestep,
        )
        assert actual == expected, (method_name, enabled, rule, merge_mode, src, tgt)
