"""Plain depth-rule names in the config (2026-10-09): interpolate_layers / index_match.

An explicit rule must execute exactly what ``depth_defaults: method`` executed for the same method; the only
difference is where the rule came from (``source``). Contradictory options fail instead of silently switching rules.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from merge_and_rebase.rebase.block_extension.config import PUBLIC_DEPTH_RULES, parse_depth_rule_schema
from merge_and_rebase.rebase.capabilities import resolve_depth_strategy
from merge_and_rebase.rebase.depth_report import describe_depth_handling
from merge_and_rebase.rebase.run_config import resolve_run_config

SUITES = {"vision8": SimpleNamespace(tasks=["MNIST", "DTD"])}
BE = {"n_batches_act": 2}


def _resolve(method, **cfg):
    return resolve_run_config({"method": method, "block_extension_enabled": True, **cfg}, suites=SUITES)


def _same_except_source(explicit, default):
    assert explicit.depth_rule.source == "config" and default.depth_rule.source == "method_default"
    assert explicit.depth_rule.kind == default.depth_rule.kind
    assert explicit.depth_rule.extension_strategy == default.depth_rule.extension_strategy
    assert explicit.depth_rule.skip_correction == default.depth_rule.skip_correction
    assert explicit.block_extension_cfg == default.block_extension_cfg
    assert {**explicit.depth_rule_resolved, "source": None} == {**default.depth_rule_resolved, "source": None}


def test_interpolate_layers_equals_the_theseus_method_default():
    explicit = _resolve("theseus", block_extension_params={**BE, "depth_rule": "interpolate_layers"})
    default = _resolve("theseus", depth_defaults="method", block_extension_params=dict(BE))
    _same_except_source(explicit, default)
    explicit.bind(4, 6)  # an explicit rule is never meaning-changed-guarded
    assert explicit.depth_rule_resolved["skip_correction"] is True


def test_index_match_equals_the_bico_method_default():
    explicit = _resolve("bico", block_extension_params={**BE, "depth_rule": "index_match"})
    default = _resolve("bico", depth_defaults="method", block_extension_params=dict(BE))
    _same_except_source(explicit, default)
    explicit.bind(4, 6)


@pytest.mark.parametrize(("plain", "internal"), sorted(PUBLIC_DEPTH_RULES.items()))
def test_plain_names_are_aliases_of_the_internal_rules(plain, internal):
    assert parse_depth_rule_schema({"block_extension_params": {"depth_rule": plain}}, warn=False).rule == internal
    assert resolve_depth_strategy("theseus", {"depth_rule": plain}, None, None).rule == internal


@pytest.mark.parametrize(
    "params",
    [{"skip_correction": False}, {"extension_strategy": "duplicate_per_weight"}],
)
def test_interpolate_layers_rejects_options_that_make_it_something_else(params):
    with pytest.raises(ValueError, match="interpolate_layers"):
        parse_depth_rule_schema({"block_extension_params": {"depth_rule": "interpolate_layers", **params}}, warn=False)


def test_index_match_with_brace_only_keys_warns_and_stays_index_match():
    params = {**BE, "depth_rule": "index_match", "extension_strategy": "duplicate_per_weight", "n_cascade_iters": 2}
    with pytest.warns(RuntimeWarning, match="ignores the BRACE-only"):
        resolved = _resolve("bico", block_extension_params=params)
    assert resolved.depth_rule.kind == "discrete_index_match"


def test_old_names_keep_working():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _resolve(
            "bico", block_extension_params={**BE, "depth_rule": "discrete_index_match"}
        ).depth_rule.kind == ("discrete_index_match")
        assert _resolve("theseus", block_extension_params={**BE, "depth_rule": "brace"}).depth_rule.kind == "brace"


def _line(method, source_depth, target_depth, **cfg):
    resolved = _resolve(method, **cfg)
    plan = resolved.bind(source_depth, target_depth)
    return describe_depth_handling(resolved, plan, source_depth, target_depth)


def test_run_start_line_names_the_executed_handling():
    theseus = _line("theseus", 24, 28, depth_defaults="method", block_extension_params=dict(BE))
    assert "theseus -> interpolate_layers" in theseus and "[0, 5, 11, 17]" in theseus and "24 -> 28" in theseus
    bico = _line("bico", 24, 28, depth_defaults="method", block_extension_params=dict(BE))
    assert "bico -> index_match" in bico and "[0, 1, 2, 3, 3, 4," in bico
    assert "none (source and target both have 24 layers)" in _line("bico", 24, 24, depth_defaults="method")
    assert "none (no depth step runs" in _line("gradfix", 12, 24)
    corrected = _line("theseus", 24, 28, block_extension_params={**BE, "skip_correction": False, "depth_rule": "brace"})
    assert "brace (extension_strategy=interpolate_per_weight, skip_correction=False)" in corrected


def test_run_start_line_for_ariadne_reports_the_pairing():
    line = _line("ariadne", 24, 28, ariadne_params={"preset": "ariadne", "num_batches": 2, "seed": 0})
    assert "ariadne -> no layers added; depth_pairing=relative" in line and "24 -> 28" in line
