"""P6.11: depth-rule schema under ``block_extension_params`` (schema only, no resolution)."""

from __future__ import annotations

import warnings

import pytest

from merge_and_rebase.rebase.block_extension.config import (
    brace_only_fields_set,
    parse_depth_rule_schema,
    resolve_block_extension_config,
)


def test_absent_is_method_default():
    s = parse_depth_rule_schema({"block_extension_params": {"n_batches_act": 1}})
    assert (s.rule, s.source, s.skip_correction, s.extension_strategy) == ("method_default", None, None, None)
    assert parse_depth_rule_schema({}).rule == "method_default"


@pytest.mark.parametrize(
    ("alias", "rule"), [("ariadne", "brace"), ("discrete_index_match", "discrete_index_match"), (" Ariadne ", "brace")]
)
def test_alias_maps(alias, rule):
    s = parse_depth_rule_schema({"depth_alignment": alias})
    assert (s.rule, s.source, s.depth_alignment_given) == (rule, "alias:depth_alignment", True)


def test_explicit_rule_and_fields():
    s = parse_depth_rule_schema(
        {"block_extension_params": {"depth_rule": "brace", "skip_correction": True, "extension_strategy": "x"}}
    )
    assert (s.rule, s.source, s.skip_correction, s.depth_rule_given) == ("brace", "config", True, True)
    assert s.extension_strategy == "x"
    assert parse_depth_rule_schema({"block_extension_params": {"skip_correction": False}}).skip_correction is False


def test_equal_alias_and_rule_accepted():
    s = parse_depth_rule_schema(
        {"depth_alignment": "ariadne", "block_extension_params": {"depth_rule": "brace"}}
    )
    assert s.rule == "brace"


def test_conflict_raises():
    with pytest.raises(ValueError, match="Conflicting depth rules"):
        parse_depth_rule_schema(
            {"depth_alignment": "ariadne", "block_extension_params": {"depth_rule": "discrete_index_match"}}
        )
    with pytest.raises(ValueError, match="Conflicting depth rules"):
        resolve_block_extension_config(
            {"depth_alignment": "discrete_index_match", "block_extension_params": {"depth_rule": "brace"}}
        )


def test_invalid_values():
    with pytest.raises(ValueError, match="depth_rule must be one of"):
        parse_depth_rule_schema({"block_extension_params": {"depth_rule": "nope"}})
    with pytest.raises(ValueError, match="depth_alignment must be one of: ariadne, discrete_index_match"):
        parse_depth_rule_schema({"depth_alignment": "nope"})


def test_brace_only_under_discrete_warns_and_lists():
    cfg = {"block_extension_params": {"depth_rule": "discrete_index_match", "ridge_identity": 0.5, "lmc_mode": "shared"}}
    with pytest.warns(RuntimeWarning, match=r"\['lmc_mode', 'ridge_identity'\]"):
        parse_depth_rule_schema(cfg)
    # via the alias, and through the real resolver
    cfg2 = {"depth_alignment": "discrete_index_match", "block_extension_params": {"dampening_factor": 0.5}}
    with pytest.warns(RuntimeWarning, match="dampening_factor"):
        resolve_block_extension_config(cfg2)


def test_defaults_and_brace_rule_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parse_depth_rule_schema(
            {"depth_alignment": "discrete_index_match", "block_extension_params": {"ridge_weight": 1e-6, "n_batches_act": 3}}
        )
        parse_depth_rule_schema({"block_extension_params": {"depth_rule": "brace", "ridge_identity": 0.5}})
        resolve_block_extension_config({"block_extension_params": {"depth_rule": "brace", "n_batches_act": 1}})


def test_brace_only_fields_set_ignores_defaults():
    assert brace_only_fields_set({"skip_correction": False, "insertion_order": "bottom-top"}) == []
    assert brace_only_fields_set({"skip_correction": True}) == ["skip_correction"]
