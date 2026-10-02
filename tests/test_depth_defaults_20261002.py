"""P5.12: per-method depth defaults, ``depth_defaults`` switch and the meaning-changed guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merge_and_rebase.rebase.run_config import ConfigMeaningChangedError, resolve_run_config

SUITES = {"vision8": SimpleNamespace(tasks=["MNIST", "DTD"])}
BE = {"n_batches_act": 2}


def _resolve(**cfg):
    base = {"method": "theseus", "block_extension_enabled": True}
    return resolve_run_config({**base, **cfg}, suites=SUITES)


def test_theseus_method_default_is_brace_skip_correction():
    r = _resolve(depth_defaults="method", block_extension_params=dict(BE))
    assert r.block_extension_cfg.skip_correction is True
    assert r.depth_rule.kind == "brace"
    assert r.depth_rule_resolved == {
        "rule": "brace",
        "extension_strategy": "interpolate_per_weight",
        "skip_correction": True,
        "depth_pairing": None,
        "source": "method_default",
    }
    r.bind(4, 6)  # method mode never guards


def test_legacy_is_byte_identical_to_old_defaults():
    r = _resolve(depth_defaults="legacy", block_extension_params=dict(BE))
    assert r.block_extension_cfg.skip_correction is False
    assert r.depth_rule.source == "legacy_defaults"
    r.bind(4, 6)
    b = _resolve(method="bico", depth_defaults="legacy")
    assert b.depth_rule.kind == "brace"


def test_absent_theseus_guarded_only_on_depth_mismatch():
    r = _resolve(block_extension_params=dict(BE))
    r.bind(4, 4)  # equal depth: the default cannot change the outcome
    with pytest.raises(ConfigMeaningChangedError, match="skip_correction.*depth_defaults.*method"):
        r.bind(4, 6)
    assert issubclass(ConfigMeaningChangedError, ValueError)


def test_explicit_skip_correction_is_never_guarded_nor_overridden():
    for skip in (True, False):
        r = _resolve(block_extension_params={**BE, "skip_correction": skip})
        assert r.block_extension_cfg.skip_correction is skip
        r.bind(4, 6)


def test_theseus_option_requiring_correction_without_skip_value_raises_at_resolve():
    with pytest.raises(ConfigMeaningChangedError, match="joint_blockwise_correction"):
        _resolve(
            block_extension_params={
                **BE,
                "lmc_mode": "shared",
                "extension_strategy": "duplicate_per_weight",
                "calibration_split": "val",
                "joint_blockwise_correction": {"enabled": True},
            }
        )


def test_bico_defaults():
    m = _resolve(method="bico", depth_defaults="method")
    assert m.depth_rule.kind == "discrete_index_match"
    assert m.depth_rule_resolved["rule"] == "discrete_index_match"
    assert m.depth_alignment_mode == "discrete_index_match"
    m.bind(4, 6)
    absent = _resolve(method="bico")
    with pytest.raises(ConfigMeaningChangedError, match="depth_alignment"):
        absent.bind(4, 6)
    explicit = _resolve(method="bico", depth_alignment="ariadne")
    assert explicit.depth_rule.kind == "brace"
    explicit.bind(4, 6)


def test_unknown_depth_defaults_value_rejected():
    with pytest.raises(ValueError, match="depth_defaults must be one of"):
        _resolve(depth_defaults="new")


def test_other_methods_unchanged():
    r = _resolve(method="gradfix", method_params={})
    assert r.depth_rule_resolved["rule"] == "none"
    r.bind(4, 6)
