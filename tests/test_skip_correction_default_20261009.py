"""skip_correction defaults to true on every path; the BRACE ridge correction is opt-in and warns (2026-10-09)."""

from __future__ import annotations

import warnings

import pytest

from merge_and_rebase.rebase.block_extension.config import BlockExtensionConfig, resolve_block_extension_config
from merge_and_rebase.rebase.capabilities import resolve_depth_strategy


def _resolve(params):
    return resolve_block_extension_config({"block_extension_params": params})[1]


def test_default_is_no_correction_and_silent():
    assert BlockExtensionConfig().skip_correction is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve({}).skip_correction is True
        assert _resolve({"n_batches_act": 100}).skip_correction is True


def test_explicit_correction_is_allowed_but_warns_experimental():
    with pytest.warns(RuntimeWarning, match="experimental"):
        config = _resolve({"skip_correction": False})
    assert config.skip_correction is False


@pytest.mark.parametrize(
    "params",
    [{"correction_scope": "interleaved_once"}, {"target_shared_correction": {"target_weight": 0.5}, "lmc_mode": "shared"}],
)
def test_correction_only_options_now_need_an_explicit_false(params):
    with pytest.raises(ValueError, match="requires skip_correction=false"):
        _resolve(params)
    with pytest.warns(RuntimeWarning, match="experimental"):
        _resolve({**params, "skip_correction": False})


def test_legacy_depth_strategy_default_is_no_correction():
    strategy = resolve_depth_strategy("theseus", {"extension_strategy": "duplicate_per_weight"}, None, None)
    assert strategy.legacy and strategy.skip_correction is True
    strategy = resolve_depth_strategy("theseus", {"skip_correction": False}, None, None)
    assert strategy.skip_correction is False
