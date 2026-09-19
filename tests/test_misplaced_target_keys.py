"""Regression tests guarding against misplaced target-* keys at the top level.

A campaign generator (target_informed_brace_20260918) produced 91 configs that set
``target_residual_completion`` and ``target_shared_correction`` as TOP-LEVEL keys of
the run config, instead of nesting them under ``block_extension_params``. The
authoritative implementation only ever reads these from
``block_extension_params.<key>`` (see ``_as_target_shared_correction`` and
``resolve_block_extension_config``), so a top-level key is silently ignored and the
run executes as a plain baseline while its filename/run_id claims to be an ablation.

``resolve_block_extension_config`` must fail loudly on this instead of silently
producing a mislabeled baseline result.
"""

from __future__ import annotations

import pytest

from merge_and_rebase.eval.block_extension import resolve_block_extension_config

_MISPLACED_KEYS = (
    "target_shared_correction",
    "target_residual_completion",
    "capture_target_residual_reference",
)


@pytest.mark.parametrize("key", _MISPLACED_KEYS)
def test_misplaced_top_level_key_raises(key: str) -> None:
    cfg = {
        "block_extension_enabled": True,
        "block_extension_params": {},
        key: {"enabled": True},
    }
    with pytest.raises(ValueError, match=rf"{key}.*top level"):
        resolve_block_extension_config(cfg)


@pytest.mark.parametrize("key", _MISPLACED_KEYS)
def test_misplaced_top_level_key_message_names_nested_location(key: str) -> None:
    cfg = {
        "block_extension_enabled": True,
        "block_extension_params": {},
        key: {"enabled": True},
    }
    with pytest.raises(ValueError, match=rf"block_extension_params\.{key}"):
        resolve_block_extension_config(cfg)


def test_correctly_nested_target_shared_correction_resolves() -> None:
    cfg = {
        "block_extension_enabled": True,
        "block_extension_params": {
            "lmc_mode": "shared",
            "skip_correction": False,
            "target_shared_correction": {"target_weight": 0.3},
        },
    }
    _, resolved = resolve_block_extension_config(cfg)
    assert resolved.target_shared_correction is not None
    assert resolved.target_shared_correction.active
    assert resolved.target_shared_correction.target_weight == pytest.approx(0.3)


def test_config_without_target_shared_correction_defaults_to_none() -> None:
    cfg = {
        "block_extension_enabled": True,
        "block_extension_params": {},
    }
    _, resolved = resolve_block_extension_config(cfg)
    assert resolved.target_shared_correction is None
