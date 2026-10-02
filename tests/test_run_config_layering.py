"""``rebase.run_config`` sits below ``eval`` in the layering: importing it must not load any
``merge_and_rebase.eval`` module, and the merge-mode helpers it uses must be the same objects the
vision merge module re-exports."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["run_config", "prestep"])
def test_rebase_stage_modules_import_no_eval_module(module: str) -> None:
    code = (
        f"import sys; import merge_and_rebase.rebase.{module}; "
        "bad=[m for m in sys.modules if m == 'merge_and_rebase.eval' or m.startswith('merge_and_rebase.eval.')]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_merge_mode_helpers_are_shared_objects() -> None:
    from merge_and_rebase.eval.vision_rebase import merge as vision_merge
    from merge_and_rebase.rebase import merge_modes, run_config

    for name in (
        "_resolve_merge_mode_config",
        "_SINGLE_TRANSPORT_MODES",
        "_VALID_MERGE_MODES",
        "_TRANSPORT_THEN_MERGE_MODES",
    ):
        assert getattr(vision_merge, name) is getattr(merge_modes, name)
    assert run_config._resolve_merge_mode_config is merge_modes._resolve_merge_mode_config
    assert run_config._SINGLE_TRANSPORT_MODES is merge_modes._SINGLE_TRANSPORT_MODES
