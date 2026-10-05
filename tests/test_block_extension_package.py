"""The ``rebase.block_extension`` package is the home of the BRACE config/schedule layers and the extenders; it must not
import any ``merge_and_rebase.eval`` module (the former ``eval.block_extension*`` shims were removed in P5.16)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

_PACKAGE_MODULES = ("config", "schedules", "adapters", "core", "vision", "decoder")


@pytest.mark.parametrize("module", _PACKAGE_MODULES)
def test_package_module_imports_nothing_from_eval(module):
    if importlib.util.find_spec(f"merge_and_rebase.rebase.block_extension.{module}") is None:
        pytest.skip(f"{module} not created yet")
    code = (
        f"import sys, merge_and_rebase.rebase.block_extension.{module}; "
        "bad=[m for m in sys.modules if m.startswith('merge_and_rebase.eval')]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["block_extension", "block_extension_llm", "direct_residual"])
def test_retired_eval_shims_are_gone(module):
    assert importlib.util.find_spec(f"merge_and_rebase.eval.{module}") is None
