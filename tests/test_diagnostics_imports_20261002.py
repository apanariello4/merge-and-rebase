"""P5.16: the diagnostic CLIs live in ``eval.diagnostics`` and stay runnable with ``python -m``."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

_MODULES = ("vision_connectivity", "vision_logit_kl", "checkpoint_stats", "vision_block_extension")


@pytest.mark.parametrize("name", _MODULES)
def test_diagnostic_module_imports_and_exposes_main(name):
    module = importlib.import_module(f"merge_and_rebase.eval.diagnostics.{name}")
    assert callable(getattr(module, "main", None))


@pytest.mark.parametrize("name", _MODULES)
def test_old_top_level_diagnostic_module_is_gone(name):
    assert importlib.util.find_spec(f"merge_and_rebase.eval.{name}") is None
