from __future__ import annotations

import sys
from types import ModuleType

import pytest


def test_harness_import_without_lm_eval() -> None:
    from merge_and_rebase.eval.lm_harness_runner import run

    assert callable(run)


def test_harness_raises_without_lm_eval(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lm_eval", None)

    from merge_and_rebase.eval.lm_harness_runner import run

    with pytest.raises(ImportError, match="lm-eval"):
        run(tasks=["hellaswag"], model=None, tokenizer=None)


def test_harness_with_mocked_lm_eval(monkeypatch) -> None:
    class _FakeResults:
        def get(self, key, default=None):
            if key == "results":
                return {
                    "hellaswag": {"acc,none": 0.42},
                    "piqa": {"acc_norm,none": 0.75},
                }
            return default

    class _FakeLMEval:
        def simple_evaluate(self, **kwargs):
            return _FakeResults()

    import torch.nn as nn

    fake_mod = ModuleType("lm_eval")
    fake_mod.simple_evaluate = lambda **kwargs: _FakeResults()
    monkeypatch.setitem(sys.modules, "lm_eval", fake_mod)

    from merge_and_rebase.eval.lm_harness_runner import run

    stub_model = nn.Module()
    result = run(tasks=["hellaswag", "piqa"], model=stub_model, tokenizer=None)
    assert result["hellaswag_acc"] == 0.42
    assert result["piqa_acc_norm"] == 0.75


def test_harness_with_math_metrics(monkeypatch) -> None:
    # Regression test: MATH-style tasks (hendrycks_math500, minerva_math500) report
    # "exact_match"/"math_verify", not "acc"/"acc_norm" -- the runner must not
    # silently drop them (it used to, returning an empty dict for these tasks).
    class _FakeResults:
        def get(self, key, default=None):
            if key == "results":
                return {
                    "hendrycks_math500": {
                        "exact_match,none": 0.12,
                        "exact_match_stderr,none": 0.01,
                        "alias": "hendrycks_math500",
                    },
                    "minerva_math500": {
                        "exact_match,none": 0.10,
                        "math_verify,none": 0.31,
                        "math_verify_stderr,none": 0.02,
                    },
                }
            return default

    import torch.nn as nn

    fake_mod = ModuleType("lm_eval")
    fake_mod.simple_evaluate = lambda **kwargs: _FakeResults()
    monkeypatch.setitem(sys.modules, "lm_eval", fake_mod)

    from merge_and_rebase.eval.lm_harness_runner import run

    stub_model = nn.Module()
    result = run(tasks=["hendrycks_math500", "minerva_math500"], model=stub_model, tokenizer=None)
    assert result["hendrycks_math500_exact_match"] == 0.12
    assert result["minerva_math500_exact_match"] == 0.10
    assert result["minerva_math500_math_verify"] == 0.31
    assert not any(k.endswith("_stderr") for k in result)
