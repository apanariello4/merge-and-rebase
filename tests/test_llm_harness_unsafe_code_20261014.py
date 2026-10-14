"""Harness: opt-in confirm_run_unsafe_code (humaneval/mbpp need it); the default call is unchanged (lm-eval mocked)."""

from __future__ import annotations

import torch.nn as nn
from test_llm_harness_filters_chat_template_20261007 import _OLD_KEYS, IFEVAL, _install_fake_lm_eval, _Tok

from merge_and_rebase.eval.llm_rebase import harness


def test_default_call_has_no_unsafe_code_kwarg(monkeypatch):
    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, IFEVAL)
    harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu")
    assert set(calls[0]) == _OLD_KEYS


def test_confirm_run_unsafe_code_is_passed(monkeypatch):
    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, IFEVAL)
    harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu", confirm_run_unsafe_code=True)
    assert calls[0]["confirm_run_unsafe_code"] is True
    assert set(calls[0]) == _OLD_KEYS | {"confirm_run_unsafe_code"}
