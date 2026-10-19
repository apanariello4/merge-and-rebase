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


def test_unsafe_code_preloads_numpy_in_forkserver(monkeypatch):
    """Code tests that import numpy (HumanEval+) need it preloaded by the forkserver (2026-10-19)."""
    import multiprocessing
    import multiprocessing.forkserver as forkserver

    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, IFEVAL)
    server = forkserver.ForkServer()
    monkeypatch.setattr(forkserver, "_forkserver", server)
    monkeypatch.setattr(multiprocessing, "set_forkserver_preload", server.set_forkserver_preload)
    harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu")
    assert "numpy" not in server._preload_modules  # default call: untouched
    harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu", confirm_run_unsafe_code=True)
    assert "numpy" in server._preload_modules


def test_unsafe_code_refuses_running_forkserver_without_numpy(monkeypatch):
    import multiprocessing.forkserver as forkserver

    import pytest

    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, IFEVAL)
    server = forkserver.ForkServer()
    server._forkserver_pid = 12345  # pretend it is running
    monkeypatch.setattr(forkserver, "_forkserver", server)
    with pytest.raises(RuntimeError, match="already running"):
        harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu", confirm_run_unsafe_code=True)
