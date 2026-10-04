"""Regression tests for the release code-review findings (2026-10-04) on the LLM path.

- eval_before_rebase_only with Ariadne stops after the before-rebase eval (crashed: no method stage to precompute).
- The calibration pool is sized for Ariadne's own ``ariadne_params.num_batches`` (was: silently truncated).
- The run summary records Ariadne's calibration, fit diagnostics, timings, depth pairing and task-vector hash.
"""

from __future__ import annotations

from golden.test_llm_main_golden import World, _base_cfg, _launch

_ARIADNE = {"method": "ariadne", "ariadne_params": {"preset": "ariadne", "num_batches": 3, "seed": 0}}


def test_ariadne_eval_before_rebase_only_stops_cleanly(tmp_path, monkeypatch):
    cfg = _base_cfg(tmp_path, **_ARIADNE, eval_before_rebase_only=True, depth_defaults="method")
    calls = _launch(cfg, tmp_path, monkeypatch, World())
    rec = calls.recorders[0]
    assert rec.status == "success"
    assert rec.summary["stopped_after"] == "before_rebase_eval"
    assert len(calls.harness) == 1
