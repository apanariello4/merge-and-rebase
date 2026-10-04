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


def _record_pool_size(monkeypatch):
    from merge_and_rebase.eval.llm_rebase import context

    seen: dict[str, int] = {}
    original = context.TextCalibrationCache.__init__

    def wrapped(self, *args, **kwargs):
        seen["n_calib_batches"] = kwargs["n_calib_batches"]
        original(self, *args, **kwargs)

    monkeypatch.setattr(context.TextCalibrationCache, "__init__", wrapped)
    return seen


def test_calibration_pool_counts_ariadne_num_batches(tmp_path, monkeypatch):
    seen = _record_pool_size(monkeypatch)
    cfg = _base_cfg(tmp_path, **_ARIADNE, depth_defaults="method")  # n_batches_act=2 < num_batches=3
    calls = _launch(cfg, tmp_path, monkeypatch, World())
    assert calls.recorders[0].status == "success"
    assert seen["n_calib_batches"] == 3


def test_ariadne_calibration_shortfall_warns(tmp_path, monkeypatch):
    import pytest

    # 6 calibration prompts at batch size 2 hold 3 batches; ask for 5.
    params = {"preset": "ariadne", "num_batches": 5, "seed": 0}
    cfg = _base_cfg(tmp_path, method="ariadne", ariadne_params=params, depth_defaults="method")
    with pytest.warns(RuntimeWarning, match="fewer than ariadne_params.num_batches=5"):
        _launch(cfg, tmp_path, monkeypatch, World())
