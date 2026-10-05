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


def test_summary_records_ariadne_fit(tmp_path, monkeypatch):
    digests = []
    for run in ("a", "b"):
        root = tmp_path / run
        calls = _launch(_base_cfg(root, **_ARIADNE, depth_defaults="method"), root, monkeypatch, World())
        summary = calls.recorders[0].summary
        report = summary["task_vectors"]["ariadne"]
        assert report["pairing"]["depth_pairing"] == "relative"
        assert len(report["pairing"]["pairing"]) == report["pairing"]["target_depth"]
        assert set(report["per_task"]) == {"task_0", "task_1"}
        for record in report["per_task"].values():
            assert len(record["task_vector_sha256"]) == 64
            assert record["calibration"]["actual_batches"] == record["calibration"]["requested_batches"] == 3
            assert record["calibration_indices_sha256"] is not None
            assert record["timing"]["correction_fit"] and record["diagnostics"]
        digests.append({t: r["task_vector_sha256"] for t, r in report["per_task"].items()})
    assert digests[0] == digests[1]


class _StubHarnessTask:
    def __init__(self, n_docs: int):
        self.eval_docs = [{"i": i} for i in range(n_docs)]

    def doc_to_text(self, doc):
        return f"the quick brown fox {doc['i']}"


def test_harness_calibration_is_held_out_with_the_default_split(tmp_path, monkeypatch):
    """Real resolver, default block_extension_params.calibration_split ("test"): the harness must not score the
    calibration docs (it used to score all of them: samples=None)."""
    import lm_eval.tasks as lm_tasks

    class _Manager:
        def load_task_or_group(self, names):
            return {n: _StubHarnessTask(20) for n in names}

    monkeypatch.setattr(lm_tasks, "TaskManager", _Manager)
    for method, extra in (("theseus", {}), ("ariadne", {"ariadne_params": {"preset": "ariadne", "num_batches": 2}})):
        root = tmp_path / method
        cfg = _base_cfg(root, method=method, **extra, depth_defaults="method")
        cfg.pop("calibration_prompts")
        assert "calibration_split" not in cfg["block_extension_params"]
        calls = _launch(cfg, root, monkeypatch, World())
        assert calls.harness, method
        for call in calls.harness:
            samples = call["samples"]
            assert samples is not None and set(samples) == {"arc_easy", "piqa"}, method
            assert all(len(v) == 18 for v in samples.values()), method  # 20 docs, 2 per task held out
