"""Invalid LLM runs fail before model loading; evaluation partitions stay independent."""

from __future__ import annotations

import pytest
from golden.test_llm_main_golden import World, _base_cfg, _launch

from merge_and_rebase.data.llm_calibration import resolve_calibration_texts
from merge_and_rebase.eval.llm_rebase.context import _task_contexts


@pytest.mark.parametrize("params", [{"num_batches": {"values": [1, 2]}}, {"num_batches": {"values": [1]}}])
def test_method_parameter_search_is_rejected_before_models_or_logging(tmp_path, monkeypatch, params):
    cfg = _base_cfg(tmp_path, hyperparam_search={"alpha": {"values": [0.5]}, "method_params": params})
    with pytest.raises(ValueError, match="alpha search only"):
        _launch(cfg, tmp_path, monkeypatch, World())
    assert not _launch.last_calls.builds and not _launch.last_calls.recorders


def test_duplicate_labels_are_rejected_before_model_loading(tmp_path, monkeypatch):
    cfg = _base_cfg(tmp_path, harness_tasks=None, tasks="snli,snli", allow_prompt_eval=True)
    with pytest.raises(ValueError, match="duplicate labels.*snli"):
        _launch(cfg, tmp_path, monkeypatch, World())
    assert not _launch.last_calls.builds and not _launch.last_calls.tuned
    with pytest.raises(ValueError, match="duplicate labels"):
        _task_contexts(["snli", "snli"], ["a", "b"])


def test_test_slice_requires_search_slice_before_any_model_is_loaded(tmp_path, monkeypatch):
    cfg = _base_cfg(tmp_path, eval_before_rebase=True, harness_test_samples={"arc_easy": [10], "piqa": [11]})
    with pytest.raises(ValueError, match="requires explicit harness_samples"):
        _launch(cfg, tmp_path, monkeypatch, World())
    assert not _launch.last_calls.builds and not _launch.last_calls.harness


def test_real_calibration_partition_separates_fit_search_and_test(tmp_path, monkeypatch):
    import lm_eval.tasks as lm_tasks

    class Task:
        eval_docs = [{"i": index} for index in range(20)]

        def doc_to_text(self, doc):
            return f"the quick brown fox {doc['i']}"

    class Manager:
        def load_task_or_group(self, names):
            return {name: Task() for name in names}

    monkeypatch.setattr(lm_tasks, "TaskManager", Manager)
    calibration = resolve_calibration_texts(harness_tasks=["arc_easy", "piqa"], n_sequences=4, seed=3)
    search = {task: indices[:3] for task, indices in calibration.eval_samples.items()}
    test = {task: indices[3:6] for task, indices in calibration.eval_samples.items()}
    cfg = _base_cfg(
        tmp_path,
        harness_samples=search,
        harness_test_samples=test,
        eval_before_rebase=True,
        hyperparam_search={"alpha": {"values": [0.0, 0.5]}},
    )
    cfg.pop("calibration_prompts")
    calls = _launch(cfg, tmp_path, monkeypatch, World())
    assert calls.recorders[0].status == "success" and len(calls.harness) == 4
    assert all(call["samples"] == search for call in calls.harness[:-1])
    assert calls.harness[-1]["samples"] == test
    for task, allowed in calibration.eval_samples.items():
        assert set(search[task]).isdisjoint(test[task])
        assert set(search[task] + test[task]).issubset(allowed)


def test_structured_runtime_owns_immutable_model_bases(tmp_path, monkeypatch):
    from merge_and_rebase.eval.llm_rebase import cli
    from merge_and_rebase.eval.llm_rebase.runtime import LlmCalibration, LlmEvaluation, LlmModels

    build = cli.build_runtime
    captured = []

    def wrapped(*args, **kwargs):
        runtime = build(*args, **kwargs)
        captured.append(runtime)
        return runtime

    monkeypatch.setattr(cli, "build_runtime", wrapped)
    _launch(_base_cfg(tmp_path), tmp_path, monkeypatch, World())
    runtime = captured[0]
    assert isinstance(runtime.models, LlmModels)
    assert isinstance(runtime.calibration, LlmCalibration)
    assert isinstance(runtime.evaluation, LlmEvaluation)
    for key, value in runtime.models.target_base_sd.items():
        assert value.data_ptr() != runtime.models.target_llm.model.state_dict()[key].data_ptr()
