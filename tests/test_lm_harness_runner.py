from __future__ import annotations

import sys
from types import ModuleType

import pytest


def _install_fake_lm_eval(monkeypatch, simple_evaluate):
    """
    Register a minimal fake `lm_eval` package in sys.modules, including the
    `lm_eval.config.task.TaskConfig` submodule that `run()` imports to patch
    around the group-task (e.g. "mmlu") OOM in _patched_task_config_to_dict.

    Setting sys.modules["lm_eval.config.task"] directly (rather than relying
    on `lm_eval`/`lm_eval.config` being real packages with __path__) is enough:
    CPython's import machinery returns an already-cached fully-qualified
    module name straight from sys.modules without walking through its parents.
    """

    class _FakeTaskConfig:
        to_dict = staticmethod(lambda self=None, keep_callable=False: {})

    lm_eval_mod = ModuleType("lm_eval")
    lm_eval_mod.simple_evaluate = simple_evaluate
    config_mod = ModuleType("lm_eval.config")
    task_mod = ModuleType("lm_eval.config.task")
    task_mod.TaskConfig = _FakeTaskConfig

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.config", config_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.config.task", task_mod)
    return _FakeTaskConfig


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

    _install_fake_lm_eval(monkeypatch, lambda **kwargs: _FakeResults())

    import torch.nn as nn

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

    _install_fake_lm_eval(monkeypatch, lambda **kwargs: _FakeResults())

    import torch.nn as nn

    from merge_and_rebase.eval.lm_harness_runner import run

    stub_model = nn.Module()
    result = run(tasks=["hendrycks_math500", "minerva_math500"], model=stub_model, tokenizer=None)
    assert result["hendrycks_math500_exact_match"] == 0.12
    assert result["minerva_math500_exact_match"] == 0.10
    assert result["minerva_math500_math_verify"] == 0.31
    assert not any(k.endswith("_stderr") for k in result)


def test_harness_aggregates_per_task_fewshot(monkeypatch) -> None:
    """
    The whole point of harness_num_fewshot supporting a list: tasks with
    different few-shot counts (e.g. arc_easy=0, arc_challenge=25, mmlu=5) must
    all be runnable from one run() call, grouped into one simple_evaluate call
    per distinct fewshot value, with results merged back into a single flat
    dict as if it had been one evaluation.
    """
    calls: list[dict] = []

    # canned per-task results, keyed by task name, returned regardless of
    # which group a task lands in
    canned = {
        "arc_easy": {"acc,none": 0.70, "acc_norm,none": 0.65},
        "arc_challenge": {"acc,none": 0.50, "acc_norm,none": 0.55},
        "hellaswag": {"acc,none": 0.60, "acc_norm,none": 0.68},
    }

    def _fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        requested = kwargs["tasks"]

        class _R:
            def get(self, key, default=None):
                if key == "results":
                    return {t: canned[t] for t in requested}
                return default

        return _R()

    _install_fake_lm_eval(monkeypatch, _fake_simple_evaluate)

    import torch.nn as nn

    from merge_and_rebase.eval.lm_harness_runner import run

    stub_model = nn.Module()
    result = run(
        tasks=["arc_easy", "arc_challenge", "hellaswag"],
        model=stub_model,
        tokenizer=None,
        num_fewshot=[0, 25, 10],
    )

    # arc_easy (0-shot) and nothing else shares its fewshot value here, so it
    # gets its own call; arc_challenge (25-shot) and hellaswag (10-shot) each
    # get their own call too -- three distinct fewshot values, three calls.
    assert len(calls) == 3
    fewshot_by_call = {tuple(sorted(c["tasks"])): c["num_fewshot"] for c in calls}
    assert fewshot_by_call[("arc_easy",)] == 0
    assert fewshot_by_call[("arc_challenge",)] == 25
    assert fewshot_by_call[("hellaswag",)] == 10

    assert result["arc_easy_acc"] == 0.70
    assert result["arc_challenge_acc_norm"] == 0.55
    assert result["hellaswag_acc"] == 0.60


def test_harness_groups_tasks_sharing_fewshot(monkeypatch) -> None:
    """Tasks with the same few-shot count are batched into one call."""
    calls: list[dict] = []

    def _fake_simple_evaluate(**kwargs):
        calls.append(kwargs)

        class _R:
            def get(self, key, default=None):
                if key == "results":
                    return {t: {"acc,none": 1.0} for t in kwargs["tasks"]}
                return default

        return _R()

    _install_fake_lm_eval(monkeypatch, _fake_simple_evaluate)

    import torch.nn as nn

    from merge_and_rebase.eval.lm_harness_runner import run

    stub_model = nn.Module()
    run(
        tasks=["winogrande", "gsm8k"],
        model=stub_model,
        tokenizer=None,
        num_fewshot={"winogrande": 5, "gsm8k": 5},
    )

    assert len(calls) == 1
    assert sorted(calls[0]["tasks"]) == ["gsm8k", "winogrande"]
    assert calls[0]["num_fewshot"] == 5


def test_resolve_fewshot_by_task_int() -> None:
    from merge_and_rebase.eval.lm_harness_runner import _resolve_fewshot_by_task

    assert _resolve_fewshot_by_task(["a", "b"], 3) == {"a": 3, "b": 3}


def test_resolve_fewshot_by_task_list() -> None:
    from merge_and_rebase.eval.lm_harness_runner import _resolve_fewshot_by_task

    assert _resolve_fewshot_by_task(["a", "b"], [0, 5]) == {"a": 0, "b": 5}


def test_resolve_fewshot_by_task_dict() -> None:
    from merge_and_rebase.eval.lm_harness_runner import _resolve_fewshot_by_task

    assert _resolve_fewshot_by_task(["a", "b"], {"a": 0, "b": 5}) == {"a": 0, "b": 5}


def test_resolve_fewshot_by_task_list_length_mismatch() -> None:
    from merge_and_rebase.eval.lm_harness_runner import _resolve_fewshot_by_task

    with pytest.raises(ValueError, match="parallel"):
        _resolve_fewshot_by_task(["a", "b"], [0])


def test_resolve_fewshot_by_task_dict_missing_task() -> None:
    from merge_and_rebase.eval.lm_harness_runner import _resolve_fewshot_by_task

    with pytest.raises(ValueError, match="missing"):
        _resolve_fewshot_by_task(["a", "b"], {"a": 0})


def test_score_by_task_weights_each_task_equally() -> None:
    """
    Regression test: a naive flat average over every "{task}_{metric}" key
    lets a group task like MMLU (dozens of subject/category keys) dominate
    the score. score_by_task must average within each task first, then
    average those per-task scores together.
    """
    from merge_and_rebase.eval.lm_harness_runner import score_by_task

    results = {
        "gsm8k_exact_match": 1.0,
        "arc_easy_acc": 0.5,
        "arc_easy_acc_norm": 0.7,
        "mmlu_acc": 0.6,
        "mmlu_abstract_algebra_acc": 0.1,
        "mmlu_anatomy_acc": 0.2,
        "mmlu_stem_acc": 0.3,
    }
    score = score_by_task(results, ["gsm8k", "arc_easy", "mmlu"])
    expected = (1.0 + (0.5 + 0.7) / 2 + (0.6 + 0.1 + 0.2 + 0.3) / 4) / 3
    assert score == pytest.approx(expected)


def test_score_by_task_ignores_tasks_with_no_results() -> None:
    from merge_and_rebase.eval.lm_harness_runner import score_by_task

    results = {"gsm8k_exact_match": 0.8}
    score = score_by_task(results, ["gsm8k", "missing_task"])
    assert score == pytest.approx(0.8)


def test_score_by_task_empty_results() -> None:
    from merge_and_rebase.eval.lm_harness_runner import score_by_task

    assert score_by_task({}, ["gsm8k"]) == 0.0
