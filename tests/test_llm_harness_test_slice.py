"""Exercise production held-out scoring and document partition validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from merge_and_rebase.eval.llm_rebase.alpha_search import score_harness_test_slice
from merge_and_rebase.eval.llm_rebase.splits import resolve_harness_slices


@pytest.fixture
def score(monkeypatch):
    runner = Mock(return_value={"ifeval_acc": 0.5})
    monkeypatch.setattr("merge_and_rebase.eval.llm_rebase.harness.run", runner)

    def evaluate(test_samples, search_samples, tasks=("ifeval",)):
        model = torch.nn.Linear(1, 1, bias=False)
        return score_harness_test_slice(
            cfg={"harness_test_samples": test_samples},
            merged_delta={"weight": torch.ones(1, 1)},
            target_base_sd={"weight": torch.zeros(1, 1)},
            target_llm=SimpleNamespace(model=model, tokenizer=None),
            harness_tasks_resolved=list(tasks),
            device="cpu",
            harness_num_fewshot=0,
            harness_batch_size=1,
            harness_samples=search_samples,
            best_alpha=0.5,
        )

    return evaluate, runner


def test_disjoint_slices_score_only_the_test_documents(score):
    evaluate, runner = score
    results, samples = evaluate({"ifeval": [5, 6, 7]}, {"ifeval": [1, 2, 3]})
    assert results == {"ifeval_acc": 0.5} and samples == {"ifeval": [5, 6, 7]}
    runner.assert_called_once()
    assert runner.call_args.kwargs["samples"] == samples


@pytest.mark.parametrize("search", [None, {}])
def test_unrestricted_search_cannot_have_a_held_out_test_slice(score, search):
    evaluate, runner = score
    with pytest.raises(ValueError):
        evaluate({"ifeval": [0, 1]}, search)
    runner.assert_not_called()


def test_overlap_is_detected_per_task(score):
    evaluate, runner = score
    with pytest.raises(ValueError, match="overlaps the alpha-search slice"):
        evaluate(
            {"ifeval": [9], "hellaswag": [1]},
            {"ifeval": [50], "hellaswag": [1]},
            tasks=("ifeval", "hellaswag"),
        )
    runner.assert_not_called()


@pytest.mark.parametrize("indices", [[-1], [True], [1.5], [], [1, 1]])
def test_invalid_indices_are_refused_before_scoring(score, indices):
    evaluate, runner = score
    with pytest.raises(ValueError):
        evaluate({"ifeval": indices}, {"ifeval": [9]})
    runner.assert_not_called()


def test_malformed_sample_map_is_refused(score):
    evaluate, runner = score
    with pytest.raises(ValueError, match="must map task names"):
        evaluate([0, 1], {"ifeval": [9]})
    runner.assert_not_called()


@pytest.mark.parametrize(
    "search,test",
    [
        ({"ifeval": [1]}, {"ifeval": [2], "hellaswag": [3]}),
        ({"ifeval": [1], "hellaswag": [2]}, {"ifeval": [3]}),
    ],
)
def test_test_partition_requires_complete_task_coverage(score, search, test):
    evaluate, runner = score
    with pytest.raises(ValueError, match="every harness task"):
        evaluate(test, search, tasks=("ifeval", "hellaswag"))
    runner.assert_not_called()


def test_explicit_search_and_test_may_partition_the_calibration_holdout():
    slices = resolve_harness_slices(
        {"harness_samples": {"ifeval": [2, 3]}, "harness_test_samples": {"ifeval": [4]}},
        ["ifeval"],
        calibration_holdout={"ifeval": [2, 3, 4, 5]},
    )
    assert slices.search == {"ifeval": [2, 3]} and slices.test == {"ifeval": [4]}


@pytest.mark.parametrize("key", ["harness_samples", "harness_test_samples"])
def test_neither_evaluation_slice_may_use_calibration_documents(key):
    cfg = {"harness_samples": {"ifeval": [2]}, "harness_test_samples": {"ifeval": [3]}}
    cfg[key] = {"ifeval": [0]}
    with pytest.raises(ValueError, match="hold-out derived from calibration"):
        resolve_harness_slices(cfg, ["ifeval"], calibration_holdout={"ifeval": [2, 3, 4]})


def test_default_search_is_the_calibration_holdout():
    slices = resolve_harness_slices({}, ["ifeval"], calibration_holdout={"ifeval": [2, 3, 4]})
    assert slices.search == {"ifeval": [2, 3, 4]} and slices.test is None
