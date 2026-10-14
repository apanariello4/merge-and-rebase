"""A task's prepared maps must be released before building the next model pair."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import torch

from merge_and_rebase.rebase.orchestration import MethodResult, TaskPipeline


def test_prepared_state_is_released_between_tasks():
    references = []
    saved = []

    class Prepared:
        def __init__(self):
            self.buffer = torch.ones(128, 128)

    class Stage:
        def run(self, env, task, pre):
            assert all(reference() is None for reference in references)
            prepared = Prepared()
            references.append(weakref.ref(prepared))
            return MethodResult(transported_delta={"w": pre.task_delta["w"] * 2}, prepared=prepared)

    class Prestep:
        def run(self, env, task, models):
            return SimpleNamespace(completion_note=None, timings={}, task_delta={"w": torch.tensor([1.0])})

        def load_delta(self, env, task, result):
            return result

    def build_models(env, task):
        assert all(reference() is None for reference in references)

    resolved = SimpleNamespace(
        direct_fit=False,
        prestep_eval=SimpleNamespace(source_only=False),
        merge=SimpleNamespace(mode="none"),
        method=SimpleNamespace(name="toy"),
    )
    pipeline = TaskPipeline(
        prestep=Prestep(),
        observers=[],
        method_stage=Stage(),
        saver=lambda task, delta: saved.append((task, delta)),
        build_models=build_models,
    )
    result = pipeline.run(
        SimpleNamespace(resolved=resolved, native_tasks=set(), run_logger=None),
        ["a", "b", "c"],
        {task: object() for task in ("a", "b", "c")},
    )
    assert all(reference() is None for reference in references)
    assert len(result.transported_deltas) == len(saved) == 3
    assert all(torch.equal(delta["w"], torch.tensor([2.0])) for _, delta in saved)
