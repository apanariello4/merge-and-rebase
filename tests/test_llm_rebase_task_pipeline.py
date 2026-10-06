"""LLM rebase runs the shared per-task loop: prepare -> transport -> free, one task at a time (S10d)."""

from __future__ import annotations

from golden.test_llm_main_golden import CASES, World, _base_cfg, _launch

from merge_and_rebase.eval.llm_rebase.method_stages import TransportStage
from merge_and_rebase.eval.llm_rebase.stages import BracePrestep


def _record(monkeypatch, events: list[str]) -> None:
    prestep_run, transport = BracePrestep.run, TransportStage.run

    def pre_run(self, env, task, models):
        events.append(f"prepare:{task.task}")
        return prestep_run(self, env, task, models)

    def stage_run(self, env, task, pre):
        events.append(f"transport:{task.task}")
        result = transport(self, env, task, pre)
        events.append(f"freed:{task.task}:{pre.source_base_model is None}")
        return result

    monkeypatch.setattr(BracePrestep, "run", pre_run)
    monkeypatch.setattr(TransportStage, "run", stage_run)


def test_tasks_are_interleaved_and_resized_models_freed(tmp_path, monkeypatch):
    events: list[str] = []
    _record(monkeypatch, events)
    world_kw, make_cfg = CASES["theseus_extend_eval_before_rebase"]
    r = tmp_path / "r"
    _launch(_base_cfg(r, **make_cfg(r)), r, monkeypatch, World(**world_kw))
    assert events == [
        "prepare:task_0",
        "transport:task_0",
        "freed:task_0:True",
        "prepare:task_1",
        "transport:task_1",
        "freed:task_1:True",
    ]


def test_eval_before_rebase_only_never_transports(tmp_path, monkeypatch):
    events: list[str] = []
    _record(monkeypatch, events)
    world_kw, make_cfg = CASES["eval_before_rebase_only_extend"]
    r = tmp_path / "r"
    _launch(_base_cfg(r, **make_cfg(r)), r, monkeypatch, World(**world_kw))
    assert events == ["prepare:task_0", "prepare:task_1"]



def test_task_names_must_match_tuned_bodies():
    import pytest

    from merge_and_rebase.eval.llm_rebase.context import _task_contexts

    assert list(_task_contexts([], ["a", "b"])) == ["task_0", "task_1"]
    assert list(_task_contexts(["x", "y"], ["a", "b"])) == ["x", "y"]
    with pytest.raises(ValueError, match="tuned bodies"):
        _task_contexts(["x"], ["a", "b"])
