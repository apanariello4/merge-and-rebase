"""P7.S5: calibration layer additions (all opt-in; defaults unchanged)."""

from __future__ import annotations

import datasets
import pytest
from _llm_fixtures import CALIB_TEXTS, local_tokenizer

from merge_and_rebase.data import llm_calibration as lc
from merge_and_rebase.data.llm_calibration import (
    build_text_calibration_loader,
    resolve_calibration_texts,
    tokenization_stats,
)
from merge_and_rebase.eval import llm_rebase


def test_loader_is_promoted_and_old_names_are_identical():
    assert llm_rebase._build_text_calibration_loader is build_text_calibration_loader
    assert llm_rebase._TokenizedPromptDataset is lc.TokenizedPromptDataset


def test_blank_prompts_rejected_but_ints_still_stringified():
    with pytest.raises(ValueError, match="empty entries"):
        resolve_calibration_texts(prompts=["a", "  "], n_sequences=1)
    assert resolve_calibration_texts(prompts=["a", 2], n_sequences=1).texts == ["a", "2"]


@pytest.fixture
def qa_dataset(monkeypatch):
    ds = datasets.Dataset.from_dict(
        {"question": ["q1", "q2", ""], "answer": ["a1", "a2", "a3"], "text": ["t1", "t2", "t3"]}
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda path, *a, **k: ds)


def test_text_columns_with_template_is_opt_in(qa_dataset):
    default = resolve_calibration_texts(calibration_dataset="x", n_sequences=5)
    assert default.texts == ["t1", "t2", "t3"]
    qa = resolve_calibration_texts(
        calibration_dataset={"path": "x", "text_columns": ["question", "answer"], "text_template": "Q: {question}\nA: {answer}"},
        n_sequences=5,
    )
    assert qa.texts == ["Q: q1\nA: a1", "Q: q2\nA: a2"]  # the row with an empty question is skipped
    assert qa.decoupled and qa.eval_samples == {}
    with pytest.raises(ValueError, match="only one"):
        resolve_calibration_texts(
            calibration_dataset={"path": "x", "text_column": "text", "text_columns": ["question"]}, n_sequences=2
        )


class _FakeTask:
    def __init__(self, with_target):
        self.eval_docs = [{"q": f"question {i}", "a": i % 2} for i in range(6)]
        self.with_target = with_target
        self.config = type("C", (), {"target_delimiter": " "})()

    def doc_to_text(self, doc):
        return doc["q"]

    def doc_to_target(self, doc):
        if not self.with_target:
            raise KeyError("no target")
        return doc["a"]

    def doc_to_choice(self, doc):
        return ["no", "yes"]


def _patch_harness(monkeypatch, task):
    import lm_eval.tasks as tasks

    class _TM:
        def load_task_or_group(self, names):
            return {names[0]: task}

    monkeypatch.setattr(tasks, "TaskManager", _TM)


def test_harness_include_target_default_off_and_maps_choice_index(monkeypatch):
    _patch_harness(monkeypatch, _FakeTask(True))
    off = resolve_calibration_texts(harness_tasks=["t"], calibration_split="test", n_sequences=3)
    assert all(t.startswith("question") and " yes" not in t and " no" not in t for t in off.texts)
    on = resolve_calibration_texts(harness_tasks=["t"], calibration_split="test", n_sequences=3, include_target=True)
    assert all(t.endswith((" yes", " no")) for t in on.texts) and on.notes == []
    assert len(on.texts) == len(off.texts)


def test_harness_include_target_falls_back_with_recorded_note(monkeypatch):
    _patch_harness(monkeypatch, _FakeTask(False))
    out = resolve_calibration_texts(harness_tasks=["t"], calibration_split="val", n_sequences=3, include_target=True)
    assert all(t.startswith("question") for t in out.texts)
    assert out.notes and "no gold target" in out.notes[0]
    prov = out.provenance()
    assert prov["n_sequences"] == len(out.texts) and prov["holdout_sha256"] and prov["notes"] == out.notes


def test_provenance_is_deterministic_and_tokenization_stats():
    a = resolve_calibration_texts(prompts=list(CALIB_TEXTS), n_sequences=1).provenance()
    b = resolve_calibration_texts(prompts=list(CALIB_TEXTS), n_sequences=1).provenance()
    assert a == b and a["holdout_sha256"] is None and a["decoupled_from_eval"] is True
    tok = local_tokenizer(CALIB_TEXTS, "right")
    stats = tokenization_stats(tok, CALIB_TEXTS, max_length=6)
    assert stats["n_sequences"] == len(CALIB_TEXTS) and stats["n_truncated"] >= 1
    assert 0.0 <= stats["pad_fraction"] < 1.0
