"""Characterization of ``resolve_calibration_texts`` (data/llm_calibration.py), fully offline.

Sources, in priority order: explicit prompts > HF dataset spec > evaluated lm-harness tasks.
The harness path carves a deterministic calibration/eval partition from ``task.eval_docs``
whose size is driven by ``n_sequences`` (the default sizing rule must not change).
"""

from __future__ import annotations

import datasets
import pytest

from merge_and_rebase.data import llm_calibration as lc
from merge_and_rebase.data.llm_calibration import CalibrationTexts, resolve_calibration_texts

# --------------------------------------------------------------------------- prompts


def test_prompts_take_priority_and_are_stringified():
    out = resolve_calibration_texts(
        prompts=["a", 2, "c"], calibration_dataset="ignored/dataset", harness_tasks=["ifeval"], n_sequences=1
    )
    assert out.texts == ["a", "2", "c"]
    assert out.source == "config['calibration_prompts']"
    assert out.eval_samples == {}
    assert len(out) == 3
    assert out.describe() == "3 sequences from config['calibration_prompts']"


def test_prompts_are_not_truncated_to_n_sequences():
    out = resolve_calibration_texts(prompts=["a", "b", "c"], n_sequences=1)
    assert out.texts == ["a", "b", "c"]


@pytest.mark.parametrize("n", [0, -3])
def test_nonpositive_n_sequences_rejected(n):
    with pytest.raises(ValueError, match="n_sequences must be > 0"):
        resolve_calibration_texts(prompts=["a"], n_sequences=n)


def test_no_source_raises():
    with pytest.raises(ValueError, match="No calibration source configured"):
        resolve_calibration_texts(n_sequences=4)


def test_empty_corpus_raises():
    with pytest.raises(ValueError, match="produced no text"):
        CalibrationTexts([], source="x")


# --------------------------------------------------------------------------- HF dataset


@pytest.fixture
def fake_load_dataset(monkeypatch):
    calls: list[tuple] = []
    registry: dict[str, datasets.Dataset] = {}

    def _load(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        return registry[path]

    monkeypatch.setattr(datasets, "load_dataset", _load)
    return calls, registry


def test_hf_dataset_auto_detects_text_column_and_caps_rows(fake_load_dataset):
    calls, registry = fake_load_dataset
    registry["org/ds"] = datasets.Dataset.from_dict(
        {"id": list(range(6)), "question": [f"q{i}" for i in range(6)], "answer": ["x"] * 6}
    )
    out = resolve_calibration_texts(calibration_dataset="org/ds", calibration_split="val", n_sequences=4)
    assert out.texts == ["q0", "q1", "q2", "q3"]
    assert out.source == "org/ds[val].question"
    assert out.eval_samples == {}
    assert calls == [("org/ds", (), {"split": "val"})]


def test_hf_dataset_text_column_and_spec_fields(fake_load_dataset):
    calls, registry = fake_load_dataset
    registry["org/ds"] = datasets.Dataset.from_dict(
        {"text": ["ignored"] * 3, "body": ["b0", "  ", "b2"], "n": [1, 2, 3]}
    )
    spec = {"path": "org/ds", "name": "cfg", "split": "train", "text_column": "body"}
    out = resolve_calibration_texts(calibration_dataset=spec, calibration_split="val", n_sequences=10)
    # Explicit column wins over the auto-detect candidate 'text'; blank rows are skipped.
    assert out.texts == ["b0", "b2"]
    assert out.source == "org/ds[train].body"
    assert calls == [("org/ds", ("cfg",), {"split": "train"})]


def test_hf_dataset_spec_accepts_dataset_and_config_aliases(fake_load_dataset):
    calls, registry = fake_load_dataset
    registry["org/ds"] = datasets.Dataset.from_dict({"sentence": ["s0", "s1"]})
    out = resolve_calibration_texts(calibration_dataset={"dataset": "org/ds", "config": "c"}, n_sequences=2)
    assert out.texts == ["s0", "s1"]
    assert calls[0][1] == ("c",)


def test_hf_dataset_candidate_column_priority(fake_load_dataset):
    _, registry = fake_load_dataset
    registry["org/ds"] = datasets.Dataset.from_dict({"Question": ["q"], "TEXT": ["t"]})
    assert resolve_calibration_texts(calibration_dataset="org/ds", n_sequences=1).texts == ["t"]


def test_hf_dataset_without_text_column_errors(fake_load_dataset):
    _, registry = fake_load_dataset
    registry["org/ds"] = datasets.Dataset.from_dict({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Could not infer a text column"):
        resolve_calibration_texts(calibration_dataset="org/ds", n_sequences=1)


def test_hf_dataset_spec_validation():
    with pytest.raises(ValueError, match="needs a 'path'"):
        resolve_calibration_texts(calibration_dataset={"split": "x"}, n_sequences=1)
    with pytest.raises(TypeError, match="path or a mapping"):
        resolve_calibration_texts(calibration_dataset=3, n_sequences=1)


# --------------------------------------------------------------------------- lm-harness


class _StubTask:
    def __init__(self, n_docs: int, tag: str = "d"):
        self.eval_docs = [{"i": i, "tag": tag} for i in range(n_docs)]

    def doc_to_text(self, doc):
        return f"{doc['tag']}-{doc['i']} prompt"


@pytest.fixture
def stub_tasks(monkeypatch):
    import lm_eval.tasks as lm_tasks

    tasks: dict[str, object] = {}

    class _Manager:
        def load_task_or_group(self, names):
            return {n: tasks[n] for n in names if n in tasks}

    monkeypatch.setattr(lm_tasks, "TaskManager", _Manager)
    return tasks


def _idx(texts, tag="d"):
    return sorted(int(t.split()[0].split("-")[1]) for t in texts if t.startswith(tag + "-"))


def test_harness_400_sequences_matches_ifeval_split(stub_tasks):
    """400 calibration sequences on a 541-doc single-split task leave exactly 141 eval docs."""
    stub_tasks["ifeval"] = _StubTask(541)
    out = resolve_calibration_texts(harness_tasks=["ifeval"], calibration_split="val", n_sequences=400)
    assert len(out) == 400
    assert len(out.eval_samples["ifeval"]) == 141
    calib = _idx(out.texts)
    assert sorted(calib + out.eval_samples["ifeval"]) == list(range(541))  # disjoint partition
    assert out.eval_samples["ifeval"] == sorted(out.eval_samples["ifeval"])
    assert out.source == "lm-harness ifeval[holdout]"
    assert out.describe() == "400 sequences from lm-harness ifeval[holdout] (held out -> ifeval: 141 eval docs)"


def test_harness_partition_is_deterministic_and_seed_dependent(stub_tasks):
    stub_tasks["t"] = _StubTask(50)
    kw = dict(harness_tasks=["t"], calibration_split="val", n_sequences=20)
    first = resolve_calibration_texts(seed=0, **kw)
    again = resolve_calibration_texts(seed=0, **kw)
    other = resolve_calibration_texts(seed=1, **kw)
    assert first.texts == again.texts and first.eval_samples == again.eval_samples
    assert first.eval_samples != other.eval_samples


@pytest.mark.parametrize(
    ("n_sequences", "n_calib", "n_eval"),
    [(1, 1, 49), (10, 10, 40), (49, 49, 1), (50, 49, 1), (400, 49, 1)],
)
def test_harness_n_sequences_determines_partition_and_always_leaves_one_eval_doc(
    stub_tasks, n_sequences, n_calib, n_eval
):
    stub_tasks["t"] = _StubTask(50)
    out = resolve_calibration_texts(harness_tasks=["t"], calibration_split="val", n_sequences=n_sequences)
    assert (len(out), len(out.eval_samples["t"])) == (n_calib, n_eval)


def test_harness_smaller_n_sequences_calibrates_on_prefix_of_same_shuffle(stub_tasks):
    stub_tasks["t"] = _StubTask(50)
    small = resolve_calibration_texts(harness_tasks=["t"], n_sequences=10, seed=3)
    large = resolve_calibration_texts(harness_tasks=["t"], n_sequences=20, seed=3)
    assert set(_idx(small.texts)) <= set(_idx(large.texts))


def test_harness_multiple_tasks_split_budget_per_task(stub_tasks):
    stub_tasks["a"] = _StubTask(30, "a")
    stub_tasks["b"] = _StubTask(30, "b")
    out = resolve_calibration_texts(harness_tasks=["a", "b"], n_sequences=11)  # ceil(11 / 2) = 6 per task
    assert len(_idx(out.texts, "a")) == len(_idx(out.texts, "b")) == 6
    assert {k: len(v) for k, v in out.eval_samples.items()} == {"a": 24, "b": 24}
    assert out.source == "lm-harness a+b[holdout]"


@pytest.mark.parametrize("split", ["val", "validation", "dev", "VAL"])
def test_harness_holdout_splits(stub_tasks, split):
    stub_tasks["t"] = _StubTask(20)
    assert resolve_calibration_texts(harness_tasks=["t"], calibration_split=split, n_sequences=5).eval_samples


def test_harness_other_split_calibrates_without_holdout(stub_tasks):
    stub_tasks["t"] = _StubTask(20)
    out = resolve_calibration_texts(harness_tasks=["t"], calibration_split="test", n_sequences=5)
    assert len(out) == 5
    assert out.eval_samples == {}
    assert out.source == "lm-harness t[test]"


def test_harness_skips_groups_and_empty_tasks_and_errors_when_nothing_left(stub_tasks):
    stub_tasks["t"] = _StubTask(10)
    stub_tasks["empty"] = _StubTask(0)
    out = resolve_calibration_texts(harness_tasks=["group_missing", "empty", "t"], n_sequences=3)
    # Characterized quirk: the per-task budget is ceil(n_sequences / len(tasks)) over ALL requested tasks,
    # including skipped ones, so skipped tasks shrink calibration below n_sequences (3 -> 1 here).
    assert len(out) == 1 and set(out.eval_samples) == {"t"}
    with pytest.raises(ValueError, match="yielded no calibration text"):
        resolve_calibration_texts(harness_tasks=["group_missing", "empty"], n_sequences=3)


def test_harness_reads_docs_via_eval_docs_only(stub_tasks):
    class _NoDocs:
        def doc_to_text(self, doc):  # pragma: no cover - never reached
            return "x"

    stub_tasks["t"] = _NoDocs()
    assert lc._task_docs(stub_tasks["t"]) == []
