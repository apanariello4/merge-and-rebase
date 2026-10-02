"""Snapshot of the 13 private helpers that ``eval/llm_merge.py`` duplicates from ``eval/llm_common.py``.

Every case runs against BOTH the ``llm_merge._x`` copy and the ``llm_common.x`` original with the same
literal expectation, so the later dedupe (S12) is gated by "same input -> same output" on either side.
Today the two copies are behaviourally identical (they differ only in comments).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from merge_and_rebase.data.text_loaders import NLITaskData
from merge_and_rebase.eval import llm_common as common
from merge_and_rebase.eval import llm_merge as merge

NAMES = [
    "resolve_tasks",
    "resolve_suite_name",
    "resolve_eval_mode",
    "default_prompt_for_task",
    "resolve_task_mask_class",
    "head_class_ids_for_task",
    "load_task_heads",
    "task_head_tensor_for_param",
    "task_head_param_overrides",
    "inject_task_head",
    "to_unit_acc",
    "normalized_acc",
    "resolve_fine_tuned_acc",
]
ALL_TASKS = ["snli", "mnli", "sick", "qnli", "rte", "scitail"]


@pytest.fixture(params=["llm_merge", "llm_common"])
def h(request):
    """Namespace exposing the 13 helpers under their public names, from either module."""

    class _NS:
        pass

    ns = _NS()
    for name in NAMES:
        setattr(ns, name, getattr(merge, "_" + name) if request.param == "llm_merge" else getattr(common, name))
    return ns


def test_all_thirteen_helpers_exist_in_both_modules():
    assert len(NAMES) == 13
    for name in NAMES:
        assert callable(getattr(merge, "_" + name)), name
        assert callable(getattr(common, name)), name
    assert merge.NLI_SUITES == common.NLI_SUITES == {"nli6": tuple(ALL_TASKS)}


def test_resolve_tasks(h):
    assert h.resolve_tasks(None) == ALL_TASKS
    assert h.resolve_tasks(" ALL ") == ALL_TASKS
    assert h.resolve_tasks("SNLI, mnli") == ["snli", "mnli"]
    assert h.resolve_tasks([" RTE ", "Sick"]) == ["rte", "sick"]
    assert h.resolve_tasks(("qnli",), suite_name="nli6") == ["qnli"]
    with pytest.raises(ValueError, match=r"Unknown tasks: \['foo'\]"):
        h.resolve_tasks("snli,foo")
    with pytest.raises(ValueError, match="Unknown tasks for suite 'nli6': \\['foo'\\]"):
        h.resolve_tasks(["foo"], suite_name="nli6")
    with pytest.raises(ValueError, match="'all', a CSV string, or a list"):
        h.resolve_tasks(3)


def test_resolve_suite_name(h):
    assert h.resolve_suite_name(None) is None
    assert h.resolve_suite_name("  ") is None
    assert h.resolve_suite_name(" NLI6 ") == "nli6"
    with pytest.raises(ValueError, match=r"Unknown suite 'bogus'. Available: \['nli6'\]"):
        h.resolve_suite_name("bogus")


def test_resolve_eval_mode(h):
    assert h.resolve_eval_mode("auto", None) == "prompt"
    assert h.resolve_eval_mode("AUTO", "heads.pt") == "head_logits"
    assert h.resolve_eval_mode(" Prompt ", "heads.pt") == "prompt"
    assert h.resolve_eval_mode("head_logits", None) == "head_logits"
    with pytest.raises(ValueError, match="eval_mode must be one of: auto, prompt, head_logits"):
        h.resolve_eval_mode("logits", None)


def test_default_prompt_for_task(h):
    data = NLITaskData(
        task="rte", examples=[], labels=["0", "1"], label_texts=["entailment", "not_entailment"], meta={}
    )
    assert h.default_prompt_for_task(data) == (
        "You are an NLI classifier.\n"
        "Given a premise and a hypothesis, predict one label from: entailment, not_entailment.\n"
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n"
        "Label:"
    )


def test_resolve_task_mask_class(h):
    assert h.resolve_task_mask_class(None) == {}
    assert h.resolve_task_mask_class({" RTE ": "1", "snli": None, "  ": 2, "mnli": 0}) == {
        "rte": 1,
        "snli": None,
        "mnli": 0,
    }
    with pytest.raises(ValueError, match="task_mask_class must be a dict"):
        h.resolve_task_mask_class([1])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(task="snli", task_num_labels=3, head_num_labels=3, masked_class=None), [0, 1, 2]),
        (dict(task="QNLI", task_num_labels=2, head_num_labels=3, masked_class=None), [0, 2]),
        (dict(task="rte", task_num_labels=2, head_num_labels=4, masked_class=None), [0, 2]),
        (dict(task="scitail", task_num_labels=2, head_num_labels=3, masked_class=None), [0, 1]),
        (dict(task="mnli", task_num_labels=2, head_num_labels=3, masked_class=1), [0, 2]),
        (dict(task="snli", task_num_labels=3, head_num_labels=4, masked_class=0), [1, 2, 3]),
    ],
)
def test_head_class_ids_for_task(h, kwargs, expected):
    assert h.head_class_ids_for_task(**kwargs) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(task="x", task_num_labels=0, head_num_labels=3, masked_class=None), "Invalid task_num_labels for 'x': 0"),
        (dict(task="x", task_num_labels=2, head_num_labels=0, masked_class=None), "Invalid head_num_labels for 'x': 0"),
        (dict(task="mnli", task_num_labels=2, head_num_labels=3, masked_class=None), "Could not infer head_class_ids"),
        (dict(task="x", task_num_labels=5, head_num_labels=3, masked_class=None), "Could not infer head_class_ids"),
        (dict(task="x", task_num_labels=2, head_num_labels=3, masked_class=3), "Invalid masked class for task 'x': 3"),
        (dict(task="x", task_num_labels=3, head_num_labels=3, masked_class=0), "Mask-derived class ids"),
    ],
)
def test_head_class_ids_for_task_errors(h, kwargs, match):
    with pytest.raises(ValueError, match=match):
        h.head_class_ids_for_task(**kwargs)


def test_load_task_heads(h, tmp_path):
    path = tmp_path / "heads.pt"
    torch.save({" SNLI ": torch.ones(2), "Rte": {"w": torch.zeros(1)}}, path)
    out = h.load_task_heads(str(path))
    assert sorted(out) == ["rte", "snli"]
    assert torch.equal(out["snli"], torch.ones(2))
    bad = tmp_path / "bad.pt"
    torch.save([1, 2], bad)
    with pytest.raises(ValueError, match="task_heads file must contain a dict"):
        h.load_task_heads(str(bad))


class _HeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_head = nn.Module()
        self.classification_head.out_proj = nn.Linear(4, 3)
        self.backbone = nn.Linear(4, 4)


def _zeroed_model():
    model = _HeadModel()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def test_task_head_tensor_for_param(h):
    param = torch.zeros(3, 4)
    full = torch.arange(12.0).reshape(3, 4)
    same = h.task_head_tensor_for_param(
        task_key="t", name="classification_head.out_proj.weight", param=param, value=full
    )
    assert torch.equal(same, full)
    sub = torch.arange(8.0).reshape(2, 4) + 1
    mapped = h.task_head_tensor_for_param(
        task_key="t", name="classification_head.out_proj.weight", param=param, value=sub, head_class_ids=[0, 2]
    )
    assert torch.equal(mapped[0], sub[0]) and torch.equal(mapped[2], sub[1]) and not mapped[1].any()
    prefix = h.task_head_tensor_for_param(
        task_key="t", name="classification_head.out_proj.weight", param=param, value=sub
    )
    assert torch.equal(prefix[:2], sub) and not prefix[2].any()
    bias = h.task_head_tensor_for_param(
        task_key="t",
        name="x.classification_head.out_proj.bias",
        param=torch.zeros(3),
        value=torch.tensor([5.0, 6.0]),
        head_class_ids=[2, 0],
    )
    assert bias.tolist() == [6.0, 0.0, 5.0]
    assert not param.any()  # input param untouched (clone)
    with pytest.raises(ValueError, match="must be unique"):
        h.task_head_tensor_for_param(
            task_key="t", name="classification_head.out_proj.weight", param=param, value=sub, head_class_ids=[1, 1]
        )
    with pytest.raises(ValueError, match="out of range"):
        h.task_head_tensor_for_param(
            task_key="t", name="classification_head.out_proj.weight", param=param, value=sub, head_class_ids=[0, 3]
        )
    with pytest.raises(ValueError, match="Head shape mismatch"):
        h.task_head_tensor_for_param(task_key="t", name="backbone.weight", param=param, value=sub)


def test_task_head_param_overrides_dict_and_tensor_payloads(h):
    model = _zeroed_model()
    w = torch.ones(3, 4)
    out = h.task_head_param_overrides(
        model=model,
        task="T",
        task_heads={"t": {"classification_head.out_proj.weight": w, "ignored": "str"}},
        head_key_pattern="classification_head",
    )
    assert list(out) == ["classification_head.out_proj.weight"] and torch.equal(
        out["classification_head.out_proj.weight"], w
    )
    # suffix match through the pattern-restricted names
    out = h.task_head_param_overrides(
        model=model,
        task="t",
        task_heads={"t": {"out_proj.bias": torch.ones(3)}},
        head_key_pattern="classification_head",
    )
    assert list(out) == ["classification_head.out_proj.bias"]
    # tensor payload: unique shape match under the pattern
    out = h.task_head_param_overrides(
        model=model, task="t", task_heads={"t": w}, head_key_pattern="classification_head"
    )
    assert list(out) == ["classification_head.out_proj.weight"]
    assert not any(p.any() for p in model.parameters())  # overrides are returned, not applied
    with pytest.raises(KeyError, match="not found in task_heads"):
        h.task_head_param_overrides(model=model, task="zzz", task_heads={}, head_key_pattern="p")
    with pytest.raises(ValueError, match="must be a Tensor or dict"):
        h.task_head_param_overrides(model=model, task="t", task_heads={"t": 3}, head_key_pattern="p")
    with pytest.raises(KeyError, match="No parameter match"):
        h.task_head_param_overrides(
            model=model, task="t", task_heads={"t": {"nope.weight": w}}, head_key_pattern="classification_head"
        )
    with pytest.raises(ValueError, match="Could not uniquely match tensor head"):
        h.task_head_param_overrides(
            model=model, task="t", task_heads={"t": torch.ones(7, 7)}, head_key_pattern="classification_head"
        )


def test_inject_task_head_copies_in_place(h):
    model = _zeroed_model()
    w = torch.arange(8.0).reshape(2, 4)
    h.inject_task_head(
        model=model,
        task="t",
        task_heads={"t": {"classification_head.out_proj.weight": w}},
        head_key_pattern="classification_head",
        head_class_ids=[0, 2],
    )
    got = model.classification_head.out_proj.weight
    assert torch.equal(got[0], w[0]) and torch.equal(got[2], w[1]) and not got[1].any()
    assert not model.backbone.weight.any()
    h.inject_task_head(
        model=model, task="t", task_heads={"t": torch.full((3, 4), 2.0)}, head_key_pattern="classification_head"
    )
    assert bool((model.classification_head.out_proj.weight == 2.0).all())
    with pytest.raises(KeyError, match="not found in task_heads"):
        h.inject_task_head(model=model, task="zzz", task_heads={}, head_key_pattern="p")
    with pytest.raises(KeyError, match="No parameter match"):
        h.inject_task_head(model=model, task="t", task_heads={"t": {"nope": torch.ones(1)}}, head_key_pattern="p")


def test_inject_task_head_ambiguous_suffix(h):
    model = nn.Module()
    model.a = nn.Module()
    model.b = nn.Module()
    model.a.head = nn.Linear(2, 2)
    model.b.head = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Ambiguous suffix match"):
        h.inject_task_head(
            model=model, task="t", task_heads={"t": {"head.weight": torch.ones(2, 2)}}, head_key_pattern="head"
        )


def test_accuracy_helpers(h):
    assert h.to_unit_acc(46.7) == pytest.approx(0.467)
    assert h.to_unit_acc(0.467) == 0.467
    assert h.to_unit_acc(1.0) == 1.0
    assert h.to_unit_acc("80") == pytest.approx(0.8)
    assert h.normalized_acc(0.4, 80) == pytest.approx(0.5)
    assert h.normalized_acc(0.4, 0.8) == pytest.approx(0.5)
    assert h.normalized_acc(0.4, 0) == 0.0
    assert h.normalized_acc(0.4, -1.0) == 0.0


def test_resolve_fine_tuned_acc(h, capsys):
    assert h.resolve_fine_tuned_acc(cfg={}, tasks=["snli"]) is None
    out = h.resolve_fine_tuned_acc(
        cfg={"fine_tuned_acc": {" SNLI ": "91.5", "bad": "x", "rte": 70}}, tasks=["snli", "mnli"]
    )
    assert out == {"snli": 91.5, "rte": 70.0}
    assert "fine_tuned_acc missing tasks ['mnli']" in capsys.readouterr().out
    with pytest.raises(ValueError, match="fine_tuned_acc must be a dict"):
        h.resolve_fine_tuned_acc(cfg={"fine_tuned_acc": [1]}, tasks=[])
