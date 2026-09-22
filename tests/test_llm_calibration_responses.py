"""Calibration text that includes a response, not only the prompt.

An instruct or reasoning fine-tune changes how the model *writes* its answer,
and a prompt-only bank reaches those positions only at the last prompt token.
Three ways to put response positions into the bank are pinned here:

- ``text_template`` on an HF ``calibration_dataset`` spec, which joins a
  dataset's own response column (GSM8K ``answer``) to its prompt;
- ``include_target`` on lm-harness calibration, which appends ``doc_to_target``
  and refuses tasks that have no text target (IFEval);
- ``append_generated_responses``, the ``align_with_gen_response`` path, which
  teacher-forces the source fine-tune's own greedy answer.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from merge_and_rebase.data.llm_calibration import (
    append_generated_responses,
    resolve_calibration_texts,
)
from merge_and_rebase.eval.llm_rebase import _parse_gen_response_cfg

_ROWS = [
    {"question": "What is 2+3?", "answer": "2+3=5\n#### 5"},
    {"question": "What is 4*2?", "answer": "4*2=8\n#### 8"},
    {"question": "What is 9-1?", "answer": "9-1=8\n#### 8"},
]


class _FakeDataset(list):
    column_names = ["question", "answer"]


@pytest.fixture
def fake_hf(monkeypatch):
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: _FakeDataset(_ROWS)
    monkeypatch.setitem(sys.modules, "datasets", module)


# --- text_template -----------------------------------------------------------


def test_text_template_joins_the_dataset_response(fake_hf):
    resolved = resolve_calibration_texts(
        calibration_dataset={"path": "openai/gsm8k", "text_template": "{question}\n{answer}"},
        n_sequences=2,
    )
    assert resolved.texts == ["What is 2+3?\n2+3=5\n#### 5", "What is 4*2?\n4*2=8\n#### 8"]
    assert "template" in resolved.source


def test_text_column_alone_keeps_the_historical_prompt_only_text(fake_hf):
    resolved = resolve_calibration_texts(
        calibration_dataset={"path": "openai/gsm8k", "text_column": "question"}, n_sequences=5
    )
    assert resolved.texts == [row["question"] for row in _ROWS]


def test_text_template_refuses_unknown_columns(fake_hf):
    with pytest.raises(ValueError, match=r"\['solution'\]"):
        resolve_calibration_texts(
            calibration_dataset={"path": "openai/gsm8k", "text_template": "{question} {solution}"},
            n_sequences=2,
        )


def test_text_template_and_text_column_are_exclusive(fake_hf):
    with pytest.raises(ValueError, match="not both"):
        resolve_calibration_texts(
            calibration_dataset={"path": "x", "text_column": "question", "text_template": "{question}"},
            n_sequences=2,
        )


# --- include_target (lm-harness) ---------------------------------------------


class _Task:
    def __init__(self, target):
        self._target = target
        self.config = types.SimpleNamespace(target_delimiter=" ")

    def doc_to_text(self, doc):
        return f"Q: {doc['q']}\nA:"

    def doc_to_target(self, doc):
        return self._target(doc)


@pytest.fixture
def fake_harness(monkeypatch):
    tasks = {}

    class _Manager:
        def load_task_or_group(self, names):
            return {name: tasks[name] for name in names}

    lm_eval = types.ModuleType("lm_eval")
    lm_eval_tasks = types.ModuleType("lm_eval.tasks")
    lm_eval_tasks.TaskManager = _Manager
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", lm_eval_tasks)
    monkeypatch.setattr(
        "merge_and_rebase.data.llm_calibration._task_docs",
        lambda task: [{"q": f"q{i}", "a": f"a{i}"} for i in range(4)],
    )
    return tasks


def test_include_target_appends_the_reference_answer(fake_harness):
    fake_harness["gsm8k"] = _Task(lambda doc: doc["a"])
    resolved = resolve_calibration_texts(
        harness_tasks=["gsm8k"], calibration_split="val", n_sequences=2, include_target=True
    )
    assert len(resolved.texts) == 2
    for text in resolved.texts:
        prompt, answer = text.rsplit(" ", 1)
        assert prompt.startswith("Q: q") and answer == "a" + prompt[4]
    # The held-out eval slice is unchanged by appending targets.
    assert len(resolved.eval_samples["gsm8k"]) == 2


def test_include_target_refuses_a_task_without_a_text_target(fake_harness):
    fake_harness["ifeval"] = _Task(lambda doc: 0)
    with pytest.raises(ValueError, match="no text reference target"):
        resolve_calibration_texts(
            harness_tasks=["ifeval"], calibration_split="val", n_sequences=2, include_target=True
        )


def test_include_target_is_refused_outside_harness_calibration(fake_hf):
    with pytest.raises(ValueError, match="only to lm-harness"):
        resolve_calibration_texts(
            calibration_dataset={"path": "openai/gsm8k"}, n_sequences=2, include_target=True
        )


# --- align_with_gen_response -------------------------------------------------


class _CharTokenizer:
    """One token per character; id 0 is pad. Enough to exercise left padding."""

    pad_token = "<pad>"
    eos_token = "<pad>"
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "right"
        self.seen_sides = []

    def __call__(self, texts, return_tensors=None, padding=False, **kwargs):
        self.seen_sides.append(self.padding_side)
        ids = [[ord(c) for c in t] for t in texts]
        width = max(len(r) for r in ids)
        pad = lambda r: ([0] * (width - len(r)) + r) if self.padding_side == "left" else (r + [0] * (width - len(r)))
        input_ids = torch.tensor([pad(r) for r in ids])
        return _Encoding(input_ids=input_ids, attention_mask=(input_ids != 0).long())

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<user>{messages[0]['content']}<assistant>"


class _Encoding(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class _EchoModel(nn.Module):
    """Greedy "response": the prompt's last real character, repeated, then pad."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls = []

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, pad_token_id):
        self.calls.append({"do_sample": do_sample, "max_new_tokens": max_new_tokens})
        last = input_ids[:, -1:]  # left padding puts the last real token here
        body = last.repeat(1, 2)
        tail = torch.full((input_ids.shape[0], max_new_tokens - 2), pad_token_id)
        return torch.cat([input_ids, body, tail], dim=1)


def test_generated_responses_are_appended_to_each_prompt():
    tokenizer, model = _CharTokenizer(), _EchoModel()
    texts, stats = append_generated_responses(
        ["ab", "xyz", "q"], model=model, tokenizer=tokenizer, max_new_tokens=4, batch_size=2, device="cpu"
    )
    assert texts == ["abbb", "xyzzz", "qqq"]
    assert stats["n"] == 3 and stats["mean_response_tokens"] == 2.0 and stats["hit_max_new_tokens"] == 0
    assert all(call["do_sample"] is False for call in model.calls)
    # Generated left-padded, and the tokenizer's own side is restored for the loader.
    assert set(tokenizer.seen_sides) == {"left"}
    assert tokenizer.padding_side == "right"


def test_generated_responses_follow_the_chat_template_when_asked():
    texts, stats = append_generated_responses(
        ["hi"], model=_EchoModel(), tokenizer=_CharTokenizer(), max_new_tokens=3,
        apply_chat_template=True, device="cpu",
    )
    assert texts == ["<user>hi<assistant>>>"]
    assert stats["apply_chat_template"] is True


def test_gen_response_config_parsing():
    assert _parse_gen_response_cfg(None, default_chat_template=False) is None
    assert _parse_gen_response_cfg(False, default_chat_template=False) is None
    assert _parse_gen_response_cfg({"enabled": False}, default_chat_template=False) is None
    parsed = _parse_gen_response_cfg(True, default_chat_template=True)
    assert parsed == {
        "model": None, "max_new_tokens": 256, "batch_size": 8,
        "apply_chat_template": True, "cache_path": None,
    }
    assert _parse_gen_response_cfg({"max_new_tokens": 64}, default_chat_template=False)["max_new_tokens"] == 64
    with pytest.raises(ValueError, match="unknown align_with_gen_response"):
        _parse_gen_response_cfg({"max_tokens": 64}, default_chat_template=False)
