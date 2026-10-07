"""Opt-in chat-template rendering of LLM calibration text (data/llm_calibration.py), fully offline.

Flag off must stay byte-identical (texts, token ids, hold-out); flag on renders through the tokenizer's chat template
(lm-eval's ``fewshot_context`` for harness docs), keeps the same hold-out, tokenizes without extra special tokens and
requires source/target tokenizers to agree.
"""

from __future__ import annotations

import hashlib
import json

import datasets
import pytest
from _llm_fixtures import local_tokenizer

from merge_and_rebase.data.llm_calibration import (
    _render_user_text,
    build_text_calibration_loader,
    check_calibration_tokenizers,
    make_chat_template_fn,
    resolve_calibration_texts,
    tokenizer_chat_template_sha256,
)

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
VOCAB_TEXTS = ["alpha beta gamma delta", "<|system|> <|user|> <|assistant|> <|end|> be brief q0 q1 q2 q3 q4 q5"]


def _tok(template: str | None = TEMPLATE, extra: str = ""):
    tok = local_tokenizer(VOCAB_TEXTS + [extra])
    tok.chat_template = template
    return tok


class _StubTask:
    def __init__(self, n_docs: int, tag: str = "d"):
        self.eval_docs = [{"i": i, "tag": tag} for i in range(n_docs)]
        self.seeds: list[int] = []
        self.fewshot_calls: list[dict] = []

    def doc_to_text(self, doc):
        return f"{doc['tag']}-{doc['i']} prompt"

    def set_fewshot_seed(self, seed=None):
        self.seeds.append(seed)

    def fewshot_context(
        self,
        doc,
        num_fewshot,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        gen_prefix=None,
    ):
        self.fewshot_calls.append(
            {"n": num_fewshot, "sys": system_instruction, "multi": fewshot_as_multiturn, "act": apply_chat_template}
        )
        messages = ([{"role": "system", "content": system_instruction}] if system_instruction else []) + [
            {"role": "user", "content": self.doc_to_text(doc)}
        ]
        return chat_template(messages)


@pytest.fixture
def stub_tasks(monkeypatch):
    import lm_eval.tasks as lm_tasks

    tasks: dict[str, object] = {}

    class _Manager:
        def load_task_or_group(self, names):
            return {n: tasks[n] for n in names if n in tasks}

    monkeypatch.setattr(lm_tasks, "TaskManager", _Manager)
    return tasks


def test_flag_off_harness_texts_and_hashes_unchanged(stub_tasks):
    stub_tasks["t"] = _StubTask(30)
    out = resolve_calibration_texts(harness_tasks=["t"], n_sequences=8, seed=3)
    # Re-derive with the historical algorithm (shuffle of range(n), seed, first n_calib sorted).
    import random

    order = list(range(30))
    random.Random(3).shuffle(order)
    expected = [f"d-{i} prompt" for i in sorted(order[:8])]
    assert list(out.texts) == expected
    assert type(out.texts) is list
    prov = out.provenance()
    assert prov["corpus_sha256"] == hashlib.sha256("\x00".join(expected).encode()).hexdigest()
    assert (
        prov["holdout_sha256"]
        == hashlib.sha256(json.dumps({"t": sorted(order[8:])}, sort_keys=True).encode()).hexdigest()
    )
    assert prov["chat_template_applied"] is False
    assert prov["chat_template_sha256"] is None


def test_flag_off_token_ids_identical_to_plain_tokenizer_call():
    tok = _tok()
    texts = ["alpha beta", "gamma delta alpha"]
    loader = build_text_calibration_loader(tokenizer=tok, texts=texts, batch_size=2, max_length=8)
    ref = tok(texts, truncation=True, max_length=8, padding="max_length")
    batch = next(iter(loader))
    assert batch["input_ids"].tolist() == ref["input_ids"]


def test_flag_on_holdout_identical_and_text_rendered_by_lm_eval_path(stub_tasks):
    tok = _tok()
    stub_tasks["t"] = _StubTask(30)
    off = resolve_calibration_texts(harness_tasks=["t"], n_sequences=8, seed=3)
    stub_tasks["t"] = task = _StubTask(30)
    on = resolve_calibration_texts(
        harness_tasks=["t"],
        n_sequences=8,
        seed=3,
        apply_chat_template=True,
        tokenizer=tok,
        system_instruction="be brief",
        num_fewshot={"t": 2},
    )
    assert on.eval_samples == off.eval_samples
    assert on.provenance()["holdout_sha256"] == off.provenance()["holdout_sha256"]
    expected = [
        tok.apply_chat_template(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for t in off.texts
    ]
    assert list(on.texts) == expected
    assert on.texts.add_special_tokens is False
    assert {c["n"] for c in task.fewshot_calls} == {2}
    assert all(c["multi"] and c["act"] and c["sys"] == "be brief" for c in task.fewshot_calls)
    assert task.seeds == [1234]
    prov = on.provenance()
    assert prov["chat_template_applied"] is True
    assert prov["rendered_texts_sha256"] != off.provenance()["rendered_texts_sha256"]


def test_flag_on_dataset_path_equals_tokenizer_template(monkeypatch):
    tok = _tok()
    ds = datasets.Dataset.from_dict({"text": ["q0 q1", "q2 q3", "q4 q5"]})
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: ds)
    plain = resolve_calibration_texts(calibration_dataset="org/ds", n_sequences=3)
    on = resolve_calibration_texts(calibration_dataset="org/ds", n_sequences=3, apply_chat_template=True, tokenizer=tok)
    assert list(on.texts) == [
        tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
        for t in plain.texts
    ]
    prompts = resolve_calibration_texts(
        prompts=["q0"], n_sequences=1, apply_chat_template=True, tokenizer=tok, system_instruction="be brief"
    )
    assert prompts.texts[0].startswith("<|system|> be brief")


def test_flag_on_loader_skips_extra_special_tokens():
    tok = _tok()
    tok.add_special_tokens({"bos_token": "<|bos|>"})
    on = resolve_calibration_texts(prompts=["q0 q1"], n_sequences=1, apply_chat_template=True, tokenizer=tok)
    batch = next(iter(build_text_calibration_loader(tokenizer=tok, texts=on.texts, batch_size=1, max_length=16)))
    ref = tok(list(on.texts), add_special_tokens=False, padding="max_length", max_length=16, truncation=True)
    assert batch["input_ids"].tolist() == ref["input_ids"]


def test_include_target_with_chat_template_not_implemented():
    with pytest.raises(NotImplementedError):
        resolve_calibration_texts(
            prompts=["a"], n_sequences=1, include_target=True, apply_chat_template=True, tokenizer=_tok()
        )


def test_missing_template_or_tokenizer_rejected():
    with pytest.raises(ValueError, match="no chat_template"):
        resolve_calibration_texts(prompts=["a"], n_sequences=1, apply_chat_template=True, tokenizer=_tok(None))
    with pytest.raises(ValueError, match="needs the source tokenizer"):
        resolve_calibration_texts(prompts=["a"], n_sequences=1, apply_chat_template=True)


def test_source_target_tokenizer_check():
    src, tgt = _tok(), _tok()
    texts = ["<|user|> q0 q1 <|assistant|>"]
    sha = check_calibration_tokenizers(src, tgt, texts, add_special_tokens=False)
    assert sha == {"source": tokenizer_chat_template_sha256(src), "target": tokenizer_chat_template_sha256(tgt)}
    with pytest.raises(ValueError, match="different chat_template"):
        check_calibration_tokenizers(src, _tok(TEMPLATE + " "), texts, add_special_tokens=False)
    # A different template string that renders differently is rejected even with a system instruction.
    with pytest.raises(ValueError, match="render the same conversation differently"):
        check_calibration_tokenizers(
            src,
            _tok(TEMPLATE + " "),
            texts,
            add_special_tokens=False,
            system_instruction="You are a helpful assistant.",
        )
    other_vocab = local_tokenizer(["zzz yyy"])  # different vocabulary -> different ids, same template
    other_vocab.chat_template = TEMPLATE
    with pytest.raises(ValueError, match="disagree on input_ids"):
        check_calibration_tokenizers(src, other_vocab, texts, add_special_tokens=False)


def test_qwen_base_vs_math_templates_need_explicit_system_instruction():
    """Qwen2.5 base and Math templates differ only in their default system prompt (real cached tokenizers)."""
    transformers = pytest.importorskip("transformers")
    try:
        base = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", local_files_only=True)
        math = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen2.5 tokenizers not in the local HF cache")
    system = "You are a helpful assistant."
    text = _render_user_text(make_chat_template_fn(math), "Question: 2+2?\nAnswer:", system)
    with pytest.raises(ValueError, match="no system instruction"):
        check_calibration_tokenizers(math, base, [text], add_special_tokens=False)
    sha = check_calibration_tokenizers(math, base, [text], add_special_tokens=False, system_instruction=system)
    assert sha["source"] != sha["target"]
    assert text == _render_user_text(make_chat_template_fn(base), "Question: 2+2?\nAnswer:", system)
