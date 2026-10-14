"""Opt-in ``chat_template_from``: a base tokenizer without a chat template borrows another tokenizer's (2026-10-14)."""

from __future__ import annotations

import pytest

from merge_and_rebase.models.text_lm import TextBuildConfig, apply_chat_template_donor


class _Tok:
    def __init__(self, vocab: dict[str, int], chat_template: str | None) -> None:
        self._vocab = vocab
        self.chat_template = chat_template

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


VOCAB = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "a": 3}
TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['content'] }}<|im_end|>{% endfor %}"


def test_default_field_is_none():
    assert TextBuildConfig(model_name_or_path="x").chat_template_from is None


def test_missing_template_is_copied():
    tok = _Tok(VOCAB, None)
    apply_chat_template_donor(tok, _Tok(VOCAB, TEMPLATE), "donor")
    assert tok.chat_template == TEMPLATE


def test_identical_template_is_a_no_op():
    tok = _Tok(VOCAB, TEMPLATE)
    apply_chat_template_donor(tok, _Tok(VOCAB, TEMPLATE), "donor")
    assert tok.chat_template == TEMPLATE


def test_different_own_template_raises():
    tok = _Tok(VOCAB, "other")
    with pytest.raises(ValueError, match="different chat_template"):
        apply_chat_template_donor(tok, _Tok(VOCAB, TEMPLATE), "donor")
    assert tok.chat_template == "other"


def test_donor_without_template_raises():
    with pytest.raises(ValueError, match="no chat_template"):
        apply_chat_template_donor(_Tok(VOCAB, None), _Tok(VOCAB, None), "donor")


def test_vocabulary_mismatch_raises():
    tok = _Tok(VOCAB, None)
    with pytest.raises(ValueError, match="vocabulary differs"):
        apply_chat_template_donor(tok, _Tok({**VOCAB, "b": 4}, TEMPLATE), "donor")
    assert tok.chat_template is None


def test_smollm2_base_receives_instruct_template():
    transformers = pytest.importorskip("transformers")
    try:
        base = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M", local_files_only=True)
        inst = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("SmolLM2 tokenizers not in the local HF cache")
    assert base.chat_template is None
    apply_chat_template_donor(base, inst, "HuggingFaceTB/SmolLM2-360M-Instruct")
    msgs = [{"role": "user", "content": "hi"}]
    kw = {"tokenize": False, "add_generation_prompt": True}
    assert base.apply_chat_template(msgs, **kw) == inst.apply_chat_template(msgs, **kw)
