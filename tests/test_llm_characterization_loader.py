"""Characterization of ``_build_text_calibration_loader`` (eval/llm_rebase.py).

Pins today's behaviour: labels are -100 exactly where attention_mask == 0 (never
derived from ``input_ids == pad_id``, because Qwen's pad id is also a content
token), and sample ids are stable across builds.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, PAD, local_tokenizer, real_qwen_tokenizer_or_skip

from merge_and_rebase.eval.llm_rebase import _build_text_calibration_loader


def _batches(loader):
    return list(loader)


def _check_labels(batch):
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    assert ids.shape == mask.shape == labels.shape
    assert torch.equal(labels == -100, mask == 0)
    keep = mask == 1
    assert torch.equal(labels[keep], ids[keep])


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("max_length", [12, 24])
def test_local_tokenizer_labels_follow_attention_mask(side, max_length):
    tok = local_tokenizer(CALIB_TEXTS, side)
    loader = _build_text_calibration_loader(tokenizer=tok, texts=CALIB_TEXTS, batch_size=4, max_length=max_length)
    batches = _batches(loader)
    assert sum(b["input_ids"].shape[0] for b in batches) == len(CALIB_TEXTS)
    for b in batches:
        assert b["input_ids"].shape[1] == max_length
        _check_labels(b)
    n_content = sum(int(b["attention_mask"].sum()) for b in batches)
    assert n_content == sum(len(t.split()) for t in CALIB_TEXTS)
    # Padding side is honoured: pads trail (right) or lead (left) every row.
    for b in batches:
        for row in b["attention_mask"]:
            idx = row.nonzero().flatten()
            if side == "right":
                assert int(idx[0]) == 0
            else:
                assert int(idx[-1]) == max_length - 1


@pytest.mark.parametrize("side", ["right", "left"])
def test_local_tokenizer_pad_id_as_content_keeps_label(side):
    """A pad-id token inside the text is content: attention_mask 1 and a real label."""
    texts = ["alpha " + PAD + " beta", "gamma delta"]
    tok = local_tokenizer(texts, side)
    batch = _batches(_build_text_calibration_loader(tokenizer=tok, texts=texts, batch_size=2, max_length=8))[0]
    pad_id = tok.pad_token_id
    content_pad = (batch["input_ids"] == pad_id) & (batch["attention_mask"] == 1)
    assert int(content_pad.sum()) == 1
    assert torch.equal(batch["labels"][content_pad], torch.tensor([pad_id]))
    _check_labels(batch)


@pytest.mark.parametrize("side", ["right", "left"])
def test_real_qwen_tokenizer_labels_follow_attention_mask(side):
    tok = real_qwen_tokenizer_or_skip(side)
    texts = ["The quick brown fox.", "Hello", "A somewhat longer calibration sentence about decoders."]
    batch = _batches(_build_text_calibration_loader(tokenizer=tok, texts=texts, batch_size=3, max_length=24))[0]
    _check_labels(batch)
    assert int((batch["labels"] == -100).sum()) == int((batch["attention_mask"] == 0).sum()) > 0


@pytest.mark.parametrize("side", ["right", "left"])
def test_real_qwen_endoftext_as_content_is_kept(side):
    """pad id == '<|endoftext|>' for Qwen2.5: as content it keeps mask 1 and its label."""
    tok = real_qwen_tokenizer_or_skip(side)
    texts = ["first doc<|endoftext|>second doc", "short"]
    batch = _batches(_build_text_calibration_loader(tokenizer=tok, texts=texts, batch_size=2, max_length=16))[0]
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    assert tok.pad_token_id == eot
    content_eot = (batch["input_ids"] == eot) & (batch["attention_mask"] == 1)
    assert int(content_eot.sum()) == 1
    assert int(batch["labels"][content_eot]) == eot
    _check_labels(batch)
    # ...while the same id at padded positions is masked out of the labels.
    padded_eot = (batch["input_ids"] == eot) & (batch["attention_mask"] == 0)
    assert int(padded_eot.sum()) > 0
    assert bool((batch["labels"][padded_eot] == -100).all())


def test_sample_ids_stable_and_shared_across_tokenizers():
    tok_a = local_tokenizer(CALIB_TEXTS, "right")
    tok_b = local_tokenizer(CALIB_TEXTS + ["extra words"], "left")
    first = _build_text_calibration_loader(tokenizer=tok_a, texts=CALIB_TEXTS, batch_size=2, max_length=12)
    again = _build_text_calibration_loader(tokenizer=tok_a, texts=CALIB_TEXTS, batch_size=2, max_length=24)
    other = _build_text_calibration_loader(tokenizer=tok_b, texts=CALIB_TEXTS, batch_size=3, max_length=12)
    ids = first.dataset.sample_ids
    assert ids == again.dataset.sample_ids == other.dataset.sample_ids
    assert len(ids) == len(set(ids)) == len(CALIB_TEXTS)
    assert all(sid.startswith(f"{i}:") and len(sid.split(":")[1]) == 12 for i, sid in enumerate(ids))
    # Stable value (sha1 of the text), not a salted str.__hash__.
    assert ids[2] == "2:" + hashlib.sha1(CALIB_TEXTS[2].encode()).hexdigest()[:12]


def test_loader_rejects_empty_texts():
    with pytest.raises(ValueError, match="at least one text"):
        _build_text_calibration_loader(tokenizer=local_tokenizer(CALIB_TEXTS), texts=[], batch_size=2, max_length=8)


def test_truncation_keeps_mask_one_and_labels():
    tok = local_tokenizer(CALIB_TEXTS, "right")
    batch = _batches(_build_text_calibration_loader(tokenizer=tok, texts=CALIB_TEXTS[:1], batch_size=1, max_length=4))[
        0
    ]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert torch.equal(batch["labels"], batch["input_ids"])
