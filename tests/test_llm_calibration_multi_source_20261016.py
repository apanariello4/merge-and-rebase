"""Opt-in list of calibration_dataset specs: equal quotas, round-robin interleave; one spec unchanged (2026-10-16)."""

from __future__ import annotations

import json

import pytest

from merge_and_rebase.data.llm_calibration import _from_hf_dataset, resolve_calibration_texts


def _corpus(tmp_path, name, n):
    d = tmp_path / name
    d.mkdir()
    with open(d / "train.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"question": f"{name}-q{i}", "other": i}) + "\n")
    return str(d)


def _spec(path):
    return {"path": path, "split": "train", "text_columns": ["question"], "text_template": "Q: {question}"}


def test_single_spec_unchanged_by_list_support(tmp_path):
    a = _corpus(tmp_path, "fr", 10)
    single = _from_hf_dataset(_spec(a), calibration_split="val", n_sequences=4)
    assert single.texts == [f"Q: fr-q{i}" for i in range(4)]
    one_list = _from_hf_dataset([_spec(a)], calibration_split="val", n_sequences=4)
    assert one_list.texts == single.texts


def test_list_round_robin_equal_quotas(tmp_path):
    specs = [_spec(_corpus(tmp_path, lang, 10)) for lang in ("fr", "es", "de")]
    out = _from_hf_dataset(specs, calibration_split="val", n_sequences=7)
    # quotas 3, 2, 2; interleaved fr, es, de, fr, es, de, fr
    assert out.texts == ["Q: fr-q0", "Q: es-q0", "Q: de-q0", "Q: fr-q1", "Q: es-q1", "Q: de-q1", "Q: fr-q2"]
    assert out.source.count(" + ") == 2 and "fr" in out.source and "de" in out.source


def test_list_source_too_small_raises(tmp_path):
    specs = [_spec(_corpus(tmp_path, "fr", 10)), _spec(_corpus(tmp_path, "es", 1))]
    with pytest.raises(ValueError, match="needs"):
        _from_hf_dataset(specs, calibration_split="val", n_sequences=6)


def test_empty_list_raises():
    with pytest.raises(ValueError):
        _from_hf_dataset([], calibration_split="val", n_sequences=4)


def test_resolve_calibration_texts_accepts_list(tmp_path):
    specs = [_spec(_corpus(tmp_path, lang, 5)) for lang in ("fr", "es")]
    res = resolve_calibration_texts(calibration_dataset=specs, n_sequences=4)
    assert res.texts == ["Q: fr-q0", "Q: es-q0", "Q: fr-q1", "Q: es-q1"]
