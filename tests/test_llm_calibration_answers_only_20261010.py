"""calibration_generated_answer.include_prompt=false: each calibration text is the generated answer alone."""

from __future__ import annotations

import hashlib
import json

import pytest
from _llm_fixtures import tiny_qwen2
from test_llm_calibration_generated_answer_20261007 import PROMPTS, _cache, _Counting, _tok

from merge_and_rebase.data.llm_calibration_generate import GeneratedAnswerConfig, _compose, parse_generated_answer_cfg


def _model(tok):
    return tiny_qwen2(vocab=len(tok))


def _sha(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def test_parse_include_prompt():
    assert parse_generated_answer_cfg({}).include_prompt is True
    cfg = parse_generated_answer_cfg({"include_prompt": False})
    assert cfg.include_prompt is False and cfg.uses_extensions
    assert not parse_generated_answer_cfg({"include_prompt": True}).uses_extensions
    with pytest.raises(ValueError, match="separator has no meaning"):
        parse_generated_answer_cfg({"include_prompt": False, "separator": "\n"})


def test_default_texts_and_provenance_unchanged(tmp_path):
    tok = _tok()
    path = tmp_path / "g.json"
    ga = {"max_new_tokens": 5, "cache_path": str(path)}
    c1 = _cache(tok, _Counting(_model(tok)), ga=ga)
    plain = list(c1.get().texts)
    answers = json.loads(path.read_text())["answers"]
    assert plain == [p + a for p, a in zip(PROMPTS, answers, strict=True)]
    c2 = _cache(tok, _Counting(_model(tok)), ga={**ga, "include_prompt": True})
    assert list(c2.get().texts) == plain
    assert c2.provenance()["generated_answer"] == c1.provenance()["generated_answer"] | {"cached": True}
    assert "include_prompt" not in c2.provenance()["generated_answer"]


def test_answers_only_reuses_cache(tmp_path):
    tok = _tok()
    path = tmp_path / "g.json"
    ga = {"max_new_tokens": 5, "cache_path": str(path)}
    _cache(tok, _Counting(_model(tok)), ga=ga).get()
    blob = json.loads(path.read_text())
    answers = blob["answers"]
    assert all(answers)
    f = _Counting(_model(tok))
    c = _cache(tok, f, ga={**ga, "include_prompt": False})
    got = list(c.get().texts)
    assert f.calls == 0 and got == answers
    prov = c.provenance()["generated_answer"]
    assert prov["include_prompt"] is False and prov["cached"] is True
    assert prov["answers_sha256"] == _sha(answers) and prov["texts_sha256"] == _sha(answers)
    assert json.loads(path.read_text()) == blob  # the cache file is not rewritten


def test_answers_only_recovers_answers_from_old_format_cache(tmp_path):
    tok = _tok()
    path = tmp_path / "g.json"
    ga = {"max_new_tokens": 5, "cache_path": str(path)}
    _cache(tok, _Counting(_model(tok)), ga=ga).get()
    blob = json.loads(path.read_text())
    answers = blob.pop("answers")
    path.write_text(json.dumps(blob))
    before = path.read_bytes()
    f = _Counting(_model(tok))
    c = _cache(tok, f, ga={**ga, "include_prompt": False})
    assert list(c.get().texts) == answers and f.calls == 0
    assert c.provenance()["generated_answer"]["answers_sha256"] == _sha(answers)
    assert path.read_bytes() == before


def test_empty_answer_raises_and_shuffle_composes():
    cfg = GeneratedAnswerConfig(include_prompt=False)
    with pytest.raises(ValueError, match="empty answers"):
        _compose(["p1", "p2"], ["a1", ""], cfg)
    assert _compose(["p1", "p2"], ["a1", "a2"], cfg) == ["a1", "a2"]
    shuf = _compose(["p1", "p2"], ["a1", "a2"], GeneratedAnswerConfig(include_prompt=False, shuffle_answers_seed=0))
    assert shuf == ["a2", "a1"]
