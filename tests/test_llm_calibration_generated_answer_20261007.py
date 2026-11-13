"""Opt-in generated-answer calibration (data/llm_calibration_generate.py + TextCalibrationCache), CPU, offline.

Off = byte-identical texts/hashes; on = prompt + the source fine-tune's greedy answer, replaced once inside the cache.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import local_tokenizer, perturbed_copy, tiny_qwen2
from test_llm_tuned_config_mismatch_20261007 import _save_tuned
from transformers import GenerationConfig

from merge_and_rebase.data.llm_calibration import CalibrationTexts
from merge_and_rebase.data.llm_calibration_generate import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    parse_generated_answer_cfg,
)
from merge_and_rebase.eval.llm_rebase import context as ctx_mod
from merge_and_rebase.eval.llm_rebase.context import TextCalibrationCache, tuned_generator_factory
from merge_and_rebase.io.text_checkpoints import load_aligned_tuned_from_ref
from merge_and_rebase.models.text_lm import TextBuildConfig

PROMPTS = ["w1 w2 w3", "w4 w5", "w6 w7 w8 w9", "w10 w11 w12"]
TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _tok():
    return local_tokenizer([" ".join(f"w{i}" for i in range(60))])  # 62 ids: pad 0, unk 1, w0..w59 -> vocab 62


def _model(tok, **kw):
    return tiny_qwen2(vocab=len(tok), **kw)


def _cache(tok, factory=None, *, ga=None, max_length=64, include_target=False, refs=("REF",), chat=False, **kw):
    return TextCalibrationCache(
        prompts=list(PROMPTS),
        calibration_dataset_cfg=None,
        block_extension_cfg=SimpleNamespace(calibration_dataset=None, calibration_task=None, calibration_split="val"),
        harness_tasks=[],
        n_sequences_cfg=None,
        n_calib_batches=1,
        calib_batch_size=len(PROMPTS),
        calib_max_length=max_length,
        seed=0,
        include_target=include_target,
        tokenizer=tok,
        apply_chat_template=chat,
        generated_answer=parse_generated_answer_cfg(ga),
        tuned_refs=list(refs),
        generator_factory=factory,
        **kw,
    )


def _expected(model, tok, prompts, n_new, add_special_tokens=True):
    """Independent reference: one prompt at a time (no padding), neutral greedy config, cut at the pad/stop id."""
    out = []
    gc = GenerationConfig(
        do_sample=False, eos_token_id=[tok.pad_token_id], pad_token_id=tok.pad_token_id, max_new_tokens=n_new
    )
    saved, model.generation_config = model.generation_config, gc
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=add_special_tokens)
        with torch.no_grad():
            row = model.generate(**ids)[0, ids["input_ids"].shape[1] :]
        row = row.tolist()
        row = row[: row.index(tok.pad_token_id)] if tok.pad_token_id in row else row
        out.append(p + tok.decode(row, skip_special_tokens=True))
    model.generation_config = saved
    return out


class _Counting:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def __call__(self, ref):
        self.calls += 1
        return self.model


def test_parse_defaults_and_rejections():
    assert parse_generated_answer_cfg(None) is None
    assert parse_generated_answer_cfg({"enabled": False}) is None
    cfg = parse_generated_answer_cfg({"enabled": True})
    assert (cfg.max_new_tokens, cfg.batch_size) == (256, 16) == (DEFAULT_MAX_NEW_TOKENS, DEFAULT_BATCH_SIZE)
    with pytest.raises(ValueError, match="unknown"):
        parse_generated_answer_cfg({"enabled": True, "max_new_token": 4})
    with pytest.raises(ValueError, match="> 0"):
        parse_generated_answer_cfg({"max_new_tokens": 0})


def test_feature_off_is_identical_to_resolver_output():
    tok = _tok()
    got = _cache(tok).get()
    assert list(got.texts) == PROMPTS
    prov = got.provenance()
    assert "generated_answer" not in prov
    assert prov["rendered_texts_sha256"] == hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()
    assert _cache(tok).provenance() is None  # nothing resolved yet


def test_on_text_is_prompt_plus_independent_greedy_continuation(recwarn):
    tok = _tok()
    model = _model(tok)
    ga = {"enabled": True, "max_new_tokens": 6, "batch_size": 3}  # batched + left-padded vs one-by-one reference
    cache = _cache(tok, _Counting(model), ga=ga)
    got = cache.get()
    assert list(got.texts) == _expected(model, tok, PROMPTS, 6)
    assert list(got.texts) != PROMPTS
    prov = cache.provenance()["generated_answer"]
    assert prov["model"] == "REF" and prov["max_new_tokens"] == 6 and prov["n"] == len(PROMPTS)
    assert prov["cached"] is False and prov["n_truncated_at_max_length"] == 0
    assert prov["texts_sha256"] == hashlib.sha256(json.dumps(list(got.texts)).encode()).hexdigest()
    assert 0 <= prov["n_hit_max_new_tokens"] <= len(PROMPTS)
    assert not [w for w in recwarn if "truncated" in str(w.message)]


def test_generation_ignores_checkpoint_generation_config():
    tok = _tok()
    model = _model(tok)
    ref = _cache(tok, _Counting(model), ga={"max_new_tokens": 6}).get().texts
    model.generation_config.repetition_penalty = 5.0
    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = 1
    saved_cfg = model.generation_config
    again = _cache(tok, _Counting(model), ga={"max_new_tokens": 6}).get().texts
    assert list(again) == list(ref)
    assert model.generation_config is saved_cfg  # restored


def test_holdout_ids_unchanged_and_rendered_hash_changes(monkeypatch):
    tok = _tok()
    model = _model(tok)

    def fake_resolve(**kw):
        return CalibrationTexts(list(PROMPTS), source="harness:stub", eval_samples={"t": [3, 4, 5]})

    monkeypatch.setattr(ctx_mod, "resolve_calibration_texts", fake_resolve)
    off = _cache(tok)
    off.get()
    on = _cache(tok, _Counting(model), ga={"max_new_tokens": 4})
    on.get()
    p_off, p_on = off.provenance(), on.provenance()
    assert p_off["holdout_sha256"] == p_on["holdout_sha256"] and p_on["holdout_sizes"] == {"t": 3}
    assert p_off["rendered_texts_sha256"] != p_on["rendered_texts_sha256"]
    assert on.get().eval_samples == {"t": [3, 4, 5]}


def test_cache_hit_miss_and_identity_mismatch(tmp_path):
    tok = _tok()
    path = tmp_path / "sub" / "gen.json"
    ga = {"max_new_tokens": 5, "cache_path": str(path)}
    f1 = _Counting(_model(tok))
    first = _cache(tok, f1, ga=ga)
    texts = list(first.get().texts)
    assert f1.calls == 1 and path.is_file() and not list(path.parent.glob("*.tmp.*"))
    assert json.loads(path.read_text())["identity"]["model"] == "REF"
    f2 = _Counting(_model(tok, seed=5))  # a different generator would give other text: proves the cache is read
    second = _cache(tok, f2, ga=ga)
    assert list(second.get().texts) == texts and f2.calls == 0
    assert second.provenance()["generated_answer"]["cached"] is True
    # identity mismatch (different max_new_tokens / model ref / prompts) -> regenerate and overwrite
    f3 = _Counting(_model(tok, seed=5))
    _cache(tok, f3, ga={**ga, "max_new_tokens": 7}).get()
    assert f3.calls == 1 and json.loads(path.read_text())["identity"]["max_new_tokens"] == 7
    f4 = _Counting(_model(tok))
    _cache(tok, f4, ga={**ga, "max_new_tokens": 7, "model": "OTHER"}, refs=("REF", "X")).get()
    assert f4.calls == 1 and json.loads(path.read_text())["identity"]["model"] == "OTHER"


def test_model_selection_and_include_target_conflict():
    tok = _tok()
    with pytest.raises(ValueError, match="exactly one tuned body"):
        _cache(tok, _Counting(_model(tok)), ga={"enabled": True}, refs=("A", "B"))
    _cache(tok, _Counting(_model(tok)), ga={"enabled": True}, refs=("A", "A"))  # one distinct body is fine
    with pytest.raises(ValueError, match="calibration_include_target"):
        _cache(tok, _Counting(_model(tok)), ga={"enabled": True}, include_target=True)


def test_truncation_is_warned_and_recorded_not_fixed():
    tok = _tok()
    model = _model(tok)
    cache = _cache(tok, _Counting(model), ga={"max_new_tokens": 6}, max_length=4)
    with pytest.warns(RuntimeWarning, match="truncated"):
        got = cache.get()
    prov = cache.provenance()
    assert prov["generated_answer"]["n_truncated_at_max_length"] >= 1
    assert prov["calibration_max_length"] == 4 and prov["generated_answer"]["max_length"] == 4
    assert list(got.texts) == _expected(model, tok, PROMPTS, 6)  # text itself untouched


def test_chat_template_prompt_precedes_answer():
    tok = _tok()
    tok.chat_template = TEMPLATE
    model = _model(tok)
    rendered = _cache(tok, chat=True).get()
    on = _cache(tok, _Counting(model), ga={"max_new_tokens": 5}, chat=True).get()
    assert on.chat_template is not None and getattr(on.texts, "add_special_tokens", True) is False
    for before, after in zip(rendered.texts, on.texts, strict=True):
        assert after.startswith(before) and before.endswith("<|assistant|>")
    expected = _expected(model, tok, list(rendered.texts), 5, add_special_tokens=False)
    assert list(on.texts) == expected


def test_tuned_ref_with_other_rope_theta_generates_under_its_own_config(tmp_path):
    tok = _tok()
    base = _model(tok)
    other, tuned = _save_tuned(tmp_path, base, "other", rope_theta=500000.0)
    same, tuned_same = _save_tuned(tmp_path, base, "same")
    build_cfg = TextBuildConfig(model_name_or_path="unused", device="cpu")
    base_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    factory = tuned_generator_factory(base, base_sd, build_cfg, load_tuned=load_aligned_tuned_from_ref)

    gen = factory(other)
    assert gen.config.rope_parameters["rope_theta"] == 500000.0
    assert base.config.rope_parameters["rope_theta"] != 500000.0
    for k, v in tuned.state_dict().items():
        assert torch.equal(gen.state_dict()[k], v), k
    assert factory(same).config.rope_parameters["rope_theta"] == base.config.rope_parameters["rope_theta"]

    ga = {"max_new_tokens": 8}
    texts = list(_cache(tok, factory, ga=ga, refs=(other,)).get().texts)
    assert texts == _expected(gen, tok, PROMPTS, 8)  # the model with its own RoPE
    wrong = perturbed_copy(base)  # same weights as `tuned`, but under the source-base geometry
    wrong.load_state_dict(tuned.state_dict())
    ids = torch.arange(8).unsqueeze(0)
    with torch.no_grad():  # the geometry matters (greedy tokens of a tiny random model are too degenerate to show it)
        assert not torch.allclose(gen(ids).logits, wrong(ids).logits, atol=1e-6)
