"""Extensions of calibration_generated_answer: generation_chat_template, separator, shuffle, raw-answer cache."""

from __future__ import annotations

import hashlib
import json

import pytest
from _llm_fixtures import tiny_qwen2
from test_llm_calibration_generated_answer_20261007 import PROMPTS, TEMPLATE, _cache, _Counting, _tok

from merge_and_rebase.data.llm_calibration_generate import derange, parse_generated_answer_cfg
from merge_and_rebase.eval.llm_rebase.harness import chat_end_of_turn_token_id


def _chat_tok():
    tok = _tok()
    tok.chat_template = TEMPLATE
    return tok


def _model(tok):
    return tiny_qwen2(vocab=len(tok))


class _Spy:
    """Factory whose model records the strings (decoded input ids) handed to generate()."""

    def __init__(self, model):
        self.model, self.calls, self.inputs = model, 0, []
        orig = model.generate

        def wrapped(*a, **kw):
            self.inputs.append(kw["input_ids"].clone())
            return orig(*a, **kw)

        model.generate = wrapped

    def __call__(self, ref):
        self.calls += 1
        return self.model


def test_defaults_unchanged(tmp_path):
    tok = _tok()
    path = tmp_path / "g.json"
    cache = _cache(tok, _Counting(_model(tok)), ga={"max_new_tokens": 5, "cache_path": str(path)})
    cache.get()
    ident = json.loads(path.read_text())["identity"]
    assert set(ident) == {
        "model", "dtype", "max_new_tokens", "apply_chat_template", "system_instruction", "chat_template_sha256",
        "prompts_sha1",
    }  # fmt: skip
    prov = cache.provenance()["generated_answer"]
    assert set(prov) == {
        "model", "max_new_tokens", "n", "mean_response_tokens", "n_hit_max_new_tokens", "cached", "texts_sha256",
        "n_truncated_at_max_length", "max_length",
    }  # fmt: skip


def test_parse_new_keys():
    cfg = parse_generated_answer_cfg(
        {
            "generation_chat_template": True,
            "generation_system_instruction": "S",
            "separator": "\n",
            "shuffle_answers_seed": 3,
        }
    )
    assert (cfg.generation_chat_template, cfg.generation_system_instruction, cfg.separator, cfg.shuffle_answers_seed) == (
        True, "S", "\n", 3,
    )  # fmt: skip
    with pytest.raises(ValueError, match="needs generation_chat_template"):
        parse_generated_answer_cfg({"generation_system_instruction": "S"})
    with pytest.raises(ValueError, match="unknown"):
        parse_generated_answer_cfg({"shuffle_seed": 1})


def test_chat_generation_prompt_but_plain_text(tmp_path):
    tok = _chat_tok()
    model = _model(tok)
    spy = _Spy(model)
    ga = {"max_new_tokens": 5, "batch_size": 1, "generation_chat_template": True}
    got = _cache(tok, spy, ga=ga).get()
    rendered = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in PROMPTS
    ]
    for ids, r in zip(spy.inputs, rendered, strict=True):
        assert ids[0].tolist() == tok(r, add_special_tokens=False)["input_ids"]
    # independent reference: generate on the rendered prompt, stop set = task_decoding_config(chat)
    from merge_and_rebase.eval.llm_rebase.harness import task_decoding_config

    gc = task_decoding_config(tok, apply_chat_template=True)
    gc.max_new_tokens = 5
    stop = set(gc.eos_token_id)
    model.generation_config = gc
    import torch

    for text, prompt, r in zip(got.texts, PROMPTS, rendered, strict=True):
        enc = tok(r, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            row = model.generate(**enc)[0, enc["input_ids"].shape[1] :].tolist()
        cut = next((i for i, t in enumerate(row) if t in stop), len(row))
        assert text == prompt + tok.decode(row[:cut], skip_special_tokens=True)
        assert "<|" not in text
    assert chat_end_of_turn_token_id(tok) in stop


def test_chat_generation_stops_at_end_of_turn():
    tok = _chat_tok()
    eot = chat_end_of_turn_token_id(tok)
    assert eot is not None

    class Fixed:  # emits "w5 <end> w6 w7": the answer must be cut before the end-of-turn token
        training = False
        generation_config = None

        def to(self, d):
            return self

        def eval(self):
            return self

        def train(self, m):
            pass

        def generate(self, input_ids, attention_mask=None, **kw):
            import torch

            tail = torch.tensor([[tok.convert_tokens_to_ids("w5"), eot, tok.convert_tokens_to_ids("w6")]])
            return torch.cat([input_ids, tail.expand(input_ids.shape[0], -1)], dim=1)

    ga = {"max_new_tokens": 3, "generation_chat_template": True}
    got = _cache(tok, lambda ref: Fixed(), ga=ga).get()
    assert list(got.texts) == [p + "w5" for p in PROMPTS]


def test_chat_generation_system_instruction_rendered():
    tok = _chat_tok()
    spy = _Spy(_model(tok))
    ga = {"max_new_tokens": 2, "generation_chat_template": True, "generation_system_instruction": "w9"}
    _cache(tok, spy, ga=ga).get()
    r = tok.apply_chat_template(
        [{"role": "system", "content": "w9"}, {"role": "user", "content": PROMPTS[0]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert (
        spy.inputs[0][0].tolist()[-len(tok(r, add_special_tokens=False)["input_ids"]) :]
        == tok(r, add_special_tokens=False)["input_ids"]
    )


def test_chat_generation_conflicts_with_harness_chat_template():
    tok = _chat_tok()
    with pytest.raises(ValueError, match="harness_apply_chat_template"):
        _cache(tok, _Counting(_model(tok)), ga={"generation_chat_template": True}, chat=True)


def test_separator_and_cache_serves_all_cells(tmp_path):
    tok = _tok()
    path = tmp_path / "g.json"
    base = {"max_new_tokens": 5, "cache_path": str(path)}
    f1 = _Counting(_model(tok))
    plain = list(_cache(tok, f1, ga=base).get().texts)
    answers = json.loads(path.read_text())["answers"]
    assert plain == [p + a for p, a in zip(PROMPTS, answers, strict=True)]
    f2 = _Counting(_model(tok))
    c2 = _cache(tok, f2, ga={**base, "separator": "\n\n"})
    sep = list(c2.get().texts)
    assert f2.calls == 0 and sep == [p + "\n\n" + a for p, a in zip(PROMPTS, answers, strict=True)]
    f3 = _Counting(_model(tok))
    shuf = list(_cache(tok, f3, ga={**base, "shuffle_answers_seed": 1}).get().texts)
    assert f3.calls == 0 and shuf != plain
    prov = c2.provenance()["generated_answer"]
    assert prov["separator"] == repr("\n\n") and prov["shuffle_answers_seed"] is None
    assert prov["generation_chat_template"] is False and prov["generation_system_instruction"] is None
    assert prov["answers_sha256"] == hashlib.sha256(json.dumps(answers).encode()).hexdigest()
    assert prov["texts_sha256"] == hashlib.sha256(json.dumps(sep).encode()).hexdigest()
    assert prov["cached"] is True


def test_derangement_properties():
    answers = [f"a{i}" for i in range(7)]
    for seed in range(20):
        out = derange(answers, seed)
        assert all(o != a for o, a in zip(out, answers, strict=True))
        assert sorted(out) == sorted(answers)
        assert out == derange(answers, seed)
    assert len({tuple(derange(answers, s)) for s in range(20)}) > 1
    assert derange(["x", "y"], 0) == ["y", "x"]
    with pytest.raises(ValueError, match="at least 2"):
        derange(["x"], 0)


def test_shuffle_text_pairs_other_answer_and_n_lt_2_errors(tmp_path):
    tok = _tok()
    model = _model(tok)
    ga = {"max_new_tokens": 5, "shuffle_answers_seed": 4, "separator": " "}
    cache = _cache(tok, _Counting(model), ga=ga)
    got = list(cache.get().texts)
    plain = list(_cache(tok, _Counting(model), ga={"max_new_tokens": 5}).get().texts)
    answers = [t[len(p) :] for t, p in zip(plain, PROMPTS, strict=True)]
    assert got == [p + " " + a for p, a in zip(PROMPTS, derange(answers, 4), strict=True)]
    assert cache.provenance()["generated_answer"]["shuffle_answers_seed"] == 4
    one = _cache(tok, _Counting(model), ga={"shuffle_answers_seed": 0})
    one._prompts = PROMPTS[:1]
    one._n_sequences_cfg = None
    one._calib_batch_size = 1
    with pytest.raises(ValueError, match="at least 2"):
        one.get()


def test_old_format_cache_loads_for_defaults_and_regenerates_for_new_options(tmp_path):
    tok = _tok()
    path = tmp_path / "g.json"
    ga = {"max_new_tokens": 5, "cache_path": str(path)}
    texts = list(_cache(tok, _Counting(_model(tok)), ga=ga).get().texts)
    blob = json.loads(path.read_text())
    del blob["answers"]
    path.write_text(json.dumps(blob))
    f = _Counting(
        _model(
            tok,
        )
        if False
        else _model(tok)
    )
    assert list(_cache(tok, f, ga=ga).get().texts) == texts and f.calls == 0
    f2 = _Counting(_model(tok))
    got = list(_cache(tok, f2, ga={**ga, "separator": "|"}).get().texts)
    assert f2.calls == 1 and "answers" in json.loads(path.read_text())
    assert got == [t[: len(p)] + "|" + t[len(p) :] for t, p in zip(texts, PROMPTS, strict=True)]


def test_chat_identity_keys_only_in_new_mode(tmp_path):
    tok = _chat_tok()
    path = tmp_path / "g.json"
    ga = {
        "max_new_tokens": 3,
        "cache_path": str(path),
        "generation_chat_template": True,
        "generation_system_instruction": "w9",
    }
    _cache(tok, _Counting(_model(tok)), ga=ga).get()
    ident = json.loads(path.read_text())["identity"]
    assert ident["generation_chat_template"] is True and ident["generation_system_instruction"] == "w9"
    assert ident["chat_template_sha256"] and ident["apply_chat_template"] is False
    f = _Counting(_model(tok))  # different system instruction -> different identity -> regenerate
    _cache(tok, f, ga={**ga, "generation_system_instruction": "w8"}).get()
    assert f.calls == 1
    f = _Counting(_model(tok))
    _cache(tok, f, ga={**ga, "generation_system_instruction": "w8", "separator": "x", "shuffle_answers_seed": 2}).get()
    assert f.calls == 0
