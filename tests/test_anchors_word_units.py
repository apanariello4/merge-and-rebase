"""``anchors`` on word units (``text_units='word'``): decoder LLMs with DIFFERENT tokenizers.

A unit is a word of the raw text (``\\S+``); a model reads it at the last token of the word in its own tokenization
(character offsets). Covered: unit extraction under two toy fast tokenizers (boundary-crossing tokens, special tokens,
truncation), equivalence with the token rows when every word is one token, the frame identity guards, an end-to-end
cross-tokenizer / cross-family fit (pairwise and GPA), and the LLM DirectFitStage / capability wiring.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_llama, tiny_qwen2

from merge_and_rebase.data.llm_calibration import build_text_calibration_loader
from merge_and_rebase.data.text_units import NOT_A_UNIT, token_word_units, tokenize_word_units, word_end_offsets
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY
from merge_and_rebase.rebase.methods.anchors import (
    AnchorsConfig,
    AnchorsRebase,
    AnchorState,
    resolve_anchors_decoder_config,
)
from merge_and_rebase.rebase.methods.anchors.config import parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.text_units import pair_unit_batches, paired_word_calibration
from merge_and_rebase.rebase.model_families import infer_family

WORDS = sorted({w for t in CALIB_TEXTS for w in t.split()})
POOL_TEXTS = [" ".join(WORDS[(7 * i + 3 * k) % len(WORDS)] for k in range(5 + i % 6)) for i in range(32)]
TASK_TEXTS = [" ".join(WORDS[(5 * i + 2 * k + 1) % len(WORDS)] for k in range(8 + i % 3)) for i in range(20)]
# the task corpus also holds phrases that tokenizer B reads as ONE boundary-crossing token ("over the", "lazy dog")
TASK_TEXTS = TASK_TEXTS + list(CALIB_TEXTS[:4])
MAX_LENGTH = 12
FOX = CALIB_TEXTS[0]  # "the quick brown fox jumps over the lazy dog"

# Tokenizer B: pieces of at most 3 characters, whitespace as its own token, a BOS, and two phrases kept whole.
_B_PATTERN = r"over the|lazy dog|\S{1,3}"


def tokenizer_b(texts=(*POOL_TEXTS, *TASK_TEXTS, *CALIB_TEXTS)):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    pre = pre_tokenizers.Split(Regex(_B_PATTERN), behavior="isolated")
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2}
    for text in texts:
        for piece, _ in pre.pre_tokenize_str(text):
            vocab.setdefault(piece, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre
    tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<s>")


def tokenizer_a(bos: bool = False):
    """Whitespace WordLevel (every word of the fixture texts is one token), optionally with a BOS."""
    tok = local_tokenizer(CALIB_TEXTS)
    if bos:
        from tokenizers import processors
        from transformers import PreTrainedTokenizerFast

        backend = tok.backend_tokenizer
        backend.add_special_tokens(["<s>"])
        bos_id = backend.token_to_id("<s>")
        backend.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", bos_id)])
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<|pad|>", unk_token="<|unk|>",
                                      bos_token="<s>")
    return tok


def _loader(tok, texts, *, words=True, max_length=MAX_LENGTH, batch_size=4):
    return build_text_calibration_loader(
        tokenizer=tok, texts=texts, batch_size=batch_size, max_length=max_length, word_units=words
    )


def _cfg(**kw):
    base = dict(
        anchor_pool="wikitext103",
        num_anchors=16,
        anchor_batches=2,
        registration_batches=2,
        num_batches=3,
        components=("mlp.c_proj",),
        ridge_estimator="empirical_bayes",
        missing_bias="materialize",
    )
    base.update(kw)
    return AnchorsConfig(**base)


def _with_bias(model, fam):
    model = deepcopy(model)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    materialize_missing_projection_biases(model, sd, family_adapter=fam, components=("mlp.c_proj",))
    return model, sd


def _tokens(tok, ids):
    return [tok.convert_ids_to_tokens(int(i)) for i in ids]


# ---------------------------------------------------------------- 1. unit extraction


def test_word_units_map_word_ends_to_last_tokens_per_tokenizer():
    ends = word_end_offsets(FOX)
    assert ends == [3, 9, 15, 19, 25, 30, 34, 39, 43]
    a, b = tokenizer_a(), tokenizer_b()
    enc_a, units_a = tokenize_word_units(a, [FOX], max_length=16)
    enc_b, units_b = tokenize_word_units(b, [FOX], max_length=24)
    # A: one token per word, no BOS -> word k at position k
    assert units_a[0][:9] == list(range(9)) and set(units_a[0][9:]) == {NOT_A_UNIT}
    # B: the word's LAST token is the one ending at the word's end offset; BOS and whitespace tokens are no units
    tok_b = _tokens(b, enc_b["input_ids"][0])
    last_b = {k: tok_b[p] for p, k in enumerate(units_b[0]) if k != NOT_A_UNIT}
    assert last_b == {0: "the", 1: "ck", 2: "wn", 3: "fox", 4: "ps", 6: "over the", 8: "lazy dog"}
    # "over" and "lazy" end inside a token crossing the word boundary: missing for B only, present for A
    assert 5 not in last_b and 7 not in last_b and {5, 7} <= set(units_a[0])
    assert units_b[0][0] == NOT_A_UNIT and tok_b[0] == "<s>"
    # every unit token really ends at its word's end offset
    offsets = b([FOX], return_offsets_mapping=True)["offset_mapping"][0]
    for p, k in enumerate(units_b[0][: len(offsets)]):
        if k != NOT_A_UNIT:
            assert offsets[p][1] == ends[k]


def test_word_units_truncation_specials_and_repeated_ends():
    b = tokenizer_b()
    # max_length 4: <s> the ' ' qui -> "quick" is cut (its last token "ck" is gone): dropped; "the" stays
    _, units = tokenize_word_units(b, [FOX], max_length=4)
    assert units[0] == [NOT_A_UNIT, 0, NOT_A_UNIT, NOT_A_UNIT]
    _, units = tokenize_word_units(b, [FOX], max_length=5)
    assert units[0] == [NOT_A_UNIT, 0, NOT_A_UNIT, NOT_A_UNIT, 1]
    # two tokens ending at the same character (a character split into byte tokens): the later one is the unit;
    # special ids and padding are never units
    out = token_word_units("ab é", [(0, 0), (0, 2), (2, 3), (3, 4), (3, 4), (0, 0)], [9, 5, 6, 7, 8, 0],
                           [1, 1, 1, 1, 1, 0], special_ids={9})
    assert out == [NOT_A_UNIT, 0, NOT_A_UNIT, NOT_A_UNIT, 1, NOT_A_UNIT]


def test_pair_unit_batches_keeps_common_units_in_one_order():
    fam = infer_family(tiny_llama())
    a, b = tokenizer_a(), tokenizer_b()
    texts = [FOX, CALIB_TEXTS[1]]
    sa = next(iter(_loader(a, texts, max_length=16, batch_size=2)))
    sb = next(iter(_loader(b, texts, max_length=24, batch_size=2)))
    wa, wb, counts = pair_unit_batches(sa, sb, fam)
    ma, mb = wa[ROW_MASK_KEY], wb[ROW_MASK_KEY]
    words_a = [int(sa["word_unit"][0, p]) for p in torch.nonzero(ma[0]).flatten()]
    words_b = [int(sb["word_unit"][0, p]) for p in torch.nonzero(mb[0]).flatten()]
    # word 0 is the first content token of A (no BOS): excluded there, so not common; 5 and 7 are missing in B
    assert words_a == words_b == [1, 2, 3, 4, 6, 8]
    assert counts["common"] == int(ma.sum()) == int(mb.sum()) and counts["source"] > counts["common"]
    tok_b = _tokens(b, sb["input_ids"][0])
    assert [tok_b[p] for p in torch.nonzero(mb[0]).flatten()] == ["ck", "wn", "fox", "ps", "over the", "lazy dog"]
    assert "word_unit" in wa and set(fam.extract_calibration_batch(wa)) == {"input_ids", "attention_mask", "labels"}


# ---------------------------------------------------------------- 2. single-token words: word rows = token rows


def _setup(source_layers=2, target_layers=3, target_hidden=32):
    src = tiny_llama(layers=source_layers, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    fam = infer_family(src)
    tgt, sd = _with_bias(tiny_llama(layers=target_layers, hidden=target_hidden, inter=2 * target_hidden, seed=2), fam)
    return src, ft, tgt, sd, fam, DiscreteLayerPairing.compute(source_layers, target_layers)


def _prepare(cfg, src, ft, tgt, sd, fam, pairing, *, src_tok, tgt_tok, words, state=None, source_id="A",
             target_id="B", tgt_max_length=MAX_LENGTH):
    return AnchorsRebase().prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd=sd,
        source_loader=_loader(src_tok, TASK_TEXTS, words=words),
        target_loader=_loader(tgt_tok, TASK_TEXTS, words=words, max_length=tgt_max_length),
        pairing=pairing,
        config=cfg,
        device="cpu",
        source_anchor_loader=_loader(src_tok, POOL_TEXTS, words=words),
        target_anchor_loader=_loader(tgt_tok, POOL_TEXTS, words=words, max_length=tgt_max_length),
        source_model_id=source_id,
        target_model_id=target_id,
        state=state,
        family_adapter=fam,
    )


@pytest.mark.parametrize("bos", [False, True])
@pytest.mark.parametrize("frame_map", ["pairwise", "gpa"])
def test_single_token_words_give_the_token_rows_and_task_vector(bos, frame_map):
    src, ft, tgt, sd, fam, pairing = _setup()
    tok = tokenizer_a(bos=bos)
    extra = dict(frame_map="gpa", anchor_batches=4, num_anchors=48) if frame_map == "gpa" else {}
    token_state, word_state = AnchorState(_cfg(**extra)), AnchorState(_cfg(text_units="word", **extra))
    kw = dict(src_tok=tok, tgt_tok=tok)
    by_token = _prepare(_cfg(**extra), src, ft, tgt, sd, fam, pairing, words=False, state=token_state, **kw)
    by_word = _prepare(_cfg(text_units="word", **extra), src, ft, tgt, sd, fam, pairing, words=True,
                       state=word_state, **kw)
    # the same anchors: token position = word index (+1 behind the BOS)
    shift = 1 if bos else 0
    assert [[s, p - shift] for s, p in token_state.frame["anchor_rows"]] == word_state.frame["anchor_rows"]
    assert set(word_state.frame["pool"]) >= {"units"} and "tokens" not in word_state.frame["pool"]
    for m in ("A", "B"):
        for pt, pw in zip(token_state.model(m).p, word_state.model(m).p, strict=True):
            assert torch.equal(pt, pw)
    assert word_state.model("A").meta["rows"] == "words" and by_word.extra["anchors"]["rows"] == "words"
    cal_t, cal_w = by_token.extra["calibration"], by_word.extra["calibration"]
    assert cal_t["n_rows_content"] == cal_w["n_rows_content"] and cal_w["n_units_missing_in_target"] == 0
    assert set(by_token.unit_task_vector) == set(by_word.unit_task_vector)
    for key, value in by_token.unit_task_vector.items():
        torch.testing.assert_close(by_word.unit_task_vector[key], value, rtol=0, atol=0)


# ---------------------------------------------------------------- frame identity


def test_word_frame_stores_unit_identity_and_refuses_other_texts_or_missing_anchors():
    src, *_ = _setup()
    fam = infer_family(src)
    a, b = tokenizer_a(), tokenizer_b()
    state = AnchorState(_cfg(text_units="word"))
    state.register_model(src, "A", _loader(a, POOL_TEXTS), device="cpu", family_adapter=fam,
                         frame_loaders=(_loader(b, POOL_TEXTS, max_length=24),))
    pool = state.frame["pool"]
    assert pool["units"]["unit"] == "word" and len(pool["units"]["texts_sha256"]) == 64
    assert state.frame["config"]["text_units"] == "word"
    assert "text_units" not in _cfg().frame()  # the token default leaves existing frames unchanged
    with pytest.raises(ValueError, match="differ"):  # the pool identity (texts) differs
        state.register_model(src, "C", _loader(a, list(reversed(POOL_TEXTS))), device="cpu", family_adapter=fam)
    # B at max_length 6 truncates words the frame's anchors use (drawn with B at 24 tokens): refused, not redefined
    with pytest.raises(ValueError, match="anchor words have no last token"):
        state.register_model(src, "D", _loader(b, POOL_TEXTS, max_length=6), device="cpu", family_adapter=fam)
    assert not state.has_model("C") and not state.has_model("D")
    with pytest.raises(ValueError, match="text_units"):
        resolve_anchors_decoder_config({"text_units": "words"}, fam)
    with pytest.raises(ValueError, match="text_units"):
        parse_anchors_config({"text_units": "word"})  # the vision (image) pool


# ---------------------------------------------------------------- 3. cross-tokenizer, cross-family smoke


@pytest.mark.parametrize("frame_map", ["pairwise", "gpa"])
def test_cross_tokenizer_fit_end_to_end(frame_map):
    a, b = tokenizer_a(), tokenizer_b()
    src = tiny_llama(layers=2, hidden=32, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt_raw = tiny_qwen2(layers=3, hidden=48, heads=4, kv_heads=2, inter=96, vocab=len(b), seed=2)
    fam = infer_family(tgt_raw)  # qwen2: the decoder layout serves the llama source as well
    assert fam.name == "qwen2" and infer_family(src).name == "llama"
    tgt, sd = _with_bias(tgt_raw, fam)
    pairing = DiscreteLayerPairing.compute(2, 3)
    extra = dict(frame_map="gpa", gpa_rank=24) if frame_map == "gpa" else {}
    cfg = _cfg(text_units="word", anchor_batches=4, num_anchors=40, rescale_target=False, **extra)
    state = AnchorState(cfg)
    prepared = _prepare(cfg, src, ft, tgt, sd, fam, pairing, src_tok=a, tgt_tok=b, words=True, state=state,
                        tgt_max_length=24)
    tv = prepared.unit_task_vector
    assert set(tv) == {f"model.layers.{j}.mlp.down_proj.{p}" for j in range(3) for p in ("weight", "bias")}
    for j in range(3):
        w = tv[f"model.layers.{j}.mlp.down_proj.weight"]
        assert w.shape == sd[f"model.layers.{j}.mlp.down_proj.weight"].shape == (48, 96)
        assert torch.isfinite(w).all() and float(w.abs().max()) > 0
    cal = prepared.extra["calibration"]
    assert cal["rows"] == "word_units" and cal["n_units_missing_in_target"] > 0  # "over", "lazy" before B's phrases
    assert cal["n_rows_content"] == cal["n_units_source"] - cal["n_units_missing_in_target"]
    assert state.model("A").p[0].shape == (40, 32) and state.model("B").p[0].shape == (40, 48)
    assert prepared.extra["anchors"]["rows"] == "words"
    # every frame anchor is a word present in both tokenizations
    _, _, meta = paired_word_calibration(_loader(a, POOL_TEXTS), _loader(b, POOL_TEXTS, max_length=24),
                                         num_batches=1, seed=None, family_adapter=fam)
    assert meta["texts_sha256"] == state.frame["pool"]["units"]["texts_sha256"]
    if frame_map == "gpa":
        assert prepared.extra["anchors"]["gpa_universe"]["fit_models"] == ["A", "B"]


def test_cross_tokenizer_task_rows_map():
    """``map_rows='task'`` (the bridge to Ariadne) on word units: the map is fitted on the common calibration words."""
    a, b = tokenizer_a(), tokenizer_b()
    src = tiny_llama(layers=2, hidden=32, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt_raw = tiny_qwen2(layers=3, hidden=48, inter=96, vocab=len(b), seed=2)
    fam = infer_family(tgt_raw)
    tgt, sd = _with_bias(tgt_raw, fam)
    cfg = _cfg(text_units="word", map_rows="task", spectral_filter="procrustes_cross", rescale_target=False)
    tv = _prepare(cfg, src, ft, tgt, sd, fam, DiscreteLayerPairing.compute(2, 3), src_tok=a, tgt_tok=b, words=True,
                  tgt_max_length=24).unit_task_vector
    assert all(torch.isfinite(v).all() for v in tv.values()) and len(tv) == 6


def test_token_units_still_require_identical_tokens():
    a, b = tokenizer_a(), tokenizer_b()
    src, ft, tgt, sd, fam, pairing = _setup()
    with pytest.raises(ValueError):
        AnchorsRebase().prepare(
            source_base_model=src, source_ft_model=ft, target_model=tgt, target_base_sd=sd,
            source_loader=_loader(a, TASK_TEXTS, words=False), target_loader=_loader(b, TASK_TEXTS, words=False),
            pairing=pairing, config=_cfg(), device="cpu",
            source_anchor_loader=_loader(a, POOL_TEXTS, words=False),
            target_anchor_loader=_loader(b, POOL_TEXTS, words=False),
            source_model_id="A", target_model_id="B", family_adapter=fam,
        )


# ---------------------------------------------------------------- pipeline wiring


def test_capability_and_family_pair_checks():
    from merge_and_rebase.eval.llm_rebase.context import anchors_word_units, check_anchors_family_pair
    from merge_and_rebase.rebase.capabilities import check_pair

    llama, qwen = infer_family(tiny_llama()), infer_family(tiny_qwen2())
    lm, qm = llama.metadata(tiny_llama()), qwen.metadata(tiny_qwen2())
    with pytest.raises(ValueError, match="family mismatch"):
        check_pair("anchors", qm, lm)
    check_pair("anchors", qm, lm, allow_cross_family=anchors_word_units("anchors", {"text_units": "word"}))
    assert not anchors_word_units("anchors", {}) and not anchors_word_units("ariadne", {"text_units": "word"})
    words = resolve_anchors_decoder_config({"text_units": "word"}, llama)
    check_anchors_family_pair(qwen, llama, words)
    with pytest.raises(ValueError, match="text_units='word'"):
        check_anchors_family_pair(qwen, llama, resolve_anchors_decoder_config({}, llama))
    check_anchors_family_pair(llama, llama, resolve_anchors_decoder_config({}, llama))


def test_llm_direct_fit_stage_runs_anchors_across_tokenizers(monkeypatch):
    from merge_and_rebase.eval.llm_rebase import anchors_pool as pool_mod
    from merge_and_rebase.eval.llm_rebase.method_stages import DirectFitStage

    monkeypatch.setattr(pool_mod, "load_pool_texts", lambda pool: list(POOL_TEXTS))
    a, b = tokenizer_a(), tokenizer_b()
    src = tiny_llama(layers=2, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    target = tiny_qwen2(layers=3, hidden=48, inter=96, vocab=len(b), seed=2)
    fam = infer_family(target)
    cfg = resolve_anchors_decoder_config(
        {"num_anchors": 16, "anchor_batches": 2, "registration_batches": 2, "num_batches": 3, "frame_map": "gpa",
         "text_units": "word", "rescale_target": False},
        fam,
    )
    resolved = SimpleNamespace(anchors_cfg=cfg, ariadne_cfg=replace(cfg.ridge_config(), exclude_first_token=True))
    runtime = SimpleNamespace(
        cfg={
            "source_model_name_or_path": "org/llama-src",
            "target_model_name_or_path": "org/qwen-tgt",
            "anchors_pool": {"batch_size": 4, "max_length": 24, "n_sequences": len(POOL_TEXTS)},
        },
        family_adapter=fam,
        source_llm=SimpleNamespace(model=src, tokenizer=a),
        target_llm=SimpleNamespace(model=target, tokenizer=b),
        target_base_sd={k: v.detach().clone() for k, v in target.state_dict().items()},
        _calibration=lambda: SimpleNamespace(texts=TASK_TEXTS),
        calib_batch_size=4,
        calib_max_length=24,
        source_depth=2,
        target_depth=3,
        device="cpu",
        task_contexts={"it": None},
    )
    stage = DirectFitStage()
    env = SimpleNamespace(runtime=runtime, resolved=resolved)
    pre = SimpleNamespace(source_base_model=src, source_ft_model=ft, task_delta={})
    result = stage.run(env, SimpleNamespace(task="it", ctx=SimpleNamespace(index=0)), pre)
    assert set(result.transported_delta) == {
        f"model.layers.{j}.mlp.down_proj.{p}" for j in range(3) for p in ("weight", "bias")
    }
    assert all(torch.isfinite(v).all() for v in result.transported_delta.values())
    record = stage.report()["per_task"]["it"]
    assert record["anchors"]["rows"] == "words" and record["calibration"]["rows"] == "word_units"
    assert record["anchors"]["frame_pool"]["units"]["unit"] == "word"
