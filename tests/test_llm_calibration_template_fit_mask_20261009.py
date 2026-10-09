"""Opt-in ``calibration_apply_chat_template`` (decouples the calibration format from the evaluation format) and
``calibration_fit_mask="user_content"`` (ARIADNE fits only on tokens inside message contents), fully offline on CPU.

Pinned here: defaults are byte-identical (texts, token ids, batch keys, provenance keys); the resolved calibration flag
reaches the cache / tokenizer check / generated-answer conflict check; the per-token mask marks exactly the content
characters (token-boundary overlap counts) and never template tokens or padding, for both pad sides; it composes with
``exclude_first_token``; invalid combinations raise; the ARIADNE fit with the mask equals a fit whose masked rows were
removed by hand, for the independent AND both sequential implementations; model forwards never see ``fit_mask``.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_qwen2

from merge_and_rebase.data.llm_calibration import (
    CalibrationTextList,
    build_text_calibration_loader,
    locate_content_spans,
    resolve_calibration_apply_chat_template,
    resolve_calibration_fit_mask,
    resolve_calibration_texts,
)
from merge_and_rebase.eval.llm_rebase.context import TextCalibrationCache
from merge_and_rebase.rebase.capabilities import MethodFamily, method_family
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne import capture as capture_mod
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY, capture_tokens, paired_calibration
from merge_and_rebase.rebase.methods._ariadne.config import resolve_ariadne_decoder_config
from merge_and_rebase.rebase.methods._ariadne.sequential_carry import (
    _BlockRangeRunner,
    fit_direct_residual_sequential_block_carry,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_sequential_streaming,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase
from merge_and_rebase.rebase.model_families import infer_family

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
#: No markers at all: the first real token is content (to see ``exclude_first_token`` compose with the mask).
BARE_TEMPLATE = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
SYSTEM = "be brief"
PROMPTS = [*CALIB_TEXTS, "a b", "the dog", *CALIB_TEXTS[:4]]  # 12 raw texts of uneven length
MAX_LEN = 40
VOCAB_TEXTS = [*CALIB_TEXTS, "a b the dog be brief system user assistant end shot one answer"]


def _tok(template=TEMPLATE, side="right"):
    tok = local_tokenizer(VOCAB_TEXTS, padding_side=side)
    tok.chat_template = template
    return tok


def _texts(tok, *, spans, system=SYSTEM, prompts=PROMPTS):
    return resolve_calibration_texts(
        prompts=prompts,
        n_sequences=len(prompts),
        apply_chat_template=True,
        system_instruction=system,
        tokenizer=tok,
        record_content_spans=spans,
    ).texts


def _kept_text(text, offsets, mask):
    """Characters of ``text`` covered by the kept tokens, one chunk per token."""
    return [text[s:e] for (s, e), m in zip(offsets, mask, strict=True) if m]


# ----------------------------------------------------------------------------------------------- defaults


def test_defaults_are_byte_identical():
    tok = _tok()
    off = resolve_calibration_texts(prompts=PROMPTS, n_sequences=len(PROMPTS))
    assert type(off.texts) is list and list(off.texts) == PROMPTS
    plain = tok(PROMPTS, truncation=True, max_length=MAX_LEN, padding="max_length")
    batch = next(iter(build_text_calibration_loader(tokenizer=tok, texts=PROMPTS, batch_size=3, max_length=MAX_LEN)))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].tolist() == plain["input_ids"][:3]
    # chat on, mask off: same texts as before this feature, no spans, no extra batch key, unchanged provenance keys
    on = resolve_calibration_texts(
        prompts=PROMPTS, n_sequences=len(PROMPTS), apply_chat_template=True, system_instruction=SYSTEM, tokenizer=tok
    )
    expected = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for t in PROMPTS
    ]
    assert list(on.texts) == expected and on.texts.content_spans is None
    assert set(on.provenance()) == {
        "source", "n_sequences", "corpus_sha256", "holdout_sha256", "holdout_sizes", "decoupled_from_eval", "notes",
        "chat_template_applied", "chat_template_sha256", "chat_template", "rendered_texts_sha256",
    }  # fmt: skip
    batch_on = next(iter(build_text_calibration_loader(tokenizer=tok, texts=on.texts, batch_size=3, max_length=MAX_LEN)))
    assert set(batch_on) == {"input_ids", "attention_mask", "labels"}


def test_spans_recording_does_not_change_texts_or_ids():
    tok = _tok()
    a, b = _texts(tok, spans=False), _texts(tok, spans=True)
    assert list(a) == list(b) and b.content_spans is not None and a.add_special_tokens is b.add_special_tokens is False
    batch_a = next(iter(build_text_calibration_loader(tokenizer=tok, texts=a, batch_size=4, max_length=MAX_LEN)))
    batch_b = next(
        iter(build_text_calibration_loader(tokenizer=tok, texts=b, batch_size=4, max_length=MAX_LEN, fit_mask="user_content"))
    )
    assert torch.equal(batch_a["input_ids"], batch_b["input_ids"])
    assert torch.equal(batch_a["attention_mask"], batch_b["attention_mask"])
    assert set(batch_b) == {"input_ids", "attention_mask", "labels", "fit_mask"}
    assert batch_b["fit_mask"].dtype == torch.bool and batch_b["fit_mask"].shape == batch_b["input_ids"].shape


# ----------------------------------------------------------------------------------------------- feature A


def test_resolved_calibration_flag():
    assert resolve_calibration_apply_chat_template({}) is False
    assert resolve_calibration_apply_chat_template({"harness_apply_chat_template": True}) is True
    assert resolve_calibration_apply_chat_template({"calibration_apply_chat_template": None, "harness_apply_chat_template": True})
    both = {"harness_apply_chat_template": True, "calibration_apply_chat_template": False}
    assert resolve_calibration_apply_chat_template(both) is False
    assert resolve_calibration_apply_chat_template({"calibration_apply_chat_template": True}) is True
    with pytest.raises(ValueError, match="bool or null"):
        resolve_calibration_apply_chat_template({"calibration_apply_chat_template": "yes"})


def _cache(tok, *, apply, target_tok=None, generated=None, record=False, fit_mask=None):
    return TextCalibrationCache(
        prompts=PROMPTS,
        calibration_dataset_cfg=None,
        block_extension_cfg=SimpleNamespace(calibration_dataset=None, calibration_task=None, calibration_split="val"),
        harness_tasks=[],
        n_sequences_cfg=None,
        n_calib_batches=2,
        calib_batch_size=2,
        calib_max_length=MAX_LEN,
        seed=0,
        include_target=False,
        tokenizer=tok,
        apply_chat_template=apply,
        system_instruction=SYSTEM,
        target_tokenizer=target_tok,
        generated_answer=generated,
        generator_factory=(lambda ref: None) if generated is not None else None,
        tuned_refs=["x"],
        fit_mask=fit_mask,
        record_calibration_format=record,
    )


def test_decoupled_flag_gives_plain_or_rendered_texts():
    tok = _tok()
    plain_run = resolve_calibration_texts(prompts=PROMPTS, n_sequences=len(PROMPTS))
    # harness template on, calibration template explicitly off -> the texts of a template-off run
    cfg = {"harness_apply_chat_template": True, "calibration_apply_chat_template": False}
    flag = resolve_calibration_apply_chat_template(cfg)
    cache = _cache(tok, apply=flag, record=True)
    got = cache.get()
    assert list(got.texts) == list(plain_run.texts) and got.chat_template is None
    assert cache.provenance()["calibration_apply_chat_template"] is False
    assert cache.provenance()["chat_template_applied"] is False
    # harness off, calibration on -> rendered texts
    flag = resolve_calibration_apply_chat_template({"calibration_apply_chat_template": True})
    rendered = _cache(tok, apply=flag, record=True).get()
    assert list(rendered.texts) == list(_texts(tok, spans=False))
    # key not set -> no new provenance key
    default = _cache(tok, apply=False)
    default.get()
    assert "calibration_apply_chat_template" not in default.provenance()


def test_tokenizer_check_and_generated_answer_conflict_use_the_resolved_flag():
    src, tgt = _tok(), _tok(TEMPLATE + " ")
    tgt.chat_template = TEMPLATE + "{# differs #}"
    # chat off: the differing templates are never compared; chat on: refused (different templates, system turn set but
    # the probes render differently? they render identically, so only the sha differs and it is accepted) -> use no system.
    _cache(src, apply=False, target_tok=tgt).get()
    dis = _tok("{% for m in messages %}[{{ m['content'] }}]{% endfor %}")
    with pytest.raises(ValueError, match="render the same conversation differently|probe|differently"):
        _cache(src, apply=True, target_tok=dis).get()
    from merge_and_rebase.data.llm_calibration_generate import GeneratedAnswerConfig

    gen = GeneratedAnswerConfig(generation_chat_template=True)
    with pytest.raises(ValueError, match="calibration_apply_chat_template"):
        _cache(src, apply=True, generated=gen)
    _cache(src, apply=False, generated=gen)  # coupled flag resolved false: no conflict


# ----------------------------------------------------------------------------------------------- feature B


@pytest.mark.parametrize("side", ["right", "left"])
def test_mask_marks_exactly_the_user_content(side):
    tok = _tok(side=side)
    texts = _texts(tok, spans=True)
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", add_special_tokens=False,
              return_offsets_mapping=True)  # fmt: skip
    loader = build_text_calibration_loader(
        tokenizer=tok, texts=texts, batch_size=len(PROMPTS), max_length=MAX_LEN, fit_mask="user_content"
    )
    batch = next(iter(loader))
    mask = batch["fit_mask"]
    assert not bool((mask & (batch["attention_mask"] == 0)).any())  # padding is 0 on both pad sides
    assert int((batch["attention_mask"] == 0).sum()) > 0
    for i, (text, prompt) in enumerate(zip(texts, PROMPTS, strict=True)):
        assert texts.content_spans[i] == [(text.index(prompt, text.index("<|user|>")), text.index(prompt, text.index("<|user|>")) + len(prompt))]
        kept = _kept_text(text, enc["offset_mapping"][i], mask[i].tolist())
        # words are single tokens here (no token straddles a content edge), so the kept tokens ARE the content words
        assert kept == prompt.split(), (kept, prompt)
        marker_ids = [j for j, (s, e) in enumerate(enc["offset_mapping"][i]) if e > s and text[s:e] in ("<", "|", ">", "<|", "|>")]
        assert not bool(mask[i][marker_ids].any())
        assert not bool(mask[i][: 0].any())
    # system words ("be", "brief") are not kept even though they are ordinary vocabulary words
    assert all("brief" not in _kept_text(t, enc["offset_mapping"][i], mask[i].tolist()) for i, t in enumerate(texts))


def test_token_straddling_a_content_edge_is_kept():
    # Offsets are character spans: a token overlapping the content edge counts as content (documented behaviour).
    from merge_and_rebase.data.llm_calibration import _content_token_masks

    enc = {"offset_mapping": [[(0, 4), (4, 9), (9, 12), (0, 0)]], "attention_mask": [[1, 1, 1, 0]]}
    assert _content_token_masks(enc, [[(3, 10)]]) == [[1, 1, 1, 0]]
    assert _content_token_masks(enc, [[(4, 9)]]) == [[0, 1, 0, 0]]


def test_fewshot_multiturn_masks_user_and_assistant_contents(monkeypatch):
    import lm_eval.tasks as lm_tasks

    class _Task:
        eval_docs = [{"i": i} for i in range(6)]

        def doc_to_text(self, doc):
            return f"the quick {doc['i']}"

        def set_fewshot_seed(self, seed=None):
            pass

        def fewshot_context(self, doc, num_fewshot, system_instruction=None, apply_chat_template=False,
                            fewshot_as_multiturn=False, chat_template=None, gen_prefix=None):  # fmt: skip
            msgs = [{"role": "system", "content": system_instruction}] if system_instruction else []
            msgs += [{"role": "user", "content": "shot one"}, {"role": "assistant", "content": "answer one"}]
            msgs += [{"role": "user", "content": self.doc_to_text(doc)}]
            return chat_template(msgs)

    class _Manager:
        def load_task_or_group(self, names):
            return {n: _Task() for n in names}

    monkeypatch.setattr(lm_tasks, "TaskManager", _Manager)
    tok = _tok()
    res = resolve_calibration_texts(
        harness_tasks=["t"], n_sequences=4, apply_chat_template=True, system_instruction=SYSTEM, tokenizer=tok,
        num_fewshot=1, record_content_spans=True,
    )  # fmt: skip
    for text, spans in zip(res.texts, res.texts.content_spans, strict=True):
        pieces = [text[a:b] for a, b in spans]
        assert pieces[:2] == ["shot one", "answer one"] and pieces[2].startswith("the quick")
        assert len(pieces) == 3 and SYSTEM not in pieces
    # the system turn only moves the cursor; contents are matched in order
    assert locate_content_spans("<s> sys <u> a <e> <u> a <e>", [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "a"}, {"role": "user", "content": "a"},
    ]) == [(12, 13), (22, 23)]  # fmt: skip
    with pytest.raises(ValueError, match="not found verbatim"):
        locate_content_spans("<u> x <e>", [{"role": "user", "content": "y"}])
    with pytest.raises(ValueError, match="more than once"):
        locate_content_spans("<u> x <e> x", [{"role": "user", "content": "x"}])


def test_generated_answer_text_counts_as_content(monkeypatch):
    tok = _tok()
    prompt_texts = _texts(tok, spans=True)
    cache = _cache(
        tok, apply=True, fit_mask="user_content",
        generated=__import__("merge_and_rebase.data.llm_calibration_generate", fromlist=["x"]).GeneratedAnswerConfig(),
    )  # fmt: skip
    import merge_and_rebase.eval.llm_rebase.context as ctx

    monkeypatch.setattr(
        ctx, "build_generated_texts",
        lambda prompts, **kw: ([p + "answer one" for p in prompts], {"answers": ["answer one"] * len(prompts), "n": len(prompts),
                                                                     "mean_response_tokens": 2.0, "n_hit_max_new_tokens": 0,
                                                                     "cached": False}),
    )  # fmt: skip
    got = cache.get()
    for text, prompt, spans in zip(got.texts, prompt_texts, got.texts.content_spans, strict=True):
        assert spans[:-1] == prompt_texts.content_spans[list(prompt_texts).index(prompt)]
        assert text[spans[-1][0] : spans[-1][1]] == "answer one"
    prov = cache.provenance()
    assert prov["calibration_fit_mask"] == "user_content" and prov["n_tokens_kept"] > 0
    assert 0 < prov["kept_share"] < 1 and len(prov["mask_sha256"]) == 64


def test_provenance_records_counts_and_hash():
    tok = _tok()
    cache = _cache(tok, apply=True, fit_mask="user_content")
    cache.get()
    prov = cache.provenance()
    loader = build_text_calibration_loader(
        tokenizer=tok, texts=cache.get().texts, batch_size=len(PROMPTS), max_length=MAX_LEN, fit_mask="user_content"
    )
    batch = next(iter(loader))
    assert prov["calibration_fit_mask"] == "user_content"
    assert prov["n_tokens_kept"] == int(batch["fit_mask"].sum())
    assert prov["n_tokens_total"] == int(batch["attention_mask"].sum())
    assert prov["kept_share"] == prov["n_tokens_kept"] / prov["n_tokens_total"]
    assert prov["mask_sha256"] == hashlib.sha256(bytes(batch["fit_mask"].to(torch.uint8).flatten().tolist())).hexdigest()
    assert "calibration_fit_mask" not in _plain_provenance(tok)


def _plain_provenance(tok):
    cache = _cache(tok, apply=True)
    cache.get()
    return cache.provenance()


def test_invalid_combinations_raise():
    ok = {"calibration_fit_mask": "user_content"}
    assert resolve_calibration_fit_mask({}, chat_rendered=False, method_is_ariadne=False) is None
    assert resolve_calibration_fit_mask(ok, chat_rendered=True, method_is_ariadne=True) == "user_content"
    with pytest.raises(ValueError, match="chat-rendered"):
        resolve_calibration_fit_mask(ok, chat_rendered=False, method_is_ariadne=True)
    with pytest.raises(ValueError, match="null or one of"):
        resolve_calibration_fit_mask({"calibration_fit_mask": "assistant"}, chat_rendered=True, method_is_ariadne=True)
    for method in ("theseus", "theseus_gqa", "bico"):
        assert method_family(method) is not MethodFamily.DIRECT_FIT
        with pytest.raises(ValueError, match="ARIADNE"):
            resolve_calibration_fit_mask(
                ok, chat_rendered=True, method_is_ariadne=method_family(method) is MethodFamily.DIRECT_FIT
            )
    assert method_family("ariadne") is MethodFamily.DIRECT_FIT and method_family("direct_residual") is MethodFamily.DIRECT_FIT
    # no spans (plain text) / slow tokenizer
    with pytest.raises(ValueError, match="content spans"):
        build_text_calibration_loader(tokenizer=_tok(), texts=PROMPTS, fit_mask="user_content")
    slow = SimpleNamespace(is_fast=False)
    with pytest.raises(ValueError, match="fast tokenizer"):
        build_text_calibration_loader(tokenizer=slow, texts=_texts(_tok(), spans=True), fit_mask="user_content")
    with pytest.raises(ValueError, match="fast tokenizers"):
        _cache(_tok(), apply=True, target_tok=slow, fit_mask="user_content").get()
    with pytest.raises(ValueError, match="chat-rendered"):
        _cache(_tok(), apply=False, fit_mask="user_content")


# ----------------------------------------------------------------------------------------------- ARIADNE


def _loaders(tok, *, mask=True, batch_size=2):
    texts = _texts(tok, spans=True)
    kw = {"fit_mask": "user_content"} if mask else {}
    return [
        build_text_calibration_loader(tokenizer=tok, texts=texts, batch_size=batch_size, max_length=MAX_LEN, **kw)
        for _ in range(2)
    ]


@pytest.mark.parametrize("side", ["right", "left"])
def test_mask_combines_with_padding_and_exclude_first_token(side):
    tok = _tok(BARE_TEMPLATE, side)
    fam = infer_family(tiny_qwen2())
    n = len(PROMPTS)

    def calib(mask, exclude):
        s, t = _loaders(tok, mask=mask)
        return paired_calibration(s, t, num_batches=n // 2, seed=None, family_adapter=fam, exclude_first_token=exclude)

    _, _, off = calib(False, False)
    src, tgt, only_mask = calib(True, False)
    _, _, both = calib(True, True)
    assert "n_rows_excluded_fit_mask" not in off and "n_rows_excluded_first_token" not in only_mask
    assert only_mask["n_rows_excluded_fit_mask"] > 0  # the system turn "be brief" is dropped from every text
    assert only_mask["n_rows_content"] == off["n_rows_content"] - only_mask["n_rows_excluded_fit_mask"]
    assert only_mask["pad_row_fraction"] == off["pad_row_fraction"]
    # bare template: the first real token is the system word, already masked; first-token exclusion then removes
    # nothing more -> same rows; with a user-only conversation it would remove the first content token instead.
    assert both["n_rows_content"] <= only_mask["n_rows_content"]
    for a, b in zip(src, tgt, strict=True):
        assert torch.equal(a[ROW_MASK_KEY], b[ROW_MASK_KEY])
        assert not bool((a[ROW_MASK_KEY] & ~a["attention_mask"].bool()).any())
        assert "fit_mask" not in a  # the wrapped batch is the model-input whitelist + the row mask only


def test_exclude_first_token_acts_on_the_first_real_token_even_when_it_is_content():
    tok = _tok(BARE_TEMPLATE, "left")
    texts = _texts(tok, spans=True, system=None)  # no system turn: the first real token is content
    fam = infer_family(tiny_qwen2())
    loader = lambda: build_text_calibration_loader(  # noqa: E731
        tokenizer=tok, texts=texts, batch_size=2, max_length=MAX_LEN, fit_mask="user_content"
    )
    _, _, meta = paired_calibration(
        loader(), loader(), num_batches=len(PROMPTS) // 2, seed=None, family_adapter=fam, exclude_first_token=True
    )
    assert meta["n_rows_excluded_first_token"] == len(PROMPTS)
    assert meta["n_rows_excluded_fit_mask"] == 0  # every real token is content here


def _arrays(tv):
    return {k: v.detach().double() for k, v in tv.items()}


def _assert_close(a, b, tol=1e-12):
    assert set(a) == set(b)
    for k in a:
        assert float((a[k] - b[k]).abs().max()) <= tol, k


def _ariadne_fit(mask, monkeypatch=None, *, patched_reference=False, dtype=torch.float64):
    src = tiny_qwen2(layers=2, hidden=16, inter=48, seed=0).to(dtype)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_qwen2(layers=3, seed=2).to(dtype)
    fam = infer_family(tgt)
    if patched_reference:
        # Reference: attach ignores fit_mask (as before the feature); each row is removed by hand in content_mask.
        orig = type(fam).content_mask
        monkeypatch.setattr(type(fam), "content_mask", lambda self, batch: orig(self, batch) & batch["fit_mask"].bool())
        monkeypatch.setattr(capture_mod, "_batch_fit_mask", lambda batch: None)
    tok = _tok()
    s, t = _loaders(tok, mask=mask or patched_reference)
    cfg = resolve_ariadne_decoder_config(
        {"num_batches": len(PROMPTS) // 2, "components": ["mlp.c_proj"], "missing_bias": "skip", "exact_form": False,
         "ridge_estimator": "empirical_bayes", "alignment_map": "ridge"}, fam)  # fmt: skip
    return AriadneRebase().prepare(
        source_base_model=src, source_ft_model=ft, target_model=tgt,
        target_base_sd={k: v.clone() for k, v in tgt.state_dict().items()},
        source_loader=s, target_loader=t, pairing=DiscreteLayerPairing.compute(2, 3), config=cfg, device="cpu",
        family_adapter=fam,
    )  # fmt: skip


def test_independent_fit_equals_hand_removed_rows(monkeypatch):
    on = _ariadne_fit(True)
    off = _ariadne_fit(False)
    # reference needs its own monkeypatch scope (patched content_mask + attach ignoring fit_mask)
    with monkeypatch.context() as m:
        ref = _ariadne_fit(False, m, patched_reference=True)
    assert on.extra["calibration"]["n_rows_excluded_fit_mask"] > 0
    assert "n_rows_excluded_fit_mask" not in off.extra["calibration"]
    assert ref.extra["calibration"]["n_rows_content"] == on.extra["calibration"]["n_rows_content"]
    _assert_close(_arrays(on.task_vector), _arrays(ref.task_vector))
    assert max(float((a - b).abs().max()) for a, b in zip(_arrays(on.task_vector).values(), _arrays(off.task_vector).values(), strict=True)) > 1e-9
    assert [d["n_rows"] for d in on.diagnostics] != [d["n_rows"] for d in off.diagnostics]


def test_capture_bank_is_the_unmasked_bank_without_the_masked_rows():
    tok = _tok()
    fam = infer_family(tiny_qwen2())
    model = tiny_qwen2(layers=2, seed=0)
    n = len(PROMPTS)
    batches = {}
    for mask in (False, True):
        s, t = _loaders(tok, mask=mask)
        batches[mask] = paired_calibration(s, t, num_batches=n // 2, seed=None, family_adapter=fam)[0]
    req = {"0": (0, "boundary")}
    bank = {
        m: torch.cat([x.reshape(-1, x.shape[-1]) for x in capture_tokens(model, b, req, "cpu", family_adapter=fam)["0"]])
        for m, b in batches.items()
    }
    keep = torch.cat([b[ROW_MASK_KEY][a[ROW_MASK_KEY]] for a, b in zip(batches[False], batches[True], strict=True)])
    assert bank[True].shape[0] < bank[False].shape[0]
    assert torch.equal(bank[True], bank[False][keep])


def _sequential(monkeypatch=None, *, mask=True, patched_reference=False, side="right"):
    src = tiny_qwen2(layers=2, hidden=16, inter=48, seed=0).to(torch.float64)
    tgt = tiny_qwen2(layers=3, seed=2).to(torch.float64)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    fam = infer_family(tgt)
    if patched_reference:
        orig = type(fam).content_mask
        monkeypatch.setattr(type(fam), "content_mask", lambda self, batch: orig(self, batch) & batch["fit_mask"].bool())
        monkeypatch.setattr(capture_mod, "_batch_fit_mask", lambda batch: None)
    tok = _tok(side=side)
    s, t = _loaders(tok, mask=mask or patched_reference)
    params = {"depth_fit": "sequential", "missing_bias": "skip", "exact_form": False, "num_batches": len(PROMPTS) // 2}
    cfg_ref = resolve_ariadne_decoder_config(params, fam)
    cfg_new = resolve_ariadne_decoder_config({**params, "sequential_impl": "block_carry"}, fam)
    pairing = DiscreteLayerPairing.compute(2, 3)
    prepared = prepare_direct_residual_streaming(
        src, tgt, s, t, pairing, num_batches=cfg_ref.num_batches, seed=cfg_ref.seed, device="cpu", family_adapter=fam
    )  # fmt: skip
    base = {k: v.clone() for k, v in tgt.state_dict().items()}
    args = (tgt, base, src, ft, prepared, pairing)
    ref = fit_direct_residual_sequential_streaming(*args, config=cfg_ref, device="cpu", family_adapter=fam)
    new = fit_direct_residual_sequential_block_carry(
        *args, config=cfg_new, device="cpu", family_adapter=fam, runner=_BlockRangeRunner(fam, "cpu")
    )  # fmt: skip
    return ref, new, prepared


@pytest.mark.parametrize("side", ["right", "left"])
def test_sequential_reference_and_block_carry_honour_the_mask(monkeypatch, side):
    ref, new, prepared = _sequential(mask=True, side=side)
    assert prepared["calibration"]["n_rows_excluded_fit_mask"] > 0
    # the two implementations still agree bit for bit under the mask
    assert set(ref[0]) == set(new[0])
    for k in ref[0]:
        assert torch.equal(ref[0][k], new[0][k]), k
    # ... and equal the fit on hand-removed rows
    with monkeypatch.context() as m:
        man_ref, man_new, _ = _sequential(m, mask=False, patched_reference=True, side=side)
    _assert_close(_arrays(ref[0]), _arrays(man_ref[0]))
    _assert_close(_arrays(new[0]), _arrays(man_new[0]))
    off_ref, _, _ = _sequential(mask=False, side=side)
    assert max(float((ref[0][k].double() - off_ref[0][k].double()).abs().max()) for k in ref[0]) > 1e-9


def test_model_forward_never_receives_fit_mask():
    seen: set[str] = set()
    src = tiny_qwen2(layers=2, hidden=16, inter=48, seed=0).to(torch.float64)
    tgt = tiny_qwen2(layers=3, seed=2).to(torch.float64)
    fam = infer_family(tgt)
    handles = []
    for model in (src, tgt):
        for module in model.modules():
            handles.append(
                module.register_forward_pre_hook(lambda _m, args, kwargs: seen.update(kwargs) or None, with_kwargs=True)
            )
    s, t = _loaders(_tok())
    cfg = resolve_ariadne_decoder_config({"num_batches": 2}, fam)
    prepare_direct_residual_streaming(
        src, tgt, s, t, DiscreteLayerPairing.compute(2, 3), num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
        family_adapter=fam,
    )  # fmt: skip
    for h in handles:
        h.remove()
    assert "input_ids" in seen and "attention_mask" in seen
    assert not {"fit_mask", ROW_MASK_KEY} & seen
    # the raw loader batch carries it, the model-input whitelist does not
    raw = next(iter(s))
    assert "fit_mask" in raw and "fit_mask" not in fam.extract_calibration_batch(raw)


def test_calibration_text_list_validates_spans():
    with pytest.raises(ValueError, match="content span lists"):
        CalibrationTextList(["a", "b"], content_spans=[[(0, 1)]])
