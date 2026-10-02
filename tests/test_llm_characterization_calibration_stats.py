"""Characterization of THESEUS / BiCo calibration statistics on tiny Qwen decoders.

Real calibration code throughout: ``_build_text_calibration_loader`` ->
``collect_activations`` (THESEUS) / ``collect_bilinear_statistics`` (BiCo) with the
family adapter llm_rebase uses. Row masks must come from attention_mask only
(Qwen pad id == a content token id), and padding must never enter statistics.

Strict xfails document confirmed bugs (see reasons); they must flip to passes
when the corresponding Phase 7 fix lands.
"""

from __future__ import annotations

import pytest
import torch
from _llm_fixtures import (
    CALIB_TEXTS,
    PAD,
    local_tokenizer,
    real_qwen_tokenizer_or_skip,
    tiny_qwen2,
    tiny_qwen3,
)

import merge_and_rebase.rebase.methods  # noqa: F401  (registers methods)
from merge_and_rebase.eval.llm_rebase import _build_text_calibration_loader
from merge_and_rebase.models.grad_recipes import causal_lm_recipe
from merge_and_rebase.rebase.methods import bico as bico_mod
from merge_and_rebase.rebase.methods import theseus as theseus_mod
from merge_and_rebase.rebase.model_families import infer_family

N_BATCHES = 3  # 3 batches x 2 sequences = all 6 CALIB_TEXTS
BATCH = 2
# `embed_tokens.in` is fed input_ids (B, L): one row per sequence with L features, not one row per token.
# Characterized separately below; the embedding is never transported.
_EMBED_KEY = "embed_tokens.in"


def _is_token_row_key(key: str) -> bool:
    """Keys whose hooked tensor is (B, L, D): one row per token. Excludes the embedding input and
    Qwen3's per-head q_norm/k_norm hooks (4-D (B, L, heads, head_dim), characterized separately)."""
    return key != _EMBED_KEY and ".q_norm." not in key and ".k_norm." not in key


def _pair(family: str):
    if family == "qwen2":
        return tiny_qwen2(seed=1), tiny_qwen2(hidden=48, heads=6, inter=96, seed=2)
    return tiny_qwen3(seed=1), tiny_qwen3(hidden=48, heads=6, inter=96, seed=2)


def _loader(tok, texts, max_length, batch_size=BATCH):
    return _build_text_calibration_loader(tokenizer=tok, texts=texts, batch_size=batch_size, max_length=max_length)


def _content_tokens(loader) -> int:
    return sum(int(b["attention_mask"].sum()) for b in loader)


def _theseus(src, tgt, loader, *, seq_align="interpolate", n_batches=N_BATCHES, **kw):
    return theseus_mod.collect_activations(
        src,
        tgt,
        loader,
        loader,
        device="cpu",
        seq_align=seq_align,
        n_batches=n_batches,
        family_adapter=infer_family(src),
        **kw,
    )


def _bico(src, tgt, loader, *, seq_align="interpolate", n_batches=N_BATCHES):
    return bico_mod.collect_bilinear_statistics(
        src,
        tgt,
        loader,
        loader,
        causal_lm_recipe(device="cpu"),
        causal_lm_recipe(device="cpu"),
        device="cpu",
        seq_align=seq_align,
        n_batches=n_batches,
        family_adapter=infer_family(src),
    )


COLLECTORS = {"theseus": _theseus, "bico": _bico}


def _assert_rows_are_content(registry, expected_rows):
    assert registry
    for key, store in registry.items():
        if not _is_token_row_key(key):
            continue
        assert store.n_samples == expected_rows, key


def _assert_cov_equal(reg_a, reg_b):
    assert reg_a.keys() == reg_b.keys()
    for key in reg_a:
        if not _is_token_row_key(key):
            continue
        torch.testing.assert_close(
            reg_a[key].get_covariance(),
            reg_b[key].get_covariance(),
            rtol=1e-4,
            atol=1e-6,
            msg=lambda m, k=key: f"{k}: {m}",
        )


# --------------------------------------------------------------------------- right padding


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("method", ["theseus", "bico"])
def test_rows_equal_content_tokens_and_covariance_invariant_to_max_length(method, family, side):
    # Left padding is invariant too: RoPE is relative and pads are masked out of attention.
    tok = local_tokenizer(CALIB_TEXTS, side)
    src, tgt = _pair(family)
    n_content = sum(len(t.split()) for t in CALIB_TEXTS)
    regs = {}
    for max_length in (12, 24):
        loader = _loader(tok, CALIB_TEXTS, max_length)
        assert _content_tokens(loader) == n_content
        regs[max_length] = COLLECTORS[method](src, tgt, loader)
        _assert_rows_are_content(regs[max_length], n_content)
    _assert_cov_equal(regs[12], regs[24])


@pytest.mark.parametrize("method", ["theseus", "bico"])
def test_embed_tokens_in_statistic_is_per_sequence_not_per_token(method):
    """Characterization: the embedding 'input' is input_ids (B, L) -> one row per sequence, L features.

    Pads are not removed (row count != mask size) and the statistic is not invariant to max_length.
    The embedding is excluded from transport, so this is inert today.
    """
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair("qwen2")
    for max_length in (12, 24):
        reg = COLLECTORS[method](src, tgt, _loader(tok, CALIB_TEXTS, max_length))
        store = reg[_EMBED_KEY]
        assert store.n_samples == len(CALIB_TEXTS)
        assert tuple(store.at_b.shape) == (max_length, max_length)


@pytest.mark.parametrize("method", ["theseus", "bico"])
def test_qwen3_qk_norm_hook_statistics_are_not_per_token(method):
    """Characterization: q_norm/k_norm hook 4-D (B, L, heads, head_dim) tensors, which the generic
    aligner folds into rows=B*heads*(...), features=L, with pad positions included (unmasked).
    These keys are not transportable (see hybrid-path tests), so this is inert today."""
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair("qwen3")
    for max_length in (12, 24):
        reg = COLLECTORS[method](src, tgt, _loader(tok, CALIB_TEXTS, max_length))
        qk = {k: s for k, s in reg.items() if ".q_norm." in k or ".k_norm." in k}
        assert qk
        n_content = sum(len(t.split()) for t in CALIB_TEXTS)
        for key, store in qk.items():
            assert tuple(store.at_b.shape) == (max_length, max_length), key
            assert store.n_samples != n_content, key


# --------------------------------------------------------------------------- pad id as content


@pytest.mark.parametrize("method", ["theseus", "bico"])
def test_local_pad_id_as_content_is_kept(method):
    texts = ["alpha " + PAD + " beta gamma", "delta epsilon", "zeta " + PAD, "eta theta iota"]
    tok = local_tokenizer(texts, "right")
    src, tgt = _pair("qwen2")
    loader = _loader(tok, texts, 10)
    pad_as_content = sum(int(((b["input_ids"] == tok.pad_token_id) & (b["attention_mask"] == 1)).sum()) for b in loader)
    assert pad_as_content == 2
    n_content = _content_tokens(loader)
    assert n_content == 4 + 2 + 2 + 3
    reg = COLLECTORS[method](src, tgt, loader, n_batches=2)
    _assert_rows_are_content(reg, n_content)


@pytest.mark.parametrize("method", ["theseus", "bico"])
def test_real_qwen_endoftext_as_content_is_kept(method):
    tok = real_qwen_tokenizer_or_skip("right")
    texts = ["first doc<|endoftext|>second doc", "short", "x<|endoftext|>", "tail words here"]
    vocab = max(len(tok), 151936)
    src = tiny_qwen2(seed=1, vocab=vocab)
    tgt = tiny_qwen2(hidden=48, heads=6, inter=96, seed=2, vocab=vocab)
    loader = _loader(tok, texts, 16)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    assert tok.pad_token_id == eot
    in_content = sum(int(((b["input_ids"] == eot) & (b["attention_mask"] == 1)).sum()) for b in loader)
    assert in_content == 2
    reg = COLLECTORS[method](src, tgt, loader, n_batches=2)
    _assert_rows_are_content(reg, _content_tokens(loader))


# --------------------------------------------------------------------------- known bugs (strict xfail)


def _manual_batches(texts, tok, max_length):
    """Plain list of collated batches: no `.dataset`, so collection takes the ordered zip path."""
    return list(_loader(tok, texts, max_length))


def test_content_row_mask_normal_case_masks_pads():
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
    out = theseus_mod._content_row_mask(mask, mask)
    assert out.tolist() == [True, True, False, False, True, False, False, False]
    # full-content batch: nothing to drop -> None (keep everything) is correct there.
    assert theseus_mod._content_row_mask(torch.ones(2, 3, dtype=torch.long), None) is None


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _content_row_mask returns None for an all-pad batch (`not mask.any()`), so every pad row is kept; "
    "decision: padding rows never enter statistics.",
)
def test_all_pad_batch_has_no_content_rows():
    mask = torch.zeros(2, 4, dtype=torch.long)
    out = theseus_mod._content_row_mask(mask, mask)
    assert out is not None and not bool(out.any())


@pytest.mark.xfail(
    strict=True,
    reason="BUG: an all-pad batch contributes pad rows to THESEUS statistics (n_samples counts pads).",
)
def test_all_pad_batch_adds_no_rows_end_to_end():
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair("qwen2")
    batches = _manual_batches(CALIB_TEXTS[:2], tok, 12)
    empty = _manual_batches(["", ""], tok, 12)
    assert int(empty[0]["attention_mask"].sum()) == 0
    reg = _theseus(src, tgt, batches + empty, n_batches=2)
    _assert_rows_are_content(reg, _content_tokens(batches))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _content_row_mask silently returns None when source/target masks differ in numel (tokenizer "
    "mismatch), keeping pad rows; decision: error for all methods, never a pad-keeping fallback.",
)
def test_mask_numel_mismatch_raises():
    a = torch.ones(2, 5, dtype=torch.long)
    b = torch.ones(2, 7, dtype=torch.long)
    a[:, 3:] = 0
    with pytest.raises(ValueError):
        theseus_mod._content_row_mask(a, b)


def _pooled_content_rows(x: torch.Tensor, mask: torch.Tensor, *, mode: str):
    """The exact per-batch steps collect_activations applies to one (source, target) hook pair."""
    rows, _ = theseus_mod._align_features(x, x.clone(), mode=mode)
    row_mask = theseus_mod._content_row_mask(mask, mask)
    return theseus_mod._drop_padding_rows(rows, rows, row_mask)[0]


def _content_means(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[i, mask[i].bool()].mean(dim=0) for i in range(x.shape[0])])


def test_seq_align_mean_without_padding_is_content_mean():
    """Control: with no padding, mean pooling equals the mean over content tokens."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3, 5, 4, generator=gen)
    mask = torch.ones(3, 5, dtype=torch.long)
    torch.testing.assert_close(_pooled_content_rows(x, mask, mode="mean"), _content_means(x, mask))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: seq_align='mean' pools over all positions incl. padding before _drop_padding_rows "
    "(pooled row count != mask numel, so nothing is dropped); rows differ from the content mean.",
)
def test_seq_align_mean_pools_only_content_tokens():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3, 5, 4, generator=gen)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    torch.testing.assert_close(_pooled_content_rows(x, mask, mode="mean"), _content_means(x, mask))


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="BUG: collect_activations(seq_align='mean') crashes on any decoder: the embed_tokens hook input is the "
    "Long input_ids tensor and mean() rejects integer dtype, so mean pooling is unusable for LLMs.",
)
def test_seq_align_mean_runs_on_decoder():
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair("qwen2")
    reg = _theseus(src, tgt, _loader(tok, CALIB_TEXTS, 12), seq_align="mean")
    assert reg["layers.0.mlp.up_proj.in"].n_samples == len(CALIB_TEXTS)
