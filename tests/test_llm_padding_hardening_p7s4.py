"""S4a: THESEUS/BiCo padding hardening beyond the flipped characterization tests.

Padding rows never enter a statistic; masks come only from attention_mask.
"""

from __future__ import annotations

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, tiny_qwen2

import merge_and_rebase.rebase.methods  # noqa: F401  (registers methods)
from merge_and_rebase.eval.llm_rebase import _build_text_calibration_loader as build_text_calibration_loader
from merge_and_rebase.models.grad_recipes import causal_lm_recipe
from merge_and_rebase.rebase.methods import _shared
from merge_and_rebase.rebase.methods import bico as bico_mod
from merge_and_rebase.rebase.methods import theseus as theseus_mod
from merge_and_rebase.rebase.model_families import infer_family


def _pair():
    return tiny_qwen2(seed=1), tiny_qwen2(hidden=48, heads=6, inter=96, seed=2)


def _batches(texts, tok, max_length, batch_size=2):
    return list(build_text_calibration_loader(tokenizer=tok, texts=texts, batch_size=batch_size, max_length=max_length))


def _theseus(src, tgt, loader, n_batches, stats=None, seq_align="interpolate"):
    return theseus_mod.collect_activations(
        src, tgt, loader, loader, device="cpu", seq_align=seq_align, n_batches=n_batches,
        family_adapter=infer_family(src), padding_stats=stats,
    )


def _bico(src, tgt, loader, n_batches, stats=None, seq_align="interpolate"):
    return bico_mod.collect_bilinear_statistics(
        src, tgt, loader, loader, causal_lm_recipe(device="cpu"), causal_lm_recipe(device="cpu"),
        device="cpu", seq_align=seq_align, n_batches=n_batches, family_adapter=infer_family(src),
        padding_stats=stats,
    )


def test_mask_mismatch_error_names_both_shapes():
    a, b = torch.ones(2, 5, dtype=torch.long), torch.ones(2, 7, dtype=torch.long)
    with pytest.raises(ValueError) as exc:
        _shared._content_row_mask(a, b)
    assert "(2, 5)" in str(exc.value) and "(2, 7)" in str(exc.value)


def test_single_mask_is_used_and_full_content_is_none():
    m = torch.tensor([[1, 0], [1, 1]])
    assert _shared._content_row_mask(m, None).tolist() == [True, False, True, True]
    assert _shared._content_row_mask(None, m).tolist() == [True, False, True, True]
    assert _shared._content_row_mask(None, None) is None


@pytest.mark.parametrize("collector", [_theseus, _bico])
def test_padding_stats_count_total_and_content_rows(collector):
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair()
    loader = _batches(CALIB_TEXTS, tok, 12)
    stats: dict[str, int] = {}
    collector(src, tgt, loader, 3, stats)
    assert stats["n_rows_total"] == 6 * 12
    assert stats["n_rows_content"] == sum(len(t.split()) for t in CALIB_TEXTS)


@pytest.mark.parametrize("collector", [_theseus, _bico])
def test_all_pad_batch_is_skipped_and_counted(collector):
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair()
    batches = _batches(CALIB_TEXTS[:2], tok, 12)
    empty = _batches(["", ""], tok, 12)
    assert int(empty[0]["attention_mask"].sum()) == 0
    stats: dict[str, int] = {}
    reg = collector(src, tgt, batches + empty, 2, stats)
    assert stats == {"n_rows_total": 48, "n_rows_content": int(batches[0]["attention_mask"].sum())}
    assert reg["layers.0.mlp.up_proj.in"].n_samples == stats["n_rows_content"]


@pytest.mark.parametrize("collector", [_theseus, _bico])
def test_no_content_row_at_all_fails(collector):
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair()
    empty = _batches(["", ""], tok, 12)
    with pytest.raises(ValueError, match="no content rows"):
        collector(src, tgt, empty, 1, {})


@pytest.mark.parametrize("collector", [_theseus, _bico])
def test_mismatched_source_target_masks_raise_in_collection(collector):
    tok = local_tokenizer(CALIB_TEXTS, "right")
    src, tgt = _pair()
    a = _batches(CALIB_TEXTS[:2], tok, 12)
    b = _batches(CALIB_TEXTS[:2], tok, 16)
    with pytest.raises(ValueError, match="cannot be paired"):
        theseus_mod.collect_activations(
            src, tgt, a, b, device="cpu", seq_align="interpolate", n_batches=1, family_adapter=infer_family(src)
        ) if collector is _theseus else bico_mod.collect_bilinear_statistics(
            src, tgt, a, b, causal_lm_recipe(device="cpu"), causal_lm_recipe(device="cpu"),
            device="cpu", seq_align="interpolate", n_batches=1, family_adapter=infer_family(src),
        )


@pytest.mark.parametrize("mode", ["mean", "cls"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_pooled_rows_are_invariant_to_max_length(mode, side):
    tok = local_tokenizer(CALIB_TEXTS, side)
    src, tgt = _pair()
    covs = {}
    for max_length in (12, 24):
        reg = _theseus(src, tgt, _batches(CALIB_TEXTS, tok, max_length), 3, seq_align=mode)
        assert reg["layers.0.mlp.up_proj.in"].n_samples == len(CALIB_TEXTS)
        covs[max_length] = reg["layers.0.mlp.up_proj.in"].get_covariance()
    torch.testing.assert_close(covs[12], covs[24], rtol=1e-4, atol=1e-6)


def test_cls_pooling_takes_first_content_token_under_left_padding():
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    rows, _ = _shared._align_features(x, x.clone(), mode="cls", content_mask=_shared._content_row_mask(mask, mask))
    torch.testing.assert_close(rows, torch.stack([x[0, 2], x[1, 0]]))


def test_pooling_drops_sequences_without_content():
    x = torch.randn(3, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    rows, tgt = _shared._align_features(x, x.clone(), mode="mean", content_mask=_shared._content_row_mask(mask, mask))
    assert rows.shape == (2, 2) and tgt.shape == (2, 2)


def test_poolable_skips_integer_inputs_only_for_pooling_modes():
    ids = torch.zeros(2, 3, dtype=torch.long)
    assert not _shared._poolable("mean", ids, ids)
    assert not _shared._poolable("cls", ids, ids)
    assert _shared._poolable("interpolate", ids, ids)
    assert _shared._poolable("mean", torch.zeros(2, 3), torch.zeros(2, 3))


def test_vision_pooling_without_mask_is_unchanged():
    x = torch.randn(2, 5, 4)
    rows, tgt = _shared._align_features(x, x.clone(), mode="mean")
    torch.testing.assert_close(rows, x.mean(dim=1))
    rows, _ = _shared._align_features(x, x.clone(), mode="cls")
    torch.testing.assert_close(rows, x[:, 0])
