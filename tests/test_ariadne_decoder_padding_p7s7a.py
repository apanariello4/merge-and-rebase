"""Phase 7 S7a: Ariadne decoder statistics never include padding rows (unconditional for decoders)."""

from __future__ import annotations

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, PAD, local_tokenizer, perturbed_copy, tiny_qwen2
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY, paired_calibration
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods._ariadne.streaming import _StreamingCrossCovariance
from merge_and_rebase.rebase.model_families import infer_family
from merge_and_rebase.rebase.registry import get_method


class _Texts(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.sample_ids = [f"t{i}" for i in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def _loader(texts, max_length, side="right"):
    tok = local_tokenizer(CALIB_TEXTS, padding_side=side)

    def collate(batch):
        return dict(tok(batch, padding="max_length", max_length=max_length, return_tensors="pt"))

    return DataLoader(_Texts(texts), batch_size=2, shuffle=False, collate_fn=collate)


TEXTS = CALIB_TEXTS * 4  # enough content rows that the Procrustes cross-covariance is full rank (d=32)


def _fit(max_length, storage, mask_enabled=True, texts=TEXTS, monkeypatch=None, alignment_map="ridge"):
    src = tiny_qwen2(layers=2, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_qwen2(layers=3, seed=2)
    fam = infer_family(src)
    if not mask_enabled:
        monkeypatch.setattr(
            type(fam), "content_mask", lambda self, b: torch.ones_like(b["input_ids"], dtype=torch.bool)
        )
    cfg = DirectResidualConfig(
        num_batches=len(texts) // 2,
        components=("mlp.c_proj",),
        activation_storage=storage,
        ridge_estimator="empirical_bayes",
        alignment_map=alignment_map,
        missing_bias="skip",
        exact_form=False,
    )
    prepared = get_method("ariadne").prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd={k: v.clone() for k, v in tgt.state_dict().items()},
        source_loader=_loader(texts, max_length),
        target_loader=_loader(texts, max_length),
        pairing=DiscreteLayerPairing.compute(2, 3),
        config=cfg,
        device="cpu",
        family_adapter=fam,
    )
    return prepared


def _assert_tv_close(a, b):
    assert a.keys() == b.keys() and a
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("storage", ["resident", "streaming"])
def test_padding_invariance(storage):
    short = _fit(12, storage)
    long = _fit(24, storage)
    _assert_tv_close(short.task_vector, long.task_vector)


@pytest.mark.parametrize("storage", ["resident", "streaming"])
def test_padding_invariance_polar_statistics(storage):
    # The tiny model's centered activations have an exactly null direction (rank d-1), so the polar factor Q is
    # not unique there; the Q-independent sufficient statistics must still be invariant to padding.
    keys = ("n_rows", "desired_norm", "effect_before_norm", "trace_centered_feature_gram")
    short = _fit(12, storage, alignment_map="polar").diagnostics
    long = _fit(24, storage, alignment_map="polar").diagnostics
    for a, b in zip(short, long, strict=True):
        assert a["n_rows"] == b["n_rows"]
        for k in keys[1:]:
            assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("storage", ["resident", "streaming"])
def test_negative_control_unmasked_differs(storage, monkeypatch):
    masked = _fit(12, storage)
    unmasked_12 = _fit(12, storage, mask_enabled=False, monkeypatch=monkeypatch)
    unmasked_24 = _fit(24, storage, mask_enabled=False, monkeypatch=monkeypatch)
    differs = any(
        not torch.allclose(unmasked_12.task_vector[k], unmasked_24.task_vector[k], rtol=1e-4, atol=1e-6)
        for k in unmasked_12.task_vector
    )
    assert differs
    assert any(
        not torch.allclose(masked.task_vector[k], unmasked_12.task_vector[k], rtol=1e-4, atol=1e-6)
        for k in masked.task_vector
    )


def test_resident_matches_streaming():
    _assert_tv_close(_fit(12, "resident").task_vector, _fit(12, "streaming").task_vector)


def test_pad_id_attended_is_content():
    tok = local_tokenizer(CALIB_TEXTS)
    fam = infer_family(tiny_qwen2())
    texts = [f"the quick {PAD} fox", "short text"] + CALIB_TEXTS[2:]
    loader = _loader(texts, 12)
    src, tgt, meta = paired_calibration(loader, _loader(texts, 12), num_batches=1, seed=None, family_adapter=fam)
    ids, mask = src[0]["input_ids"], src[0][ROW_MASK_KEY]
    pad_id = tok.pad_token_id
    assert bool(((ids == pad_id) & mask).any()), "attended pad-id token must stay content"
    assert int(mask[0].sum()) == 4 and int(mask[1].sum()) == 2
    assert torch.equal(src[0][ROW_MASK_KEY], tgt[0][ROW_MASK_KEY])
    assert meta["n_rows_content"] == int(mask.sum()) and meta["n_rows_total"] == mask.numel()


def test_mask_shape_mismatch_raises():
    fam = infer_family(tiny_qwen2())
    with pytest.raises(ValueError, match="different shapes"):
        paired_calibration(_loader(CALIB_TEXTS, 12), _loader(CALIB_TEXTS, 16), num_batches=1, family_adapter=fam)


def test_vision_batches_untouched():
    imgs = [(torch.zeros(3), 0) for _ in range(4)]

    class D(Dataset):
        sample_ids = ["a", "b", "c", "d"]

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return imgs[i]

    s, t, meta = paired_calibration(DataLoader(D(), batch_size=2), DataLoader(D(), batch_size=2), num_batches=2)
    assert isinstance(s[0], list) and "n_rows_total" not in meta


def test_stats_and_cov_row_mask_equal_preselection():
    g = torch.Generator().manual_seed(0)
    h, e = torch.randn(10, 4, generator=g), torch.randn(10, 3, generator=g)
    t_out = torch.eye(3)
    mask = torch.tensor([True, False] * 5)
    a, b = ResidualSufficientStatistics(), ResidualSufficientStatistics()
    a.update(h, e, None, t_out, row_mask=mask)
    b.update(h[mask], e[mask], None, t_out)
    assert a.n_rows == b.n_rows == 5 and torch.equal(a.s, b.s) and torch.equal(a.b, b.b)
    h_nan = h.clone()
    h_nan[1] = float("nan")  # masked-out row: must not trip the finiteness check
    a.update(h_nan, e, None, t_out, row_mask=mask)
    ResidualSufficientStatistics().update(h, e, None, t_out, row_mask=torch.zeros(10, dtype=torch.bool))
    c1, c2 = _StreamingCrossCovariance(), _StreamingCrossCovariance()
    x, y = torch.randn(10, 4, generator=g), torch.randn(10, 3, generator=g)
    c1.update(x, y, row_mask=mask)
    c2.update(x[mask], y[mask])
    assert torch.equal(c1.cross(), c2.cross())
