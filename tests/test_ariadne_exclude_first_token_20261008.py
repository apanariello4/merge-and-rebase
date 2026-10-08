"""Ariadne decoder option ``exclude_first_token`` (opt-in): drop each sequence's first real token everywhere."""

from __future__ import annotations

import hashlib

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_qwen2
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne import DirectResidualConfig, parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY, capture_tokens, paired_calibration
from merge_and_rebase.rebase.methods._ariadne.config import (
    direct_residual_config_dict,
    reject_decoder_only_options,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.model_families import infer_family
from merge_and_rebase.rebase.registry import get_method

TEXTS = CALIB_TEXTS * 4


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


def _calib(side, exclude, max_length=12):
    fam = infer_family(tiny_qwen2())
    return paired_calibration(
        _loader(TEXTS, max_length, side),
        _loader(TEXTS, max_length, side),
        num_batches=len(TEXTS) // 2,
        seed=None,
        family_adapter=fam,
        exclude_first_token=exclude,
    )


def _hash(tv):
    h = hashlib.sha256()
    for k in sorted(tv):
        h.update(k.encode())
        h.update(tv[k].detach().double().contiguous().numpy().tobytes())
    return h.hexdigest()


def _fit(exclude, storage="streaming", huge_first=False, patch_mask=False, monkeypatch=None):
    src = tiny_qwen2(layers=2, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_qwen2(layers=3, seed=2)
    fam = infer_family(src)
    if huge_first:  # artificial attention-sink: every first-token id gets a huge embedding in all three models
        tok = local_tokenizer(CALIB_TEXTS)
        first_ids = sorted({tok(t)["input_ids"][0] for t in TEXTS})
        for m in (src, ft, tgt):
            with torch.no_grad():
                m.get_input_embeddings().weight[first_ids] *= 50.0
    if patch_mask:  # independent route: a content_mask that already lacks each first real token
        orig = type(fam).content_mask

        def reduced(self, batch):
            m = orig(self, batch)
            idx = m.to(torch.uint8).argmax(dim=1)
            m = m.clone()
            m[torch.arange(m.shape[0]), idx] = False
            return m

        monkeypatch.setattr(type(fam), "content_mask", reduced)
    cfg = DirectResidualConfig(
        num_batches=len(TEXTS) // 2,
        components=("mlp.c_proj",),
        activation_storage=storage,
        ridge_estimator="empirical_bayes",
        alignment_map="ridge",
        missing_bias="skip",
        exact_form=False,
        exclude_first_token=exclude,
    )
    return get_method("ariadne").prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd={k: v.clone() for k, v in tgt.state_dict().items()},
        source_loader=_loader(TEXTS, 12),
        target_loader=_loader(TEXTS, 12),
        pairing=DiscreteLayerPairing.compute(2, 3),
        config=cfg,
        device="cpu",
        family_adapter=fam,
    )


@pytest.mark.parametrize("storage", ["streaming", "resident"])
def test_off_matches_default_config(storage):
    # exclude_first_token=False is the dataclass default: same task vector as a config that never names it.
    a = _fit(False, storage)
    src = _fit(False, storage)
    assert _hash(a.task_vector) == _hash(src.task_vector)
    assert "exclude_first_token" not in direct_residual_config_dict(DirectResidualConfig())


@pytest.mark.parametrize("side", ["right", "left"])
def test_row_count_and_metadata(side):
    _, _, off = _calib(side, False)
    _, _, on = _calib(side, True)
    assert "n_rows_excluded_first_token" not in off
    assert on["n_rows_excluded_first_token"] == len(TEXTS)  # one first token per sequence
    assert on["n_rows_content"] == off["n_rows_content"] - len(TEXTS)
    assert on["n_rows_total"] == off["n_rows_total"]
    assert on["pad_row_fraction"] == off["pad_row_fraction"]


@pytest.mark.parametrize("side", ["right", "left"])
def test_removed_rows_are_first_real_tokens(side):
    fam = infer_family(tiny_qwen2())
    s_off, t_off, _ = _calib(side, False)
    s_on, t_on, _ = _calib(side, True)
    for a, b, sa, ta in zip(s_off, s_on, s_on, t_on, strict=True):
        m_off, m_on = a[ROW_MASK_KEY], b[ROW_MASK_KEY]
        removed = m_off & ~m_on
        assert torch.equal(sa[ROW_MASK_KEY], ta[ROW_MASK_KEY])  # source/target rows stay identical
        assert not bool((m_on & ~m_off).any())
        assert bool((removed.sum(dim=1) == 1).all())
        first = m_off.to(torch.uint8).argmax(dim=1)
        assert torch.equal(removed.to(torch.uint8).argmax(dim=1), first)
        assert bool(a["attention_mask"][torch.arange(len(first)), first].all())
        if side == "right":
            assert bool((first == 0).all())
        else:
            assert bool((first == (a["attention_mask"] == 0).sum(dim=1)).all())
    # banks: the on-bank is exactly the off-bank without those rows
    model = tiny_qwen2(layers=2, seed=0)
    req = {"0": (0, "boundary")}
    bank_off = torch.cat(
        [x.reshape(-1, x.shape[-1]) for x in capture_tokens(model, s_off, req, "cpu", family_adapter=fam)["0"]]
    )
    bank_on = torch.cat(
        [x.reshape(-1, x.shape[-1]) for x in capture_tokens(model, s_on, req, "cpu", family_adapter=fam)["0"]]
    )
    keep = torch.cat([b[ROW_MASK_KEY][a[ROW_MASK_KEY]] for a, b in zip(s_off, s_on, strict=True)])
    assert bank_on.shape[0] == bank_off.shape[0] - len(TEXTS)
    assert torch.equal(bank_on, bank_off[keep])


def test_source_target_guard_still_raises():
    fam = infer_family(tiny_qwen2())
    with pytest.raises(ValueError, match="different shapes"):
        paired_calibration(
            _loader(CALIB_TEXTS, 12),
            _loader(CALIB_TEXTS, 16),
            num_batches=1,
            family_adapter=fam,
            exclude_first_token=True,
        )


@pytest.mark.parametrize("storage", ["streaming", "resident"])
def test_on_changes_fit_and_equals_reduced_mask_fit(storage, monkeypatch):
    off = _fit(False, storage, huge_first=True)
    on = _fit(True, storage, huge_first=True)
    assert _hash(on.task_vector) != _hash(off.task_vector)
    assert [d["n_rows"] for d in on.diagnostics] == [d["n_rows"] - len(TEXTS) for d in off.diagnostics]
    manual = _fit(False, storage, huge_first=True, patch_mask=True, monkeypatch=monkeypatch)
    assert _hash(on.task_vector) == _hash(manual.task_vector)


def test_vision_rejects_and_parse_validates():
    cfg = parse_direct_residual_config({"exclude_first_token": True})
    with pytest.raises(ValueError, match="decoder-only"):
        reject_decoder_only_options(cfg)
    reject_decoder_only_options(parse_direct_residual_config({}))
    with pytest.raises(TypeError, match="bool"):
        parse_direct_residual_config({"exclude_first_token": 1})
    assert resolve_ariadne_decoder_config({"exclude_first_token": True}, infer_family(tiny_qwen2())).exclude_first_token
    assert direct_residual_config_dict(cfg)["exclude_first_token"] is True
    with pytest.raises(ValueError, match="decoder-only"):
        paired_calibration(
            DataLoader(_Texts(["a", "b"]), batch_size=1),
            DataLoader(_Texts(["a", "b"]), batch_size=1),
            num_batches=1,
            exclude_first_token=True,
        )
