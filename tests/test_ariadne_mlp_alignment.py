"""Tests for the opt-in Ariadne ``alignment='mlp'`` (non-linear alignment ceiling; independent streaming fit, vision only).

Default (alignment absent / 'procrustes') must be bit-identical; an untrained residual-init MLP must reproduce the
Procrustes desired change; plain init is seeded; the held-out image split is deterministic and disjoint; config validation.
Fixture (tiny CPU fakes, distinct source/target widths) duplicated per the no-cross-test-import rule.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import (
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.mlp_alignment import (
    AlignmentMLP,
    heldout_image_split,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)

WS, WT = 6, 5


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _setup(source_depth=4, target_depth=2, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(WS, source_depth).eval()
    target_base = _Model(WT, target_depth).eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(8, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(8)), batch_size=2, shuffle=False)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


BASE = dict(
    activation_storage="streaming",
    num_batches=4,
    ridge_estimator="empirical_bayes",
    components=("mlp.c_proj",),
    seed=54,
)


def _sha(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _run(setup, overrides=None, prep_kwargs=None):
    source_base, source_ft, target_base, data, pairing, sd = setup
    cfg = parse_direct_residual_config({**BASE, **(overrides or {})})
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
        **(prep_kwargs or {}),
    )
    corr, rows = fit_direct_residual_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    return prepared, corr, rows


def test_default_bit_identical_and_not_serialized():
    setup = _setup()
    p0, c0, _ = _run(setup)
    p1, c1, _ = _run(setup, {"alignment": "procrustes"}, {"alignment": "procrustes"})
    assert "mlp_alignment" not in p0 and "mlp_alignment" not in p1
    assert _sha(c0) == _sha(c1)
    d = direct_residual_config_dict(parse_direct_residual_config({**BASE, "alignment": "procrustes"}))
    assert "alignment" not in d and "alignment_mlp_init" not in d
    d = direct_residual_config_dict(parse_direct_residual_config({**BASE, "alignment": "mlp"}))
    assert d["alignment"] == "mlp" and "alignment_mlp_init" not in d  # 'residual' is the default
    d = direct_residual_config_dict(
        parse_direct_residual_config({**BASE, "alignment": "mlp", "alignment_mlp_init": "plain"})
    )
    assert d["alignment"] == "mlp" and d["alignment_mlp_init"] == "plain"


def test_residual_init_zero_epochs_equals_procrustes():
    setup = _setup()
    _, c0, _ = _run(setup)
    p, c1, rows = _run(setup, {"alignment": "mlp"}, {"alignment": "mlp", "mlp_max_epochs": 0, "mlp_n_heldout": 2})
    assert set(c0) == set(c1)
    for k in c0:
        torch.testing.assert_close(c1[k], c0[k], rtol=1e-4, atol=1e-5)
    # the per-batch desired change itself
    fn_models = p["mlp_alignment"]["models"]
    for j, g in fn_models.items():
        x = torch.randn(7, WS)
        torch.testing.assert_close(g(x), x @ p["q_by_position"][j], rtol=1e-5, atol=1e-6)


def test_training_records_diagnostics_and_never_worse_than_init_on_heldout():
    setup = _setup()
    p, c, _ = _run(setup, {"alignment": "mlp"}, {"alignment": "mlp", "mlp_max_epochs": 30, "mlp_n_heldout": 2})
    assert set(c)
    for j, row in p["procrustes_diagnostics"].items():
        for key in (
            "mlp_heldout_rel_err",
            "procrustes_heldout_rel_err",
            "mlp_train_rel_err",
            "procrustes_train_rel_err",
            "mlp_epochs",
            "mlp_best_epoch",
        ):
            assert key in row, (j, key)
        assert row["mlp_train_images"] == 6 and row["mlp_heldout_images"] == 2
        assert row["mlp_heldout_rel_err"] <= row["procrustes_heldout_rel_err"] * (1 + 1e-4)
        assert 0 <= row["mlp_best_epoch"] <= row["mlp_epochs"] <= 30


def test_plain_init_is_seeded_and_deterministic():
    setup = _setup()
    kw = {"alignment": "mlp", "alignment_mlp_init": "plain", "mlp_max_epochs": 3, "mlp_n_heldout": 2}
    ov = {"alignment": "mlp", "alignment_mlp_init": "plain"}
    p_a, c_a, _ = _run(setup, ov, kw)
    p_b, c_b, _ = _run(setup, ov, kw)
    assert _sha(c_a) == _sha(c_b)
    p_c, _, _ = _run(setup, {**ov, "seed": 55}, kw)
    w_a = p_a["mlp_alignment"]["models"][0].l1.weight
    assert torch.equal(w_a, p_b["mlp_alignment"]["models"][0].l1.weight)
    assert not torch.equal(w_a, p_c["mlp_alignment"]["models"][0].l1.weight)
    assert p_a["mlp_alignment"]["models"][0].q is None


def test_residual_last_layer_is_zero_and_plain_is_not():
    q = torch.randn(WS, WT)
    r = AlignmentMLP(WS, WT, init="residual", q=q, seed=1)
    assert float(r.l2.weight.detach().abs().sum()) == 0.0 and float(r.l2.bias.detach().abs().sum()) == 0.0
    p = AlignmentMLP(WS, WT, init="plain", q=None, seed=1)
    assert float(p.l2.weight.detach().abs().sum()) > 0.0


def test_heldout_split_deterministic_and_disjoint():
    tr, ho = heldout_image_split(320, 54)
    tr2, ho2 = heldout_image_split(320, 54)
    assert torch.equal(tr, tr2) and torch.equal(ho, ho2)
    assert tr.numel() == 280 and ho.numel() == 40
    assert not (set(tr.tolist()) & set(ho.tolist()))
    assert sorted(tr.tolist() + ho.tolist()) == list(range(320))
    _, ho3 = heldout_image_split(320, 55)
    assert not torch.equal(ho, ho3)


def test_config_validation():
    with pytest.raises(ValueError, match="alignment must be"):
        parse_direct_residual_config({**BASE, "alignment": "nonlinear"})
    with pytest.raises(ValueError, match="alignment_mlp_init"):
        parse_direct_residual_config({**BASE, "alignment": "mlp", "alignment_mlp_init": "x"})
    with pytest.raises(ValueError, match="only applies"):
        parse_direct_residual_config({**BASE, "alignment_mlp_init": "plain"})
    with pytest.raises(ValueError, match="requires"):
        parse_direct_residual_config({**BASE, "alignment": "mlp", "depth_fit": "sequential"})
    with pytest.raises(ValueError, match="requires"):
        parse_direct_residual_config({**BASE, "alignment": "mlp", "activation_storage": "resident"})
    with pytest.raises(ValueError, match="requires"):
        parse_direct_residual_config({**BASE, "alignment": "mlp", "residual_target": "transported_endpoint"})
    parse_direct_residual_config({**BASE, "alignment": "mlp", "alignment_mlp_init": "plain"})


def test_vision_only():
    with pytest.raises(ValueError, match="vision-only"):
        resolve_ariadne_decoder_config({"preset": "ariadne", "alignment": "mlp"}, family_adapter=object())
    setup = _setup()
    source_base, _, target_base, data, pairing, _ = setup
    with pytest.raises(ValueError, match="vision-only"):
        prepare_direct_residual_streaming(
            source_base,
            target_base,
            data,
            data,
            pairing,
            num_batches=4,
            seed=54,
            device="cpu",
            alignment="mlp",
            alignment_map="ridge",
        )
