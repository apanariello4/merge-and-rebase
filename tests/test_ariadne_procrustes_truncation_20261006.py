"""Opt-in rank truncation of the Ariadne Procrustes map (streaming path, polar + activation only).

`procrustes_rank` / `procrustes_energy` keep the top-k singular pairs of the centered cross-covariance,
Q^k = U[:, :k] Vh[:k]. Defaults (None) must leave the fit bit-identical. Fixture duplicated from
`tests/test_direct_residual_streaming.py` (no cross-test imports).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import _truncated_procrustes_from_cross
from merge_and_rebase.rebase.methods._ariadne.config import (
    DirectResidualConfig,
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)


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


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _setup(source_depth, target_depth, width=5, seed=11, same_object=False):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = source_base if (same_object and source_depth == target_depth) else _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _sha(d):
    h = hashlib.sha256()
    for key in sorted(d):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _run(config, depths=(3, 4), **prep_kwargs):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(*depths)
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=config.num_batches,
        seed=config.seed,
        device="cpu",
        **prep_kwargs,
    )
    corr, _diag = fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config,
        device="cpu",
    )
    return prepared, corr


def _cfg(**kw):
    return DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming", **kw)


def test_default_task_vector_unchanged_with_explicit_none():
    prepared_a, corr_a = _run(_cfg())
    prepared_b, corr_b = _run(_cfg(), procrustes_rank=None, procrustes_energy=None)
    assert _sha(corr_a) == _sha(corr_b)
    for j in prepared_a["q_by_position"]:
        assert torch.equal(prepared_a["q_by_position"][j], prepared_b["q_by_position"][j])
        assert "procrustes_rank_used" not in prepared_a["procrustes_diagnostics"][j]


@pytest.mark.parametrize("kw", [{"procrustes_rank": 5}, {"procrustes_rank": 99}, {"procrustes_energy": 1.0}])
def test_full_equivalents_are_bit_identical(kw):
    prepared_d, corr_d = _run(_cfg())
    prepared_t, corr_t = _run(_cfg(**kw), **kw)
    for j in prepared_d["q_by_position"]:
        assert torch.equal(prepared_d["q_by_position"][j], prepared_t["q_by_position"][j])
        assert prepared_t["procrustes_diagnostics"][j]["procrustes_rank_used"] == 5
    assert _sha(corr_d) == _sha(corr_t)


def test_rank_k_is_partial_isometry_matching_independent_svd():
    torch.manual_seed(3)
    cross = torch.randn(7, 5, dtype=torch.float64)
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    for k in (1, 3, 4):
        q, used, mass, kept_energy = _truncated_procrustes_from_cross(cross, rank=k)
        assert used == k
        assert torch.equal(q, u[:, :k] @ vh[:k])
        assert mass == pytest.approx(float(s[:k].sum() / s.sum()))
        assert kept_energy == pytest.approx(float((s[:k] ** 2).sum() / (s**2).sum()))
        proj = q @ q.T
        assert torch.allclose(proj @ proj, proj, atol=1e-12)
        assert float(torch.trace(proj)) == pytest.approx(k)
        assert int(torch.linalg.matrix_rank(q)) == k


def test_rank_k_in_pipeline_is_rank_k_partial_isometry():
    prepared, _ = _run(_cfg(procrustes_rank=2), procrustes_rank=2)
    for j, q in prepared["q_by_position"].items():
        proj = (q @ q.T).double()
        assert float(torch.trace(proj)) == pytest.approx(2, abs=1e-4)
        assert torch.allclose(proj @ proj, proj, atol=1e-4)
        row = prepared["procrustes_diagnostics"][j]
        assert row["procrustes_rank_used"] == 2
        assert 0 < row["procrustes_kept_singular_mass"] <= 1
        assert row["procrustes_kept_singular_mass"] <= row["procrustes_kept_singular_energy"] + 1e-12


@pytest.mark.parametrize("frac", [0.1, 0.5, 0.8, 0.95, 0.999])
def test_energy_picks_smallest_k_on_plain_singular_values(frac):
    torch.manual_seed(5)
    for shape in ((6, 6), (4, 9)):
        cross = torch.randn(*shape, dtype=torch.float64)
        s = torch.linalg.svdvals(cross)
        want = next(k for k in range(1, len(s) + 1) if float(s[:k].sum() / s.sum()) >= frac)
        q, used, mass, _ = _truncated_procrustes_from_cross(cross, energy=frac)
        assert used == want
        assert mass >= frac
        assert want == 1 or float(s[: want - 1].sum() / s.sum()) < frac
        assert int(torch.linalg.matrix_rank(q)) == want


def test_energy_recorded_in_diagnostics():
    prepared, _ = _run(_cfg(procrustes_energy=0.6), procrustes_energy=0.6)
    for row in prepared["procrustes_diagnostics"].values():
        assert 1 <= row["procrustes_rank_used"] <= 5
        assert row["procrustes_kept_singular_mass"] >= 0.6 - 1e-12


@pytest.mark.parametrize(
    "overrides",
    [
        {"procrustes_rank": 2, "procrustes_energy": 0.5},
        {"procrustes_rank": 0},
        {"procrustes_rank": -1},
        {"procrustes_rank": 1.5},
        {"procrustes_rank": True},
        {"procrustes_energy": 0.0},
        {"procrustes_energy": 1.5},
        {"procrustes_energy": -0.1},
        {"procrustes_energy": float("nan")},
        {"procrustes_rank": 2, "alignment_map": "ridge"},
        {"procrustes_energy": 0.5, "alignment_map": "random_isometry"},
        {"procrustes_rank": 2, "procrustes_source": "gradient"},
        {"procrustes_rank": 2, "activation_storage": "resident"},
    ],
)
def test_validation_errors(overrides):
    params = {"activation_storage": "streaming", **overrides}
    with pytest.raises(ValueError):
        parse_direct_residual_config(params)


def test_prepare_rejects_incompatible_options():
    source_base, _ft, target_base, data, pairing, _sd = _setup(2, 2)
    with pytest.raises(ValueError):
        prepare_direct_residual_streaming(
            source_base,
            target_base,
            data,
            data,
            pairing,
            num_batches=2,
            seed=0,
            device="cpu",
            procrustes_rank=1,
            procrustes_energy=0.5,
        )
    with pytest.raises(ValueError):
        prepare_direct_residual_streaming(
            source_base,
            target_base,
            data,
            data,
            pairing,
            num_batches=2,
            seed=0,
            device="cpu",
            procrustes_rank=1,
            alignment_map="ridge",
        )


def test_serialization_only_when_set():
    default = direct_residual_config_dict(DirectResidualConfig())
    assert "procrustes_rank" not in default and "procrustes_energy" not in default
    r = direct_residual_config_dict(_cfg(procrustes_rank=3))
    assert r["procrustes_rank"] == 3 and "procrustes_energy" not in r
    e = direct_residual_config_dict(_cfg(procrustes_energy=0.9))
    assert e["procrustes_energy"] == 0.9 and "procrustes_rank" not in e


class _DecoderAdapter:
    """Minimal stand-in: resolve_ariadne_decoder_config only needs a non-None adapter."""


def test_decoder_config_accepts_fields():
    cfg = resolve_ariadne_decoder_config({"procrustes_rank": 4}, _DecoderAdapter())
    assert cfg.procrustes_rank == 4 and cfg.activation_storage == "streaming"
    cfg = resolve_ariadne_decoder_config({"procrustes_energy": 0.9}, _DecoderAdapter())
    assert cfg.procrustes_energy == 0.9
