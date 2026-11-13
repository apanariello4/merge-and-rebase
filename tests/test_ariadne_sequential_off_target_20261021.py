"""Tests for the opt-in Ariadne ``sequential_subtract="off_target"`` (plain sequential block-boundary streaming fit).

Block j is regressed on ``E_j = D_j - (U_j - c_j D_j)`` with ``c_j = <U_j, D_j> / ||D_j||^2`` summed over all calibration rows
(``U_j`` = upstream change at boundary j with blocks < j mounted), instead of ``E_j = D_j - U_j`` ("all", the default).
Expectations are recomputed from plain forward hooks on an independently mounted copy of the target. Fixture duplicated per the
no-cross-test-import rule.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import direct_residual_config_dict, parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    _OffTargetUpstream,
    fit_direct_residual_sequential_streaming,
    prepare_direct_residual_streaming,
)

WIDTH = 5


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


def _setup(source_depth, target_depth, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(WIDTH, source_depth).eval()
    target_base = _Model(WIDTH, target_depth).eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(0.2 * torch.randn_like(block.attn.out_proj.weight))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(6, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=2, shuffle=False)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


BASE = dict(
    activation_storage="streaming",
    depth_fit="sequential",
    num_batches=3,
    ridge_estimator="empirical_bayes",
    components=("mlp.c_proj",),
)
REGIMES = [pytest.param(2, 4, id="extend"), pytest.param(4, 2, id="shrink"), pytest.param(3, 3, id="same_arch")]
PFX = "visual.transformer.resblocks.{}.mlp.c_proj"


def _sha(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _run(setup, overrides=None):
    source_base, source_ft, target_base, data, pairing, sd = setup
    cfg = parse_direct_residual_config({**BASE, **(overrides or {})})
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu"
    )
    tau, rows = fit_direct_residual_sequential_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    return tau, rows, prepared


def _boundaries(model, batches):
    """Per-block output (residual stream after the block), float64 rows ``[block][N, width]``."""
    blocks = model.visual.transformer.resblocks
    out = [[] for _ in blocks]
    handles = [
        b.register_forward_hook(lambda _m, _i, v, *, s=out[k]: s.append(v.detach().double()))
        for k, b in enumerate(blocks)
    ]
    try:
        with torch.no_grad():
            for batch in batches:
                model.encode_image(batch[0])
    finally:
        for h in handles:
            h.remove()
    return [torch.cat([x.reshape(-1, x.shape[-1]) for x in o]) for o in out]


def _cproj_inputs(model, batches, j):
    store = []
    h = model.visual.transformer.resblocks[j].mlp.c_proj.register_forward_hook(
        lambda _m, inputs, _v: store.append(inputs[0].detach().double())
    )
    try:
        with torch.no_grad():
            for batch in batches:
                model.encode_image(batch[0])
    finally:
        h.remove()
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in store])


def _explicit_ridge(h_rows, d_rows):
    n = h_rows.shape[0]
    mh, md = h_rows.mean(0), d_rows.mean(0)
    hc, dc = h_rows - mh, d_rows - md
    s = hc.T @ hc
    lam = torch.trace(s) / (n - 1)
    x = torch.linalg.solve(s + lam * torch.eye(s.shape[0], dtype=s.dtype), hc.T @ dc)
    return x.T, md - x.T @ mh


def _rel(a, b):
    return float((a.double() - b.double()).norm() / max(float(b.double().norm()), 1e-30))


def _rel(a, b):
    return float((a.double() - b.double()).norm() / max(float(b.double().norm()), 1e-30))


def _explicit_sequential(setup, prepared, mode):
    """Block-by-block recomputation: D_j, U_j (own mounted copy), c_j, E_j, ridge. ``{j: (w, b, c, ||U||, ||U_off||)}``."""
    source_base, source_ft, target_base, _data, pairing, _sd = setup
    batches = prepared["target_batches"]
    sb = _boundaries(source_base, prepared["source_batches"])
    sf = _boundaries(source_ft, prepared["source_batches"])
    pristine = _boundaries(target_base, batches)
    cur = deepcopy(target_base)
    out = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        d = (sf[i] - sb[i]) @ prepared["q_by_position"][j].double()
        u = _boundaries(cur, batches)[j] - pristine[j]
        c = float((u * d).sum() / (d * d).sum()) if float((d * d).sum()) > 0 else 0.0
        e = d - u if mode == "all" else d - (u - c * d)
        w, b = _explicit_ridge(_cproj_inputs(cur, batches, j), e)
        with torch.no_grad():
            proj = cur.visual.transformer.resblocks[j].mlp.c_proj
            proj.weight.add_(w.to(proj.weight.dtype))
            proj.bias.add_(b.to(proj.bias.dtype))
        out[j] = (w, b, c, float(u.norm()), float((u - c * d).norm()))
    return out


# (a) default is hash-identical ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_all_hash_identical_to_unset(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    default, rows_default, _ = _run(setup)
    explicit, rows_explicit, _ = _run(setup, dict(sequential_subtract="all"))
    assert _sha(default) == _sha(explicit)
    keys = {"sequential_subtract", "upstream_along_target", "upstream_norm", "upstream_off_target_norm"}
    assert all(not keys & set(r) for r in rows_default + rows_explicit)
    assert "sequential_subtract" not in direct_residual_config_dict(parse_direct_residual_config(BASE))
    off = parse_direct_residual_config({**BASE, "sequential_subtract": "off_target"})
    assert direct_residual_config_dict(off)["sequential_subtract"] == "off_target"


# (b) explicit recomputation ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_off_target_matches_explicit_recomputation(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    tau, rows, prepared = _run(setup, dict(sequential_subtract="off_target"))
    want = _explicit_sequential(setup, prepared, "off_target")
    by_pos = {r["position"]: r for r in rows}
    for j in range(target_depth):
        w, b, c, u_norm, u_off = want[j]
        assert _rel(tau[PFX.format(j) + ".weight"], w) < 1e-6, f"weight j={j}"
        assert _rel(tau[PFX.format(j) + ".bias"], b) < 1e-6, f"bias j={j}"
        r = by_pos[j]
        assert r["sequential_subtract"] == "off_target"
        assert abs(r["upstream_along_target"] - c) <= 1e-6 * max(abs(c), 1e-12) + 1e-9
        assert abs(r["upstream_norm"] - u_norm) <= 1e-6 * max(u_norm, 1e-12) + 1e-9
        assert abs(r["upstream_off_target_norm"] - u_off) <= 1e-6 * max(u_off, 1e-12) + 1e-9
        assert abs(r["upstream_norm"] - r["upstream_effect_norm"]) <= 1e-12 * max(1.0, u_norm)
    assert any(abs(by_pos[j]["upstream_along_target"]) > 1e-3 for j in range(1, target_depth))


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_all_matches_explicit_recomputation(source_depth, target_depth):
    """Guards the recomputation itself: with E_j = D_j - U_j it reproduces the unchanged default fit."""
    setup = _setup(source_depth, target_depth)
    tau, _, prepared = _run(setup)
    want = _explicit_sequential(setup, prepared, "all")
    for j in range(target_depth):
        assert _rel(tau[PFX.format(j) + ".weight"], want[j][0]) < 1e-6


# (c) block 0 ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_block0_equals_all(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    a, _, _ = _run(setup)
    o, rows, _ = _run(setup, dict(sequential_subtract="off_target"))
    k0 = [k for k in a if ".resblocks.0." in k]
    assert k0 and _sha({k: a[k] for k in k0}) == _sha({k: o[k] for k in k0})
    r0 = next(r for r in rows if r["position"] == 0)
    assert r0["upstream_along_target"] == 0.0 and r0["upstream_norm"] == 0.0 and r0["upstream_off_target_norm"] == 0.0
    assert _sha(a) != _sha(o)  # later blocks do differ


# (d) U parallel to D -> E = D -----------------------------------------------------------------------------------------


def test_upstream_parallel_to_target_gives_target():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(3, 8, 6, generator=g)
    d = torch.randn(3, 8, 5, generator=g)
    t_out = torch.eye(5)
    ref = ResidualSufficientStatistics(device="cpu")
    mix = ResidualSufficientStatistics(device="cpu")
    ot = _OffTargetUpstream()
    for k in range(3):
        ref.update(h[k], d[k], None, t_out)
        mix.update(h[k], d[k], None, t_out)
        ot.update(h[k], d[k], 0.7 * d[k], t_out, "cpu")
    ot.merge_into(mix)
    assert abs(ot.along_target - 0.7) < 1e-6 and ot.off_target_norm < 1e-6 * float(d.norm())
    assert _rel(mix.b, ref.b) < 1e-6 and _rel(mix.sum_e, ref.sum_e) < 1e-6
    assert abs(mix.sum_e2 - ref.sum_e2) < 1e-6 * ref.sum_e2
    # D = 0: c_j = 0 and E = -U (nothing to keep along D)
    zero = _OffTargetUpstream()
    zero.update(h[0], torch.zeros_like(d[0]), d[0], t_out, "cpu")
    assert zero.along_target == 0.0 and abs(zero.off_target_norm - float(d[0].double().norm())) < 1e-9


# (e) config rejections --------------------------------------------------------------------------------------------------


def test_config_accepts():
    cfg = parse_direct_residual_config({**BASE, "sequential_subtract": "off_target"})
    assert cfg.sequential_subtract == "off_target"


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequential_subtract": "bogus"},
        {"depth_fit": "independent"},
        {"activation_storage": "resident"},
        {"block_split": "backfit", "components": ("attn.out_proj", "mlp.c_proj")},
        {"sequential_points": "end_end", "components": ("attn.out_proj", "mlp.c_proj")},
        {"sequential_impl": "block_carry"},
        {"component_target": "output_total"},
        {"block_target": "increment"},
        {"pair_target_split": "interpolate"},
        {"component_fit": "gradient"},
        {"residual_target": "native_delta"},
    ],
)
def test_config_rejects(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**BASE, "sequential_subtract": "off_target", **overrides})
