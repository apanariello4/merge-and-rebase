"""Tests for the opt-in Ariadne ``block_target="increment"`` (independent block-boundary streaming fit).

Target block j (paired with source block i(j), run size g(j), p(j) = last target position before j's run) is regressed on
``inc_j = [(f_i - b_i) - (f_p - b_p)] Q_j / g(j)`` (first run: ``(f_i - b_i) Q_j / g(j)``) instead of ``D_j = (f_i - b_i) Q_j``.
The desired rows are read through a ``stats_factory`` that records the error rows (effect is zero against the pristine
target), and every expectation is recomputed from plain forward hooks. Fixture duplicated per the no-cross-test-import rule.
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
    increment_references,
    parse_direct_residual_config,
)
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
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
    num_batches=3,
    ridge_estimator="empirical_bayes",
    components=("mlp.c_proj",),
)
REGIMES = [pytest.param(2, 4, id="extend"), pytest.param(4, 2, id="shrink"), pytest.param(3, 3, id="same_arch")]


class _Recorder(ResidualSufficientStatistics):
    """Unchanged statistics that also keep the (h, e) rows each batch contributed."""

    def __init__(self, store, key, **kwargs):
        super().__init__(**kwargs)
        self._store, self._key = store, key

    def update(self, h, e, t_in, t_out, row_mask=None):
        self._store.setdefault(self._key, {"h": [], "e": []})
        self._store[self._key]["h"].append(h.double().clone())
        self._store[self._key]["e"].append(e.double().clone())
        super().update(h, e, t_in, t_out, row_mask)


def _sha(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _run(setup, overrides=None, q_identity=False, record=False):
    source_base, source_ft, target_base, data, pairing, sd = setup
    cfg = parse_direct_residual_config({**BASE, **(overrides or {})})
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu"
    )
    if q_identity:
        prepared = {**prepared, "q_by_position": {j: torch.eye(WIDTH) for j in prepared["q_by_position"]}}
    store: dict = {}
    factory = (lambda pos, comp, dev: _Recorder(store, pos, device=dev)) if record else None
    tau, rows = fit_direct_residual_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu", stats_factory=factory
    )
    return tau, rows, prepared, store


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


def _explicit_desired(setup, prepared, q_by_position=None, block_target="increment"):
    """``{j: inc_j (or D_j)}`` as float64 rows, from plain hooks (token counts equal, so no alignment)."""
    source_base, source_ft, target_base, _data, pairing, _sd = setup
    sb = _boundaries(source_base, prepared["source_batches"])
    sf = _boundaries(source_ft, prepared["source_batches"])
    pi = pairing.pairing
    refs = increment_references(pi)
    groups = {}
    for j in range(len(pi)):
        groups[j] = sum(1 for k in range(len(pi)) if pi[k] == pi[j])
    out = {}
    for j in range(len(pi)):
        q = (q_by_position or prepared["q_by_position"])[j].double()
        delta = sf[pi[j]] - sb[pi[j]]
        if block_target == "total":
            out[j] = delta @ q
            continue
        if refs[j] is not None:
            delta = delta - (sf[pi[refs[j]]] - sb[pi[refs[j]]])
        out[j] = delta @ q / groups[j]
    return out


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


# (a) explicit recomputation ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_increment_matches_explicit_recomputation(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    tau, rows, prepared, store = _run(setup, dict(block_target="increment"), record=True)
    want = _explicit_desired(setup, prepared)
    target_base, data = setup[2], prepared["target_batches"]
    for j in range(target_depth):
        got = torch.cat(store[j]["e"])
        assert _rel(got, want[j]) < 1e-6, f"desired rows j={j}"
        w, b = _explicit_ridge(_cproj_inputs(target_base, data, j), want[j])
        prefix = f"visual.transformer.resblocks.{j}.mlp.c_proj"
        assert _rel(tau[f"{prefix}.weight"], w) < 1e-6, f"weight j={j}"
        assert _rel(tau[f"{prefix}.bias"], b) < 1e-6, f"bias j={j}"
    by_pos = {r["position"]: r for r in rows}
    total = _explicit_desired(setup, prepared, block_target="total")
    for j in range(target_depth):
        assert by_pos[j]["block_target"] == "increment"
        assert abs(by_pos[j]["total_target_norm"] - float(total[j].norm())) < 1e-5 * float(total[j].norm())


# (b) telescoping, one-to-one pairing, identical Q --------------------------------------------------------------------


def test_telescoping_one_to_one():
    setup = _setup(3, 3)
    _, _, _, inc = _run(setup, dict(block_target="increment"), q_identity=True, record=True)
    _, _, _, tot = _run(setup, dict(block_target="total"), q_identity=True, record=True)
    running = torch.zeros_like(torch.cat(inc[0]["e"]))
    for j in range(3):
        running = running + torch.cat(inc[j]["e"])
        assert _rel(running, torch.cat(tot[j]["e"])) < 1e-6, f"sum_k<=j inc_k vs D_j at j={j}"


# (c) many-to-one pairing -----------------------------------------------------------------------------------------------


def test_many_to_one_equal_split_and_first_run():
    setup = _setup(2, 4)  # pairing (0, 0, 1, 1)
    assert setup[4].pairing == (0, 0, 1, 1) or tuple(setup[4].pairing) == (0, 0, 1, 1)
    _, _, _, inc = _run(setup, dict(block_target="increment"), q_identity=True, record=True)
    _, _, _, tot = _run(setup, dict(block_target="total"), q_identity=True, record=True)
    e = {j: torch.cat(inc[j]["e"]) for j in range(4)}
    d = {j: torch.cat(tot[j]["e"]) for j in range(4)}
    assert _rel(e[0], e[1]) < 1e-12  # equal split inside the run
    assert _rel(e[0], d[0] / 2) < 1e-6  # first run: D_0 / g
    assert _rel(e[0] + e[1], d[0]) < 1e-6  # the run reproduces the step
    assert _rel(e[2], e[3]) < 1e-12
    assert _rel(e[2] + e[3], d[2] - d[0]) < 1e-6  # second run: difference of the two references
    assert _rel(e[0] + e[1] + e[2] + e[3], d[2]) < 1e-6


# (d) total is the default, hash-identical --------------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_total_hash_identical_to_default(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    default, rows_default, _, _ = _run(setup)
    explicit, rows_explicit, _, _ = _run(setup, dict(block_target="total"))
    assert _sha(default) == _sha(explicit)
    assert all("block_target" not in r for r in rows_default + rows_explicit)
    assert "block_target" not in direct_residual_config_dict(parse_direct_residual_config(BASE))
    assert (
        direct_residual_config_dict(parse_direct_residual_config({**BASE, "block_target": "increment"}))["block_target"]
        == "increment"
    )


def test_increment_differs_from_total():
    setup = _setup(3, 3)
    assert _sha(_run(setup)[0]) != _sha(_run(setup, dict(block_target="increment"))[0])


# (e) config rejections ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"block_target": "bogus"},
        {"activation_storage": "resident"},
        {"depth_fit": "sequential"},
        {"block_split": "joint"},
        {"component_target": "output_total"},
        {"residual_target": "transported_endpoint"},
        {"pair_target_split": "per_block_share"},
        {"fit_blocks": (0,)},
        {"component_fit": "gradient_exact"},
        {"residual_target": "native_oracle", "native_target_ckpt_template": "x"},
    ],
)
def test_config_rejections(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**BASE, "block_target": "increment", **overrides})


def test_config_accepts_increment_with_several_components():
    cfg = parse_direct_residual_config(
        {**BASE, "block_target": "increment", "components": ("attn.out_proj", "mlp.c_proj")}
    )
    assert cfg.block_target == "increment"


# (f) non-monotone pairing -------------------------------------------------------------------------------------------------------


def test_non_monotone_pairing_raises():
    setup = _setup(3, 3)
    source_base, source_ft, target_base, data, pairing, sd = setup
    bad = DiscreteLayerPairing.compute(3, 3)
    object.__setattr__(bad, "pairing", (0, 2, 1)) if hasattr(bad, "__dataclass_fields__") else setattr(
        bad, "pairing", (0, 2, 1)
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        _run((source_base, source_ft, target_base, data, bad, sd), dict(block_target="increment"))
    with pytest.raises(ValueError, match="non-decreasing"):
        increment_references((0, 2, 1))
    assert increment_references((0, 0, 1, 1)) == {0: None, 1: None, 2: 1, 3: 1}
