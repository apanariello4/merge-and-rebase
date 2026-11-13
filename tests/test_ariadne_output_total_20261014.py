"""Tests for the opt-in ``component_target="output_total"`` Ariadne fit (vision only, streaming).

Per layer c in {attn.out_proj, mlp.c_proj} of target block j (paired with source block i), the regression target is
``D_j^c = (A_c(FT) - A_c(base)) Q_j^c`` in the projection-output space (``A_c`` = the projection's own output, before
LayerScale), ``Q_j^c`` the centered rectangular Procrustes map fitted on base component outputs. Independent mode fits
every layer against the pristine target; sequential mode fits ``O_0, D_0, O_1, D_1, ...`` with target ``D - U`` (``U`` the
change of ``A_c`` caused by the already fitted layers). Every expectation below is recomputed explicitly in float64 from
hooks installed in the test, independently of the capture/accumulation code under test.

Fixture duplicated from `tests/test_ariadne_sequential_depth_fit_20261005.py` (no cross-test imports), with a block whose
MLP reads the attention-updated stream (so attn.out_proj changes reach mlp.c_proj).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from _llm_fixtures import tiny_qwen2
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import centered_rectangular_procrustes
from merge_and_rebase.rebase.methods._ariadne.config import (
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.output_total import (
    fit_output_total_independent_streaming,
    fit_output_total_sequential_streaming,
    prepare_output_total_streaming,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import prepare_direct_residual_streaming
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase
from merge_and_rebase.rebase.model_families import infer_family

WIDTH, DEPTH = 5, 3
COMPS = ("attn.out_proj", "mlp.c_proj")


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mix = torch.nn.Linear(width, width)
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(torch.nn.functional.gelu(self.mix(x)))


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
        x = x + self.attn(x)
        return x + self.ls_2(self.mlp(x))


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


def _tuned(model, seed, scale=0.2, blocks=None, comps=COMPS):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for j, block in enumerate(tuned.visual.transformer.resblocks):
            if blocks is not None and j not in blocks:
                continue
            for c in comps:
                mod = block.attn.out_proj if c == "attn.out_proj" else block.mlp.c_proj
                mod.weight.add_(scale * torch.randn_like(mod.weight))
                mod.bias.add_(0.1 * scale * torch.randn_like(mod.bias))
    return tuned


def _setup(source_depth=DEPTH, target_depth=DEPTH, seed=11, ft_blocks=None, ft_comps=COMPS):
    torch.manual_seed(seed)
    source_base = _Model(WIDTH, source_depth).eval()
    target_base = _Model(WIDTH, target_depth).eval()
    source_ft = _tuned(source_base, seed + 1, blocks=ft_blocks, comps=ft_comps)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


def _cfg(depth_fit="independent", components=COMPS, **extra):
    return parse_direct_residual_config(
        dict(
            activation_storage="streaming",
            component_target="output_total",
            num_batches=3,
            ridge_estimator="empirical_bayes",
            components=components,
            depth_fit=depth_fit,
            **extra,
        )
    )


def _prepare(source_base, target_base, data, pairing, cfg):
    return prepare_output_total_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
        components=cfg.components,
    )


def _fit(source_base, source_ft, target_base, data, pairing, sd, cfg):
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    fit = (
        fit_output_total_sequential_streaming
        if cfg.depth_fit == "sequential"
        else fit_output_total_independent_streaming
    )
    tau, rows = fit(target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu")
    return tau, rows, prepared


def _sha(d):
    h = hashlib.sha256()
    for key in sorted(d):
        h.update(key.encode())
        h.update(d[key].numpy().tobytes())
    return h.hexdigest()


# --- explicit (independent) recomputation helpers ---------------------------------------------------------------


def _layer_module(model, j, c):
    block = model.visual.transformer.resblocks[j]
    return block.attn.out_proj if c == "attn.out_proj" else block.mlp.c_proj


def _hooks(model, batches, positions, comps):
    """``{(j, c): {"h": [per-batch input], "a": [per-batch output]}}`` from plain forward hooks."""
    out = {(j, c): {"h": [], "a": []} for j in positions for c in comps}
    handles = []
    for (j, c), store in out.items():

        def hook(_m, inputs, value, *, s=store):
            s["h"].append(inputs[0].detach().float())
            s["a"].append(value.detach().float())

        handles.append(_layer_module(model, j, c).register_forward_hook(hook))
    try:
        with torch.no_grad():
            for batch in batches:
                model.encode_image(batch[0])
    finally:
        for h in handles:
            h.remove()
    return out


def _rows(batches):
    return torch.cat([b.reshape(-1, b.shape[-1]) for b in batches]).double()


def _explicit_q(src_a, tgt_a):
    q, _, _ = centered_rectangular_procrustes(_rows(src_a), _rows(tgt_a))
    return q


def _explicit_ridge(h_rows, d_rows):
    """Empirical-Bayes ridge with unpenalised intercept, L = I, float64: ``(W correction, bias correction)``."""
    n = h_rows.shape[0]
    mh, md = h_rows.mean(0), d_rows.mean(0)
    hc, dc = h_rows - mh, d_rows - md
    s = hc.T @ hc
    lam = torch.trace(s) / (n - 1)
    x = torch.linalg.solve(s + lam * torch.eye(s.shape[0], dtype=s.dtype), hc.T @ dc)
    return x.T, md - x.T @ mh


def _key(j, c, kind="weight"):
    return f"visual.transformer.resblocks.{j}.{c}.{kind}"


def _explicit_targets(source_base, source_ft, target_base, prepared, positions, comps):
    """``{(j, c): D_j^c}`` (float64 rows) and the pristine-target captures, from independent hooks (equal depths)."""
    sb = _hooks(source_base, prepared["source_batches"], positions, comps)
    sf = _hooks(source_ft, prepared["source_batches"], positions, comps)
    tb = _hooks(target_base, prepared["target_batches"], positions, comps)
    q = {p: _explicit_q(sb[p]["a"], tb[p]["a"]) for p in sb}
    d = {p: (_rows(sf[p]["a"]) - _rows(sb[p]["a"])) @ q[p] for p in sb}
    return d, q, tb


def _mounted(base_model, sd, tau, keep):
    model = deepcopy(base_model)
    model.load_state_dict({k: v + tau[k] if k in tau and keep(k) else v for k, v in sd.items()}, strict=True)
    return model.eval()


def _block_of(key):
    return int(key.split(".")[3])


def _assert_close(got, want, tol, what):
    got, want = got.double(), want.double()
    err = float((got - want).norm() / max(float(want.norm()), 1e-30))
    assert err < tol, f"{what}: relative error {err:.3e}"
    return err


# --- (a) independent: explicit recomputation -------------------------------------------------------------------


def test_independent_matches_explicit_recomputation():
    source_base, source_ft, target_base, data, pairing, sd = _setup()
    cfg = _cfg()
    tau, rows, prepared = _fit(source_base, source_ft, target_base, data, pairing, sd, cfg)
    positions = range(DEPTH)
    d, _q, tb = _explicit_targets(source_base, source_ft, target_base, prepared, positions, COMPS)
    assert len(rows) == DEPTH * 2
    worst = 0.0
    for j in positions:
        for c in COMPS:
            w, b = _explicit_ridge(_rows(tb[(j, c)]["h"]), d[(j, c)])
            worst = max(worst, _assert_close(tau[_key(j, c)], w, 1e-5, f"W {j} {c}"))
            worst = max(worst, _assert_close(tau[_key(j, c, "bias")], b, 1e-5, f"b {j} {c}"))
    print(f"independent vs explicit float64: worst relative error {worst:.3e} (not bit-equal: float32 vs float64)")
    for row in rows:
        assert row["component_target"] == "output_total" and row["mode"] == "direct_target"
        assert "depth_fit" not in row
        assert 0.0 <= row["relative_residual_after"] <= row["relative_residual_before"] + 1e-9
        # desired_norm is the norm of the layer's OWN target D_j^c
        want = float(d[(row["position"], row["component"])].norm())
        assert row["desired_norm"] == pytest.approx(want, rel=1e-5)
        assert row["ridge"] > 0 and row["correction_norm"] > 0


# --- (b) block 0 of the source fine-tuned only: D^c_0 is the local weight-change effect -------------------------


@pytest.mark.parametrize("c", COMPS)
def test_block0_only_target_is_local_weight_change_effect(c):
    # Only layer c of source block 0 is fine-tuned: its input is then unchanged, so D^c_0 is exactly the local effect
    # A_c(X^{s0}, W_ft) - A_c(X^{s0}, W_base) (the historical output_local identity), pushed through Q.
    source_base, source_ft, target_base, data, pairing, sd = _setup(ft_blocks={0}, ft_comps=(c,))
    cfg = _cfg()
    tau, rows, prepared = _fit(source_base, source_ft, target_base, data, pairing, sd, cfg)
    sb = _hooks(source_base, prepared["source_batches"], [0], COMPS)
    tb = _hooks(target_base, prepared["target_batches"], [0], COMPS)
    base_mod, ft_mod = _layer_module(source_base, 0, c), _layer_module(source_ft, 0, c)
    x = _rows(sb[(0, c)]["h"])
    linear = torch.nn.functional.linear
    local = linear(x, ft_mod.weight.detach().double(), ft_mod.bias.detach().double()) - linear(
        x, base_mod.weight.detach().double(), base_mod.bias.detach().double()
    )
    q = _explicit_q(sb[(0, c)]["a"], tb[(0, c)]["a"])
    w, b = _explicit_ridge(_rows(tb[(0, c)]["h"]), local @ q)
    _assert_close(tau[_key(0, c)], w, 1e-5, f"W0 {c}")
    _assert_close(tau[_key(0, c, "bias")], b, 1e-5, f"b0 {c}")
    row = next(r for r in rows if r["position"] == 0 and r["component"] == c)
    assert row["desired_norm"] == pytest.approx(float((local @ q).norm()), rel=1e-5)
    if c == "mlp.c_proj":
        # nothing in block 0 upstream of c_proj changed: the O target of block 0 is zero (later blocks are not: the
        # fine-tuned stream drifts, which is exactly what output_total keeps)
        o_row = next(r for r in rows if r["position"] == 0 and r["component"] == "attn.out_proj")
        assert o_row["desired_norm"] < 1e-5 * row["desired_norm"]


# --- (c) an upstream source change changes later blocks' targets -------------------------------------------------


def test_upstream_source_change_changes_later_targets():
    source_base, ft_a, target_base, data, pairing, sd = _setup(ft_blocks={0})
    ft_b = deepcopy(ft_a)
    with torch.no_grad():
        for c in COMPS:
            _layer_module(ft_b, 0, c).weight.mul_(1.5)  # blocks 1, 2 untouched (equal to ft_a's)
    for j in (1, 2):
        for c in COMPS:
            assert torch.equal(_layer_module(ft_a, j, c).weight, _layer_module(ft_b, j, c).weight)
    cfg = _cfg()
    tau_a, rows_a, _ = _fit(source_base, ft_a, target_base, data, pairing, sd, cfg)
    tau_b, rows_b, _ = _fit(source_base, ft_b, target_base, data, pairing, sd, cfg)
    for ra, rb in zip(rows_a, rows_b, strict=True):
        assert ra["desired_norm"] != pytest.approx(rb["desired_norm"], rel=1e-3), (ra["position"], ra["component"])
    for j in (1, 2):
        assert not torch.equal(tau_a[_key(j, "mlp.c_proj")], tau_b[_key(j, "mlp.c_proj")])


# --- (d) Q_j^c is fitted on component outputs --------------------------------------------------------------------


def test_q_is_fitted_on_component_outputs():
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth=2, target_depth=3)
    cfg = _cfg()
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    sb = _hooks(source_base, prepared["source_batches"], range(2), COMPS)
    tb = _hooks(target_base, prepared["target_batches"], range(3), COMPS)
    boundary = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=3, seed=cfg.seed, device="cpu"
    )
    for j in range(3):
        i = pairing.pairing[j]
        for c in COMPS:
            q = prepared["q_component_by_position"][(j, c)]
            assert q.dtype == torch.float32 and tuple(q.shape) == (WIDTH, WIDTH)
            want = _explicit_q(sb[(i, c)]["a"], tb[(j, c)]["a"])
            assert float((q.double() - want).abs().max()) < 1e-5
            assert not torch.allclose(q, boundary["q_by_position"][j], atol=1e-3)  # not the block-boundary map
            assert prepared["component_procrustes_diagnostics"][(j, c)]["procrustes_source"] == "activation"


# --- (e) config -------------------------------------------------------------------------------------------------


BASE = dict(activation_storage="streaming", component_target="output_total")


def test_config_acceptance():
    assert parse_direct_residual_config({**BASE}).component_target == "output_total"
    assert parse_direct_residual_config({**BASE, "components": ["attn.out_proj"]}).components == ("attn.out_proj",)
    assert parse_direct_residual_config({**BASE, "components": ["mlp.c_proj"]}).components == ("mlp.c_proj",)
    for comps in (["mlp.c_proj"], ["attn.out_proj"], list(COMPS)):
        cfg = parse_direct_residual_config({**BASE, "depth_fit": "sequential", "components": comps})
        assert cfg.depth_fit == "sequential"
    assert direct_residual_config_dict(parse_direct_residual_config({**BASE}))["component_target"] == "output_total"


@pytest.mark.parametrize(
    "bad",
    [
        {"activation_storage": "resident"},
        {"block_split": "backfit"},
        {"components": ["attn.v_proj"]},
        {"components": ["mlp.c_fc", "mlp.c_proj"]},
        {"component_fit": "gradient_exact"},
        {"procrustes_source": "gradient"},
        {"tv_scaling": "global"},
        {"residual_target": "transported_endpoint"},
        {"v_fit": "preimage", "components": ["attn.v_proj"]},
        {"pair_target_split": "per_block_share"},
        {"fit_blocks": [0]},
        {"depth_pairing": "reversed"},
        {"realization_diagnostics": True},
        {"depth_fit": "sequential", "sequential_points": "end_end"},
        {"depth_fit": "sequential", "sequential_impl": "block_carry"},
        {"depth_fit": "sequential", "block_split": "backfit"},
    ],
)
def test_config_rejections(bad):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**BASE, **bad})


def test_output_local_and_default_serialization():
    with pytest.raises(ValueError, match="retired"):
        parse_direct_residual_config({"component_target": "output_local"})
    with pytest.raises(ValueError, match="retired"):
        parse_direct_residual_config({**BASE, "component_target": "output_local"})
    default = direct_residual_config_dict(parse_direct_residual_config({}))
    assert default["component_target"] == "block_boundary"
    assert not {"depth_fit", "sequential_points"} & set(default)


# --- (f) sequential: first fitted layer equals the independent fit -----------------------------------------------


def test_sequential_first_layer_equals_independent_bitwise():
    source_base, source_ft, target_base, data, pairing, sd = _setup()
    tau_i, _, prepared = _fit(source_base, source_ft, target_base, data, pairing, sd, _cfg())
    cfg_s = _cfg("sequential")
    tau_s, rows_s = fit_output_total_sequential_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg_s, device="cpu"
    )
    first = [_key(0, "attn.out_proj"), _key(0, "attn.out_proj", "bias")]
    assert _sha({k: tau_s[k] for k in first}) == _sha({k: tau_i[k] for k in first})
    assert rows_s[0]["upstream_effect_norm"] == 0.0 and rows_s[0]["depth_fit"] == "sequential"
    # the second layer of block 0 already sees O_0 mounted: it differs from the independent fit
    assert not torch.equal(tau_s[_key(0, "mlp.c_proj")], tau_i[_key(0, "mlp.c_proj")])
    assert [(r["position"], r["component"]) for r in rows_s] == [(j, c) for j in range(DEPTH) for c in COMPS]


# --- (g) sequential, one component: target = D - measured upstream change ----------------------------------------


def test_sequential_single_component_target_is_d_minus_upstream():
    source_base, source_ft, target_base, data, pairing, sd = _setup()
    c = "mlp.c_proj"
    cfg = _cfg("sequential", components=(c,))
    tau, rows, prepared = _fit(source_base, source_ft, target_base, data, pairing, sd, cfg)
    d, _q, tb = _explicit_targets(source_base, source_ft, target_base, prepared, range(DEPTH), (c,))
    worst = 0.0
    for j in range(DEPTH):
        cur = _mounted(target_base, sd, tau, lambda k, j=j: _block_of(k) < j)
        cap = _hooks(cur, prepared["target_batches"], [j], (c,))[(j, c)]
        upstream = _rows(cap["a"]) - _rows(tb[(j, c)]["a"])
        if j == 0:
            assert float(upstream.abs().max()) == 0.0
        else:
            assert float(upstream.norm()) > 1e-6
        w, b = _explicit_ridge(_rows(cap["h"]), d[(j, c)] - upstream)
        worst = max(worst, _assert_close(tau[_key(j, c)], w, 1e-5, f"W {j}"))
        worst = max(worst, _assert_close(tau[_key(j, c, "bias")], b, 1e-5, f"b {j}"))
        row = next(r for r in rows if r["position"] == j)
        assert row["upstream_effect_norm"] == pytest.approx(float(upstream.norm()), rel=1e-4, abs=1e-12)
        assert row["desired_norm"] == pytest.approx(float(d[(j, c)].norm()), rel=1e-5)
    print(f"sequential (D only) vs explicit float64: worst relative error {worst:.3e}")


# --- (h) sequential O+D: D's upstream effect includes the O correction of the same block --------------------------


def test_sequential_d_sees_same_block_o_correction():
    source_base, source_ft, target_base, data, pairing, sd = _setup()
    cfg = _cfg("sequential")
    tau, rows, prepared = _fit(source_base, source_ft, target_base, data, pairing, sd, cfg)
    d, _q, tb = _explicit_targets(source_base, source_ft, target_base, prepared, range(DEPTH), COMPS)
    for j in range(DEPTH):
        for c in COMPS:
            # layers fitted before (j, c): all blocks < j, plus O_j when c is the MLP projection
            def keep(k, j=j, c=c):
                b = _block_of(k)
                return b < j or (b == j and c == "mlp.c_proj" and ".attn.out_proj." in k)

            cur = _mounted(target_base, sd, tau, keep)
            cap = _hooks(cur, prepared["target_batches"], [j], (c,))[(j, c)]
            upstream = _rows(cap["a"]) - _rows(tb[(j, c)]["a"])
            w, b = _explicit_ridge(_rows(cap["h"]), d[(j, c)] - upstream)
            _assert_close(tau[_key(j, c)], w, 1e-5, f"W {j} {c}")
            _assert_close(tau[_key(j, c, "bias")], b, 1e-5, f"b {j} {c}")
            row = next(r for r in rows if r["position"] == j and r["component"] == c)
            assert row["upstream_effect_norm"] == pytest.approx(float(upstream.norm()), rel=1e-4, abs=1e-12)
            if c == "mlp.c_proj":
                without_o = _mounted(target_base, sd, tau, lambda k, j=j: _block_of(k) < j)
                cap0 = _hooks(without_o, prepared["target_batches"], [j], (c,))[(j, c)]
                u0 = _rows(cap0["a"]) - _rows(tb[(j, c)]["a"])
                assert float((upstream - u0).norm()) > 1e-4  # the O_j correction moves D's output
                assert row["upstream_effect_norm"] != pytest.approx(float(u0.norm()), rel=1e-6)


# --- (i) decoders, determinism, dispatch, state restoration ------------------------------------------------------


def test_decoders_raise_not_implemented():
    src = tiny_qwen2(layers=2, seed=0)
    fam = infer_family(src)
    with pytest.raises(NotImplementedError):
        resolve_ariadne_decoder_config({"component_target": "output_total"}, fam)
    pairing = DiscreteLayerPairing.compute(2, 2)
    with pytest.raises(NotImplementedError):
        prepare_output_total_streaming(
            src, src, None, None, pairing, num_batches=1, seed=0, device="cpu", components=COMPS, family_adapter=fam
        )
    source_base, source_ft, target_base, data, vpairing, sd = _setup()
    cfg = _cfg()
    prepared = _prepare(source_base, target_base, data, vpairing, cfg)
    for fit in (fit_output_total_independent_streaming, fit_output_total_sequential_streaming):
        with pytest.raises(NotImplementedError):
            fit(
                target_base,
                sd,
                source_base,
                source_ft,
                prepared,
                vpairing,
                config=cfg,
                device="cpu",
                family_adapter=fam,
            )
    with pytest.raises(NotImplementedError):
        AriadneRebase().prepare(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target_base,
            target_base_sd=sd,
            source_loader=data,
            target_loader=data,
            pairing=vpairing,
            config=cfg,
            device="cpu",
            family_adapter=fam,
        )


@pytest.mark.parametrize("depth_fit", ["independent", "sequential"])
def test_deterministic_dispatch_and_state_restored(depth_fit):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth=2, target_depth=3)
    cfg = _cfg(depth_fit)
    before = _sha({k: v.detach().contiguous() for k, v in target_base.state_dict().items()})
    hashes = []
    for _ in range(2):
        prep = AriadneRebase().prepare(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target_base,
            target_base_sd=sd,
            source_loader=data,
            target_loader=data,
            pairing=pairing,
            config=cfg,
            device="cpu",
        )
        hashes.append(_sha(prep.task_vector))
    assert hashes[0] == hashes[1]
    assert _sha({k: v.detach().contiguous() for k, v in target_base.state_dict().items()}) == before
    assert prep.extra["alignment_diagnostics"] is None
    assert {r["component_target"] for r in prep.diagnostics} == {"output_total"}
    assert len(prep.diagnostics) == 3 * 2
    # direct call equals the method path
    tau, _, _ = _fit(source_base, source_ft, target_base, data, pairing, sd, cfg)
    assert _sha(tau) == hashes[0]
    # strength-scaled unit vector is the same fit
    assert set(prep.unit_task_vector) == set(tau)
