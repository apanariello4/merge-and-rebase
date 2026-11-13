"""Closed-form pre-image correction of the attention V projection (``v_fit="preimage"``, Ariadne, vision).

Pinned: the per-head mixed inputs ``Z_h = A_h X`` reproduce the pre-``out_proj`` rows through the V slice; mounting
``(dV, db)`` moves those rows exactly by ``Z_h dV_h^T + 1 db_h^T``; the pre-image ``T = D L^+`` satisfies ``T L = D``; with a
vanishing ridge the per-head solve is the explicit least squares with intercept; the empirical-Bayes ridge is the O/D path's
formula on the same features; only the V rows of the packed ``in_proj`` change; the residual-space diagnostic equals an
independent first-order brute force; config validation and default serialization.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.capture import (
    _stock_mha_head_mixed_input,
    _stock_mha_out_proj_input,
    _v_apply_mixed,
    capture_tokens,
)
from merge_and_rebase.rebase.methods._ariadne.config import (
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods._ariadne.layouts import _layout_for
from merge_and_rebase.rebase.methods._ariadne.streaming import prepare_direct_residual_streaming
from merge_and_rebase.rebase.methods._ariadne.v_fit import (
    VFitBlockStatistics,
    fit_direct_residual_vfit_streaming,
    vfit_write_map,
)

WIDTH, HEADS = 8, 2
DH = WIDTH // HEADS


class _LayerScale(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = torch.nn.Parameter(0.5 + torch.rand(width))

    def forward(self, x):
        return x * self.gamma


class _Block(torch.nn.Module):
    def __init__(self, scaled):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(WIDTH)
        self.attn = torch.nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.ls_1 = _LayerScale(WIDTH) if scaled else torch.nn.Identity()
        self.ln_2 = torch.nn.LayerNorm(WIDTH)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(WIDTH, 2 * WIDTH)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(2 * WIDTH, WIDTH)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.ls_1(self.attn(h, h, h, need_weights=False)[0])
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class _Visual(torch.nn.Module):
    def __init__(self, depth, scaled):
        super().__init__()
        self.input = torch.nn.Linear(4, WIDTH)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(scaled) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, depth, scaled):
        super().__init__()
        self.visual = _Visual(depth, scaled)

    def encode_image(self, x):
        return self.visual(x)


def _setup(regime=(2, 4), scaled=False, seed=31):
    torch.manual_seed(seed)
    source_base = _Model(regime[0], scaled).double().eval()
    target_base = _Model(regime[1], scaled).double().eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            for p in (block.mlp.c_proj.weight, block.attn.out_proj.weight, block.attn.in_proj_weight):
                p.add_(0.2 * torch.randn_like(p))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(6, 5, 4, generator=generator).double()
    data = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=2, shuffle=False)
    pairing = DiscreteLayerPairing.compute(*regime)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


BASE = dict(
    activation_storage="streaming",
    num_batches=3,
    ridge_relative=0.05,
    ridge_estimator="empirical_bayes",
    components=("attn.v_proj",),
    v_fit="preimage",
)


def _mha(seed=3):
    torch.manual_seed(seed)
    return torch.nn.MultiheadAttention(WIDTH, HEADS, batch_first=True).double().eval()


def _mixed_and_rows(attn, x):
    """Float64 ``(Z [B,T,H*d], out_proj input rows [B,T,d])`` of a self-attention call, via the production helpers."""
    out = {}

    def hook(mod, args, kwargs, _value):
        out["z"] = _stock_mha_head_mixed_input(mod, args, kwargs)
        out["rows"] = _stock_mha_out_proj_input(mod, args, kwargs)

    handle = attn.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            attn(x, x, x, need_weights=False)
    finally:
        handle.remove()
    return out["z"], out["rows"]


def _toy_stats(scaled=False, n_batches=4, seed=3, solve_cfg=None):
    attn = _mha(seed)
    block = _Block(scaled).double().eval()
    block.attn = attn
    torch.manual_seed(seed + 5)
    zs, rows = [], []
    for _ in range(n_batches):
        x = torch.randn(3, 7, WIDTH).double()
        z, r = _mixed_and_rows(attn, x)
        zs.append(z)
        rows.append(r)
    z = torch.cat(zs).reshape(-1, HEADS * WIDTH)
    desired = torch.randn(z.shape[0], WIDTH, dtype=torch.float64)
    shim = _layout_for(None)
    stats = VFitBlockStatistics(vfit_write_map(shim, block), HEADS)
    stats.update(z, desired)
    return attn, block, z, desired, stats


def _cfg(**over):
    return parse_direct_residual_config({**BASE, **over})


# ---- capture ---------------------------------------------------------------------------------------------------------


def test_mixed_inputs_reproduce_out_proj_input():
    attn = _mha()
    x = torch.randn(3, 7, WIDTH).double()
    z, rows = _mixed_and_rows(attn, x)
    assert z.shape == (3, 7, HEADS * WIDTH)
    assert (_v_apply_mixed(attn, z) - rows).abs().max() < 1e-12


def test_mixed_capture_hook_verifies_and_rejects_non_self_attention():
    attn = _mha()
    x = torch.randn(2, 5, WIDTH).double()
    model = _Model(1, False).double().eval()
    model.visual.transformer.resblocks[0].attn = attn
    got = capture_tokens(model, [(x[:, :, :4],)], {"z": (0, "attn_mixed_input")}, "cpu")
    assert got["z"][0].shape == (2, 5, HEADS * WIDTH)
    with pytest.raises(ValueError, match="self-attention"):
        _stock_mha_head_mixed_input(attn, (x, x.clone(), x.clone()), {})


# ---- (a) mounting is exact ---------------------------------------------------------------------------------------------


def test_mounting_dv_db_moves_pre_o_rows_exactly():
    attn, _, z, _, stats = _toy_stats()
    dv, db, _ = stats.solve(_cfg())
    dv, db = dv.float().double(), db.float().double()  # what the fit writes (float32)
    mounted = deepcopy(attn)
    with torch.no_grad():
        mounted.in_proj_weight[2 * WIDTH :] += dv
        mounted.in_proj_bias[2 * WIDTH :] += db
    torch.manual_seed(77)
    x = torch.randn(3, 7, WIDTH).double()
    z0, rows0 = _mixed_and_rows(attn, x)
    z1, rows1 = _mixed_and_rows(mounted, x)
    assert (z1 - z0).abs().max() < 1e-12  # V changes do not move the attention probabilities or X
    predicted = torch.cat(
        [
            z0[..., h * WIDTH : (h + 1) * WIDTH] @ dv[h * DH : (h + 1) * DH].T + db[h * DH : (h + 1) * DH]
            for h in range(HEADS)
        ],
        dim=-1,
    )
    assert (rows1 - rows0 - predicted).abs().max() <= 1e-10


# ---- (b) lambda -> 0 is explicit least squares ------------------------------------------------------------------------


def test_vanishing_ridge_equals_explicit_lstsq_with_intercept():
    _, _, z, desired, stats = _toy_stats()
    dv, db, _ = stats.solve(_cfg(ridge_estimator="fixed_relative", ridge_relative=1e-12))
    t = stats.target(desired)
    for h in range(HEADS):
        zh = z[:, h * WIDTH : (h + 1) * WIDTH]
        design = torch.cat([zh, torch.ones(zh.shape[0], 1, dtype=torch.float64)], dim=1)
        sol = torch.linalg.lstsq(design, t[:, h * DH : (h + 1) * DH], driver="gelsd").solution
        assert torch.allclose(dv[h * DH : (h + 1) * DH].double(), sol[:-1].T, atol=1e-5, rtol=1e-5)
        assert torch.allclose(db[h * DH : (h + 1) * DH].double(), sol[-1], atol=1e-5, rtol=1e-5)


# ---- (c) pre-image ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scaled", [False, True])
def test_preimage_times_write_map_is_the_target(scaled):
    _, block, z, desired, stats = _toy_stats(scaled=scaled)
    t = stats.target(desired)
    assert (t @ stats.write_map - desired).abs().max() < 1e-10
    # L is the real write map: H -> ls_1(out_proj(H)) minus the constant out_proj bias
    h = torch.randn(5, WIDTH, dtype=torch.float64)
    with torch.no_grad():
        full = block.ls_1(block.attn.out_proj(h)) - block.ls_1(block.attn.out_proj(torch.zeros_like(h)))
    assert (h @ stats.write_map - full).abs().max() < 1e-12
    assert stats.cond >= 1.0


# ---- (d) empirical-Bayes ridge ----------------------------------------------------------------------------------------


def test_eb_ridge_is_the_o_path_formula_on_the_same_features():
    _, _, z, desired, stats = _toy_stats()
    _, _, info = stats.solve(_cfg())
    t = stats.target(desired)
    n = z.shape[0]
    for h in range(HEADS):
        zh = z[:, h * WIDTH : (h + 1) * WIDTH]
        zc = zh - zh.mean(0)
        by_hand = float((zc**2).sum()) / (n - 1)
        o_path = ResidualSufficientStatistics()
        o_path.update(zh, t[:, h * DH : (h + 1) * DH], None, torch.eye(DH))
        _, diag = o_path.solve(ridge_relative=0.05, ridge_estimator="empirical_bayes", exact_form=True)
        assert info["ridge_per_head"][h] == pytest.approx(by_hand, rel=1e-10)
        assert diag["ridge"] == pytest.approx(by_hand, rel=1e-10)


# ---- end to end -------------------------------------------------------------------------------------------------------


def _block_io(model, images):
    """Per block: (input, output) of the pristine block on ``images`` (hooks)."""
    ins, outs = {}, {}
    handles = []
    for j, block in enumerate(model.visual.transformer.resblocks):
        handles.append(block.register_forward_pre_hook(lambda m, a, j=j: ins.__setitem__(j, a[0].detach())))
        handles.append(block.register_forward_hook(lambda m, a, o, j=j: outs.__setitem__(j, o.detach())))
    try:
        with torch.no_grad():
            model.encode_image(images)
    finally:
        for h in handles:
            h.remove()
    return ins, outs


def _fit(scaled=False, **over):
    source_base, source_ft, target_base, data, pairing, sd = _setup(scaled=scaled)
    cfg = _cfg(**over)
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu"
    )
    before = {k: v.clone() for k, v in target_base.state_dict().items()}
    tau, rows = fit_direct_residual_vfit_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    for k, v in target_base.state_dict().items():
        assert torch.equal(v, before[k])  # target restored
    return tau, rows, prepared, (source_base, source_ft, target_base, pairing, sd)


@pytest.mark.parametrize("scaled", [False, True])
def test_end_to_end_slice_only_and_residual_space_diagnostic(scaled):
    tau, rows, prepared, (source_base, source_ft, target_base, pairing, sd) = _fit(scaled=scaled)
    depth = pairing.target_depth
    d = WIDTH
    # (e) only the V rows of in_proj_weight / in_proj_bias; nothing else
    expected_keys = {
        f"visual.transformer.resblocks.{j}.attn.{n}" for j in range(depth) for n in ("in_proj_weight", "in_proj_bias")
    }
    assert set(tau) == expected_keys
    for key, value in tau.items():
        assert value.shape == sd[key].shape
        assert torch.count_nonzero(value[: 2 * d]) == 0
        assert torch.count_nonzero(value[2 * d :]) > 0
    assert [r["position"] for r in rows] == list(range(depth))
    for row in rows:
        assert row["component"] == "attn.v_proj" and row["v_fit"] == "preimage"
        assert row["preimage_rel_error"] < 1e-9 and row["out_proj_cond"] >= 1.0
        assert row["ridge_min"] <= row["ridge"] <= row["ridge_max"] and len(row["ridge_per_head"]) == HEADS
        assert 0.0 <= row["relative_residual_preimage"] <= 1.0 + 1e-9
        assert row["correction_norm"] > 0 and row["bias_norm"] > 0 and row["desired_norm"] > 0
    # brute force: first-order residual-space error of the mounted correction vs D = (f - b) Q, per block
    q = prepared["q_by_position"]
    err_sq = {j: 0.0 for j in range(depth)}
    d_sq = {j: 0.0 for j in range(depth)}
    for batch in prepared["target_batches"]:
        images = batch[0]
        _, b_out = _block_io(source_base, images)
        _, f_out = _block_io(source_ft, images)
        ins, _ = _block_io(target_base, images)
        for j in range(depth):
            block = target_base.visual.transformer.resblocks[j]
            mounted = deepcopy(block)
            with torch.no_grad():
                mounted.attn.in_proj_weight += tau[f"visual.transformer.resblocks.{j}.attn.in_proj_weight"].double()
                mounted.attn.in_proj_bias += tau[f"visual.transformer.resblocks.{j}.attn.in_proj_bias"].double()
                h = block.ln_1(ins[j])
                delta = mounted.ls_1(mounted.attn(h, h, h, need_weights=False)[0]) - block.ls_1(
                    block.attn(h, h, h, need_weights=False)[0]
                )
            i = pairing.pairing[j]
            target = (f_out[i] - b_out[i]) @ q[j].double()
            err_sq[j] += float(((delta - target) ** 2).sum())
            d_sq[j] += float((target**2).sum())
    for row in rows:
        j = row["position"]
        assert row["relative_residual_after"] == pytest.approx((err_sq[j] / d_sq[j]) ** 0.5, rel=1e-4)
        assert row["desired_norm"] == pytest.approx(d_sq[j] ** 0.5, rel=1e-5)


def test_position_chunking_is_equivalent():
    full, _, _, _ = _fit()
    chunked, _, _, _ = _fit(streaming_position_chunk=1)
    for key in full:
        assert torch.equal(full[key], chunked[key])


# ---- config -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"v_fit": "bogus"},
        {"components": ("attn.v_proj", "mlp.c_proj")},
        {"components": ("attn.out_proj",)},
        {"component_fit": "gradient_exact"},
        {"depth_fit": "sequential"},
        {"activation_storage": "resident"},
        {"residual_target": "transported_endpoint"},
        {"pair_target_split": "per_block_share"},
        {"fit_blocks": (0, 1)},
        {"tv_scaling": "global"},
        {"block_split": "joint"},
        {"exact_form": False},
        {"fidelity_holdout": True},
    ],
)
def test_validation_rejects_invalid_combinations(bad):
    with pytest.raises(ValueError):
        _cfg(**bad)


def test_v_proj_without_v_fit_is_still_rejected_and_default_is_unchanged():
    with pytest.raises(ValueError):
        parse_direct_residual_config({**BASE, "v_fit": None})  # closed form cannot fit attn.v_proj
    default = parse_direct_residual_config({"activation_storage": "streaming", "components": ("mlp.c_proj",)})
    assert default.v_fit is None
    assert "v_fit" not in direct_residual_config_dict(default)
    assert direct_residual_config_dict(_cfg())["v_fit"] == "preimage"


def test_decoder_path_is_not_implemented():
    with pytest.raises(NotImplementedError, match="vision-only"):
        resolve_ariadne_decoder_config(dict(BASE), family_adapter=object())
