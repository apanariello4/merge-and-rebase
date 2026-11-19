"""S1 / S2 / S3 of the anchors synthesis (anchors/synthesis.py fit_tracking): closed-loop depth fit and label-free gain.

The toy network is exactly affine (block: ``x + c_proj(c_fc(x))``, no activation, attention output zero), so the
answer is known: the desired change ``D_j`` is the cumulative stream change of a hidden 'true' update, hence
realizable, and a correct closed-loop fit realizes it exactly (norm ratio 1) while the independent fit, whose every
block rewrites the whole cumulative change on top of the earlier blocks, overshoots.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.capture import capture_tokens, iter_capture_tokens
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.fit import fit_direct_residual
from merge_and_rebase.rebase.methods._ariadne.layouts import COMPONENT_INPUT_KIND
from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.synthesis import (
    GenericInputStats,
    capture_generic_inputs,
    fit_tracking,
    make_synthesis_hook,
)

WIDTH, DEPTH, TOKENS = 6, 4, 5
KEY = "visual.transformer.resblocks.{}.mlp.c_proj"


class _Zero(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.out_proj = _Zero()
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(width, 2 * width)), ("act", torch.nn.Identity()),
            ("c_proj", torch.nn.Linear(2 * width, width))]))
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn.out_proj(x) + self.ls_2(self.mlp(x))


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


def _batches(n_batches, seed):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(2 * n_batches, TOKENS, 4, generator=g)
    return list(DataLoader(TensorDataset(images, torch.zeros(2 * n_batches)), batch_size=2, shuffle=False))


def _toy(depth=DEPTH, scale=0.5, seed=3, n_batches=8):
    """``(base model, base state, batches, captured, D, truth)``; D_j = cumulative change of the hidden true update."""
    torch.manual_seed(seed)
    base = _Model(WIDTH, depth).eval()
    truth = deepcopy(base)
    with torch.no_grad():
        for blk in truth.visual.transformer.resblocks:
            blk.mlp.c_proj.weight.add_(scale * torch.randn_like(blk.mlp.c_proj.weight))
            blk.mlp.c_proj.bias.add_(0.1 * torch.randn_like(blk.mlp.c_proj.bias))
    batches = _batches(n_batches, seed + 1)
    req = {str(j): (j, "boundary") for j in range(depth)}
    b_out = capture_tokens(base, batches, req, "cpu")
    t_out = capture_tokens(truth, batches, req, "cpu")
    desired = {j: [t - b for t, b in zip(t_out[str(j)], b_out[str(j)], strict=True)] for j in range(depth)}
    captured = {"target_batches": batches, "target_base_outputs_by_position": {j: b_out[str(j)] for j in range(depth)}}
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    return base, sd, batches, captured, desired, truth


CFG = DirectResidualConfig(components=("mlp.c_proj",), ridge_estimator="fixed_relative", ridge_relative=1e-9,
                           num_batches=8)


def _mount(base_sd, corrections, alpha=1.0):
    out = {k: v.clone() for k, v in base_sd.items()}
    for k, v in corrections.items():
        out[k] = out[k] + alpha * v.to(out[k].dtype)
    return out


def _ratios(model, sd, batches, desired, corrections, depth=DEPTH):
    """Per block: ``||realized|| / ||desired||`` and the cosine, realized = boundary change of the mounted model."""
    edited = deepcopy(model)
    edited.load_state_dict(_mount(sd, corrections))
    req = {str(j): (j, "boundary") for j in range(depth)}
    base_out = capture_tokens(model, batches, req, "cpu")
    new_out = capture_tokens(edited, batches, req, "cpu")
    ratio, cos = [], []
    for j in range(depth):
        r = torch.cat([(n - b).reshape(-1) for n, b in zip(new_out[str(j)], base_out[str(j)], strict=True)]).double()
        d = torch.cat([x.reshape(-1) for x in desired[j]]).double()
        ratio.append(float(r.norm() / d.norm()))
        cos.append(float(r @ d / (r.norm() * d.norm())))
    return ratio, cos


def _independent(model, sd, captured, desired, hook=None):
    pairing = DiscreteLayerPairing.compute(DEPTH, DEPTH)
    cfg = DirectResidualConfig(**{**CFG.__dict__})
    corr, rows = fit_direct_residual(model, sd, captured, desired, pairing, config=cfg, device="cpu", solve_hook=hook)
    return corr, rows


def test_independent_overshoots_and_closed_loop_reaches_unit_ratio():
    model, sd, _, captured, desired, truth = _toy()
    batches = captured["target_batches"]
    ind, _ = _independent(model, sd, captured, desired)
    r_ind, _ = _ratios(model, sd, batches, desired, ind)
    trk, rows = fit_tracking(model, sd, captured, desired, config=CFG, device="cpu")
    r_trk, c_trk = _ratios(model, sd, batches, desired, trk)
    assert r_ind[0] == pytest.approx(1.0, abs=1e-3)  # block 0 has no upstream
    assert r_ind[-1] > 1.5 and max(r_ind) > 1.5       # the independent fit overshoots in depth
    assert all(r == pytest.approx(1.0, abs=2e-3) for r in r_trk), r_trk
    assert all(c > 0.999 for c in c_trk)
    assert rows[0]["upstream_norm"] == 0.0
    # the model is restored
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd[k])


def test_closed_loop_with_preserve_hook_stays_near_unit_ratio_where_independent_preserve_overshoots():
    model, sd, _, captured, desired, _ = _toy()
    batches = captured["target_batches"]
    gen_batches = _batches(4, 99)
    generic = capture_generic_inputs(model, gen_batches, DEPTH, component="mlp.c_proj", tokens_per_block=40, seed=0,
                                     device="cpu")
    weights = {j: sd[KEY.format(j) + ".weight"] for j in range(DEPTH)}
    hook = make_synthesis_hook(generic=generic, rho_g=1e-3, base_weights=weights)
    ind, _ = _independent(model, sd, captured, desired, hook)
    trk, rows = fit_tracking(model, sd, captured, desired, config=CFG, device="cpu", solve_hook=hook)
    r_ind, _ = _ratios(model, sd, batches, desired, ind)
    r_trk, _ = _ratios(model, sd, batches, desired, trk)
    assert max(r_ind) > 1.5
    assert all(abs(r - 1.0) < 0.05 for r in r_trk), r_trk
    assert all("generic_change" in r for r in rows)


def test_preserve_on_updated_inputs_is_recaptured_per_block():
    from merge_and_rebase.rebase.methods.anchors.synthesis import LazyGenericInputs

    model, sd, _, captured, desired, _ = _toy()
    gen_batches = _batches(4, 99)
    lazy = LazyGenericInputs(model, gen_batches, component="mlp.c_proj", tokens_per_block=40, seed=0, device="cpu")
    weights = {j: sd[KEY.format(j) + ".weight"] for j in range(DEPTH)}
    hook = make_synthesis_hook(generic=lazy, rho_g=1.0, base_weights=weights)
    fit_tracking(model, sd, captured, desired, config=CFG, device="cpu", solve_hook=hook)
    assert lazy.captures == DEPTH  # one sweep per block, each on the model with the upstream blocks mounted
    base_g = capture_generic_inputs(model, gen_batches, DEPTH, component="mlp.c_proj", tokens_per_block=40, seed=0,
                                    device="cpu")
    # block 0: the updated model equals the base; deeper blocks see a different input scatter
    assert isinstance(base_g[0], GenericInputStats)


def test_gain_equals_the_analytic_value_for_one_block():
    model, sd, _, captured, desired, _ = _toy(depth=1)
    fitted, _ = fit_tracking(model, sd, captured, {0: desired[0]}, config=CFG, device="cpu")
    shrunk = {k: 0.6 * v for k, v in fitted.items()}  # realizes 0.6 D: the exact gain is 1 / 0.6
    scaled, rows = fit_tracking(model, sd, captured, {0: desired[0]}, config=CFG, device="cpu", fixed=shrunk,
                                gain="lsq", holdout=0)
    assert rows[0]["gain"] == pytest.approx(1 / 0.6, rel=1e-3)
    assert rows[0]["gain_lsq_raw"] == pytest.approx(1 / 0.6, rel=1e-3)
    for k in fitted:
        torch.testing.assert_close(scaled[k], fitted[k], rtol=2e-3, atol=2e-4)
    assert rows[0]["realized_over_desired_gain_rows"] == pytest.approx(1.0, abs=1e-3)
    # holdout rows: same value on exactly linear data; and a gain above the bound is clipped
    _, hold = fit_tracking(model, sd, captured, {0: desired[0]}, config=CFG, device="cpu", fixed=shrunk, gain="lsq",
                           holdout=2)
    assert hold[0]["gain"] == pytest.approx(1 / 0.6, rel=1e-3) and hold[0]["gain_rows"] == "holdout"
    tiny = {k: 0.1 * v for k, v in fitted.items()}
    _, clipped = fit_tracking(model, sd, captured, {0: desired[0]}, config=CFG, device="cpu", fixed=tiny, gain="lsq")
    assert clipped[0]["gain_lsq_raw"] == pytest.approx(10.0, rel=1e-3) and clipped[0]["gain"] == 2.0


def test_gain_of_the_independent_fit_matches_the_formula_in_depth_order_and_removes_the_overshoot():
    model, sd, _, captured, desired, _ = _toy()
    batches = captured["target_batches"]
    ind, _ = _independent(model, sd, captured, desired)
    scaled, rows = fit_tracking(model, sd, captured, desired, config=CFG, device="cpu", fixed=ind, gain="lsq", holdout=0)
    gains = [r["gain"] for r in rows]
    assert gains[0] == pytest.approx(1.0, abs=1e-3)  # block 0: no upstream, the fit already realizes D_0
    assert all(g < 0.9 for g in gains[2:]) and all(0.0 <= g <= 2.0 for g in gains)
    # g_1 from first principles: U_1 with block 0 mounted (gain 1), own_1 = H_1 dW_1 + beta_1 on that model
    edited = deepcopy(model)
    edited.load_state_dict(_mount(sd, {k: v for k, v in ind.items() if ".resblocks.0." in k}))
    num = den = 0.0
    for k, vals in enumerate(iter_capture_tokens(edited, batches, {"out": (1, "boundary"), "h": (1, COMPONENT_INPUT_KIND["mlp.c_proj"])}, "cpu")):
        base_out = captured["target_base_outputs_by_position"][1][k]
        res = (desired[1][k] - (vals["out"] - base_out)).double().reshape(-1, WIDTH)
        h = vals["h"].double().reshape(-1, 2 * WIDTH)
        own = h @ ind[KEY.format(1) + ".weight"].double().T + ind[KEY.format(1) + ".bias"].double()
        num += float((own * res).sum())
        den += float((own * own).sum())
    assert gains[1] == pytest.approx(num / den, rel=2e-3)
    ratio, _ = _ratios(model, sd, batches, desired, scaled)
    r_ind, _ = _ratios(model, sd, batches, desired, ind)
    # a scalar per block cannot undo what the upstream already overshoots (g >= 0): a partial fix, far below the
    # independent overshoot of this strongly amplifying toy, exact at the first blocks
    assert abs(ratio[1] - 1.0) < 0.02 and max(ratio) < max(r_ind) / 5
    assert gains == sorted(gains, reverse=True)


def test_tracking_with_gain_and_holdout_runs_and_fits_on_the_remaining_rows():
    model, sd, _, captured, desired, _ = _toy()
    scaled, rows = fit_tracking(model, sd, captured, desired, config=CFG, device="cpu", gain="lsq", holdout=2)
    assert all(r["gain_rows"] == "holdout" and abs(r["gain"] - 1.0) < 1e-2 for r in rows)  # exact closed loop: g = 1
    with pytest.raises(ValueError, match="holdout"):
        fit_tracking(model, sd, captured, desired, config=CFG, device="cpu", gain="lsq", holdout=8)


# ---------------------------------------------------------------- config and end to end


def test_config_options_defaults_and_validation():
    d = AnchorsConfig()
    assert (d.gain_calibration, d.preserve_inputs, d.gain_holdout_batches, d.depth_fit) == ("none", "base", 2, "independent")
    s1 = parse_anchors_config({"depth_fit": "sequential", "synthesis": "ridge_preserve"})
    assert s1.synthesis == "ridge_preserve"
    assert parse_anchors_config({"depth_fit": "sequential"}).synthesis == "ridge"  # the default is unchanged
    s2 = parse_anchors_config({"depth_fit": "tracking", "preserve_inputs": "updated"})
    assert (s2.synthesis, s2.rho_g) == ("ridge_preserve", 0.1)
    for bad in ({"preserve_inputs": "updated"}, {"preserve_inputs": "x"}, {"gain_calibration": "x"},
                {"gain_calibration": "lsq", "depth_fit": "spread", "measure_depths": [1.0]},
                {"gain_calibration": "lsq", "gain_holdout_batches": 10}, {"depth_fit": "tracking", "map_rows": "task"}):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)


def _method_setup():
    from tests.test_anchors_method import _cfg, _prepare, _setup  # noqa: PLC0415

    return _cfg, _prepare, _setup


def test_tracking_with_the_plain_ridge_is_the_sequential_fit():
    _cfg, _prepare, _setup = _method_setup()
    setup = _setup(2, 4)
    seq = _prepare(_cfg(depth_fit="sequential"), setup)
    trk = _prepare(_cfg(depth_fit="tracking", synthesis="ridge"), setup)
    assert set(seq.unit_task_vector) == set(trk.unit_task_vector)
    for k, v in seq.unit_task_vector.items():
        torch.testing.assert_close(trk.unit_task_vector[k], v, rtol=1e-3, atol=1e-5)
    assert all(r["mode"] == "anchors_tracking" and r["closed_loop"] for r in trk.diagnostics)


def test_s1_sequential_preserve_end_to_end():
    _cfg, _prepare, _setup = _method_setup()
    setup = _setup(2, 4)
    seq = _prepare(_cfg(depth_fit="sequential"), setup)
    zero = _prepare(_cfg(depth_fit="sequential", synthesis="ridge_preserve", rho_g=0.0, preserve_tokens=6), setup)
    for k, v in seq.unit_task_vector.items():  # rho_g = 0 is the plain sequential ridge
        torch.testing.assert_close(zero.unit_task_vector[k], v, rtol=1e-3, atol=1e-5)
    pos = _prepare(_cfg(depth_fit="sequential", synthesis="ridge_preserve", rho_g=10.0, preserve_tokens=6), setup)
    rows = [r for r in pos.diagnostics if "generic_change" in r]
    assert len(rows) == 4 and all(r["generic_change"] <= r["generic_change_ridge"] + 1e-9 for r in rows)
    upd = _prepare(_cfg(depth_fit="sequential", synthesis="ridge_preserve", rho_g=10.0, preserve_tokens=6,
                        preserve_inputs="updated"), setup)
    w0, w3 = (f"visual.transformer.resblocks.{j}.mlp.c_proj.weight" for j in (0, 3))
    torch.testing.assert_close(upd.unit_task_vector[w0], pos.unit_task_vector[w0], rtol=1e-5, atol=1e-6)  # no upstream
    assert not torch.allclose(upd.unit_task_vector[w3], pos.unit_task_vector[w3], atol=1e-7)


def test_s3_independent_gain_end_to_end_and_default_is_untouched():
    _cfg, _prepare, _setup = _method_setup()
    setup = _setup(2, 4)
    ref = _prepare(_cfg(), setup)
    same = _prepare(_cfg(gain_calibration="none"), setup)
    for k, v in ref.unit_task_vector.items():
        assert torch.equal(same.unit_task_vector[k], v)
    ins = _prepare(_cfg(gain_calibration="lsq", gain_holdout_batches=0), setup)
    hold = _prepare(_cfg(gain_calibration="lsq", gain_holdout_batches=1), setup)
    for out, rows_kind in ((ins, "in_sample"), (hold, "holdout")):
        assert all(r["gain_rows"] == rows_kind and 0.0 <= r["gain"] <= 2.0 for r in out.diagnostics)
        assert all("ridge" in r or "residual_norm_after" in r for r in out.diagnostics)  # the ridge's own rows kept
    g = {int(r["position"]): r["gain"] for r in ins.diagnostics}
    w = "visual.transformer.resblocks.1.mlp.c_proj.weight"
    torch.testing.assert_close(ins.unit_task_vector[w], g[1] * ref.unit_task_vector[w], rtol=1e-4, atol=1e-6)


def test_expert_cproj_restriction_keeps_only_the_cproj_deltas():
    from merge_and_rebase.eval.diagnostics.expert_cproj_restriction import cproj_restricted

    base = {"visual.transformer.resblocks.0.mlp.c_proj.weight": torch.zeros(2, 2),
            "visual.transformer.resblocks.0.mlp.c_proj.bias": torch.zeros(2),
            "visual.transformer.resblocks.0.attn.out_proj.weight": torch.zeros(2, 2),
            "visual.transformer.resblocks.0.mlp.c_fc.weight": torch.zeros(2, 2)}
    expert = {k: torch.ones_like(v) for k, v in base.items()}
    out = cproj_restricted(base, expert)
    assert {k: float(v.sum()) for k, v in out.items() if "c_proj" in k} == {
        "visual.transformer.resblocks.0.mlp.c_proj.weight": 4.0, "visual.transformer.resblocks.0.mlp.c_proj.bias": 2.0}
    assert all(float(v.sum()) == 0.0 for k, v in out.items() if "c_proj" not in k)
    with pytest.raises(KeyError):
        cproj_restricted(base, {"visual.transformer.resblocks.0.attn.out_proj.weight": torch.ones(2, 2)})
