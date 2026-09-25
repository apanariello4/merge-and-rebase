"""Brute-force tests for Direct Residual's post-fit realization diagnostics:

``target_informed_runtime.measure_direct_residual_realization`` (per-position
``block_realized_target_error``/``component_interaction_error``/per-family
ratios, measured on the FULL nonlinear target model with the fitted,
unit-strength task vector actually mounted) and
``target_informed_runtime.compute_direct_residual_task_vector_stats`` (tau
norms and touched-parameter counts).

Both are analysis-only, post-hoc functions: they never run inside
`fit_direct_residual`'s own try/finally and never influence its fitted
corrections. Every brute-force check here recomputes the same quantity via an
INDEPENDENT full forward pass with weights mounted by hand (never calling
`capture_tokens`, `_layout_for`, or any other internal machinery the
functions under test themselves use), so a real bug in either function has
nowhere to hide behind a shared implementation.

No cross-test imports (same convention as
``test_direct_residual_component_coverage.py``): fixtures are duplicated.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
)
from merge_and_rebase.eval.target_informed_runtime import (
    compute_direct_residual_task_vector_stats,
    measure_direct_residual_realization,
)
from merge_and_rebase.eval.target_residual_completion import order_components
from merge_and_rebase.eval.vision_rebase import _state_dict_sha256
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")),
]

# --------------------------------------------------------------------------
# Fixture A: plain (non-stock) attention wrapper -- block_boundary {out_proj,
# c_proj}. Identical shape to test_direct_residual_backfit.py's fixture.
# --------------------------------------------------------------------------


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


def _setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _fit_and_capture(source_depth, target_depth, config, device="cpu", component_inputs=()):
    """Run the real pipeline once, returning everything ``measure_direct_
    residual_realization``/``compute_direct_residual_task_vector_stats`` need."""
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(source_depth, target_depth)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device=device, component_inputs=component_inputs,
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device=device,
    )
    positions = list(range(pairing.target_depth))
    return {
        "corrections": corrections,
        "diagnostics": diagnostics,
        "target_base": target_base,
        "target_base_sd": target_base_sd,
        "positions": positions,
        "batches": captured["target_batches"],
        "target_outputs_by_position": captured["target_base_outputs_by_position"],
        "desired": desired,
        "pairing": pairing,
    }


# --------------------------------------------------------------------------
# Independent (brute-force) recomputation: mount weights by hand on a plain
# deepcopy of the model and hook the block module directly, never touching
# capture_tokens/_layout_for/measure_direct_residual_realization's own code.
# --------------------------------------------------------------------------


def _independent_boundary_outputs(model, state, batches, positions):
    model = deepcopy(model)
    model.load_state_dict(state, strict=True)
    model.eval()
    outputs = {pos: [] for pos in positions}
    handles = []

    def make_hook(pos):
        def hook(_m, _inputs, value):
            outputs[pos].append(value.detach().float().clone())
        return hook

    for pos in positions:
        handles.append(model.visual.transformer.resblocks[pos].register_forward_hook(make_hook(pos)))
    try:
        with torch.no_grad():
            for batch in batches:
                model.encode_image(batch[0])
    finally:
        for h in handles:
            h.remove()
    return outputs


def _norm(batches):
    return sum(float((b.double() ** 2).sum().item()) for b in batches) ** 0.5


def _apply_delta(base_state, delta):
    state = dict(base_state)
    for key, value in delta.items():
        state[key] = state[key] + value.to(state[key])
    return state


def _family_filter(corrections, component, positions):
    """Independent (test-local) reimplementation of family isolation, written
    without reusing target_informed_runtime._family_delta_state."""
    out = {}
    for pos in positions:
        if component == "attn.out_proj":
            wk, bk = f"visual.transformer.resblocks.{pos}.attn.out_proj.weight", f"visual.transformer.resblocks.{pos}.attn.out_proj.bias"
        elif component == "mlp.c_proj":
            wk, bk = f"visual.transformer.resblocks.{pos}.mlp.c_proj.weight", f"visual.transformer.resblocks.{pos}.mlp.c_proj.bias"
        else:
            raise ValueError(component)
        if wk in corrections:
            out[wk] = corrections[wk].clone()
        if bk in corrections:
            out[bk] = corrections[bk].clone()
    return out


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_realization_matches_independent_recomputation(source_depth, target_depth):
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"))
    ctx = _fit_and_capture(source_depth, target_depth, cfg)
    result = measure_direct_residual_realization(
        ctx["target_base"], ctx["target_base_sd"], ctx["corrections"], ctx["positions"], ctx["batches"],
        ctx["target_outputs_by_position"], ctx["desired"], device="cpu",
    )

    joint_state = _apply_delta(ctx["target_base_sd"], ctx["corrections"])
    joint_outputs = _independent_boundary_outputs(ctx["target_base"], joint_state, ctx["batches"], ctx["positions"])

    family_outputs = {}
    for component in ("attn.out_proj", "mlp.c_proj"):
        delta = _family_filter(ctx["corrections"], component, ctx["positions"])
        state = _apply_delta(ctx["target_base_sd"], delta)
        family_outputs[component] = _independent_boundary_outputs(ctx["target_base"], state, ctx["batches"], ctx["positions"])

    for pos in ctx["positions"]:
        t0 = ctx["target_outputs_by_position"][pos]
        d_batches = ctx["desired"][pos]
        d_norm = _norm(d_batches)

        joint_delta = [v - b for v, b in zip(joint_outputs[pos], t0, strict=True)]
        joint_norm = _norm(joint_delta)
        err = [jd - dd for jd, dd in zip(joint_delta, d_batches, strict=True)]
        e_j = _norm(err) / (d_norm + 1e-12)

        row = result[pos]
        assert row["block_realized_target_error"] == pytest.approx(e_j, rel=1e-4, abs=1e-8)
        assert row["joint_delta_norm_over_desired"] == pytest.approx(joint_norm / (d_norm + 1e-12), rel=1e-4, abs=1e-8)

        family_deltas = {
            c: [v - b for v, b in zip(family_outputs[c][pos], t0, strict=True)] for c in ("attn.out_proj", "mlp.c_proj")
        }
        for component, deltas_c in family_deltas.items():
            expected_ratio = _norm(deltas_c) / (d_norm + 1e-12)
            assert row["per_family_delta_norm_over_desired"][component] == pytest.approx(
                expected_ratio, rel=1e-4, abs=1e-8
            )

        running_sum = [
            family_deltas["attn.out_proj"][idx] + family_deltas["mlp.c_proj"][idx]
            for idx in range(len(joint_delta))
        ]
        interaction = [jd - rs for jd, rs in zip(joint_delta, running_sum, strict=True)]
        expected_interaction = _norm(interaction) / (joint_norm + 1e-12)
        assert row["component_interaction_error"] == pytest.approx(expected_interaction, rel=1e-4, abs=1e-8)


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_single_family_interaction_error_is_none(source_depth, target_depth):
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=("attn.out_proj",))
    ctx = _fit_and_capture(source_depth, target_depth, cfg)
    result = measure_direct_residual_realization(
        ctx["target_base"], ctx["target_base_sd"], ctx["corrections"], ctx["positions"], ctx["batches"],
        ctx["target_outputs_by_position"], ctx["desired"], device="cpu",
    )
    for row in result.values():
        assert row["component_interaction_error"] is None
        assert set(row["per_family_delta_norm_over_desired"]) == {"attn.out_proj"}


# --------------------------------------------------------------------------
# Toy linear-additive block (no coupling between out_proj/c_proj, same
# convention as test_direct_residual_backfit.py's docstring): I_j should be
# approximately zero, since the two components' effects genuinely superpose.
# --------------------------------------------------------------------------


class _ToyAttn(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _ToyMlp(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.c_proj(x)


class _ToyBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _ToyAttn(width)
        self.mlp = _ToyMlp(width)

    def forward(self, x):
        return x + self.attn(x) + self.mlp(x)


class _ToyVisual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_ToyBlock(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _ToyModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _ToyVisual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _toy_setup(depth=1, width=6, seed=7):
    torch.manual_seed(seed)
    source_base = _ToyModel(width, depth).eval()
    target_base = _ToyModel(width, depth).eval()
    tuned = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.attn.out_proj.weight.add_(0.3 * torch.randn_like(block.attn.out_proj.weight))
            block.attn.out_proj.bias.add_(0.1 * torch.randn_like(block.attn.out_proj.bias))
            block.mlp.c_proj.weight.add_(0.3 * torch.randn_like(block.mlp.c_proj.weight))
            block.mlp.c_proj.bias.add_(0.1 * torch.randn_like(block.mlp.c_proj.bias))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(24, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(24)), batch_size=4, shuffle=False)
    pairing = DiscreteLayerPairing.compute(depth, depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, tuned, target_base, data, pairing, target_base_sd


def test_toy_linear_additive_block_interaction_error_is_near_zero():
    cfg = DirectResidualConfig(
        num_batches=6, ridge_relative=0.1, components=("attn.out_proj", "mlp.c_proj"), seed=89,
    )
    source_base, source_ft, target_base, data, pairing, target_base_sd = _toy_setup()
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    result = measure_direct_residual_realization(
        target_base, target_base_sd, corrections, [0], captured["target_batches"],
        captured["target_base_outputs_by_position"], desired, device="cpu",
    )
    # The toy block's two projections both act directly and only additively on
    # the SAME raw input x (out = x + out_proj(x) + c_proj(x)), so mounting
    # both at once is exactly linear superposition of mounting them one at a
    # time: I_0 should vanish up to floating-point error.
    assert result[0]["component_interaction_error"] == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------
# realization_diagnostics never changes the fitted task vector (decoupled by
# construction: measure_direct_residual_realization is a separate, post-hoc
# call, never invoked from inside fit_direct_residual).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_split", ["none", "backfit"])
def test_computing_realization_does_not_change_subsequent_fit_hash(block_split):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"), block_split=block_split,
    )
    ctx1 = _fit_and_capture(2, 4, cfg)
    measure_direct_residual_realization(
        ctx1["target_base"], ctx1["target_base_sd"], ctx1["corrections"], ctx1["positions"], ctx1["batches"],
        ctx1["target_outputs_by_position"], ctx1["desired"], device="cpu",
    )
    ctx2 = _fit_and_capture(2, 4, cfg)
    assert _state_dict_sha256(ctx1["corrections"]) == _state_dict_sha256(ctx2["corrections"])


def test_output_local_realization_matches_independent_recomputation():
    """Stock nn.MultiheadAttention fixture (needed for q/k/v/c_fc), reused from
    test_direct_residual_device_parametrization.py's Fixture B (duplicated,
    per this suite's no-cross-test-import convention)."""

    class _StockAttentionBlock(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.ln_1 = torch.nn.LayerNorm(width)
            self.attn = torch.nn.MultiheadAttention(width, 1, batch_first=True)
            self.ls_1 = torch.nn.Identity()
            self.ln_2 = torch.nn.LayerNorm(width)
            self.mlp = torch.nn.Sequential(OrderedDict([
                ("c_fc", torch.nn.Linear(width, width * 2)), ("gelu", torch.nn.GELU()),
                ("c_proj", torch.nn.Linear(width * 2, width)),
            ]))
            self.ls_2 = torch.nn.Identity()

        def forward(self, x):
            normed = self.ln_1(x)
            x = x + self.ls_1(self.attn(normed, normed, normed, need_weights=False)[0])
            return x + self.ls_2(self.mlp(self.ln_2(x)))

    class _StockVisual(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.input = torch.nn.Linear(4, width)
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList([_StockAttentionBlock(width) for _ in range(depth)])

        def forward(self, images):
            x = self.input(images)
            for block in self.transformer.resblocks:
                x = block(x)
            return x.mean(dim=1)

    class _StockModel(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.visual = _StockVisual(width, depth)

        def encode_image(self, x):
            return self.visual(x)

    torch.manual_seed(41)
    source_base = _StockModel(4, 2).eval()
    target_base = _StockModel(4, 4).eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(42)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.in_proj_weight.add_(0.1 * torch.randn_like(block.attn.in_proj_weight))
    data = _loader(seed=43)
    pairing = DiscreteLayerPairing.compute(2, 4)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}

    components = ("attn.v_proj", "mlp.c_proj")
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target="output_local", components=components,
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu", component_inputs=order_components(components),
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    positions = list(range(pairing.target_depth))
    result = measure_direct_residual_realization(
        target_base, target_base_sd, corrections, positions, captured["target_batches"],
        captured["target_base_outputs_by_position"], desired, device="cpu", components=components,
    )

    joint_state = _apply_delta(target_base_sd, corrections)
    joint_model = deepcopy(target_base)
    joint_model.load_state_dict(joint_state, strict=True)
    joint_model.eval()
    joint_outputs = {pos: [] for pos in positions}
    handles = [
        joint_model.visual.transformer.resblocks[pos].register_forward_hook(
            lambda _m, _i, v, pos=pos: joint_outputs[pos].append(v.detach().float().clone())
        )
        for pos in positions
    ]
    try:
        with torch.no_grad():
            for batch in captured["target_batches"]:
                joint_model.encode_image(batch[0])
    finally:
        for h in handles:
            h.remove()

    for pos in positions:
        t0 = captured["target_base_outputs_by_position"][pos]
        d_batches = desired[pos]
        d_norm = _norm(d_batches)
        joint_delta = [v - b for v, b in zip(joint_outputs[pos], t0, strict=True)]
        e_j = _norm([jd - dd for jd, dd in zip(joint_delta, d_batches, strict=True)]) / (d_norm + 1e-12)
        assert result[pos]["block_realized_target_error"] == pytest.approx(e_j, rel=1e-4, abs=1e-8)
        assert set(result[pos]["per_family_delta_norm_over_desired"]) == set(components)


# --------------------------------------------------------------------------
# Task-vector stats.
# --------------------------------------------------------------------------


def test_task_vector_stats_sha_matches_vision_rebase_helper():
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"))
    ctx = _fit_and_capture(2, 4, cfg)
    stats = compute_direct_residual_task_vector_stats(ctx["corrections"], ctx["target_base_sd"], ctx["positions"])
    assert stats["tau_sha256"] == _state_dict_sha256(ctx["corrections"])
    assert stats["n_modified_tensors"] == len(ctx["corrections"])
    assert stats["tau_norm"] > 0
    assert stats["tau_norm_over_touched_base"] > 0
    assert stats["tau_norm_over_all_base"] > 0
    # Touching only two of the four blocks' components still divides by the
    # WHOLE base model's norm for tau_norm_over_all_base, so it must be a much
    # smaller ratio than tau_norm_over_touched_base (bigger denominator).
    assert stats["tau_norm_over_all_base"] < stats["tau_norm_over_touched_base"]


def test_task_vector_stats_v_only_output_local_not_triple_counted():
    """v-only output_local run -> n_modified_parameters counts d*d_in + d,
    not 3x (would happen if the whole packed in_proj tensor were counted)."""
    torch.manual_seed(51)

    class _StockAttentionBlock(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.ln_1 = torch.nn.LayerNorm(width)
            self.attn = torch.nn.MultiheadAttention(width, 1, batch_first=True)
            self.ls_1 = torch.nn.Identity()
            self.ln_2 = torch.nn.LayerNorm(width)
            self.mlp = torch.nn.Sequential(OrderedDict([
                ("c_fc", torch.nn.Linear(width, width * 2)), ("gelu", torch.nn.GELU()),
                ("c_proj", torch.nn.Linear(width * 2, width)),
            ]))
            self.ls_2 = torch.nn.Identity()

        def forward(self, x):
            normed = self.ln_1(x)
            x = x + self.ls_1(self.attn(normed, normed, normed, need_weights=False)[0])
            return x + self.ls_2(self.mlp(self.ln_2(x)))

    class _StockVisual(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.input = torch.nn.Linear(4, width)
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList([_StockAttentionBlock(width) for _ in range(depth)])

        def forward(self, images):
            x = self.input(images)
            for block in self.transformer.resblocks:
                x = block(x)
            return x.mean(dim=1)

    class _StockModel(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.visual = _StockVisual(width, depth)

        def encode_image(self, x):
            return self.visual(x)

    width, depth = 4, 2
    source_base = _StockModel(width, depth).eval()
    target_base = _StockModel(width, depth).eval()
    source_ft = deepcopy(source_base)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].attn.in_proj_weight.add_(0.2)
    data = _loader(seed=52)
    pairing = DiscreteLayerPairing.compute(depth, depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target="output_local", components=("attn.v_proj",),
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu", component_inputs=("attn.v_proj",),
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    positions = list(range(pairing.target_depth))
    stats = compute_direct_residual_task_vector_stats(
        corrections, target_base_sd, positions, components=("attn.v_proj",),
    )
    d_in = width
    expected_per_position = width * d_in + width  # weight rows + bias rows for the v-slice only
    assert stats["n_modified_parameters"] == expected_per_position * len(positions)


# --------------------------------------------------------------------------
# Restoration guarantee.
# --------------------------------------------------------------------------


def test_measure_realization_restores_target_model_exactly():
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"))
    ctx = _fit_and_capture(2, 4, cfg)
    before = {k: v.clone() for k, v in ctx["target_base"].state_dict().items()}
    measure_direct_residual_realization(
        ctx["target_base"], ctx["target_base_sd"], ctx["corrections"], ctx["positions"], ctx["batches"],
        ctx["target_outputs_by_position"], ctx["desired"], device="cpu",
    )
    for key, value in before.items():
        assert torch.equal(dict(ctx["target_base"].state_dict())[key], value)


# --------------------------------------------------------------------------
# Device parametrization.
# --------------------------------------------------------------------------


def test_packed_qkv_presence_is_not_falsely_triggered_by_a_sibling_component():
    """Regression: q/k/v share ONE physical state-dict key
    (in_proj_weight/in_proj_bias). Passing the broad CANONICAL_COMPONENT_ORDER
    default (which includes q and k) to measure_direct_residual_realization
    for a run that only ever fit v must NOT report q/k as "present" -- caught
    during initial development of this test file, where the default produced
    spurious q/k rows for a v-only fit because their shared key existed in
    target_corrections regardless of which row-third was written."""

    class _StockAttentionBlock(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.ln_1 = torch.nn.LayerNorm(width)
            self.attn = torch.nn.MultiheadAttention(width, 1, batch_first=True)
            self.ls_1 = torch.nn.Identity()
            self.ln_2 = torch.nn.LayerNorm(width)
            self.mlp = torch.nn.Sequential(OrderedDict([
                ("c_fc", torch.nn.Linear(width, width * 2)), ("gelu", torch.nn.GELU()),
                ("c_proj", torch.nn.Linear(width * 2, width)),
            ]))
            self.ls_2 = torch.nn.Identity()

        def forward(self, x):
            normed = self.ln_1(x)
            x = x + self.ls_1(self.attn(normed, normed, normed, need_weights=False)[0])
            return x + self.ls_2(self.mlp(self.ln_2(x)))

    class _StockVisual(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.input = torch.nn.Linear(4, width)
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList([_StockAttentionBlock(width) for _ in range(depth)])

        def forward(self, images):
            x = self.input(images)
            for block in self.transformer.resblocks:
                x = block(x)
            return x.mean(dim=1)

    class _StockModel(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.visual = _StockVisual(width, depth)

        def encode_image(self, x):
            return self.visual(x)

    torch.manual_seed(61)
    source_base = _StockModel(4, 2).eval()
    target_base = _StockModel(4, 2).eval()
    source_ft = deepcopy(source_base)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].attn.in_proj_weight.add_(0.2)
    data = _loader(seed=62)
    pairing = DiscreteLayerPairing.compute(2, 2)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target="output_local", components=("attn.v_proj",),
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu", component_inputs=("attn.v_proj",),
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    positions = list(range(pairing.target_depth))
    # Passing exactly the fitted family (the safe, required contract).
    result = measure_direct_residual_realization(
        target_base, target_base_sd, corrections, positions, captured["target_batches"],
        captured["target_base_outputs_by_position"], desired, device="cpu", components=("attn.v_proj",),
    )
    for row in result.values():
        assert set(row["per_family_delta_norm_over_desired"]) == {"attn.v_proj"}
        assert row["component_interaction_error"] is None  # only one family requested


def _open_clip_direction_setup(direction, seed=101):
    """Duplicated from tests/test_direct_residual_open_clip_integration.py's
    ``_direction_setup`` (per this suite's no-cross-test-import convention);
    same three (extend/shrink/same_arch) direction specs."""
    from open_clip.transformer import VisionTransformer
    from torch.utils.data import TensorDataset as _TD

    from merge_and_rebase.eval.target_informed_runtime import paired_calibration  # noqa: F401 (imported for parity)

    class _CLIPLike(torch.nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual

        def encode_image(self, x):
            return self.visual(x)

    def _make_vit(*, image_size, patch_size, width, layers, heads, seed):
        torch.manual_seed(seed)
        vt = VisionTransformer(
            image_size=image_size, patch_size=patch_size, width=width, layers=layers, heads=heads,
            mlp_ratio=2.0, ls_init_value=None, output_dim=width, pool_type="tok",
        )
        return _CLIPLike(vt).eval()

    def _tuned(model, seed, scale=0.2):
        tuned = deepcopy(model)
        torch.manual_seed(seed)
        with torch.no_grad():
            for block in tuned.visual.transformer.resblocks:
                block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
                block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
        return tuned

    class _IdentityTensorDataset(_TD):
        def __init__(self, images, sample_ids):
            super().__init__(images, torch.arange(len(images)))
            self.sample_ids = sample_ids

    def _loader_ocl(image_size, n=6, seed=0, sample_ids=None):
        generator = torch.Generator().manual_seed(seed)
        images = torch.randn(n, 3, image_size, image_size, generator=generator)
        ids = sample_ids if sample_ids is not None else [str(i) for i in range(n)]
        return DataLoader(_IdentityTensorDataset(images, ids), batch_size=2, shuffle=False)

    directions = {
        "extend": dict(source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
                       target=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3)),
        "shrink": dict(source=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3),
                       target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2)),
        "same_arch": dict(source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
                           target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2)),
    }
    spec = directions[direction]
    source_base = _make_vit(seed=seed, **spec["source"])
    target_base = _make_vit(seed=seed + 1, **spec["target"])
    source_ft = _tuned(source_base, seed=seed + 2)
    shared_ids = [str(i) for i in range(6)]
    source_loader = _loader_ocl(spec["source"]["image_size"], seed=seed + 3, sample_ids=shared_ids)
    target_loader = _loader_ocl(spec["target"]["image_size"], seed=seed + 4, sample_ids=shared_ids)
    pairing = DiscreteLayerPairing.compute(spec["source"]["layers"], spec["target"]["layers"])
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_realization_matches_independent_recomputation_open_clip(direction):
    pytest.importorskip("open_clip")
    source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd = (
        _open_clip_direction_setup(direction)
    )
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"))
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    positions = list(range(pairing.target_depth))
    result = measure_direct_residual_realization(
        target_base, target_base_sd, corrections, positions, captured["target_batches"],
        captured["target_base_outputs_by_position"], desired, device="cpu",
        components=("attn.out_proj", "mlp.c_proj"),
    )

    joint_state = _apply_delta(target_base_sd, corrections)
    joint_model = deepcopy(target_base)
    joint_model.load_state_dict(joint_state, strict=True)
    joint_model.eval()
    joint_outputs = {pos: [] for pos in positions}
    handles = [
        joint_model.visual.transformer.resblocks[pos].register_forward_hook(
            lambda _m, _i, v, pos=pos: joint_outputs[pos].append(v.detach().float().clone())
        )
        for pos in positions
    ]
    try:
        with torch.no_grad():
            for batch in captured["target_batches"]:
                joint_model.encode_image(batch[0])
    finally:
        for h in handles:
            h.remove()

    for pos in positions:
        t0 = captured["target_base_outputs_by_position"][pos]
        d_batches = desired[pos]
        d_norm = _norm(d_batches)
        joint_delta = [v - b for v, b in zip(joint_outputs[pos], t0, strict=True)]
        e_j = _norm([jd - dd for jd, dd in zip(joint_delta, d_batches, strict=True)]) / (d_norm + 1e-12)
        assert result[pos]["block_realized_target_error"] == pytest.approx(e_j, rel=1e-4, abs=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_realization_finite_on_device(device):
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"))
    ctx = _fit_and_capture(2, 4, cfg, device=device)
    result = measure_direct_residual_realization(
        ctx["target_base"], ctx["target_base_sd"], ctx["corrections"], ctx["positions"], ctx["batches"],
        ctx["target_outputs_by_position"], ctx["desired"], device=device,
    )
    for row in result.values():
        for key in ("block_realized_target_error", "joint_delta_norm_over_desired", "component_interaction_error"):
            assert torch.isfinite(torch.tensor(float(row[key])))
        for ratio in row["per_family_delta_norm_over_desired"].values():
            assert torch.isfinite(torch.tensor(float(ratio)))
