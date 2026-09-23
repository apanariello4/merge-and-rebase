"""Tests for Direct Residual's ``component_target='output_local'`` mode.

Direct Residual (``src/merge_and_rebase/eval/direct_residual.py``) answers the
same transport-free local-effect question ARIADNE's Proposal 1 ``direct_target``
mode does, but pairs an arbitrary source depth with an arbitrary target depth
via the flat ``DiscreteLayerPairing`` -- so it, not the P1 path, is where this
ablation actually lives (P1's ``ResidualCompletionConfig`` and
``complete_residuals_direct`` are unmodified by this feature; see the golden
hashes below and in ``tests/test_direct_target_independent_fast_path.py`` /
``tests/test_direct_p1_trajectory_and_components.py``).

``component_target='block_boundary'`` (default) fits every requested
component against the SAME block-boundary target, exactly as before this
ablation. ``'output_local'`` gives each component its own target built only
from its own (local) weight change, with a span-aware set of weighted source
contributions per target position -- ``position_source_contributions`` -- that
is the same code for extend, shrink and same-arch:

* extend/same-arch: one contribution, the paired source block, weighted
  ``1/m_i`` (``m_i`` = how many target positions share source block ``i``).
* shrink: every source block whose discrete "reverse" pairing collapses onto
  this target position, each at full weight 1 (residual writers only;
  internal components always use only the forward-paired block at weight 1).

Per CLAUDE.md's determinism convention, the null control (``component_target``
omitted/``'block_boundary'``) is pinned at sha256-hash level against golden
values recorded at HEAD ``d77bbce`` via a ``git worktree`` (see the
implementation report for the exact commands), reusing the fixture of
``tests/test_direct_residual_fit.py``. This module has no cross-test imports.
"""

from __future__ import annotations

import hashlib
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
    parse_direct_residual_config,
    position_source_contributions,
)
from merge_and_rebase.eval.target_residual_completion import order_components
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.utils.cost_accounting import PhaseCostRecorder, recording

# --------------------------------------------------------------------------
# Golden hashes for the untouched block_boundary path, recorded at HEAD
# d77bbce (clean tree, via `git worktree add ... d77bbce`) on the fixture of
# test_direct_residual_fit.py, BEFORE any of this ablation's code was written.
# --------------------------------------------------------------------------
_GOLDEN_EXTEND = "e6f02b6922df921f802cb02ca245a60f6f988d44a70a2425061c7cd10f2e6dbc"
_GOLDEN_SHRINK = "111eafdb947b940ebee0366366c5d726830ff6661d8f78222f0a1c67f13dfe94"
_GOLDEN_SAME_ARCH = "3b0600d6b14887867dc83d8d026903dea27bf0de20baaaa5055848fb0385c9d3"


def _state_dict_sha256(d: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Shared synthetic fixture (copied from test_direct_residual_fit.py -- this
# module has no cross-test imports).
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


def _fit(source_depth, target_depth, config=None, **setup_kwargs):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(
        source_depth, target_depth, **setup_kwargs
    )
    config = config or DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    )
    return corrections, diagnostics


# --------------------------------------------------------------------------
# 1. Golden-hash pinning: the block_boundary path is untouched.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth, golden", [
    (2, 4, _GOLDEN_EXTEND), (4, 2, _GOLDEN_SHRINK), (3, 3, _GOLDEN_SAME_ARCH),
])
def test_golden_hash_block_boundary(source_depth, target_depth, golden):
    corrections, _diag = _fit(source_depth, target_depth)
    assert _state_dict_sha256(corrections) == golden


@pytest.mark.parametrize("source_depth, target_depth, golden", [
    (2, 4, _GOLDEN_EXTEND), (4, 2, _GOLDEN_SHRINK), (3, 3, _GOLDEN_SAME_ARCH),
])
def test_golden_hash_block_boundary_under_cost_recording(source_depth, target_depth, golden):
    # Cost accounting only synchronizes and reads counters: the pinned default
    # path is bit-identical with an active recorder.
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder):
        corrections, _diag = _fit(source_depth, target_depth)
    assert _state_dict_sha256(corrections) == golden
    phases = recorder.summary()["phases"]
    assert phases["activation_collection"]["segments"] > 0
    assert phases["transformation"]["segments"] > 0


def test_component_target_defaults_to_block_boundary():
    cfg = DirectResidualConfig()
    assert cfg.component_target == "block_boundary"
    assert cfg.realization_diagnostics is False
    parsed = parse_direct_residual_config(None)
    assert parsed.component_target == "block_boundary"


# --------------------------------------------------------------------------
# 2. position_source_contributions: coverage/partition and weight rules.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", [(12, 24), (12, 12)])
def test_extend_and_same_arch_weights_sum_to_one_per_source_block(source_depth, target_depth):
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    contributions = position_source_contributions(pairing)
    assert set(contributions) == set(range(target_depth))
    totals: dict[int, float] = {}
    for j, terms in contributions.items():
        assert len(terms) == 1, "extend/same-arch positions have exactly one contribution"
        source_idx, weight = terms[0]
        assert source_idx == pairing.pairing[j]
        totals[source_idx] = totals.get(source_idx, 0.0) + weight
    for source_idx, total in totals.items():
        assert total == pytest.approx(1.0), source_idx


def test_same_arch_reduces_to_plain_single_block_weight_one():
    pairing = DiscreteLayerPairing.compute(8, 8)
    contributions = position_source_contributions(pairing)
    for j in range(8):
        assert contributions[j] == [(j, 1.0)]


def test_shrink_partition_covers_every_source_block_exactly_once():
    pairing = DiscreteLayerPairing.compute(24, 12)
    contributions = position_source_contributions(pairing)
    assert set(contributions) == set(range(12))
    covered = sorted(i for terms in contributions.values() for i, _w in terms)
    assert covered == list(range(24))
    for j, terms in contributions.items():
        for _i, weight in terms:
            assert weight == 1.0
        assert pairing.pairing[j] in {i for i, _w in terms}


def test_shrink_12_to_12_is_same_arch_shaped():
    """source_depth == target_depth takes the extend/same-arch branch
    (target_depth >= source_depth), not the shrink span-partition branch."""
    pairing = DiscreteLayerPairing.compute(12, 12)
    contributions = position_source_contributions(pairing)
    for j in range(12):
        assert contributions[j] == [(j, 1.0)]


# --------------------------------------------------------------------------
# 3. Parser coverage.
# --------------------------------------------------------------------------

_ALL_SIX = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj")
_SIX_ARMS = [
    ("mlp.c_proj",),
    ("attn.out_proj",),
    ("attn.out_proj", "mlp.c_proj"),
    ("attn.v_proj", "attn.out_proj", "mlp.c_proj"),
    ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.c_proj"),
    _ALL_SIX,
]


@pytest.mark.parametrize("components", _SIX_ARMS)
def test_all_six_component_sets_parse_in_output_local(components):
    cfg = parse_direct_residual_config(
        {"component_target": "output_local", "components": list(components)}
    )
    assert order_components(cfg.components) == tuple(c for c in _ALL_SIX if c in components)


@pytest.mark.parametrize("components", _SIX_ARMS)
def test_all_six_component_sets_parse_in_output_total(components):
    cfg = parse_direct_residual_config(
        {"component_target": "output_total", "components": list(components)}
    )
    assert order_components(cfg.components) == tuple(c for c in _ALL_SIX if c in components)


def test_output_total_rejects_block_split_backfit():
    with pytest.raises(ValueError, match="block_split='backfit' requires component_target"):
        parse_direct_residual_config({"component_target": "output_total", "block_split": "backfit"})


def test_internal_components_rejected_for_block_boundary():
    with pytest.raises(ValueError, match="unknown components"):
        parse_direct_residual_config({"components": ["attn.q_proj", "mlp.c_proj"]})


def test_out_proj_alone_is_allowed_for_block_boundary():
    """Direct Residual is always independent (no cascade), so the P1-era
    "c_proj anchor" restriction does not apply here; DT-O is a legal arm."""
    cfg = parse_direct_residual_config({"components": ["attn.out_proj"]})
    assert cfg.components == ("attn.out_proj",)


def test_c_proj_alone_is_still_allowed_for_block_boundary():
    cfg = parse_direct_residual_config({"components": ["mlp.c_proj"]})
    assert cfg.components == ("mlp.c_proj",)


def test_output_local_out_proj_alone_is_allowed():
    cfg = parse_direct_residual_config({"component_target": "output_local", "components": ["attn.out_proj"]})
    assert cfg.components == ("attn.out_proj",)


# --------------------------------------------------------------------------
# 4. End-to-end fit against a stock nn.MultiheadAttention target (the real
#    OpenCLIP shape), across extend / shrink / same_arch.
# --------------------------------------------------------------------------


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


def _stock_loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _stock_setup(source_depth, target_depth, width=4, seed=41):
    torch.manual_seed(seed)
    source_base = _StockModel(width, source_depth).eval()
    target_base = _StockModel(width, target_depth).eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
            block.mlp.c_fc.weight.add_(0.15 * torch.randn_like(block.mlp.c_fc.weight))
            block.attn.out_proj.weight.add_(0.2 * torch.randn_like(block.attn.out_proj.weight))
            block.attn.in_proj_weight.add_(0.1 * torch.randn_like(block.attn.in_proj_weight))
    data = _stock_loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _stock_output_local_fit(source_depth, target_depth, components, **overrides):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _stock_setup(source_depth, target_depth)
    params = dict(num_batches=3, ridge_relative=0.05, component_target="output_local", components=tuple(components))
    params.update(overrides)
    config = DirectResidualConfig(**params)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
        component_inputs=order_components(config.components),
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    )
    return corrections, diagnostics, target_base, target_base_sd


def test_output_local_realization_diagnostics_flag_does_not_change_fitted_corrections():
    """Part 3: realization_diagnostics is analysis-only for output_local too
    (already had the diagnostic fields before this ablation; this pins that
    toggling the flag never perturbs the task vector itself)."""
    corrections_off, _diag_off, _m1, _sd1 = _stock_output_local_fit(2, 4, _ALL_SIX, realization_diagnostics=False)
    corrections_on, _diag_on, _m2, _sd2 = _stock_output_local_fit(2, 4, _ALL_SIX, realization_diagnostics=True)
    assert _state_dict_sha256(corrections_off) == _state_dict_sha256(corrections_on)


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_output_local_end_to_end_all_six_components_finite(source_depth, target_depth):
    corrections, diagnostics, target_base, _sd = _stock_output_local_fit(source_depth, target_depth, _ALL_SIX)
    assert corrections
    for key, value in corrections.items():
        assert torch.isfinite(value).all(), key
    assert {row["component"] for row in diagnostics} == set(_ALL_SIX)
    for row in diagnostics:
        assert row["component_target"] == "output_local"
        assert "source_contributions" in row
    in_proj_keys = [k for k in corrections if k.endswith("attn.in_proj_weight")]
    assert in_proj_keys


@pytest.mark.parametrize("component", ["attn.q_proj", "attn.k_proj", "attn.v_proj"])
def test_packed_qkv_slice_independence_at_hash_level(component):
    """A component's own row slice of the joint in_proj_weight/bias
    correction is bitwise equal to the slice written by that component's
    solo run (extend regime, where packed q/k/v share weight 1/m across
    duplicated target positions but the row-slice writes stay independent)."""
    slice_index = {"attn.q_proj": 0, "attn.k_proj": 1, "attn.v_proj": 2}[component]
    joint_corrections, _diag, _model, _sd = _stock_output_local_fit(2, 4, _ALL_SIX)
    solo_corrections, _diag2, _model2, _sd2 = _stock_output_local_fit(2, 4, (component,))
    for key, value in solo_corrections.items():
        is_bias = key.endswith("in_proj_bias")
        d = value.shape[0] // 3
        rows = slice(slice_index * d, (slice_index + 1) * d)
        solo_slice = value[rows] if is_bias else value[rows, :]
        joint_slice = joint_corrections[key][rows] if is_bias else joint_corrections[key][rows, :]
        torch.testing.assert_close(joint_slice, solo_slice, rtol=0, atol=0)


def test_shrink_residual_writer_target_is_sum_of_per_block_terms():
    """Brute force: a shrink position's mlp.c_proj correction must equal the
    correction from a single-contribution fit whose "source" is a merged
    model built by summing each span member's own local delta (weight 1
    each) -- i.e. the multi-term D_{j,c} really is a plain sum, checked by
    reproducing the accumulation independently of _fit_component_outputs_
    from_contributions's own internal loop.

    Concretely: depth 4 -> 2 puts source blocks {0,1} in span 0 and {2,3} in
    span 1 (reverse pairing). This test recomputes D_{0,mlp.c_proj} by hand
    from the two per-block terms and compares the fitted correction's
    achieved residual reduction against a ridge-free (ridge_relative tiny)
    fit of that same by-hand target, using the identical target input bank.
    """
    source_depth, target_depth = 4, 2
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    contributions = position_source_contributions(pairing)
    assert contributions[0] == [(0, 1.0), (1, 1.0)]
    assert contributions[1] == [(2, 1.0), (3, 1.0)]

    corrections, diagnostics, _model, _sd = _stock_output_local_fit(
        source_depth, target_depth, ("mlp.c_proj",), ridge_relative=1e-6,
    )
    rows = [r for r in diagnostics if r["position"] == 0 and r["component"] == "mlp.c_proj"]
    assert len(rows) == 1
    row = rows[0]
    assert sorted(row["source_contributions"]) == [(0, 1.0), (1, 1.0)]
    assert len(row["procrustes_ranks"]) == 2
    assert torch.isfinite(corrections["visual.transformer.resblocks.0.mlp.c_proj.weight"]).all()


def test_output_local_ignores_upstream_source_block_weight_changes():
    """output_local's target for an extend position must be unaffected by a
    weight change on a DIFFERENT source block: there is no upstream-induced
    input drift by construction (X^s0 is each contributing block's own,
    unperturbed input)."""
    source_depth, target_depth = 2, 4
    source_base, source_ft_a, target_base, data, pairing, target_base_sd = _stock_setup(source_depth, target_depth)
    source_ft_b = deepcopy(source_ft_a)
    with torch.no_grad():
        source_ft_b.visual.transformer.resblocks[0].attn.out_proj.weight.add_(0.5)

    config = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target="output_local", components=("mlp.c_proj",),
    )

    def _run(source_ft):
        captured = capture_paired_boundary_activations(
            source_base, source_ft, deepcopy(target_base), data, data, pairing,
            num_batches=config.num_batches, seed=config.seed, device="cpu",
            component_inputs=order_components(config.components),
        )
        desired = compute_desired_effects(captured, pairing)
        return fit_direct_residual(
            deepcopy(target_base), target_base_sd, captured, desired, pairing, config=config, device="cpu",
        )

    corrections_a, _ = _run(source_ft_a)
    corrections_b, _ = _run(source_ft_b)
    # pairing.pairing[j] for j in {2,3} is source block 1 (since target_depth
    # 4 = 2*source_depth), never source block 0: unaffected by its change.
    for j in (2, 3):
        if pairing.pairing[j] != 0:
            key = f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"
            torch.testing.assert_close(corrections_a[key], corrections_b[key], rtol=0, atol=0)


# --------------------------------------------------------------------------
# 5. Part 1 review-fix regressions: live-parameter/bank consistency, and the
#    internal-component extend weight (paired-entry weight, not hardcoded 1.0).
# --------------------------------------------------------------------------


def test_output_local_fit_survives_live_parameter_dtype_mismatch_with_banks():
    """Regression for the device/dtype bug in
    ``_fit_component_outputs_from_contributions``: ``target_w``/``target_b``
    used to come straight from the live module (whatever device/dtype it
    happens to be at call time), while every captured bank (``h_batches``,
    source component inputs/weights) is unconditionally forced to CPU
    float32 by ``capture_tokens``/``capture_source_component_references``
    (both call ``.float().cpu()`` regardless of the source model's own
    dtype). A single-device CPU-only CI box cannot exercise the device half
    of that mismatch (no CUDA), so this test builds the entire model +
    calibration pipeline in float64 instead: every forward pass then runs
    consistently in double precision (no unrelated dtype error inside
    ``capture_tokens``' own forward pass), but the CAPTURED banks still come
    back float32 by the cast above, while a not-yet-fixed ``target_w`` read
    straight off the double module would stay float64 -- reproducing,
    without CUDA, the exact operand-provenance bug the fix addresses at the
    ``F.linear(h, target_w, target_b)`` call site.
    """
    source_depth, target_depth = 2, 4
    source_base, source_ft, target_base, data, pairing, target_base_sd = _stock_setup(source_depth, target_depth)
    source_base = source_base.double()
    source_ft = source_ft.double()
    target_base = target_base.double()
    target_base_sd = {k: v.double() for k, v in target_base_sd.items()}
    double_data = DataLoader(
        TensorDataset(data.dataset.tensors[0].double(), data.dataset.tensors[1]), batch_size=2, shuffle=False
    )
    config = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target="output_local", components=("mlp.c_proj",),
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, double_data, double_data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
        component_inputs=order_components(config.components),
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    )
    # No RuntimeError from F.linear's dtype promotion check (the pre-fix
    # symptom): reaching here at all is most of what this test is for.
    assert diagnostics
    for value in corrections.values():
        assert torch.isfinite(value).all()
    # The weight correction is always cast to float32 by ResidualSufficient
    # Statistics.solve(); only the bias is deliberately re-cast to match the
    # target state dict's own dtype (float64 here) so it can be added back in
    # directly.
    weight_keys = [k for k in corrections if k.endswith(".weight")]
    assert weight_keys and all(corrections[k].dtype == torch.float32 for k in weight_keys)
    # The live model itself must be untouched by the fit (fit_direct_residual
    # restores original_state in its `finally`), so it stays float64.
    assert next(target_base.parameters()).dtype == torch.float64


def test_internal_component_extend_weight_matches_contribution_not_hardcoded_one():
    """12->24 extend: every source block is realized by exactly two target
    positions (m_i=2 for all i), so ``position_source_contributions`` gives
    each of them weight 0.5. An internal component (q/k/v/c_fc) at such a
    position must be fit against a target scaled by that same 0.5 -- not by a
    hardcoded 1.0, which would double the desired effect relative to the
    residual-writing components fit at the same position.
    """
    source_depth, target_depth = 12, 24
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    contributions = position_source_contributions(pairing)
    # Sanity: this is exactly the "m_i=2 for every source block" extend case
    # the plan's item 2 names.
    for j in range(target_depth):
        i, w = contributions[j][0]
        assert i == pairing.pairing[j]
        assert w == pytest.approx(0.5)

    corrections, diagnostics, _model, _sd = _stock_output_local_fit(
        source_depth, target_depth, ("attn.v_proj",), ridge_relative=1e-6,
    )
    rows = {row["position"]: row for row in diagnostics if row["component"] == "attn.v_proj"}
    assert len(rows) == target_depth
    for j, row in rows.items():
        assert row["source_contributions"] == [(pairing.pairing[j], pytest.approx(0.5))]
