"""Tests for Direct Residual's ``residual_target='transported_endpoint'`` variant.

Today Direct Residual builds every position's target as the transported delta
``D_j^Delta = (S_{j,1} - S_{j,0}) Q_j`` (``compute_desired_effects``'s default
``residual_target="transported_delta"``, unchanged and golden-hash pinned).
This module adds the affine "transported endpoint" alternative ``D_j^end =
(S_{j,1} - mu_s) Q_j + mu_t - T_j^0``, which agrees with the delta target
exactly when ``E_j := (S_{j,0} - mu_s) Q_j - (T_j^0 - mu_t)`` -- the residual
the centered Procrustes fit ``Q_j`` minimizes -- is zero (see
``compute_alignment_diagnostics``'s docstring in
``src/merge_and_rebase/eval/direct_residual.py``). That diagnostics function
is deliberately separate from ``compute_desired_effects`` (not folded into
it, and not called from ``vision_rebase._run_direct_residual_fit``'s
``alignment_calibration`` timing bracket): it recomputes the same Procrustes
fit a second time purely for analysis, and keeping it out of the fit's own
cost is what keeps the ``alignment_calibration_seconds``/``_peak_memory_bytes``
numbers comparable across code generations even in the default path.

Reuses the synthetic CLIP-shaped fixture from ``tests/test_direct_residual_
fit.py`` / ``tests/test_direct_residual_component_coverage.py``; this module
has no cross-test imports (repo convention).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_alignment_diagnostics,
    compute_desired_effects,
    fit_direct_residual,
    parse_direct_residual_config,
)
from merge_and_rebase.eval.target_residual_completion import centered_rectangular_procrustes
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

# --------------------------------------------------------------------------
# Golden hashes for the untouched transported_delta path, recorded at HEAD
# d77bbce (clean tree, via `git worktree add ... d77bbce`) on this module's
# fixture -- identical fixture/config to
# test_direct_residual_component_coverage.py's block_boundary golden values.
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


def _capture(source_depth, target_depth, config, **setup_kwargs):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(
        source_depth, target_depth, **setup_kwargs
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=config.num_batches,
        seed=config.seed,
        device="cpu",
    )
    return captured, target_base, target_base_sd, pairing


def _fit(source_depth, target_depth, config=None, **setup_kwargs):
    config = config or DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    captured, target_base, target_base_sd, pairing = _capture(source_depth, target_depth, config, **setup_kwargs)
    desired = compute_desired_effects(captured, pairing, residual_target=config.residual_target)
    alignment_diagnostics = compute_alignment_diagnostics(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=config,
        device="cpu",
    )
    return corrections, diagnostics, alignment_diagnostics


REGIMES = [
    pytest.param(2, 4, id="extend"),
    pytest.param(4, 2, id="shrink"),
    pytest.param(3, 3, id="same_arch"),
]


# --------------------------------------------------------------------------
# 1. Default unchanged: transported_delta (default/explicit) is golden-hash
#    pinned, exactly as test_direct_residual_component_coverage.py's
#    block_boundary path.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_depth, target_depth, golden",
    [(2, 4, _GOLDEN_EXTEND), (4, 2, _GOLDEN_SHRINK), (3, 3, _GOLDEN_SAME_ARCH)],
)
def test_golden_hash_default_residual_target(source_depth, target_depth, golden):
    corrections, _diag, _align = _fit(source_depth, target_depth)
    assert _state_dict_sha256(corrections) == golden


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_explicit_transported_delta_matches_default(source_depth, target_depth):
    """An absent `residual_target` and an explicit `"transported_delta"` are
    the same behaviour: identical tau sha256, not merely close."""
    default_corrections, _diag, _align = _fit(source_depth, target_depth)
    explicit_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, residual_target="transported_delta")
    explicit_corrections, _diag2, _align2 = _fit(source_depth, target_depth, config=explicit_config)
    assert _state_dict_sha256(explicit_corrections) == _state_dict_sha256(default_corrections)


def test_default_config_field_and_parser():
    cfg = DirectResidualConfig()
    assert cfg.residual_target == "transported_delta"
    parsed = parse_direct_residual_config(None)
    assert parsed.residual_target == "transported_delta"
    parsed_explicit = parse_direct_residual_config({"residual_target": "transported_delta"})
    assert parsed_explicit.residual_target == "transported_delta"


# --------------------------------------------------------------------------
# 2. Algebraic identity: D_end - D_delta == E on random rows (pure algebra,
#    float64, independent of the fixture/model machinery above).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n, d_s, d_t", [(50, 6, 6), (50, 4, 9), (50, 9, 4)])
def test_algebraic_identity_endpoint_minus_delta_equals_procrustes_error(n, d_s, d_t):
    generator = torch.Generator().manual_seed(0)
    s0 = torch.randn(n, d_s, generator=generator, dtype=torch.float64)
    s1 = s0 + 0.3 * torch.randn(n, d_s, generator=generator, dtype=torch.float64)
    t0 = torch.randn(n, d_t, generator=generator, dtype=torch.float64)

    q, mu_s, mu_t = centered_rectangular_procrustes(s0, t0)
    e = (s0 - mu_s) @ q - (t0 - mu_t)
    d_delta = (s1 - s0) @ q
    d_end = (s1 - mu_s) @ q + mu_t - t0

    torch.testing.assert_close(d_end - d_delta, e, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------
# 3. Equivalence limit: when the Procrustes fit is exact (E == 0), the two
#    targets and their fitted task vectors agree to tolerance. The concrete
#    instance used is R = I, c = 0, d_s = d_t (target_base IS source_base):
#    a valid, maximally simple case of "T0 = (S0 - mu)R + mu_t + c" with R
#    exactly orthogonal, since a Procrustes fit of any point cloud onto
#    itself returns Q = I and mu_s = mu_t exactly (up to float64 round-off).
#    Not byte-level equal to the transported_delta path: the endpoint
#    arithmetic differs (a different sequence of additions/subtractions), so
#    only approximate agreement is asserted.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", [(3, 3), (2, 2)])
def test_equivalence_limit_when_target_equals_source(source_depth, target_depth):
    torch.manual_seed(21)
    width = 5
    source_base = _Model(width, source_depth).eval()
    source_ft = _tuned_copy(source_base, seed=22)
    data = _loader(seed=23)
    # target_base IS source_base: pairing is same-arch identity, so every
    # position's T_j^0 literally equals its paired S_{j,0} -- the R=I, c=0
    # instance of the equivalence limit.
    target_base = source_base
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)

    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
    )
    desired_delta = compute_desired_effects(captured, pairing, residual_target="transported_delta")
    desired_end = compute_desired_effects(captured, pairing, residual_target="transported_endpoint")
    # compute_alignment_diagnostics doesn't take residual_target -- it always
    # recomputes the same underlying Procrustes fit compute_desired_effects
    # used internally, regardless of which target that call built.
    alignment_diagnostics = compute_alignment_diagnostics(captured, pairing)

    for j in range(pairing.target_depth):
        diag = alignment_diagnostics[j]
        assert diag["procrustes_error_norm"] == pytest.approx(0.0, abs=1e-6)
        assert diag["procrustes_relative_error"] == pytest.approx(0.0, abs=1e-6)
        for batch_delta, batch_end in zip(desired_delta[j], desired_end[j], strict=True):
            torch.testing.assert_close(batch_delta, batch_end, atol=1e-4, rtol=1e-4)

    config_delta = replace(config, residual_target="transported_delta")
    config_end = replace(config, residual_target="transported_endpoint")
    corrections_delta, _diag_delta = fit_direct_residual(
        target_base, target_base_sd, captured, desired_delta, pairing, config=config_delta, device="cpu"
    )
    corrections_end, _diag_end = fit_direct_residual(
        target_base, target_base_sd, captured, desired_end, pairing, config=config_end, device="cpu"
    )
    assert set(corrections_delta) == set(corrections_end)
    for key in corrections_delta:
        torch.testing.assert_close(corrections_delta[key], corrections_end[key], atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------
# 4. Rectangular split: in^2 + out^2 == ||E||^2, and out == the target's own
#    out-of-range component, only when d_t > d_s (semi-orthogonal Q).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n, d_s, d_t", [(60, 4, 9), (60, 6, 6), (60, 9, 4)])
def test_rectangular_split_pythagorean_and_out_of_range_formula(n, d_s, d_t):
    generator = torch.Generator().manual_seed(7)
    s0 = torch.randn(n, d_s, generator=generator, dtype=torch.float64)
    t0 = torch.randn(n, d_t, generator=generator, dtype=torch.float64)
    q, mu_s, mu_t = centered_rectangular_procrustes(s0, t0)
    e = (s0 - mu_s) @ q - (t0 - mu_t)

    e_in_range = (e @ q.T) @ q
    e_out_of_range = e - e_in_range
    in_norm = float(torch.linalg.norm(e_in_range))
    out_norm = float(torch.linalg.norm(e_out_of_range))
    total_norm = float(torch.linalg.norm(e))

    assert in_norm**2 + out_norm**2 == pytest.approx(total_norm**2, rel=1e-8, abs=1e-8)

    identity = torch.eye(d_t, dtype=torch.float64)
    projector = q.T @ q
    expected_out = float(torch.linalg.norm((t0 - mu_t) @ (identity - projector)))
    assert out_norm == pytest.approx(expected_out, rel=1e-6, abs=1e-8)

    if d_t <= d_s:
        assert out_norm == pytest.approx(0.0, abs=1e-6)
    else:
        assert out_norm > 1e-6


# --------------------------------------------------------------------------
# 5. Validation.
# --------------------------------------------------------------------------


def test_transported_endpoint_requires_block_boundary():
    with pytest.raises(ValueError, match="transported_endpoint"):
        parse_direct_residual_config(
            {
                "residual_target": "transported_endpoint",
                "component_target": "output_local",
                "components": ["attn.out_proj"],
            }
        )


def test_transported_endpoint_with_block_boundary_is_valid():
    cfg = parse_direct_residual_config(
        {"residual_target": "transported_endpoint", "component_target": "block_boundary"}
    )
    assert cfg.residual_target == "transported_endpoint"


def test_unknown_residual_target_raises():
    with pytest.raises(ValueError, match="residual_target"):
        parse_direct_residual_config({"residual_target": "not_a_real_mode"})


def test_compute_desired_effects_rejects_unknown_residual_target():
    captured = {
        "source_base_outputs": {0: [torch.zeros(1, 2, 3)]},
        "source_ft_outputs": {0: [torch.zeros(1, 2, 3)]},
        "target_base_outputs_by_position": {0: [torch.zeros(1, 2, 3)]},
    }
    pairing = DiscreteLayerPairing.compute(1, 1)
    with pytest.raises(ValueError, match="residual_target"):
        compute_desired_effects(captured, pairing, residual_target="bogus")


# --------------------------------------------------------------------------
# 6. Determinism: two endpoint-mode runs give an identical tau sha256.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_endpoint_mode_is_deterministic(source_depth, target_depth):
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, residual_target="transported_endpoint")
    corrections_a, _diag_a, _align_a = _fit(source_depth, target_depth, config=config)
    corrections_b, _diag_b, _align_b = _fit(source_depth, target_depth, config=config)
    assert _state_dict_sha256(corrections_a) == _state_dict_sha256(corrections_b)


# --------------------------------------------------------------------------
# 7. compute_desired_effects's implicit default (`residual_target` omitted)
#    agrees byte-for-byte with the explicit "transported_delta" call, and
#    compute_alignment_diagnostics is keyed identically to compute_desired_
#    effects's return -- exercising both functions' public shapes directly,
#    not just through the fitted tau (test 1 above).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_desired_effects_default_matches_explicit_delta_and_diagnostics_shape(source_depth, target_depth):
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    captured, _target_base, _target_base_sd, pairing = _capture(source_depth, target_depth, config)
    desired_default = compute_desired_effects(captured, pairing)
    desired_explicit = compute_desired_effects(captured, pairing, residual_target="transported_delta")
    alignment_diagnostics = compute_alignment_diagnostics(captured, pairing)
    assert set(desired_default) == set(desired_explicit) == set(alignment_diagnostics)
    for j in desired_default:
        for a, b in zip(desired_default[j], desired_explicit[j], strict=True):
            assert torch.equal(a, b)
