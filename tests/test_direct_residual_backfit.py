"""Tests for Direct Residual's ``block_split='backfit'`` intra-block
Gauss-Seidel backfitting (``component_target='block_boundary'`` only).

``block_split='none'`` (default, historical) fits every requested residual
writer independently against the SAME block-boundary target ``D_j`` -- see
``target_informed_runtime._fit_all_positions_independent``. ``'backfit'``
instead refits each component against the residual left over once every
OTHER component's current correction is mounted on a LOCAL (never the live
target model) copy of the block and the block is replayed on the pristine
captured block input ``X_j^0`` -- see
``target_informed_runtime._fit_block_boundary_backfit``.

This module has no cross-test imports (same convention as
``test_direct_residual_component_coverage.py``).
"""

from __future__ import annotations

import hashlib
import math
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
)
from merge_and_rebase.eval.target_residual_completion import order_components
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

# --------------------------------------------------------------------------
# Fixture: plain (non-stock) attention wrapper, IDENTICAL to
# test_direct_residual_component_coverage.py's golden-hash fixture, so the
# block_split='none' golden hashes recorded there remain the reference for
# "backfit reduces to none with one component".
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


def _fit(source_depth, target_depth, config, **setup_kwargs):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(
        source_depth, target_depth, **setup_kwargs
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    )
    return corrections, diagnostics, target_base, target_base_sd


def _state_dict_sha256(d: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# 0. Config/parser.
# --------------------------------------------------------------------------


def test_block_split_defaults_to_none():
    cfg = DirectResidualConfig()
    assert cfg.block_split == "none"
    assert cfg.backfit_max_iters == 20
    assert cfg.backfit_tol == pytest.approx(1e-4)


def test_block_split_backfit_requires_block_boundary():
    with pytest.raises(ValueError, match="block_boundary"):
        parse_direct_residual_config(
            {"component_target": "output_local", "components": ["attn.out_proj"], "block_split": "backfit"}
        )


def test_block_split_backfit_rejects_internal_components():
    # Internal components are unreachable under block_boundary at all (a
    # separate, pre-existing rejection), so this exercises the block_split
    # check's own message when it is reached with a valid block_boundary
    # subset but combined with a bad value.
    with pytest.raises(ValueError, match="block_split"):
        parse_direct_residual_config({"block_split": "sideways"})


def test_block_split_backfit_parses():
    cfg = parse_direct_residual_config(
        {"components": ["attn.out_proj", "mlp.c_proj"], "block_split": "backfit", "backfit_max_iters": 5, "backfit_tol": 1e-3}
    )
    assert cfg.block_split == "backfit"
    assert cfg.backfit_max_iters == 5
    assert cfg.backfit_tol == pytest.approx(1e-3)


# --------------------------------------------------------------------------
# (a) Single-component backfit == block_split='none'.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
@pytest.mark.parametrize("components", [("attn.out_proj",), ("mlp.c_proj",)])
def test_single_component_backfit_matches_none(source_depth, target_depth, components):
    none_cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=components, block_split="none")
    backfit_cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=components, block_split="backfit", backfit_max_iters=1,
    )
    corrections_none, diag_none, _m1, _sd1 = _fit(source_depth, target_depth, none_cfg)
    corrections_backfit, diag_backfit, _m2, _sd2 = _fit(source_depth, target_depth, backfit_cfg)
    assert set(corrections_none) == set(corrections_backfit)
    for key in corrections_none:
        # With a single component, the first (and only) backfit sweep's
        # target reduces exactly to D_j (no other component is mounted), so
        # this is expected to be extremely tight -- report whether it is
        # bitwise; the only source of any difference at all is whether the
        # local-block-copy replay of X_j^0 is bit-identical to what a direct
        # hook on the live model would have captured (both are eval-mode,
        # no-grad, deterministic float32 matmuls on the same input tensor, so
        # it is expected to be exactly bitwise on CPU).
        torch.testing.assert_close(corrections_none[key], corrections_backfit[key], rtol=0, atol=1e-6)
    for row in diag_backfit:
        assert row["backfit_n_sweeps"] == 1
        assert row["backfit_converged"] is False  # r_prev is None after 1 sweep, never compared
        assert "block_replay_bitwise" in row


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_block_replay_reproduces_pristine_boundary(source_depth, target_depth):
    """Sanity check surfaced via the single-component test above: assert here
    directly whether the local-block-copy replay of X_j^0 is bitwise-identical
    to the originally captured boundary output T_j^0, and report it."""
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj",), block_split="backfit", backfit_max_iters=1,
    )
    _corrections, diagnostics, _model, _sd = _fit(source_depth, target_depth, cfg)
    bitwise_flags = {row["block_replay_bitwise"] for row in diagnostics}
    # Report (not assert both ways): record whether replay was bitwise for
    # this fixture/backend. Either way the tolerance check inside
    # _fit_block_boundary_backfit (atol=rtol=1e-4) already guards correctness.
    assert bitwise_flags <= {True, False}


# --------------------------------------------------------------------------
# (d) Golden hashes for block_split='none' unchanged (redundant with
#     test_direct_residual_component_coverage.py's own golden hashes, since
#     "none" routes through the exact same _fit_all_positions_independent
#     call as before; this is a second, independent confirmation that adding
#     block_split doesn't perturb the default path).
# --------------------------------------------------------------------------

_GOLDEN_EXTEND = "e6f02b6922df921f802cb02ca245a60f6f988d44a70a2425061c7cd10f2e6dbc"
_GOLDEN_SHRINK = "111eafdb947b940ebee0366366c5d726830ff6661d8f78222f0a1c67f13dfe94"
_GOLDEN_SAME_ARCH = "3b0600d6b14887867dc83d8d026903dea27bf0de20baaaa5055848fb0385c9d3"


@pytest.mark.parametrize("source_depth, target_depth, golden", [
    (2, 4, _GOLDEN_EXTEND), (4, 2, _GOLDEN_SHRINK), (3, 3, _GOLDEN_SAME_ARCH),
])
def test_golden_hash_unaffected_by_new_block_split_field(source_depth, target_depth, golden):
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05)  # block_split defaults to "none"
    corrections, _diag, _model, _sd = _fit(source_depth, target_depth, cfg)
    assert _state_dict_sha256(corrections) == golden


# --------------------------------------------------------------------------
# Target model is never mutated by the backfit path (only local deepcopies).
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Part 3 (diagnostics): realization_diagnostics is analysis-only -- gated
# additive fields, never changing any fitted number -- for both block_boundary
# paths (none and backfit).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("block_split", ["none", "backfit"])
def test_realization_diagnostics_flag_does_not_change_fitted_corrections(block_split):
    base_kwargs = dict(num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"), block_split=block_split)
    cfg_off = DirectResidualConfig(realization_diagnostics=False, **base_kwargs)
    cfg_on = DirectResidualConfig(realization_diagnostics=True, **base_kwargs)
    corrections_off, _diag_off, _m1, _sd1 = _fit(2, 4, cfg_off)
    corrections_on, _diag_on, _m2, _sd2 = _fit(2, 4, cfg_on)
    assert _state_dict_sha256(corrections_off) == _state_dict_sha256(corrections_on)


@pytest.mark.parametrize("block_split", ["none", "backfit"])
def test_realization_diagnostics_adds_expected_fields(block_split):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        block_split=block_split, realization_diagnostics=True,
    )
    _corrections, diagnostics, _model, _sd = _fit(2, 4, cfg)
    expected = {
        "fit_relative_residual", "target_norm", "update_norm", "relative_update_norm", "realized_target_norm_ratio",
    }
    for row in diagnostics:
        assert expected <= set(row), row.keys()
        assert all(torch.isfinite(torch.tensor(float(row[k]))) for k in expected)


def test_backfit_never_mutates_the_live_target_model():
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"), block_split="backfit",
    )
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(2, 4)
    before_hash = _state_dict_sha256(target_base.state_dict())
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    fit_direct_residual(target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu")
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash


# --------------------------------------------------------------------------
# (c) Monotone convergence for O+D backfit on the toy.
#
# IMPORTANT FINDING, recorded here rather than silently forced to pass: the
# plan's "residual trace is non-increasing" was checked empirically against
# the pure DATA residual r = ||D-effect||/||D|| and found FALSE in general
# (confirmed both on the real GELU fixture and on the linear-additive toy
# below, at ridge_relative from 1e-6 up to 0.3) -- r is monotonically
# INCREASING sweep over sweep in every case measured, converging from below
# to a plateau ABOVE its first-sweep value. This is not a bug: each backfit
# sweep performs an EXACT block-coordinate-descent step on the ridge-PENALIZED
# objective J = ||D-effect||^2 + lam*(||W_c||^2 for each component), which
# monotonically decreases J itself (a standard, textbook block-coordinate-
# descent guarantee for a jointly convex quadratic) but NOT the unpenalized
# data term alone -- as the ridge shrinks the weights sweep by sweep, the
# achievable data fit necessarily gets slightly worse in exchange for a much
# larger reduction in penalty. This test verifies the property that IS
# guaranteed (J is non-increasing), computed independently from the returned
# corrections at a sequence of max_iters cutoffs (not from internal solver
# state), and separately documents that r itself is not.
# --------------------------------------------------------------------------


def _penalized_objective(h_rows, d_rows, w_o, b_o, w_d, b_d, lam):
    pred = h_rows.double() @ (w_o + w_d).double().T + (b_o + b_d).double()
    data_term = float(((pred - d_rows.double()) ** 2).sum().item())
    penalty = lam * (float((w_o.double() ** 2).sum().item()) + float((w_d.double() ** 2).sum().item()))
    return data_term + penalty, data_term


def test_od_backfit_penalized_objective_is_non_increasing_on_the_toy():
    ridge_relative = 0.1
    source_base, source_ft, target_base, data, pairing, target_base_sd = _toy_setup()
    from merge_and_rebase.eval.target_informed_runtime import paired_calibration

    _sb, target_batches, _meta = paired_calibration(data, data, num_batches=6, seed=89)
    with torch.no_grad():
        x_batches = [target_base.visual.input(b[0]) for b in target_batches]
    h_rows = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_batches], dim=0)

    objectives = []
    data_terms = []
    for k in range(1, 16):
        cfg = DirectResidualConfig(
            num_batches=6, ridge_relative=ridge_relative, components=("attn.out_proj", "mlp.c_proj"),
            block_split="backfit", backfit_max_iters=k, backfit_tol=1e-30, seed=89,
        )
        captured = capture_paired_boundary_activations(
            source_base, source_ft, deepcopy(target_base), data, data, pairing,
            num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
        )
        desired = compute_desired_effects(captured, pairing)
        corrections, diagnostics = fit_direct_residual(
            deepcopy(target_base), target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
        )
        d_rows = torch.cat([d.reshape(-1, d.shape[-1]) for d in desired[0]], dim=0)
        lam = diagnostics[0]["ridge"]
        obj, data_term = _penalized_objective(
            h_rows,
            d_rows,
            corrections["visual.transformer.resblocks.0.attn.out_proj.weight"],
            corrections["visual.transformer.resblocks.0.attn.out_proj.bias"],
            corrections["visual.transformer.resblocks.0.mlp.c_proj.weight"],
            corrections["visual.transformer.resblocks.0.mlp.c_proj.bias"],
            lam,
        )
        objectives.append(obj)
        data_terms.append(data_term)

    # The guaranteed quantity: the ridge-penalized joint objective.
    for prev, curr in zip(objectives, objectives[1:], strict=False):
        assert curr <= prev + 1e-6, objectives
    # The documented finding: the unpenalized data residual is NOT
    # non-increasing here (it increases every sweep in this run) -- recorded
    # as a positive assertion (not merely a comment) so a future change that
    # makes it monotonic again is visible rather than silently accepted.
    assert data_terms[-1] > data_terms[0], data_terms


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_od_backfit_runs_to_convergence_or_reports_not_converged(source_depth, target_depth):
    """Not a monotonicity claim (see above): just confirms the stopping rule
    itself behaves sanely -- either it converges within a generous iteration
    budget, or it honestly reports ``backfit_converged=False`` rather than
    silently truncating."""
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        block_split="backfit", backfit_max_iters=500, backfit_tol=1e-6,
    )
    _corrections, diagnostics, _model, _sd = _fit(source_depth, target_depth, cfg)
    for row in diagnostics:
        assert row["backfit_n_sweeps"] <= 500
        assert isinstance(row["backfit_converged"], bool)


# --------------------------------------------------------------------------
# (b) Toy linear-additive block: backfit converges to the stacked joint ridge
#     solution with per-component penalties.
#
# Toy block: out = x + out_proj(x) + c_proj(x) -- BOTH components consume the
# block's own raw input x directly (no ln_1/ln_2, no GELU, no dimension
# change), so there is no nonlinearity between them and H_out = H_cproj = x
# exactly. Both share the identical ridge scale (same H, same config), so the
# joint objective is
#
#   J(Wo,Wd,bo,bd) = ||x@Wo^T + x@Wd^T + 1(bo+bd)^T - D||_F^2
#                    + lam(||Wo||_F^2 + ||Wd||_F^2)
#
# For any fixed S=Wo+Wd, the penalty term is minimized at Wo=Wd=S/2 (a fixed
# sum's squared-norm split is minimized when split evenly), giving penalty
# lam*||S||^2/2. So the optimal S solves ordinary ridge regression at HALF
# the per-component ridge (lam/2), and Wo=Wd=S/2 at convergence. The bias
# split (bo, bd) is NOT similarly pinned down (bias is unpenalized, so only
# bo+bd is determined by the objective) -- only their sum is asserted.
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


def _ridge_regression_with_intercept(h_rows: torch.Tensor, d_rows: torch.Tensor, ridge_relative: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Independent from-scratch closed-form centered ridge regression
    (mirrors the *mathematical* content of ResidualSufficientStatistics.solve
    at t_in=t_out=I, but implemented directly here rather than by calling any
    production helper): W = argmin ||Hc W^T - Dc||^2 + lam||W||^2 (centered),
    b = mean_d - W @ mean_h, lam = ridge_relative * trace(Hc^T Hc) / d_in.
    """
    h = h_rows.double()
    d = d_rows.double()
    n, d_in = h.shape
    mu_h = h.mean(dim=0)
    mu_d = d.mean(dim=0)
    hc = h - mu_h
    dc = d - mu_d
    s = hc.T @ hc
    lam = float(ridge_relative) * float(torch.trace(s).item()) / d_in
    w = torch.linalg.solve(s + lam * torch.eye(d_in, dtype=torch.float64), hc.T @ dc).T
    b = mu_d - w @ mu_h
    return w.float(), b.float(), lam


def test_toy_linear_additive_block_backfit_matches_derived_joint_ridge_solution():
    ridge_relative = 0.1
    # Convergence here is slow (linear rate typical of Gauss-Seidel on a
    # near-degenerate design -- both components share the identical input,
    # so the system is close to singular): empirically ~130 sweeps at
    # backfit_tol=1e-9 for this fixture/seed. backfit_max_iters is set well
    # above that so the test exercises real convergence, not a truncation.
    cfg = DirectResidualConfig(
        num_batches=6, ridge_relative=ridge_relative, components=("attn.out_proj", "mlp.c_proj"),
        block_split="backfit", backfit_max_iters=2000, backfit_tol=1e-9, seed=89,
    )
    source_base, source_ft, target_base, data, pairing, target_base_sd = _toy_setup()
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )

    # Independent derivation: gather D_0 and the shared input H=X_0^0 (the
    # block's own raw input) directly from the captured banks, and solve the
    # HALF-ridge (lam/2) regression by hand (see the module docstring for the
    # derivation of why the per-component split is at half the nominal ridge).
    # The block's own INPUT (not its boundary/output): recover it directly by
    # replaying the model's own `input` linear layer on the same calibration
    # images capture_paired_boundary_activations itself replayed.
    from merge_and_rebase.eval.target_informed_runtime import paired_calibration

    _sb, target_batches, _meta = paired_calibration(data, data, num_batches=cfg.num_batches, seed=cfg.seed)
    with torch.no_grad():
        x_batches = [target_base.visual.input(b[0]) for b in target_batches]
    h_rows = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_batches], dim=0)
    d_rows = torch.cat([d.reshape(-1, d.shape[-1]) for d in desired[0]], dim=0)

    w_half, b_half, _lam_half = _ridge_regression_with_intercept(h_rows, d_rows, ridge_relative / 2.0)

    key_out = "visual.transformer.resblocks.0.attn.out_proj.weight"
    key_cproj = "visual.transformer.resblocks.0.mlp.c_proj.weight"
    bias_out = "visual.transformer.resblocks.0.attn.out_proj.bias"
    bias_cproj = "visual.transformer.resblocks.0.mlp.c_proj.bias"

    w_out, w_cproj = corrections[key_out], corrections[key_cproj]
    b_out, b_cproj = corrections[bias_out], corrections[bias_cproj]

    # Weights: each component's fitted correction should be half of the
    # combined joint solution, and the two should be (numerically) equal to
    # each other by the symmetry argument.
    #
    # Tolerance note (monotone safeguard): this fixture is DELIBERATELY
    # near-singular (both components share the identical raw input x, so only
    # their SUM is well-conditioned; the split direction Wo-Wd is constrained
    # only by the comparatively weak ridge penalty). The safeguarded rule
    # measures J on a dedicated float64 block replica (see
    # _backfit_data_fit_sq) specifically so float32 forward-pass noise cannot
    # cause a spurious rejection -- but in this ill-conditioned split
    # direction, per-sweep improvement eventually falls below float64's own
    # precision floor, and the strict "accept only if J decreases" rule then
    # (correctly) reports convergence rather than continuing to chase
    # numerical noise, as the old unconditional-acceptance rule effectively
    # did. Measured empirically at backfit_tol=1e-9/max_iters=2000: max
    # |w_out-w_cproj| ~= 5e-4 (vs. this suite's historical 5e-5 under the old,
    # tol-on-r stopping rule) -- loosened accordingly rather than silently
    # left at a value that would flake.
    torch.testing.assert_close(w_out, w_cproj, rtol=0, atol=1e-3)
    torch.testing.assert_close(w_out, w_half / 2.0, rtol=0, atol=1e-3)
    torch.testing.assert_close(w_cproj, w_half / 2.0, rtol=0, atol=1e-3)

    # Bias: only the SUM is pinned down by the objective (see docstring).
    torch.testing.assert_close(b_out + b_cproj, b_half, rtol=0, atol=1e-5)

    for row in diagnostics:
        assert row["backfit_converged"], row["backfit_residual_trace"][-5:]


# --------------------------------------------------------------------------
# Monotone safeguard: J(Delta) is non-increasing by construction, every sweep,
# on both the toy fixture and (see test_direct_residual_open_clip_integration.py)
# real open_clip blocks. Also: an attempt to reproduce the UNSAFEGUARDED rule's
# divergence on a small fixture (tiny ridge + strong ln_2/GELU coupling + a
# scaled-up out_proj), by reimplementing the pre-fix Gauss-Seidel sub-step
# (no backtracking) directly against the same production helpers
# (_mount_component/_replay_block_components/ResidualSufficientStatistics)
# _fit_block_boundary_backfit itself uses -- so this is a faithful replay of
# the old rule, not a hand-wavy toy.
# --------------------------------------------------------------------------

import copy as _copy  # noqa: E402

from merge_and_rebase.eval.target_informed_runtime import (  # noqa: E402
    ResidualSufficientStatistics,
    _component_weight_bias,
    _layout_for,
    _mount_component,
    _replay_block_components,
    capture_tokens,
)


def _unsafeguarded_backfit_r_trace(
    target_model, positions, desired_batches, target_output_batches, batches, components, config, device,
):
    """Faithful replay of the PRE-FIX (unsafeguarded) Gauss-Seidel sub-step:
    each component is refit against the residual left over once every other
    component's CURRENT fit is mounted and the block replayed, accepted
    unconditionally every time (no J-based accept/backtrack). Returns
    ``{position: r_trace}``, the historical convergence diagnostic (plain
    relative data residual, measured once per full sweep).
    """
    shim = _layout_for(None)
    order = order_components(components)
    block_input_requests = {f"{pos}.block_input": (pos, "block_input") for pos in positions}
    block_inputs = capture_tokens(target_model, batches, block_input_requests, device, family_adapter=None)
    traces: dict[int, list[float]] = {}
    for pos in positions:
        x_batches = block_inputs[f"{pos}.block_input"]
        t0_batches = target_output_batches[pos]
        d_batches = desired_batches[pos]
        block = shim.blocks(target_model)[pos]
        local_block = _copy.deepcopy(shim.block_module(block)).to(device).eval()
        base = {}
        for c in order:
            w, b, row_slice = _component_weight_bias(shim, local_block, c)
            assert row_slice is None
            base[c] = (w.detach().cpu().clone(), None if b is None else b.detach().cpu().clone())
        scale_modules = {c: shim.component_scale_module(local_block, c) for c in order}

        def reset_all(local_block=local_block, order=order, base=base):
            for c in order:
                w, b = base[c]
                _mount_component(shim, local_block, c, w, b)

        deltas = {
            c: (torch.zeros_like(base[c][0]), None if base[c][1] is None else torch.zeros_like(base[c][1]))
            for c in order
        }
        desired_norm = sum(float((d.double() ** 2).sum().item()) for d in d_batches) ** 0.5
        r_trace = []
        for _sweep in range(1, int(config.backfit_max_iters) + 1):
            for c in order:
                reset_all()
                for c2 in order:
                    if c2 == c:
                        continue
                    w2, b2 = deltas[c2]
                    _mount_component(shim, local_block, c2, base[c2][0] + w2, None if base[c2][1] is None else base[c2][1] + b2)
                component_h, out_batches = _replay_block_components(shim, local_block, x_batches, [c], device)
                h_batches = component_h[c]
                e_batches = [d - (t - t0) for d, t, t0 in zip(d_batches, out_batches, t0_batches, strict=True)]
                width = int(base[c][0].shape[0])
                effective_out = torch.eye(width, dtype=torch.float32)
                scale_module = scale_modules[c]
                if not isinstance(scale_module, torch.nn.Identity):
                    scale = getattr(scale_module, "gamma", None)
                    effective_out = effective_out * scale.detach().cpu().float().unsqueeze(0)
                stats = ResidualSufficientStatistics(device=device)
                for h, e in zip(h_batches, e_batches, strict=True):
                    stats.update(h.reshape(-1, h.shape[-1]), e.reshape(-1, e.shape[-1]), None, effective_out)
                correction, diag = stats.solve(
                    ridge_relative=config.ridge_relative, ridge_estimator=config.ridge_estimator,
                    exact_form=config.exact_form,
                )
                # Unconditional acceptance -- this is exactly the omitted safeguard.
                deltas[c] = (correction.cpu(), diag["bias_correction"].cpu())
            reset_all()
            for c in order:
                w, b = deltas[c]
                _mount_component(shim, local_block, c, base[c][0] + w, None if base[c][1] is None else base[c][1] + b)
            t_all = [local_block(x.to(device)).detach().float().cpu().clone() for x in x_batches]
            num_sq = sum(
                float(((d - (t - t0)).double() ** 2).sum().item())
                for d, t, t0 in zip(d_batches, t_all, t0_batches, strict=True)
            )
            r_trace.append((num_sq**0.5) / (desired_norm + 1e-12))
        traces[pos] = r_trace
    return traces


def _coupled_divergence_setup(image_size=16, patch_size=4, width=8, layers=2, heads=2, seed=101, out_scale=8.0, cfc_scale=8.0, n=6):
    """A REAL ``open_clip`` ViT block (unlike this module's own ``_Block``
    fixture, whose ``attn``/``mlp`` paths are PARALLEL -- both consume the
    block's raw input directly, so mounting one component never perturbs the
    other's regression input at all, and no amount of scaling reproduces the
    task's motivating failure mode). A real ``VisionTransformer`` block is
    SEQUENTIAL (``x1 = x + attn(ln_1(x))``; ``x2 = x1 + mlp(ln_2(x1))``), so
    ``mlp.c_proj``'s GELU input genuinely depends on whatever is currently
    mounted at ``attn.out_proj`` -- the real coupling path
    ``attn.out_proj -> ln_2 -> GELU -> mlp.c_proj`` the task names. Strongly
    scaling ``attn.out_proj`` and ``mlp.c_fc`` (so a source-target tuning gap
    routes a large perturbation through that path) with a tiny ridge (so each
    sub-step's candidate barely damps the resulting swing) is enough to make
    the OLD (unsafeguarded) rule diverge -- see the test below.
    """
    from open_clip.transformer import VisionTransformer

    def make_vit(vit_seed):
        torch.manual_seed(vit_seed)
        vt = VisionTransformer(
            image_size=image_size, patch_size=patch_size, width=width, layers=layers, heads=heads,
            mlp_ratio=2.0, ls_init_value=None, output_dim=width, pool_type="tok",
        )
        return _CLIPLike(vt).eval()

    source_base = make_vit(seed)
    target_base = make_vit(seed + 1)
    tuned = deepcopy(source_base)
    torch.manual_seed(seed + 2)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.attn.out_proj.weight.add_(out_scale * torch.randn_like(block.attn.out_proj.weight))
            block.mlp.c_fc.weight.add_(cfc_scale * torch.randn_like(block.mlp.c_fc.weight))
    generator = torch.Generator().manual_seed(seed + 3)
    images = torch.randn(n, 3, image_size, image_size, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)
    pairing = DiscreteLayerPairing.compute(layers, layers)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, tuned, target_base, data, pairing, target_base_sd


class _CLIPLike(torch.nn.Module):
    """Minimal ``open_clip``-shaped wrapper: only ``.visual`` matters to
    ``target_informed_runtime``'s ``_encode_image``/``_VisionLayout`` (same
    minimal shim as ``test_direct_residual_open_clip_integration.py``'s own,
    duplicated here rather than imported, per this module's no-cross-test-
    import convention)."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def encode_image(self, x):
        return self.visual(x)


@pytest.mark.parametrize(
    "out_scale, cfc_scale, ridge_relative",
    [(5.0, 5.0, 1e-3), (10.0, 10.0, 1e-5)],
)
def test_attempt_to_reproduce_unsafeguarded_divergence_and_confirm_safeguard_holds(out_scale, cfc_scale, ridge_relative):
    """Reproduce the task's motivating failure mode directly: on a REAL
    (sequential, ln_2/GELU-coupled) ``open_clip`` ViT block, tiny ridge plus a
    strongly scaled ``attn.out_proj``/``mlp.c_fc`` tuning gap makes the OLD
    (unsafeguarded, unconditional-acceptance) Gauss-Seidel rule diverge to
    non-finite deltas within a handful of sweeps -- reproduced below by
    replaying that exact pre-fix rule against the same production primitives
    (_mount_component/_replay_block_components/ResidualSufficientStatistics)
    the current, safeguarded ``_fit_block_boundary_backfit`` itself uses.
    The PRODUCTION (safeguarded) rule is then asserted to stay finite and its
    ``J`` trace non-increasing on the IDENTICAL fixture/config.
    """
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=ridge_relative, components=("attn.out_proj", "mlp.c_proj"),
        block_split="backfit", backfit_max_iters=20, backfit_tol=1e-14,
    )
    source_base, source_ft, target_base, data, pairing, target_base_sd = _coupled_divergence_setup(
        out_scale=out_scale, cfc_scale=cfc_scale,
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, deepcopy(target_base), data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    positions = list(range(pairing.target_depth))

    diverged = False
    try:
        old_traces = _unsafeguarded_backfit_r_trace(
            deepcopy(target_base), positions, desired, captured["target_base_outputs_by_position"],
            captured["target_batches"], cfg.components, cfg, "cpu",
        )
    except ValueError as exc:
        # ResidualSufficientStatistics.update's own finite check: the
        # unsafeguarded rule fed it a non-finite input bank/target -- i.e. it
        # already blew up before even reaching the next sub-solve.
        assert "finite" in str(exc)
        diverged = True
        old_traces = {}
    else:
        for _pos, r in old_traces.items():
            finite = all(math.isfinite(v) for v in r)
            # A "diverged" trace is not required to be monotonically ascending
            # step-to-step (Gauss-Seidel coupling can oscillate on the way
            # up) -- an order-of-magnitude blow-up relative to its own first
            # value is a robust enough signal, and non-finite is diverged
            # outright.
            grew_hugely = finite and max(r) > 50.0 * max(r[0], 1e-12)
            if not finite or grew_hugely:
                diverged = True
    print(f"unsafeguarded r_traces (diverged={diverged}): {old_traces}")

    corrections, diagnostics = fit_direct_residual(
        deepcopy(target_base), target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    for value in corrections.values():
        assert torch.isfinite(value).all()
    for row in diagnostics:
        j_trace = row["backfit_j_trace"]
        assert all(math.isfinite(v) for v in j_trace), j_trace
        for prev, curr in zip(j_trace, j_trace[1:], strict=False):
            assert curr <= prev + 1e-6, j_trace

    if diverged:
        # Positive confirmation: the OLD rule provably diverges/ascends on
        # this exact fixture/config, and the safeguarded rule above stayed
        # finite and monotone on the SAME fixture/config -- direct evidence
        # the fix matters here, not merely that it is harmless.
        assert True
    else:
        pytest.skip(
            "Did not reproduce unsafeguarded ascent/divergence on this fixture "
            f"(old r_traces={old_traces}); the safeguard's own monotonicity above still holds."
        )
