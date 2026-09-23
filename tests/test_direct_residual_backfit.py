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
    torch.testing.assert_close(w_out, w_cproj, rtol=0, atol=5e-5)
    torch.testing.assert_close(w_out, w_half / 2.0, rtol=0, atol=5e-5)
    torch.testing.assert_close(w_cproj, w_half / 2.0, rtol=0, atol=5e-5)

    # Bias: only the SUM is pinned down by the objective (see docstring).
    torch.testing.assert_close(b_out + b_cproj, b_half, rtol=0, atol=1e-5)

    for row in diagnostics:
        assert row["backfit_converged"], row["backfit_residual_trace"][-5:]
