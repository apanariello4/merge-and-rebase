"""DirectResidualConfig.ridge_estimator='none' (dr_ablation_v2_20260925 ablation
switch 3): exact least squares (lambda=0) from the same normal equations
ResidualSufficientStatistics.solve already builds for fixed_relative/
empirical_bayes, with a condition-number guard that fails loudly on a
singular/ill-conditioned system instead of silently returning the
regularized-solver's pseudo-inverse fallback.

Covers: config validation/plumbing; agreement with an independent numpy/torch
lstsq on a tiny well-conditioned problem; a loud failure (not a silent
pseudo-inverse) on a singular design; condition_number recorded in solve()'s
diagnostics; and resident/streaming agreement end to end.
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
    fit_direct_residual_streaming,
    parse_direct_residual_config,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.eval.target_residual_completion import ResidualSufficientStatistics
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

# ---- config plumbing -----------------------------------------------------------


def test_config_accepts_ridge_estimator_none():
    cfg = parse_direct_residual_config({"ridge_estimator": "none"})
    assert cfg.ridge_estimator == "none"


def test_config_still_requires_positive_ridge_relative_for_none():
    # Do not loosen ridge_relative > 0 for fixed_relative/anything else, even
    # though ridge_estimator="none" never reads it.
    with pytest.raises(ValueError):
        parse_direct_residual_config({"ridge_estimator": "none", "ridge_relative": 0.0})
    with pytest.raises(ValueError):
        parse_direct_residual_config({"ridge_estimator": "none", "ridge_relative": -1.0})


def test_config_rejects_unknown_ridge_estimator():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"ridge_estimator": "bogus"})


# ---- ResidualSufficientStatistics.solve(ridge_estimator="none") ---------------


def test_none_matches_independent_lstsq_well_conditioned():
    torch.manual_seed(0)
    n, d_in, d_out = 200, 4, 3
    h = torch.randn(n, d_in, dtype=torch.float64)
    true_w = torch.randn(d_out, d_in, dtype=torch.float64)
    true_b = torch.randn(d_out, dtype=torch.float64)
    e = h @ true_w.T + true_b + 0.01 * torch.randn(n, d_out, dtype=torch.float64)
    t_out = torch.eye(d_out, dtype=torch.float64)

    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, t_out)
    # solve() already returns the weight in nn.Linear orientation ([d_out,
    # d_in], ready as `h @ weight.T`) -- no further transpose needed.
    weight, diag = stats.solve(ridge_relative=0.05, ridge_estimator="none", exact_form=True)
    assert diag["ridge"] == pytest.approx(0.0)
    assert diag["condition_number"] is not None
    fitted_w = weight.double()
    fitted_b = diag["bias_correction"].double()

    # Independent reference: ordinary least squares with an explicit
    # intercept column, via torch.linalg.lstsq.
    ones = torch.ones(n, 1, dtype=torch.float64)
    design = torch.cat([h, ones], dim=1)
    sol = torch.linalg.lstsq(design, e).solution  # [d_in+1, d_out]
    ref_w = sol[:d_in].T
    ref_b = sol[d_in]

    assert torch.allclose(fitted_w, ref_w, rtol=1e-4, atol=1e-5)
    assert torch.allclose(fitted_b, ref_b, rtol=1e-4, atol=1e-5)


def test_none_recovers_exact_fit_when_realizable():
    """An exactly realizable, well-conditioned system: residual must be ~0."""
    torch.manual_seed(1)
    n, d_in, d_out = 50, 4, 3
    h = torch.randn(n, d_in, dtype=torch.float64)
    true_w = torch.randn(d_out, d_in, dtype=torch.float64)
    true_b = torch.randn(d_out, dtype=torch.float64)
    e = h @ true_w.T + true_b  # no noise: exactly realizable
    t_out = torch.eye(d_out, dtype=torch.float64)

    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, t_out)
    _x, diag = stats.solve(ridge_relative=0.05, ridge_estimator="none", exact_form=True)
    assert diag["residual_norm_after"] < 1e-6 * max(diag["residual_norm_before"], 1.0)


def test_none_raises_loudly_on_singular_design():
    torch.manual_seed(2)
    n, d_out = 20, 2
    base = torch.randn(n, 1, dtype=torch.float64)
    # Every column is a multiple of `base`: rank-1 design, singular gram.
    h = base @ torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    e = torch.randn(n, d_out, dtype=torch.float64)
    t_out = torch.eye(d_out, dtype=torch.float64)

    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, t_out)
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        stats.solve(ridge_relative=0.05, ridge_estimator="none", exact_form=True)


def test_fixed_relative_and_empirical_bayes_do_not_report_condition_number():
    torch.manual_seed(3)
    h = torch.randn(30, 4, dtype=torch.float64)
    e = torch.randn(30, 3, dtype=torch.float64)
    t_out = torch.eye(3, dtype=torch.float64)
    for estimator in ("fixed_relative", "empirical_bayes"):
        stats = ResidualSufficientStatistics()
        stats.update(h, e, None, t_out)
        _x, diag = stats.solve(ridge_relative=0.1, ridge_estimator=estimator, exact_form=True)
        assert diag["condition_number"] is None


# ---- end-to-end Direct Residual fit --------------------------------------------


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


N_CLASSES = 6


def _loader(n=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n) % N_CLASSES), batch_size=3, shuffle=False)


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


def test_ridge_none_changes_tau_and_records_condition_number_resident():
    # mlp.c_proj only: this fixture's raw images have only 4 degrees of
    # freedom (`_Visual.input = nn.Linear(4, width=5)`) mapped into a width=5
    # embedding, which makes `attn.out_proj`'s own input (that same
    # embedding, since this fixture's toy attention is itself just a linear
    # map of the block input -- no softmax) EXACTLY rank-deficient before any
    # nonlinearity -- see test_ridge_none_raises_end_to_end_on_ill_conditioned_
    # component below, which exercises exactly that case through the full
    # pipeline. mlp.c_proj's input (post-GELU, width*2=10-dim) is not
    # bottlenecked the same way and is well-conditioned here.
    setup = _setup(2, 4)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    base_config = DirectResidualConfig(num_batches=4, ridge_relative=0.05, components=("mlp.c_proj",))
    none_config = DirectResidualConfig(
        num_batches=4, ridge_relative=0.05, ridge_estimator="none", components=("mlp.c_proj",)
    )

    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=base_config.num_batches,
        seed=base_config.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing, residual_target=base_config.residual_target)
    corr_default, _rows_default = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=base_config, device="cpu"
    )
    corr_none, rows_none = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=none_config, device="cpu"
    )
    assert set(corr_default) == set(corr_none)
    changed = any(not torch.equal(corr_default[k], corr_none[k]) for k in corr_default)
    assert changed
    for row in rows_none:
        assert row["ridge_estimator"] == "none"
        assert row["ridge"] == pytest.approx(0.0)
        assert row["condition_number"] is not None
        assert row["condition_number"] > 0


def test_ridge_none_raises_end_to_end_on_ill_conditioned_component():
    """The full fit_direct_residual pipeline, not just the isolated solver,
    must fail loudly (not silently fall back to a pseudo-inverse) when a
    requested component's own design is ill-conditioned. mlp.c_proj's input
    in this tiny fixture (bottlenecked through a width->2*width c_fc then
    GELU at small input scale) is exactly such a case."""
    setup = _setup(2, 4)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    config = DirectResidualConfig(
        num_batches=4, ridge_relative=0.05, ridge_estimator="none", components=("attn.out_proj",)
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
    desired = compute_desired_effects(captured, pairing, residual_target=config.residual_target)
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        fit_direct_residual(target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu")


def test_ridge_none_resident_streaming_agree():
    setup = _setup(4, 2)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    config_r = DirectResidualConfig(
        num_batches=4, ridge_relative=0.05, ridge_estimator="none", components=("mlp.c_proj",)
    )
    config_s = DirectResidualConfig(
        num_batches=4,
        ridge_relative=0.05,
        ridge_estimator="none",
        activation_storage="streaming",
        components=("mlp.c_proj",),
    )

    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=config_r.num_batches,
        seed=config_r.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing, residual_target=config_r.residual_target)
    corr_r, _rows_r = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config_r, device="cpu"
    )
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=config_s.num_batches,
        seed=config_s.seed,
        device="cpu",
        source_ft_model=source_ft,
    )
    corr_s, _rows_s = fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config_s,
        device="cpu",
    )
    assert set(corr_r) == set(corr_s)
    for key in corr_r:
        assert torch.allclose(corr_r[key], corr_s[key], rtol=1e-4, atol=1e-5), key
