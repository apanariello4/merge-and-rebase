"""Tests for the transport-free ``direct_target`` Proposal-1 mode.

The transport-aware arm (``mode='transport_residual'``) completes the residual
left by an already-transported task vector.  This arm asks whether the desired
local functional effect can be written into the native target base *without
any parameter transport at all*, i.e. whether a useful target task vector
exists with ``tau_t^transport = 0``.

Two claims need pinning down, because the implementation deliberately reuses
the transport-aware solver rather than restating the normal equations:

1.  ``t_in=None`` (identity input map, never materialized) is numerically the
    same solve as passing an explicit identity, and both agree with the
    textbook centered ridge closed form.  If that ever drifts, the two modes
    stop being comparable ablations of one method.
2.  The per-block ``relative_residual_after`` diagnostic is reported from the
    solver's own objective rather than re-measured with a second forward pass.
    That is only legitimate because ``c_proj`` is the last operation writing
    into the residual stream and its input does not depend on its own weight,
    so mounting the correction changes the block output by exactly the
    quantity the solver evaluated.  The equality is asserted here instead of
    paying 2x the calibration forward passes in every production run.

Hash-level comparison is used for the null controls, following CLAUDE.md and
``tests/test_brace_rebase_extension_spread_mod_20260909.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from merge_and_rebase.eval.target_informed_runtime import (
    capture_tokens,
    complete_residuals_direct,
    scale_completion,
)
from merge_and_rebase.eval.target_residual_completion import (
    ResidualCompletionConfig,
    ResidualSufficientStatistics,
    parse_residual_completion_config,
)
from merge_and_rebase.eval.vision_rebase import (
    _maybe_capture_target_residual_references,
    _maybe_complete_target_residual_task_vector,
    _state_dict_sha256,
)

# The tiny ViT stand-in is duplicated from
# tests/test_target_residual_completion_wiring.py rather than imported: the
# suite has no cross-test imports and no tests package, and every module here
# is self-contained.


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
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(width, width * 2)), ("gelu", torch.nn.GELU()),
            ("c_proj", torch.nn.Linear(width * 2, width)),
        ]))
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


def _loader():
    return DataLoader(TensorDataset(torch.randn(6, 5, 4), torch.arange(6)), batch_size=2, shuffle=False)


def _fixture():
    """Source depth 2 (un-resized), target depth 4 (already doubled)."""
    torch.manual_seed(12)
    source = _Model(3, 2).eval()
    target = _Model(5, 4).eval()
    source_ft = deepcopy(source)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.2)
        source_ft.visual.transformer.resblocks[1].mlp.c_proj.weight.sub_(0.15)
    data = _loader()
    layout = {"inserted_blocks": [{"position": 1, "source_orig_idx": 0}, {"position": 3, "source_orig_idx": 1}]}
    target_base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    torch.manual_seed(99)
    baseline_delta = {k: 0.05 * torch.randn_like(v) for k, v in target_base_sd.items()}
    return source, source_ft, target, data, layout, None, target_base_sd, baseline_delta


# --------------------------------------------------------------------------
# 1. The solver identity that lets direct mode reuse the transport-aware code
# --------------------------------------------------------------------------


def _banks(rows=64, d_in=7, d_out=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(rows, d_in, generator=generator) + 0.8
    e = torch.randn(rows, d_out, generator=generator) - 0.4
    return h, e


@pytest.mark.parametrize("exact_form", [True, False])
def test_identity_input_map_matches_an_explicit_identity(exact_form):
    """``t_in=None`` must be the same solve as ``t_in=I``, not merely similar.

    The direct path avoids materializing the identity purely for cost (4096x4096
    per batch on a ViT-L/14 target); it must not become a second, subtly
    different estimator.
    """
    h, e = _banks()
    t_out = torch.eye(e.shape[1])

    implicit = ResidualSufficientStatistics()
    implicit.update(h, e, None, t_out)
    weight_implicit, diag_implicit = implicit.solve(ridge_relative=0.01, exact_form=exact_form)

    explicit = ResidualSufficientStatistics()
    explicit.update(h, e, torch.eye(h.shape[1]), t_out)
    weight_explicit, diag_explicit = explicit.solve(ridge_relative=0.01, exact_form=exact_form)

    torch.testing.assert_close(weight_implicit, weight_explicit, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(diag_implicit["bias_correction"], diag_explicit["bias_correction"], rtol=1e-6, atol=1e-6)
    assert diag_implicit["residual_norm_after"] == pytest.approx(diag_explicit["residual_norm_after"], rel=1e-9)


def test_direct_solve_matches_the_centered_ridge_closed_form():
    """Independent check against the textbook affine ridge, solved directly.

    ``min ||H X + 1 beta^T - E||^2 + lam ||X||^2`` has
    ``X = (Hc^T Hc + lam I)^-1 Hc^T Ec`` and ``beta = mu_e - mu_h X``.  This is
    computed here from scratch, not from the module under test.
    """
    h, e = _banks(seed=3)
    ridge_relative = 0.01
    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, torch.eye(e.shape[1]))
    weight, diag = stats.solve(ridge_relative=ridge_relative, exact_form=True)

    h64, e64 = h.double(), e.double()
    mu_h, mu_e = h64.mean(dim=0), e64.mean(dim=0)
    hc, ec = h64 - mu_h, e64 - mu_e
    # The module scales the ridge by trace(Sc)/m_in times trace(G)/d_out; with
    # t_out = I the second factor is exactly 1.
    lam = ridge_relative * torch.trace(hc.T @ hc).item() / h.shape[1]
    x = torch.linalg.solve(hc.T @ hc + lam * torch.eye(h.shape[1], dtype=torch.float64), hc.T @ ec)
    beta = mu_e - mu_h @ x

    torch.testing.assert_close(weight, x.T.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(diag["bias_correction"], beta.float(), rtol=1e-5, atol=1e-6)
    assert diag["ridge"] == pytest.approx(lam, rel=1e-9)


def test_absolute_ridge_uses_one_raw_lambda_and_reports_trace_reference():
    h, e = _banks(seed=19)
    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, torch.eye(e.shape[1]))
    raw_lambda = 7.25
    _, diag = stats.solve(
        ridge_relative=0.01,
        ridge_mode="absolute",
        ridge_absolute=raw_lambda,
        exact_form=True,
    )
    assert diag["ridge"] == pytest.approx(raw_lambda)
    assert diag["ridge_mode"] == "absolute"
    assert diag["ridge_absolute"] == pytest.approx(raw_lambda)
    assert diag["ridge_trace_normalized"] != pytest.approx(raw_lambda)


def test_layerscale_enters_the_direct_fit_as_a_diagonal_output_map():
    """diag(gamma) must be inside the objective, not applied afterwards."""
    h, e = _banks(seed=5)
    gamma = torch.linspace(0.2, 1.5, e.shape[1])
    scaled = ResidualSufficientStatistics()
    scaled.update(h, e, None, torch.eye(e.shape[1]) * gamma.unsqueeze(0))
    weight_scaled, _ = scaled.solve(ridge_relative=0.01, exact_form=True)

    plain = ResidualSufficientStatistics()
    plain.update(h, e, None, torch.eye(e.shape[1]))
    weight_plain, _ = plain.solve(ridge_relative=0.01, exact_form=True)
    assert not torch.allclose(weight_scaled, weight_plain)


# --------------------------------------------------------------------------
# 2. Config surface
# --------------------------------------------------------------------------


def test_mode_defaults_to_the_historical_transport_aware_path():
    assert ResidualCompletionConfig().mode == "transport_residual"
    assert parse_residual_completion_config({"enabled": True}).mode == "transport_residual"


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be"):
        parse_residual_completion_config({"enabled": True, "mode": "direct"})


# --------------------------------------------------------------------------
# 3. End-to-end behaviour of the direct arm
# --------------------------------------------------------------------------


def _direct_config(**overrides):
    params = {
        "enabled": True,
        "mode": "direct_target",
        "ridge_relative": 0.05,
        "num_batches": 3,
        "strength": 1.0,
    }
    params.update(overrides)
    return ResidualCompletionConfig(**params)


def _references(source, source_ft, target, data, config, seed=0):
    return _maybe_capture_target_residual_references(
        config=config, source_base_model=source, source_ft_model=source_ft, target_model=target,
        source_loader=data, target_loader=data, seed=seed, device="cpu",
    )


def test_direct_completion_writes_only_the_selected_cproj_keys():
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = _direct_config()
    references = _references(source, source_ft, target, data, config)
    corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    assert [row["position"] for row in diagnostics] == [1, 3]
    assert all(row["mode"] == "direct_target" for row in diagnostics)
    assert set(corrections) == {
        "visual.transformer.resblocks.1.mlp.c_proj.weight",
        "visual.transformer.resblocks.1.mlp.c_proj.bias",
        "visual.transformer.resblocks.3.mlp.c_proj.weight",
        "visual.transformer.resblocks.3.mlp.c_proj.bias",
    }
    for key, correction in corrections.items():
        assert correction.shape == target_base_sd[key].shape


def test_first_block_residual_equals_the_desired_effect_exactly():
    """E_j == D_j at the first fitted block: nothing has perturbed the base yet."""
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = _direct_config()
    references = _references(source, source_ft, target, data, config)
    _corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    first = diagnostics[0]
    assert first["effect_before_norm"] == pytest.approx(0.0, abs=1e-6)
    assert first["residual_norm_before"] == pytest.approx(first["desired_norm"], rel=1e-5)
    assert first["relative_residual_before"] == pytest.approx(1.0, rel=1e-5)


def test_later_blocks_see_the_corrections_already_mounted_upstream():
    """The cascade must be sequential: block 3 repairs only what block 1 left."""
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = _direct_config()
    references = _references(source, source_ft, target, data, config)
    _corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    second = diagnostics[1]
    assert second["effect_before_norm"] > 1e-6, (
        "the upstream correction did not reach the downstream block: the fit is not sequential"
    )


def test_reported_post_fit_residual_equals_the_re_measured_one():
    """Justifies reporting ``relative_residual_after`` without a second capture pass.

    c_proj is the last operation writing into the residual stream in its block
    and its input does not depend on its own weight, so mounting the fitted
    correction changes the block output by exactly the quantity the solver
    minimized. This measures that directly on a single fitted block.
    """
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = _direct_config()
    references = _references(source, source_ft, target, data, config)
    single = {"inserted_blocks": [layout["inserted_blocks"][0]]}
    # Restrict the references to the same single position so the scope check passes.
    trimmed = dict(references)
    position = single["inserted_blocks"][0]["position"]
    for field in ("desired", "target_base_outputs", "maps"):
        trimmed[field] = {position: references[field][position]}
    corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, trimmed, single, data, config=config, device="cpu",
    )

    mounted = deepcopy(target)
    state = {k: v.clone() for k, v in target_base_sd.items()}
    for key, correction in corrections.items():
        state[key] = state[key] + correction.to(state[key])
    mounted.load_state_dict(state, strict=True)

    meta = references["calibration"]
    batches = list(DataLoader(Subset(data.dataset, meta["indices"]), batch_size=meta["batch_size"],
                              shuffle=False, num_workers=0, collate_fn=data.collate_fn))
    captured = capture_tokens(mounted, batches, {"out": (position, "boundary")}, "cpu")
    residual_sq = 0.0
    for out, desired_batch, base_out in zip(
        captured["out"], trimmed["desired"][position], trimmed["target_base_outputs"][position], strict=True
    ):
        error = desired_batch.double() - (out.double() - base_out.double())
        residual_sq += float((error ** 2).sum().item())
    assert residual_sq ** 0.5 == pytest.approx(diagnostics[0]["residual_norm_after"], rel=1e-3)


def test_the_correction_actually_reduces_the_local_residual():
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = _direct_config()
    references = _references(source, source_ft, target, data, config)
    _corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    for row in diagnostics:
        assert row["residual_norm_after"] < row["residual_norm_before"], row["position"]


def test_target_base_weights_are_restored_after_the_direct_fit():
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = _direct_config()
    references = _references(source, source_ft, target, data, config)
    pre_hash = _state_dict_sha256(target.state_dict())
    complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    assert _state_dict_sha256(target.state_dict()) == pre_hash


def test_direct_mode_is_deterministic_under_a_fixed_seed():
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = _direct_config()
    first, _ = complete_residuals_direct(
        target, target_base_sd, _references(source, source_ft, target, data, config, seed=7),
        layout, data, config=config, device="cpu",
    )
    second, _ = complete_residuals_direct(
        target, target_base_sd, _references(source, source_ft, target, data, config, seed=7),
        layout, data, config=config, device="cpu",
    )
    assert _state_dict_sha256(first) == _state_dict_sha256(second)


def test_transport_residual_mode_refuses_to_run_the_direct_solver():
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    config = ResidualCompletionConfig(enabled=True, num_batches=2)
    references = _references(source, source_ft, target, data, config)
    with pytest.raises(ValueError, match="requires mode='direct_target'"):
        complete_residuals_direct(
            target, target_base_sd, references, layout, data, config=config, device="cpu",
        )


# --------------------------------------------------------------------------
# 4. The glue: gamma semantics and the absence of transport
# --------------------------------------------------------------------------


def _complete(config, transported_delta):
    source, source_ft, target, data, layout, _prepared, target_base_sd, _delta = _fixture()
    references = _references(source, source_ft, target, data, config)
    return _maybe_complete_target_residual_task_vector(
        config=config,
        references=references,
        # Any use of ``prepared`` would be a transport read; None makes that fatal.
        prepared=None,
        layout=layout,
        target_model=target,
        target_base_sd=target_base_sd,
        transported_delta=transported_delta,
        target_loader=data,
        device="cpu",
    )


def test_direct_mode_never_reads_a_fitted_transport():
    """``prepared=None`` would raise inside ``projection_transforms``."""
    completed, diagnostics = _complete(_direct_config(), {})
    assert diagnostics is not None and len(diagnostics) == 2
    assert completed, "the direct arm must return a task vector of its own"


def test_gamma_zero_is_an_exact_native_target_base_control():
    completed, diagnostics = _complete(_direct_config(strength=0.0), {})
    assert diagnostics is not None and len(diagnostics) == 2, "the fit still runs at gamma=0"
    assert completed, "the zero task vector must still carry its keys"
    for value in completed.values():
        assert torch.count_nonzero(value) == 0


def test_gamma_scales_the_direct_task_vector_linearly():
    half, _ = _complete(_direct_config(strength=0.5), {})
    full, _ = _complete(_direct_config(strength=1.0), {})
    for key, value in full.items():
        torch.testing.assert_close(half[key], 0.5 * value, rtol=1e-5, atol=1e-7)


def test_direct_mode_refuses_a_non_empty_transported_vector():
    source, source_ft, target, data, layout, _prepared, target_base_sd, baseline_delta = _fixture()
    config = _direct_config()
    references = _references(source, source_ft, target, data, config)
    with pytest.raises(ValueError, match="requires an empty transported task vector"):
        _maybe_complete_target_residual_task_vector(
            config=config, references=references, prepared=None, layout=layout,
            target_model=target, target_base_sd=target_base_sd,
            transported_delta=baseline_delta, target_loader=data, device="cpu",
        )


def test_scale_completion_keeps_gamma_semantics_shared_with_the_transport_arm():
    correction = {"a": torch.ones(3)}
    baseline = {"a": torch.zeros(3)}
    torch.testing.assert_close(scale_completion(baseline, correction, 0.25)["a"], torch.full((3,), 0.25))
