"""Synthetic-scale correctness tests for `direct_residual.fit_direct_residual`.

Covers extend, shrink and same-arch depth pairings against tiny CLIP-shaped
synthetic models (the same fixture pattern `tests/test_direct_target_p1.py`
uses, duplicated per this suite's no-cross-test-import convention).
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    _aligned,
    _rows,
    capture_paired_boundary_activations,
    capture_tokens,
    centered_rectangular_procrustes,
    compute_desired_effects,
    fit_direct_residual,
    fit_sequential_source_endpoints,
    parse_direct_residual_config,
)
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing


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
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=config,
        device="cpu",
    )
    return corrections, diagnostics, target_base, target_base_sd


def test_sequential_endpoints_fit_on_mounted_base_and_restore_model():
    source_base, source_ft, target, data, pairing, target_sd = _setup(2, 4)
    config = parse_direct_residual_config({
        "endpoint_construction": "sequential_source_endpoints",
        "components": ["mlp.c_proj"],
        "num_batches": 3,
        "ridge_estimator": "empirical_bayes",
    })
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target, data, data, pairing,
        num_batches=3, seed=config.seed, device="cpu",
    )
    before = {k: v.clone() for k, v in target.state_dict().items()}
    sequential, rows, diagnostics = fit_sequential_source_endpoints(
        target, target_sd, captured, pairing, config=config, device="cpu",
    )
    ordinary, _ = fit_direct_residual(
        target, target_sd, captured, compute_desired_effects(captured, pairing), pairing,
        config=config, device="cpu",
    )
    assert sequential.keys() == ordinary.keys()
    assert {row["endpoint_stage"] for row in rows} == {"pretrained", "finetuned"}
    assert diagnostics["pretrained_correction_norm"] > 0
    assert diagnostics["task_vector_norm"] > 0
    assert any(not torch.allclose(sequential[k], ordinary[k], atol=1e-5, rtol=1e-5) for k in sequential)
    changed_ft = dict(captured)
    changed_ft["source_ft_outputs"] = {
        i: [batch + 0.5 for batch in batches]
        for i, batches in captured["source_ft_outputs"].items()
    }
    _, changed_rows, changed_diagnostics = fit_sequential_source_endpoints(
        target, target_sd, changed_ft, pairing, config=config, device="cpu",
    )
    assert diagnostics["pretrained_correction_norm"] == changed_diagnostics["pretrained_correction_norm"]
    for original, changed in zip(
        (r for r in rows if r["endpoint_stage"] == "pretrained"),
        (r for r in changed_rows if r["endpoint_stage"] == "pretrained"),
        strict=True,
    ):
        assert original["correction_norm"] == changed["correction_norm"]
    for key, value in before.items():
        assert torch.equal(target.state_dict()[key], value), key


def test_sequential_delta_on_synthesized_base_zero_update_and_native_control():
    source_base, source_ft, target, data, pairing, target_sd = _setup(2, 4)
    source_ft.load_state_dict(source_base.state_dict(), strict=True)
    config = parse_direct_residual_config({
        "endpoint_construction": "sequential_delta_on_synthesized_base",
        "components": ["mlp.c_proj"],
        "num_batches": 3,
        "ridge_estimator": "empirical_bayes",
    })
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target, data, data, pairing,
        num_batches=3, seed=config.seed, device="cpu",
    )
    before = {k: v.clone() for k, v in target.state_dict().items()}
    zero_vector, rows, diagnostics = fit_sequential_source_endpoints(
        target, target_sd, captured, pairing, config=config, device="cpu",
    )
    assert diagnostics["pretrained_correction_norm"] > 0
    assert diagnostics["task_vector_norm"] == 0.0
    assert {row["endpoint_stage"] for row in rows} == {"pretrained", "finetuned"}
    assert all(torch.count_nonzero(value) == 0 for value in zero_vector.values())
    for key, value in before.items():
        assert torch.equal(target.state_dict()[key], value), key

    # With a nonzero source update, the synthesized-base design defines a
    # distinct hypothesis from the ordinary native-delta fit.
    source_ft = _tuned_copy(source_base, seed=91)
    captured["source_ft_outputs"] = capture_paired_boundary_activations(
        source_base, source_ft, target, data, data, pairing,
        num_batches=3, seed=config.seed, device="cpu",
    )["source_ft_outputs"]
    synthesized, _rows, synthesized_diagnostics = fit_sequential_source_endpoints(
        target, target_sd, captured, pairing, config=config, device="cpu",
    )
    native, _ = fit_direct_residual(
        target, target_sd, captured, compute_desired_effects(captured, pairing), pairing,
        config=config, device="cpu",
    )
    assert synthesized_diagnostics["task_vector_norm"] > 0
    assert any(not torch.allclose(synthesized[k], native[k], atol=1e-5, rtol=1e-5) for k in synthesized)


def test_matched_synthesized_design_endpoint_subtraction_matches_delta_fit():
    source_base, source_ft, target, data, pairing, target_sd = _setup(2, 4)
    config = parse_direct_residual_config({
        "endpoint_construction": "sequential_delta_on_synthesized_base",
        "components": ["mlp.c_proj"],
        "num_batches": 3,
        "ridge_estimator": "empirical_bayes",
    })
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target, data, data, pairing,
        num_batches=3, seed=config.seed, device="cpu",
    )
    mapped_base = {}
    mapped_ft = {}
    base_desired = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        native = captured["target_base_outputs_by_position"][j]
        s0 = _aligned(captured["source_base_outputs"][i], native)
        s1 = _aligned(captured["source_ft_outputs"][i], native)
        q, mu_s, mu_t = centered_rectangular_procrustes(_rows(s0).double(), _rows(native).double())
        q, mu_s, mu_t = q.float(), mu_s.float(), mu_t.float()
        mapped_base[j] = [(b - mu_s) @ q + mu_t for b in s0]
        mapped_ft[j] = [(f - mu_s) @ q + mu_t for f in s1]
        base_desired[j] = [m - t for m, t in zip(mapped_base[j], native, strict=True)]

    base_correction, _ = fit_direct_residual(
        target, target_sd, captured, base_desired, pairing, config=config, device="cpu",
    )
    synthesized = {k: v.clone() for k, v in target_sd.items()}
    for key, correction in base_correction.items():
        synthesized[key] += correction.to(synthesized[key])
    target.load_state_dict(synthesized, strict=True)
    requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    raw = capture_tokens(target, captured["target_batches"], requests, "cpu")
    synthesized_outputs = {int(j): batches for j, batches in raw.items()}
    target.load_state_dict(target_sd, strict=True)
    stage2_captured = dict(captured)
    stage2_captured["target_base_outputs_by_position"] = synthesized_outputs
    endpoint_base = {
        j: [m - h for m, h in zip(mapped_base[j], synthesized_outputs[j], strict=True)]
        for j in range(pairing.target_depth)
    }
    endpoint_ft = {
        j: [m - h for m, h in zip(mapped_ft[j], synthesized_outputs[j], strict=True)]
        for j in range(pairing.target_depth)
    }
    source_delta = {
        j: [ft - base for base, ft in zip(mapped_base[j], mapped_ft[j], strict=True)]
        for j in range(pairing.target_depth)
    }
    correction_base, _ = fit_direct_residual(
        target, synthesized, stage2_captured, endpoint_base, pairing, config=config, device="cpu",
    )
    correction_ft, _ = fit_direct_residual(
        target, synthesized, stage2_captured, endpoint_ft, pairing, config=config, device="cpu",
    )
    correction_delta, _ = fit_direct_residual(
        target, synthesized, stage2_captured, source_delta, pairing, config=config, device="cpu",
    )
    for key in correction_delta:
        assert torch.allclose(
            correction_ft[key] - correction_base[key], correction_delta[key], atol=2e-5, rtol=2e-5,
        ), key


REGIMES = [
    pytest.param(2, 4, id="extend"),
    pytest.param(4, 2, id="shrink"),
    pytest.param(3, 3, id="same_arch"),
]


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_relative_residual_before_is_always_one(source_depth, target_depth):
    """Direct Residual is always 'independent' by construction: no cascade
    mounts a correction before the next position's fit, so every position
    sees E_j == D_j exactly (relative_residual_before == 1.0), not just the
    first one -- unlike ARIADNE's target_scope='all' path, which can only
    guarantee this for its very first fitted position."""
    _corrections, diagnostics, _model, _sd = _fit(source_depth, target_depth)
    for row in diagnostics:
        assert row["relative_residual_before"] == pytest.approx(1.0, rel=1e-4), row["position"]


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_strength_zero_reproduces_native_target_base_exactly(source_depth, target_depth):
    """scale_completion(gamma=0) is a full state-dict no-op; prove the fitted
    corrections never touch anything outside their own weight/bias keys by
    diffing the FULL state dict, not a components whitelist."""
    corrections, _diagnostics, target_model, target_base_sd = _fit(source_depth, target_depth)
    from merge_and_rebase.eval.target_informed_runtime import scale_completion

    completed_at_zero = scale_completion(target_base_sd, corrections, 0)
    assert set(completed_at_zero) == set(target_base_sd)
    for key, value in target_base_sd.items():
        assert torch.equal(completed_at_zero[key], value), key


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_target_corrections_only_touch_configured_components(source_depth, target_depth):
    corrections, _diagnostics, _model, target_base_sd = _fit(source_depth, target_depth)
    for key in corrections:
        assert (".mlp.c_proj." in key) or (".attn.out_proj." in key), key
        assert corrections[key].shape == target_base_sd[key].shape


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_target_model_is_restored_after_fit(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(source_depth, target_depth)
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)
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
    desired = compute_desired_effects(captured, pairing)
    pre_state = {k: v.clone() for k, v in target_base.state_dict().items()}
    fit_direct_residual(target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu")
    post_state = target_base.state_dict()
    for key, value in pre_state.items():
        assert torch.equal(post_state[key], value), key
