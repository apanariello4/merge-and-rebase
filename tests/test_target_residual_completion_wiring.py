"""Integration tests for ARIADNE proposal-1 (target residual completion) wiring.

The math (``target_residual_completion.py``) and model-side orchestration
(``target_informed_runtime.py``) are independently unit-tested elsewhere and
verified against a Kronecker reference. What was missing was the glue in
``vision_rebase.py`` that (1) captures native reference banks before block
extension resizes the source model and (2) completes the transported task
vector's inserted ``c_proj`` keys after transport is fitted. These tests
exercise that glue -- ``_maybe_capture_target_residual_references`` and
``_maybe_complete_target_residual_task_vector`` -- directly, the same way
``tests/test_vision_rebase_theseus.py`` unit-tests ``_build_rebase_prepared``
without booting the full CLI pipeline.

Hash-level equality (rather than ``==`` on floats) is used throughout because
CLAUDE.md requires provenance/determinism to be testable at hash granularity,
matching the pattern in
``tests/test_brace_rebase_extension_spread_mod_20260909.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import resolve_block_extension_config, run_block_extension
from merge_and_rebase.eval.target_residual_completion import ResidualCompletionConfig
from merge_and_rebase.eval.vision_rebase import (
    _maybe_capture_target_residual_references,
    _maybe_complete_target_residual_task_vector,
    _state_dict_sha256,
)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 2)), ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(width * 2, width)),
        ]))
        self.ls_2 = nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Attention(nn.Module):
    """Minimal residual-writing attention stand-in for identity insertion."""

    def __init__(self, width):
        super().__init__()
        self.out_proj = nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class Visual(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = nn.Linear(4, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class Model(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _loader():
    return DataLoader(TensorDataset(torch.randn(6, 5, 4), torch.arange(6)), batch_size=2, shuffle=False)


def _fixture():
    """Source depth 2 (un-resized), target depth 4 (already doubled), as ARIADNE requires."""
    torch.manual_seed(12)
    source = Model(3, 2).eval()
    target = Model(5, 4).eval()
    source_ft = deepcopy(source)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.2)
        source_ft.visual.transformer.resblocks[1].mlp.c_proj.weight.sub_(0.15)
    data = _loader()
    layout = {"inserted_blocks": [{"position": 1, "source_orig_idx": 0}, {"position": 3, "source_orig_idx": 1}]}
    transforms_by_key = {}
    for pos in (1, 3):
        key = f"transformer.resblocks.{pos}.mlp.c_proj.weight"
        transforms_by_key[key] = SimpleNamespace(
            kind="weight",
            t_in=torch.linalg.qr(torch.randn(10, 6)).Q.T,
            t_out=torch.linalg.qr(torch.randn(5, 3)).Q.T,
        )
    prepared = {"transforms_by_key": transforms_by_key}
    target_base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    torch.manual_seed(99)
    baseline_delta = {k: 0.05 * torch.randn_like(v) for k, v in target_base_sd.items()}
    return source, source_ft, target, data, layout, prepared, target_base_sd, baseline_delta


def _run(config: ResidualCompletionConfig, seed: int = 0):
    source, source_ft, target, data, layout, prepared, target_base_sd, baseline_delta = _fixture()
    references = _maybe_capture_target_residual_references(
        config=config,
        source_base_model=source,
        source_ft_model=source_ft,
        target_model=target,
        source_loader=data,
        target_loader=data,
        seed=seed,
        device="cpu",
    )
    completed, diagnostics = _maybe_complete_target_residual_task_vector(
        config=config,
        references=references,
        prepared=prepared,
        layout=layout,
        target_model=target,
        target_base_sd=target_base_sd,
        transported_delta=baseline_delta,
        target_loader=data,
        device="cpu",
    )
    return baseline_delta, completed, diagnostics


_INSERTED_KEYS = {
    "visual.transformer.resblocks.1.mlp.c_proj.weight",
    "visual.transformer.resblocks.1.mlp.c_proj.bias",
    "visual.transformer.resblocks.3.mlp.c_proj.weight",
    "visual.transformer.resblocks.3.mlp.c_proj.bias",
}


def _diagnostics_equal(diag_a, diag_b):
    """Compare per-block diagnostics dicts, hash-level for the tensor entry.

    ``diag`` now carries ``bias_correction`` (a tensor), so a plain ``==`` on
    the dicts raises "Boolean value of Tensor with more than one value is
    ambiguous". Every other value is a plain Python scalar and is compared
    with ``==`` as before.
    """
    if len(diag_a) != len(diag_b):
        return False
    for a, b in zip(diag_a, diag_b, strict=True):
        if set(a) != set(b):
            return False
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
                if not (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)):
                    return False
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


def test_disabled_is_a_pure_noop_and_never_calls_into_the_math():
    """enabled=false must be byte-identical to the pre-existing path.

    ``_maybe_capture_target_residual_references`` returns ``None`` without
    touching the models, and ``_maybe_complete_target_residual_task_vector``
    returns the exact same ``transported_delta`` object -- so a state-dict
    hash comparison is trivially satisfied by construction, not by chance.
    """
    config = ResidualCompletionConfig(enabled=False)
    baseline_delta, completed, diagnostics = _run(config)
    assert completed is baseline_delta
    assert diagnostics is None
    assert _state_dict_sha256(completed) == _state_dict_sha256(baseline_delta)


def test_identity_initialization_p1_wires_capture_layout_transport_and_completion():
    """The exploratory arm captures before identity extension and completes after transport.

    This is intentionally a small mocked transport: the fitted projection maps
    stand in for Theseus/BiCo, while the real reference capture, BRACE layout,
    projection lookup, and completion code all run in their production order.
    """
    source, source_ft, target, data, _layout, _prepared, target_base_sd, baseline_delta = _fixture()
    _, config = resolve_block_extension_config(
        {
            "block_extension_enabled": True,
            "block_extension_params": {
                "target_layers_total": 4,
                "n_batches_act": 2,
                "skip_correction": True,
                "lmc_mode": "shared",
                "inserted_block_mode": "residual_identity",
                "target_residual_completion": {
                    "enabled": True,
                    "num_batches": 3,
                    "ridge_relative": 0.05,
                    "strength": 1.0,
                },
            },
        }
    )
    references = _maybe_capture_target_residual_references(
        config=config.target_residual_completion,
        source_base_model=source,
        source_ft_model=source_ft,
        target_model=target,
        source_loader=data,
        target_loader=data,
        seed=0,
        device="cpu",
    )
    assert references is not None

    layout = {}
    run_block_extension(
        source_base_model=source,
        source_ft_model=source_ft,
        calibration_loader=data,
        target_layers_total=4,
        config=config,
        device="cpu",
        layout_out=layout,
    )
    assert len(source.visual.transformer.resblocks) == 4
    assert [row["position"] for row in layout["inserted_blocks"]] == [1, 3]

    transforms = {}
    for position in (1, 3):
        transforms[f"transformer.resblocks.{position}.mlp.c_proj.weight"] = SimpleNamespace(
            kind="weight",
            t_in=torch.linalg.qr(torch.randn(10, 6)).Q.T,
            t_out=torch.linalg.qr(torch.randn(5, 3)).Q.T,
        )
    prepared = {"transforms_by_key": transforms}
    completed, diagnostics = _maybe_complete_target_residual_task_vector(
        config=config.target_residual_completion,
        references=references,
        prepared=prepared,
        layout=layout,
        target_model=target,
        target_base_sd=target_base_sd,
        transported_delta=baseline_delta,
        target_loader=data,
        device="cpu",
    )
    assert diagnostics is not None and [row["position"] for row in diagnostics] == [1, 3]
    changed = {key for key in baseline_delta if not torch.equal(completed[key], baseline_delta[key])}
    assert changed
    assert all("c_proj" in key for key in changed)


def test_enabled_with_zero_strength_is_a_true_null_ablation():
    """gamma=1 fitting still runs, but strength=0.0 must reproduce baseline exactly."""
    config = ResidualCompletionConfig(enabled=True, ridge_relative=0.05, num_batches=3, strength=0.0)
    baseline_delta, completed, diagnostics = _run(config)
    assert diagnostics is not None and len(diagnostics) == 2
    assert set(completed) == set(baseline_delta)
    assert _state_dict_sha256(completed) == _state_dict_sha256(baseline_delta)
    for key in baseline_delta:
        torch.testing.assert_close(completed[key], baseline_delta[key], rtol=0, atol=0)


def test_enabled_with_positive_strength_changes_only_inserted_c_proj_keys():
    config = ResidualCompletionConfig(enabled=True, ridge_relative=0.05, num_batches=3, strength=0.7)
    baseline_delta, completed, diagnostics = _run(config)
    assert diagnostics is not None and len(diagnostics) == 2

    changed = {k for k in baseline_delta if not torch.equal(completed[k], baseline_delta[k])}
    assert changed, "expected the completion to change at least one task-vector key"
    assert changed <= _INSERTED_KEYS, f"completion touched keys outside the inserted c_proj layers: {changed}"
    assert changed == _INSERTED_KEYS, "expected both inserted blocks to receive a nonzero correction"
    assert _state_dict_sha256(completed) != _state_dict_sha256(baseline_delta)


def _fixture_with_mean_shift():
    """Same fixture, but with a deliberately nonzero-mean fine-tuning residual.

    Shifting the fine-tuned source's ``c_proj`` bias by a constant (rather
    than only its weight, as ``_fixture`` does) gives the boundary residual a
    nonzero mean, so the exact-form affine intercept (Eq. 8-11) has something
    unambiguous to fit and the reduced form (no intercept term) has no way to
    reproduce it.
    """
    source, source_ft, target, data, layout, prepared, target_base_sd, baseline_delta = _fixture()
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.bias.add_(0.5)
        source_ft.visual.transformer.resblocks[1].mlp.c_proj.bias.add_(-0.3)
    return source, source_ft, target, data, layout, prepared, target_base_sd, baseline_delta


def _run_with(source, source_ft, target, data, layout, prepared, target_base_sd, baseline_delta, config, seed=0):
    references = _maybe_capture_target_residual_references(
        config=config, source_base_model=source, source_ft_model=source_ft, target_model=target,
        source_loader=data, target_loader=data, seed=seed, device="cpu",
    )
    completed, diagnostics = _maybe_complete_target_residual_task_vector(
        config=config, references=references, prepared=prepared, layout=layout,
        target_model=target, target_base_sd=target_base_sd, transported_delta=baseline_delta,
        target_loader=data, device="cpu",
    )
    return baseline_delta, completed, diagnostics


_BIAS_KEYS = {
    "visual.transformer.resblocks.1.mlp.c_proj.bias",
    "visual.transformer.resblocks.3.mlp.c_proj.bias",
}


def test_exact_form_applies_a_nonzero_intercept_that_the_reduced_form_discards():
    """The exact-form intercept must actually reach the task vector, not be silently reduced.

    With a deliberately nonzero-mean residual, ``exact_form=True`` must
    produce a nonzero bias delta at the inserted ``c_proj`` layers, while
    ``exact_form=False`` (no intercept term, ``beta`` fixed at zero by
    construction in ``ResidualSufficientStatistics.solve``) must leave those
    bias keys exactly at the baseline value.
    """
    fixture = _fixture_with_mean_shift()
    baseline_delta = fixture[-1]

    exact_config = ResidualCompletionConfig(
        enabled=True, ridge_relative=0.05, num_batches=3, strength=1.0, exact_form=True
    )
    _, completed_exact, _ = _run_with(*fixture, exact_config, seed=3)
    for key in _BIAS_KEYS:
        assert not torch.equal(completed_exact[key], baseline_delta[key]), (
            f"expected the exact-form intercept to change {key}"
        )

    reduced_config = ResidualCompletionConfig(
        enabled=True, ridge_relative=0.05, num_batches=3, strength=1.0, exact_form=False
    )
    _, completed_reduced, _ = _run_with(*fixture, reduced_config, seed=3)
    for key in _BIAS_KEYS:
        assert torch.equal(completed_reduced[key], baseline_delta[key]), (
            f"expected the reduced form to leave {key} at the baseline value"
        )


def test_completion_is_deterministic_given_a_fixed_seed():
    config = ResidualCompletionConfig(enabled=True, ridge_relative=0.05, num_batches=3, strength=0.7)
    _, completed_a, diag_a = _run(config, seed=7)
    _, completed_b, diag_b = _run(config, seed=7)
    assert _state_dict_sha256(completed_a) == _state_dict_sha256(completed_b)
    assert _diagnostics_equal(diag_a, diag_b)


def test_different_seeds_select_different_calibration_and_can_change_the_result():
    config = ResidualCompletionConfig(enabled=True, ridge_relative=0.05, num_batches=2, strength=0.7)
    _, completed_a, _ = _run(config, seed=1)
    _, completed_b, _ = _run(config, seed=2)
    # Not a hard requirement of the method, but pins down that the seed
    # actually reaches calibration sampling instead of being silently dropped.
    assert _state_dict_sha256(completed_a) != _state_dict_sha256(completed_b)


def test_target_base_weights_are_never_mutated_by_completion():
    config = ResidualCompletionConfig(enabled=True, ridge_relative=0.05, num_batches=3, strength=0.7)
    source, source_ft, target, data, layout, prepared, target_base_sd, baseline_delta = _fixture()
    references = _maybe_capture_target_residual_references(
        config=config, source_base_model=source, source_ft_model=source_ft, target_model=target,
        source_loader=data, target_loader=data, seed=0, device="cpu",
    )
    pre_hash = _state_dict_sha256(target.state_dict())
    _maybe_complete_target_residual_task_vector(
        config=config, references=references, prepared=prepared, layout=layout,
        target_model=target, target_base_sd=target_base_sd, transported_delta=baseline_delta,
        target_loader=data, device="cpu",
    )
    assert _state_dict_sha256(target.state_dict()) == pre_hash


@pytest.mark.parametrize("enabled", [False, True])
def test_capture_is_skipped_entirely_when_disabled(enabled):
    """No forward pass over the source/target models happens unless enabled."""
    source, source_ft, target, data, _layout, _prepared, _base, _delta = _fixture()
    calls = []
    original = source.encode_image
    source.encode_image = lambda x: (calls.append(1) or original(x))
    config = ResidualCompletionConfig(enabled=enabled)
    _maybe_capture_target_residual_references(
        config=config, source_base_model=source, source_ft_model=source_ft, target_model=target,
        source_loader=data, target_loader=data, seed=0, device="cpu",
    )
    assert bool(calls) == enabled
