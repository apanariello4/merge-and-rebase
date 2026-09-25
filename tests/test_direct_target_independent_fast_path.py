"""``cascade_order="independent"`` is fit from one shared forward sweep.

Both ``complete_residuals_direct`` (ARIADNE's ``target_scope in {"inserted",
"all"}`` path) and ``direct_residual.fit_direct_residual`` call the shared,
private ``_fit_direct_target_position`` once per position, and internally
once per ``(position, component)`` pair -- a separate ``capture_tokens``
calibration sweep every time -- even though under ``cascade_order=
"independent"`` the target model is never mutated between fits (the only
mount site is unconditionally skipped in that mode). ``_fit_
all_positions_independent`` fixes this: it builds one combined
``capture_tokens`` request spanning every ``(position, component)`` pair and
solves each one from the shared banks.

This module pins two claims at synthetic scale (cheap, no GPU, runs in CI):

1.  Bit-identity: fitting through ``_fit_all_positions_independent`` produces
    the exact same corrections and diagnostics (float32/float64 bit-for-bit,
    modulo the harmless extra pristine-effect assertions it performs) as the
    historical per-``(position, component)``-capture loop through
    ``_fit_direct_target_position``, given the same inputs.
2.  Forward-pass-count reduction: the number of ``capture_tokens`` calls (each
    one sweep over every calibration batch) drops from
    ``len(positions) * len(components)`` to exactly ``1``.

Per CLAUDE.md's determinism convention, equivalence is checked at hash/exact
level rather than "looks close", and this module is self-contained -- no
cross-test imports -- matching the rest of this suite.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval import target_informed_runtime as runtime
from merge_and_rebase.eval.target_informed_runtime import (
    _fit_all_positions_independent,
    _fit_direct_target_position,
    capture_tokens,
)
from merge_and_rebase.eval.target_residual_completion import (
    ResidualCompletionConfig,
    order_components,
)


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
        self.ls_1 = torch.nn.Identity()
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.ls_1(self.attn(x)) + self.ls_2(self.mlp(x))


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


def _loader(n=8, batch_size=2):
    return DataLoader(TensorDataset(torch.randn(n, 5, 4), torch.arange(n)), batch_size=batch_size, shuffle=False)


def _fixture(depth=4, width=6):
    """A small multi-position, multi-component synthetic ViT stand-in."""
    torch.manual_seed(7)
    target = _Model(width, depth).eval()
    target_base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    positions = list(range(depth))
    source_coordinates = {pos: float(pos) + 0.5 for pos in positions}

    # Fabricate "desired effect" and "native target boundary output" batches
    # exactly as the real captured references would supply them: fixed,
    # independent of anything the fit does, matching batch-by-batch what a
    # real capture_paired_boundary_activations / desired-effect computation
    # produces.
    batches = list(_loader())
    desired_batches: dict[int, list[torch.Tensor]] = {}
    target_output_batches: dict[int, list[torch.Tensor]] = {}
    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        raw_out = capture_tokens(target, batches, {str(p): (p, "boundary") for p in positions}, "cpu")
    for pos in positions:
        target_output_batches[pos] = [tensor.clone() for tensor in raw_out[str(pos)]]
        desired_batches[pos] = [
            base + 0.1 * torch.randn(base.shape, generator=gen) for base in target_output_batches[pos]
        ]
    return target, target_base_sd, positions, source_coordinates, desired_batches, target_output_batches, batches


def _config(**overrides):
    params = {
        "enabled": True,
        "mode": "direct_target",
        "ridge_relative": 0.05,
        "num_batches": 4,
        "strength": 1.0,
        "cascade_order": "independent",
        "components": ("attn.out_proj", "mlp.c_proj"),
    }
    params.update(overrides)
    return ResidualCompletionConfig(**params)


def _old_per_pair_capture_fit(
    target,
    target_base_sd,
    positions,
    source_coordinates,
    desired_batches,
    target_output_batches,
    batches,
    components,
    config,
    device,
):
    """The historical behaviour: one `_fit_direct_target_position` call per
    position, each internally issuing one `capture_tokens` sweep per
    component. Kept here, unchanged, purely as the ground truth this test
    compares the new fast path against."""
    current_state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
    target.load_state_dict(current_state, strict=True)
    results = {}
    try:
        for pos in positions:
            position_corrections, block_rows = _fit_direct_target_position(
                target,
                current_state,
                pos,
                source_coordinates[pos],
                desired_batches[pos],
                target_output_batches[pos],
                batches,
                components,
                config,
                device,
                assert_pristine_effect=True,
            )
            results[pos] = (position_corrections, block_rows)
    finally:
        target.load_state_dict(target_base_sd, strict=True)
    return results


def test_fast_path_is_bit_identical_to_the_per_pair_capture_loop():
    target, target_base_sd, positions, source_coordinates, desired_batches, target_output_batches, batches = _fixture()
    config = _config()
    components = order_components(config.components)

    old_target = deepcopy(target)
    old_results = _old_per_pair_capture_fit(
        old_target,
        target_base_sd,
        positions,
        source_coordinates,
        desired_batches,
        target_output_batches,
        batches,
        components,
        config,
        "cpu",
    )

    new_target = deepcopy(target)
    current_state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
    new_target.load_state_dict(current_state, strict=True)
    new_results = _fit_all_positions_independent(
        new_target,
        current_state,
        positions,
        source_coordinates,
        desired_batches,
        target_output_batches,
        batches,
        components,
        config,
        "cpu",
    )

    assert set(old_results) == set(new_results) == set(positions)
    for pos in positions:
        old_corrections, old_rows = old_results[pos]
        new_corrections, new_rows = new_results[pos]
        assert set(old_corrections) == set(new_corrections)
        for key in old_corrections:
            torch.testing.assert_close(old_corrections[key], new_corrections[key], rtol=0, atol=0)
        assert len(old_rows) == len(new_rows)
        for old_row, new_row in zip(old_rows, new_rows, strict=True):
            assert set(old_row) == set(new_row)
            for field in old_row:
                old_value, new_value = old_row[field], new_row[field]
                if isinstance(old_value, torch.Tensor):
                    torch.testing.assert_close(old_value, new_value, rtol=0, atol=0)
                elif isinstance(old_value, float):
                    assert old_value == new_value, field
                else:
                    assert old_value == new_value, field


def test_forward_pass_count_drops_to_one_shared_sweep(monkeypatch):
    target, target_base_sd, positions, source_coordinates, desired_batches, target_output_batches, batches = _fixture()
    config = _config()
    components = order_components(config.components)

    calls = {"n": 0}
    real_capture_tokens = runtime.capture_tokens

    def counting_capture_tokens(*args, **kwargs):
        calls["n"] += 1
        return real_capture_tokens(*args, **kwargs)

    monkeypatch.setattr(runtime, "capture_tokens", counting_capture_tokens)

    old_target = deepcopy(target)
    old_current_state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
    old_target.load_state_dict(old_current_state, strict=True)
    for pos in positions:
        runtime._fit_direct_target_position(
            old_target,
            old_current_state,
            pos,
            source_coordinates[pos],
            desired_batches[pos],
            target_output_batches[pos],
            batches,
            components,
            config,
            "cpu",
            assert_pristine_effect=True,
        )
    old_calls = calls["n"]
    expected_old_calls = len(positions) * len(components)
    assert old_calls == expected_old_calls

    calls["n"] = 0
    new_target = deepcopy(target)
    new_current_state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
    new_target.load_state_dict(new_current_state, strict=True)
    runtime._fit_all_positions_independent(
        new_target,
        new_current_state,
        positions,
        source_coordinates,
        desired_batches,
        target_output_batches,
        batches,
        components,
        config,
        "cpu",
    )
    new_calls = calls["n"]
    assert new_calls == 1
    assert old_calls > new_calls
    # 4 positions x 2 components = 8 old sweeps collapsed into 1.
    assert old_calls == 8 and new_calls == 1


def _perturbed_banks(target_output_batches, scale):
    gen = torch.Generator().manual_seed(5)
    return {
        pos: [t * (1.0 + scale * torch.randn(t.shape, generator=gen)) for t in banks]
        for pos, banks in target_output_batches.items()
    }


def _fit_both_paths(banks):
    target, target_base_sd, positions, source_coordinates, desired_batches, _, batches = _fixture()
    config = _config()
    components = order_components(config.components)
    per_position_target = deepcopy(target)
    state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
    per_position_target.load_state_dict(state, strict=True)
    _fit_direct_target_position(
        per_position_target, state, positions[0], source_coordinates[positions[0]],
        desired_batches[positions[0]], banks[positions[0]], batches, components, config, "cpu",
        assert_pristine_effect=True,
    )
    shared_target = deepcopy(target)
    state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
    shared_target.load_state_dict(state, strict=True)
    _fit_all_positions_independent(
        shared_target, state, positions, source_coordinates, desired_batches, banks, batches,
        components, config, "cpu",
    )


def test_pristine_guard_tolerates_rounding_level_reference_drift():
    # A materialized zero bias can change the bf16 GEMM kernel, so the reference
    # bank and the pre-fit capture can differ by an ulp (Qwen2.5-0.5B down_proj
    # at 512 rows: ~3e-6 relative energy).
    _, _, _, _, _, target_output_batches, _ = _fixture()
    _fit_both_paths(_perturbed_banks(target_output_batches, 4e-3))


def test_pristine_guard_still_rejects_a_stale_reference_bank():
    _, _, _, _, _, target_output_batches, _ = _fixture()
    stale = _perturbed_banks(target_output_batches, 0.5)
    try:
        _fit_both_paths(stale)
    except RuntimeError as exc:
        assert "not the native base" in str(exc)
    else:
        raise AssertionError("a stale reference bank passed the pristine-effect guard")
