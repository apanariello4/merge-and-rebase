"""Tests for Direct Residual's memory-bounded ("streaming") activation-storage path.

`prepare_direct_residual_streaming`/`fit_direct_residual_streaming` accumulate
per-position Procrustes cross-covariance and ridge sufficient statistics
batch-by-batch instead of holding every batch's boundary activation bank
resident (see `capture_paired_boundary_activations`/`fit_direct_residual`),
so host RAM stays O(1) in `num_batches`. Correctness is pinned against the
resident path bit-for-bit (extend/shrink/same_arch), plus chunk-invariance,
config validation and pristine-target-model detection.

Fixture duplicated from `tests/test_direct_residual_fit.py` per this suite's
no-cross-test-import convention.
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
    _StreamingCrossCovariance,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
    fit_direct_residual_streaming,
    parse_direct_residual_config,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.eval.target_residual_completion import centered_rectangular_procrustes
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


def _setup(source_depth, target_depth, width=5, seed=11, same_object=False):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = source_base if (same_object and source_depth == target_depth) else _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _fit_resident(source_base, source_ft, target_base, data, pairing, target_base_sd, config):
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    return fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    ), captured


def _fit_streaming(source_base, source_ft, target_base, data, pairing, target_base_sd, config):
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
    )
    result = fit_direct_residual_streaming(
        target_base, target_base_sd, source_base, source_ft, prepared, pairing, config=config, device="cpu",
    )
    return result, prepared


REGIMES = [
    pytest.param(2, 4, id="extend"),
    pytest.param(4, 2, id="shrink"),
    pytest.param(3, 3, id="same_arch"),
]


def _drift(corr_a, corr_b):
    """Max abs/rel diff between two correction dicts, plus the fraction of
    float32 entries that are not bit-identical. Returned as a plain dict so
    callers (this test module and scratchpad/streaming_drift.py) can print or
    assert on it without depending on pytest's capsys."""
    max_abs = 0.0
    max_rel = 0.0
    total = 0
    mismatched = 0
    for key in corr_a:
        a, b = corr_a[key].double(), corr_b[key].double()
        diff = (a - b).abs()
        denom = a.abs().max().item() + 1e-12
        max_abs = max(max_abs, diff.max().item())
        max_rel = max(max_rel, (diff.max().item() / denom))
        total += a.numel()
        mismatched += int((corr_a[key] != corr_b[key]).sum().item())
    return {
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "fraction_not_bit_identical": (mismatched / total) if total else 0.0,
    }


# ---- (a) _StreamingCrossCovariance vs centered_rectangular_procrustes -------


def test_streaming_cross_covariance_matches_batched_procrustes():
    torch.manual_seed(0)
    n, d_src, d_tgt = 37, 6, 8
    x = torch.randn(n, d_src, dtype=torch.float64)
    y = torch.randn(n, d_tgt, dtype=torch.float64)
    q_ref, _mu_s, _mu_t = centered_rectangular_procrustes(x, y)

    # Unequal, non-uniform batch split.
    splits = [5, 1, 13, 4, 14]
    assert sum(splits) == n
    acc = _StreamingCrossCovariance()
    start = 0
    for s in splits:
        acc.update(x[start : start + s], y[start : start + s])
        start += s

    src_mean = x.mean(dim=0)
    tgt_mean = y.mean(dim=0)
    cross_ref = (x - src_mean).T @ (y - tgt_mean)

    assert torch.allclose(acc.cross(), cross_ref, rtol=1e-12, atol=1e-12)
    from merge_and_rebase.eval.target_residual_completion import _procrustes_from_cross

    q_stream = _procrustes_from_cross(acc.cross())
    assert torch.allclose(q_stream, q_ref, rtol=1e-12, atol=1e-12)


# ---- (b) streaming vs resident end-to-end -----------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_matches_resident_end_to_end(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(source_depth, target_depth)
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    stream_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming")

    (corr_resident, diag_resident), captured = _fit_resident(
        source_base, source_ft, target_base, data, pairing, target_base_sd, config
    )
    (corr_stream, diag_stream), prepared = _fit_streaming(
        source_base, source_ft, target_base, data, pairing, target_base_sd, stream_config
    )

    assert prepared["calibration"] == captured["calibration"]
    assert set(corr_resident) == set(corr_stream)
    for key in corr_resident:
        assert torch.allclose(corr_resident[key], corr_stream[key], rtol=1e-5, atol=1e-7), key

    by_pos_component_resident = {(r["position"], r["component"]): r for r in diag_resident}
    by_pos_component_stream = {(r["position"], r["component"]): r for r in diag_stream}
    assert set(by_pos_component_resident) == set(by_pos_component_stream)
    numeric_fields = [
        "desired_norm", "effect_before_norm", "relative_residual_before", "relative_residual_after",
        "correction_rank", "n_rows", "ridge", "residual_norm_before", "residual_norm_after",
        "correction_norm", "bias_norm",
    ]
    for key, row_r in by_pos_component_resident.items():
        row_s = by_pos_component_stream[key]
        for field in numeric_fields:
            assert row_r[field] == pytest.approx(row_s[field], rel=1e-5), (key, field)

    drift = _drift(corr_resident, corr_stream)
    print(f"[{source_depth}->{target_depth}] drift={drift}")
    assert drift["max_abs_diff"] < 1e-6
    assert drift["max_rel_diff"] < 1e-4


# ---- (c) chunk invariance ----------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_position_chunk_is_bitwise_invariant(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(source_depth, target_depth)
    config_all = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming")
    config_chunked = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, activation_storage="streaming", streaming_position_chunk=1
    )

    (corr_all, _diag_all), _ = _fit_streaming(source_base, source_ft, target_base, data, pairing, target_base_sd, config_all)
    (corr_chunked, _diag_chunked), _ = _fit_streaming(
        source_base, source_ft, target_base, data, pairing, target_base_sd, config_chunked
    )

    assert set(corr_all) == set(corr_chunked)
    for key in corr_all:
        assert torch.equal(corr_all[key], corr_chunked[key]), key

    def _sha(d):
        h = hashlib.sha256()
        for key in sorted(d):
            h.update(key.encode())
            h.update(d[key].numpy().tobytes())
        return h.hexdigest()

    assert _sha(corr_all) == _sha(corr_chunked)


# ---- (d) config validation ---------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"component_target": "output_local", "components": ("attn.out_proj",)},
        {"block_split": "backfit"},
        {"realization_diagnostics": True},
    ],
)
def test_streaming_rejects_unsupported_config_combinations(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({"activation_storage": "streaming", **overrides})


@pytest.mark.parametrize("bad_value", ["bogus", 1, 1.0, None])
def test_activation_storage_rejects_bad_values(bad_value):
    with pytest.raises((ValueError, TypeError)):
        parse_direct_residual_config({"activation_storage": bad_value})


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, True, "3"])
def test_streaming_position_chunk_rejects_bad_values(bad_value):
    with pytest.raises((ValueError, TypeError)):
        parse_direct_residual_config({"streaming_position_chunk": bad_value})


def test_streaming_position_chunk_accepts_none_and_positive_int():
    cfg = parse_direct_residual_config({"streaming_position_chunk": None})
    assert cfg.streaming_position_chunk is None
    cfg = parse_direct_residual_config({"streaming_position_chunk": 3})
    assert cfg.streaming_position_chunk == 3


# ---- (e) pristine-target-model detection between pass A and pass B ----------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_mutated_target_between_passes_raises_pristine_error(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(source_depth, target_depth)
    stream_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming")
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing,
        num_batches=stream_config.num_batches, seed=stream_config.seed, device="cpu",
    )
    mutated_target_base_sd = {k: v.clone() for k, v in target_base_sd.items()}
    any_key = next(iter(mutated_target_base_sd))
    mutated_target_base_sd[any_key] = mutated_target_base_sd[any_key] + 1.0

    with pytest.raises(RuntimeError, match="not the native base"):
        fit_direct_residual_streaming(
            target_base, mutated_target_base_sd, source_base, source_ft, prepared, pairing,
            config=stream_config, device="cpu",
        )
    # Restored even on the raised path.
    for key, value in target_base_sd.items():
        assert torch.equal(target_base.state_dict()[key], value), key


# ---- (f) defaults -------------------------------------------------------------


def test_default_activation_storage_is_resident():
    assert DirectResidualConfig().activation_storage == "resident"
    assert parse_direct_residual_config(None).activation_storage == "resident"
    assert parse_direct_residual_config({}).activation_storage == "resident"


# ---- (g) same-object source_base/target ---------------------------------------


def test_streaming_matches_resident_when_source_and_target_share_object():
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(3, 3, same_object=True)
    assert source_base is target_base
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    stream_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming")

    (corr_resident, _diag_resident), _captured = _fit_resident(
        source_base, source_ft, target_base, data, pairing, target_base_sd, config
    )
    (corr_stream, _diag_stream), _prepared = _fit_streaming(
        source_base, source_ft, target_base, data, pairing, target_base_sd, stream_config
    )

    assert set(corr_resident) == set(corr_stream)
    for key in corr_resident:
        assert torch.allclose(corr_resident[key], corr_stream[key], rtol=1e-5, atol=1e-7), key
    # The shared source_base/target object must come back pristine.
    for key, value in target_base_sd.items():
        assert torch.equal(target_base.state_dict()[key], value), key
