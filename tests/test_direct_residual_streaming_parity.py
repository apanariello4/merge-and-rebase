"""Streaming (activation_storage='streaming') parity with the resident path for the
features added after the streaming path: gradient-fitted Procrustes, the transported-
endpoint target, alignment diagnostics, realization diagnostics and tv_scaling.

Every comparison is resident vs streaming on identical calibration samples. The streaming
path accumulates sums per batch and fits Procrustes from a Chan-merged cross-covariance,
so results agree to floating-point tolerance, not bit-for-bit; chunking positions must
not change anything (bitwise). block_split='backfit'/'joint' stay rejected under streaming.

Fixture duplicated from `tests/test_direct_residual_streaming.py` per this suite's
no-cross-test-import convention.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    apply_tv_scaling,
    capture_paired_boundary_activations,
    compute_alignment_diagnostics,
    compute_alignment_diagnostics_streaming,
    compute_desired_effects,
    fit_direct_residual,
    fit_direct_residual_streaming,
    measure_direct_residual_realization,
    measure_streaming_realization_for,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.eval.target_informed_runtime import (
    ResidualSufficientStatistics,
    _realized_pred_sq_from_stats,
    capture_block_gradients,
    iter_capture_block_gradients,
)
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.utils.cost_accounting import PhaseCostRecorder, recording


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


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n) % N_CLASSES), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _recipe(width, seed):
    text = torch.randn(N_CLASSES, width, generator=torch.Generator().manual_seed(seed))

    def recipe(model, batch):
        images, labels = batch
        logits = model.encode_image(images) @ text.T
        return F.cross_entropy(logits, labels.long()), []

    return recipe


def _setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


REGIMES = [
    pytest.param(2, 4, id="extend"),
    pytest.param(4, 2, id="shrink"),
    pytest.param(3, 3, id="same_arch"),
]


def _resident(setup, config, recipes=(None, None)):
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
        procrustes_source=config.procrustes_source, source_recipe=recipes[0], target_recipe=recipes[1],
    )
    diag_out: dict = {}
    desired = compute_desired_effects(
        captured, pairing, residual_target=config.residual_target,
        procrustes_source=config.procrustes_source,
        alignment_map=config.alignment_map,
        alignment_row_weighting=config.alignment_row_weighting,
        diagnostics_out=diag_out,
    )
    corr, rows = fit_direct_residual(target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu")
    return corr, rows, captured, desired, diag_out


def _streaming(setup, config, recipes=(None, None)):
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
        source_ft_model=source_ft,
        procrustes_source=config.procrustes_source, source_recipe=recipes[0], target_recipe=recipes[1],
        alignment_map=config.alignment_map, alignment_row_weighting=config.alignment_row_weighting,
    )
    corr, rows = fit_direct_residual_streaming(
        target_base, target_base_sd, source_base, source_ft, prepared, pairing, config=config, device="cpu",
    )
    return corr, rows, prepared


def _assert_corr_close(a, b):
    assert set(a) == set(b)
    for key in a:
        assert torch.allclose(a[key], b[key], rtol=1e-5, atol=1e-7), key


def _state_hash(model):
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---- gradient capture generator ---------------------------------------------


def test_iter_capture_block_gradients_matches_capture_block_gradients():
    source_base, _sf, _tb, data, _pairing, _sd = _setup(3, 3)
    batches = list(data)
    requests = {str(i): i for i in range(3)}
    recipe = _recipe(5, 0)
    before = _state_hash(source_base)
    flags = {n: p.requires_grad for n, p in source_base.named_parameters()}
    ref = capture_block_gradients(source_base, batches, requests, recipe, "cpu")
    streamed = list(iter_capture_block_gradients(source_base, batches, requests, recipe, "cpu"))
    assert len(streamed) == len(batches)
    for key in requests:
        for k, batch_values in enumerate(streamed):
            assert torch.equal(batch_values[key], ref[key][k])
    assert _state_hash(source_base) == before
    assert {n: p.requires_grad for n, p in source_base.named_parameters()} == flags
    assert all(p.grad is None for p in source_base.parameters())


# ---- gradient Procrustes ------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_gradient_procrustes_matches_resident(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    recipes = (_recipe(5, 1), _recipe(5, 2))
    base = dict(num_batches=3, ridge_relative=0.05, procrustes_source="gradient")
    corr_r, _rows_r, _captured, _desired, diag_r = _resident(setup, DirectResidualConfig(**base), recipes)
    corr_s, _rows_s, prepared = _streaming(setup, DirectResidualConfig(**base, activation_storage="streaming"), recipes)
    _assert_corr_close(corr_r, corr_s)
    for j, row in diag_r.items():
        row_s = prepared["procrustes_diagnostics"][j]
        assert row_s["procrustes_source"] == "gradient"
        assert row_s["procrustes_rank"] == row["procrustes_rank"]
        assert row_s["activation_gradient_procrustes_overlap"] == pytest.approx(
            row["activation_gradient_procrustes_overlap"], rel=1e-6
        )


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
@pytest.mark.parametrize(
    "alignment_map,alignment_row_weighting",
    [("polar", "cls_balanced"), ("polar", "delta_magnitude"), ("ridge", "uniform")],
)
def test_alignment_variants_match_resident_streaming(source_depth, target_depth, alignment_map, alignment_row_weighting):
    setup = _setup(source_depth, target_depth, seed=71)
    common = dict(
        num_batches=3, ridge_relative=0.05,
        alignment_map=alignment_map, alignment_row_weighting=alignment_row_weighting,
    )
    corr_r, _rows_r, captured, _desired_r, _diag = _resident(setup, DirectResidualConfig(**common))
    corr_s, _rows_s, prepared = _streaming(setup, DirectResidualConfig(**common))
    _assert_corr_close(corr_r, corr_s)
    from merge_and_rebase.eval.direct_residual import _aligned, _fit_activation_map

    for j in range(setup[4].target_depth):
        i = setup[4].pairing[j]
        x = _aligned(captured["source_base_outputs"][i], captured["target_base_outputs_by_position"][j])
        ft = _aligned(captured["source_ft_outputs"][i], captured["target_base_outputs_by_position"][j])
        y = captured["target_base_outputs_by_position"][j]
        q, *_ = _fit_activation_map(
            x, y, ft, alignment_map=alignment_map, row_weighting=alignment_row_weighting
        )
        assert torch.allclose(q.float(), prepared["q_by_position"][j], rtol=1e-5, atol=1e-6)
    assert prepared["alignment_map"] == alignment_map


def test_delta_magnitude_weights_are_per_image_scale_normalized_and_grid_aligned():
    from merge_and_rebase.eval.direct_residual import _fit_activation_map
    from merge_and_rebase.eval.target_informed_runtime import _interp_2d_tokens

    torch.manual_seed(171)
    # float64 so the invariance is checked exactly, not up to float32 rounding of the rescaled delta.
    f64 = torch.float64
    source = [torch.randn(3, 5, 4, dtype=f64), torch.randn(2, 5, 4, dtype=f64)]
    ft = [x + torch.randn_like(x) * torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=f64) for x in source]
    target = [torch.randn(3, 7, 3, dtype=f64), torch.randn(2, 7, 3, dtype=f64)]
    aligned = [_interp_2d_tokens(x, 7) for x in source]
    aligned_ft = [_interp_2d_tokens(x, 7) for x in ft]
    q1, *_ = _fit_activation_map(aligned, target, aligned_ft, row_weighting="delta_magnitude")
    # Per-image normalization makes positive rescaling of an image's whole
    # delta irrelevant.
    changed = [x.clone() for x in aligned_ft]
    changed[0][0] = aligned[0][0] + 13.0 * (changed[0][0] - aligned[0][0])
    q2, *_ = _fit_activation_map(aligned, target, changed, row_weighting="delta_magnitude")
    assert torch.allclose(q1, q2, atol=1e-10, rtol=1e-10)
    zero_delta = [x.clone() for x in aligned_ft]
    zero_delta[1][0] = aligned[1][0]
    q_zero, *_ = _fit_activation_map(aligned, target, zero_delta, row_weighting="delta_magnitude")
    assert torch.isfinite(q_zero).all()


# ---- endpoint target ----------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_endpoint_target_matches_resident(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    base = dict(num_batches=3, ridge_relative=0.05, residual_target="transported_endpoint")
    corr_r, rows_r, *_ = _resident(setup, DirectResidualConfig(**base))
    corr_s, rows_s, _ = _streaming(setup, DirectResidualConfig(**base, activation_storage="streaming"))
    _assert_corr_close(corr_r, corr_s)
    by_r = {(r["position"], r["component"]): r for r in rows_r}
    by_s = {(r["position"], r["component"]): r for r in rows_s}
    for key, row in by_r.items():
        assert by_s[key]["desired_norm"] == pytest.approx(row["desired_norm"], rel=1e-5)


@pytest.mark.parametrize("overrides", [{"residual_target": "transported_endpoint"}, {"procrustes_source": "gradient"}])
def test_streaming_new_features_are_chunk_invariant_bitwise(overrides):
    setup = _setup(2, 4)
    recipes = (_recipe(5, 1), _recipe(5, 2)) if overrides.get("procrustes_source") == "gradient" else (None, None)
    base = dict(num_batches=3, ridge_relative=0.05, activation_storage="streaming", **overrides)
    corr_all, *_ = _streaming(setup, DirectResidualConfig(**base), recipes)
    corr_one, *_ = _streaming(setup, DirectResidualConfig(**base, streaming_position_chunk=1), recipes)
    assert set(corr_all) == set(corr_one)
    for key in corr_all:
        assert torch.equal(corr_all[key], corr_one[key]), key


# ---- alignment diagnostics ----------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_alignment_diagnostics_match_resident(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, _data, pairing, target_base_sd = setup
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    _corr, _rows, captured, _desired, _ = _resident(setup, config)
    _corr_s, _rows_s, prepared = _streaming(setup, DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming"))
    before = _state_hash(target_base)
    diag_r = compute_alignment_diagnostics(captured, pairing)
    diag_s = compute_alignment_diagnostics_streaming(
        source_base, source_ft, target_base, target_base_sd, prepared, pairing, device="cpu"
    )
    assert _state_hash(target_base) == before
    assert set(diag_r) == set(diag_s)
    for j, row in diag_r.items():
        assert set(row) == set(diag_s[j])
        for field, value in row.items():
            assert diag_s[j][field] == pytest.approx(value, rel=1e-5, abs=1e-9), (j, field)


# ---- realization diagnostics ---------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_realization_measurement_matches_resident(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, _data, pairing, target_base_sd = setup
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05)
    stream_config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming")
    corr, _rows, captured, desired, _ = _resident(setup, config)
    _corr_s, _rows_s, prepared = _streaming(setup, stream_config)
    positions = list(range(pairing.target_depth))
    rows_r = measure_direct_residual_realization(
        target_base, target_base_sd, corr, positions, captured["target_batches"],
        captured["target_base_outputs_by_position"], desired, device="cpu",
        components=("attn.out_proj", "mlp.c_proj"),
    )
    before = _state_hash(target_base)
    measure = measure_streaming_realization_for(
        target_base, target_base_sd, source_base, source_ft, prepared, pairing, config=stream_config, device="cpu"
    )
    rows_s = measure(corr)
    assert _state_hash(target_base) == before
    for j, row in rows_r.items():
        for field in ("desired_norm", "joint_delta_norm", "block_realized_target_error",
                      "joint_delta_norm_over_desired", "component_interaction_error"):
            assert rows_s[j][field] == pytest.approx(row[field], rel=1e-5, abs=1e-9), (j, field)
        for c, value in row["per_family_delta_norm_over_desired"].items():
            assert rows_s[j]["per_family_delta_norm_over_desired"][c] == pytest.approx(value, rel=1e-5)


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_per_component_realization_fields_match_resident(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    base = dict(num_batches=3, ridge_relative=0.05, realization_diagnostics=True)
    _corr_r, rows_r, *_ = _resident(setup, DirectResidualConfig(**base))
    _corr_s, rows_s, _ = _streaming(setup, DirectResidualConfig(**base, activation_storage="streaming"))
    by_r = {(r["position"], r["component"]): r for r in rows_r}
    by_s = {(r["position"], r["component"]): r for r in rows_s}
    for key, row in by_r.items():
        for field in ("fit_relative_residual", "target_norm", "update_norm", "relative_update_norm",
                      "realized_target_norm_ratio"):
            assert by_s[key][field] == pytest.approx(row[field], rel=1e-5, abs=1e-9), (key, field)


def test_realized_pred_sq_from_stats_equals_bank_sum():
    torch.manual_seed(3)
    stats = ResidualSufficientStatistics()
    banks = [torch.randn(7, 4) for _ in range(3)]
    e_out = torch.diag(torch.rand(5) + 0.5)
    for h in banks:
        stats.update(h, torch.randn(7, 5), None, e_out)
    correction = torch.randn(5, 4)
    bias = torch.randn(5)
    bank_sum = 0.0
    for h in banks:
        pred = (h.double() @ correction.double().T + bias.double()) @ e_out.double()
        bank_sum += float((pred**2).sum().item())
    assert _realized_pred_sq_from_stats(stats, correction, bias, e_out) == pytest.approx(bank_sum, rel=1e-10)


# ---- tv_scaling ------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["global", "per_block"])
@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_tv_scaling_matches_resident(mode, source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    source_base, source_ft, target_base, _data, pairing, target_base_sd = setup
    base = dict(num_batches=3, ridge_relative=0.05, tv_scaling=mode, tv_scaling_iters=2)
    config = DirectResidualConfig(**base)
    stream_config = DirectResidualConfig(**base, activation_storage="streaming")
    corr, _rows, captured, desired, _ = _resident(setup, config)
    corr_s, _rows_s, prepared = _streaming(setup, stream_config)
    positions = list(range(pairing.target_depth))
    final_r, diag_r = apply_tv_scaling(
        target_base, target_base_sd, corr, positions, captured, desired, config=config, device="cpu"
    )
    measure = measure_streaming_realization_for(
        target_base, target_base_sd, source_base, source_ft, prepared, pairing, config=stream_config, device="cpu"
    )
    final_s, diag_s = apply_tv_scaling(
        target_base, target_base_sd, corr_s, positions, None, None, config=stream_config, device="cpu",
        measure_fn=measure,
    )
    _assert_corr_close(final_r, final_s)
    if mode == "global":
        assert diag_s["c"] == pytest.approx(diag_r["c"], rel=1e-5)
    else:
        for trace_r, trace_s in zip(diag_r["s_traces"], diag_s["s_traces"], strict=True):
            for j in positions:
                assert trace_s[j] == pytest.approx(trace_r[j], rel=1e-5)


# ---- cost accounting ---------------------------------------------------------


@pytest.mark.parametrize("overrides", [{}, {"procrustes_source": "gradient"}, {"residual_target": "transported_endpoint"}])
def test_streaming_fit_bit_identical_under_cost_recording(overrides):
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming", **overrides)
    recipes = (_recipe(5, 1), _recipe(5, 2)) if overrides.get("procrustes_source") == "gradient" else (None, None)
    plain, _rows, _prepared = _streaming(_setup(2, 4), config, recipes)
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder):
        recorded, _rows, _prepared = _streaming(_setup(2, 4), config, recipes)
    assert set(plain) == set(recorded)
    for key in plain:
        assert torch.equal(plain[key], recorded[key]), key
    phases = recorder.summary()["phases"]
    assert phases["activation_collection"]["segments"] > 0
    assert phases["transformation"]["segments"] > 0
