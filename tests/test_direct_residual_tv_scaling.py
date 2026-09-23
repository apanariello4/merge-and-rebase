"""Tests for Direct Residual's ``tv_scaling`` ablation (``direct_residual.py``'s
``apply_tv_scaling``).

Motivation (see ``direct_residual.DirectResidualConfig.tv_scaling`` docstring):
block_boundary O+D realizes, with all of a unit-strength task vector tau
mounted, a block-output change ``||delta T_j||`` that can be far from
``||D_j||`` (``measure_direct_residual_realization``'s
``joint_delta_norm_over_desired``), growing with depth. ``tv_scaling``
rescales tau -- label-free, from activations only -- AFTER the fit but
BEFORE any alpha-search, to put alpha-search on a better-conditioned grid.

Per CLAUDE.md: ``tv_scaling="none"`` (the default) must leave the historical,
golden-hash-pinned path byte-identical -- this module never modifies
``fit_direct_residual`` itself, only adds a strictly-gated post-hoc step in
``apply_tv_scaling`` / ``vision_rebase._run_direct_residual_fit``. This module
has no cross-test imports (matching the convention of every other Direct
Residual test file in this repo).
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from open_clip.transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    apply_tv_scaling,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
    parse_direct_residual_config,
)
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")),
]


def _state_dict_sha256(d) -> str:
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ==========================================================================
# Fixture 1: standard SEQUENTIAL toy model (residual stream chains block to
# block), copied from test_direct_residual_component_coverage.py -- this
# module deliberately keeps no cross-test imports, per repo convention.
# Used for (a) golden-hash / no-op checks and (b) "global" mode.
# ==========================================================================


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


def _capture_and_fit(source_depth, target_depth, config, width=5, seed=11):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(
        source_depth, target_depth, width=width, seed=seed
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=config.num_batches, seed=config.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu",
    )
    return target_base, target_base_sd, captured, desired, pairing, corrections, diagnostics


def _brute_force_mounted_boundary(target_model, target_base_sd, delta, positions, target_batches):
    """Mount ``target_base_sd + delta`` and capture every position's
    block-boundary OUTPUT via a plain ``register_forward_hook`` replay over
    ``target_batches`` -- deliberately NOT ``measure_direct_residual_realization``
    or ``capture_tokens``, so tests (b)/(c) recompute r_j from raw tensors
    rather than re-calling the implementation's own measurement helper."""
    original = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    state = {k: v.clone() for k, v in target_base_sd.items()}
    for key, value in delta.items():
        state[key] = state[key] + value.to(state[key])
    target_model.load_state_dict(state, strict=True)
    outputs = {j: [] for j in positions}
    handles = []

    def make_hook(j):
        def hook(_module, _inputs, output):
            outputs[j].append(output.detach().clone())

        return hook

    for j in positions:
        handles.append(target_model.visual.transformer.resblocks[j].register_forward_hook(make_hook(j)))
    try:
        with torch.no_grad():
            for batch in target_batches:
                images = batch[0]
                target_model.encode_image(images)
    finally:
        for h in handles:
            h.remove()
        target_model.load_state_dict(original, strict=True)
    return outputs


def _brute_force_r_j(delta_outputs, base_outputs, desired, positions):
    r = {}
    for j in positions:
        num_sq = sum(
            float(((a - b).double() ** 2).sum().item())
            for a, b in zip(delta_outputs[j], base_outputs[j], strict=True)
        )
        den_sq = sum(float((d.double() ** 2).sum().item()) for d in desired[j])
        r[j] = (num_sq**0.5) / ((den_sq**0.5) + 1e-12)
    return r


# ==========================================================================
# (a) tv_scaling="none" is a strict no-op / golden hashes unaffected.
# ==========================================================================


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
@pytest.mark.parametrize("block_split", ["none", "backfit", "joint"])
def test_tv_scaling_none_is_exact_identity(source_depth, target_depth, block_split):
    """apply_tv_scaling(..., tv_scaling='none') must return the input
    dict's tensors byte-identical, for every block_split -- this is the
    no-op contract the historical, golden-hash-pinned path (see
    tests/test_direct_residual_component_coverage.py) relies on. Note
    fit_direct_residual itself is completely untouched by this feature
    (apply_tv_scaling is never called by it); this test only pins
    apply_tv_scaling's own no-op behaviour.
    """
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, block_split=block_split, backfit_max_iters=3)
    target_base, target_base_sd, captured, desired, pairing, corrections, _diag = _capture_and_fit(
        source_depth, target_depth, cfg
    )
    before_hash = _state_dict_sha256(corrections)
    final, diag = apply_tv_scaling(
        target_base, target_base_sd, corrections, list(range(pairing.target_depth)),
        captured, desired, config=cfg, device="cpu",
    )
    assert diag == {"mode": "none"}
    assert _state_dict_sha256(final) == before_hash
    for key in corrections:
        torch.testing.assert_close(final[key], corrections[key], rtol=0, atol=0)


def test_parser_rejects_tv_scaling_with_backfit_or_joint():
    for split in ("backfit", "joint"):
        with pytest.raises(ValueError, match="block_split"):
            parse_direct_residual_config({"tv_scaling": "global", "block_split": split})
        with pytest.raises(ValueError, match="block_split"):
            parse_direct_residual_config({"tv_scaling": "per_block", "block_split": split})


def test_parser_accepts_tv_scaling_values_only():
    assert parse_direct_residual_config({"tv_scaling": "none"}).tv_scaling == "none"
    assert parse_direct_residual_config({"tv_scaling": "global"}).tv_scaling == "global"
    assert parse_direct_residual_config({"tv_scaling": "per_block"}).tv_scaling == "per_block"
    with pytest.raises(ValueError, match="tv_scaling"):
        parse_direct_residual_config({"tv_scaling": "bogus"})


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_parser_rejects_bad_tv_scaling_iters(bad):
    with pytest.raises(ValueError, match="tv_scaling_iters"):
        parse_direct_residual_config({"tv_scaling_iters": bad})


def test_parser_accepts_positive_tv_scaling_iters():
    assert parse_direct_residual_config({"tv_scaling_iters": 7}).tv_scaling_iters == 7


# ==========================================================================
# (b) "global": tau_out == tau_in / c exactly; c matches a brute-force
#     independent recomputation of median_j(r_j) from raw captured tensors.
# ==========================================================================


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_tv_scaling_global_matches_brute_force_median(source_depth, target_depth):
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, tv_scaling="global")
    target_base, target_base_sd, captured, desired, pairing, corrections, _diag = _capture_and_fit(
        source_depth, target_depth, cfg
    )
    positions = list(range(pairing.target_depth))
    final, diag = apply_tv_scaling(
        target_base, target_base_sd, corrections, positions, captured, desired, config=cfg, device="cpu",
    )
    assert diag["mode"] == "global"

    # Brute-force: mount the UNSCALED corrections, replay target_batches by
    # hand with plain forward hooks, compute r_j from raw tensors, and take
    # the median independently of apply_tv_scaling's internal computation.
    delta_outputs = _brute_force_mounted_boundary(
        target_base, target_base_sd, corrections, positions, captured["target_batches"]
    )
    r_j = _brute_force_r_j(delta_outputs, captured["target_base_outputs_by_position"], desired, positions)
    values = sorted(r_j[j] for j in positions)
    n = len(values)
    c_expected = values[n // 2] if n % 2 == 1 else 0.5 * (values[n // 2 - 1] + values[n // 2])

    assert diag["c"] == pytest.approx(c_expected, rel=1e-9)
    for key in corrections:
        expected = corrections[key] / c_expected
        torch.testing.assert_close(final[key], expected, rtol=1e-6, atol=1e-9)
        # Exact (not merely close) elementwise equality against the
        # implementation's OWN c, per the "tau_out == tau_in / c EXACTLY"
        # requirement.
        torch.testing.assert_close(final[key], corrections[key] / diag["c"], rtol=0, atol=0)


def test_tv_scaling_global_restores_target_model_state():
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, tv_scaling="global")
    target_base, target_base_sd, captured, desired, pairing, corrections, _diag = _capture_and_fit(3, 3, cfg)
    before_hash = _state_dict_sha256(target_base.state_dict())
    apply_tv_scaling(
        target_base, target_base_sd, corrections, list(range(pairing.target_depth)),
        captured, desired, config=cfg, device="cpu",
    )
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash


# ==========================================================================
# (d) A linear toy model where blocks do NOT interact: each block reads the
# shared embedded input directly (not the previous block's output), so a
# correction at block k has zero effect on block j != k's boundary output.
# Every correction here is applied to a projection that is the LAST linear
# operation in its path (out_proj / c_proj), so each block's boundary output
# is also exactly AFFINE in a per-block scale factor s_j. Both properties
# together mean the Jacobi update s_j <- s_j / r_j^(0) is exact, not
# approximate: per_block must converge to r_j == 1 in exactly one iteration.
# ==========================================================================


class _IndependentVisual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x0 = self.input(images)
        # Parallel, not chained: every block transforms the SAME shared
        # embedding x0 independently, so block k's own weights never affect
        # block j != k's boundary output (no residual-stream chaining across
        # positions). The final scalar output (never used by Direct
        # Residual's boundary captures) just averages the blocks.
        outs = [block(x0) for block in self.transformer.resblocks]
        return torch.stack(outs, dim=0).mean(dim=0).mean(dim=1)


class _IndependentModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _IndependentVisual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _independent_setup(depth, width=5, seed=23):
    torch.manual_seed(seed)
    source_base = _Model(width, depth).eval()  # source may be ordinary/sequential; only the TARGET must be independent
    target_base = _IndependentModel(width, depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(depth, depth)  # same-arch: no span mixing
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def test_per_block_converges_in_one_iteration_when_blocks_do_not_interact():
    depth = 4
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, tv_scaling="per_block", tv_scaling_iters=3, seed=17,
    )
    source_base, source_ft, target_base, data, pairing, target_base_sd = _independent_setup(depth)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    positions = list(range(pairing.target_depth))
    final, diag = apply_tv_scaling(
        target_base, target_base_sd, corrections, positions, captured, desired, config=cfg, device="cpu",
    )
    assert diag["mode"] == "per_block"
    max_dev_trace = diag["max_dev_trace"]
    assert len(max_dev_trace) == 3
    # Before the loop, s_j = 1 everywhere -- iteration 0's own measurement IS
    # that pre-loop max deviation.
    pre_loop_max_dev = max_dev_trace[0]
    # After exactly one Jacobi update (the mounted combination measured at
    # the START of iteration 1 uses s_j found from iteration 0's r_j), the
    # ratio must be 1 to floating-point precision: no second update should be
    # needed given block independence + per-block affinity in the scale.
    assert max_dev_trace[1] < 1e-5, max_dev_trace
    assert max_dev_trace[1] < pre_loop_max_dev
    # A third (redundant) iteration must leave it converged (idempotent
    # fixed point), not regress.
    assert max_dev_trace[2] < 1e-5, max_dev_trace

    for j in positions:
        assert math.isfinite(diag["s_traces"][-1][j])
    for it_trace in diag["r_traces"]:
        for j in positions:
            if it_trace[j] is not None:
                assert math.isfinite(it_trace[j])

    # Brute-force recomputation of the FINAL r_j from raw tensors, matching
    # the implementation's own post_scaling_r_j.
    delta_outputs = _brute_force_mounted_boundary(
        target_base, target_base_sd, final, positions, captured["target_batches"]
    )
    r_j_bruteforce = _brute_force_r_j(delta_outputs, captured["target_base_outputs_by_position"], desired, positions)
    for j in positions:
        assert r_j_bruteforce[j] == pytest.approx(diag["post_scaling_r_j"][j], rel=1e-5, abs=1e-6)
        assert r_j_bruteforce[j] == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_per_block_restores_target_model_state():
    depth = 3
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, tv_scaling="per_block", tv_scaling_iters=2, seed=17)
    source_base, source_ft, target_base, data, pairing, target_base_sd = _independent_setup(depth)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu",
    )
    before_hash = _state_dict_sha256(target_base.state_dict())
    apply_tv_scaling(
        target_base, target_base_sd, corrections, list(range(pairing.target_depth)),
        captured, desired, config=cfg, device="cpu",
    )
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash


# ==========================================================================
# (c) "per_block" on a tiny real open_clip ViT -- fixtures adapted from
# tests/test_direct_residual_open_clip_integration.py (copied, not imported,
# per this repo's no-cross-test-import convention for Direct Residual tests).
# ==========================================================================


class _CLIPLike(torch.nn.Module):
    def __init__(self, visual: VisionTransformer):
        super().__init__()
        self.visual = visual

    def encode_image(self, x):
        return self.visual(x)


def _make_vit(*, image_size, patch_size, width, layers, heads, seed):
    torch.manual_seed(seed)
    vt = VisionTransformer(
        image_size=image_size, patch_size=patch_size, width=width, layers=layers, heads=heads,
        mlp_ratio=2.0, ls_init_value=None, output_dim=width, pool_type="tok",
    )
    return _CLIPLike(vt).eval()


def _tuned_vit_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


class _IdentityTensorDataset(TensorDataset):
    def __init__(self, images, sample_ids):
        super().__init__(images, torch.arange(len(images)))
        self.sample_ids = sample_ids


def _clip_loader(image_size, n=6, seed=0, sample_ids=None):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, image_size, image_size, generator=generator)
    ids = sample_ids if sample_ids is not None else [str(i) for i in range(n)]
    return DataLoader(_IdentityTensorDataset(images, ids), batch_size=2, shuffle=False)


def _open_clip_setup(seed=101):
    """A small extend pairing (2 -> 4 layers), matching the
    test_direct_residual_open_clip_integration.py "extend" direction."""
    source_spec = dict(image_size=16, patch_size=4, width=8, layers=2, heads=2)
    target_spec = dict(image_size=24, patch_size=4, width=12, layers=4, heads=3)
    source_base = _make_vit(seed=seed, **source_spec)
    target_base = _make_vit(seed=seed + 1, **target_spec)
    source_ft = _tuned_vit_copy(source_base, seed=seed + 2)
    shared_ids = [str(i) for i in range(6)]
    source_loader = _clip_loader(source_spec["image_size"], seed=seed + 3, sample_ids=shared_ids)
    target_loader = _clip_loader(target_spec["image_size"], seed=seed + 4, sample_ids=shared_ids)
    pairing = DiscreteLayerPairing.compute(source_spec["layers"], target_spec["layers"])
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd


@pytest.mark.parametrize("device", DEVICES)
def test_per_block_open_clip_integration(device):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        tv_scaling="per_block", tv_scaling_iters=4, seed=101,
    )
    source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd = _open_clip_setup()
    before_hash = _state_dict_sha256(target_base.state_dict())
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device=device,
    )
    desired = compute_desired_effects(captured, pairing)
    target_base.to(device)
    corrections, _diag = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device=device,
    )
    positions = list(range(pairing.target_depth))
    final, diag = apply_tv_scaling(
        target_base, target_base_sd, corrections, positions, captured, desired, config=cfg, device=device,
    )
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash, "target model must be restored after apply_tv_scaling"

    max_dev_trace = diag["max_dev_trace"]
    assert len(max_dev_trace) == 4
    assert max_dev_trace[-1] <= max_dev_trace[0], max_dev_trace  # report the full trace regardless of monotonicity
    for j in positions:
        assert math.isfinite(diag["s_traces"][-1][j])

    if device == "cpu":
        # Brute-force recomputation of the FINAL iteration's r_j from raw
        # captured tensors (not measure_direct_residual_realization).
        delta_outputs = _brute_force_mounted_boundary(
            target_base, target_base_sd, final, positions, captured["target_batches"]
        )
        r_j_bruteforce = _brute_force_r_j(
            delta_outputs, captured["target_base_outputs_by_position"], desired, positions
        )
        for j in positions:
            assert r_j_bruteforce[j] == pytest.approx(diag["post_scaling_r_j"][j], rel=1e-4, abs=1e-5)
