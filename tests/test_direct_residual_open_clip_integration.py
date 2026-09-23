"""End-to-end Direct Residual integration test against real ``open_clip``
``VisionTransformer`` instances (offline, random init, no pretrained weights
downloaded -- fully compatible with CLAUDE.md's "Offline compute" rule).

Unlike every other Direct Residual test in this repo (which uses a hand-built
toy ``_Model``/``_StockModel`` fixture, see e.g.
``tests/test_direct_residual_component_coverage.py``), this module builds the
actual ``open_clip.transformer.VisionTransformer`` class Direct Residual is
meant to run against in production, to catch anything the hand-built
fixtures' simplifications (single head, no patch embedding, no positional
embedding, no LayerNorm) could hide. ``ls_init_value=None`` gives
``nn.Identity`` LayerScale on both paths automatically -- the same
requirement ``_assert_layerscale_identity`` enforces for
``component_target='output_local'`` -- so no special-casing is needed to
satisfy it.

Source and target are given different ``width``/``layers`` (extend, shrink)
and different ``image_size``/``patch_size`` (so source and target native
token counts differ and the source-side token interpolation
(``_aligned``/``_interp_2d_tokens``) actually has work to do), plus one
same-architecture pair as a control.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
)
from merge_and_rebase.eval.target_residual_completion import order_components
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")),
]

_ALL_SIX = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj")


class _CLIPLike(torch.nn.Module):
    """Minimal ``open_clip``-shaped wrapper: only ``.visual`` matters to
    ``target_informed_runtime._encode_image``/``_VisionLayout``."""

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


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.mlp.c_fc.weight.add_(scale * torch.randn_like(block.mlp.c_fc.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
            block.attn.in_proj_weight.add_(scale * torch.randn_like(block.attn.in_proj_weight))
    return tuned


class _IdentityTensorDataset(TensorDataset):
    """A ``TensorDataset`` that also carries ``sample_ids``: ``paired_calibration``
    (``target_informed_runtime._dataset_identity``) requires that source and
    target loaders' datasets carry matching ``sample_ids`` whenever they are
    not literally the same Python object -- true here, since source and
    target use different native image resolutions and therefore need two
    separate tensors, unlike every other Direct Residual test fixture (same
    resolution on both sides, so they share one loader object)."""

    def __init__(self, images, sample_ids):
        super().__init__(images, torch.arange(len(images)))
        self.sample_ids = sample_ids


def _loader(image_size, n=6, seed=0, sample_ids=None):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, image_size, image_size, generator=generator)
    ids = sample_ids if sample_ids is not None else [str(i) for i in range(n)]
    return DataLoader(_IdentityTensorDataset(images, ids), batch_size=2, shuffle=False)


# --------------------------------------------------------------------------
# Three direction fixtures: extend (different token counts too), shrink
# (different token counts), same_arch (identical shapes/token counts).
# --------------------------------------------------------------------------

_DIRECTIONS = {
    "extend": dict(source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
                   target=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3)),
    "shrink": dict(source=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3),
                   target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2)),
    "same_arch": dict(source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
                       target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2)),
}


def _direction_setup(direction, seed=101):
    spec = _DIRECTIONS[direction]
    source_base = _make_vit(seed=seed, **spec["source"])
    target_base = _make_vit(seed=seed + 1, **spec["target"])
    source_ft = _tuned_copy(source_base, seed=seed + 2)
    shared_ids = [str(i) for i in range(6)]
    source_loader = _loader(spec["source"]["image_size"], seed=seed + 3, sample_ids=shared_ids)
    target_loader = _loader(spec["target"]["image_size"], seed=seed + 4, sample_ids=shared_ids)
    pairing = DiscreteLayerPairing.compute(spec["source"]["layers"], spec["target"]["layers"])
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd


def _state_dict_sha256(d) -> str:
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _fit(direction, config, device, component_inputs=()):
    source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd = _direction_setup(
        direction
    )
    before_hash = _state_dict_sha256(target_base.state_dict())
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=config.num_batches, seed=config.seed, device=device, component_inputs=component_inputs,
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device=device,
    )
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash, "target base model must be restored after fit_direct_residual"
    return corrections, diagnostics, target_base, target_base_sd


# --------------------------------------------------------------------------
# 1. Token counts genuinely differ for extend/shrink (interpolation exercised).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["extend", "shrink"])
def test_source_and_target_token_counts_differ(direction):
    spec = _DIRECTIONS[direction]
    source_tokens = (spec["source"]["image_size"] // spec["source"]["patch_size"]) ** 2 + 1
    target_tokens = (spec["target"]["image_size"] // spec["target"]["patch_size"]) ** 2 + 1
    assert source_tokens != target_tokens


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
@pytest.mark.parametrize("device", DEVICES)
def test_block_boundary_none_end_to_end_finite(direction, device):
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"))
    corrections, diagnostics, _model, sd = _fit(direction, cfg, device)
    touched = set(corrections)
    for key, value in corrections.items():
        assert torch.isfinite(value).all(), key
    # Untouched parameters absent: only out_proj/c_proj weight+bias keys.
    for key in touched:
        assert (".attn.out_proj." in key) or (".mlp.c_proj." in key), key
    assert diagnostics


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
@pytest.mark.parametrize("device", DEVICES)
def test_block_boundary_backfit_od_end_to_end_finite(direction, device):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        block_split="backfit", backfit_max_iters=3,
    )
    corrections, diagnostics, _model, _sd = _fit(direction, cfg, device)
    for value in corrections.values():
        assert torch.isfinite(value).all()
    for row in diagnostics:
        assert isinstance(row["backfit_n_sweeps"], int)


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_block_boundary_backfit_j_trace_is_non_increasing(direction):
    """The monotone safeguard's own guarantee (see
    target_informed_runtime._fit_block_boundary_backfit's docstring): J(Delta),
    measured on the real (un-linearized) local block replay every sweep, must
    never increase -- on a REAL open_clip block (real ln_2/GELU coupling, real
    LayerNorm/patch embedding), not only the hand-built toy fixture covered in
    test_direct_residual_backfit.py.
    """
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        block_split="backfit", backfit_max_iters=10, backfit_tol=1e-12,
    )
    _corrections, diagnostics, _model, _sd = _fit(direction, cfg, "cpu")
    for row in diagnostics:
        j_trace = row["backfit_j_trace"]
        assert len(j_trace) == row["backfit_n_sweeps"]
        assert all(math.isfinite(v) for v in j_trace), (direction, row["component"], j_trace)
        for prev, curr in zip(j_trace, j_trace[1:], strict=False):
            assert curr <= prev + 1e-6, (direction, row["component"], j_trace)
        assert row["backfit_round1_j"] == pytest.approx(j_trace[0])


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
@pytest.mark.parametrize("device", DEVICES)
def test_output_local_all_six_end_to_end_finite(direction, device):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target="output_local", components=_ALL_SIX,
    )
    corrections, diagnostics, _model, _sd = _fit(direction, cfg, device, component_inputs=order_components(_ALL_SIX))
    assert corrections
    for key, value in corrections.items():
        assert torch.isfinite(value).all(), key
    assert {row["component"] for row in diagnostics} == set(_ALL_SIX)

    # Packed in_proj: untouched rows must be exactly zero (each of q/k/v owns
    # a disjoint 1/3 row slice; here all three are requested, so nothing
    # should be exactly zero by omission, but the shape/dtype contract is
    # asserted directly).
    for key, value in corrections.items():
        if key.endswith("attn.in_proj_weight"):
            assert value.shape[0] % 3 == 0


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_output_local_untouched_qkv_rows_are_exactly_zero(direction):
    """Requesting only attn.v_proj must leave the q/k row-thirds of the
    packed in_proj_weight/bias correction exactly zero."""
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, component_target="output_local", components=("attn.v_proj",),
    )
    corrections, _diag, _model, _sd = _fit(direction, cfg, "cpu", component_inputs=("attn.v_proj",))
    for key, value in corrections.items():
        if key.endswith("attn.in_proj_weight") or key.endswith("attn.in_proj_bias"):
            d = value.shape[0] // 3
            q_slice = value[:d]
            k_slice = value[d : 2 * d]
            v_slice = value[2 * d :]
            assert torch.count_nonzero(q_slice) == 0, "q row-third must stay exactly zero"
            assert torch.count_nonzero(k_slice) == 0, "k row-third must stay exactly zero"
            assert torch.count_nonzero(v_slice) > 0, "v row-third is the one requested component"


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_output_local_independence_at_hash_level(direction):
    """Every component's correction in the ALL run bitwise equals its own
    single-component run (extends the toy-fixture version of this check in
    test_direct_residual_component_coverage.py to a real open_clip block)."""
    joint_corrections, _diag, _model, _sd = _fit(
        direction,
        DirectResidualConfig(num_batches=3, ridge_relative=0.05, component_target="output_local", components=_ALL_SIX),
        "cpu",
        component_inputs=order_components(_ALL_SIX),
    )
    for component in ("attn.out_proj", "mlp.c_fc", "mlp.c_proj"):
        solo_corrections, _diag2, _model2, _sd2 = _fit(
            direction,
            DirectResidualConfig(
                num_batches=3, ridge_relative=0.05, component_target="output_local", components=(component,)
            ),
            "cpu",
            component_inputs=(component,),
        )
        for key, value in solo_corrections.items():
            torch.testing.assert_close(joint_corrections[key], value, rtol=0, atol=0)
    for component, idx in (("attn.q_proj", 0), ("attn.k_proj", 1), ("attn.v_proj", 2)):
        solo_corrections, _diag2, _model2, _sd2 = _fit(
            direction,
            DirectResidualConfig(
                num_batches=3, ridge_relative=0.05, component_target="output_local", components=(component,)
            ),
            "cpu",
            component_inputs=(component,),
        )
        for key, value in solo_corrections.items():
            is_bias = key.endswith("in_proj_bias")
            d = value.shape[0] // 3
            rows = slice(idx * d, (idx + 1) * d)
            solo_slice = value[rows] if is_bias else value[rows, :]
            joint_slice = joint_corrections[key][rows] if is_bias else joint_corrections[key][rows, :]
            torch.testing.assert_close(joint_slice, solo_slice, rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_realization_diagnostics_finite_real_model(direction):
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        realization_diagnostics=True,
    )
    _corrections, diagnostics, _model, _sd = _fit(direction, cfg, "cpu")
    expected = {
        "fit_relative_residual", "target_norm", "update_norm", "relative_update_norm", "realized_target_norm_ratio",
    }
    for row in diagnostics:
        assert expected <= set(row)
        for key in expected:
            assert torch.isfinite(torch.tensor(float(row[key])))


# --------------------------------------------------------------------------
# 2. q/k/v recomputation matches F._in_projection_packed's own slicing, and
#    c_fc matches a direct forward hook -- both against the real
#    nn.MultiheadAttention / mlp.c_fc modules Direct Residual reads from.
# --------------------------------------------------------------------------


def test_qkv_recomputation_matches_in_projection_packed_on_real_block():
    torch.manual_seed(7)
    vt = VisionTransformer(
        image_size=16, patch_size=4, width=8, layers=1, heads=2, mlp_ratio=2.0,
        ls_init_value=None, output_dim=8, pool_type="tok",
    ).eval()
    block = vt.transformer.resblocks[0]
    x = torch.randn(3, 17, 8)  # (B, T, D) for width=8, image_size=16, patch=4 -> 16 patches + 1 cls
    normed = block.ln_1(x)

    with torch.no_grad():
        q_ref, k_ref, v_ref = F._in_projection_packed(normed, normed, normed, block.attn.in_proj_weight, block.attn.in_proj_bias)

    d = block.attn.embed_dim
    w = block.attn.in_proj_weight
    b = block.attn.in_proj_bias
    with torch.no_grad():
        q_ours = F.linear(normed, w[0:d], b[0:d])
        k_ours = F.linear(normed, w[d : 2 * d], b[d : 2 * d])
        v_ours = F.linear(normed, w[2 * d : 3 * d], b[2 * d : 3 * d])

    torch.testing.assert_close(q_ours, q_ref, rtol=0, atol=1e-6)
    torch.testing.assert_close(k_ours, k_ref, rtol=0, atol=1e-6)
    torch.testing.assert_close(v_ours, v_ref, rtol=0, atol=1e-6)


def test_c_fc_recomputation_matches_forward_hook_on_real_block():
    torch.manual_seed(7)
    vt = VisionTransformer(
        image_size=16, patch_size=4, width=8, layers=1, heads=2, mlp_ratio=2.0,
        ls_init_value=None, output_dim=8, pool_type="tok",
    ).eval()
    block = vt.transformer.resblocks[0]
    x = torch.randn(3, 17, 8)

    captured_input = {}
    captured_output = {}

    def hook(_m, inputs, value):
        captured_input["x"] = inputs[0].detach().clone()
        captured_output["y"] = value.detach().clone()

    handle = block.mlp.c_fc.register_forward_hook(hook)
    try:
        with torch.no_grad():
            vt(torch.randn(3, 3, 16, 16))
            _ = block(x)
    finally:
        handle.remove()

    with torch.no_grad():
        recomputed = F.linear(captured_input["x"], block.mlp.c_fc.weight, block.mlp.c_fc.bias)
    torch.testing.assert_close(recomputed, captured_output["y"], rtol=0, atol=1e-6)
