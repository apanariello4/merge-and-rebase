"""``VisionAdapter`` / ``DecoderAdapter`` reproduce the original extender methods tensor-for-tensor.

The adapters are not used by the extenders yet; these tests pin them against the old methods on the
golden tiny fixtures (OpenCLIP-style toy vision model, tiny Qwen2) so a later switch-over is a
verifiable no-op. They also pin the documented vision/decoder divergences (execution log B11/B12 and
the bias-scaling difference in ``dampen_block_output``), which must be preserved, not unified.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from merge_and_rebase.eval.block_extension import BlockExtender
from merge_and_rebase.eval.block_extension_llm import DecoderBlockExtender
from merge_and_rebase.rebase.block_extension.adapters import (
    ComponentSpec,
    DecoderAdapter,
    VisionAdapter,
)
from merge_and_rebase.rebase.model_families import infer_family
from tests.golden._hashing import deterministic_cpu
from tests.golden.test_release_golden_hashes import (
    _brace_vision_models,
    _class_loader,
    _llm_loader,
    _llm_source_pair,
)


@pytest.fixture(autouse=True)
def _det():
    with deterministic_cpu():
        yield


def _assert_same_state(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


def _vision():
    base, ft = _brace_vision_models(3)
    ext = BlockExtender(base, ft, "cpu", verbose=False, show_progress=False)
    return base, ft, ext, VisionAdapter()


def _decoder():
    base, ft = _llm_source_pair(2)
    family = infer_family(base)
    ext = DecoderBlockExtender(base, ft, family, "cpu", verbose=False, show_progress=False)
    return base, ft, ext, DecoderAdapter(family)


def _out_dim(adapter, block, spec) -> int:
    inner = adapter.inner_block(block)
    if isinstance(adapter, VisionAdapter):
        return {
            "ln_1": inner.ln_1.weight,
            "ln_2": inner.ln_2.weight,
            "q": inner.attn.in_proj_weight[: inner.attn.in_proj_weight.shape[0] // 3],
            "k": inner.attn.in_proj_weight[: inner.attn.in_proj_weight.shape[0] // 3],
            "v": inner.attn.in_proj_weight[: inner.attn.in_proj_weight.shape[0] // 3],
            "out_proj": inner.attn.out_proj.weight,
            "c_fc": inner.mlp.c_fc.weight,
            "c_proj": inner.mlp.c_proj.weight,
        }[spec.name].shape[0]
    name = spec.name
    if spec.kind == "norm_diag":
        return getattr(block, name).weight.shape[0]
    module = (
        getattr(block.self_attn, name)
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
        else getattr(block.mlp, name)
    )
    return module.weight.shape[0]


def _corrections(adapter, block, seed=5):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for spec in adapter.components:
        n = _out_dim(adapter, block, spec)
        W = torch.eye(n) + 0.1 * torch.randn(n, n, generator=gen)
        b = 0.1 * torch.randn(n, generator=gen)
        out[spec.name] = (W, b)
    return out


@pytest.mark.parametrize("make", [_vision, _decoder], ids=["vision", "decoder"])
def test_apply_correction_all_components_matches_apply_block_corrections(make):
    base, _ft, ext, adapter = make()
    model_old, model_new = deepcopy(base), deepcopy(base)
    pos = 1
    corrections = _corrections(adapter, adapter.layers(model_new)[pos])
    ext._apply_block_corrections(model_old, pos, corrections)
    block = adapter.layers(model_new)[pos]
    for spec in adapter.components:
        adapter.apply_correction(block, spec, *corrections[spec.name])
    _assert_same_state(model_old, model_new)
    assert not all(
        torch.equal(p, q) for p, q in zip(base.state_dict().values(), model_new.state_dict().values(), strict=True)
    )


@pytest.mark.parametrize("make", [_vision, _decoder], ids=["vision", "decoder"])
def test_apply_correction_each_component_alone(make):
    base, _ft, ext, adapter = make()
    corrections = _corrections(adapter, adapter.layers(base)[0], seed=9)
    for spec in adapter.components:
        model_old, model_new = deepcopy(base), deepcopy(base)
        ext._apply_block_corrections(model_old, 0, {spec.name: corrections[spec.name]})
        adapter.apply_correction(adapter.layers(model_new)[0], spec, *corrections[spec.name])
        _assert_same_state(model_old, model_new)


@pytest.mark.parametrize("make", [_vision, _decoder], ids=["vision", "decoder"])
def test_interpolate_block_weights_matches(make):
    base, ft, ext, adapter = make()
    model_old, model_new = deepcopy(base), deepcopy(base)
    ext._interpolate_block_weights(adapter.layers(model_old)[0], adapter.layers(ft)[1], 0.3)
    adapter.interpolate_block_weights(adapter.layers(model_new)[0], adapter.layers(ft)[1], 0.3)
    _assert_same_state(model_old, model_new)
    assert not all(
        torch.equal(p, q) for p, q in zip(base.state_dict().values(), model_new.state_dict().values(), strict=True)
    )


def test_vision_dampen_scales_weights_only():
    base, _ft, ext, adapter = _vision()
    model_old, model_new = deepcopy(base), deepcopy(base)
    ext._dampen_block_output(adapter.layers(model_old)[1], 0.5)
    adapter.dampen_block_output(adapter.layers(model_new)[1], 0.5)
    _assert_same_state(model_old, model_new)
    inner_new, inner_ref = adapter.layers(model_new)[1], adapter.layers(base)[1]
    assert torch.equal(inner_new.attn.out_proj.weight, inner_ref.attn.out_proj.weight * 0.5)
    # Divergence from the decoder: vision biases are left untouched.
    assert torch.equal(inner_new.attn.out_proj.bias, inner_ref.attn.out_proj.bias)
    assert torch.equal(inner_new.mlp.c_proj.bias, inner_ref.mlp.c_proj.bias)


def test_decoder_dampen_scales_weights_and_biases():
    base, _ft, ext, adapter = _decoder()
    for layer in adapter.layers(base):
        for lin in (layer.self_attn.o_proj, layer.mlp.down_proj):
            lin.bias = torch.nn.Parameter(torch.linspace(0.1, 1.0, lin.weight.shape[0]))
    model_old, model_new = deepcopy(base), deepcopy(base)
    ext._dampen_block_output(adapter.layers(model_old)[1], 0.5)
    adapter.dampen_block_output(adapter.layers(model_new)[1], 0.5)
    _assert_same_state(model_old, model_new)
    ref, new = adapter.layers(base)[1], adapter.layers(model_new)[1]
    assert torch.equal(new.self_attn.o_proj.bias, ref.self_attn.o_proj.bias * 0.5)
    assert torch.equal(new.mlp.down_proj.bias, ref.mlp.down_proj.bias * 0.5)


def test_vision_zero_output_projections_matches():
    base, _ft, ext, adapter = _vision()
    model_old, model_new = deepcopy(base), deepcopy(base)
    ext._zero_block_output_projections(adapter.layers(model_old)[1])
    adapter.zero_output_projections(adapter.layers(model_new)[1])
    _assert_same_state(model_old, model_new)
    assert float(adapter.layers(model_new)[1].mlp.c_proj.weight.abs().sum()) == 0.0


def test_decoder_has_no_zero_projection_baseline():
    _base, _ft, _ext, adapter = _decoder()
    assert not adapter.capabilities.zero_projection_baselines
    with pytest.raises(NotImplementedError):
        adapter.zero_output_projections(adapter.layers(_base)[0])


_VISION_CAPTURE_SPECS = [*VisionAdapter.components, ComponentSpec("attn", "attn_output", "linear")]
_DECODER_CAPTURE_SPECS = [*DecoderAdapter.components, ComponentSpec("attn", "attn_output", "linear")]


@pytest.mark.parametrize("spec", _VISION_CAPTURE_SPECS, ids=lambda s: s.name)
def test_vision_capture_component_matches(spec):
    base, _ft, ext, adapter = _vision()
    loader = _class_loader(n=16, in_dim=6, batch_size=4, seed=3)
    old_name = "attn" if spec.name == "out_proj" else spec.name
    old = ext._capture_component_output(deepcopy(base), 1, old_name, loader, 3)
    new = adapter.capture_component(deepcopy(base), 1, spec, loader, 3, "cpu")
    assert old.numel() > 0 and torch.equal(old, new)


@pytest.mark.parametrize("target", [0, 2, "final"])
def test_vision_capture_block_input_matches(target):
    base, _ft, ext, adapter = _vision()
    loader = _class_loader(n=16, in_dim=6, batch_size=4, seed=3)
    old = ext._capture_single_input(deepcopy(base), target, loader, 3)
    new = adapter.capture_block_input(deepcopy(base), target, loader, 3, "cpu")
    assert old.numel() > 0 and torch.equal(old, new)


@pytest.mark.parametrize("spec", _DECODER_CAPTURE_SPECS, ids=lambda s: s.name)
def test_decoder_capture_component_matches(spec):
    base, _ft, ext, adapter = _decoder()
    old = ext._capture_component_output(base, 1, spec.name, _llm_loader(), 2)
    new = adapter.capture_component(base, 1, spec, _llm_loader(), 2, "cpu")
    assert old.numel() > 0 and torch.equal(old, new)


@pytest.mark.parametrize("target", [0, 1, "final"])
def test_decoder_capture_block_input_matches(target):
    base, _ft, ext, adapter = _decoder()
    old = ext._capture_single_input(base, target, _llm_loader(), 2)
    new = adapter.capture_block_input(base, target, _llm_loader(), 2, "cpu")
    assert old.numel() > 0 and torch.equal(old, new)


def test_unsupported_component_errors_match():
    vbase, _f, vext, vad = _vision()
    bad = ComponentSpec("nope", "nope_output", "linear")
    loader = _class_loader()
    with pytest.raises(ValueError) as old:
        vext._capture_component_output(vbase, 0, "nope", loader, 1)
    with pytest.raises(ValueError) as new:
        vad.capture_component(vbase, 0, bad, loader, 1, "cpu")
    assert str(old.value) == str(new.value)


def test_layers_and_inner_block():
    vbase, _f, _e, vad = _vision()
    assert vad.layers(vbase) is vbase.visual.transformer.resblocks
    block = vad.layers(vbase)[0]
    assert vad.inner_block(block) is block
    wrapped = torch.nn.Module()
    wrapped.block = block
    assert vad.inner_block(wrapped) is block
    dbase, _f, _e, dad = _decoder()
    assert len(dad.layers(dbase)) == 2
    assert dad.inner_block(dad.layers(dbase)[0]) is dad.layers(dbase)[0]


def test_component_specs_match_cascade_order_and_ref_keys():
    assert [s.name for s in VisionAdapter.components] == ["ln_1", "q", "k", "v", "out_proj", "ln_2", "c_fc", "c_proj"]
    assert [s.slice_index for s in VisionAdapter.components if s.kind == "fused_slice"] == [0, 1, 2]
    assert [s.name for s in VisionAdapter.components if s.residual_aware] == ["out_proj", "c_proj"]
    assert [s.name for s in VisionAdapter.components if s.blendable_target] == ["c_proj"]
    assert [s.ref_key for s in DecoderAdapter.components] == [
        "input_layernorm_output",
        "q_proj_output",
        "k_proj_output",
        "v_proj_output",
        "attn_output",
        "post_attn_ln_output",
        "gate_proj_output",
        "up_proj_output",
        "down_proj_output",
    ]
    assert not any(s.residual_aware or s.blendable_target for s in DecoderAdapter.components)


# ---- collapse policy (B11 / B12) -------------------------------------------------------------

_GRID = [
    (curr, n, order, density)
    for curr in (2, 3, 6)
    for n in (1, 2, 3)
    for order in ("bottom-top", "top-bottom", "random")
    for density in ("spread", "spread_mod", "clump")
    if n < curr
]


def _call(fn, *args):
    np.random.seed(123)
    try:
        return ("ok", fn(*args))
    except Exception as exc:  # noqa: BLE001 - comparing exception type/message across implementations
        return ("err", type(exc).__name__, str(exc))


@pytest.mark.parametrize(
    "cls,adapter",
    [(BlockExtender, VisionAdapter()), (DecoderBlockExtender, DecoderAdapter(None))],
    ids=["vision", "decoder"],
)
def test_collapse_schedule_policy_matches_extender(cls, adapter):
    policy = adapter.collapse_policy()
    for args in _GRID:
        assert _call(cls._build_collapse_schedule, *args) == _call(policy.build, *args), args


def _chain(spans):
    return [{"orig_idxs": list(span)} for span in spans]


@pytest.mark.parametrize(
    "cls,adapter",
    [(BlockExtender, VisionAdapter()), (DecoderBlockExtender, DecoderAdapter(None))],
    ids=["vision", "decoder"],
)
def test_locate_collapse_pos_policy_matches_extender(cls, adapter):
    policy = adapter.collapse_policy()
    chain = _chain([(0,), (1, 2), (3,), (4,)])
    for anchor in range(-1, 6):
        assert _call(cls._locate_collapse_pos, chain, anchor) == _call(policy.locate, chain, anchor), anchor


def test_b11_spread_mod_collapse_diverges_between_families():
    args = (6, 3, "top-bottom", "spread_mod")
    vision = VisionAdapter().collapse_policy().build(*args)
    decoder = DecoderAdapter(None).collapse_policy().build(*args)
    assert vision == BlockExtender._build_collapse_schedule(*args)
    assert decoder == DecoderBlockExtender._build_collapse_schedule(*args)
    assert decoder == [0, 1, 2]  # i % (D - 1), insertion_order ignored
    assert vision == [4, 2, 0]  # linspace anchors mirrored for top-bottom
    assert vision != decoder


def test_b12_locate_collapse_pos_clamp_diverges_between_families():
    chain = _chain([(0,), (1,), (2,)])
    assert VisionAdapter().collapse_policy().locate(chain, 2) == 1  # clamped to len(chain) - 2
    assert DecoderAdapter(None).collapse_policy().locate(chain, 2) == 2  # unclamped membership


def test_build_layout_matches_layout_functions():
    from merge_and_rebase.rebase.block_extension.schedules import build_extension_layout, build_reduction_layout

    extension_chain = [
        {"orig_idx": 0, "inserted": False},
        {"orig_idx": 0, "inserted": True, "neighbour_orig_idx": 1},
        {"orig_idx": 1, "inserted": False},
    ]
    reduction_chain = [{"orig_idxs": [0]}, {"orig_idxs": [1, 2]}]
    for adapter in (VisionAdapter(), DecoderAdapter(None)):
        assert adapter.build_layout(extension_chain) == build_extension_layout(extension_chain)
        assert _call(lambda c, adapter=adapter: adapter.build_layout(c, reduction=True), reduction_chain) == _call(
            build_reduction_layout, reduction_chain
        )
