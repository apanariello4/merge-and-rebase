"""Phase 7 S2: the family adapter is the single source of decoder layout."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import tiny_llama, tiny_qwen2, tiny_qwen3

from merge_and_rebase.rebase.block_extension.adapters import DECODER_COMPONENTS
from merge_and_rebase.rebase.capabilities import check_pair
from merge_and_rebase.rebase.model_families import accessors, infer_family
from merge_and_rebase.rebase.model_families.base import ModelFamilyMetadata

BUILDERS = {"llama": tiny_llama, "qwen2": tiny_qwen2, "qwen3": tiny_qwen3}
OLD_TUPLE = tuple(spec.name for spec in DECODER_COMPONENTS)


@pytest.fixture(params=sorted(BUILDERS))
def model_and_family(request):
    model = BUILDERS[request.param](layers=3)
    family = infer_family(model)
    assert family is not None and family.name == request.param
    return model, family


def test_param_key_in_state_dict_for_every_position(model_and_family):
    model, family = model_and_family
    sd = model.state_dict()
    for pos in range(len(family.layers(model))):
        for suffix in ("mlp.down_proj.weight", "self_attn.o_proj.weight", "input_layernorm.weight"):
            assert family.param_key(pos, suffix) in sd
        for canon, rel in family.CANONICAL_COMPONENTS.items():
            assert family.param_key(pos, rel + ".weight") in sd, canon
    assert family.CANONICAL_COMPONENTS == {"mlp.c_proj": "mlp.down_proj", "attn.out_proj": "self_attn.o_proj"}


def test_block_components_equal_old_hardcoded_tuple(model_and_family):
    model, family = model_and_family
    block = family.layers(model)[0]
    comps = family.block_components(block)
    assert tuple(comps) == OLD_TUPLE
    assert comps["down_proj"] is block.mlp.down_proj is family.residual_writer(block)
    assert comps["o_proj"] is block.self_attn.o_proj is family.attn_output(block)
    assert comps["q_proj"] is block.self_attn.q_proj
    assert comps["post_attention_layernorm"] is block.post_attention_layernorm
    assert family.final_norm(model) is model.model.norm


def _extend_by_duplicating(model, family, n_new):
    blocks = list(family.layers(model))
    blocks = blocks + [copy.deepcopy(b) for b in blocks[-n_new:]]
    family.set_layers(model, blocks)


@pytest.mark.parametrize("op", ["extend", "shrink"])
def test_set_layers_reindexes_and_model_runs(model_and_family, op):
    model, family = model_and_family
    if op == "extend":
        _extend_by_duplicating(model, family, 2)
    else:
        family.set_layers(model, list(family.layers(model))[:2])
    depth = len(family.layers(model))
    assert depth == (5 if op == "extend" else 2)
    assert model.config.num_hidden_layers == depth
    layer_types = getattr(model.config, "layer_types", None)  # absent on Llama in some versions
    if layer_types is not None:
        # grown on extend; left as-is (HF slices it by num_hidden_layers) on shrink -- legacy behaviour kept
        assert len(layer_types) >= depth
    for i, layer in enumerate(family.layers(model)):
        assert layer.self_attn.layer_idx == i
        if layer_types is not None and hasattr(layer, "attention_type"):
            assert layer.attention_type == layer_types[i]

    fired = [0] * depth
    hooks = [
        layer.register_forward_hook(lambda *_a, i=i: fired.__setitem__(i, fired[i] + 1))
        for i, layer in enumerate(family.layers(model))
    ]
    ids = torch.randint(0, 60, (2, 6))
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    assert fired == [1] * depth

    out = model.generate(input_ids=ids[:, :4], max_new_tokens=3, do_sample=False, pad_token_id=0)
    assert out.shape == (2, 7)


def test_qwen3_qk_norm_present_but_not_transportable():
    model = tiny_qwen3()
    family = infer_family(model)
    sd = model.state_dict()
    qk = {k for k in sd if k.endswith(("q_norm.weight", "k_norm.weight"))}
    assert qk
    assert not (qk & family.transportable_keys(sd))


def test_content_mask_equals_attention_mask_and_ignores_ids(model_and_family):
    _, family = model_and_family
    pad_id = 3
    ids = torch.tensor([[5, 6, pad_id, pad_id], [pad_id, 7, 8, pad_id]])
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])  # last row: pad id attended as content
    got = family.content_mask({"input_ids": ids, "attention_mask": mask})
    assert got.dtype == torch.bool and torch.equal(got, mask.bool())
    assert bool(got[1, 3])
    no_mask = family.content_mask({"input_ids": ids})
    assert bool(no_mask.all()) and no_mask.shape == ids.shape


def test_calibration_forward_is_backbone_only(model_and_family):
    model, family = model_and_family
    ids = torch.randint(0, 60, (2, 5))
    with torch.no_grad():
        out = family.calibration_forward(model, {"input_ids": ids, "labels": ids}, "cpu")
    assert out.last_hidden_state.shape == (2, 5, model.config.hidden_size)
    assert getattr(out, "past_key_values", None) is None


@pytest.mark.parametrize("model_type", ["qwen2_moe", "qwen3_moe"])
def test_moe_config_rejected(model_type):
    cfg = SimpleNamespace(
        model_type=model_type, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2
    )
    family = infer_family(cfg)
    meta = family.metadata(cfg)
    assert meta.is_moe
    dense = ModelFamilyMetadata(
        family=meta.family, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2
    )
    with pytest.raises(ValueError, match="Mixture-of-experts"):
        check_pair("theseus", meta, meta)
    with pytest.raises(ValueError, match="Mixture-of-experts"):
        check_pair("theseus", dense, meta)
    check_pair("theseus", dense, dense)


def test_accessor_fallbacks_for_fakes():
    model = tiny_qwen2(layers=2)

    class Fake:
        def transport_scope(self, m):
            return m.model

    fake = Fake()
    block = accessors.layers(fake, model)[0]
    assert accessors.residual_writer(fake, block) is block.mlp.down_proj
    assert accessors.attn_output(fake, block) is block.self_attn.o_proj
    assert accessors.param_key(fake, 1, "mlp.down_proj.weight") in model.state_dict()
    assert tuple(accessors.block_components(fake, block)) == OLD_TUPLE
    accessors.set_layers(fake, model, list(accessors.layers(fake, model))[:1])
    assert model.config.num_hidden_layers == 1
