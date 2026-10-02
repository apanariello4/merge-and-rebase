"""Phase 7 S6: HF-correct discrete depth reindexing of decoders."""

from __future__ import annotations

import copy

import pytest
import torch
from _llm_fixtures import tiny_qwen2

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing, build_discrete_indexed_decoder
from merge_and_rebase.rebase.model_families import infer_family


def _ids():
    torch.manual_seed(3)
    return torch.randint(2, 60, (2, 7))


@pytest.fixture
def src():
    model = tiny_qwen2(layers=2)
    return model, infer_family(model)


def test_identity_pairing_bitwise_logits(src):
    model, fam = src
    out = build_discrete_indexed_decoder(model, DiscreteLayerPairing.compute(2, 2), fam)
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, out(ids).logits)


@pytest.mark.parametrize("target", [3, 1])
def test_depth_and_unique_layer_idx(src, target):
    model, fam = src
    pairing = DiscreteLayerPairing.compute(2, target)
    out = build_discrete_indexed_decoder(model, pairing, fam)
    layers = fam.layers(out)
    assert len(layers) == target == out.config.num_hidden_layers
    assert [layer.self_attn.layer_idx for layer in layers] == list(range(target))
    assert len(out.config.layer_types) == target
    for j, i in enumerate(pairing.pairing):
        a, b = fam.layers(model)[i].state_dict(), layers[j].state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert len({id(layer) for layer in layers}) == target


def test_mutation_isolation(src):
    model, fam = src
    before = copy.deepcopy(model.state_dict())
    out = build_discrete_indexed_decoder(model, DiscreteLayerPairing.compute(2, 3), fam)
    assert len(fam.layers(model)) == 2 and model.config.num_hidden_layers == 2
    with torch.no_grad():
        for p in out.parameters():
            p.add_(1.0)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_generate_three_tokens_extended(src):
    model, fam = src
    out = build_discrete_indexed_decoder(model, DiscreteLayerPairing.compute(2, 3), fam)
    gen = out.generate(_ids(), max_new_tokens=3, do_sample=False, pad_token_id=0)
    assert gen.shape[1] == 7 + 3
