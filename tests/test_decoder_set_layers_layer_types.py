from _llm_fixtures import tiny_qwen2

from merge_and_rebase.rebase.model_families import Qwen2DecoderAdapter


def test_set_layers_truncates_layer_types_on_shrink():
    model = tiny_qwen2(layers=4)
    ad = Qwen2DecoderAdapter()
    assert len(model.config.layer_types) == 4
    ad.set_layers(model, list(ad.layers(model))[:2])
    assert model.config.num_hidden_layers == 2
    assert len(model.config.layer_types) == 2


def test_set_layers_grow_keeps_layer_types_in_sync():
    model = tiny_qwen2(layers=2)
    ad = Qwen2DecoderAdapter()
    blocks = list(ad.layers(model))
    ad.set_layers(model, blocks + [blocks[-1]])
    assert len(model.config.layer_types) == 3
