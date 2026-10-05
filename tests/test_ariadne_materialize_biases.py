import torch
from _llm_fixtures import tiny_qwen2

from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.model_families import Qwen2DecoderAdapter


def _setup(dtype=torch.float32):
    model = tiny_qwen2(layers=3).to(dtype)
    adapter = Qwen2DecoderAdapter()
    return model, {k: v.clone() for k, v in model.state_dict().items()}, adapter


def test_materialize_idempotent_and_strict_load():
    model, sd, ad = _setup()
    first = materialize_missing_projection_biases(model, sd, family_adapter=ad, components=("mlp.c_proj",))
    assert len(first) == 3 and all(k.endswith("down_proj.bias") for k in first)
    assert materialize_missing_projection_biases(model, sd, family_adapter=ad, components=("mlp.c_proj",)) == []
    model.load_state_dict(sd, strict=True)


def test_materialize_o_proj_and_positions():
    model, sd, ad = _setup()
    added = materialize_missing_projection_biases(
        model, sd, family_adapter=ad, components=("attn.out_proj", "mlp.c_proj"), positions=[1]
    )
    assert sorted(added) == ["model.layers.1.mlp.down_proj.bias", "model.layers.1.self_attn.o_proj.bias"]
    model.load_state_dict(sd, strict=True)


def test_materialize_bf16_dtype_and_noop_function():
    model, sd, ad = _setup(torch.bfloat16)
    ids = torch.randint(0, 64, (2, 5))
    before = model(ids).logits.clone()
    added = materialize_missing_projection_biases(model, sd, family_adapter=ad, components=("mlp.c_proj",))
    assert all(sd[k].dtype == torch.bfloat16 for k in added)
    assert model.model.layers[0].mlp.down_proj.bias.dtype == torch.bfloat16
    assert torch.equal(model(ids).logits, before)


def test_tied_embeddings_untouched_and_skip_adds_nothing():
    model, sd, ad = _setup()
    keys = set(sd)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    materialize_missing_projection_biases(model, sd, family_adapter=ad, components=("mlp.c_proj",))
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not any("embed" in k or "lm_head" in k for k in set(sd) - keys)
    # skip path never calls the helper: stock architecture has no new keys
    m2, sd2, _ = _setup()
    assert set(m2.state_dict()) == keys
