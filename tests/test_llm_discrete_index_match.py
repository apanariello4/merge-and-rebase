import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from merge_and_rebase.eval.block_extension_llm import run_discrete_index_match_llm
from merge_and_rebase.rebase.discrete_layer_match import discrete_layer_pairing
from merge_and_rebase.rebase.model_families.registry import infer_family


def _tiny_qwen(depth: int) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=depth,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_discrete_index_match_copies_paired_blocks_and_runs():
    model = _tiny_qwen(4)
    orig = [{k: v.clone() for k, v in layer.state_dict().items()} for layer in model.model.layers]
    depth = run_discrete_index_match_llm(model=model, target_layers_total=7, family_adapter=infer_family(model))

    assert depth == 7
    assert model.config.num_hidden_layers == 7
    pairing = discrete_layer_pairing(4, 7)
    for j, layer in enumerate(model.model.layers):
        assert layer.self_attn.layer_idx == j
        for k, v in layer.state_dict().items():
            assert torch.equal(v, orig[pairing[j]][k])
    # Positions 0 and 1 both copy source block 0; they must not share storage.
    assert pairing[0] == pairing[1] == 0
    layers = model.model.layers
    assert layers[0].mlp.down_proj.weight.data_ptr() != layers[1].mlp.down_proj.weight.data_ptr()

    ids = torch.randint(0, 64, (2, 5))
    out = model(input_ids=ids, use_cache=True)
    assert out.logits.shape == (2, 5, 64)
    assert len(out.past_key_values) == 7
