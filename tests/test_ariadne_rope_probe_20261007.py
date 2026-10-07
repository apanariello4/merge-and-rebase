import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from merge_and_rebase.eval.diagnostics.ariadne_norm.rope_probe import (
    block_record,
    block_states,
    first_token_rows,
    load_model,
    native_rope_theta,
    set_rope_theta,
)


def _tiny(path, theta):
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    set_rope_theta(cfg, theta)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(path)


def _batches():
    ids = torch.randint(0, 64, (3, 10))
    mask = torch.ones(3, 10, dtype=torch.long)
    mask[1, 7:] = 0
    return [{"input_ids": ids, "attention_mask": mask}]


def test_load_model_sets_rope_and_changes_states(tmp_path):
    _tiny(tmp_path, 1e6)
    own = load_model(str(tmp_path), 1e6, "cpu", torch.float32)
    alt = load_model(str(tmp_path), 1e4, "cpu", torch.float32)
    assert native_rope_theta(own.config) == 1e6 and native_rope_theta(alt.config) == 1e4
    b = _batches()
    s_own, s_own2, s_alt = block_states(own, b, "cpu"), block_states(own, b, "cpu"), block_states(alt, b, "cpu")
    assert s_own[0].shape[0] == int(b[0]["attention_mask"].sum())
    assert all(torch.equal(x, y) for x, y in zip(s_own, s_own2, strict=True))
    # Only rope differs: the first token (position 0, no rotation) is identical, later tokens are not.
    first = first_token_rows(b)
    assert torch.allclose(s_own[1][first], s_alt[1][first], atol=1e-6)
    assert (s_own[1][~first] - s_alt[1][~first]).abs().max() > 1e-6


def test_block_record_decomposition():
    torch.manual_seed(1)
    b_own, b_ft, m_own, m_base = (torch.randn(20, 8) for _ in range(4))
    r = block_record(b_own, b_ft, m_own, m_base)
    assert r["identity_max_abs"] < 1e-6
    assert abs(r["share_transported_along_rope"] - r["cos_transported_rope"] ** 2) < 1e-12
    same = block_record(b_own, b_own, m_own, m_base)
    assert same["rel_rope_only"] == 0.0 and abs(same["cos_transported_ft_matched"] - 1.0) < 1e-12
