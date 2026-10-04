"""A dense sequence-classification ref keeps its classification head (release review fix, 2026-10-04).

``_load_dense_hf_state_dict`` loaded ``model_kind="sequence_classification"`` refs with AutoModelForCausalLM: the
state dict carried an ``lm_head`` instead of the fine-tuned ``score`` head, so the head delta was lost.
"""

from __future__ import annotations

import torch
import transformers

from merge_and_rebase.io.text_checkpoints import _load_dense_hf_state_dict
from merge_and_rebase.models.text_lm import TextBuildConfig


def test_sequence_classification_ref_loads_the_classification_head(tmp_path):
    cfg = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_labels=3,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForSequenceClassification(cfg)
    model.save_pretrained(tmp_path)
    build = TextBuildConfig(model_name_or_path=str(tmp_path), model_kind="sequence_classification", num_labels=3)
    sd = _load_dense_hf_state_dict(str(tmp_path), build)
    assert "score.weight" in sd and not any(k.startswith("lm_head") for k in sd)
    assert torch.equal(sd["score.weight"], model.score.weight.detach())
