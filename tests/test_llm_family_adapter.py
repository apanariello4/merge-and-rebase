from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.rebase.model_families import infer_family, list_families
from merge_and_rebase.rebase.model_families.hf_decoder import (
    LlamaDecoderAdapter,
    Qwen2DecoderAdapter,
    Qwen3DecoderAdapter,
)


class _FakeConfig:
    model_type = "llama"
    hidden_size = 64
    intermediate_size = 128
    num_hidden_layers = 4
    num_attention_heads = 4


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
        for layer in self.model.layers:
            layer.input_layernorm = nn.LayerNorm(64)
            layer.post_attention_layernorm = nn.LayerNorm(64)
            attn = nn.Module()
            attn.q_proj = nn.Linear(64, 64)
            attn.k_proj = nn.Linear(64, 64)
            attn.v_proj = nn.Linear(64, 64)
            attn.o_proj = nn.Linear(64, 64)
            layer.self_attn = attn
            mlp = nn.Module()
            mlp.gate_proj = nn.Linear(64, 128)
            mlp.up_proj = nn.Linear(64, 128)
            mlp.down_proj = nn.Linear(128, 64)
            layer.mlp = mlp
        self.model.norm = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 320)
        self.model.embed_tokens = nn.Embedding(320, 64)


def test_families_registered() -> None:
    assert "llama" in list_families()
    assert "qwen2" in list_families()
    assert "qwen3" in list_families()


def test_llama_adapter_matches() -> None:
    adapter = LlamaDecoderAdapter()
    assert adapter._matches_model_type("llama")
    assert not adapter._matches_model_type("qwen2")


def test_qwen2_adapter_matches() -> None:
    adapter = Qwen2DecoderAdapter()
    assert adapter._matches_model_type("qwen2")
    assert adapter._matches_model_type("qwen2_moe")


def test_qwen3_adapter_matches() -> None:
    adapter = Qwen3DecoderAdapter()
    assert adapter._matches_model_type("qwen3")
    assert adapter._matches_model_type("qwen3_moe")
    assert not adapter._matches_model_type("qwen2")


def test_infer_family() -> None:
    model = _FakeModel()
    adapter = infer_family(model)
    assert adapter is not None
    assert adapter.name == "llama"


def test_metadata() -> None:
    model = _FakeModel()
    adapter = LlamaDecoderAdapter()
    meta = adapter.metadata(model)
    assert meta.family == "llama"
    assert meta.hidden_size == 64
    assert meta.intermediate_size == 128
    assert meta.num_hidden_layers == 4
    assert meta.num_attention_heads == 4


def test_transport_scope() -> None:
    model = _FakeModel()
    adapter = LlamaDecoderAdapter()
    scope = adapter.transport_scope(model)
    assert scope is model.model


def test_transportable_keys() -> None:
    model = _FakeModel()
    sd = {k: v for k, v in model.named_parameters()}
    adapter = LlamaDecoderAdapter()
    keys = adapter.transportable_keys(sd)
    assert "model.layers.0.input_layernorm.weight" in keys
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.0.self_attn.k_proj.weight" in keys
    assert "model.layers.0.self_attn.v_proj.weight" in keys
    assert "model.layers.0.self_attn.o_proj.weight" in keys
    assert "model.layers.0.post_attention_layernorm.weight" in keys
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.layers.0.mlp.up_proj.weight" in keys
    assert "model.layers.0.mlp.down_proj.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" not in keys
    assert "model.embed_tokens.weight" not in keys


def test_block_count() -> None:
    model = _FakeModel()
    adapter = LlamaDecoderAdapter()
    assert adapter.block_count(model) == 4


def test_extract_calibration_batch_from_dict() -> None:
    adapter = LlamaDecoderAdapter()
    batch = {
        "input_ids": torch.randint(0, 100, (4, 128)),
        "attention_mask": torch.ones(4, 128),
        "labels": torch.randint(0, 100, (4, 128)),
    }
    result = adapter.extract_calibration_batch(batch)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result


def test_extract_calibration_batch_from_tuple() -> None:
    adapter = LlamaDecoderAdapter()
    batch = (torch.randint(0, 100, (4, 128)), torch.randint(0, 100, (4, 128)))
    result = adapter.extract_calibration_batch(batch)
    assert "input_ids" in result
