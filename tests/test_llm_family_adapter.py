from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.rebase.capabilities import check_pair
from merge_and_rebase.rebase.model_families import infer_family, list_families
from merge_and_rebase.rebase.model_families.hf_decoder import (
    Gemma3DecoderAdapter,
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


class _FakeGemma3Config:
    """Gemma 3's config, in the two ways it differs from Llama/Qwen2.

    ``head_dim`` is *not* ``hidden_size // num_attention_heads`` (256 vs 160 on
    the real 270m), and ``layer_types`` alternates sliding/full attention on a
    5:1 pattern keyed by block position.
    """

    model_type = "gemma3_text"
    hidden_size = 640
    intermediate_size = 2048
    num_hidden_layers = 18
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256
    layer_types = [
        "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
        for i in range(18)
    ]


class _FakeGemma3Model(nn.Module):
    """A two-block stand-in carrying Gemma 3's extra per-block norms."""

    def __init__(self, hidden: int = 640, inter: int = 2048, depth: int = 2):
        super().__init__()
        self.config = _FakeGemma3Config()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(depth)])
        for layer in self.model.layers:
            layer.input_layernorm = nn.LayerNorm(hidden)
            layer.post_attention_layernorm = nn.LayerNorm(hidden)
            # Gemma 3 only: a second norm pair around the MLP.
            layer.pre_feedforward_layernorm = nn.LayerNorm(hidden)
            layer.post_feedforward_layernorm = nn.LayerNorm(hidden)
            attn = nn.Module()
            # Gemma 3 carries no biases anywhere, unlike Qwen2's q/k/v.
            attn.q_proj = nn.Linear(hidden, 1024, bias=False)
            attn.k_proj = nn.Linear(hidden, 256, bias=False)
            attn.v_proj = nn.Linear(hidden, 256, bias=False)
            attn.o_proj = nn.Linear(1024, hidden, bias=False)
            # Gemma 3 only: per-head QK-RMSNorm, as on Qwen3.
            attn.q_norm = nn.LayerNorm(256)
            attn.k_norm = nn.LayerNorm(256)
            layer.self_attn = attn
            mlp = nn.Module()
            mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
            mlp.up_proj = nn.Linear(hidden, inter, bias=False)
            mlp.down_proj = nn.Linear(inter, hidden, bias=False)
            layer.mlp = mlp
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, 320, bias=False)
        self.model.embed_tokens = nn.Embedding(320, hidden)


def test_families_registered() -> None:
    assert "llama" in list_families()
    assert "qwen2" in list_families()
    assert "qwen3" in list_families()
    assert "gemma3" in list_families()


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


def test_gemma3_adapter_matches() -> None:
    adapter = Gemma3DecoderAdapter()
    # Text-only checkpoints (270m, 1b) report "gemma3_text"; the multimodal
    # ones report "gemma3" and wrap the same decoder.
    assert adapter._matches_model_type("gemma3_text")
    assert adapter._matches_model_type("gemma3")
    assert not adapter._matches_model_type("gemma2")
    assert not adapter._matches_model_type("qwen2")


def test_infer_family() -> None:
    model = _FakeModel()
    adapter = infer_family(model)
    assert adapter is not None
    assert adapter.name == "llama"


def test_infer_family_gemma3() -> None:
    # Without this, infer_family returns None, check_pair silently skips every
    # validation, the block-extension pre-step never runs, and a Proposal-1 run
    # dies at the guard in llm_rebase rather than at load time.
    adapter = infer_family(_FakeGemma3Model())
    assert adapter is not None
    assert adapter.name == "gemma3"


def test_gemma3_metadata_prefers_config_head_dim() -> None:
    meta = Gemma3DecoderAdapter().metadata(_FakeGemma3Model())
    assert meta.family == "gemma3"
    assert meta.hidden_size == 640
    assert meta.intermediate_size == 2048
    assert meta.num_hidden_layers == 18
    # 640 // 4 == 160; the config value must win.
    assert meta.head_dim == 256
    assert meta.num_key_value_heads == 1


def test_gemma3_pair_passes_capability_check() -> None:
    """270m -> 1b: same family, width up and depth up, via theseus."""
    adapter = Gemma3DecoderAdapter()
    source = _FakeGemma3Model(hidden=640, inter=2048)
    target = _FakeGemma3Model(hidden=1152, inter=6912)
    target.config.hidden_size = 1152
    target.config.intermediate_size = 6912
    target.config.num_hidden_layers = 26
    # A depth mismatch is only legal with the block-extension pre-step, which
    # is exactly what the Proposal-1 arm runs.
    check_pair(
        "theseus",
        adapter.metadata(source),
        adapter.metadata(target),
        allow_depth_mismatch=True,
    )


def test_gemma3_transportable_keys_exclude_gemma_only_norms() -> None:
    model = _FakeGemma3Model()
    sd = dict(model.named_parameters())
    keys = Gemma3DecoderAdapter().transportable_keys(sd)

    # The write surfaces Proposal 1 fits.
    assert "model.layers.0.mlp.down_proj.weight" in keys
    assert "model.layers.0.self_attn.o_proj.weight" in keys
    assert "model.layers.0.input_layernorm.weight" in keys
    assert "model.layers.0.post_attention_layernorm.weight" in keys
    assert "model.norm.weight" in keys

    # Gemma-only norms have no Qwen2 counterpart and are absent from
    # _DECODER_TRANSPORTABLE_SUFFIXES, so they fall through to passthrough --
    # which the direct_target arm drops. Same treatment as Qwen3's q/k_norm.
    for suffix in (
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ):
        assert f"model.layers.0.{suffix}" not in keys

    assert "lm_head.weight" not in keys
    assert "model.embed_tokens.weight" not in keys


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
