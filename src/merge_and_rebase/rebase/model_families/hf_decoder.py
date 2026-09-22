from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.nn as nn

from ...merge.task_vectors import default_key_filter
from .base import ModelFamilyMetadata

_DECODER_TRANSPORTABLE_SUFFIXES = frozenset({
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.bias",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.bias",
    "mlp.up_proj.weight",
    "mlp.up_proj.bias",
    "mlp.down_proj.weight",
    "mlp.down_proj.bias",
})

_DECODER_EXCLUDED_ROOTS = frozenset({
    "embed_tokens",
    "embed_positions",
    "lm_head",
    "rotary_emb",
})


class HfDecoderAdapter:
    """Shared adapter for HF decoder-style models with 'model.layers' layout."""

    name: str = "hf_decoder"

    LAYER_PREFIX: str = "model.layers"
    FINAL_NORM_KEY: str = "model.norm"

    _MODEL_TYPES: frozenset[str] = frozenset()

    @classmethod
    def _matches_model_type(cls, model_type: str) -> bool:
        return model_type in cls._MODEL_TYPES

    def metadata(self, model: nn.Module) -> ModelFamilyMetadata:
        cfg = getattr(model, "config", model)
        num_attention_heads = int(getattr(cfg, "num_attention_heads", 0))
        hidden_size = int(getattr(cfg, "hidden_size", 0))
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None and num_attention_heads:
            head_dim = hidden_size // num_attention_heads
        return ModelFamilyMetadata(
            family=self.name,
            hidden_size=hidden_size,
            intermediate_size=int(getattr(cfg, "intermediate_size", 0)),
            num_hidden_layers=int(getattr(cfg, "num_hidden_layers", 0)),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", None),
            head_dim=int(head_dim) if head_dim is not None else None,
        )

    def transport_scope(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            return model.model
        return model

    def transportable_keys(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> set[str]:
        keys: set[str] = set()
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not default_key_filter(k, v):
                continue
            if any(k.startswith(root + ".") or k == root for root in _DECODER_EXCLUDED_ROOTS):
                continue
            if self._is_layer_param(k):
                suffix = self._layer_param_suffix(k)
                if suffix in _DECODER_TRANSPORTABLE_SUFFIXES:
                    keys.add(k)
                    continue
            if k == self.FINAL_NORM_KEY + ".weight":
                keys.add(k)
        return keys

    def param_to_module(self, model: nn.Module) -> dict[str, str]:
        scope = self.transport_scope(model)
        out: dict[str, str] = {}
        for module_name, module in scope.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                rel_key = f"{module_name}.{param_name}" if module_name else param_name
                out[rel_key] = module_name
                out[f"model.{rel_key}"] = module_name
        return out

    def iter_blocks(self, model: nn.Module) -> Iterator[nn.Module]:
        scope = self.transport_scope(model)
        layers = getattr(scope, "layers", None)
        if layers is not None:
            yield from layers

    def block_count(self, model: nn.Module) -> int:
        scope = self.transport_scope(model)
        layers = getattr(scope, "layers", None)
        if layers is not None:
            return len(layers)
        return 0

    def extract_calibration_batch(
        self, batch: Any
    ) -> dict[str, torch.Tensor]:
        if isinstance(batch, Mapping) or hasattr(batch, "get"):
            out: dict[str, torch.Tensor] = {}
            for key in ("input_ids", "attention_mask", "labels"):
                val = batch.get(key)
                if isinstance(val, torch.Tensor):
                    out[key] = val
            return out
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            first = batch[0]
            second = batch[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                return {"input_ids": first, "labels": second}
        return {"input_ids": batch} if isinstance(batch, torch.Tensor) else {}

    def excluded_keys(self) -> set[str]:
        return set(_DECODER_EXCLUDED_ROOTS)

    @staticmethod
    def _is_layer_param(key: str) -> bool:
        return bool(re.match(r"^model\.layers\.\d+\.", key))

    @staticmethod
    def _layer_param_suffix(key: str) -> str:
        m = re.match(r"^model\.layers\.\d+\.(.+)$", key)
        if m:
            return m.group(1)
        return key


class LlamaDecoderAdapter(HfDecoderAdapter):
    name: str = "llama"
    _MODEL_TYPES = frozenset({"llama"})


class Qwen2DecoderAdapter(HfDecoderAdapter):
    name: str = "qwen2"
    _MODEL_TYPES = frozenset({"qwen2", "qwen2_moe"})


class Qwen3DecoderAdapter(HfDecoderAdapter):
    name: str = "qwen3"
    _MODEL_TYPES = frozenset({"qwen3", "qwen3_moe"})
    # Qwen3 adds per-head QK-RMSNorm ("self_attn.q_norm.weight" / "k_norm.weight"),
    # which have no Qwen2 counterpart. They're intentionally excluded from
    # _DECODER_TRANSPORTABLE_SUFFIXES, so they're simply left untouched (passthrough)
    # rather than transported.


class Gemma3DecoderAdapter(HfDecoderAdapter):
    name: str = "gemma3"
    # Text-only Gemma 3 checkpoints (270m, 1b) report "gemma3_text"; the
    # multimodal ones report "gemma3" and wrap the same decoder.
    _MODEL_TYPES = frozenset({"gemma3_text", "gemma3"})
    # Gemma 3 adds four per-block norms on top of the Qwen2 layout:
    # "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
    # and per-head QK-RMSNorm ("self_attn.q_norm.weight" / "k_norm.weight").
    # As with Qwen3's q_norm/k_norm they're absent from
    # _DECODER_TRANSPORTABLE_SUFFIXES and fall through to passthrough rather
    # than being transported.
    #
    # Two more Gemma 3 quirks that the shared adapter already handles, noted
    # here because neither holds on Qwen2/Llama:
    #   - head_dim (256) is not hidden_size // num_attention_heads, so
    #     metadata() must read cfg.head_dim -- which it does, preferring the
    #     config value and only deriving as a fallback.
    #   - blocks alternate sliding/full attention on a 5:1 pattern keyed by
    #     position. The depth resize keeps config.layer_types authoritative and
    #     reassigns each block's attention_type; see
    #     eval/block_extension_llm.py::_resolve_layer_types.
