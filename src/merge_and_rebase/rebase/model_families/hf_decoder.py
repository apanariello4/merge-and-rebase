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


# Canonical (vision-style) component names mapped to the decoder block's relative module path.
CANONICAL_COMPONENTS: dict[str, str] = {
    "mlp.c_proj": "mlp.down_proj",
    "attn.out_proj": "self_attn.o_proj",
}

# Correctable block components (name -> module path inside the block); order is the extender's.
BLOCK_COMPONENT_PATHS: dict[str, str] = {
    "input_layernorm": "input_layernorm",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "post_attention_layernorm": "post_attention_layernorm",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}

_MOE_MODEL_TYPES = frozenset({"qwen2_moe", "qwen3_moe"})


def _resolve_path(root: nn.Module, path: str) -> nn.Module:
    obj: Any = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def decoder_set_layers(scope: nn.Module, model: nn.Module, blocks: Any) -> None:
    """Install ``blocks`` on ``scope`` and reindex depth / ``layer_idx`` / ``layer_types``."""
    scope.layers = nn.ModuleList(blocks)
    _set_depth(model, scope, len(scope.layers))
    _reindex_layers(model, scope.layers)


def _set_depth(model: nn.Module, scope: nn.Module, depth: int) -> None:
    # HF decoders iterate `self.layers[: self.config.num_hidden_layers]`, so
    # a longer ModuleList alone does nothing: the appended blocks never run.
    # Everything downstream still sees them (state_dict reports them, deltas
    # are computed over them), which makes the truncation invisible -- the
    # model simply behaves as if it were never extended.
    for holder in (model, scope):
        config = getattr(holder, "config", None)
        if config is None:
            continue
        if getattr(config, "num_hidden_layers", None) == depth:
            continue
        config.num_hidden_layers = depth


def _resolve_layer_types(model: nn.Module, layers: nn.ModuleList) -> list[str] | None:
    """Resize `config.layer_types` (grow or truncate) to the new depth, keeping it authoritative.

    Models with alternating attention patterns key off this list, and the
    reindex below reads it positionally. Left short, every layer past the
    original depth keeps whichever `attention_type` it was duplicated with
    while the config claims a shorter model.
    """
    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return None
    resolved = list(layer_types)
    if len(resolved) >= len(layers):
        # Shrink: the config must follow the new depth too (correctness, unconditional).
        resolved = resolved[: len(layers)]
        config.layer_types = resolved
        return resolved
    for idx in range(len(resolved), len(layers)):
        own = getattr(layers[idx], "attention_type", None)
        resolved.append(own if own is not None else resolved[-1])
    config.layer_types = resolved
    return resolved


def _reindex_layers(model: nn.Module, layers: nn.ModuleList) -> None:
    # Each decoder layer's attention module caches its own `layer_idx`
    # (set at construction) to key into the shared KV cache during a
    # forward pass. Duplicating/reordering layers without updating it
    # leaves two layers pointing at the same cache slot: the second one
    # to run has its `update()` call concatenate onto the first's
    # leftover keys/values, silently doubling the sequence length the
    # rest of that layer's attention sees (crashes as a seq-length
    # mismatch against the attention mask, or worse, doesn't crash).
    layer_types = _resolve_layer_types(model, layers)
    for new_idx, layer in enumerate(layers):
        for holder in (layer, getattr(layer, "self_attn", None)):
            if holder is not None and hasattr(holder, "layer_idx"):
                holder.layer_idx = new_idx
        if layer_types is not None and hasattr(layer, "attention_type") and new_idx < len(layer_types):
            layer.attention_type = layer_types[new_idx]


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
            is_moe=str(getattr(cfg, "model_type", "")).strip().lower() in _MOE_MODEL_TYPES,
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

    # ---- decoder layout ------------------------------------------------------------------------------

    CANONICAL_COMPONENTS = CANONICAL_COMPONENTS

    def layers(self, model: nn.Module) -> nn.ModuleList:
        return self.transport_scope(model).layers

    def set_layers(self, model: nn.Module, blocks: Any) -> None:
        decoder_set_layers(self.transport_scope(model), model, blocks)

    def final_norm(self, model: nn.Module) -> nn.Module:
        return self.transport_scope(model).norm

    def block_components(self, block: nn.Module) -> dict[str, nn.Module]:
        return {name: _resolve_path(block, path) for name, path in BLOCK_COMPONENT_PATHS.items()}

    def attn_module(self, block: nn.Module) -> nn.Module:
        return block.self_attn

    def residual_writer(self, block: nn.Module) -> nn.Module:
        return _resolve_path(block, CANONICAL_COMPONENTS["mlp.c_proj"])

    def attn_output(self, block: nn.Module) -> nn.Module:
        return _resolve_path(block, CANONICAL_COMPONENTS["attn.out_proj"])

    def param_key(self, position: int, suffix: str) -> str:
        return f"{self.LAYER_PREFIX}.{int(position)}.{suffix}"

    def calibration_forward(self, model: nn.Module, batch: Any, device: Any) -> Any:
        """Backbone-only forward: no labels, no LM head, ``use_cache=False``.

        Grad mode is the caller's. Not yet used by the capture paths (their compute path is
        unchanged until the calibration switch-over).
        """
        inputs = self.extract_calibration_batch(batch)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return self.transport_scope(model)(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    def content_mask(self, batch: Any) -> torch.Tensor:
        """Bool ``[B, T]`` from ``attention_mask`` only; a pad-id token that is attended stays content."""
        inputs = self.extract_calibration_batch(batch)
        mask = inputs.get("attention_mask")
        if mask is None:
            return torch.ones_like(inputs["input_ids"], dtype=torch.bool)
        return mask.bool()

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
