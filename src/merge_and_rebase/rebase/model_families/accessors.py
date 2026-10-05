"""Duck-typed decoder-layout accessors.

Each function asks the family adapter for the layout and falls back to the HF decoder
defaults when the adapter predates the layout API (test fakes). Real adapters never reach
the fallbacks, so the adapter stays the single source of layout.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .hf_decoder import BLOCK_COMPONENT_PATHS, CANONICAL_COMPONENTS, _resolve_path, decoder_set_layers

LAYER_PREFIX = "model.layers"


def layers(family_adapter: Any, model: nn.Module) -> nn.ModuleList:
    fn = getattr(family_adapter, "layers", None)
    return fn(model) if fn is not None else family_adapter.transport_scope(model).layers


def set_layers(family_adapter: Any, model: nn.Module, blocks: Any) -> None:
    fn = getattr(family_adapter, "set_layers", None)
    if fn is not None:
        fn(model, blocks)
    else:
        decoder_set_layers(family_adapter.transport_scope(model), model, blocks)


def final_norm(family_adapter: Any, model: nn.Module) -> nn.Module:
    fn = getattr(family_adapter, "final_norm", None)
    return fn(model) if fn is not None else family_adapter.transport_scope(model).norm


def block_components(family_adapter: Any, block: nn.Module) -> dict[str, nn.Module]:
    fn = getattr(family_adapter, "block_components", None)
    if fn is not None:
        return fn(block)
    return {name: _resolve_path(block, path) for name, path in BLOCK_COMPONENT_PATHS.items()}


def attn_module(family_adapter: Any, block: nn.Module) -> nn.Module:
    fn = getattr(family_adapter, "attn_module", None)
    return fn(block) if fn is not None else block.self_attn


def residual_writer(family_adapter: Any, block: nn.Module) -> nn.Module:
    fn = getattr(family_adapter, "residual_writer", None)
    return fn(block) if fn is not None else _resolve_path(block, CANONICAL_COMPONENTS["mlp.c_proj"])


def attn_output(family_adapter: Any, block: nn.Module) -> nn.Module:
    fn = getattr(family_adapter, "attn_output", None)
    return fn(block) if fn is not None else _resolve_path(block, CANONICAL_COMPONENTS["attn.out_proj"])


def param_key(family_adapter: Any, position: int, suffix: str) -> str:
    fn = getattr(family_adapter, "param_key", None)
    return fn(position, suffix) if fn is not None else f"{LAYER_PREFIX}.{int(position)}.{suffix}"


def content_mask(family_adapter: Any, batch: Any):
    """Bool ``[B, T]`` content mask of a calibration batch, from ``attention_mask`` only."""
    import torch

    fn = getattr(family_adapter, "content_mask", None)
    if fn is not None:
        return fn(batch)
    inputs = family_adapter.extract_calibration_batch(batch)
    mask = inputs.get("attention_mask")
    if mask is None:
        return torch.ones_like(inputs["input_ids"], dtype=torch.bool)
    return mask.bool()


def canonical_components(family_adapter: Any) -> dict[str, str]:
    mapping = getattr(family_adapter, "CANONICAL_COMPONENTS", None)
    return dict(mapping) if mapping is not None else dict(CANONICAL_COMPONENTS)
