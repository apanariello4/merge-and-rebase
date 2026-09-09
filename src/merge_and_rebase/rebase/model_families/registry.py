from __future__ import annotations

from typing import Any

from .base import ModelFamilyAdapter
from .hf_decoder import LlamaDecoderAdapter, Qwen2DecoderAdapter, Qwen3DecoderAdapter

_FAMILIES: dict[str, ModelFamilyAdapter] = {}


def register(adapter: ModelFamilyAdapter) -> None:
    _FAMILIES[adapter.name] = adapter


def get_family(name: str) -> ModelFamilyAdapter:
    if name not in _FAMILIES:
        raise KeyError(
            f"Unknown model family '{name}'. Available: {sorted(_FAMILIES)}"
        )
    return _FAMILIES[name]


def list_families() -> list[str]:
    return sorted(_FAMILIES)


def infer_family(model_or_config: Any) -> ModelFamilyAdapter | None:
    if hasattr(model_or_config, "config"):
        cfg = model_or_config.config
    else:
        cfg = model_or_config

    model_type = str(getattr(cfg, "model_type", "")).strip().lower()
    for _name, adapter in _FAMILIES.items():
        if hasattr(adapter, "_matches_model_type") and adapter._matches_model_type(model_type):
            return adapter
    return None


register(LlamaDecoderAdapter())
register(Qwen2DecoderAdapter())
register(Qwen3DecoderAdapter())
