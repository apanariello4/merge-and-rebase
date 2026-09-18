from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .model_families.base import ModelFamilyMetadata


@dataclass(frozen=True)
class _PairSupport:
    cross_size: bool = False
    required: bool = True


_METHOD_SUPPORT: dict[str, _PairSupport] = {
    "theseus": _PairSupport(cross_size=True),
    "theseus_gqa": _PairSupport(cross_size=True),
    "bico": _PairSupport(cross_size=True),
    "identity": _PairSupport(cross_size=False),
    "orthogonal_shift": _PairSupport(cross_size=False),
    "gradfix": _PairSupport(cross_size=False),
    "transfusion": _PairSupport(cross_size=False, required=False),
}

# Families that don't share a model_type but are the same "hf_decoder" shape
# (model.layers.N.self_attn/mlp.* naming) closely enough for activation-driven
# transport (theseus) to be meaningful across them. Qwen3 only adds per-head
# QK-RMSNorm weights on top of the Qwen2 layout, which aren't transportable
# keys to begin with (see Qwen3DecoderAdapter), so they're just left as-is.
_COMPATIBLE_CROSS_FAMILY_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"qwen2", "qwen3"}),
})


def check_pair(
    method_name: str,
    source_meta: ModelFamilyMetadata | None,
    target_meta: ModelFamilyMetadata | None,
    source_state_dict: Mapping[str, torch.Tensor] | None = None,
    target_state_dict: Mapping[str, torch.Tensor] | None = None,
    allow_depth_mismatch: bool = False,
) -> None:
    support = _METHOD_SUPPORT.get(method_name)
    if support is None:
        raise ValueError(
            f"Unknown rebase method '{method_name}'. "
            f"Supported: {sorted(_METHOD_SUPPORT)}"
        )

    if not support.required:
        raise ValueError(
            f"Method '{method_name}' is not available for text/decoder rebasing in v1. "
            f"Supported: {sorted(n for n, s in _METHOD_SUPPORT.items() if s.required)}"
        )

    if source_meta is None or target_meta is None:
        return

    if source_meta.family != target_meta.family:
        pair = frozenset({source_meta.family, target_meta.family})
        if pair not in _COMPATIBLE_CROSS_FAMILY_PAIRS:
            raise ValueError(
                f"Model family mismatch: source='{source_meta.family}', target='{target_meta.family}'. "
                "Cross-family rebasing is not supported in v1."
            )

    source_depth = source_meta.num_hidden_layers
    target_depth = target_meta.num_hidden_layers

    if source_depth > target_depth and not allow_depth_mismatch:
        raise ValueError(
            f"Source has more layers than target ({source_depth} > {target_depth}). "
            "Downsizing preprocess is not implemented in v1."
        )

    same_size = (
        source_meta.hidden_size == target_meta.hidden_size
        and source_meta.intermediate_size == target_meta.intermediate_size
    )

    if not same_size and not support.cross_size:
        raise ValueError(
            f"Method '{method_name}' does not support cross-size rebasing. "
            f"Source hidden={source_meta.hidden_size}/{source_meta.intermediate_size} "
            f"vs target hidden={target_meta.hidden_size}/{target_meta.intermediate_size}. "
            "Cross-size support: theseus, bico"
        )

    if same_size and source_depth != target_depth and not allow_depth_mismatch:
        raise ValueError(
            f"Same-size models have different depths ({source_depth} vs {target_depth}). "
            "Depth mismatch requires block-extension prealign."
        )


def needs_depth_upsize(
    source_meta: ModelFamilyMetadata,
    target_meta: ModelFamilyMetadata,
) -> bool:
    return source_meta.num_hidden_layers < target_meta.num_hidden_layers


def is_same_size(
    source_meta: ModelFamilyMetadata,
    target_meta: ModelFamilyMetadata,
) -> bool:
    return (
        source_meta.hidden_size == target_meta.hidden_size
        and source_meta.intermediate_size == target_meta.intermediate_size
    )
