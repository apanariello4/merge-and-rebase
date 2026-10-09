from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from .model_families.base import ModelFamilyMetadata
from .registry import canonical_method_name


class MethodFamily(enum.Enum):
    """How a rebase method obtains the target task vector."""

    #: The source task vector is mapped through maps estimated from calibration activations / gradients
    #: (THESEUS, BiCo).
    ACTIVATION_ALIGNED = "activation_aligned"
    #: The source task vector is transformed in weight space (permutation, gradient-sign mask, identity, rotation).
    WEIGHT_SPACE = "weight_space"
    #: The target task vector is fitted from paired source/target activations; nothing is transported (Ariadne).
    DIRECT_FIT = "direct_fit"


@dataclass(frozen=True)
class MethodTraits:
    """Static traits of a registered rebase method: the single table every dispatch decision reads."""

    family: MethodFamily
    cross_size: bool = False
    required: bool = True
    # Optional method-specific explanation used instead of the generic message when
    # ``required`` is False.
    unavailable_reason: str | None = None
    # Depth-mismatched pairs are handled by the method itself (no block-extension prealign needed).
    any_depth: bool = False
    # The method reads calibration activations; on LLM harness runs that calibration text is held out of the
    # evaluation docs, so the evaluation slice must be resolved before the first evaluation.
    calibrates: bool = False
    # Default depth prestep for a depth-mismatched pair (``None``: the method runs no depth prestep) ...
    depth_prestep: Literal["brace", "discrete_index_match"] | None = None
    # ... and the entrypoints that run it (``theseus_gqa`` exists only for decoders).
    depth_prestep_entrypoints: frozenset[str] = frozenset({"vision", "llm"})


_ACT = MethodFamily.ACTIVATION_ALIGNED
_WEIGHT = MethodFamily.WEIGHT_SPACE

METHOD_TRAITS: dict[str, MethodTraits] = {
    "theseus": MethodTraits(_ACT, cross_size=True, calibrates=True, depth_prestep="brace"),
    "theseus_gqa": MethodTraits(
        _ACT, cross_size=True, calibrates=True, depth_prestep="brace", depth_prestep_entrypoints=frozenset({"llm"})
    ),
    "bico": MethodTraits(_ACT, cross_size=True, calibrates=True, depth_prestep="discrete_index_match"),
    "identity": MethodTraits(_WEIGHT, cross_size=False),
    "orthogonal_shift": MethodTraits(_WEIGHT, cross_size=False),
    "gradfix": MethodTraits(_WEIGHT, cross_size=False),
    "transfusion": MethodTraits(_WEIGHT, cross_size=False, required=False),
    # "direct_residual" is a registry alias of "ariadne" and resolves to this entry.
    "ariadne": MethodTraits(MethodFamily.DIRECT_FIT, cross_size=True, any_depth=True, calibrates=True),
}


def method_traits(method_name: str) -> MethodTraits | None:
    """Traits of a registered method (aliases resolved); ``None`` for an unknown name."""
    return METHOD_TRAITS.get(canonical_method_name(method_name))


def method_family(method_name: str) -> MethodFamily | None:
    traits = method_traits(method_name)
    return None if traits is None else traits.family


def depth_prestep_methods(entrypoint: str) -> frozenset[str]:
    """Methods whose depth-mismatched pairs run the depth prestep in ``entrypoint`` (``"vision"`` / ``"llm"``)."""
    return frozenset(
        name
        for name, traits in METHOD_TRAITS.items()
        if traits.depth_prestep is not None and entrypoint in traits.depth_prestep_entrypoints
    )


def default_depth_prestep(method_name: str, entrypoint: str) -> str | None:
    """The method's default depth prestep rule in ``entrypoint``, or ``None`` when it runs none there."""
    if method_name not in depth_prestep_methods(entrypoint):
        return None
    return METHOD_TRAITS[method_name].depth_prestep


def supports_cross_size(method_name: str) -> bool:
    """Whether the method can rebase across different hidden/intermediate sizes."""
    support = method_traits(method_name)
    if support is None:
        raise ValueError(f"Unknown rebase method '{method_name}'. Supported: {sorted(METHOD_TRAITS)}")
    return support.cross_size


def is_text_supported(method_name: str) -> bool:
    """Whether the method is wired for text/decoder rebasing."""
    support = method_traits(method_name)
    if support is None:
        raise ValueError(f"Unknown rebase method '{method_name}'. Supported: {sorted(METHOD_TRAITS)}")
    return support.required


def uses_calibration(method_name: str) -> bool:
    """Whether the method fits on calibration activations (unknown methods: no)."""
    support = method_traits(method_name)
    return bool(support is not None and support.calibrates)


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
    block_extension_params: Mapping[str, object] | None = None,
) -> None:
    support = method_traits(method_name)
    if support is None:
        raise ValueError(
            f"Unknown rebase method '{method_name}'. "
            f"Supported: {sorted(METHOD_TRAITS)}"
        )

    if not support.required:
        if support.unavailable_reason is not None:
            raise ValueError(
                f"Method '{method_name}' is not available for text/decoder rebasing: {support.unavailable_reason} "
                f"Supported: {sorted(n for n, s in METHOD_TRAITS.items() if s.required)}"
            )
        raise ValueError(
            f"Method '{method_name}' is not available for text/decoder rebasing in v1. "
            f"Supported: {sorted(n for n, s in METHOD_TRAITS.items() if s.required)}"
        )

    if source_meta is None or target_meta is None:
        return

    for role, meta in (("source", source_meta), ("target", target_meta)):
        if getattr(meta, "is_moe", False):
            raise ValueError(
                f"Mixture-of-experts {role} model (family '{meta.family}') is not supported: "
                "block transport assumes a dense MLP (mlp.gate_proj/up_proj/down_proj)."
            )

    if source_meta.family != target_meta.family:
        pair = frozenset({source_meta.family, target_meta.family})
        if pair not in _COMPATIBLE_CROSS_FAMILY_PAIRS:
            raise ValueError(
                f"Model family mismatch: source='{source_meta.family}', target='{target_meta.family}'. "
                "Cross-family rebasing is not supported in v1."
            )

    source_depth = source_meta.num_hidden_layers
    target_depth = target_meta.num_hidden_layers

    # Ariadne pairs blocks itself; BiCo with a discrete index match works on the reindexed stack.
    if support.any_depth or (
        support.depth_prestep == "discrete_index_match"
        and resolve_depth_strategy(method_name, block_extension_params, source_meta, target_meta).rule
        == "discrete_index_match"
    ):
        allow_depth_mismatch = True

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


_LEGACY_DEPTH_KEYS = ("depth_rule", "extension_strategy", "skip_correction")


@dataclass(frozen=True)
class DepthStrategy:
    rule: str  # "none" | "brace" | "discrete_index_match"
    skip_correction: bool | None = None
    extension_strategy: str | None = None
    legacy: bool = False


def resolve_depth_strategy(
    method_name: str,
    block_extension_params: Mapping[str, object] | None,
    source_meta: ModelFamilyMetadata | None,
    target_meta: ModelFamilyMetadata | None,
) -> DepthStrategy:
    """Depth rule for a decoder pair (pure).

    Defaults: THESEUS-like -> BRACE ``interpolate_per_weight`` + ``skip_correction=True``; BiCo -> discrete index
    match on the reindexed stack; Ariadne -> none (it pairs blocks itself). Equal depths -> ``none``.
    Any legacy key in ``block_extension_params`` (depth_rule / extension_strategy / skip_correction) keeps the
    legacy semantics: the given values verbatim (unset ``skip_correction`` is the default, True).
    """
    traits = method_traits(method_name)
    params = block_extension_params or {}
    # Methods without a depth prestep (Ariadne pairs blocks itself) and equal depths need no rule.
    if traits is None or traits.depth_prestep is None:
        return DepthStrategy(rule="none")
    if source_meta is not None and target_meta is not None:
        if source_meta.num_hidden_layers == target_meta.num_hidden_layers:
            return DepthStrategy(rule="none")
    if any(k in params for k in _LEGACY_DEPTH_KEYS):
        rule = str(params.get("depth_rule") or "")
        if rule in ("", "method_default"):
            rule = (
                "discrete_index_match"
                if traits.depth_prestep == "discrete_index_match" and "skip_correction" not in params
                else "brace"
            )
        ext = params.get("extension_strategy")
        return DepthStrategy(
            rule=rule,
            skip_correction=bool(params.get("skip_correction", True)),
            extension_strategy=None if ext is None else str(ext),
            legacy=True,
        )
    if traits.depth_prestep == "discrete_index_match":
        return DepthStrategy(rule="discrete_index_match")
    return DepthStrategy(rule="brace", skip_correction=True, extension_strategy="interpolate_per_weight")
