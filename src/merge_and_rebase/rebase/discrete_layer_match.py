"""Discrete layer pairing between two transformer depths.

The pairing formula ``i(j) = round(j * (D_A - 1) / (D_B - 1))`` maps each
target-model block position ``j`` in ``[0, D_B)`` to the source-model block
position ``i(j)`` in ``[0, D_A)`` it is aligned with, where ``D_A`` is
``source_depth`` and ``D_B`` is ``target_depth``. It is a purely discrete,
index-only correspondence: no interpolation, no weight blending, no fit.

This is the one piece of code that both Direct Residual
(``merge_and_rebase.eval.direct_residual``) and the faithful BiCo/THESEUS
structural-resize control must agree on bit-for-bit, so it is defined exactly
once, here, and imported by both consumers.

``round()`` is Python's built-in banker's rounding (round-half-to-even); it is
used as-is rather than replaced with a different rounding convention such as
round-half-away-from-zero, so the exact tie-breaking behaviour at any
half-integer boundary is whatever the interpreter's built-in gives.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

Tensor = torch.Tensor

# Matches a resblock-scoped state-dict key, with or without a leading module
# prefix (e.g. "visual." for a full CLIP model state dict, or none at all for
# a state dict already scoped to the visual tower). Group 1 is the block
# index, group 2 is everything after the trailing dot (e.g.
# "mlp.c_proj.weight").
_DEFAULT_BLOCK_PATTERN = r"^(?P<prefix>.*?)transformer\.resblocks\.(?P<index>\d+)\.(?P<suffix>.+)$"


def discrete_layer_pairing(source_depth: int, target_depth: int) -> list[int]:
    """Discrete block-index pairing from a target depth to a source depth.

    Returns a list of length ``target_depth`` whose ``j``-th entry is the
    source block index ``i(j) = round(j * (source_depth - 1) / (target_depth - 1))``
    that target position ``j`` is paired with. ``target_depth == 1`` is a
    special case (``[0]``), since the formula's denominator would otherwise
    be zero.
    """
    if source_depth < 1:
        raise ValueError(f"source_depth must be >= 1, got {source_depth}")
    if target_depth < 1:
        raise ValueError(f"target_depth must be >= 1, got {target_depth}")

    if target_depth == 1:
        return [0]

    scale = (source_depth - 1) / (target_depth - 1)
    return [round(j * scale) for j in range(target_depth)]


@dataclass(frozen=True)
class DiscreteLayerPairing:
    """The pairing formula's result, bundled with the depths it was computed from."""

    source_depth: int
    target_depth: int
    pairing: tuple[int, ...]  # len == target_depth, values in [0, source_depth)

    @classmethod
    def compute(cls, source_depth: int, target_depth: int) -> DiscreteLayerPairing:
        pairing = tuple(discrete_layer_pairing(source_depth, target_depth))
        return cls(source_depth=source_depth, target_depth=target_depth, pairing=pairing)


def reindex_state_dict(
    source_sd: Mapping[str, Tensor],
    pairing: DiscreteLayerPairing,
    *,
    block_pattern: str = _DEFAULT_BLOCK_PATTERN,
) -> dict[str, Tensor]:
    """Reindex a resblock-scoped state dict from source depth to target depth.

    Target block ``j``'s tensors are the source block ``pairing.pairing[j]``'s
    tensors, verbatim -- no interpolation, no scaling, no duplication-aware
    reweighting. Keys that do not match ``block_pattern`` (patch embed,
    positional embedding, ln_pre/ln_post, proj, and anything else outside the
    resblock stack) pass through unchanged, copied from ``source_sd`` as-is.
    """
    pattern = re.compile(block_pattern)

    by_source_index: dict[int, list[tuple[str, str, Tensor]]] = {}
    passthrough: dict[str, Tensor] = {}

    for key, tensor in source_sd.items():
        match = pattern.match(key)
        if match is None:
            passthrough[key] = tensor
            continue
        source_index = int(match.group("index"))
        by_source_index.setdefault(source_index, []).append((match.group("prefix"), match.group("suffix"), tensor))

    out: dict[str, Tensor] = dict(passthrough)
    for target_index, source_index in enumerate(pairing.pairing):
        entries = by_source_index.get(source_index)
        if entries is None:
            raise ValueError(
                f"pairing references source block {source_index}, but source_sd has no "
                f"keys matching block_pattern for that index"
            )
        for prefix, suffix, tensor in entries:
            out[f"{prefix}transformer.resblocks.{target_index}.{suffix}"] = tensor

    return out


def build_discrete_indexed_model(
    source_model: nn.Module,
    pairing: DiscreteLayerPairing,
    *,
    resblocks_attr: str = "visual.transformer.resblocks",
) -> nn.Module:
    """Build a depth-reindexed copy of ``source_model`` per ``pairing``.

    Deep-copies ``source_model``, then reassigns the resblocks ``ModuleList``
    at ``resblocks_attr`` to
    ``nn.ModuleList([deepcopy(orig[pairing.pairing[j]]) for j in range(target_depth)])``,
    mirroring the resblocks-reassignment pattern used throughout the codebase's
    block-extension machinery (``block_extension.py``). No correction, no
    cascade, no interpolation is applied.
    """
    result = deepcopy(source_model)

    *parent_parts, attr_name = resblocks_attr.split(".")
    parent = operator.attrgetter(".".join(parent_parts))(result) if parent_parts else result
    orig = getattr(parent, attr_name)

    new_resblocks = nn.ModuleList([deepcopy(orig[pairing.pairing[j]]) for j in range(pairing.target_depth)])
    setattr(parent, attr_name, new_resblocks)

    return result
