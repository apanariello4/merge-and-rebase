"""Depth pairings derived from the BRACE structural schedule (pure; no model, no extender run).

``spread_duplicate_pairing`` answers: if BRACE resized a ``source_depth`` stack to ``target_depth``, which
source block is the ancestor of each realized target position? It replays the same schedule and chain
bookkeeping as the vision ``BlockExtender`` (``BlockExtenderCore._extend_per_weight`` /
``_shrink_per_weight``) on plain indices, so it equals the realized layout of a real
``BlockExtender(skip_correction=True)`` run (pinned by ``tests/test_spread_duplicate_pairing_20261002.py``).

* Extension: an original block is its own ancestor; an inserted block's ancestor is the original block it was
  duplicated from (``source_orig_idx`` of ``build_extension_layout``).
* Shrink (the VISION collapse schedule and locate logic): a collapsed span maps to its FIRST original index.
  ``build_reduction_layout`` records the span END as ``source_orig_idx``; the full span is kept in
  ``span_orig_idxs``, whose first element is what this pairing uses (D-P6b).
* The schedule is fixed to BRACE's defaults (``bottom-top``, ``spread``; vision default ``cascade`` collapse); there are
  no schedule parameters, and random insertion has no deterministic pairing.
"""

from __future__ import annotations

from typing import Any

from .block_extension.core import BlockExtenderCore
from .block_extension.schedules import (
    vision_collapse_schedule,
    vision_locate_collapse_pos,
)
from .discrete_layer_match import DiscreteLayerPairing


def spread_duplicate_pairing(source_depth: int, target_depth: int) -> DiscreteLayerPairing:
    """``pairing[j]`` = source block that is the BRACE ancestor of target position ``j``."""
    source_depth, target_depth = int(source_depth), int(target_depth)
    if source_depth <= 0 or target_depth <= 0:
        raise ValueError("brace ancestry pairing needs positive depths")
    insertion_order, extension_density = "bottom-top", "spread"
    if target_depth == source_depth:
        return DiscreteLayerPairing(source_depth, target_depth, tuple(range(source_depth)))

    if target_depth > source_depth:
        schedule = BlockExtenderCore._build_duplication_schedule(
            curr_layers=source_depth,
            n_needed=target_depth - source_depth,
            insertion_order=insertion_order,
            extension_density=extension_density,
        )
        chain: list[int] = list(range(source_depth))  # ancestor (orig_idx) per position
        for src_idx in schedule:
            insert_pos = -1
            for i, orig_idx in enumerate(chain):
                if orig_idx == int(src_idx):
                    insert_pos = i
            chain.insert(insert_pos + 1, int(src_idx))
        return DiscreteLayerPairing(source_depth, target_depth, tuple(chain))

    n_to_remove = source_depth - target_depth
    schedule = vision_collapse_schedule(source_depth, n_to_remove, insertion_order, extension_density)
    spans: list[dict[str, Any]] = [{"orig_idxs": (i,)} for i in range(source_depth)]
    for anchor in schedule:
        pos = vision_locate_collapse_pos(spans, int(anchor))
        merged = tuple(spans[pos]["orig_idxs"] + spans[pos + 1]["orig_idxs"])
        spans[pos : pos + 2] = [{"orig_idxs": merged}]
    return DiscreteLayerPairing(source_depth, target_depth, tuple(int(s["orig_idxs"][0]) for s in spans))
