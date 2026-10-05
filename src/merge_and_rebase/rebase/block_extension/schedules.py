"""Pure schedule and layout functions for BRACE block extension and reduction.

No model access: these map depths, insertion orders and ancestry chains to anchor
schedules and layout dicts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def spread_anchor_schedule(n_anchors: int, n_positions: int, insertion_order: str) -> list[int]:
    """Anchor blocks for ``extension_density="spread"``, spaced evenly over the depth.

    ``spread`` used to take the first ``n_anchors`` entries of an ordered
    priority list, which only spreads once at least one anchor per block is
    needed. Below that it piled every change onto one end of the model: growing
    28 -> 36 layers duplicated blocks 0-7 and left 8-27 untouched, measurably
    worse than spacing them out (Qwen2.5-1.5B, interpolate, no correction:
    wikitext-2 ppl 145 that way vs 26 spaced evenly). It now splits
    ``range(n_positions)`` into ``n_anchors`` equal runs and anchors at the
    start of each. ``spread_mod`` still reproduces the old bottom-top schedule
    for comparing against earlier results.

    Splitting into runs (rather than taking even fractions of the closed range
    ``[0, n_positions - 1]``) also keeps the last anchor one run short of the
    top, which matters: see the caller's note on why the final block must not
    become an anchor.

    ``insertion_order`` picks which end the anchors are laid out from;
    ``random`` keeps its meaning of an arbitrary (deliberately unspread) choice.
    """
    if n_anchors <= 0 or n_positions <= 0:
        return []

    if insertion_order == "random":
        anchors: list[int] = []
        while len(anchors) < n_anchors:
            cycle = list(range(n_positions))
            np.random.shuffle(cycle)
            anchors.extend(cycle[: n_anchors - len(anchors)])
        return anchors

    if insertion_order not in {"bottom-top", "top-bottom"}:
        raise ValueError(
            f"Unsupported insertion_order. Expected one of: bottom-top, top-bottom, random. Got: {insertion_order}"
        )

    anchors = [(i * n_positions) // n_anchors for i in range(n_anchors)]
    if insertion_order == "top-bottom":
        anchors = [n_positions - 1 - a for a in anchors]
    return anchors


def balanced_collapse_spans(curr_layers: int, final_depth: int, insertion_order: str) -> list[tuple[int, ...]]:
    """Partition ``curr_layers`` blocks into ``final_depth`` contiguous spans.

    This is the reduction-direction analogue of what
    :func:`spread_anchor_schedule` does for insertion, and it exists because
    the anchor schedule alone does not survive the reduction's own side
    effects. Anchors there are spaced over the *original* chain, but each
    collapse merges two entries, so a later anchor can land inside a span that
    an earlier step already merged and absorb a further block into it. At
    24 -> 12 that yields spans of sizes ``[3, 2, 2, ..., 2, 1]``: the bottom
    span swallows three blocks and the top block is never collapsed at all --
    precisely the pile-up at one end that ``spread`` was introduced to avoid.

    Splitting the depth up front instead makes each span disjoint by
    construction. Sizes are as equal as the depth allows, using the same
    ``(i * n) // k`` run split as the insertion schedule, and
    ``insertion_order`` decides which end any leftover blocks accumulate at.

    This is opt-in (``collapse_schedule='disjoint_spans'``). The cascading
    behaviour remains the default because every completed reduction campaign
    was produced with it.
    """
    if final_depth <= 0 or curr_layers <= 0:
        raise ValueError("balanced collapse spans need a positive depth")
    if final_depth > curr_layers:
        raise ValueError("cannot collapse into more spans than there are blocks")
    if insertion_order not in {"bottom-top", "top-bottom"}:
        raise ValueError(
            "collapse_schedule='disjoint_spans' supports insertion_order "
            "'bottom-top' or 'top-bottom'; 'random' has no meaning for a fixed "
            f"disjoint partition. Got: {insertion_order}"
        )
    sizes = [((i + 1) * curr_layers) // final_depth - (i * curr_layers) // final_depth for i in range(final_depth)]
    if insertion_order == "top-bottom":
        sizes = sizes[::-1]
    spans: list[tuple[int, ...]] = []
    start = 0
    for size in sizes:
        spans.append(tuple(range(start, start + size)))
        start += size
    return spans


def disjoint_collapse_schedule(curr_layers: int, n_to_remove: int, insertion_order: str) -> list[int]:
    """Anchor sequence realizing :func:`balanced_collapse_spans`.

    Each span is collapsed by repeatedly anchoring at its *first* original
    index: the merge loop locates the chain entry containing that index and
    absorbs the entry above it, so ``len(span) - 1`` repeats fold exactly that
    span and nothing else. Steps are emitted bottom-top; because anchors are
    original indices and merging one span never changes another span's
    membership, the step order cannot affect the realized partition.
    """
    spans = balanced_collapse_spans(curr_layers, curr_layers - n_to_remove, insertion_order)
    schedule: list[int] = []
    for span in spans:
        schedule.extend([span[0]] * (len(span) - 1))
    return schedule


def plan_inserted_positions(curr_layers: int, schedule: Sequence[int]) -> list[int]:
    """Final chain positions of each inserted block, in schedule order.

    The correction loop knows only ``insert_pos``, the position in the chain as
    it stands at that step, which later insertions below can still shift. A
    target-side reference has to be addressed by the block's *final* position,
    so replay the insertion bookkeeping on plain indices first. This mirrors
    ``_extend_per_weight`` exactly, including that a descendant shares its
    source's ``orig_idx`` and so is itself a valid insertion anchor.
    """
    chain: list[dict[str, Any]] = [{"orig_idx": i, "inserted": False, "step": None} for i in range(int(curr_layers))]
    for step, src_idx in enumerate(schedule):
        insert_pos = -1
        for i, item in enumerate(chain):
            if item["orig_idx"] == int(src_idx):
                insert_pos = i
        insert_pos += 1
        chain.insert(insert_pos, {"orig_idx": int(src_idx), "inserted": True, "step": step})

    positions = [0] * len(schedule)
    for position, item in enumerate(chain):
        if item["inserted"]:
            positions[int(item["step"])] = position
    return positions


def build_extension_layout(chain: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the realized block chain of an extended model.

    Positions index the extended ``resblocks`` list. Each inserted entry also
    records where the two original blocks that initialized it ended up, which
    is what the interpolated-activation baseline needs in order to address the
    pair of activation banks that bracket an inserted position.
    """
    original_positions: dict[int, int] = {}
    for position, item in enumerate(chain):
        if not bool(item.get("inserted", False)):
            original_positions[int(item["orig_idx"])] = position

    inserted_blocks: list[dict[str, int]] = []
    final_blocks: list[dict[str, Any]] = []
    for position, item in enumerate(chain):
        source_orig_idx = int(item["orig_idx"])
        if bool(item.get("inserted", False)):
            final_blocks.append({"position": position, "source_orig_idx": source_orig_idx, "block_kind": "inserted"})
        else:
            final_blocks.append({"position": position, "source_orig_idx": source_orig_idx, "block_kind": "original"})
        if not bool(item.get("inserted", False)):
            continue
        neighbour_orig_idx = int(item["neighbour_orig_idx"])
        inserted_blocks.append(
            {
                "position": position,
                "source_orig_idx": source_orig_idx,
                "neighbour_orig_idx": neighbour_orig_idx,
                "source_position": original_positions[source_orig_idx],
                "neighbour_position": original_positions[neighbour_orig_idx],
            }
        )

    return {
        # Labelled explicitly so a consumer never has to infer the direction
        # from "are there inserted blocks?". The reduction layout carries the
        # same two keys with different values (see build_reduction_layout).
        "direction": "extend",
        "p1_source_ancestry": "inserted_position",
        "final_depth": len(chain),
        "original_positions": {idx: original_positions[idx] for idx in sorted(original_positions)},
        "inserted_blocks": tuple(inserted_blocks),
        "final_blocks": tuple(final_blocks),
    }


def build_reduction_layout(chain: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the realized block chain of a *reduced* model.

    A reduction has no inserted blocks: every realized target position is a
    collapsed span of one or more original source blocks. A target residual
    solve therefore has to address every realized position (``target_scope
    ='all'``) and cannot use the extension's ``2i+1`` insertion arithmetic,
    which has no meaning here.

    The recorded ancestry is the span's **last** original block. That is not a
    convention chosen here: the collapsed block occupies the span's output
    boundary in the residual stream, and ``_shrink_per_weight`` already fits
    its correction against precisely that boundary (``span_end_idx``, and the
    input of the block above it). Recording the span end keeps a later P1
    solve reading the same boundary the structural correction was fitted at,
    instead of guessing a position. The full span is kept in
    ``span_orig_idxs`` so a consumer that needs the whole collapsed group --
    to report compression, or to address the span's input boundary -- does not
    have to reconstruct it.
    """
    final_blocks: list[dict[str, Any]] = []
    original_positions: dict[int, int] = {}
    for position, item in enumerate(chain):
        span = tuple(int(index) for index in item["orig_idxs"])
        if not span:
            raise ValueError(f"reduction chain position {position} has an empty source span")
        for index in span:
            original_positions[index] = position
        final_blocks.append(
            {
                "position": position,
                "source_orig_idx": span[-1],
                "span_orig_idxs": span,
                "block_kind": "collapsed" if len(span) > 1 else "original",
            }
        )
    return {
        "direction": "shrink",
        "p1_source_ancestry": "span_end_boundary",
        "final_depth": len(chain),
        # Many-to-one here, unlike the extension: every original block maps to
        # the realized position of the span that absorbed it.
        "original_positions": {idx: original_positions[idx] for idx in sorted(original_positions)},
        "inserted_blocks": (),
        "final_blocks": tuple(final_blocks),
    }


def vision_collapse_schedule(
    curr_layers: int,
    n_to_remove: int,
    insertion_order: str,
    extension_density: str,
):
    if n_to_remove <= 0:
        return []
    if curr_layers < 2:
        raise ValueError("Cannot collapse blocks when the model depth is less than 2.")

    max_anchor = curr_layers - 2
    if extension_density == "clump":
        if insertion_order == "top-bottom":
            return [max_anchor] * n_to_remove
        if insertion_order == "random":
            return [int(np.random.randint(0, max_anchor + 1)) for _ in range(n_to_remove)]
        if insertion_order != "bottom-top":
            raise ValueError(
                f"Unsupported insertion_order. Expected one of: bottom-top, top-bottom, random. Got: {insertion_order}"
            )
        return [0] * n_to_remove

    if extension_density == "spread":
        return spread_anchor_schedule(n_to_remove, max_anchor + 1, insertion_order)

    if extension_density != "spread_mod":
        raise ValueError(
            f"Unsupported extension_density. Expected one of: spread, spread_mod, clump. Got: {extension_density}"
        )

    if n_to_remove == 1:
        anchors = [0]
    else:
        anchors = [int(round(v)) for v in np.linspace(0, max_anchor, num=n_to_remove)]

    if insertion_order == "bottom-top":
        return anchors
    if insertion_order == "top-bottom":
        return [max_anchor - a for a in anchors]
    if insertion_order == "random":
        anchors = list(anchors)
        np.random.shuffle(anchors)
        return anchors
    raise ValueError(
        f"Unsupported insertion_order. Expected one of: bottom-top, top-bottom, random. Got: {insertion_order}"
    )


def vision_locate_collapse_pos(chain: list[dict[str, Any]], anchor_orig_idx: int) -> int:
    for pos, item in enumerate(chain):
        orig_idxs = item["orig_idxs"]
        if orig_idxs[0] <= anchor_orig_idx <= orig_idxs[-1]:
            return min(pos, len(chain) - 2)
    raise ValueError(f"Could not locate collapse anchor {anchor_orig_idx} in the current block chain.")


def decoder_collapse_schedule(
    curr_layers: int,
    n_to_remove: int,
    insertion_order: str,
    extension_density: str,
) -> list[int]:
    if n_to_remove <= 0:
        return []
    if curr_layers < 2:
        raise ValueError("Cannot collapse blocks when the model depth is less than 2.")

    max_anchor = curr_layers - 2
    if extension_density == "clump":
        if insertion_order == "top-bottom":
            return [max_anchor] * n_to_remove
        if insertion_order == "random":
            return [int(np.random.randint(0, max_anchor + 1)) for _ in range(n_to_remove)]
        if insertion_order != "bottom-top":
            raise ValueError(
                f"Unsupported insertion_order. Expected: bottom-top, top-bottom, random. Got: {insertion_order}"
            )
        return [0] * n_to_remove

    if extension_density == "spread_mod":
        n_gaps = curr_layers - 1
        return [i % n_gaps for i in range(n_to_remove)]

    if extension_density != "spread":
        raise ValueError(
            f"Unsupported extension_density. Expected: spread, spread_mod, clump. Got: {extension_density}"
        )

    return spread_anchor_schedule(n_to_remove, max_anchor + 1, insertion_order)


def decoder_locate_collapse_pos(chain: list[dict[str, Any]], anchor_orig_idx: int) -> int:
    for i, item in enumerate(chain):
        if anchor_orig_idx in item["orig_idxs"]:
            return i
    raise ValueError(f"Could not locate anchor_orig_idx={anchor_orig_idx} in chain.")
