"""Merge-mode configuration and delta-space merge helpers for the vision rebase entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..io.ckpt import align_to_base_keys
from ..merge.base import PreparedMergeMethod
from ..merge.methods._common import axpy_state_dict
from ..merge.registry import get_method as get_merge_method


def _scale_delta(delta_sd: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
    w = float(weight)
    if w == 1.0:
        return delta_sd
    return {k: (v * w) for k, v in delta_sd.items()}


def _check_untransported_compatibility(
    base_sd: dict[str, torch.Tensor],
    delta_sd: dict[str, torch.Tensor],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for k, d in delta_sd.items():
        b = base_sd.get(k)
        if b is None:
            issues.append(f"{k}: missing in target base")
            continue
        if tuple(d.shape) != tuple(b.shape):
            issues.append(f"{k}: delta shape {tuple(d.shape)} != target shape {tuple(b.shape)}")
    return (len(issues) == 0), issues


def _average_visual_state_dicts(
    states_by_task: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Average independently transformed visual endpoint bases.

    Only floating-point visual tensors present with identical shapes in every
    task endpoint participate. This is deliberately separate from task-vector
    construction: callers must form each ``ft_ind_t - base_ind_t`` first.
    """
    if not states_by_task:
        raise ValueError("Cannot average independent endpoint bases for an empty task set.")

    task_items = list(states_by_task.items())
    common_keys = set(task_items[0][1])
    for _, state in task_items[1:]:
        common_keys &= set(state)

    usable_keys: list[str] = []
    for key in sorted(common_keys):
        values = [state[key] for _, state in task_items]
        if not key.startswith("visual.") or any(not value.is_floating_point() for value in values):
            continue
        shape = tuple(values[0].shape)
        if any(tuple(value.shape) != shape for value in values[1:]):
            continue
        usable_keys.append(key)

    if not usable_keys:
        raise ValueError("Independent endpoint bases have no common floating-point visual tensors to average.")

    average: dict[str, torch.Tensor] = {}
    for key in usable_keys:
        values = [state[key].detach().to(device="cpu", dtype=torch.float32) for _, state in task_items]
        average[key] = torch.stack(values, dim=0).mean(dim=0)
    return average, usable_keys


def _relative_visual_state_distance(
    state: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    keys: Sequence[str],
) -> float:
    """Return ||state-reference||_2 / ||reference||_2 over selected tensors."""
    distance_sq = torch.zeros((), dtype=torch.float64)
    reference_sq = torch.zeros((), dtype=torch.float64)
    for key in keys:
        value = state[key].detach().to(device="cpu", dtype=torch.float64)
        ref = reference[key].detach().to(device="cpu", dtype=torch.float64)
        distance_sq += torch.sum((value - ref) ** 2)
        reference_sq += torch.sum(ref**2)
    denominator = float(torch.sqrt(reference_sq))
    if denominator == 0.0:
        return 0.0 if float(torch.sqrt(distance_sq)) == 0.0 else float("inf")
    return float(torch.sqrt(distance_sq)) / denominator


_VALID_MERGE_MODES = (
    "none",
    "rebase_then_merge",
    "merge_then_rebase",
    "brace_transport_then_merge",
    "brace_merge_then_transport",
    "merge_then_brace_then_transport",
)


_TRANSPORT_THEN_MERGE_MODES = {"rebase_then_merge", "brace_transport_then_merge"}


_SINGLE_TRANSPORT_MODES = {
    "merge_then_rebase",
    "brace_merge_then_transport",
    "merge_then_brace_then_transport",
}


def _resolve_merge_mode_config(
    cfg: dict[str, Any],
    alpha_selection: str,
) -> tuple[str, str, dict[str, Any], bool]:
    """Resolve and validate the merge-mode knobs.

    Returns (merge_mode, merge_method_name, merge_params, global_alpha_search).
    ``merge_mode="none"`` keeps the historical per-task transfer evaluation;
    ``rebase_then_merge`` and its explicit campaign alias
    ``brace_transport_then_merge`` support hierarchical search (per-task alphas
    followed by a global merge alpha) when ``alpha_selection="per_task"``.
    """
    merge_mode = str(cfg.get("merge_mode", "none")).strip().lower()
    if merge_mode not in _VALID_MERGE_MODES:
        raise ValueError(f"merge_mode must be one of: {', '.join(_VALID_MERGE_MODES)}")

    if merge_mode not in _TRANSPORT_THEN_MERGE_MODES and merge_mode != "none" and alpha_selection == "per_task":
        raise ValueError(
            f"{merge_mode} requires alpha_selection='shared': per-task alpha search "
            "is only defined for individually transported deltas on the target base. "
            "Use merge_mode='brace_transport_then_merge' for hierarchical per-task alphas."
        )

    merge_method_name = str(cfg.get("merge_method", "task_arithmetic"))
    get_merge_method(merge_method_name)  # validate early for clearer UX

    raw_params = cfg.get("merge_params", {}) or {}
    if not isinstance(raw_params, dict):
        raise ValueError("merge_params must be a JSON object / mapping when provided.")

    global_alpha_search = cfg.get("global_alpha_search", True)
    if not isinstance(global_alpha_search, bool):
        raise ValueError("global_alpha_search must be a boolean (true/false).")
    return merge_mode, merge_method_name, dict(raw_params), global_alpha_search


def _pseudo_tuned(
    base_sd: dict[str, torch.Tensor],
    deltas: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Synthesize tuned states ``base + delta_i`` as expected by MergeMethod APIs."""
    return [axpy_state_dict(base_sd, d, alpha=1.0) for d in deltas]


def _scale_deltas_by(
    deltas: list[dict[str, torch.Tensor]],
    alphas: Sequence[float],
) -> list[dict[str, torch.Tensor]]:
    """Scale each delta by its own alpha (bakes per-task alphas into the deltas)."""
    if len(deltas) != len(alphas):
        raise ValueError("deltas and alphas must have the same length.")
    out: list[dict[str, torch.Tensor]] = []
    for delta, alpha in zip(deltas, alphas, strict=True):
        a = float(alpha)
        if a == 1.0:
            out.append(delta)
        else:
            out.append({k: v * a for k, v in delta.items()})
    return out


def _visual_key_fingerprint(sd: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compact shape fingerprint of a checkpoint's visual backbone for diagnostics."""
    depths = [k for k in sd if ".resblocks." in k]
    block_ids = {int(k.split(".resblocks.")[1].split(".")[0]) for k in depths if ".resblocks." in k}
    visual_widths = sorted({int(v.shape[0]) for k, v in sd.items() if k.startswith("visual.") and v.dim() >= 1})
    return {
        "n_visual_keys": sum(1 for k in sd if k.startswith("visual.")),
        "max_block_id": max(block_ids) if block_ids else None,
        "n_blocks": len(block_ids),
        "visual_out_dims_sample": visual_widths[:4],
    }


def _ckpt_visual_base_coverage(
    sd: Mapping[str, torch.Tensor],
    base_sd: Mapping[str, torch.Tensor],
) -> float:
    """Fraction of the base's visual keys covered (key + shape) by sd after conservative alignment."""
    visual_base_keys = [k for k, v in base_sd.items() if k.startswith("visual.") and isinstance(v, torch.Tensor)]
    if not visual_base_keys:
        return 0.0
    aligned = align_to_base_keys(sd, base_sd)
    covered = sum(1 for k in visual_base_keys if k in aligned)
    return covered / len(visual_base_keys)


def _infer_ckpt_base(
    sd: Mapping[str, torch.Tensor],
    *,
    source_base_sd: Mapping[str, torch.Tensor],
    target_base_sd: Mapping[str, torch.Tensor],
    native_coverage_threshold: float = 0.5,
) -> str | None:
    """Classify a raw tuned checkpoint as 'source' or 'target' by visual-key coverage.

    Returns 'target' when the checkpoint matches the target visual backbone
    (a native target checkpoint), 'source' otherwise (ties prefer source so
    transport stays the default), or None when it matches neither base.
    """
    coverage_source = _ckpt_visual_base_coverage(sd, source_base_sd)
    coverage_target = _ckpt_visual_base_coverage(sd, target_base_sd)
    if coverage_target >= native_coverage_threshold and coverage_target > coverage_source:
        return "target"
    if coverage_source > 0.0:
        return "source"
    return None


def _merge_direction(
    *,
    base_sd: dict[str, torch.Tensor],
    deltas: list[dict[str, torch.Tensor]],
    merge_method_name: str,
    weights: Sequence[float],
    merge_params: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Merge task deltas into one direction in delta-space via the public merge registry.

    For PreparedMergeMethod implementations ``prepare`` already yields the
    weighted direction; plain methods are merged at alpha=1.0 and the base is
    subtracted back out. The result feeds ``rebased_deltas=[direction] * n`` so
    the existing per-task sweep machinery evaluates the same merged model.
    """
    merge_method = get_merge_method(merge_method_name)
    tuned = _pseudo_tuned(base_sd, deltas)
    method_kwargs = dict(merge_params)

    if isinstance(merge_method, PreparedMergeMethod):
        _prepared_base, direction = merge_method.prepare(
            base=base_sd,
            tuned=tuned,
            weights=list(weights),
            **method_kwargs,
        )
        return dict(direction)

    merged = merge_method.merge(base=base_sd, tuned=tuned, weights=list(weights), **method_kwargs)
    return {k: v - base_sd[k] for k, v in merged.items() if k in base_sd}
