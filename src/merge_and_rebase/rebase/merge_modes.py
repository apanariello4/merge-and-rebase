"""Merge-mode constants and config resolution shared by ``eval.vision_rebase`` and ``rebase.run_config``.

Moved verbatim from ``eval/vision_rebase/merge.py`` (layering: ``rebase`` must not import ``eval``);
that module re-exports these names as the same objects.
"""

from __future__ import annotations

from typing import Any

from ..merge.registry import get_method as get_merge_method

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
    try:
        get_merge_method(merge_method_name)  # validate early for clearer UX
    except KeyError as exc:  # B10: a config error is a ValueError (the registry keeps its KeyError)
        raise ValueError(str(exc.args[0])) from None

    raw_params = cfg.get("merge_params", {}) or {}
    if not isinstance(raw_params, dict):
        raise ValueError("merge_params must be a JSON object / mapping when provided.")

    global_alpha_search = cfg.get("global_alpha_search", True)
    if not isinstance(global_alpha_search, bool):
        raise ValueError("global_alpha_search must be a boolean (true/false).")
    return merge_mode, merge_method_name, dict(raw_params), global_alpha_search
