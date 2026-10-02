"""Delta-level merge helpers: norm bookkeeping, norm-match rescale, merged-delta summary and zero-gate."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _delta_norm(delta: Mapping[str, torch.Tensor], keys: Iterable[str] | None = None) -> float:
    """Frobenius norm of a task vector, optionally restricted to `keys`."""
    total = 0.0
    for key, value in delta.items():
        if keys is not None and key not in keys:
            continue
        total += float(value.float().pow(2).sum())
    return total ** 0.5


def resolve_delta_source(cfg: Mapping[str, Any]) -> str:
    # Which task vector gets transported. "corrected" is the status quo:
    # activations and delta both come from the corrected resize.
    # "uncorrected" keeps the corrected model for activation capture -- so
    # the fitted alignment map is unchanged -- but transports the delta from
    # the uncorrected resize, isolating whether correction helps the map or
    # only distorts the vector.
    delta_source = str(cfg.get("transport_delta_source", "corrected")).strip().lower()
    if delta_source not in {"corrected", "uncorrected"}:
        raise ValueError(f"transport_delta_source must be 'corrected' or 'uncorrected'. Got: {delta_source!r}")
    return delta_source


def resolve_norm_match(cfg: Mapping[str, Any]) -> str | None:
    # Rescale the transported delta to the uncorrected task vector's norm.
    # Procrustes transport is orthogonal and norm-preserving, so without
    # this the correction's effect on scale reaches the target model in full
    # and a fixed alpha cannot distinguish scale from direction.
    norm_match = cfg.get("delta_norm_match", None)
    norm_match = str(norm_match).strip().lower() if norm_match is not None else None
    if norm_match not in {None, "none", "uncorrected"}:
        raise ValueError(f"delta_norm_match must be null or 'uncorrected'. Got: {norm_match!r}")
    return norm_match


def norm_match_transported(
    transported: dict[str, torch.Tensor],
    *,
    corrected_delta: Mapping[str, torch.Tensor],
    reference_delta: Mapping[str, torch.Tensor],
    transport_keys: Iterable[str] | None,
    norm_match: str | None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    # Norms are reported for every run, not only when matching is on, so
    # the scale effect of correction is visible in the summary.
    n_corrected = _delta_norm(corrected_delta, transport_keys)
    n_uncorrected = _delta_norm(reference_delta, transport_keys)
    n_transported = _delta_norm(transported)
    scale = 1.0
    if norm_match == "uncorrected" and n_transported > 0.0:
        scale = n_uncorrected / n_transported
        transported = {k: v * scale for k, v in transported.items()}
    norms = {
        "source_corrected": n_corrected,
        "source_uncorrected": n_uncorrected,
        "transported_before_match": n_transported,
        "norm_match_scale": scale,
        "transported_after_match": n_transported * scale,
    }
    print(
        f"  ||tv|| source corrected={n_corrected:.2f} uncorrected={n_uncorrected:.2f}"
        f" transported={n_transported:.2f}" + (f" -> rescaled x{scale:.3e}" if norm_match == "uncorrected" else "")
    )
    return transported, norms


def _summarize_merged_delta(
    merged_delta: dict[str, torch.Tensor],
    target_base: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Measure how much of the target model the transported delta actually moves.

    A transport that silently zeroes every key still returns a full set of
    correctly shaped tensors, so the alpha sweep looks healthy while every
    candidate evaluates the same untouched base model. These numbers go into the
    run summary so that failure mode is visible in the JSON, not only in stdout.
    """
    sq_delta = 0.0
    sq_base = 0.0
    nonzero = 0
    for key, value in merged_delta.items():
        val = value.float()
        sq_delta += float(val.pow(2).sum())
        if float(val.abs().sum()) > 0.0:
            nonzero += 1
        base_ref = target_base.get(key)
        if base_ref is not None:
            sq_base += float(base_ref.float().pow(2).sum())

    delta_norm = sq_delta ** 0.5
    base_norm = sq_base ** 0.5
    return {
        "key_count": float(len(merged_delta)),
        "nonzero_key_count": float(nonzero),
        "merged_delta_norm": delta_norm,
        "merged_delta_rel_norm": (delta_norm / base_norm) if base_norm > 0.0 else 0.0,
    }


def report_merged_delta(delta_stats: Mapping[str, float]) -> None:
    print(
        f"\nMerged delta: keys={delta_stats['key_count']} "
        f"nonzero_keys={delta_stats['nonzero_key_count']} "
        f"norm={delta_stats['merged_delta_norm']:.4f} "
        f"rel_norm={delta_stats['merged_delta_rel_norm']:.6f}"
    )
    if delta_stats["nonzero_key_count"] == 0:
        raise RuntimeError(
            "Merged transported delta is identically zero: every alpha would evaluate "
            "the untouched target base model. Check the transport diagnostics above."
        )
