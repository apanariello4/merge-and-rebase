"""Conditioning applied to transported task vectors immediately before merging.

The BRACE -> transport -> merge pipeline sums task vectors that a width
transport has already mapped into the target coordinate frame.  Two properties
of that transported set are not controlled by the merger itself:

* **Scale heterogeneity.**  An orthogonal Procrustes transport preserves the
  norm of whatever it captures and discards the rest, so the usable magnitude
  of a transported vector varies per task.  The 2026-09-13 campaign selected
  per-task pre-merge alphas spanning 0.0 to 9.9 for Theseus against 0.8 to 4.7
  for BICO, which is that heterogeneity made visible.  Under one shared merge
  alpha the tasks therefore do not contribute comparable energy.
* **Transport noise.**  A Procrustes rotation is only determined on the
  high-variance part of the activation spectrum, so the transported vector
  carries an unidentified low-variance residual.  Summing eight such residuals
  accumulates them.  Whitening-style mergers amplify that residual, whereas
  magnitude or rank truncation removes it.
* **A shared spurious direction.**  Measured on the 2026-09-14 bank, the eight
  transported Theseus vectors have mean pairwise ``|cos|`` 0.0056 against
  BICO's 0.0006, and their sum is superadditive (1.0196 against 1.0011 times
  the root-sum-square of the individual norms).  Both transports are
  norm-preserving semi-orthogonal maps of the same source task vectors, so the
  per-task norms are *identical* between the two methods: whatever separates
  them at the merge stage is directional, and part of it is a component common
  to every task.  ``mean_removal`` subtracts that component.

Both are corrections to the *inputs* of the merger, not to the merger, so they
live here and are applied by the runner between the hash-checked transported
bank and ``merge.prepare``.  ``mode="off"`` is the default and returns the
deltas unchanged, so an existing protocol never shifts silently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

MODES = ("off", "norm_match", "mean_removal", "magnitude_trim", "rank_trim")
_NORM_SCOPES = ("global", "per_tensor")
_TRIM_SCOPES = ("global", "per_tensor")
_REFERENCES = ("mean", "geomean", "median")

TensorDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ConditioningSpec:
    """One conditioning operation, fully described for the run summary."""

    mode: str = "off"
    scope: str = "global"
    reference: str = "mean"
    density: float = 1.0
    rescale: bool = False
    min_ndim: int = 2
    strength: float = 1.0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"tv_conditioning.mode must be one of {MODES}; got {self.mode!r}.")
        if self.mode == "norm_match":
            if self.scope not in _NORM_SCOPES:
                raise ValueError(f"norm_match scope must be one of {_NORM_SCOPES}; got {self.scope!r}.")
            if self.reference not in _REFERENCES:
                raise ValueError(f"norm_match reference must be one of {_REFERENCES}; got {self.reference!r}.")
        if self.mode in {"magnitude_trim", "rank_trim"}:
            if not 0.0 < float(self.density) <= 1.0:
                raise ValueError(f"{self.mode} density must be in (0, 1]; got {self.density!r}.")
            if self.mode == "magnitude_trim" and self.scope not in _TRIM_SCOPES:
                raise ValueError(f"magnitude_trim scope must be one of {_TRIM_SCOPES}; got {self.scope!r}.")
        if self.mode == "mean_removal" and not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError(f"mean_removal strength must be in [0, 1]; got {self.strength!r}.")
        if int(self.min_ndim) < 1:
            raise ValueError("min_ndim must be >= 1.")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode}
        if self.mode == "norm_match":
            payload.update(scope=self.scope, reference=self.reference)
        elif self.mode == "mean_removal":
            payload.update(strength=float(self.strength))
        elif self.mode == "magnitude_trim":
            payload.update(scope=self.scope, density=float(self.density), rescale=bool(self.rescale))
        elif self.mode == "rank_trim":
            payload.update(density=float(self.density), min_ndim=int(self.min_ndim), rescale=bool(self.rescale))
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def spec_from_config(raw: Mapping[str, Any] | None) -> ConditioningSpec:
    """Build a spec from a config block, defaulting to the untouched pipeline."""

    if not raw:
        return ConditioningSpec()
    known = {"mode", "scope", "reference", "density", "rescale", "min_ndim", "strength"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown tv_conditioning fields: {unknown}.")
    return ConditioningSpec(
        mode=str(raw.get("mode", "off")).strip().lower(),
        scope=str(raw.get("scope", "global")).strip().lower(),
        reference=str(raw.get("reference", "mean")).strip().lower(),
        density=float(raw.get("density", 1.0)),
        rescale=bool(raw.get("rescale", False)),
        min_ndim=int(raw.get("min_ndim", 2)),
        strength=float(raw.get("strength", 1.0)),
    )


def _global_norm(delta: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in delta.values():
        total += float(tensor.detach().to(torch.float64).pow(2).sum())
    return math.sqrt(total)


def _reference_value(values: list[float], reference: str) -> float:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return 0.0
    if reference == "mean":
        return sum(positive) / len(positive)
    if reference == "geomean":
        return math.exp(sum(math.log(value) for value in positive) / len(positive))
    ordered = sorted(positive)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _norm_match(
    deltas: Mapping[str, TensorDict], spec: ConditioningSpec
) -> tuple[dict[str, TensorDict], dict[str, Any]]:
    tasks = list(deltas)
    if spec.scope == "global":
        norms = {task: _global_norm(deltas[task]) for task in tasks}
        target = _reference_value(list(norms.values()), spec.reference)
        scales = {task: (target / norms[task] if norms[task] > 0.0 else 1.0) for task in tasks}
        out = {
            task: {key: tensor * scales[task] for key, tensor in deltas[task].items()}
            for task in tasks
        }
        return out, {
            "reference_norm": target,
            "pre_global_norm": norms,
            "scale": scales,
        }

    keys = sorted({key for task in tasks for key in deltas[task]})
    out = {task: dict(deltas[task]) for task in tasks}
    per_key_scale: dict[str, dict[str, float]] = {}
    per_key_reference: dict[str, float] = {}
    for key in keys:
        norms = {
            task: float(deltas[task][key].detach().to(torch.float64).norm())
            for task in tasks
            if key in deltas[task]
        }
        target = _reference_value(list(norms.values()), spec.reference)
        per_key_reference[key] = target
        per_key_scale[key] = {}
        for task, norm in norms.items():
            scale = target / norm if norm > 0.0 else 1.0
            per_key_scale[key][task] = scale
            out[task][key] = deltas[task][key] * scale
    return out, {
        "reference_norm_by_key": per_key_reference,
        "scale_by_key": per_key_scale,
    }


def _mean_removal(
    deltas: Mapping[str, TensorDict], spec: ConditioningSpec
) -> tuple[dict[str, TensorDict], dict[str, Any]]:
    """Subtract ``strength`` times the set's centroid from every task vector.

    At ``strength = 1`` the conditioned set has zero mean, so task arithmetic's
    summed direction is exactly zero and only the merger's own reweighting
    survives; the interesting regime is therefore the partial removal that
    keeps the shared component from dominating without deleting the sum.
    """

    tasks = list(deltas)
    keys = sorted({key for task in tasks for key in deltas[task]})
    strength = float(spec.strength)
    centroid: TensorDict = {}
    for key in keys:
        present = [deltas[task][key] for task in tasks if key in deltas[task]]
        stacked = torch.stack([tensor.detach().to(torch.float64) for tensor in present])
        centroid[key] = stacked.mean(dim=0)

    out: dict[str, TensorDict] = {}
    removed_fraction: dict[str, float] = {}
    for task in tasks:
        conditioned: TensorDict = {}
        removed_sq = 0.0
        original_sq = 0.0
        for key, tensor in deltas[task].items():
            shifted = tensor.detach().to(torch.float64) - strength * centroid[key]
            original_sq += float(tensor.detach().to(torch.float64).pow(2).sum())
            removed_sq += float((strength * centroid[key]).pow(2).sum())
            conditioned[key] = shifted.to(dtype=tensor.dtype)
        out[task] = conditioned
        removed_fraction[task] = math.sqrt(removed_sq / original_sq) if original_sq else 0.0

    return out, {
        "strength": strength,
        "centroid_norm": math.sqrt(sum(float(t.pow(2).sum()) for t in centroid.values())),
        "removed_over_original_norm": removed_fraction,
    }


def _magnitude_trim_one(delta: TensorDict, spec: ConditioningSpec) -> tuple[TensorDict, dict[str, float]]:
    density = float(spec.density)
    factor = (1.0 / density) if spec.rescale else 1.0
    kept = 0
    total = 0
    out: TensorDict = {}

    if spec.scope == "global":
        flat = torch.cat([tensor.detach().reshape(-1).abs().to(torch.float32) for tensor in delta.values()])
        total = int(flat.numel())
        keep = max(1, int(math.ceil(density * total)))
        if keep >= total:
            threshold = -1.0
        else:
            # kthvalue over the ascending order: everything strictly above the
            # (total - keep)-th smallest magnitude is retained.
            threshold = float(torch.kthvalue(flat, total - keep).values)
        del flat
        for key, tensor in delta.items():
            mask = tensor.detach().abs() > threshold
            kept += int(mask.sum())
            out[key] = tensor * mask.to(tensor.dtype) * factor
    else:
        for key, tensor in delta.items():
            values = tensor.detach().reshape(-1).abs()
            numel = int(values.numel())
            total += numel
            keep = max(1, int(math.ceil(density * numel)))
            if keep >= numel:
                mask = torch.ones_like(tensor, dtype=torch.bool)
            else:
                threshold = float(torch.kthvalue(values.to(torch.float32), numel - keep).values)
                mask = tensor.detach().abs() > threshold
            kept += int(mask.sum())
            out[key] = tensor * mask.to(tensor.dtype) * factor

    return out, {
        "kept_entries": float(kept),
        "total_entries": float(total),
        "achieved_density": (kept / total) if total else 0.0,
    }


def _rank_trim_one(delta: TensorDict, spec: ConditioningSpec) -> tuple[TensorDict, dict[str, float]]:
    density = float(spec.density)
    factor = (1.0 / density) if spec.rescale else 1.0
    out: TensorDict = {}
    retained_energy = 0.0
    total_energy = 0.0
    truncated = 0
    for key, tensor in delta.items():
        original = tensor.detach()
        total_energy += float(original.to(torch.float64).pow(2).sum())
        if original.ndim < int(spec.min_ndim):
            out[key] = tensor
            retained_energy += float(original.to(torch.float64).pow(2).sum())
            continue
        matrix = original.to(torch.float64).reshape(original.shape[0], -1)
        rank = max(1, int(math.ceil(density * min(matrix.shape))))
        if rank >= min(matrix.shape):
            out[key] = tensor
            retained_energy += float(original.to(torch.float64).pow(2).sum())
            continue
        u, s, v_h = torch.linalg.svd(matrix, full_matrices=False)
        low = (u[:, :rank] * s[:rank].unsqueeze(0)) @ v_h[:rank, :]
        retained_energy += float(low.pow(2).sum())
        out[key] = (low.reshape(original.shape) * factor).to(dtype=tensor.dtype)
        truncated += 1
    return out, {
        "truncated_tensors": float(truncated),
        "retained_energy_fraction": (retained_energy / total_energy) if total_energy else 0.0,
    }


def condition_transported_deltas(
    deltas: Mapping[str, TensorDict], spec: ConditioningSpec
) -> tuple[dict[str, TensorDict], dict[str, Any]]:
    """Apply ``spec`` to a task -> delta mapping and report what it changed.

    The returned diagnostics always carry the pre- and post-conditioning global
    norms per task, because the whole point of the operation is to change the
    relative energy each task contributes to the merge.
    """

    pre_norms = {task: _global_norm(delta) for task, delta in deltas.items()}
    if spec.mode == "off":
        return {task: dict(delta) for task, delta in deltas.items()}, {
            "spec": spec.as_dict(),
            "pre_global_norm": pre_norms,
            "post_global_norm": dict(pre_norms),
        }

    if spec.mode == "norm_match":
        out, details = _norm_match(deltas, spec)
    elif spec.mode == "mean_removal":
        out, details = _mean_removal(deltas, spec)
    elif spec.mode == "magnitude_trim":
        out, per_task = {}, {}
        for task, delta in deltas.items():
            out[task], per_task[task] = _magnitude_trim_one(delta, spec)
        details = {"per_task": per_task}
    else:
        out, per_task = {}, {}
        for task, delta in deltas.items():
            out[task], per_task[task] = _rank_trim_one(delta, spec)
        details = {"per_task": per_task}

    diagnostics: dict[str, Any] = {
        "spec": spec.as_dict(),
        "pre_global_norm": pre_norms,
        "post_global_norm": {task: _global_norm(delta) for task, delta in out.items()},
    }
    diagnostics.update(details)
    return out, diagnostics
