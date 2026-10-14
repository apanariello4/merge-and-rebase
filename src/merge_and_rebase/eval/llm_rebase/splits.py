"""Resolve evaluation document selections before fitting or scoring any model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Samples = dict[str, list[int]]


@dataclass(frozen=True)
class HarnessSlices:
    search: Samples | None
    test: Samples | None


def _parse_samples(raw: Any, key: str) -> Samples | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"config['{key}'] must map task names to document-index lists.")
    result = {}
    for task, indices in raw.items():
        if not isinstance(indices, list) or not all(type(index) is int and index >= 0 for index in indices):
            raise ValueError(f"config['{key}'] values must be lists of non-negative document indices.")
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"config['{key}'] needs non-empty, unique document indices for task {task!r}.")
        result[str(task)] = list(indices)
    if not result:
        raise ValueError(f"config['{key}'] must select at least one task.")
    return result


def resolve_harness_slices(
    cfg: Mapping[str, Any],
    tasks: Sequence[str],
    *,
    calibration_holdout: Samples | None = None,
) -> HarnessSlices:
    """Explicit search/test slices must be disjoint and outside harness calibration.

    ``None`` search means the entire evaluation corpus, or the entire calibration
    holdout when one exists. It cannot establish an independent test selection.
    """
    search = _parse_samples(cfg.get("harness_samples"), "harness_samples")
    test = _parse_samples(cfg.get("harness_test_samples"), "harness_test_samples")
    if test is not None:
        if search is None:
            raise ValueError("harness_test_samples requires explicit harness_samples for alpha search.")
        for key, samples in (("harness_samples", search), ("harness_test_samples", test)):
            if set(samples) != set(tasks):
                raise ValueError(f"{key} must select documents for every harness task: {list(tasks)}.")
        overlap = {task: sorted(set(search[task]) & set(test[task])) for task in tasks}
        leaking = {task: indices for task, indices in overlap.items() if indices}
        if leaking:
            raise ValueError(
                "harness_test_samples overlaps the alpha-search slice for "
                f"{ {task: len(indices) for task, indices in leaking.items()} }; the reported number would be "
                "selected on documents it is scored on."
            )
    if calibration_holdout is not None:
        for key, samples in (("harness_samples", search), ("harness_test_samples", test)):
            if samples is not None and any(
                task not in calibration_holdout or not set(indices).issubset(calibration_holdout[task])
                for task, indices in samples.items()
            ):
                raise ValueError(
                    f"config['{key}'] disagrees with the IFEval hold-out derived from calibration; "
                    "use the derived samples or an independent calibration corpus."
                )
        if search is None:
            search = {task: list(indices) for task, indices in calibration_holdout.items()}
    return HarnessSlices(search=search, test=test)
