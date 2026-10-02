"""Alpha-search scoring helpers for the vision rebase entrypoint."""

from __future__ import annotations

from ...utils.alpha_search import average_scores
from ..rebase_metrics import normalized_accuracy_ratio


def _norm_acc(result_acc: float, baseline_acc: float) -> float:
    return normalized_accuracy_ratio(result_acc, baseline_acc)


def _average_defined(values: list[float]) -> float:
    defined = [float(v) for v in values if float(v) == float(v)]
    return average_scores(defined) if defined else float("nan")
