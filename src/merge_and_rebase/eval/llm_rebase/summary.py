"""Run-summary assembly for the LLM rebase entrypoint (key order is part of the summary contract)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...hyperparam_search import summarize_search_results


def assemble_harness_summary(
    *,
    ignored_block_extension_fields: Any,
    calibration_provenance: Any,
    method_name: str,
    best_alpha: float,
    best_harness_results: Any,
    harness_test_results: Any,
    test_samples: Any,
    baseline_harness_results: Any,
    run_block_extension_prestep: bool,
    harness_results_by_alpha: dict[float, Any],
    delta_stats: Any,
    task_vector_report: Any,
    search_planner: Any,
    harness_search_results: Any,
    saved_merged_path: Any,
) -> dict[str, Any]:
    return {
        "ignored_block_extension_fields": ignored_block_extension_fields,
        "calibration_provenance": calibration_provenance,
        "method": method_name,
        "best_alpha": best_alpha,
        "backend": "lm_harness",
        "harness_results": best_harness_results,
        # Selected on the search slice; quote harness_results_test
        # instead whenever it is present.
        "harness_results_test": harness_test_results,
        "harness_test_sample_counts": ({t: len(v) for t, v in test_samples.items()} if harness_test_results else None),
        "harness_results_before_rebase": baseline_harness_results,
        "before_rebase_model": ("extended_source_base" if run_block_extension_prestep else "source_base"),
        # Named per-alpha metrics: search_results only keeps a flat
        # per_task_acc list, which loses which task each number is.
        "harness_results_by_alpha": {f"{a:g}": r for a, r in sorted(harness_results_by_alpha.items())},
        "merged_delta": delta_stats,
        "task_vectors": task_vector_report,
        "search_strategy": search_planner.search_summary(),
        "search_results": summarize_search_results(harness_search_results),
        "saved_merged_path": saved_merged_path,
    }


def assemble_nli_summary(
    *,
    ignored_block_extension_fields: Any,
    calibration_provenance: Any,
    method_name: str,
    best_alpha: float,
    task_data: list[Any],
    delta_stats: Any,
    task_vector_report: Any,
    search_planner: Any,
    search_results: Any,
    best_vals: list[float],
    saved_merged_path: Any,
) -> dict[str, Any]:
    return {
        "ignored_block_extension_fields": ignored_block_extension_fields,
        "calibration_provenance": calibration_provenance,
        "method": method_name,
        "best_alpha": best_alpha,
        "tasks": [td.task for td in task_data],
        "merged_delta": delta_stats,
        "task_vectors": task_vector_report,
        "search_strategy": search_planner.search_summary(),
        "search_results": summarize_search_results(search_results),
        "best_per_task_acc": {td.task: float(best_vals[i]) for i, td in enumerate(task_data)},
        "saved_merged_path": saved_merged_path,
    }


@dataclass
class RunRecord:
    """What the evaluation backend produced: ``backend`` ("harness" | "nli") plus the assembler's keyword arguments."""

    backend: str
    fields: dict[str, Any]


def assemble_summary(record: RunRecord) -> dict[str, Any]:
    if record.backend == "harness":
        return assemble_harness_summary(**record.fields)
    return assemble_nli_summary(**record.fields)
