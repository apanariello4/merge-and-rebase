"""Summary assembly for the vision rebase entrypoint.

``RunRecord`` carries the run state ``main()`` accumulates; ``assemble_summary`` builds the final summary
dictionary from it (key order is part of the output contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...rebase.block_extension.config import block_extension_protocol
from ...rebase.methods._ariadne.config import direct_residual_config_dict
from ...rebase.registry import canonical_method_name


@dataclass(kw_only=True)
class RunRecord:
    suite_name: Any
    tasks: Any
    method_label: Any
    merge_mode: Any
    target_hash_before: Any
    target_hash_after: Any
    single_transport_calibration_metadata: Any
    brace_calibration_metadata: Any
    base_construction: Any
    hierarchical_premerge_alpha_curve: Any
    global_alpha_curve: Any
    alpha_selection: Any
    selected_validation_results: Any
    baseline_label: Any
    block_extension_eval_rows: Any
    source_lmc_rows: Any
    cross_task_lmc_rows: Any = field(default_factory=list)
    all_task_lmc_rows: Any = field(default_factory=list)
    transported_artifacts: Any
    transport_timings: Any
    transport_calibration_meta: Any
    cost_phase_timings: Any
    alignment_calibration_timings: Any
    correction_fit_timings: Any
    depth_alignment_mode: Any
    depth_rule_resolved: Any
    saved_merged_path: Any
    method: Any
    merge_method_name: Any
    merge_params: Any
    block_extension_cfg: Any
    native_tasks: Any
    hierarchical: Any
    global_alpha_search: Any
    best_alpha: Any
    best_baseline_alpha: Any
    direct_fit: Any
    # The independent-endpoint diagnostics are never populated (dead since the endpoint-average baseline moved
    # into the BRACE stage); the summary keeps the keys.
    independent_base_dispersion: Any = None
    independent_base_distance_by_task: Any = field(default_factory=dict)
    independent_source_merge_param_count: Any = None
    independent_base_diagnostics_path: Any = None
    independent_direct_delta_key_count: Any = field(default_factory=dict)
    single_tv_test_accs: Any
    single_tv_alpha_protocol: Any
    direct_residual_calibration_meta: Any
    direct_residual_diagnostics: Any
    direct_residual_realization: Any
    direct_residual_task_vector_stats: Any
    direct_residual_alignment_diagnostics: Any
    direct_residual_calibration_by_task: Any
    direct_residual_tv_scaling: Any
    direct_residual_pairing_record: Any
    direct_residual_fidelity_holdout: Any
    direct_residual_sequential_endpoints: Any
    loaded_direct_residual_tvs: Any
    per_task_premerge_alphas: Any
    selected_alpha_by_task: Any
    per_task: Any
    selected_baseline_alpha_by_task: Any
    direct_residual_cfg: Any
    method_name: Any
    independent_base_average: Any = None
    baseline_test_accs: Any
    rebase_test_accs: Any
    norm_accs: Any
    direct_residual_preset: Any
    single_tv_val_best_alpha: Any
    single_tv_val_best_acc: Any
    #: Additive ``save_policy`` entry; ``None`` (key omitted) unless the config named ``save_transported_tvs``.
    save_policy: Any = None

    @classmethod
    def from_run(
        cls,
        resolved: Any,
        *,
        native_tasks: Any,
        per_task: Any,
        loop_outputs: Any,
        calibration: Any,
        merge_plan: Any,
        alpha: Any,
        ariadne: Any,
        block_extension_eval_rows: Any,
        source_lmc_rows: Any,
        transported_artifacts: Any,
        target_hash_before: Any,
        target_hash_after: Any,
        saved_merged_path: Any,
        save_policy: Any = None,
    ) -> RunRecord:
        """Collect the run's structured outputs (the summary keys keep their legacy names and order)."""
        return cls(
            suite_name=resolved.suite_name,
            tasks=resolved.tasks,
            method_label=resolved.method_label,
            merge_mode=resolved.merge.mode,
            target_hash_before=target_hash_before,
            target_hash_after=target_hash_after,
            single_transport_calibration_metadata=merge_plan.calibration_metadata,
            brace_calibration_metadata=calibration.brace_calibration_metadata,
            base_construction=resolved.merge.base_construction,
            hierarchical_premerge_alpha_curve=alpha.hierarchical_premerge_alpha_curve,
            global_alpha_curve=alpha.global_alpha_curve,
            alpha_selection=resolved.alpha.selection,
            selected_validation_results=alpha.selected_validation_results,
            baseline_label=alpha.baseline_label,
            block_extension_eval_rows=block_extension_eval_rows,
            source_lmc_rows=source_lmc_rows,
            transported_artifacts=transported_artifacts,
            transport_timings=loop_outputs.transport_timings,
            transport_calibration_meta=calibration.transport_calibration_meta,
            cost_phase_timings=loop_outputs.cost_phase_timings,
            alignment_calibration_timings=loop_outputs.alignment_calibration_timings,
            correction_fit_timings=loop_outputs.correction_fit_timings,
            depth_alignment_mode=resolved.depth_alignment_mode,
            depth_rule_resolved=resolved.depth_rule_resolved,
            saved_merged_path=saved_merged_path,
            method=resolved.method,
            merge_method_name=resolved.merge.method_name,
            merge_params=resolved.merge.params,
            block_extension_cfg=resolved.block_extension_cfg,
            native_tasks=native_tasks,
            hierarchical=alpha.hierarchical,
            global_alpha_search=resolved.merge.global_alpha_search,
            best_alpha=alpha.best_alpha,
            best_baseline_alpha=alpha.best_baseline_alpha,
            direct_fit=resolved.direct_fit,
            single_tv_test_accs=alpha.single_tv_test_accs,
            single_tv_alpha_protocol=alpha.single_tv_alpha_protocol,
            direct_residual_calibration_meta=ariadne.calibration_meta,
            direct_residual_diagnostics=ariadne.diagnostics,
            direct_residual_realization=ariadne.realization,
            direct_residual_task_vector_stats=ariadne.task_vector_stats,
            direct_residual_alignment_diagnostics=ariadne.alignment_diagnostics,
            direct_residual_calibration_by_task=ariadne.calibration_by_task,
            direct_residual_tv_scaling=ariadne.tv_scaling,
            direct_residual_pairing_record=ariadne.pairing,
            direct_residual_fidelity_holdout=ariadne.fidelity_holdout,
            direct_residual_sequential_endpoints=ariadne.sequential_endpoints,
            loaded_direct_residual_tvs=ariadne.loaded_vectors,
            per_task_premerge_alphas=alpha.per_task_premerge_alphas,
            selected_alpha_by_task=alpha.selected_alpha_by_task,
            per_task=per_task,
            selected_baseline_alpha_by_task=alpha.selected_baseline_alpha_by_task,
            direct_residual_cfg=resolved.ariadne_cfg,
            method_name=resolved.method_name,
            baseline_test_accs=alpha.baseline_test_accs,
            rebase_test_accs=alpha.rebase_test_accs,
            norm_accs=alpha.norm_accs,
            direct_residual_preset=resolved.ariadne_preset,
            single_tv_val_best_alpha=alpha.single_tv_val_best_alpha,
            single_tv_val_best_acc=alpha.single_tv_val_best_acc,
            save_policy=save_policy,
        )


def _protocol_record(record: RunRecord) -> dict[str, Any]:
    """``block_extension_protocol`` with the structural rule that actually ran (B7)."""
    protocol = block_extension_protocol(record.block_extension_cfg)
    if (record.depth_rule_resolved or {}).get("rule") == "discrete_index_match":
        # No BRACE step ran: the label used to say "ariadne" for BiCo/THESEUS discrete-index-match runs.
        protocol = {**protocol, "label": "discrete_index_match"}
    return protocol


def assemble_summary(record: RunRecord) -> dict[str, Any]:
    summary = {
        "suite": record.suite_name,
        "tasks": record.tasks,
        "method": record.method.name,
        "method_label": record.method_label,
        "merge_mode": record.merge_mode,
        "merge_method": record.merge_method_name if record.merge_mode != "none" else None,
        "merge_params": record.merge_params if record.merge_mode != "none" else None,
        "target_hash_before": record.target_hash_before,
        "target_hash_after": record.target_hash_after,
        "strict_diagnostics": {"missing": 0, "failures": 0, "wrong_shape": 0},
        "single_transport_calibration": record.single_transport_calibration_metadata,
        "brace_calibration": record.brace_calibration_metadata,
        "block_extension_protocol": _protocol_record(record),
        "base_construction": record.base_construction,
        "independent_endpoint_baseline": (
            {
                "task_vector_definition": "tau_ind_t = ft_ind_t - base_ind_t",
                "base_average_definition": "base_ind_avg = mean_t(base_ind_t)",
                "base_average_key_scope": "common floating-point visual tensors",
                "n_common_visual_keys": len(record.independent_base_average)
                if record.independent_base_average is not None
                else None,
                "base_dispersion_ind": record.independent_base_dispersion,
                "per_task_distance_to_mean_base": record.independent_base_distance_by_task,
                "source_merge_direction_param_count": record.independent_source_merge_param_count,
                "diagnostics_path": record.independent_base_diagnostics_path,
                "direct_delta_key_count": record.independent_direct_delta_key_count,
                "direct_endpoint_difference_used": record.base_construction == "independent_endpoint_average",
            }
            if record.base_construction == "independent_endpoint_average"
            else None
        ),
        "native_target_tasks": sorted(record.native_tasks) if record.native_tasks else [],
        "global_alpha_search": record.global_alpha_search if record.hierarchical else None,
        "per_task_premerge_alphas": (
            {item["task"]: float(record.per_task_premerge_alphas[i]) for i, item in enumerate(record.per_task)}
            if record.hierarchical and record.per_task_premerge_alphas is not None
            else None
        ),
        "hierarchical_premerge_alpha_curve": record.hierarchical_premerge_alpha_curve,
        "global_alpha_curve": record.global_alpha_curve,
        "alpha_selection": record.alpha_selection,
        "validation_results": record.selected_validation_results,
        "best_alpha": float(record.best_alpha),
        "best_baseline_alpha": float(record.best_baseline_alpha),
        "baseline_label": record.baseline_label,
        "metric_definitions": {
            "absolute_accuracy": "top-1 accuracy in [0, 1] (rebased/transported at the rebased's own best alpha)",
            "baseline_accuracy": "untransported baseline top-1 at the baseline's own best alpha",
            "normalized_accuracy_ratio": (
                "absolute_accuracy (at rebased best alpha) / baseline_accuracy (at baseline best alpha); "
                "each stream independently optimizes alpha on the alpha-search split"
            ),
            "normalized_accuracy_ratio_display": (
                "ratio (decimal, not a percentage); values above 1.0 indicate the rebased/transported "
                "model exceeds the untransported baseline; report as a decimal ratio, never multiplied by 100"
            ),
        },
        "test_results": {
            # Explicit names prevent a table exporter from treating a ratio as raw accuracy.
            "per_task_baseline_accuracy": {
                item["task"]: float(record.baseline_test_accs[i]) for i, item in enumerate(record.per_task)
            },
            "per_task_absolute_accuracy": {
                item["task"]: float(record.rebase_test_accs[i]) for i, item in enumerate(record.per_task)
            },
            "per_task_normalized_accuracy_ratio": {
                item["task"]: float(record.norm_accs[i]) for i, item in enumerate(record.per_task)
            },
            "per_task_baseline": {
                item["task"]: float(record.baseline_test_accs[i]) for i, item in enumerate(record.per_task)
            },
            "per_task_rebased": {
                item["task"]: float(record.rebase_test_accs[i]) for i, item in enumerate(record.per_task)
            },
            "per_task_norm": {item["task"]: float(record.norm_accs[i]) for i, item in enumerate(record.per_task)},
            "avg_rebased": float(sum(record.rebase_test_accs) / len(record.rebase_test_accs)),
            "avg_norm": float(sum(record.norm_accs) / len(record.norm_accs)),
        },
        "single_tv_diagnostic": (
            {
                "definition": (
                    "For each task t, evaluate target_base + alpha_t * transported/native task_vector_t "
                    "on task t's test set; alpha_t is selected on validation only."
                ),
                "alpha_protocol": record.single_tv_alpha_protocol,
                "per_task_validation_alpha": {
                    item["task"]: float(record.single_tv_val_best_alpha[i]) for i, item in enumerate(record.per_task)
                },
                "per_task_validation_accuracy": {
                    item["task"]: float(record.single_tv_val_best_acc[i]) for i, item in enumerate(record.per_task)
                },
                "per_task_test_accuracy": {
                    item["task"]: float(record.single_tv_test_accs[i]) for i, item in enumerate(record.per_task)
                },
                "avg_test_accuracy": float(sum(record.single_tv_test_accs) / len(record.single_tv_test_accs)),
                "merged_avg_test_accuracy": float(sum(record.rebase_test_accs) / len(record.rebase_test_accs)),
                "merge_gap_single_minus_merged": float(
                    sum(record.single_tv_test_accs) / len(record.single_tv_test_accs)
                    - sum(record.rebase_test_accs) / len(record.rebase_test_accs)
                ),
            }
            if record.single_tv_test_accs is not None
            else None
        ),
        "selected_alpha_by_task": {
            item["task"]: float(record.selected_alpha_by_task[i]) for i, item in enumerate(record.per_task)
        },
        "selected_baseline_alpha_by_task": {
            item["task"]: float(record.selected_baseline_alpha_by_task[i]) for i, item in enumerate(record.per_task)
        },
        "block_extension_target_dataset_eval": record.block_extension_eval_rows,
        "source_lmc": record.source_lmc_rows,
        "cross_task_source_lmc": record.cross_task_lmc_rows,
        "all_task_source_lmc": record.all_task_lmc_rows,
        "transported_artifacts": record.transported_artifacts,
        "transport_timings": record.transport_timings,
        "transport_calibration": record.transport_calibration_meta,
        "cost_phase_timings": record.cost_phase_timings,
        # Always present (default {}) regardless of method/path, so a
        # downstream summary-JSON parser can read these keys uniformly
        # across every method, not only depth_alignment='discrete_index_match'
        # or method='direct_residual' runs.
        "alignment_calibration_timings": record.alignment_calibration_timings,
        "correction_fit_timings": record.correction_fit_timings,
        "direct_residual": (
            {
                "config": direct_residual_config_dict(record.direct_residual_cfg),
                # Canonical registry name ("direct_residual" is an alias of "ariadne");
                # the top-level "method" keeps whatever spelling the config used.
                "canonical_method": canonical_method_name(record.method_name),
                # Present only when the params mapping selected a named preset
                # (the preset's fields are already resolved into "config").
                **({"preset": record.direct_residual_preset} if record.direct_residual_preset is not None else {}),
                # Which images every fit calibrated on (see
                # DirectResidualConfig.calibration_data): the dataset and,
                # for vision8_mix, the balanced plan's fingerprint.
                "calibration": record.direct_residual_calibration_meta,
                "diagnostics_by_task": record.direct_residual_diagnostics,
                # Additive, analysis-only: both are None per task unless
                # direct_residual_cfg.realization_diagnostics is set (see
                # _run_direct_residual_fit / measure_direct_residual_realization
                # / compute_direct_residual_task_vector_stats).
                "realization_by_task": record.direct_residual_realization,
                "task_vector_stats_by_task": record.direct_residual_task_vector_stats,
                # Analysis-only Procrustes-alignment diagnostics (never
                # fed to any fit), always populated regardless of
                # config.residual_target or realization_diagnostics; see
                # compute_alignment_diagnostics.
                "alignment_diagnostics_by_task": record.direct_residual_alignment_diagnostics,
                "calibration_by_task": record.direct_residual_calibration_by_task,
                # Additive, analysis-only: None per task unless
                # direct_residual_cfg.tv_scaling != "none" (see
                # apply_tv_scaling / _run_direct_residual_fit). tv_scaling
                # mode + iters are already carried by "config" above
                # (asdict(direct_residual_cfg)); this key carries the
                # per-task measurement (r_j traces, s_j / c, tau stats
                # before/after).
                "tv_scaling_by_task": record.direct_residual_tv_scaling,
                # depth_pairing ablation: the pi(j) tuple actually used
                # (post apply_depth_pairing_override), for the ablation
                # to compare against DirectResidualConfig.depth_pairing
                # in "config" above without recomputing it.
                "pairing": record.direct_residual_pairing_record,
                # fidelity_holdout diagnostic (analysis-only, never fed to
                # any fit): None per task unless
                # direct_residual_cfg.fidelity_holdout is set.
                "fidelity_holdout_by_task": record.direct_residual_fidelity_holdout,
                "sequential_endpoints_by_task": record.direct_residual_sequential_endpoints,
                "loaded_vectors_by_task": record.loaded_direct_residual_tvs,
            }
            if record.direct_fit
            else None
        ),
        # Reports whichever depth-alignment mode was active for a Theseus-/
        # BiCo-like method; always present so a downstream parser can rely
        # on the key, even though the default "ariadne" path never touches
        # anything new added by this change.
        "depth_alignment": record.depth_alignment_mode,
        "depth_rule_resolved": record.depth_rule_resolved,
        "saved_merged_path": record.saved_merged_path,
    }
    if record.save_policy is not None:
        summary["save_policy"] = record.save_policy
    return summary
