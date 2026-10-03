"""Per-task rebase pipeline of the LLM entrypoint (mirrors eval/vision_rebase/pipeline.py).

``run_rebase`` is the former middle of ``cli.main``: before-rebase reference evals, then the shared per-task loop
(``rebase.orchestration.TaskPipeline``: prepare -> transport -> free, one task at a time) and the weighted merge of the
transported deltas. The LLM stages (``stages``, ``method_stages``) implement the same ``DepthPrestep`` /
``MethodStage`` protocols as the vision ones, driven by the ``ResolvedRunConfig`` / ``RunPlan`` on ``LlmRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...io.ckpt import load_into_model
from ...merge.runtime import compose_weighted_deltas
from ...rebase.orchestration import TaskPipeline
from ...rebase.prestep import StageEnv
from ...rebase.run_config import MethodKind
from .merge import _summarize_merged_delta, report_merged_delta
from .method_stages import build_method_stage
from .stages import build_prestep, build_prestep_observers, build_task_models


@dataclass
class LlmRuntime:
    """Everything ``cli.main`` resolved before the first task: config, models, state dicts, calibration, flags."""

    cfg: Any
    method_name: Any
    method: Any
    method_params: Any
    source_llm: Any
    target_llm: Any
    source_build_cfg: Any
    source_base_sd: Any
    target_base_sd: Any
    source_family: Any
    target_family: Any
    family_adapter: Any
    device: Any
    num_labels: Any
    block_extension_cfg: Any
    run_block_extension_prestep: Any
    source_depth: Any
    target_depth: Any
    tuned_ref_list: Any
    tasks: Any
    suite_name: Any
    is_harness_only: Any
    harness_tasks_resolved: Any
    harness_num_fewshot: Any
    harness_batch_size: Any
    harness_limit: Any
    harness_samples: Any
    run_before_rebase_eval: Any
    eval_before_rebase_only: Any
    ignored_block_extension_fields: Any
    _calibration: Any
    _calibration_provenance: Any
    calib_batch_size: Any
    calib_max_length: Any
    calib_n_batches: Any
    blockext_calib_loader: Any
    tp_keys: Any
    full_fp_keys: Any
    eval_mode: Any
    head_key_pattern: Any
    task_heads_path: Any
    _eval_before_rebase: Any
    _baseline_summary: Any
    load_tuned: Any
    #: Shared run contract (``rebase.run_config``): resolved LLM config and its post-model plan.
    resolved: Any
    plan: Any
    #: Task label -> ``LlmTaskContext`` (tuned checkpoint ref + position), in task order.
    task_contexts: Any


@dataclass
class RebaseOutputs:
    merged_delta: dict[str, torch.Tensor]
    delta_stats: Any
    task_vector_report: dict[str, Any]


def run_rebase(rt: LlmRuntime, run_logger: Any) -> RebaseOutputs | None:
    """Returns None when the run stops after the before-rebase eval (eval_before_rebase_only)."""
    cfg = rt.cfg
    method_name = rt.method_name
    source_llm = rt.source_llm
    source_base_sd = rt.source_base_sd
    target_base_sd = rt.target_base_sd
    device = rt.device
    run_block_extension_prestep = rt.run_block_extension_prestep
    source_depth = rt.source_depth
    target_depth = rt.target_depth
    tuned_ref_list = rt.tuned_ref_list
    tasks = rt.tasks
    run_before_rebase_eval = rt.run_before_rebase_eval
    eval_before_rebase_only = rt.eval_before_rebase_only
    ignored_block_extension_fields = rt.ignored_block_extension_fields
    _calibration_provenance = rt._calibration_provenance
    _eval_before_rebase = rt._eval_before_rebase
    _baseline_summary = rt._baseline_summary

    if run_before_rebase_eval and not run_block_extension_prestep:
        # No depth change: the model transport starts from is the plain
        # source base, so one pass is enough for every task.
        load_into_model(source_llm.model, source_base_sd, strict=False)
        _eval_before_rebase(source_llm.model, "source_base")
        source_llm.model.to("cpu")
    elif run_before_rebase_eval and bool(cfg.get("eval_source_before_extension", False)):
        # The unextended source, scored on the same eval slice. Without it
        # the only "before" number is the extended source base, so there is
        # nothing to say whether correction restores the original model or
        # improves on it -- the two are indistinguishable from the extended
        # score alone.
        load_into_model(source_llm.model, source_base_sd, strict=False)
        _eval_before_rebase(source_llm.model, "source_base_unextended")
        source_llm.model.to("cpu")
    weights_raw = cfg.get("weights", None)
    if weights_raw is None:
        weights = [1.0] * len(tuned_ref_list)
    else:
        w = weights_raw if isinstance(weights_raw, (list, tuple)) else [float(weights_raw)]
        if len(w) < len(tuned_ref_list):
            w = w * len(tuned_ref_list)
        weights = [float(x) for x in w[: len(tuned_ref_list)]]

    # Per task: prepare the depth-matched delta, transport it, free the task-local models.
    if not eval_before_rebase_only:
        print(f"\n=== Transporting {len(tasks) if tasks else 1} task vectors with {method_name} ===")
    env = StageEnv(
        resolved=rt.resolved,
        plan=rt.plan,
        cfg=cfg,
        device=device,
        source_base_sd=source_base_sd,
        target_base_sd=target_base_sd,
        run_logger=None,  # the LLM entrypoint has never emitted per-task transport events
        runtime=rt,
    )
    # eval_before_rebase_only stops each task after its prestep, so its transport options are never resolved.
    method_stage = None if eval_before_rebase_only else build_method_stage(cfg, ariadne=rt.resolved.method_kind is MethodKind.ARIADNE)
    pipeline = TaskPipeline(
        prestep=build_prestep(rt.plan),
        observers=build_prestep_observers(rt.plan),
        method_stage=method_stage,
        saver=lambda task, delta: None,
        build_models=build_task_models,
    )
    loop = pipeline.run(env, list(rt.task_contexts), rt.task_contexts)

    if eval_before_rebase_only:
        # Everything the before-rebase reference needs is done: block
        # extension has run and the extended source base has been scored.
        # Transport is the expensive half and contributes nothing here.
        print("\nStopping after the before-rebase eval (eval_before_rebase_only).")
        if run_logger is not None:
            run_logger.log_summary({
                "ignored_block_extension_fields": ignored_block_extension_fields,
                "calibration_provenance": _calibration_provenance(),
                "method": method_name,
                "backend": "lm_harness",
                "stopped_after": "before_rebase_eval",
                "harness_results_before_rebase": _baseline_summary(),
                "before_rebase_model": (
                    "extended_source_base" if run_block_extension_prestep else "source_base"
                ),
                "source_depth": source_depth,
                "target_depth": target_depth,
            })
            run_logger.finish("success")
        return None

    # Merge transported deltas
    merged_delta = compose_weighted_deltas(loop.transported_deltas, weights)
    delta_stats = _summarize_merged_delta(merged_delta, target_base_sd)
    task_vector_report = {
        "transport_delta_source": method_stage.delta_source,
        "delta_norm_match": method_stage.norm_match or "none",
        "per_task": loop.task_vector_norms,
    }
    if getattr(method_stage, "materialized_bias_keys", None) is not None:
        task_vector_report["materialized_bias_keys"] = list(method_stage.materialized_bias_keys)
    report_merged_delta(delta_stats)
    return RebaseOutputs(merged_delta=merged_delta, delta_stats=delta_stats, task_vector_report=task_vector_report)
