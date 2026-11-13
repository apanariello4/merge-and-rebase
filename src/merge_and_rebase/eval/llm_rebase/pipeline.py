"""Per-task rebase pipeline of the LLM entrypoint (mirrors eval/vision_rebase/pipeline.py).

``run_rebase`` is the former middle of ``cli.main``: before-rebase reference evals, then the shared per-task loop
(``rebase.orchestration.TaskPipeline``: prepare -> transport -> free, one task at a time) and the weighted merge of the
transported deltas. The LLM stages (``stages``, ``method_stages``) implement the same ``DepthPrestep`` /
``MethodStage`` protocols as the vision ones, driven by the ``ResolvedRunConfig`` / ``RunPlan`` on ``LlmRuntime``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from ...io.ckpt import load_into_model
from ...merge.runtime import compose_weighted_deltas
from ...rebase.orchestration import TaskPipeline
from ...rebase.prestep import StageEnv
from .merge import _summarize_merged_delta, report_merged_delta
from .method_stages import build_method_stage
from .runtime import LlmRuntime
from .stages import build_prestep, build_prestep_observers, build_task_models


@dataclass
class RebaseOutputs:
    merged_delta: dict[str, torch.Tensor]
    delta_stats: Any
    task_vector_report: dict[str, Any]


def _resolve_task_weights(raw: Any, count: int) -> list[float]:
    """Broadcast a scalar, but require an explicit list to name every task."""
    if raw is None:
        return [1.0] * count
    if isinstance(raw, (list, tuple)):
        if len(raw) != count:
            raise ValueError(
                f"weights length must match tuned checkpoints: got {len(raw)} weights for {count} checkpoints. "
                "Use a scalar to apply the same weight to every checkpoint."
            )
        return [float(value) for value in raw]
    return [float(raw)] * count


def run_rebase(rt: LlmRuntime, run_logger: Any) -> RebaseOutputs | None:
    """Returns None when the run stops after the before-rebase eval (eval_before_rebase_only)."""
    cfg = rt.cfg
    method_name = rt.resolved.method_name
    source_llm = rt.models.source_llm
    source_base_sd = rt.models.source_base_sd
    target_base_sd = rt.models.target_base_sd
    device = rt.models.device
    run_block_extension_prestep = rt.plan.run_block_extension_prestep
    source_depth = rt.plan.source_depth
    target_depth = rt.plan.target_depth
    tuned_ref_list = rt.evaluation.tuned_ref_list
    tasks = rt.evaluation.tasks
    run_before_rebase_eval = rt.evaluation.run_before_rebase_eval
    eval_before_rebase_only = rt.evaluation.eval_before_rebase_only
    ignored_block_extension_fields = rt.ignored_block_extension_fields
    _calibration_provenance = rt.calibration.cache.provenance
    _eval_before_rebase = partial(rt.evaluation.evaluate_before_rebase, rt.models)
    _baseline_summary = rt.evaluation.baseline_summary
    weights = _resolve_task_weights(cfg.get("weights"), len(tuned_ref_list))

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
    method_stage = None if eval_before_rebase_only else build_method_stage(cfg, direct_fit=rt.resolved.direct_fit)
    prestep = build_prestep(rt.plan)
    pipeline = TaskPipeline(
        prestep=prestep,
        observers=build_prestep_observers(rt.plan),
        method_stage=method_stage,
        saver=lambda task, delta: None,
        build_models=build_task_models,
    )
    # Construction cost of the task vectors (model building, depth prestep, transport / fit), excluding every
    # evaluation pass: the same window for every method, the quantity a fit-time comparison reports.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rebase_t0 = time.perf_counter()
    loop = pipeline.run(env, list(rt.task_contexts), rt.task_contexts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rebase_cost = {
        "rebase_wall_seconds": time.perf_counter() - rebase_t0,
        "rebase_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }

    if eval_before_rebase_only:
        # Everything the before-rebase reference needs is done: block
        # extension has run and the extended source base has been scored.
        # Transport is the expensive half and contributes nothing here.
        print("\nStopping after the before-rebase eval (eval_before_rebase_only).")
        if run_logger is not None:
            overrides = {t: c.config_overrides for t, c in rt.task_contexts.items() if c.config_overrides}
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
                # additive: present only when a tuned body carries its own computation config
                **({"tuned_config_overrides": overrides} if overrides else {}),
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
        **rebase_cost,
        # What resolved and what ran: the configured depth rule and the prestep that actually executed (with the
        # BRACE settings it used), so a summary alone proves which depth handling produced the task vector.
        "depth_handling": {
            "rule": rt.resolved.depth_rule.kind,
            "rule_source": rt.resolved.depth_rule.source,
            "executed_prestep": prestep.kind.value,
            **(
                {
                    "extension_strategy": rt.resolved.block_extension_cfg.extension_strategy,
                    "skip_correction": rt.resolved.block_extension_cfg.skip_correction,
                }
                if prestep.kind.value == "brace"
                else {}
            ),
        },
    }
    forward_geometry = {t: c.forward_geometry for t, c in rt.task_contexts.items() if c.forward_geometry}
    if forward_geometry:
        task_vector_report["forward_geometry"] = forward_geometry
    config_overrides = {t: c.config_overrides for t, c in rt.task_contexts.items() if c.config_overrides}
    if config_overrides:
        # Tuned bodies whose own HF config differs from the source base's: {task: {field: [source, tuned]}}.
        task_vector_report["tuned_config_overrides"] = config_overrides
        task_vector_report["allow_tuned_config_mismatch"] = bool(cfg.get("allow_tuned_config_mismatch", False))
    if getattr(method_stage, "materialized_bias_keys", None) is not None:
        task_vector_report["materialized_bias_keys"] = list(method_stage.materialized_bias_keys)
    if hasattr(method_stage, "report"):
        task_vector_report["ariadne"] = method_stage.report()
    report_merged_delta(delta_stats)
    return RebaseOutputs(merged_delta=merged_delta, delta_stats=delta_stats, task_vector_report=task_vector_report)
