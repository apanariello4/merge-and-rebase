"""Per-task rebase pipeline of the LLM entrypoint (mirrors eval/vision_rebase/pipeline.py).

``run_rebase`` is the former middle of ``cli.main``: before-rebase reference evals, the per-task prestep, the
per-task transport and the weighted merge of the transported deltas. It stays a two-phase loop (all task deltas are
prepared, then all are transported) and does not use ``rebase.orchestration.TaskPipeline``: that pipeline is driven
by a ``ResolvedRunConfig`` / ``StageEnv`` the LLM entrypoint has no equivalent of, and its interleaved per-task order
would change when the before-rebase evals and the model copies happen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from ...io.ckpt import load_into_model
from ...merge.runtime import compose_weighted_deltas
from .merge import (
    _summarize_merged_delta,
    norm_match_transported,
    report_merged_delta,
    resolve_delta_source,
    resolve_norm_match,
)
from .method_stages import build_method_stage
from .stages import _PreparedTaskDelta, build_prestep


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
    method_stage = build_method_stage()


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
    prepared_tasks: list[_PreparedTaskDelta] = []
    prestep = build_prestep(run_block_extension_prestep)
    for task_idx, ckpt_ref in enumerate(tuned_ref_list):
        task_label = tasks[task_idx] if task_idx < len(tasks) else f"task_{task_idx}"
        prepared_tasks.append(prestep.run(rt, task_label, ckpt_ref))

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
        return

    weights_raw = cfg.get("weights", None)
    if weights_raw is None:
        weights = [1.0] * len(tuned_ref_list)
    else:
        w = weights_raw if isinstance(weights_raw, (list, tuple)) else [float(weights_raw)]
        if len(w) < len(tuned_ref_list):
            w = w * len(tuned_ref_list)
        weights = [float(x) for x in w[: len(tuned_ref_list)]]

    # Transport each task delta
    print(f"\n=== Transporting {len(tasks) if tasks else 1} task vectors with {method_name} ===")
    transported_deltas: list[dict[str, torch.Tensor]] = []

    delta_source = resolve_delta_source(cfg)
    norm_match = resolve_norm_match(cfg)
    task_vector_norms: list[dict[str, float]] = []
    for idx, prepared_task in enumerate(prepared_tasks):
        corrected_delta = prepared_task.delta
        reference_delta = prepared_task.uncorrected_delta or corrected_delta
        delta = reference_delta if delta_source == "uncorrected" else corrected_delta
        transport_keys = prepared_task.transport_keys
        if tasks:
            label = tasks[idx]
        else:
            label = f"task_{idx}"
        print(f"\n--- '{label}' ({idx + 1}/{len(prepared_tasks)}) ---")
        t0 = time.time()

        if run_block_extension_prestep:
            prepared_task.source_model.to(device)

        transported = method_stage.run(rt, prepared_task, delta, transport_keys)
        elapsed = time.time() - t0
        print(f"  transported {len(transported)} keys in {elapsed:.1f}s")

        transported, norms = norm_match_transported(
            transported,
            corrected_delta=corrected_delta,
            reference_delta=reference_delta,
            transport_keys=transport_keys,
            norm_match=norm_match,
        )
        task_vector_norms.append(norms)
        transported_deltas.append(transported)
        if run_block_extension_prestep:
            # Release each task-local resized model immediately after its
            # matching transport completes.
            prepared_task.source_model.to("cpu")
            del prepared_task.source_model

    # Merge transported deltas
    merged_delta = compose_weighted_deltas(transported_deltas, weights)
    delta_stats = _summarize_merged_delta(merged_delta, target_base_sd)
    task_vector_report = {
        "transport_delta_source": delta_source,
        "delta_norm_match": norm_match or "none",
        "per_task": task_vector_norms,
    }
    report_merged_delta(delta_stats)
    return RebaseOutputs(merged_delta=merged_delta, delta_stats=delta_stats, task_vector_report=task_vector_report)
