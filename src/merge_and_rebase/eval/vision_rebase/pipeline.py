"""Vision rebase run: the pipeline between the resolved config and the summary (Phase 5.11).

``run_rebase`` takes the resolved run config, the built models (``VisionRuntime``) and the run logger and returns
the final summary. It owns the pre-model-loop checks (attention patch, checkpoint base classification), the
calibration contexts, the stage objects, the per-task loop (``rebase.orchestration.TaskPipeline``), the merge-mode
dispatch, the alpha search and the summary assembly. Bodies are the former inline blocks of ``main()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...data.vision_loaders import build_vision_calibration_loader
from ...io.ckpt import load_ckpt
from ...io.peft_helpers import normalize_attn_patch_cfg
from ...merge.methods._common import axpy_state_dict
from ...rebase.orchestration import AriadneRunRecord, CompletionRecord, TaskPipeline
from ...rebase.prestep import StageEnv
from ..block_extension import calibration_dataset_spec
from ..print_utils import pretty_print_task_accuracies
from ..utils import patch_base_for_attn, to_cpu_fp32
from .alpha_search import AlphaSearchSpec, TargetEvaluator, run_alpha_search
from .artifacts import TransportedTvSaver, TransportedTvSaveSpec, _state_dict_sha256
from .completion import build_completion_stages
from .context import build_run_calibration
from .merge import (
    _SINGLE_TRANSPORT_MODES,
    _TRANSPORT_THEN_MERGE_MODES,
    _ckpt_visual_base_coverage,
    _infer_ckpt_base,
    _visual_key_fingerprint,
    compose_rebased_deltas,
)
from .method_stages import build_method_stage
from .stages import build_prestep, build_prestep_observers, build_task_models
from .summary import RunRecord, assemble_summary


@dataclass
class VisionRuntime:
    """The built models and run-level inputs ``run_rebase`` needs beyond the resolved config."""

    cfg: dict[str, Any]
    plan: Any
    clf_source: Any
    clf_target: Any
    source_cfg: Any
    target_cfg: Any
    tuned_by_task: Mapping[str, str]
    merge_weights: list[float]
    #: Directory of the run summary (``save_transported_tvs="auto"`` derives its directory from it).
    summary_dir: Path | None


def run_rebase(resolved: Any, runtime: VisionRuntime, run_logger: Any) -> dict[str, Any]:
    """Run the per-task transport, merge dispatch and alpha search; returns the final summary dictionary."""
    cfg = runtime.cfg
    plan = runtime.plan
    clf_source = runtime.clf_source
    clf_target = runtime.clf_target
    source_cfg = runtime.source_cfg
    target_cfg = runtime.target_cfg
    tuned_by_task = runtime.tuned_by_task
    merge_weights = runtime.merge_weights
    method = resolved.method
    method_label = resolved.method_label
    block_extension_cfg = resolved.block_extension_cfg
    blockext_like_method = resolved.blockext_like_method
    block_extension_enabled = resolved.block_extension_enabled
    transfusion_mode = resolved.transfusion_mode
    strict_load = resolved.strict_load
    device = resolved.device
    merge_mode = resolved.merge.mode
    base_construction = resolved.merge.base_construction
    alpha_selection = resolved.alpha.selection
    suite = resolved.suite
    tasks = resolved.tasks
    source_depth = plan.source_depth
    target_depth = plan.target_depth

    run_block_extension_prestep = plan.run_block_extension_prestep
    if blockext_like_method:
        calibration_dataset = calibration_dataset_spec(block_extension_cfg)
        if run_block_extension_prestep:
            print(
                "Block extension preprocess: enabled "
                f"(source_depth={source_depth} -> target_depth={target_depth}, "
                f"split={block_extension_cfg.calibration_split}, "
                f"dataset={calibration_dataset!r}, "
                f"n_batches_act={block_extension_cfg.n_batches_act})."
            )
        else:
            reason = "disabled by config"
            if not block_extension_enabled:
                reason = "disabled by config"
            elif source_depth == target_depth:
                reason = "source/target depth already match"
            print(
                "Block extension preprocess: skipped "
                f"({reason}, source_depth={source_depth}, target_depth={target_depth})."
            )

    block_extension_calibration_loader = None
    calibration_dataset = calibration_dataset_spec(block_extension_cfg)
    if run_block_extension_prestep and not block_extension_cfg.skip_correction and calibration_dataset is not None:
        block_extension_calibration_loader = build_vision_calibration_loader(
            calibration_dataset,
            resolver=suite.resolver,
            preprocess=clf_source.preprocess,
            calibration_split=block_extension_cfg.calibration_split,
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=int(cfg.get("num_workers", 6)),
            pin_memory=True,
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            seed=int(cfg.get("seed", 42)),
        )
        print(
            f"Block extension preprocess: using one task-independent calibration loader from {calibration_dataset!r}."
        )

    attn_patch_cfg_raw = cfg.get("attn_patch_cfg", None)
    if attn_patch_cfg_raw is not None and not isinstance(attn_patch_cfg_raw, dict):
        raise ValueError("config['attn_patch_cfg'] must be a dict when provided.")
    patch_attn_before_rebase = bool(cfg.get("patched_attn", attn_patch_cfg_raw is not None))
    attn_patch_cfg = normalize_attn_patch_cfg(attn_patch_cfg_raw) if patch_attn_before_rebase else None

    if patch_attn_before_rebase:
        print(f"Patching source/target attention before rebase: {attn_patch_cfg}")
        source_base_sd = to_cpu_fp32(
            patch_base_for_attn(
                clf=clf_source,
                base_ckpt=None,
                strict_load=strict_load,
                attn_patch_cfg=attn_patch_cfg,
            )
        )
        target_base_sd = to_cpu_fp32(
            patch_base_for_attn(
                clf=clf_target,
                base_ckpt=None,
                strict_load=strict_load,
                attn_patch_cfg=attn_patch_cfg,
            )
        )
    else:
        source_base_sd = to_cpu_fp32({k: v for k, v in clf_source.model.state_dict().items()})
        target_base_sd = to_cpu_fp32({k: v for k, v in clf_target.model.state_dict().items()})
    target_hash_before = _state_dict_sha256(target_base_sd)

    use_humanized_classnames = not bool(cfg.get("no_humanize", True))
    print(f"Classname mode: {'humanized' if use_humanized_classnames else 'raw'}")
    print(f"Rebase method: {method_label}")

    # ---- Mixed-merging pre-pass: classify each tuned checkpoint by its base ----
    native_tasks_requested = [str(t) for t in (cfg.get("native_target_tasks", []) or [])]
    unknown_native_tasks = [t for t in native_tasks_requested if t not in tasks]
    if unknown_native_tasks:
        raise ValueError(f"native_target_tasks contains tasks not in the task list: {unknown_native_tasks}")
    auto_detect_ckpt_base = bool(cfg.get("auto_detect_ckpt_base", True))
    native_tasks: set[str] = set(native_tasks_requested)

    if native_tasks or auto_detect_ckpt_base:
        print("Checkpoint base classification (visual-key coverage vs source/target):")
        for task in tasks:
            if task in native_tasks:
                print(f"  {task}: native target checkpoint (explicit)")
                continue
            raw_sd = load_ckpt(str(tuned_by_task[task]))
            inferred = _infer_ckpt_base(raw_sd, source_base_sd=source_base_sd, target_base_sd=target_base_sd)
            if inferred is None:
                fingerprint = _visual_key_fingerprint(raw_sd)
                raise ValueError(
                    f"Tuned checkpoint for task '{task}' matches neither the source nor the target "
                    f"visual backbone ({tuned_by_task[task]}). Checkpoint fingerprint: {fingerprint}. "
                    f"Source fingerprint: {_visual_key_fingerprint(source_base_sd)}. "
                    f"Target fingerprint: {_visual_key_fingerprint(target_base_sd)}."
                )
            if inferred == "target":
                if not auto_detect_ckpt_base:
                    raise ValueError(
                        f"Tuned checkpoint for task '{task}' matches the target architecture; "
                        "add it to native_target_tasks or set auto_detect_ckpt_base=true."
                    )
                native_tasks.add(task)
                print(f"  {task}: native target checkpoint (auto-detected)")
            else:
                if strict_load:
                    coverage = _ckpt_visual_base_coverage(raw_sd, source_base_sd)
                    if coverage != 1.0:
                        raise ValueError(
                            f"Strict visual checkpoint coverage failed for task '{task}': "
                            f"coverage={coverage:.6f}, expected=1.0 ({tuned_by_task[task]})."
                        )
                print(f"  {task}: source checkpoint (transport required)")
            del raw_sd
    else:
        # auto_detect_ckpt_base=false with no native_target_tasks: nothing may be classified as native, but a
        # target-architecture checkpoint must still be refused (B3: it used to be treated as a source checkpoint
        # and silently produced an empty task vector).
        for task in tasks:
            raw_sd = load_ckpt(str(tuned_by_task[task]))
            if _infer_ckpt_base(raw_sd, source_base_sd=source_base_sd, target_base_sd=target_base_sd) == "target":
                raise ValueError(
                    f"Tuned checkpoint for task '{task}' matches the target architecture; "
                    "add it to native_target_tasks or set auto_detect_ckpt_base=true."
                )
            del raw_sd

    if native_tasks:
        if merge_mode == "none":
            raise ValueError(
                "Native target checkpoints require a merge mode; merge_mode='none' evaluates "
                "per-task transported deltas only. Use merge_mode='rebase_then_merge'."
            )
        if merge_mode in _SINGLE_TRANSPORT_MODES:
            raise ValueError(
                "Native target checkpoints cannot participate in merge_then_rebase: the merge "
                "happens on the source base, where native target deltas do not exist."
            )
        if transfusion_mode:
            raise NotImplementedError(
                "Native target checkpoints with transfusion are not supported: the permutation "
                "prepare step swaps the target keyspace. Use a theseus/bico transport method."
            )

    if base_construction == "independent_endpoint_average":
        if merge_mode not in _TRANSPORT_THEN_MERGE_MODES:
            raise ValueError(
                "base_construction='independent_endpoint_average' requires merge_mode='brace_transport_then_merge'."
            )
        if alpha_selection != "shared":
            raise ValueError(
                "base_construction='independent_endpoint_average' requires "
                "alpha_selection='shared'; per-task alpha search is not part of this baseline."
            )
        if native_tasks:
            raise ValueError(
                "base_construction='independent_endpoint_average' requires every task to be "
                "an independently transformed source endpoint; native target tasks are not allowed."
            )

    calibration = build_run_calibration(
        resolved,
        plan,
        cfg=cfg,
        clf_source=clf_source,
        clf_target=clf_target,
        source_cfg=source_cfg,
        target_cfg=target_cfg,
        native_tasks=native_tasks,
        use_humanized_classnames=use_humanized_classnames,
        block_extension_calibration_loader=block_extension_calibration_loader,
        run_logger=run_logger,
    )
    per_task = calibration.per_task
    env = StageEnv(
        resolved=resolved,
        plan=plan,
        cfg=cfg,
        device=device,
        clf_source=clf_source,
        clf_target=clf_target,
        tuned_by_task=tuned_by_task,
        native_tasks=native_tasks,
        patch_attn_before_rebase=patch_attn_before_rebase,
        source_base_sd=source_base_sd,
        target_base_sd=target_base_sd,
        target_hash_before=target_hash_before,
        block_extension_calibration_loader=calibration.block_extension_calibration_loader,
        run_logger=run_logger,
    )
    saver = TransportedTvSaver(
        TransportedTvSaveSpec.from_config(
            cfg,
            method_name=method.name,
            ariadne_like=resolved.direct_residual_like,
            ariadne_cfg=resolved.ariadne_cfg,
            summary_dir=runtime.summary_dir,
        ),
        env,
    )
    eval_observer, lmc_observer = build_prestep_observers()
    method_stage = build_method_stage(
        env,
        transport_calibration_ctx=calibration.transport_calibration_ctx,
        task_contexts=calibration.task_context_by_name,
        tasks=tasks,
        merge_weights=merge_weights,
        ariadne_calibration_ctx=calibration.ariadne_calibration_ctx,
        ariadne_calibration_meta=calibration.ariadne_calibration_meta,
    )
    ariadne_record = method_stage.record if resolved.direct_residual_like else AriadneRunRecord()
    completion_record = CompletionRecord()
    # merge_then_brace_then_transport merges deltas on the native source base first and only then runs its own
    # once-only structural step, so neither prestep fires per-task under it (gating resolved in
    # `ResolvedRunConfig.bind`).
    pipeline = TaskPipeline(
        prestep=build_prestep(plan),
        observers=(eval_observer, lmc_observer),
        method_stage=method_stage,
        completion_stages=build_completion_stages(plan, block_extension_cfg, completion_record),
        saver=saver,
        build_models=build_task_models,
    )
    loop_outputs = pipeline.run(env, tasks, calibration.task_context_by_name)

    if resolved.lmc.source_only:
        # source_only skips transport, merge and target evaluation (B2: this used to crash on a zip length
        # mismatch); only the source-side observers' rows exist.
        source_only_hash_after = _state_dict_sha256(env.target_base_sd)
        if source_only_hash_after != env.target_hash_before:
            raise RuntimeError(
                "Native target base was mutated during source-only preparation: "
                f"before={env.target_hash_before}, after={source_only_hash_after}."
            )
        return {
            "suite": resolved.suite_name,
            "tasks": resolved.tasks,
            "method": method.name,
            "method_label": method_label,
            "source_only": True,
            "target_hash_before": env.target_hash_before,
            "target_hash_after": source_only_hash_after,
            "block_extension_target_dataset_eval": eval_observer.rows,
            "source_lmc": lmc_observer.rows,
            "depth_alignment": resolved.depth_alignment_mode,
            "depth_rule_resolved": resolved.depth_rule_resolved,
        }

    merge_plan = compose_rebased_deltas(
        env,
        per_task=per_task,
        transported_deltas=loop_outputs.transported_deltas,
        original_deltas=loop_outputs.original_deltas,
        merge_weights=merge_weights,
        source_cfg=source_cfg,
        target_cfg=target_cfg,
    )
    if saver.merged_single_transport_enabled and merge_plan.single_transport_delta is not None:
        saver(f"merged_{merge_mode}", merge_plan.single_transport_delta, merged=True)

    alpha = run_alpha_search(
        AlphaSearchSpec.from_run(
            resolved,
            per_task=per_task,
            merge_weights=merge_weights,
            method_label=method_label,
            run_logger=run_logger,
        ),
        TargetEvaluator.from_plan(env, merge_plan, per_task=per_task),
        merge_plan,
    )

    alpha_display_label = "hierarchical(per_task+global)" if alpha.hierarchical else alpha_selection
    pretty_print_task_accuracies(
        resolved.suite_name,
        f"{method_label}, alpha={alpha_display_label}",
        f"A={source_cfg.pretrained} → B={target_cfg.pretrained}",
        per_task,
        alpha.rebase_test_accs,
        alpha.norm_accs,
        single_accs=alpha.baseline_test_accs,
        baseline_label=alpha.baseline_label,
        result_label=alpha.result_label,
    )

    if alpha.hierarchical and alpha.per_task_premerge_alphas is not None:
        print("\nHierarchical alphas (per-task premerge + global merge alpha):")
        for item, a in zip(per_task, alpha.per_task_premerge_alphas, strict=True):
            print(f"  {item['task']}: premerge={a:.3f}  global={alpha.best_alpha:.3f}")
    elif alpha_selection == "per_task":
        print("\nSelected test-time alpha by task:")
        for item, r_a, b_a in zip(
            per_task, alpha.selected_alpha_by_task, alpha.selected_baseline_alpha_by_task, strict=True
        ):
            print(f"  {item['task']}: rebase={r_a:.3f}  baseline={b_a:.3f}")

    saved_merged_path: str | None = None
    if cfg.get("save_merged"):
        if merge_mode == "none":
            print(
                "save_merged was requested, but task-independent alpha mode does not produce a single merged checkpoint; skipping save."
            )
        else:
            best_merged_state = axpy_state_dict(
                env.target_base_sd,
                alpha.rebased_deltas[0],
                alpha=float(alpha.best_alpha),
            )
            out_path = str(cfg["save_merged"])
            out_parent = Path(out_path).parent
            if str(out_parent):
                out_parent.mkdir(parents=True, exist_ok=True)
            torch.save(to_cpu_fp32(best_merged_state), out_path)
            saved_merged_path = out_path
            print(f"Saved merged model (alpha={alpha.best_alpha:.3f}) -> {out_path}")

    # TransFusion's once-only prepare may have swapped these run-level objects on ``env``.
    target_hash_before = env.target_hash_before
    target_hash_after = _state_dict_sha256(env.target_base_sd)
    if target_hash_after != target_hash_before:
        raise RuntimeError(
            "Native target base was mutated during merge/transport preparation: "
            f"before={target_hash_before}, after={target_hash_after}."
        )

    return assemble_summary(
        RunRecord.from_run(
            resolved,
            native_tasks=native_tasks,
            per_task=per_task,
            loop_outputs=loop_outputs,
            calibration=calibration,
            merge_plan=merge_plan,
            alpha=alpha,
            ariadne=ariadne_record,
            completion=completion_record,
            block_extension_eval_rows=eval_observer.rows,
            source_lmc_rows=lmc_observer.rows,
            transported_artifacts=saver.artifacts,
            target_hash_before=target_hash_before,
            target_hash_after=target_hash_after,
            saved_merged_path=saved_merged_path,
            save_policy=saver.spec.summary_record(),
        )
    )
