"""Concrete vision depth-prestep stages and their observers (Phase 5.8).

The generic records and protocols live in ``rebase/prestep.py`` (which must not import ``eval``);
everything here is vision-specific: OpenCLIP model copies, the ``visual.`` key filter,
``run_block_extension`` and the source-LMC / target-dataset diagnostics. The bodies are the former
inline per-task block of ``main()``, moved verbatim: same call order, same prints, same events.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from ...io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ...merge.task_vectors import TaskVector
from ...rebase.discrete_layer_match import DiscreteLayerPairing, build_discrete_indexed_model
from ...rebase.methods.theseus import InterpolatedBlockActivations
from ...rebase.prestep import (
    CapturedReferences,
    DepthPrestep,
    PrestepKind,
    PrestepResult,
    StageEnv,
    TaskInputs,
    TaskModels,
    select_prestep_kind,
)
from ..block_extension import run_block_extension, select_loader
from ..target_informed_runtime import capture_residual_references, capture_resized_joint_source_inputs
from ..utils import to_cpu_fp32
from .artifacts import _state_dict_sha256
from .completion import _maybe_capture_target_residual_references
from .source_lmc import _evaluate_source_lmc, _evaluate_source_model_top1


def _visual_only_filter(k: str, v: torch.Tensor) -> bool:
    if not v.is_floating_point():
        return False
    if ".aligner." in k:
        return False
    return k.startswith("visual.")


def _resolve_source_activation_plan(
    block_extension_cfg: Any,
    extension_layout: Mapping[str, Any] | None,
) -> InterpolatedBlockActivations | None:
    """Build the interpolated-activation baseline plan, or ``None`` for ARIADNE.

    The plan is derived from the layout the extender actually realized rather
    than re-derived from the schedule, so it stays correct for insertion orders
    that draw source blocks at random.
    """
    if str(getattr(block_extension_cfg, "transport_activation_mode", "model")) == "model":
        return None
    if not extension_layout:
        raise RuntimeError(
            "transport_activation_mode='interpolate_neighbors' requires a recorded block "
            "extension layout; the extension prestep did not run."
        )
    return InterpolatedBlockActivations.from_extension_layout(extension_layout)


def build_task_models(env: StageEnv, task: str) -> TaskModels | None:
    """Per-task source base / fine-tuned copies, only when a prestep or the target-dataset eval needs them."""
    plan = env.plan
    if not (
        env.resolved.blockext_like_method
        and (
            plan.task_block_extension_prestep
            or plan.task_discrete_layer_match_prestep
            or plan.run_same_depth_direct_target
            or env.resolved.lmc.block_extension_eval_enabled
        )
    ):
        return None
    source_base_model_task = deepcopy(env.clf_source.model)
    source_ft_model_task = deepcopy(env.clf_source.model)
    load_into_model(source_base_model_task, env.source_base_sd, strict=True)
    load_into_model(source_ft_model_task, env.source_base_sd, strict=True)
    load_into_model(source_ft_model_task, load_ckpt(str(env.tuned_by_task[task])), strict=False)
    return TaskModels(source_base=source_base_model_task, source_ft=source_ft_model_task)


class _NativeDeltaMixin:
    """The no-structural-prestep delta: TransFusion's once-only prepare, else tuned ckpt minus source base."""

    kind: PrestepKind

    def load_delta(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> PrestepResult:
        if pre.task_delta is not None:
            return pre
        resolved = env.resolved
        cfg = env.cfg
        method = resolved.method
        device = env.device
        t = task.task
        if resolved.transfusion_mode:
            if env.transfusion_prepared is None:
                env.transfusion_prepared = method.prepare(
                    clf_source=env.clf_source,
                    clf_target=env.clf_target,
                    source_loaders=task.source_loaders,
                    classnames=task.classnames,
                    source_build_cfg=task.source_build_cfg_task,
                    device=device,
                    seed=int(cfg.get("seed", 42)),
                    **resolved.method_params,
                )
                transfusion_prepared = env.transfusion_prepared
                env.source_base_sd = transfusion_prepared["source_base_sd"]
                env.target_base_sd = transfusion_prepared["target_base_sd"]
                env.target_hash_before = _state_dict_sha256(env.target_base_sd)
                env.clf_target.model = transfusion_prepared["target_model_patched"]
                if transfusion_prepared.get("sanity_check_pre") is not None:
                    print(
                        f"  TransFusion perm sanity (once): "
                        f"{transfusion_prepared['sanity_check_pre']:.6f} -> "
                        f"{transfusion_prepared['sanity_check_post']:.6f} "
                        f"(delta={transfusion_prepared['sanity_check_post'] - transfusion_prepared['sanity_check_pre']:+.6f})"
                    )
                    env.run_logger.log_event(
                        "transfusion_perm_sanity",
                        metrics={
                            f"transfusion/{t}/source_zeroshot": float(transfusion_prepared["sanity_check_pre"]),
                            f"transfusion/{t}/permuted_zeroshot": float(transfusion_prepared["sanity_check_post"]),
                            f"transfusion/{t}/perm_delta": float(
                                transfusion_prepared["sanity_check_post"] - transfusion_prepared["sanity_check_pre"]
                            ),
                        },
                        context={"task": t},
                    )

            tuned_sd = method.load_task_checkpoint(
                str(env.tuned_by_task[t]),
                env.transfusion_prepared["source_model_unpatched"],
            )
            pre.task_delta = method.compute_task_delta(tuned_sd, env.source_base_sd)
        else:
            ckpt_path = str(env.tuned_by_task[t])
            sd = load_ckpt(ckpt_path)
            aligned = align_to_base_keys(sd, env.source_base_sd)
            if not aligned:
                raise ValueError(
                    f"No tensors from tuned checkpoint aligned to source base keys for task '{t}': {ckpt_path}. "
                    f"{'The base model was attention-patched before rebase, so the checkpoint must use the same patched keyspace.' if env.patch_attn_before_rebase else ''}"
                )
            tuned_sd = to_cpu_fp32(aligned)
            pre.task_delta = TaskVector.from_checkpoints(
                env.source_base_sd,
                tuned_sd,
                strict=False,
                key_filter=_visual_only_filter,
            ).delta
            if resolved.merge.base_construction == "independent_endpoint_average":
                env.endpoints.base_by_task[t] = pre.source_base_sd
                env.endpoints.ft_by_task[t] = tuned_sd

        n_keys = len(tuned_sd)
        print(f"Loaded tuned checkpoint for '{t}' ({n_keys} keys)")
        return pre


class NoPrestep(_NativeDeltaMixin):
    kind = PrestepKind.NONE

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        return PrestepResult(
            kind=self.kind,
            source_base_sd=env.source_base_sd,
            source_base_model=None if models is None else models.source_base,
            source_ft_model=None if models is None else models.source_ft,
        )


class BracePrestep:
    """BRACE/ARIADNE block-extension prestep: reference capture, resize, layout, delta, bookkeeping."""

    kind = PrestepKind.BRACE

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        resolved = env.resolved
        block_extension_cfg = resolved.block_extension_cfg
        cfg = env.cfg
        device = env.device
        t = task.task
        loaders = task.loaders
        source_loaders = task.source_loaders
        target_depth = env.target_depth
        source_base_model_task = None if models is None else models.source_base
        source_ft_model_task = None if models is None else models.source_ft
        references = CapturedReferences()

        if source_loaders is None:
            raise ValueError("Block extension preprocess requires source_loaders for calibration.")
        if source_base_model_task is None or source_ft_model_task is None:
            raise RuntimeError("Block extension preprocess expected initialized source task models.")

        calibration_loader = env.block_extension_calibration_loader
        if calibration_loader is None:
            calibration_loader = select_loader(
                block_extension_cfg.calibration_split,
                train_loader=source_loaders.train,
                test_loader=source_loaders.test,
                val_loader=source_loaders.val,
            )
        references.source_calibration_loader = calibration_loader
        if block_extension_cfg.target_residual_completion.enabled:
            # ARIADNE proposal 1: capture the native reference banks
            # (source base/ft boundary activations, paired against the
            # pretrained target model) BEFORE block extension resizes
            # source_base_model_task/source_ft_model_task in place.
            # These are the un-transported, un-inserted references the
            # completion step later regresses each inserted block's
            # c_proj projection against.
            references.residual_target_loader = select_loader(
                block_extension_cfg.calibration_split,
                train_loader=loaders.train,
                test_loader=loaders.test,
                val_loader=loaders.val,
            )
            references.residual = _maybe_capture_target_residual_references(
                config=block_extension_cfg.target_residual_completion,
                source_base_model=source_base_model_task,
                source_ft_model=source_ft_model_task,
                target_model=env.clf_target.model,
                source_loader=calibration_loader,
                target_loader=references.residual_target_loader,
                seed=int(cfg.get("seed", 42)),
                device=device,
            )
        if block_extension_cfg.joint_blockwise_correction.enabled:
            references.joint_target_loader = select_loader(
                block_extension_cfg.calibration_split,
                train_loader=loaders.train,
                test_loader=loaders.test,
                val_loader=loaders.val,
            )
            references.joint = capture_residual_references(
                source_base_model_task,
                source_ft_model_task,
                env.clf_target.model,
                calibration_loader,
                references.joint_target_loader,
                num_batches=block_extension_cfg.n_batches_act,
                seed=int(cfg.get("seed", 42)),
                device=device,
                capture_joint=True,
            )
        if block_extension_cfg.direct_p1_correction.enabled:
            references.direct_p1_target_loader = select_loader(
                block_extension_cfg.calibration_split,
                train_loader=loaders.train,
                test_loader=loaders.test,
                val_loader=loaders.val,
            )
            references.direct_p1 = capture_residual_references(
                source_base_model_task,
                source_ft_model_task,
                env.clf_target.model,
                calibration_loader,
                references.direct_p1_target_loader,
                num_batches=block_extension_cfg.n_batches_act,
                seed=int(cfg.get("seed", 42)),
                device=device,
                capture_joint=True,
            )

        task_extension_layout: dict[str, Any] = {}
        final_depth = run_block_extension(
            source_base_model=source_base_model_task,
            source_ft_model=source_ft_model_task,
            calibration_loader=calibration_loader,
            target_layers_total=target_depth,
            config=block_extension_cfg,
            device=device,
            layout_out=task_extension_layout,
            # Only the target-informed correction option reads this; every
            # standard ARIADNE path leaves the target backbone untouched.
            target_model=(env.clf_target.model if block_extension_cfg.target_shared_correction is not None else None),
        )
        env.recorded_extension_layout = dict(task_extension_layout)
        task_source_activation_plan = _resolve_source_activation_plan(block_extension_cfg, task_extension_layout)
        if final_depth != target_depth:
            raise RuntimeError(
                f"Block extension preprocess failed for task '{t}': final_depth={final_depth}, expected={target_depth}."
            )
        if block_extension_cfg.joint_blockwise_correction.enabled:
            references.joint = capture_resized_joint_source_inputs(
                source_base_model_task,
                calibration_loader,
                references.joint,
                task_extension_layout,
                device=device,
            )

        task_source_base_sd = to_cpu_fp32({k: v for k, v in source_base_model_task.state_dict().items()})
        task_source_ft_sd = to_cpu_fp32({k: v for k, v in source_ft_model_task.state_dict().items()})
        task_delta = TaskVector.from_checkpoints(
            task_source_base_sd,
            task_source_ft_sd,
            strict=True,
            key_filter=_visual_only_filter,
        ).delta
        endpoints = env.endpoints
        merge_mode = resolved.merge.mode
        base_construction = resolved.merge.base_construction
        if merge_mode == "brace_merge_then_transport" or base_construction == "independent_endpoint_average":
            endpoints.base_by_task[t] = task_source_base_sd
        if base_construction == "independent_endpoint_average":
            endpoints.ft_by_task[t] = task_source_ft_sd
        if merge_mode == "brace_merge_then_transport" and endpoints.corrected_source_template is None:
            endpoints.corrected_source_template = deepcopy(source_base_model_task).cpu()

        return PrestepResult(
            kind=self.kind,
            source_base_sd=task_source_base_sd,
            source_ft_sd=task_source_ft_sd,
            task_delta=task_delta,
            source_base_model=source_base_model_task,
            source_ft_model=source_ft_model_task,
            layout=task_extension_layout,
            activation_plan=task_source_activation_plan,
            references=references,
            final_depth=final_depth,
            completion_note=(
                f"  {t}: block extension preprocess completed "
                f"(source_depth={env.source_depth} -> {final_depth}, delta_keys={len(task_delta)})."
            ),
        )

    def load_delta(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> PrestepResult:
        return pre


class DiscreteIndexPrestep:
    """Faithful BiCo/THESEUS structural-resize control: closed-form reindex, no interpolation or correction."""

    kind = PrestepKind.DISCRETE_INDEX

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        device = env.device
        t = task.task
        if models is None or models.source_base is None or models.source_ft is None:
            raise RuntimeError("Discrete layer match expected initialized source task models.")
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        alignment_started = time.perf_counter()
        pairing = DiscreteLayerPairing.compute(env.source_depth, env.target_depth)
        source_base_model_task = build_discrete_indexed_model(models.source_base, pairing)
        source_ft_model_task = build_discrete_indexed_model(models.source_ft, pairing)
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize()
            alignment_peak_memory_bytes = float(torch.cuda.max_memory_allocated())
        else:
            alignment_peak_memory_bytes = 0.0
        timings = {
            "alignment_calibration": {
                "alignment_calibration_seconds": time.perf_counter() - alignment_started,
                "alignment_calibration_peak_memory_bytes": alignment_peak_memory_bytes,
            }
        }
        task_source_base_sd = to_cpu_fp32(source_base_model_task.state_dict())
        task_source_ft_sd = to_cpu_fp32(source_ft_model_task.state_dict())
        task_delta = TaskVector.from_checkpoints(
            task_source_base_sd, task_source_ft_sd, strict=True, key_filter=_visual_only_filter
        ).delta
        return PrestepResult(
            kind=self.kind,
            source_base_sd=task_source_base_sd,
            source_ft_sd=task_source_ft_sd,
            task_delta=task_delta,
            source_base_model=source_base_model_task,
            source_ft_model=source_ft_model_task,
            timings=timings,
            completion_note=(
                f"  {t}: discrete layer match reindex completed "
                f"(source_depth={env.source_depth} -> {env.target_depth}, delta_keys={len(task_delta)})."
            ),
        )

    def load_delta(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> PrestepResult:
        return pre


class SameDepthDirectTargetPrestep(_NativeDeltaMixin):
    """Equal-depth direct-target P1: native reference capture plus an identity layout; delta is the native one."""

    kind = PrestepKind.SAME_DEPTH_DIRECT_TARGET

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        block_extension_cfg = env.resolved.block_extension_cfg
        source_loaders = task.source_loaders
        loaders = task.loaders
        if source_loaders is None or models is None or models.source_base is None or models.source_ft is None:
            raise RuntimeError("Same-depth direct-target P1 requires source models and calibration loaders.")
        references = CapturedReferences()
        calibration_loader = select_loader(
            block_extension_cfg.calibration_split,
            train_loader=source_loaders.train,
            test_loader=source_loaders.test,
            val_loader=source_loaders.val,
        )
        references.residual_target_loader = select_loader(
            block_extension_cfg.calibration_split,
            train_loader=loaders.train,
            test_loader=loaders.test,
            val_loader=loaders.val,
        )
        references.residual = _maybe_capture_target_residual_references(
            config=block_extension_cfg.target_residual_completion,
            source_base_model=models.source_base,
            source_ft_model=models.source_ft,
            target_model=env.clf_target.model,
            source_loader=calibration_loader,
            target_loader=references.residual_target_loader,
            seed=int(env.cfg.get("seed", 42)),
            device=env.device,
        )
        task_extension_layout = {
            "direction": "extend",
            "final_blocks": [
                {
                    "position": pos,
                    "source_orig_idx": pos,
                    "span_orig_idxs": [pos],
                    "block_kind": "original",
                }
                for pos in range(env.target_depth)
            ],
            "inserted_blocks": [],
        }
        env.recorded_extension_layout = dict(task_extension_layout)
        return PrestepResult(
            kind=self.kind,
            source_base_sd=env.source_base_sd,
            source_base_model=models.source_base,
            source_ft_model=models.source_ft,
            layout=task_extension_layout,
            references=references,
        )


def build_prestep(plan: Any) -> DepthPrestep:
    """Select the per-task prestep once from the ``RunPlan``."""
    kind = select_prestep_kind(plan)
    if kind is PrestepKind.BRACE:
        return BracePrestep()
    if kind is PrestepKind.SAME_DEPTH_DIRECT_TARGET:
        return SameDepthDirectTargetPrestep()
    if kind is PrestepKind.DISCRETE_INDEX:
        return DiscreteIndexPrestep()
    return NoPrestep()


class BraceTargetEvalObserver:
    """Source zero-shot / fine-tuned top-1 on the target task's dataset, before and after the prestep."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def before(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> None:
        resolved = env.resolved
        lmc = resolved.lmc
        if not (lmc.block_extension_eval_enabled and task.source_loaders is not None):
            return
        device = env.device
        source_base_model_task = None if models is None else models.source_base
        source_ft_model_task = None if models is None else models.source_ft
        eval_row: dict[str, Any] = {
            "task": task.task,
            "split": lmc.block_extension_eval_split,
            "first_n_batches": (
                int(lmc.block_extension_eval_first_n_batches)
                if lmc.block_extension_eval_first_n_batches is not None
                else None
            ),
            "extension_applied": bool(env.plan.task_block_extension_prestep),
        }
        if source_base_model_task is None or source_ft_model_task is None:
            raise RuntimeError("Block-extension eval requested but source task models were not initialized.")

        def _top1(model: torch.nn.Module) -> float:
            return float(
                _evaluate_source_model_top1(
                    model=model,
                    clf_source=env.clf_source,
                    loaders_obj=task.source_loaders,
                    classnames_task=task.classnames,
                    source_build_cfg_task=task.source_build_cfg_task,
                    split=lmc.block_extension_eval_split,
                    first_n_batches=lmc.block_extension_eval_first_n_batches,
                    device=device,
                )
            )

        if env.plan.task_block_extension_prestep:
            eval_row["zero_shot_pre"] = _top1(source_base_model_task)
            eval_row["ft_pre"] = _top1(source_ft_model_task)
        else:
            eval_row["zero_shot"] = _top1(source_base_model_task)
            eval_row["ft"] = _top1(source_ft_model_task)
        self.rows.append(eval_row)

    def after(self, env: StageEnv, task: TaskInputs, models: TaskModels | None, result: PrestepResult) -> None:
        lmc = env.resolved.lmc
        t = task.task
        if result.kind is PrestepKind.BRACE:
            if not lmc.block_extension_eval_enabled:
                return
            device = env.device
            zero_post = _evaluate_source_model_top1(
                model=result.source_base_model,
                clf_source=env.clf_source,
                loaders_obj=task.source_loaders,
                classnames_task=task.classnames,
                source_build_cfg_task=task.source_build_cfg_task,
                split=lmc.block_extension_eval_split,
                first_n_batches=lmc.block_extension_eval_first_n_batches,
                device=device,
            )
            ft_post = _evaluate_source_model_top1(
                model=result.source_ft_model,
                clf_source=env.clf_source,
                loaders_obj=task.source_loaders,
                classnames_task=task.classnames,
                source_build_cfg_task=task.source_build_cfg_task,
                split=lmc.block_extension_eval_split,
                first_n_batches=lmc.block_extension_eval_first_n_batches,
                device=device,
            )
            last_row = self.rows[-1]
            last_row["zero_shot_post"] = float(zero_post)
            last_row["ft_post"] = float(ft_post)
            print(
                f"  {t}: source target-dataset eval "
                f"zero_shot {last_row['zero_shot_pre']:.6f}->{zero_post:.6f} "
                f"ft {last_row['ft_pre']:.6f}->{ft_post:.6f}"
            )
            env.run_logger.log_event(
                "block_extension_eval",
                metrics={
                    f"block_extension/eval/{t}/zero_shot_pre": float(last_row["zero_shot_pre"]),
                    f"block_extension/eval/{t}/zero_shot_post": float(zero_post),
                    f"block_extension/eval/{t}/ft_pre": float(last_row["ft_pre"]),
                    f"block_extension/eval/{t}/ft_post": float(ft_post),
                },
                context=last_row,
            )
        elif result.kind is PrestepKind.NONE and lmc.block_extension_eval_enabled and self.rows:
            last_row = self.rows[-1]
            print(f"  {t}: source target-dataset eval zero_shot={last_row['zero_shot']:.6f} ft={last_row['ft']:.6f}")
            env.run_logger.log_event(
                "block_extension_eval",
                metrics={
                    f"block_extension/eval/{t}/zero_shot": float(last_row["zero_shot"]),
                    f"block_extension/eval/{t}/ft": float(last_row["ft"]),
                },
                context=last_row,
            )


class SourceLmcObserver:
    """Source-endpoint LMC barrier before and after BRACE, one row per BRACE task."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._row: dict[str, Any] | None = None

    def before(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> None:
        self._row = None
        lmc = env.resolved.lmc
        if not (lmc.eval and env.plan.task_block_extension_prestep):
            return
        t = task.task
        source_base_model_task = None if models is None else models.source_base
        source_ft_model_task = None if models is None else models.source_ft
        if task.source_loaders is None or source_base_model_task is None or source_ft_model_task is None:
            raise RuntimeError("Source LMC evaluation requires initialized source models and loaders.")
        source_lmc_row: dict[str, Any] = {
            "task": t,
            "lmc_mode": env.resolved.block_extension_cfg.lmc_mode,
        }
        source_pre_base_sd = to_cpu_fp32({key: value for key, value in source_base_model_task.state_dict().items()})
        source_pre_ft_sd = to_cpu_fp32({key: value for key, value in source_ft_model_task.state_dict().items()})
        print(f"  {t}: evaluating source LMC before block extension")
        source_lmc_row["before_brace"] = _evaluate_source_lmc(
            model=source_base_model_task,
            restore_sd=source_pre_base_sd,
            endpoint_a_sd=source_pre_base_sd,
            endpoint_b_sd=source_pre_ft_sd,
            clf_source=env.clf_source,
            loaders_obj=task.source_loaders,
            classnames_task=task.classnames,
            source_build_cfg_task=task.source_build_cfg_task,
            split=lmc.eval_split,
            first_n_batches=lmc.first_n_batches,
            alphas=lmc.alphas,
            device=env.device,
        )
        self._row = source_lmc_row

    def after(self, env: StageEnv, task: TaskInputs, models: TaskModels | None, result: PrestepResult) -> None:
        source_lmc_row = self._row
        if source_lmc_row is None or result.kind is not PrestepKind.BRACE:
            return
        lmc = env.resolved.lmc
        t = task.task
        print(f"  {t}: evaluating source LMC after block extension")
        source_lmc_row["after_brace"] = _evaluate_source_lmc(
            model=result.source_base_model,
            restore_sd=result.source_base_sd,
            endpoint_a_sd=result.source_base_sd,
            endpoint_b_sd=result.source_ft_sd,
            clf_source=env.clf_source,
            loaders_obj=task.source_loaders,
            classnames_task=task.classnames,
            source_build_cfg_task=task.source_build_cfg_task,
            split=lmc.eval_split,
            first_n_batches=lmc.first_n_batches,
            alphas=lmc.alphas,
            device=env.device,
        )
        self.rows.append(source_lmc_row)
        env.run_logger.log_event(
            "source_lmc",
            metrics={
                f"source_lmc/{t}/before/max_loss_barrier": source_lmc_row["before_brace"]["max_loss_barrier"],
                f"source_lmc/{t}/after/max_loss_barrier": source_lmc_row["after_brace"]["max_loss_barrier"],
                f"source_lmc/{t}/before/max_error_barrier": source_lmc_row["before_brace"]["max_error_barrier"],
                f"source_lmc/{t}/after/max_error_barrier": source_lmc_row["after_brace"]["max_error_barrier"],
            },
            context={"task": t, "lmc_mode": env.resolved.block_extension_cfg.lmc_mode},
        )


def build_prestep_observers() -> tuple[BraceTargetEvalObserver, SourceLmcObserver]:
    """Observers in ``before`` order; ``after`` runs them in reverse (LMC after, then eval post)."""
    return BraceTargetEvalObserver(), SourceLmcObserver()
