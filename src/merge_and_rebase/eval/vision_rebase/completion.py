"""Reference capture and completion stages for the ARIADNE target-informed corrections.

``_maybe_capture_*`` / ``_maybe_complete_*`` are re-exported from ``eval.vision_rebase`` (tests import them).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...rebase.orchestration import CompletionRecord, CompletionStage, MethodResult
from ...rebase.prestep import PrestepResult, StageEnv, TaskInputs
from ..target_informed_runtime import (
    capture_residual_references,
    complete_direct_p1_shared_correction,
    complete_joint_blockwise,
    complete_residuals,
    complete_residuals_direct,
    projection_transforms,
    scale_completion,
)
from ..target_residual_completion import JointCorrectionConfig, ResidualCompletionConfig


def _maybe_capture_target_residual_references(
    *,
    config: ResidualCompletionConfig,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_loader: Any,
    target_loader: Any,
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    """Capture ARIADNE proposal-1 native reference banks, or no-op when disabled.

    Must be called before ``run_block_extension`` structurally resizes
    ``source_base_model``/``source_ft_model``: the native references are the
    un-resized source model's own boundary activations, paired against the
    pretrained target model at the doubled positions those source blocks will
    be inserted at. Returns ``None`` when the option is disabled, so callers
    that thread the result through unconditionally get a byte-identical no-op.
    """
    if not config.enabled:
        return None
    return capture_residual_references(
        source_base_model,
        source_ft_model,
        target_model,
        source_loader,
        target_loader,
        num_batches=config.num_batches,
        seed=seed,
        device=device,
        target_scope=config.target_scope,
    )


def _maybe_complete_target_residual_task_vector(
    *,
    config: ResidualCompletionConfig,
    references: dict[str, Any] | None,
    prepared: Any,
    layout: Mapping[str, Any],
    target_model: torch.nn.Module,
    target_base_sd: Mapping[str, torch.Tensor],
    transported_delta: dict[str, torch.Tensor],
    target_loader: Any,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
    """Complete the transported task vector's residual-writing keys, or no-op.

    Which projections those are is ``config.components``: ``mlp.c_proj`` alone
    by default, optionally ``attn.out_proj`` as well in ``direct_target`` mode.

    Runs after transport is fitted, using the already-fitted ``prepared``
    transforms; it only ever adds to the *task vector*, never the target base
    weights. ``config.enabled=False`` or missing ``references`` (the option
    was disabled when references would have been captured) returns
    ``transported_delta`` completely unchanged -- same dict object, so a
    caller comparing state-dict hashes sees byte-identical output. Fitting
    always happens at gamma=1 (see ``complete_residuals``); ``config.strength``
    is applied afterwards by ``scale_completion``, and ``strength=0.0`` is a
    true null ablation because ``scale_completion`` short-circuits to the
    baseline in that case.
    """
    if not config.enabled or references is None:
        return transported_delta, None
    if config.mode == "direct_target":
        # Transport-free arm. There is no tau_t to complete: the caller has
        # already skipped the transport fit, so ``transported_delta`` must be
        # empty and the fitted correction is the entire task vector. The
        # baseline it is scaled against is therefore an explicit zero delta,
        # which keeps gamma=0 an exact native-target-base control (identical
        # semantics to the transport-aware arm's gamma=0).
        if transported_delta:
            raise ValueError(
                "mode='direct_target' requires an empty transported task vector: the "
                f"caller passed {len(transported_delta)} transported keys, so the arm would "
                "not be transport-free"
            )
        target_corrections, diagnostics = complete_residuals_direct(
            target_model,
            target_base_sd,
            references,
            layout,
            target_loader,
            config=config,
            device=device,
        )
        zero_baseline = {key: torch.zeros_like(value) for key, value in target_corrections.items()}
        return scale_completion(zero_baseline, target_corrections, config.strength), diagnostics
    transforms = projection_transforms(prepared, layout, target_scope=config.target_scope)
    _source_corrections, target_corrections, diagnostics = complete_residuals(
        target_model,
        target_base_sd,
        transported_delta,
        references,
        transforms,
        layout,
        target_loader,
        config=config,
        device=device,
    )
    completed_delta = scale_completion(transported_delta, target_corrections, config.strength)
    return completed_delta, diagnostics


def _maybe_complete_joint_blockwise_task_vector(
    *,
    config: JointCorrectionConfig,
    references: dict[str, Any] | None,
    prepared: Any,
    layout: Mapping[str, Any],
    target_model: torch.nn.Module,
    target_base_sd: Mapping[str, torch.Tensor],
    transported_delta: dict[str, torch.Tensor],
    target_loader: Any,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
    """Run the opt-in frozen-map Option 3 blockwise solve."""
    if not config.enabled or references is None:
        return transported_delta, None
    transforms = projection_transforms(prepared, layout, target_scope="inserted")
    _source, target_corrections, diagnostics = complete_joint_blockwise(
        target_model,
        target_base_sd,
        transported_delta,
        references,
        transforms,
        layout,
        target_loader,
        config=config,
        device=device,
    )
    completed = dict(transported_delta)
    for key, correction in target_corrections.items():
        if key in completed:
            completed[key] = completed[key] + correction.to(completed[key])
        else:
            completed[key] = correction
    return completed, diagnostics


def _maybe_complete_direct_p1_task_vector(
    *,
    config: JointCorrectionConfig,
    references: dict[str, Any] | None,
    prepared: Any,
    layout: Mapping[str, Any],
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    target_base_sd: Mapping[str, torch.Tensor],
    transported_delta: dict[str, torch.Tensor],
    source_loader: Any,
    target_loader: Any,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
    """Fit the direct shared-geometry/P1 correction with frozen maps."""
    if not config.enabled or references is None:
        return transported_delta, None
    transforms = projection_transforms(prepared, layout, target_scope="inserted")
    target_corrections, diagnostics = complete_direct_p1_shared_correction(
        source_base_model,
        source_ft_model,
        target_model,
        target_base_sd,
        transported_delta,
        references,
        transforms,
        layout,
        source_loader,
        target_loader,
        config=config,
        device=device,
    )
    completed = dict(transported_delta)
    for key, correction in target_corrections.items():
        if key in completed:
            completed[key] = completed[key] + correction.to(completed[key])
        else:
            completed[key] = correction
    return completed, diagnostics


class TargetResidualStage:
    """ARIADNE proposal 1: complete the transported task vector's residual-writing projections.

    In the same-depth direct-target path the references and the identity layout are prepared by the
    prestep; both cases leave the target base unchanged.
    """

    def __init__(self, record: CompletionRecord) -> None:
        self.record = record

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult, result: MethodResult) -> MethodResult:
        config = env.resolved.block_extension_cfg.target_residual_completion
        result.transported_delta, task_residual_diagnostics = _maybe_complete_target_residual_task_vector(
            config=config,
            references=pre.references.residual,
            prepared=result.prepared,
            layout=pre.layout,
            target_model=env.clf_target.model,
            target_base_sd=env.target_base_sd,
            transported_delta=result.transported_delta,
            target_loader=pre.references.residual_target_loader,
            device=env.device,
        )
        if task_residual_diagnostics is not None:
            self.record.residual[task.task] = task_residual_diagnostics
        return result


class JointBlockwiseStage:
    """Opt-in frozen-map Option 3 blockwise solve."""

    def __init__(self, record: CompletionRecord) -> None:
        self.record = record

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult, result: MethodResult) -> MethodResult:
        t = task.task
        result.transported_delta, task_joint_diagnostics = _maybe_complete_joint_blockwise_task_vector(
            config=env.resolved.block_extension_cfg.joint_blockwise_correction,
            references=pre.references.joint,
            prepared=result.prepared,
            layout=pre.layout,
            target_model=env.clf_target.model,
            target_base_sd=env.target_base_sd,
            transported_delta=result.transported_delta,
            target_loader=pre.references.joint_target_loader,
            device=env.device,
        )
        if task_joint_diagnostics is not None:
            self.record.joint_blockwise[t] = task_joint_diagnostics
            env.run_logger.log_event(
                "joint_blockwise_correction",
                metrics={
                    f"rebase/{t}/joint_blocks": float(len(task_joint_diagnostics)),
                    f"rebase/{t}/joint_objective_after": float(
                        sum(row["objective_after"] for row in task_joint_diagnostics)
                    ),
                },
                context={"task": t, "method": env.resolved.method.name},
            )
        return result


class DirectP1Stage:
    """Direct shared-geometry/P1 correction with frozen maps."""

    def __init__(self, record: CompletionRecord) -> None:
        self.record = record

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult, result: MethodResult) -> MethodResult:
        t = task.task
        result.transported_delta, task_direct_p1_diagnostics = _maybe_complete_direct_p1_task_vector(
            config=env.resolved.block_extension_cfg.direct_p1_correction,
            references=pre.references.direct_p1,
            prepared=result.prepared,
            layout=pre.layout,
            source_base_model=pre.source_base_model,
            source_ft_model=pre.source_ft_model,
            target_model=env.clf_target.model,
            target_base_sd=env.target_base_sd,
            transported_delta=result.transported_delta,
            source_loader=pre.references.source_calibration_loader,
            target_loader=pre.references.direct_p1_target_loader,
            device=env.device,
        )
        if task_direct_p1_diagnostics is not None:
            self.record.direct_p1[t] = task_direct_p1_diagnostics
            env.run_logger.log_event(
                "direct_p1_correction",
                metrics={
                    f"rebase/{t}/direct_p1_blocks": float(len(task_direct_p1_diagnostics)),
                    f"rebase/{t}/direct_p1_objective_after": float(
                        sum(row["objective_after"] for row in task_direct_p1_diagnostics)
                    ),
                },
                context={"task": t, "method": env.resolved.method.name},
            )
        return result


def build_completion_stages(
    plan: Any, block_extension_cfg: Any, record: CompletionRecord
) -> tuple[CompletionStage, ...]:
    """Completion stages in their fixed order (target residual, joint blockwise, direct P1), built once."""
    stages: list[CompletionStage] = []
    if (
        plan.task_block_extension_prestep or plan.run_same_depth_direct_target
    ) and block_extension_cfg.target_residual_completion.enabled:
        stages.append(TargetResidualStage(record))
    if plan.task_block_extension_prestep and block_extension_cfg.joint_blockwise_correction.enabled:
        stages.append(JointBlockwiseStage(record))
    if plan.task_block_extension_prestep and block_extension_cfg.direct_p1_correction.enabled:
        stages.append(DirectP1Stage(record))
    return tuple(stages)
