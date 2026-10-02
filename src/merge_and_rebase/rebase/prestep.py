"""Depth-prestep stage objects shared by the per-task rebase pipeline (Phase 5.8).

This module is deliberately model- and dataset-agnostic: it defines the result record
(``PrestepResult``), the run-level mutable environment the stages read (``StageEnv``), the
``DepthPrestep`` / ``PrestepObserver`` protocols and the pure selection of the prestep kind from a
``RunPlan``. It must not import ``merge_and_rebase.eval`` (fresh-process test in
``tests/test_run_config_layering.py``): the concrete vision stages, which need OpenCLIP model copies,
``run_block_extension`` and the vision key filter, live in ``eval/vision_rebase/stages.py`` and are
selected there from ``select_prestep_kind``.

(The target-informed completion stages and their reference capture were retired on 2026-10-02.)
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .run_config import ResolvedRunConfig, RunPlan


class PrestepKind(enum.Enum):
    NONE = "none"
    BRACE = "brace"
    DISCRETE_INDEX = "discrete_index"


def select_prestep_kind(plan: RunPlan) -> PrestepKind:
    """Per-task prestep kind. The precedence is the legacy ``if/elif`` order of ``main()``."""
    if plan.task_block_extension_prestep:
        return PrestepKind.BRACE
    if plan.task_discrete_layer_match_prestep:
        return PrestepKind.DISCRETE_INDEX
    return PrestepKind.NONE


@dataclass
class TaskInputs:
    """One task's identity and its (vision) context: loaders, classnames, build configs."""

    task: str
    ctx: Any

    @property
    def loaders(self) -> Any:
        return self.ctx.loaders

    @property
    def source_loaders(self) -> Any:
        return self.ctx.source_loaders

    @property
    def classnames(self) -> Any:
        return self.ctx.classnames

    @property
    def build_cfg_task(self) -> Any:
        return self.ctx.build_cfg_task

    @property
    def source_build_cfg_task(self) -> Any:
        return self.ctx.source_build_cfg_task


@dataclass
class TaskModels:
    """The per-task source base / fine-tuned model copies a prestep may resize in place."""

    source_base: Any
    source_ft: Any


@dataclass
class IndependentEndpoints:
    """Run-level independent-endpoint bookkeeping (BRACE-then-merge / endpoint-average baselines)."""

    base_by_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    ft_by_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    corrected_source_template: Any | None = None
    # Deprecated (P5.14/B5): never filled or read any more; kept so older code can still construct the record.
    corrected_ft_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    corrected_ft_templates: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapturedReferences:
    """Run-time references a prestep hands on (None when not requested)."""

    #: Source-side calibration loader the BRACE prestep used.
    source_calibration_loader: Any = None


@dataclass
class PrestepResult:
    """Everything the method stage needs from the depth prestep."""

    kind: PrestepKind
    source_base_sd: dict[str, Any]
    source_ft_sd: dict[str, Any] | None = None
    task_delta: dict[str, Any] | None = None
    source_base_model: Any | None = None
    source_ft_model: Any | None = None
    layout: dict[str, Any] = field(default_factory=dict)
    activation_plan: Any | None = None
    references: CapturedReferences = field(default_factory=CapturedReferences)
    #: ``{"alignment_calibration": {...}}`` when the prestep recorded a timing bracket.
    timings: dict[str, dict[str, float]] = field(default_factory=dict)
    final_depth: int | None = None
    #: Printed by the pipeline after the prestep observers ran (legacy print order).
    completion_note: str | None = None


@dataclass
class StageEnv:
    """Run-level state read (and, for TransFusion's once-only prepare, written) by the stages."""

    resolved: ResolvedRunConfig
    plan: RunPlan
    cfg: Mapping[str, Any]
    device: str
    clf_source: Any
    clf_target: Any
    tuned_by_task: Mapping[str, str]
    native_tasks: set[str]
    patch_attn_before_rebase: bool
    source_base_sd: dict[str, Any]
    target_base_sd: dict[str, Any]
    target_hash_before: str
    block_extension_calibration_loader: Any
    run_logger: Any
    transfusion_prepared: dict[str, Any] | None = None
    recorded_extension_layout: dict[str, Any] = field(default_factory=dict)
    endpoints: IndependentEndpoints = field(default_factory=IndependentEndpoints)

    @property
    def source_depth(self) -> int:
        return self.plan.source_depth

    @property
    def target_depth(self) -> int:
        return self.plan.target_depth


class DepthPrestep(Protocol):
    """One depth-handling strategy, selected once per run and called once per task."""

    kind: PrestepKind

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        """Structural step. May leave ``task_delta`` unset (NoPrestep, same-depth direct target)."""
        ...

    def load_delta(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> PrestepResult:
        """Fill ``task_delta`` when ``run`` left it unset (identity for BRACE / discrete index)."""
        ...


class PrestepObserver(Protocol):
    """Diagnostics hooked around the prestep (target-dataset eval, source LMC)."""

    def before(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> None: ...

    def after(self, env: StageEnv, task: TaskInputs, models: TaskModels | None, result: PrestepResult) -> None: ...
