"""Method-stage records and protocols for the per-task rebase pipeline (Phase 5.9).

Like ``rebase/prestep.py`` this module is model- and dataset-agnostic and must not import
``merge_and_rebase.eval``: the concrete vision stages (``TransportMethodStage``, ``AriadneStage``)
live in ``eval/vision_rebase/method_stages.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .merge_modes import _SINGLE_TRANSPORT_MODES
from .prestep import DepthPrestep, PrestepObserver, PrestepResult, StageEnv, TaskInputs, TaskModels


@dataclass
class MethodResult:
    """One task's transported task vector plus the timing brackets the run summary records."""

    transported_delta: dict[str, Any]
    #: Fitted transport state (``None`` for Ariadne).
    prepared: Any = None
    transport_timing: dict[str, float] | None = None
    cost_phases: dict[str, Any] | None = None
    #: Ariadne only: ``alignment_calibration`` / ``correction_fit`` timing brackets.
    alignment_calibration: dict[str, float] | None = None
    correction_fit: dict[str, float] | None = None
    #: LLM only: per-task transported-vs-reference norm record (``task_vectors.per_task`` in the summary).
    task_vector_norms: dict[str, float] | None = None


class MethodStage(Protocol):
    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> MethodResult: ...


@dataclass
class AriadneRunRecord:
    """Per-run Ariadne outputs: replaces nine parallel ``direct_residual_*`` dicts (same summary keys)."""

    calibration_meta: dict[str, Any] = field(default_factory=lambda: {"calibration_data": "task_local"})
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    realization: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    task_vector_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    alignment_diagnostics: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    calibration_by_task: dict[str, Any] = field(default_factory=dict)
    tv_scaling: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    fidelity_holdout: dict[str, Any] = field(default_factory=dict)
    sequential_endpoints: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    loaded_vectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The depth pairing actually used, recorded once per run (``None`` until the first Ariadne use).
    pairing: dict[str, Any] | None = None

    def record_pairing(self, pairing: Any, depth_pairing: str) -> None:
        self.pairing = {
            "source_depth": pairing.source_depth,
            "target_depth": pairing.target_depth,
            "depth_pairing": depth_pairing,
            "pairing": list(pairing.pairing),
        }

    def record_fit(
        self,
        task: str,
        diagnostics: list[dict[str, Any]],
        extra: Mapping[str, Any],
        *,
        sequential: bool = True,
    ) -> None:
        """Store one task's fit outputs. The once-only merged fit does not carry sequential endpoints."""
        self.diagnostics[task] = diagnostics
        self.realization[task] = extra["realization_by_position"]
        self.task_vector_stats[task] = extra["task_vector_stats"]
        self.alignment_diagnostics[task] = extra["alignment_diagnostics"]
        self.calibration_by_task[task] = extra["calibration"]
        self.tv_scaling[task] = extra["tv_scaling"]
        self.fidelity_holdout[task] = extra["fidelity_holdout"]
        if sequential and "sequential_endpoints" in extra:
            self.sequential_endpoints[task] = extra["sequential_endpoints"]


@dataclass
class TaskLoopOutputs:
    """What the per-task loop hands to the merge / alpha-search / summary stages."""

    #: Transported deltas, one per transported (non-native) task, in task order.
    transported_deltas: list[dict[str, Any]] = field(default_factory=list)
    #: Source-side task deltas of every non-native task (collected, not transported, under single-transport modes).
    original_deltas: list[dict[str, Any]] = field(default_factory=list)
    transport_timings: dict[str, dict[str, float]] = field(default_factory=dict)
    # Timing/memory brackets (wandb-visible), parallel to ``transport_timings``: ``alignment_calibration_timings``
    # covers whatever depth/width-alignment step runs before any correction is fitted, ``correction_fit_timings``
    # the Ariadne fit only. Both default to ``{}`` and are always present in the summary, even for methods that
    # never populate them, so downstream JSON parsing is uniform across every method.
    alignment_calibration_timings: dict[str, dict[str, float]] = field(default_factory=dict)
    correction_fit_timings: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Per-task activation_collection / transformation / transport cost split (``utils.cost_accounting``).
    cost_phase_timings: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: LLM only: one norm record per transported task, in task order.
    task_vector_norms: list[dict[str, float]] = field(default_factory=list)


class TaskPipeline:
    """The per-task loop: prestep (with observers) -> method stage -> saver.

    Everything model- or dataset-specific is injected (``build_models`` and the stage objects), so this class
    imports nothing from ``eval``. ``saver(task, transported_delta)`` is called once per transported task.
    """

    def __init__(
        self,
        *,
        prestep: DepthPrestep,
        observers: Sequence[PrestepObserver],
        method_stage: Any,
        saver: Callable[[str, dict[str, Any]], None],
        build_models: Callable[[StageEnv, str], TaskModels | None],
    ) -> None:
        self.prestep = prestep
        self.observers = tuple(observers)
        self.method_stage = method_stage
        self.saver = saver
        self.build_models = build_models

    def run(self, env: StageEnv, tasks: Sequence[str], task_contexts: Mapping[str, Any]) -> TaskLoopOutputs:
        resolved = env.resolved
        out = TaskLoopOutputs()
        # merge_in_source_then_fit (an Ariadne config field, distinct from the top-level `merge_mode` cfg key)
        # merges every task's native delta ONCE before the per-task loop and fits one shared correction; the
        # stage caches it and the loop reuses it for every task.
        # No method stage when the run stops after the before-rebase eval (LLM eval_before_rebase_only).
        if resolved.direct_residual_like and self.method_stage is not None:
            self.method_stage.precompute(env)

        for task in tasks:
            task_ctx = task_contexts[task]

            if task in env.native_tasks:
                print(f"  {task}: native target checkpoint — skipping transport")
                continue

            task_in = TaskInputs(task, task_ctx)
            # Drop the previous task's models / prestep result before building the next ones (peak memory).
            task_models = pre = None
            task_models = self.build_models(env, task)
            for observer in self.observers:
                observer.before(env, task_in, task_models)
            pre = self.prestep.run(env, task_in, task_models)
            # LMC "after" is logged before the target-dataset eval "post" (legacy event order).
            for observer in reversed(self.observers):
                observer.after(env, task_in, task_models, pre)
            if pre.completion_note is not None:
                print(pre.completion_note)
            if "alignment_calibration" in pre.timings:
                out.alignment_calibration_timings[task] = pre.timings["alignment_calibration"]

            if resolved.lmc.source_only:
                continue

            # TransFusion's once-only prepare rebinds run-level objects on ``env`` (see NoPrestep.load_delta).
            pre = self.prestep.load_delta(env, task_in, pre)
            task_delta = pre.task_delta

            print(f"\n--- Transporting '{task}' with method '{resolved.method.name}' ---")
            if resolved.merge.mode not in _SINGLE_TRANSPORT_MODES:
                method_result = self.method_stage.run(env, task_in, pre)
                out.transport_timings[task] = method_result.transport_timing
                out.cost_phase_timings[task] = method_result.cost_phases
                if method_result.task_vector_norms is not None:
                    out.task_vector_norms.append(method_result.task_vector_norms)
                if method_result.alignment_calibration is not None:
                    out.alignment_calibration_timings[task] = method_result.alignment_calibration
                if method_result.correction_fit is not None:
                    out.correction_fit_timings[task] = method_result.correction_fit

                transported_delta = method_result.transported_delta

                out.transported_deltas.append(transported_delta)
                out.original_deltas.append(task_delta)
                print(f"  {task}: transported delta computed for {len(transported_delta)} params")
                if env.run_logger is not None:
                    env.run_logger.log_event(
                        "transport_task_end",
                        metrics={f"rebase/{task}/transported_param_count": float(len(transported_delta))},
                        context={"task": task, "method": resolved.method.name},
                    )

                self.saver(task, transported_delta)
            else:
                out.original_deltas.append(task_delta)
                print(f"  {task}: delta collected for merge_then_rebase ({len(task_delta)} params)")
        return out
