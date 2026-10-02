"""Method-stage records and protocols for the per-task rebase pipeline (Phase 5.9).

Like ``rebase/prestep.py`` this module is model- and dataset-agnostic and must not import
``merge_and_rebase.eval``: the concrete vision stages (``TransportMethodStage``, ``AriadneStage``)
live in ``eval/vision_rebase/method_stages.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .prestep import PrestepResult, StageEnv, TaskInputs


@dataclass
class MethodResult:
    """One task's transported task vector plus the timing brackets the run summary records."""

    transported_delta: dict[str, Any]
    #: Fitted transport state (``None`` for Ariadne and for the transport-free direct-target arm);
    #: the completion stages reuse its projection transforms.
    prepared: Any = None
    transport_timing: dict[str, float] | None = None
    cost_phases: dict[str, Any] | None = None
    #: Ariadne only: ``alignment_calibration`` / ``correction_fit`` timing brackets.
    alignment_calibration: dict[str, float] | None = None
    correction_fit: dict[str, float] | None = None


class MethodStage(Protocol):
    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> MethodResult: ...


class CompletionStage(Protocol):
    """Target-informed correction applied to a fitted task vector (target residual, joint, direct P1)."""

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult, result: MethodResult) -> MethodResult: ...


@dataclass
class CompletionRecord:
    """Per-task diagnostics of the completion stages (summary keys keep their legacy names)."""

    residual: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    joint_blockwise: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    direct_p1: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def direct_target_p1_requested(plan: Any, block_extension_cfg: Any) -> bool:
    """Transport-free proposal-1 arm: ordinary parameter transport is skipped entirely."""
    return bool(
        (plan.task_block_extension_prestep or plan.run_same_depth_direct_target)
        and block_extension_cfg.target_residual_completion.enabled
        and block_extension_cfg.target_residual_completion.mode == "direct_target"
    )


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
