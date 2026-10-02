"""Ariadne (formerly Direct Residual) as a registered rebase method.

AriadneRebase owns the full fit pipeline -- paired boundary capture, desired
effects / sequential endpoints, resident or streaming ridge fit, label-free
task-vector scaling and the realization / fidelity diagnostics. Unlike the
state-dict transport methods (THESEUS, BiCo, ...) it never transports a source
task vector: it fits the target task vector directly from paired activations, so
the state-dict-only transport API is intentionally unavailable and
prepare takes live models and calibration loaders.

"direct_residual" is a registry alias of "ariadne" (same object).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from ...utils.cost_accounting import PhaseCostRecorder, cost_excluded, cost_phase, recording
from ..discrete_layer_match import DiscreteLayerPairing
from ..registry import register, register_alias
from ._ariadne.ablations import apply_tv_scaling
from ._ariadne.alignment import (
    apply_depth_pairing_override,
    centered_rectangular_procrustes,
    centered_ridge_alignment,
    compute_alignment_diagnostics,
    compute_desired_effects,
)
from ._ariadne.capture import (
    capture_block_gradients,
    capture_paired_boundary_activations,
    capture_tokens,
    iter_capture_block_gradients,
    iter_capture_tokens,
    paired_calibration,
)
from ._ariadne.config import (
    COMPONENT_FORWARD_ORDER,
    DirectResidualConfig,
    order_components,
    parse_direct_residual_config,
    resolve_direct_residual_preset,
)
from ._ariadne.diagnostics import (
    compute_direct_residual_task_vector_stats,
    compute_fidelity_holdout_diagnostics,
    draw_fidelity_holdout_calibration,
    measure_direct_residual_realization,
    measure_direct_residual_realization_streaming,
)
from ._ariadne.fit import (
    ResidualSufficientStatistics,
    fit_direct_residual,
    fit_sequential_source_endpoints,
)
from ._ariadne.streaming import (
    compute_alignment_diagnostics_streaming,
    compute_gradient_delta_disagreement_streaming,
    fit_direct_residual_streaming,
    measure_streaming_realization_for,
    prepare_direct_residual_streaming,
)

logger = logging.getLogger(__name__)

_SEQUENTIAL_ENDPOINTS = {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}


@dataclass(frozen=True)
class AriadnePrepared:
    """Result of AriadneRebase.prepare.

    task_vector is the fitted correction at config.strength; unit_task_vector is the unit-strength
    correction (after any tv_scaling), from which apply(..., strength=s) rescales.
    """

    task_vector: dict[str, torch.Tensor]
    unit_task_vector: dict[str, torch.Tensor]
    timing: dict[str, dict[str, float]]
    diagnostics: list[dict[str, Any]]
    extra: dict[str, Any]
    config: DirectResidualConfig = field(repr=False)


def _fit_body(
    recorder: PhaseCostRecorder,
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    target_base_sd: dict[str, torch.Tensor],
    source_loader: Any,
    target_loader: Any,
    pairing: DiscreteLayerPairing,
    config: DirectResidualConfig,
    device: str,
    family_adapter: Any = None,
    clf_source: Any = None,
    clf_target: Any = None,
    classnames: list[str] | None = None,
    source_build_cfg_task: Any = None,
    build_cfg_task: Any = None,
    source_text_features: torch.Tensor | None = None,
    target_text_features: torch.Tensor | None = None,
) -> tuple[
    dict[str, torch.Tensor], dict[str, dict[str, float]], list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]
]:
    """Run Ariadne's capture -> desired-effect -> fit -> scale pipeline once.

    Returns ``(scaled_delta, timing, diagnostics, extra, target_corrections)``. ``target_corrections`` is the
    unit-strength correction (after any tv_scaling); ``scaled_delta = strength * correction`` is applied here, not
    inside the fit (which always fits at unit strength), so ``strength=0`` is an exact native-target-base control
    (empty dict), matching ``scale_completion``'s ``gamma=0``. ``scale_completion`` is not called because it
    requires every corrected key to already be in its baseline, and Ariadne has no transport step to populate it.

    ``timing`` has ``alignment_calibration`` (capture + desired effects) and ``correction_fit`` (ridge solve)
    sub-dicts shaped like ``transport_timings[task]`` entries, each with seconds, peak device memory and peak host
    RSS (VmHWM). Peaks are read through ``recorder.mark()``/``peaks_since`` so the recorder's per-segment counter
    resets never corrupt them.

    ``extra`` holds realization_by_position / task_vector_stats (None unless ``config.realization_diagnostics``;
    computed from the unit-strength corrections, before ``strength``), alignment_diagnostics (analysis-only, never
    feeds a fit, computed in an untimed call outside both brackets), calibration, tv_scaling, fidelity_holdout and,
    for sequential endpoints, sequential_endpoints. Diagnostics do not mutate ``target_model``'s entry state
    (restored internally and asserted via a state-dict hash).

    ``procrustes_source == "gradient"``: builds source/target ``clip_contrastive_recipe`` recipes like the BiCo
    branch of ``vision_rebase`` so ``Q_j`` is fit on block-boundary gradients; ``clf_source``, ``clf_target``,
    ``classnames``, ``source_build_cfg_task``, ``build_cfg_task`` and the text features are required only in that
    mode. The activation-vs-gradient overlap diagnostic is merged into each position's diagnostics row.

    ``config.activation_storage`` selects the resident path (full per-batch banks) or the streaming path
    (O(1)-in-``num_batches`` host RAM); ``realization_diagnostics`` is supported under both.
    """
    alignment_mark = recorder.mark()
    alignment_started = time.perf_counter()
    streaming = config.activation_storage == "streaming"
    captured = None
    desired = None
    prepared = None
    gradient_mode = config.procrustes_source == "gradient"
    procrustes_diagnostics: dict[int, dict[str, Any]] = {}
    # Scalar-only per-position rows merged into the diagnostics rows in BOTH storage paths (cross-covariance rank).
    rank_diagnostics: dict[int, dict[str, Any]] = {}
    source_recipe = target_recipe = None
    if gradient_mode:
        missing = [
            name
            for name, value in (
                ("clf_source", clf_source),
                ("clf_target", clf_target),
                ("classnames", classnames),
                ("source_build_cfg_task", source_build_cfg_task),
                ("build_cfg_task", build_cfg_task),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"procrustes_source='gradient' requires {missing} to be provided")
        from ...models.grad_recipes import clip_contrastive_recipe

        source_recipe = clip_contrastive_recipe(
            clf_source,
            classnames,
            source_build_cfg_task,
            device=device,
            text_features=source_text_features,
        )
        target_recipe = clip_contrastive_recipe(
            clf_target,
            classnames,
            build_cfg_task,
            device=device,
            text_features=target_text_features,
        )
    if streaming:
        prepared = prepare_direct_residual_streaming(
            source_base_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            family_adapter=family_adapter,
            source_ft_model=source_ft_model,
            procrustes_source=config.procrustes_source,
            alignment_map=config.alignment_map,
            alignment_row_weighting=config.alignment_row_weighting,
            alignment_seed=config.alignment_seed,
            fidelity_alignment_diagnostics=bool(config.fidelity_holdout),
            source_recipe=source_recipe,
            target_recipe=target_recipe,
        )
        procrustes_diagnostics.update(prepared["procrustes_diagnostics"])
    else:
        captured = capture_paired_boundary_activations(
            source_base_model,
            source_ft_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            family_adapter=family_adapter,
            procrustes_source=config.procrustes_source,
            source_recipe=source_recipe,
            target_recipe=target_recipe,
        )
        if config.endpoint_construction == "native_delta":
            desired = compute_desired_effects(
                captured,
                pairing,
                residual_target=config.residual_target,
                procrustes_source=config.procrustes_source,
                alignment_map=config.alignment_map,
                alignment_row_weighting=config.alignment_row_weighting,
                alignment_seed=config.alignment_seed,
                # fidelity_holdout needs the SAME fitted Q_j/mu produced by this call
                # (diagnostics_out[j]["q"]/["mu_s"]/["mu_t"]), whether or not gradient_mode needs it.
                diagnostics_out=procrustes_diagnostics if (gradient_mode or config.fidelity_holdout) else None,
                rank_out=rank_diagnostics,
            )
    alignment_peak_memory_bytes, alignment_calibration_host_peak = recorder.peaks_since(alignment_mark)
    alignment_timing = {
        "alignment_calibration_seconds": time.perf_counter() - alignment_started,
        "alignment_calibration_peak_memory_bytes": alignment_peak_memory_bytes,
        "alignment_calibration_peak_host_rss_bytes": alignment_calibration_host_peak,
    }

    # Deliberately outside BOTH the alignment_calibration bracket (closed) and the correction_fit bracket (not yet
    # open): compute_alignment_diagnostics refits the centered Procrustes map in float64 purely for analysis, and
    # folding it into either bracket would inflate recorded seconds/peak memory, contaminating cost comparisons.
    # It describes the ACTIVATION-space map the fit used (configured alignment_map and row weighting); under
    # procrustes_source="gradient" that is not the Q_j the fit used, so it stays None there.
    alignment_diagnostics = None
    with cost_excluded():
        if streaming and gradient_mode:
            # Resident gradient mode computes the activation-vs-gradient delta disagreement inside
            # compute_desired_effects; streaming needs one extra (untimed) sweep for it, because
            # both maps are only known after Pass A. Merged into the rows below like the rest.
            for j, value in compute_gradient_delta_disagreement_streaming(
                source_base_model,
                source_ft_model,
                target_model,
                target_base_sd,
                prepared,
                pairing,
                device=device,
                family_adapter=family_adapter,
            ).items():
                procrustes_diagnostics[j]["activation_gradient_delta_disagreement"] = value
        if config.procrustes_source == "activation":
            if streaming:
                alignment_diagnostics = compute_alignment_diagnostics_streaming(
                    source_base_model,
                    source_ft_model,
                    target_model,
                    target_base_sd,
                    prepared,
                    pairing,
                    device=device,
                    family_adapter=family_adapter,
                )
            else:
                alignment_diagnostics = compute_alignment_diagnostics(
                    captured,
                    pairing,
                    alignment_map=config.alignment_map,
                    alignment_row_weighting=config.alignment_row_weighting,
                    alignment_seed=config.alignment_seed,
                )

    fit_mark = recorder.mark()
    fit_started = time.perf_counter()
    endpoint_diagnostics = None
    if config.endpoint_construction in {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}:
        target_corrections, diagnostics, endpoint_diagnostics = fit_sequential_source_endpoints(
            target_model,
            target_base_sd,
            captured,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
    elif streaming:
        target_corrections, diagnostics = fit_direct_residual_streaming(
            target_model,
            target_base_sd,
            source_base_model,
            source_ft_model,
            prepared,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
    else:
        target_corrections, diagnostics = fit_direct_residual(
            target_model,
            target_base_sd,
            captured,
            desired,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
    fit_peak_memory_bytes, correction_fit_host_peak = recorder.peaks_since(fit_mark)
    fit_timing = {
        "correction_fit_seconds": time.perf_counter() - fit_started,
        "correction_fit_peak_memory_bytes": fit_peak_memory_bytes,
        "correction_fit_peak_host_rss_bytes": correction_fit_host_peak,
    }
    # Streaming's procrustes_diagnostics already carries the rank keys; resident's carries them only in
    # gradient / fidelity_holdout mode (as the full diagnostics_out dict), so merge rank_diagnostics first.
    row_diagnostics = {
        j: {**rank_diagnostics.get(j, {}), **procrustes_diagnostics.get(j, {})}
        for j in set(rank_diagnostics) | set(procrustes_diagnostics)
    }
    non_unique = sorted(j for j, d in row_diagnostics.items() if d.get("procrustes_q_non_unique"))
    if non_unique:
        logger.warning(
            "Ariadne: the Procrustes cross-covariance is rank-deficient at target position(s) %s "
            "(rank < min(d_source, d_target), see the procrustes_rank / procrustes_min_dim row fields), so the "
            "polar factor Q is not unique there; the fitted Q is unchanged (diagnostic flag only).",
            non_unique,
        )
    if row_diagnostics:
        for row in diagnostics:
            extra = row_diagnostics.get(int(row.get("position", -1)))
            if extra:
                row.update(extra)

    streaming_measure = (
        measure_streaming_realization_for(
            target_model,
            target_base_sd,
            source_base_model,
            source_ft_model,
            prepared,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
        if streaming
        else None
    )
    tv_scaling_diagnostics = None
    if config.tv_scaling != "none":
        # Label-free; applied AFTER the unit-strength tau is assembled and BEFORE the per-task alpha-search (and
        # before the diagnostics block below, so both report the FINAL tau). "none" is a strict no-op, so the
        # default path is unchanged.
        target_corrections, tv_scaling_diagnostics = apply_tv_scaling(
            target_model,
            target_base_sd,
            target_corrections,
            list(range(pairing.target_depth)),
            captured,
            desired,
            config=config,
            device=device,
            family_adapter=family_adapter,
            measure_fn=streaming_measure,
        )

    realization_by_position = None
    task_vector_stats = None
    with cost_excluded():
        if bool(config.realization_diagnostics):
            positions = list(range(pairing.target_depth))
            # Explicit, never the broad CANONICAL_COMPONENT_ORDER default: packed q/k/v share ONE state-dict key
            # (attn.in_proj_weight), so key presence cannot tell which of q/k/v were fit. Passing exactly
            # order_components(config.components) checks only the components actually requested.
            fitted_components = order_components(config.components)
            if streaming:
                realization_by_position = streaming_measure(target_corrections)
            else:
                realization_by_position = measure_direct_residual_realization(
                    target_model,
                    target_base_sd,
                    target_corrections,
                    positions,
                    captured["target_batches"],
                    captured["target_base_outputs_by_position"],
                    desired,
                    device=device,
                    family_adapter=family_adapter,
                    components=fitted_components,
                )
            task_vector_stats = compute_direct_residual_task_vector_stats(
                target_corrections,
                target_base_sd,
                positions,
                components=fitted_components,
                family_adapter=family_adapter,
            )

    # fidelity_holdout: analysis-only, computed strictly AFTER target_corrections (tau) is fitted and fixed; it only
    # reads tau, so it cannot change the task vector (bit-identical-tau test). Excluded from cost accounting.
    fidelity_holdout_diagnostics = None
    with cost_excluded():
        if bool(config.fidelity_holdout):
            if streaming:
                q_by_position = prepared["q_by_position"]
                mu_s_by_position = {j: m.float() for j, m in prepared["source_mean_by_position"].items()}
                mu_t_by_position = {j: m.float() for j, m in prepared["target_mean_by_position"].items()}
            else:
                q_by_position = {j: procrustes_diagnostics[j]["q"] for j in range(pairing.target_depth)}
                mu_s_by_position = {j: procrustes_diagnostics[j]["mu_s"] for j in range(pairing.target_depth)}
                mu_t_by_position = {j: procrustes_diagnostics[j]["mu_t"] for j in range(pairing.target_depth)}
            fidelity_holdout_diagnostics = compute_fidelity_holdout_diagnostics(
                source_base_model,
                source_ft_model,
                target_model,
                target_base_sd,
                target_corrections,
                source_loader,
                target_loader,
                pairing,
                config=config,
                q_by_position=q_by_position,
                mu_s_by_position=mu_s_by_position,
                mu_t_by_position=mu_t_by_position,
                device=device,
                family_adapter=family_adapter,
            )

    strength = float(config.strength)
    with cost_phase("transport"):
        scaled_delta = (
            {} if strength == 0.0 else {key: strength * correction for key, correction in target_corrections.items()}
        )
    extra = {
        "realization_by_position": realization_by_position,
        "task_vector_stats": task_vector_stats,
        "alignment_diagnostics": alignment_diagnostics,
        "calibration": (prepared if streaming else captured)["calibration"],
        "tv_scaling": tv_scaling_diagnostics,
        "fidelity_holdout": fidelity_holdout_diagnostics,
    }
    if endpoint_diagnostics is not None:
        extra["sequential_endpoints"] = endpoint_diagnostics
    return (
        scaled_delta,
        {"alignment_calibration": alignment_timing, "correction_fit": fit_timing},
        diagnostics,
        extra,
        target_corrections,
    )


@dataclass(frozen=True)
class AriadneRebase:
    """Ariadne: fit the target task vector directly from paired activations.

    ``prepare`` runs the whole fit (it needs live source/target models and
    calibration loaders); ``apply`` returns the fitted task vector, optionally
    re-scaled to a different strength; ``transport`` is not available because
    Ariadne never transports a source state-dict delta.
    """

    name: str = "ariadne"

    def prepare(
        self,
        *,
        source_base_model: torch.nn.Module,
        source_ft_model: torch.nn.Module,
        target_model: torch.nn.Module,
        target_base_sd: Mapping[str, torch.Tensor],
        source_loader: Any,
        target_loader: Any,
        pairing: DiscreteLayerPairing,
        config: DirectResidualConfig,
        device: str,
        family_adapter: Any = None,
        recorder: PhaseCostRecorder | None = None,
        clf_source: Any = None,
        clf_target: Any = None,
        classnames: list[str] | None = None,
        source_build_cfg_task: Any = None,
        build_cfg_task: Any = None,
        source_text_features: torch.Tensor | None = None,
        target_text_features: torch.Tensor | None = None,
    ) -> AriadnePrepared:
        """Fit Ariadne once; see ``_fit_body`` for the pipeline and its brackets.

        With ``recorder=None`` the fit runs under its own ``PhaseCostRecorder``
        and ``timing["cost_phases"]`` carries its summary; a caller that already
        owns a recorder passes it and receives only the two fit brackets.
        ``family_adapter=None`` selects the vision (OpenCLIP) paths verbatim.
        """
        kwargs = dict(
            source_base_model=source_base_model,
            source_ft_model=source_ft_model,
            target_model=target_model,
            target_base_sd=target_base_sd,
            source_loader=source_loader,
            target_loader=target_loader,
            pairing=pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
            clf_source=clf_source,
            clf_target=clf_target,
            classnames=classnames,
            source_build_cfg_task=source_build_cfg_task,
            build_cfg_task=build_cfg_task,
            source_text_features=source_text_features,
            target_text_features=target_text_features,
        )
        if recorder is not None:
            scaled, timing, diagnostics, extra, unit = _fit_body(recorder, **kwargs)
        else:
            with recording(PhaseCostRecorder(device)) as own_recorder:
                scaled, timing, diagnostics, extra, unit = _fit_body(own_recorder, **kwargs)
            timing["cost_phases"] = own_recorder.summary()
        return AriadnePrepared(
            task_vector=scaled,
            unit_task_vector=unit,
            timing=timing,
            diagnostics=diagnostics,
            extra=extra,
            config=config,
        )

    def apply(
        self,
        prepared: AriadnePrepared,
        *,
        delta: Mapping[str, torch.Tensor] | None = None,
        strength: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the fitted task vector.

        The default (``strength=None``) is exactly what ``prepare`` produced
        (the fit at ``config.strength``). ``strength=s`` re-scales the
        unit-strength vector (``s == 0`` gives the empty native-target control,
        like ``prepare`` at strength 0). ``delta`` exists only for protocol
        compatibility: Ariadne has no input delta and refuses one.
        """
        if delta is not None:
            raise ValueError("Ariadne fits its own task vector; apply() does not accept an input delta.")
        if strength is None:
            return prepared.task_vector
        s = float(strength)
        if s == 0.0:
            return {}
        return {key: s * correction for key, correction in prepared.unit_task_vector.items()}

    def transport(self, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Ariadne does not transport a source state-dict delta: it fits the target task vector from paired "
            "source/target activations. Call AriadneRebase().prepare(...) with live models and calibration "
            "loaders (see merge_and_rebase.rebase.methods.ariadne)."
        )


__all__ = [
    "AriadnePrepared",
    "AriadneRebase",
    "COMPONENT_FORWARD_ORDER",
    "DirectResidualConfig",
    "ResidualSufficientStatistics",
    "apply_depth_pairing_override",
    "apply_tv_scaling",
    "capture_block_gradients",
    "capture_paired_boundary_activations",
    "capture_tokens",
    "centered_rectangular_procrustes",
    "centered_ridge_alignment",
    "compute_alignment_diagnostics",
    "compute_alignment_diagnostics_streaming",
    "compute_desired_effects",
    "compute_direct_residual_task_vector_stats",
    "compute_fidelity_holdout_diagnostics",
    "draw_fidelity_holdout_calibration",
    "fit_direct_residual",
    "fit_direct_residual_streaming",
    "fit_sequential_source_endpoints",
    "iter_capture_block_gradients",
    "iter_capture_tokens",
    "measure_direct_residual_realization",
    "measure_direct_residual_realization_streaming",
    "measure_streaming_realization_for",
    "order_components",
    "paired_calibration",
    "parse_direct_residual_config",
    "prepare_direct_residual_streaming",
    "resolve_direct_residual_preset",
]

register(AriadneRebase())
register_alias("direct_residual", "ariadne")
