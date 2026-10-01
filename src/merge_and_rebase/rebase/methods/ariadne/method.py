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

from ....utils.cost_accounting import PhaseCostRecorder, cost_excluded, cost_phase, recording
from ...discrete_layer_match import DiscreteLayerPairing
from ...registry import register, register_alias
from .ablations import apply_tv_scaling
from .alignment import compute_alignment_diagnostics, compute_desired_effects
from .capture import capture_paired_boundary_activations
from .config import DirectResidualConfig, order_components
from .diagnostics import (
    compute_direct_residual_task_vector_stats,
    compute_fidelity_holdout_diagnostics,
    measure_direct_residual_realization,
)
from .fit import fit_direct_residual, fit_sequential_source_endpoints
from .streaming import (
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

    task_vector is the fitted correction at config.strength (what the
    historical _direct_residual_fit_body returned as scaled_delta);
    unit_task_vector is the unit-strength correction (after any tv_scaling),
    from which apply(..., strength=s) rescales.
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

    Mirrors the exact ``torch.cuda.reset_peak_memory_stats()`` /
    ``torch.cuda.synchronize()`` / ``torch.cuda.max_memory_allocated()`` /
    ``time.perf_counter()`` idiom the existing ``transport_timings`` bracket
    uses, split into two brackets: one around alignment/capture
    (``capture_paired_boundary_activations`` + ``compute_desired_effects``,
    the "calibration" half) and one around the ridge solve itself
    (``fit_direct_residual``). ``theta_j_corrected = theta_j_native +
    strength * correction_j`` is applied here (not inside
    ``fit_direct_residual``, which always fits at unit strength) so that
    ``strength=0`` is an exact native-target-base control, matching
    ``target_informed_runtime.scale_completion``'s ``gamma=0`` contract --
    replicated directly rather than called through ``scale_completion``
    itself, since that helper requires every corrected key to already be
    present in its ``baseline`` argument, and Direct Residual's baseline is
    the empty ``transported_delta={}`` (there is no transport step to have
    populated it).

    Returns ``(scaled_delta, timing, diagnostics, extra)`` where ``timing`` has
    ``"alignment_calibration"``/``"correction_fit"`` sub-dicts, each shaped
    like a ``transport_timings[task]`` entry, and ``extra`` is
    ``{"realization_by_position": ..., "task_vector_stats": ...,
    "alignment_diagnostics": ...}``. The first two are ``None`` unless
    ``config.realization_diagnostics`` is set, and are computed from the
    unscaled, unit-strength ``target_corrections`` ``fit_direct_residual``
    returns -- i.e. before ``strength`` is applied -- matching the plan's
    "AFTER the task vector tau (unit strength, as returned by the fit) is
    assembled" requirement. ``alignment_diagnostics`` is always present (keyed
    by target position): it comes from a separate, untimed call to
    ``compute_alignment_diagnostics`` -- deliberately outside both the
    ``alignment_calibration`` and ``correction_fit`` timing/peak-memory
    brackets, so its own float64 recomputation cost never contaminates either
    (see the inline comment at the call site) -- is analysis-only, and never
    feeds any fit regardless of ``config.residual_target``. Neither
    realization-diagnostics call mutates ``target_model``'s entry state (both
    restore it internally and assert so via a state-dict hash).

    When ``config.procrustes_source == "gradient"``, builds source/target
    ``clip_contrastive_recipe`` gradient recipes exactly like the BiCo branch
    of ``vision_rebase`` (same classifier/classnames/build-cfg/text-features arguments) and
    passes them into ``capture_paired_boundary_activations`` so ``Q_j`` is
    fit on block-boundary gradients instead of activations; the six extra
    kwargs (``clf_source``, ``clf_target``, ``classnames``,
    ``source_build_cfg_task``, ``build_cfg_task``, ``source_text_features``/
    ``target_text_features``) are required only in that mode. The
    activation-vs-gradient Procrustes overlap diagnostic is merged into each
    position's diagnostics row by position.

    ``config.activation_storage`` branches between the resident path above
    (full per-batch banks, unchanged) and the streaming path
    (``prepare_direct_residual_streaming`` + ``fit_direct_residual_streaming``,
    O(1)-in-``num_batches`` host RAM); ``parse_direct_residual_config`` has
    accepts ``realization_diagnostics`` under streaming too (the streaming
    realization measurement, ``measure_streaming_realization_for``). Both paths additionally record each bracket's exact peak host RSS
    (``{bracket}_peak_host_rss_bytes``, VmHWM reset at the bracket start via
    ``recorder``; see utils.cost_accounting) so campaigns can see streaming's
    host memory stay flat as ``num_batches`` grows while resident's does not.
    Every bracket peak is read through ``recorder.mark()``/``peaks_since`` so the
    recorder's per-segment counter resets never corrupt it.
    """
    alignment_mark = recorder.mark()
    alignment_started = time.perf_counter()
    streaming = config.activation_storage == "streaming"
    captured = None
    desired = None
    prepared = None
    gradient_mode = config.procrustes_source == "gradient"
    procrustes_diagnostics: dict[int, dict[str, Any]] = {}
    # Scalar-only per-position rows merged into the diagnostics rows in BOTH storage paths (rank
    # diagnostics of the cross-covariance the map was solved from; see procrustes_rank_diagnostics).
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
        from ....models.grad_recipes import clip_contrastive_recipe

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
                # fidelity_holdout needs the SAME fitted Q_j/mu this compute_desired_effects
                # call produces, stored under diagnostics_out[j]["q"]/["mu_s"]/["mu_t"]
                # (see compute_desired_effects's docstring) -- collected here whether
                # or not gradient_mode also needs it, so its own request never has to
                # special-case which mode it's running under.
                diagnostics_out=procrustes_diagnostics if (gradient_mode or config.fidelity_holdout) else None,
                rank_out=rank_diagnostics,
            )
    alignment_peak_memory_bytes, alignment_calibration_host_peak = recorder.peaks_since(alignment_mark)
    alignment_timing = {
        "alignment_calibration_seconds": time.perf_counter() - alignment_started,
        "alignment_calibration_peak_memory_bytes": alignment_peak_memory_bytes,
        "alignment_calibration_peak_host_rss_bytes": alignment_calibration_host_peak,
    }

    # Deliberately outside BOTH the alignment_calibration bracket above (just
    # closed) and the correction_fit bracket below (not yet opened):
    # compute_alignment_diagnostics recomputes the same centered Procrustes
    # fit a second time purely for analysis, with several float64 N x d_t
    # temporaries (N in the tens of thousands of rows). Folding it into
    # either bracket would inflate that bracket's recorded seconds/peak-
    # memory bytes -- even in the default residual_target="transported_delta"
    # path -- contaminating any cross-code-generation cost comparison for a
    # quantity these diagnostics never feed into.
    # Both compute_alignment_diagnostics variants describe the ACTIVATION-space map the fit
    # actually used (the configured alignment_map and row weighting, not a fresh uniform polar
    # fit); under procrustes_source="gradient" that is not the Q_j the fit used, so it is not
    # reported there (None) rather than reported for the wrong map.
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
        # Label-free, applied AFTER the unit-strength tau is assembled but
        # BEFORE the caller's per-task alpha-search (and therefore before the
        # realization_diagnostics/task_vector_stats block below, so both
        # report the FINAL tau that alpha-search actually sees). tv_scaling
        # defaults to "none" (a strict no-op, see apply_tv_scaling), so this
        # branch never executes for the historical, golden-hash-pinned path.
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
            # Explicit, never the broad CANONICAL_COMPONENT_ORDER default: packed
            # q/k/v share ONE physical state-dict key (attn.in_proj_weight), so a
            # presence check keyed only off "is this key in target_corrections"
            # cannot tell which of q/k/v were actually fit -- e.g. a v-only
            # run's in_proj_weight key exists in target_corrections
            # with only its v-rows nonzero, and checking q/k against that same
            # key would falsely report them "present" too. Passing exactly
            # order_components(config.components) sidesteps this: only names the
            # caller actually asked to fit are ever checked.
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

    # fidelity_holdout diagnostic: analysis-only, computed strictly AFTER
    # target_corrections (tau) is already fitted and fixed -- it only reads
    # target_corrections, never feeds back into it -- so it cannot, by
    # construction, change the task vector this run produces (see
    # tests/test_direct_residual_fidelity_holdout_20260925.py's bit-identical
    # -tau assertion). Excluded from cost accounting for the same reason
    # realization_diagnostics is above: it is not part of the method's cost.
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
            "loaders (see merge_and_rebase.rebase.methods.ariadne.method)."
        )


register(AriadneRebase())
register_alias("direct_residual", "ariadne")
