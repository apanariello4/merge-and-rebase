"""Opt-in ``component_target="output_total"`` fit (vision only): a per-layer target in the projection-output space.

For target block ``j`` paired with source block ``i = pairing[j]`` (weight 1) and ``c`` in
{``attn.out_proj``, ``mlp.c_proj``}, ``A_c`` is the projection module's own output (before LayerScale and the residual
add). The regression target of layer ``c`` is

    ``D_j^c = (A_c(source FT, block i) - A_c(source base, block i)) Q_j^c``

(token-aligned with ``_aligned``), where ``Q_j^c`` is the centered rectangular Procrustes map fitted from the source-base
``A_c`` at block ``i`` onto the target-base ``A_c`` at block ``j`` (base models only, the same polar factor as the
block-boundary ``Q_j``). The fit is the shared `ResidualSufficientStatistics` / `_finalize_independent_component` ridge
(empirical-Bayes penalty and unpenalised intercept as configured) on the layer's input ``H`` with output map
``L = I``: the target lives in the layer's own output space, so no LayerScale factor enters (it scales the write into
the residual stream, after ``A_c``).

Two depth modes share Pass A (`prepare_output_total_streaming`):

* ``depth_fit="independent"``: every layer is fitted against the pristine target (`fit_output_total_independent_streaming`).
* ``depth_fit="sequential"``: layers are fitted in forward order ``O_0, D_0, O_1, D_1, ...`` (only the configured ones);
  layer ``c`` of block ``j`` sees the target carrying every previously fitted layer, with ``U_j^c = A_c(current) -
  A_c(pristine)`` and residual target ``E_j^c = D_j^c - U_j^c`` (`fit_output_total_sequential_streaming`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

import torch

from ...discrete_layer_match import DiscreteLayerPairing
from .alignment import _procrustes_from_cross, procrustes_rank_diagnostics
from .capture import iter_capture_tokens, paired_calibration
from .config import DirectResidualConfig, order_components
from .fit import ResidualSufficientStatistics, _finalize_independent_component
from .layouts import COMPONENT_INPUT_KIND, COMPONENT_OUTPUT_KIND, _aligned, _layout_for
from .streaming import _check_boundary_fingerprint, _StreamingCrossCovariance

Tensor = torch.Tensor


def _require_vision(family_adapter) -> None:
    if family_adapter is not None:
        # A decoder port is a layout entry (COMPONENT_OUTPUT_KIND / COMPONENT_INPUT_KIND for the decoder projections).
        raise NotImplementedError("component_target='output_total' is vision-only")


def _check_components(components) -> tuple[str, ...]:
    components = order_components(components)
    if not components or any(c not in COMPONENT_OUTPUT_KIND for c in components):
        raise ValueError(f"component_target='output_total' supports only {sorted(COMPONENT_OUTPUT_KIND)}")
    return components


def prepare_output_total_streaming(
    source_base_model,
    target_base_model,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    *,
    num_batches: int,
    seed: int | None,
    device,
    components,
    family_adapter=None,
) -> dict[str, Any]:
    """Pass A of ``output_total``: one centered Procrustes map ``Q_j^c`` per (target position, component).

    Same calibration call (``paired_calibration``, same sample IDs) and ``_aligned`` helper as the block-boundary Pass A;
    the cross-covariance is accumulated batch by batch (Chan) between the source-base ``A_c`` at block ``pairing[j]`` and
    the target-base ``A_c`` at block ``j``. ``component_fingerprints[(batch, j, c)]`` is the ``(sum, sumsq)`` of the
    target-base ``A_c``, which Pass B checks to detect a target mutated between the passes.
    """
    _require_vision(family_adapter)
    components = _check_components(components)
    if pairing.target_depth < 1 or pairing.source_depth < 1:
        raise ValueError("pairing depths must be positive")
    source_batches, target_batches, metadata = paired_calibration(
        source_loader, target_loader, num_batches=num_batches, seed=seed, family_adapter=family_adapter
    )
    distinct_source_indices = sorted(set(pairing.pairing))
    src_requests = {f"{i}.{c}": (i, COMPONENT_OUTPUT_KIND[c]) for i in distinct_source_indices for c in components}
    tgt_requests = {f"{j}.{c}": (j, COMPONENT_OUTPUT_KIND[c]) for j in range(pairing.target_depth) for c in components}
    accumulators = {
        (j, c): _StreamingCrossCovariance(device=device) for j in range(pairing.target_depth) for c in components
    }
    fingerprints: dict[tuple[int, int, str], tuple[float, float]] = {}
    gens = (
        iter_capture_tokens(
            source_base_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
        ),
        iter_capture_tokens(
            target_base_model, target_batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
        ),
    )
    with contextlib.ExitStack() as stack:
        for gen in gens:
            stack.enter_context(contextlib.closing(gen))
        for k, (src, tgt) in enumerate(zip(*gens, strict=True)):
            for j in range(pairing.target_depth):
                i = pairing.pairing[j]
                for c in components:
                    t = tgt[f"{j}.{c}"]
                    x = _aligned([src[f"{i}.{c}"]], [t])[0]
                    accumulators[(j, c)].update(
                        x.reshape(-1, x.shape[-1]).double(), t.reshape(-1, t.shape[-1]).double()
                    )
                    t64 = t.double()
                    fingerprints[(k, j, c)] = (float(t64.sum().item()), float((t64**2).sum().item()))
    q_by_component: dict[tuple[int, str], Tensor] = {}
    diagnostics: dict[tuple[int, str], dict[str, Any]] = {}
    for pair, acc in accumulators.items():
        q64, rank = _procrustes_from_cross(acc.cross(), return_rank=True)
        q_by_component[pair] = q64.cpu().float()
        diagnostics[pair] = {
            "procrustes_source": "activation",
            **procrustes_rank_diagnostics(rank, q64.shape[0], q64.shape[1], polar_derived=True),
        }
    return {
        "q_component_by_position": q_by_component,
        "component_fingerprints": fingerprints,
        "procrustes_source": "activation",
        "procrustes_diagnostics": {},  # position-keyed rows do not exist here; see component_procrustes_diagnostics
        "component_procrustes_diagnostics": diagnostics,
        "source_batches": source_batches,
        "target_batches": target_batches,
        "calibration": metadata,
        "distinct_source_indices": distinct_source_indices,
    }


def _finish_rows(rows, prepared, pos, component) -> None:
    for row in rows:
        row["source_position"] = row.pop("source_coordinate")
        row["component_target"] = "output_total"
        row.update(prepared["component_procrustes_diagnostics"][(pos, component)])


def _unique_models(*models):
    unique: dict[int, Any] = {}
    for m in models:
        unique.setdefault(id(m), m)
    return unique, {mid: (next(m.parameters()).device, m.training) for mid, m in unique.items()}


@torch.no_grad()
def fit_output_total_independent_streaming(
    target_model,
    target_base_state: Mapping[str, Tensor],
    source_base_model,
    source_ft_model,
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
    stats_factory=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Pass B of ``output_total`` with ``depth_fit="independent"``: every (position, component) against the pristine target.

    Per batch, source-base and source-FT ``A_c`` at block ``pairing[j]`` and the target's layer input ``H`` and output
    ``A_c`` (the latter only for the pass-A fingerprint check) are captured in lockstep; ``D_j^c = (f - b) Q_j^c`` is
    formed on the fly and one `ResidualSufficientStatistics` per (position, component) accumulates ``(H, D_j^c)`` with
    ``L = I``. Position chunks (``streaming_position_chunk``) bound host RAM as in the block-boundary fit. Rows carry the
    block-boundary schema with ``desired_norm`` / ``relative_residual_*`` against the layer's own target, plus
    ``component_target="output_total"``.
    """
    _require_vision(family_adapter)
    components = _check_components(config.components)
    batches, source_batches = prepared["target_batches"], prepared["source_batches"]
    q_by = prepared["q_component_by_position"]
    fingerprints = prepared["component_fingerprints"]
    positions = list(range(pairing.target_depth))
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    solver_config = replace(config, cascade_order="independent") if config.cascade_order != "independent" else config
    shim = _layout_for(family_adapter)
    chunk_size = config.streaming_position_chunk or len(positions)
    chunks = [positions[s : s + chunk_size] for s in range(0, len(positions), chunk_size)]
    unique_models, originals = _unique_models(target_model, source_base_model, source_ft_model)
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        for m in unique_models.values():
            m.to(device).eval()
        for chunk in chunks:
            needed = sorted({pairing.pairing[pos] for pos in chunk})
            src_requests = {f"{i}.{c}": (i, COMPONENT_OUTPUT_KIND[c]) for i in needed for c in components}
            tgt_requests: dict[str, tuple[int, str]] = {}
            for pos in chunk:
                for c in components:
                    tgt_requests[f"{pos}.{c}.h"] = (pos, COMPONENT_INPUT_KIND[c])
                    tgt_requests[f"{pos}.{c}.a"] = (pos, COMPONENT_OUTPUT_KIND[c])
            stats_by, desired_sq, eff_out = {}, {}, {}
            for pos in chunk:
                for c in components:
                    pair = (pos, c)
                    width = int(current_state[shim.component_key(pos, c, prefixed=True)].shape[0])
                    eff_out[pair] = torch.eye(width, dtype=torch.float32)
                    stats_by[pair] = (
                        ResidualSufficientStatistics(device=device)
                        if stats_factory is None
                        else stats_factory(pos, c, device)
                    )
                    desired_sq[pair] = 0.0
            gens = (
                iter_capture_tokens(
                    source_base_model,
                    source_batches,
                    src_requests,
                    device,
                    family_adapter=family_adapter,
                    store_device=device,
                ),
                iter_capture_tokens(
                    source_ft_model,
                    source_batches,
                    src_requests,
                    device,
                    family_adapter=family_adapter,
                    store_device=device,
                ),
                iter_capture_tokens(
                    target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
                ),
            )
            with contextlib.ExitStack() as stack:
                for gen in gens:
                    stack.enter_context(contextlib.closing(gen))
                for k, (sb, sf, tgt) in enumerate(zip(*gens, strict=True)):
                    for pos in chunk:
                        i = pairing.pairing[pos]
                        for c in components:
                            a_cpu = tgt[f"{pos}.{c}.a"].cpu()
                            _check_boundary_fingerprint(a_cpu, fingerprints[(k, pos, c)], config, pos, k)
                            b = _aligned([sb[f"{i}.{c}"].cpu()], [a_cpu])[0]
                            f = _aligned([sf[f"{i}.{c}"].cpu()], [a_cpu])[0]
                            desired = (f - b) @ q_by[(pos, c)]
                            h = tgt[f"{pos}.{c}.h"].cpu()
                            desired_sq[(pos, c)] += float((desired.double() ** 2).sum().item())
                            stats_by[(pos, c)].update(
                                h.reshape(-1, h.shape[-1]),
                                desired.reshape(-1, desired.shape[-1]),
                                None,
                                eff_out[(pos, c)],
                            )
            for pos in chunk:
                position_corrections: dict[str, Tensor] = {}
                for c in components:
                    block_rows: list[dict[str, Any]] = []
                    _finalize_independent_component(
                        stats_by[(pos, c)],
                        desired_sq[(pos, c)],
                        0.0,
                        pos,
                        c,
                        shim.component_key(pos, c, prefixed=True),
                        current_state,
                        eff_out[(pos, c)],
                        solver_config,
                        source_coordinates,
                        position_corrections,
                        block_rows,
                        h_batches=None,
                    )
                    _finish_rows(block_rows, prepared, pos, c)
                    diagnostics.extend(block_rows)
                target_corrections.update(position_corrections)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    return target_corrections, diagnostics


@torch.no_grad()
def fit_output_total_sequential_streaming(
    target_model,
    target_base_state: Mapping[str, Tensor],
    source_base_model,
    source_ft_model,
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Pass B of ``output_total`` with ``depth_fit="sequential"`` (layers in forward order, one streaming pass each).

    Layers are visited ``O_0, D_0, O_1, D_1, ...`` (only the configured components). Before layer ``c`` of block ``j`` the
    target holds the base plus every already fitted layer (weights and bias). Per batch (source base, source FT, current
    target and a pristine copy swept in lockstep) it forms ``D_j^c`` as in the independent fit, the upstream effect
    ``U_j^c = A_c(current) - A_c(pristine)`` at the layer's own output and the target ``E_j^c = D_j^c - U_j^c``, and
    accumulates ``(H_current, E_j^c)`` with ``L = I``. The fitted layer is mounted before the next one, so ``D``'s upstream
    effect includes the ``O_j`` correction of the same block. The first layer fitted (block 0, first component) has
    ``U = 0`` and equals the independent fit. Rows add ``depth_fit="sequential"`` and ``upstream_effect_norm``
    (``sqrt(sum ||U||^2)``); ``relative_residual_after`` is against the layer's own target ``E``.
    """
    _require_vision(family_adapter)
    components = _check_components(config.components)
    batches, source_batches = prepared["target_batches"], prepared["source_batches"]
    q_by = prepared["q_component_by_position"]
    fingerprints = prepared["component_fingerprints"]
    positions = list(range(pairing.target_depth))
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    shim = _layout_for(family_adapter)
    unique_models, originals = _unique_models(target_model, source_base_model, source_ft_model)
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    pristine_model = None
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        pristine_model = deepcopy(target_model)
        pristine_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        sb_model = pristine_model if source_base_model is target_model else source_base_model
        sf_model = pristine_model if source_ft_model is target_model else source_ft_model
        for m in {id(x): x for x in (target_model, sb_model, sf_model, pristine_model)}.values():
            m.to(device).eval()
        for pos in positions:
            i = pairing.pairing[pos]
            for c in components:
                key = shim.component_key(pos, c, prefixed=True)
                width = int(current_state[key].shape[0])
                eff_out = torch.eye(width, dtype=torch.float32)
                q = q_by[(pos, c)]
                src_requests = {f"{i}.{c}": (i, COMPONENT_OUTPUT_KIND[c])}
                a_key, h_key = f"{pos}.{c}.a", f"{pos}.{c}.h"
                tgt_requests = {a_key: (pos, COMPONENT_OUTPUT_KIND[c]), h_key: (pos, COMPONENT_INPUT_KIND[c])}
                ref_requests = {a_key: (pos, COMPONENT_OUTPUT_KIND[c])}
                gens = (
                    iter_capture_tokens(
                        sb_model,
                        source_batches,
                        src_requests,
                        device,
                        family_adapter=family_adapter,
                        store_device=device,
                    ),
                    iter_capture_tokens(
                        sf_model,
                        source_batches,
                        src_requests,
                        device,
                        family_adapter=family_adapter,
                        store_device=device,
                    ),
                    iter_capture_tokens(
                        target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
                    ),
                    iter_capture_tokens(
                        pristine_model,
                        batches,
                        ref_requests,
                        device,
                        family_adapter=family_adapter,
                        store_device=device,
                    ),
                )
                stats = ResidualSufficientStatistics(device=device)
                desired_sq = upstream_sq = 0.0
                with contextlib.ExitStack() as stack:
                    for gen in gens:
                        stack.enter_context(contextlib.closing(gen))
                    for k, (sb, sf, tgt, ref) in enumerate(zip(*gens, strict=True)):
                        ref_cpu = ref[a_key].cpu()
                        _check_boundary_fingerprint(ref_cpu, fingerprints[(k, pos, c)], config, pos, k)
                        a_cpu = tgt[a_key].cpu()
                        b = _aligned([sb[f"{i}.{c}"].cpu()], [a_cpu])[0]
                        f = _aligned([sf[f"{i}.{c}"].cpu()], [a_cpu])[0]
                        desired = (f - b) @ q
                        upstream = a_cpu - ref_cpu
                        error = desired - upstream
                        desired_sq += float((desired.double() ** 2).sum().item())
                        upstream_sq += float((upstream.double() ** 2).sum().item())
                        h = tgt[h_key].cpu()
                        stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, eff_out)
                position_corrections: dict[str, Tensor] = {}
                block_rows: list[dict[str, Any]] = []
                # effect_sq=0.0: the upstream effect is part of the target (recorded as upstream_effect_norm).
                _finalize_independent_component(
                    stats,
                    desired_sq,
                    0.0,
                    pos,
                    c,
                    key,
                    current_state,
                    eff_out,
                    config,
                    source_coordinates,
                    position_corrections,
                    block_rows,
                    h_batches=None,
                )
                target_corrections.update(position_corrections)
                _finish_rows(block_rows, prepared, pos, c)
                for row in block_rows:
                    row["depth_fit"] = "sequential"
                    row["upstream_effect_norm"] = upstream_sq**0.5
                    diagnostics.append(row)
                # Mount this layer (weight and bias, as updated in current_state) before the next one.
                target_model.load_state_dict({name: current_state[name] for name in position_corrections}, strict=False)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
        del pristine_model
    return target_corrections, diagnostics
