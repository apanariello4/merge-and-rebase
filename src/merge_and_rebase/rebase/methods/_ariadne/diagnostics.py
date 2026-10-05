"""Post-hoc diagnostics: realization measurement, task-vector statistics and fidelity holdout."""

from __future__ import annotations

import contextlib
import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import torch

from ...discrete_layer_match import DiscreteLayerPairing
from .capture import capture_tokens, iter_capture_tokens, paired_calibration
from .config import CANONICAL_COMPONENT_ORDER, DirectResidualConfig, order_components
from .fit import _task_vector_sha256
from .layouts import (
    _PACKED_QKV_SLICE,
    COMPONENT_INPUT_KIND,
    _aligned,
    _component_effective_out,
    _family_bias_key,
    _layout_for,
    _rows,
)

Tensor = torch.Tensor


def compute_direct_residual_task_vector_stats(
    target_corrections: Mapping[str, torch.Tensor],
    target_base_state: Mapping[str, torch.Tensor],
    positions: list[int],
    components: tuple[str, ...] = CANONICAL_COMPONENT_ORDER,
    *,
    family_adapter=None,
) -> dict[str, Any]:
    """Analysis-only task-vector stats; nothing is refitted and no forward pass runs.

    ``target_corrections`` is the unscaled (unit-strength) task vector from ``fit_direct_residual``.
    ``n_modified_parameters`` counts touched numel per ``(position, component)`` pair present in
    ``target_corrections`` (a packed q/k/v correction that wrote one row-third counts ``d * d_in`` (+ ``d`` bias),
    not 3x); packed components own disjoint row slices (``_PACKED_QKV_SLICE``), so nothing double-counts.
    ``n_modified_tensors`` counts physical tensors (dict keys). ``tau_norm_over_touched_base`` divides by the
    Frobenius norm of the base at exactly the touched slices (row-aware for packed q/k/v);
    ``tau_norm_over_all_base`` by that of every floating-point tensor in the base state dict.
    """
    shim = _layout_for(family_adapter)
    tau_norm_sq = 0.0
    for value in target_corrections.values():
        tau_norm_sq += float((value.detach().double() ** 2).sum().item())
    tau_norm = tau_norm_sq**0.5

    all_base_sq = 0.0
    for value in target_base_state.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            all_base_sq += float((value.detach().double() ** 2).sum().item())
    all_base_norm = all_base_sq**0.5

    touched_base_sq = 0.0
    # WARNING: q/k/v share ONE physical key (in_proj_weight/bias), so key presence cannot tell which of q/k/v were
    # fit. Callers MUST pass exactly the fitted family list (e.g. `order_components(config.components)`), never a
    # broader default, or rows are over-counted.
    n_modified_parameters = 0
    for pos in positions:
        for component in components:
            key = shim.component_key(pos, component, prefixed=True)
            if key not in target_base_state:
                continue
            weight = target_base_state[key]
            bias_key = _family_bias_key(key)
            if component in _PACKED_QKV_SLICE:
                if key not in target_corrections and (bias_key is None or bias_key not in target_corrections):
                    continue
                d = weight.shape[0] // 3
                row_slice = slice(_PACKED_QKV_SLICE[component] * d, (_PACKED_QKV_SLICE[component] + 1) * d)
                w_slice = weight[row_slice]
                n_modified_parameters += w_slice.numel()
                touched_base_sq += float((w_slice.detach().double() ** 2).sum().item())
                if bias_key is not None and bias_key in target_base_state:
                    b_slice = target_base_state[bias_key][row_slice]
                    n_modified_parameters += b_slice.numel()
                    touched_base_sq += float((b_slice.detach().double() ** 2).sum().item())
            else:
                if key not in target_corrections:
                    continue
                n_modified_parameters += weight.numel()
                touched_base_sq += float((weight.detach().double() ** 2).sum().item())
                if bias_key is not None and bias_key in target_base_state:
                    n_modified_parameters += target_base_state[bias_key].numel()
                    touched_base_sq += float((target_base_state[bias_key].detach().double() ** 2).sum().item())
    touched_base_norm = touched_base_sq**0.5

    return {
        "n_modified_tensors": len(target_corrections),
        "n_modified_parameters": int(n_modified_parameters),
        "tau_norm": tau_norm,
        "tau_norm_over_touched_base": tau_norm / (touched_base_norm + 1e-12),
        "tau_norm_over_all_base": tau_norm / (all_base_norm + 1e-12),
        "tau_sha256": _task_vector_sha256(target_corrections),
    }


def _family_delta_state(
    shim,
    component: str,
    positions: list[int],
    target_corrections: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The subset of ``target_corrections`` belonging to one component family,
    zero-padded to full tensor shape (packed q/k/v: zero outside that
    component's own row slice)."""
    out: dict[str, torch.Tensor] = {}
    for pos in positions:
        key = shim.component_key(pos, component, prefixed=True)
        if key not in target_corrections:
            continue
        full = target_corrections[key]
        bias_key = _family_bias_key(key)
        if component in _PACKED_QKV_SLICE:
            d = base_state[key].shape[0] // 3
            row_slice = slice(_PACKED_QKV_SLICE[component] * d, (_PACKED_QKV_SLICE[component] + 1) * d)
            zeroed = torch.zeros_like(base_state[key])
            zeroed[row_slice] = full[row_slice]
            out[key] = zeroed
            if bias_key is not None and bias_key in target_corrections:
                zb = torch.zeros_like(base_state[bias_key])
                zb[row_slice] = target_corrections[bias_key][row_slice]
                out[bias_key] = zb
        else:
            out[key] = full.clone()
            if bias_key is not None and bias_key in target_corrections:
                out[bias_key] = target_corrections[bias_key].clone()
    return out


@torch.no_grad()
def measure_direct_residual_realization(
    target_model,
    target_base_state: Mapping[str, torch.Tensor],
    target_corrections: Mapping[str, torch.Tensor],
    positions: list[int],
    batches: list,
    target_outputs_by_position: Mapping[int, list],
    desired_by_position: Mapping[int, list],
    *,
    device,
    components: tuple[str, ...] = CANONICAL_COMPONENT_ORDER,
    family_adapter=None,
) -> dict[int, dict[str, Any]]:
    """Measure how well the unit-strength task vector ``tau`` realizes each position's desired block-boundary
    effect ``D_j`` on the FULL (nonlinear) target model, unlike the fit's own linear-prediction diagnostics
    (``_realization_diagnostic_fields``).

    For the ``joint`` variant (all of ``tau``) and one variant per component family present in ``tau`` (row-sliced,
    zero-elsewhere for packed q/k/v), mounts ``target_base_state + tau_variant``, captures block-boundary outputs at
    ``positions`` over ``batches`` and compares ``delta_j = T_j^variant - T_j^0`` against ``D_j`` (the same
    ``compute_desired_effects`` target the fit uses). Per position returns:
      * ``block_realized_target_error``: ``||delta_j^joint - D_j||_F / (||D_j||_F + eps)``.
      * ``joint_delta_norm_over_desired``: ``||delta_j^joint||_F / (||D_j||_F + eps)``.
      * ``component_interaction_error``: ``||delta_j^joint - sum_c delta_j^(c)||_F / (||delta_j^joint||_F + eps)``,
        ``None`` with a single family.
      * ``per_family_delta_norm_over_desired``: ``{component: ||delta_j^(c)||_F / (||D_j||_F + eps)}``.

    WARNING on ``components``: q/k/v share ONE state-dict key (``in_proj_weight``/``in_proj_bias``), so key presence
    cannot tell which were fit (a v-only run would falsely report q/k present). Callers MUST pass exactly the fitted
    family list (e.g. ``order_components(config.components)``), never the broad ``CANONICAL_COMPONENT_ORDER``
    default -- see ``vision_rebase._run_direct_residual_fit``.

    Memory: variant-outer loop; one variant's banks at a time plus accumulators (joint delta bank, running sum of
    family deltas, scalar squared norms). The entry state of ``target_model`` is restored in a ``finally`` and
    checked by state-dict hash: no residue may be left on ``target_model``.
    """
    shim = _layout_for(family_adapter)
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    entry_hash = _task_vector_sha256(entry_state)
    base_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}

    def mount(delta: Mapping[str, torch.Tensor]) -> None:
        state = dict(base_state)
        for key, value in delta.items():
            state[key] = state[key] + value.to(state[key])
        target_model.load_state_dict(state, strict=True)

    def capture_all(delta: Mapping[str, torch.Tensor]) -> dict[int, list]:
        mount(delta)
        requests = {str(pos): (pos, "boundary") for pos in positions}
        raw = capture_tokens(target_model, batches, requests, device, family_adapter=family_adapter)
        return {pos: raw[str(pos)] for pos in positions}

    try:
        joint_banks = capture_all(target_corrections)
        joint_delta = {
            pos: [v - t0 for v, t0 in zip(joint_banks[pos], target_outputs_by_position[pos], strict=True)]
            for pos in positions
        }
        del joint_banks
        joint_norm_sq = {pos: sum(float((d.double() ** 2).sum().item()) for d in joint_delta[pos]) for pos in positions}

        present_families = [
            c
            for c in components
            if any(shim.component_key(pos, c, prefixed=True) in target_corrections for pos in positions)
        ]

        running_sum = {pos: [torch.zeros_like(d) for d in joint_delta[pos]] for pos in positions}
        family_norm_sq: dict[str, dict[int, float]] = {c: {} for c in present_families}
        for component in present_families:
            delta = _family_delta_state(shim, component, positions, target_corrections, base_state)
            banks = capture_all(delta)
            for pos in positions:
                deltas_c = [v - t0 for v, t0 in zip(banks[pos], target_outputs_by_position[pos], strict=True)]
                family_norm_sq[component][pos] = sum(float((d.double() ** 2).sum().item()) for d in deltas_c)
                for idx, d in enumerate(deltas_c):
                    running_sum[pos][idx] = running_sum[pos][idx] + d
            del banks, delta

        results: dict[int, dict[str, Any]] = {}
        for pos in positions:
            d_batches = desired_by_position[pos]
            d_norm_sq = sum(float((d.double() ** 2).sum().item()) for d in d_batches)
            d_norm = d_norm_sq**0.5
            joint_norm = joint_norm_sq[pos] ** 0.5
            err_sq = sum(
                float(((jd - dd).double() ** 2).sum().item())
                for jd, dd in zip(joint_delta[pos], d_batches, strict=True)
            )
            row: dict[str, Any] = {
                "position": pos,
                "desired_norm": d_norm,
                "joint_delta_norm": joint_norm,
                "block_realized_target_error": (err_sq**0.5) / (d_norm + 1e-12),
                "joint_delta_norm_over_desired": joint_norm / (d_norm + 1e-12),
                "per_family_delta_norm_over_desired": {
                    c: (family_norm_sq[c][pos] ** 0.5) / (d_norm + 1e-12) for c in present_families
                },
            }
            if len(present_families) > 1:
                interaction_sq = sum(
                    float(((jd - rs).double() ** 2).sum().item())
                    for jd, rs in zip(joint_delta[pos], running_sum[pos], strict=True)
                )
                row["component_interaction_error"] = (interaction_sq**0.5) / (joint_norm + 1e-12)
            else:
                row["component_interaction_error"] = None
            results[pos] = row
        return results
    finally:
        target_model.load_state_dict(entry_state, strict=True)
        exit_hash = _task_vector_sha256({k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()})
        if exit_hash != entry_hash:
            raise RuntimeError(
                "measure_direct_residual_realization failed to restore the target model's entry state exactly"
            )


@torch.no_grad()
def measure_direct_residual_realization_streaming(
    target_model,
    target_base_state: Mapping[str, torch.Tensor],
    target_corrections: Mapping[str, torch.Tensor],
    positions: list[int],
    target_batches: list,
    desired_fn,
    source_iters_fn,
    *,
    device,
    components: tuple[str, ...] = CANONICAL_COMPONENT_ORDER,
    family_adapter=None,
) -> dict[int, dict[str, Any]]:
    """Streaming counterpart of `measure_direct_residual_realization` (same row schema).

    Runs ONE lockstep sweep over the calibration batches: the pristine target, one deep copy per variant (joint
    ``tau`` plus one per family present, mounted as ``base + tau_variant``) and the generators from
    ``source_iters_fn()``. Per batch ``desired_fn(k, source_values, t0)`` recomputes ``{pos: D_j}`` and norms
    accumulate as per-batch sums of squares, so results equal the resident path up to floating-point summation
    order (``D_j`` differs only via the Chan-accumulated Procrustes map). The entry state of ``target_model`` is
    restored and hash-checked.
    """
    shim = _layout_for(family_adapter)
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    entry_hash = _task_vector_sha256(entry_state)
    base_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}

    def mounted_copy(delta: Mapping[str, torch.Tensor]):
        state = dict(base_state)
        for key, value in delta.items():
            state[key] = state[key] + value.to(state[key])
        model = copy.deepcopy(target_model)
        model.load_state_dict(state, strict=True)
        return model

    present_families = [
        c
        for c in components
        if any(shim.component_key(pos, c, prefixed=True) in target_corrections for pos in positions)
    ]
    requests = {str(pos): (pos, "boundary") for pos in positions}
    variants = {"joint": target_corrections}
    for component in present_families:
        variants[component] = _family_delta_state(shim, component, positions, target_corrections, base_state)
    copies = {name: mounted_copy(delta) for name, delta in variants.items()}

    d_sq = {pos: 0.0 for pos in positions}
    joint_sq = {pos: 0.0 for pos in positions}
    err_sq = {pos: 0.0 for pos in positions}
    fam_sq = {c: {pos: 0.0 for pos in positions} for c in present_families}
    inter_sq = {pos: 0.0 for pos in positions}
    try:
        target_model.load_state_dict(base_state, strict=True)
        gens = {
            "__t0__": iter_capture_tokens(target_model, target_batches, requests, device, family_adapter=family_adapter)
        }
        for name, model in copies.items():
            gens[name] = iter_capture_tokens(model, target_batches, requests, device, family_adapter=family_adapter)
        source_gens = source_iters_fn()
        names = list(gens)
        with contextlib.ExitStack() as stack:
            for gen in list(gens.values()) + list(source_gens.values()):
                stack.enter_context(contextlib.closing(gen))
            zipped = zip(*(gens[n] for n in names), *source_gens.values(), strict=True)
            for k, values in enumerate(zipped):
                by_name = dict(zip(names, values[: len(names)], strict=True))
                source_values = dict(zip(source_gens, values[len(names) :], strict=True))
                t0 = by_name["__t0__"]
                desired = desired_fn(k, source_values, t0)
                for pos in positions:
                    base_out = t0[str(pos)]
                    d = desired[pos]
                    jd = by_name["joint"][str(pos)] - base_out
                    d_sq[pos] += float((d.double() ** 2).sum().item())
                    joint_sq[pos] += float((jd.double() ** 2).sum().item())
                    err_sq[pos] += float(((jd - d).double() ** 2).sum().item())
                    running = torch.zeros_like(jd)
                    for component in present_families:
                        dc = by_name[component][str(pos)] - base_out
                        fam_sq[component][pos] += float((dc.double() ** 2).sum().item())
                        running = running + dc
                    if len(present_families) > 1:
                        inter_sq[pos] += float(((jd - running).double() ** 2).sum().item())
        results: dict[int, dict[str, Any]] = {}
        for pos in positions:
            d_norm = d_sq[pos] ** 0.5
            joint_norm = joint_sq[pos] ** 0.5
            row: dict[str, Any] = {
                "position": pos,
                "desired_norm": d_norm,
                "joint_delta_norm": joint_norm,
                "block_realized_target_error": (err_sq[pos] ** 0.5) / (d_norm + 1e-12),
                "joint_delta_norm_over_desired": joint_norm / (d_norm + 1e-12),
                "per_family_delta_norm_over_desired": {
                    c: (fam_sq[c][pos] ** 0.5) / (d_norm + 1e-12) for c in present_families
                },
                "component_interaction_error": (
                    (inter_sq[pos] ** 0.5) / (joint_norm + 1e-12) if len(present_families) > 1 else None
                ),
            }
            results[pos] = row
        return results
    finally:
        del copies
        target_model.load_state_dict(entry_state, strict=True)
        exit_hash = _task_vector_sha256({k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()})
        if exit_hash != entry_hash:
            raise RuntimeError(
                "measure_direct_residual_realization_streaming failed to restore the target model's entry state exactly"
            )


def draw_fidelity_holdout_calibration(
    source_loader,
    target_loader,
    *,
    num_batches: int,
    holdout_batches: int,
    seed: int | None,
    family_adapter=None,
) -> tuple[list, list, dict[str, Any]]:
    """Draw a held-out batch set disjoint from the ``num_batches`` calibration set tau was fit on.

    Both sets are slices of the SAME seeded ``paired_calibration`` permutation, so a non-``None`` ``seed`` is
    required: ``paired_calibration(num_batches + holdout_batches)`` reproduces the identical leading ``num_batches``
    slice and the next ``holdout_batches`` slice is disjoint by construction (still verified from the recorded
    sample indices). Returns ``(holdout_source_batches, holdout_target_batches, holdout_metadata)``; the metadata
    carries both slices' sample-index sha256 fingerprints.
    """
    if seed is None:
        raise ValueError("fidelity_holdout requires a deterministic (non-None) calibration seed")
    total_batches = int(num_batches) + int(holdout_batches)
    all_source, all_target, metadata = paired_calibration(
        source_loader, target_loader, num_batches=total_batches, seed=seed, family_adapter=family_adapter
    )
    if len(all_source) < total_batches:
        raise ValueError(
            f"fidelity_holdout_batches={holdout_batches} requires {total_batches} batches "
            f"but only {len(all_source)} are available in the calibration split"
        )
    bs = metadata["batch_size"]
    calibration_indices = metadata["indices"][: num_batches * bs]
    holdout_indices = metadata["indices"][num_batches * bs : total_batches * bs]
    if set(calibration_indices) & set(holdout_indices):
        raise RuntimeError(
            "fidelity_holdout: calibration and holdout sample indices are not disjoint "
            "(this should be unreachable -- paired_calibration's permutation slices overlapped)"
        )
    holdout_metadata = {
        "holdout_batches": int(holdout_batches),
        "actual_holdout_batches": len(all_source) - num_batches,
        "batch_size": bs,
        "sampling_seed": seed,
        "dataset_identity": metadata["dataset_identity"],
        "calibration_indices_sha256": hashlib.sha256(repr(calibration_indices).encode()).hexdigest(),
        "holdout_indices_sha256": hashlib.sha256(repr(holdout_indices).encode()).hexdigest(),
        "calibration_holdout_disjoint": True,
    }
    return all_source[num_batches:total_batches], all_target[num_batches:total_batches], holdout_metadata


def compute_fidelity_holdout_diagnostics(
    source_base_model,
    source_ft_model,
    target_model,
    target_base_state: Mapping[str, Tensor],
    target_corrections: Mapping[str, Tensor],
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    q_by_position: Mapping[int, Tensor],
    mu_s_by_position: Mapping[int, Tensor] | None,
    mu_t_by_position: Mapping[int, Tensor] | None,
    device,
    family_adapter=None,
) -> dict[str, Any]:
    """``DirectResidualConfig.fidelity_holdout`` diagnostic: analysis-only, never read by any fit and never mutates
    ``target_corrections``.

    For BOTH the calibration split and a disjoint held-out split (``draw_fidelity_holdout_calibration``), per target
    position ``j`` (and, for ``e_local``, per fitted component):
      * ``e_local``  = ``||(H_j Delta_W_j + 1 beta_j^T) @ effective_out_j - D_j||_F / ||D_j||_F``, the component's
        own local linear-fit residual; ``H_j`` is the base target's component input at block ``j``
        (``COMPONENT_INPUT_KIND``). ``D_j`` is recomputed on this split with the SAME fitted ``Q_j`` (and, for
        ``residual_target='transported_endpoint'``, the same ``mu_s``/``mu_t``) from ``q_by_position``/
        ``mu_s_by_position``/``mu_t_by_position`` -- never refit here.
      * ``e_mounted`` = ``block_realized_target_error`` from `measure_direct_residual_realization` (mounts
        ``target_corrections`` at unit strength, alpha=1, full nonlinear forward). With a single fitted component
        it equals that component's ``e_local`` algebraically.

    Every norm is reported alongside its ratio (``||D_j||`` and the raw numerator).
    """
    if not config.fidelity_holdout:
        raise ValueError("compute_fidelity_holdout_diagnostics called with fidelity_holdout=False")
    positions = list(range(pairing.target_depth))
    components = order_components(config.components)
    shim = _layout_for(family_adapter)
    distinct_source_indices = sorted(set(pairing.pairing))

    holdout_source, holdout_target, holdout_meta = draw_fidelity_holdout_calibration(
        source_loader,
        target_loader,
        num_batches=config.num_batches,
        holdout_batches=config.fidelity_holdout_batches,
        seed=config.seed,
        family_adapter=family_adapter,
    )
    calibration_source, calibration_target, _calib_meta = paired_calibration(
        source_loader, target_loader, num_batches=config.num_batches, seed=config.seed, family_adapter=family_adapter
    )

    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    out: dict[str, Any] = {
        "holdout_indices_sha256": holdout_meta["holdout_indices_sha256"],
        "calibration_indices_sha256": holdout_meta["calibration_indices_sha256"],
        "calibration_holdout_disjoint": holdout_meta["calibration_holdout_disjoint"],
        "holdout_batches": holdout_meta["actual_holdout_batches"],
        "splits": {},
    }
    try:
        target_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        for split_name, (src_batches, tgt_batches) in (
            ("calibration", (calibration_source, calibration_target)),
            ("holdout", (holdout_source, holdout_target)),
        ):
            src_requests = {str(i): (i, "boundary") for i in distinct_source_indices}
            tgt_requests: dict[str, tuple[int, str]] = {str(j): (j, "boundary") for j in positions}
            for component in components:
                for j in positions:
                    tgt_requests[f"{j}.{component}.h"] = (j, COMPONENT_INPUT_KIND[component])
            source_base_raw = capture_tokens(
                source_base_model, src_batches, src_requests, device, family_adapter=family_adapter
            )
            source_ft_raw = capture_tokens(
                source_ft_model, src_batches, src_requests, device, family_adapter=family_adapter
            )
            target_raw = capture_tokens(target_model, tgt_batches, tgt_requests, device, family_adapter=family_adapter)

            desired: dict[int, list[Tensor]] = {}
            for j in positions:
                i = pairing.pairing[j]
                t = target_raw[str(j)]
                b = _aligned(source_base_raw[str(i)], t)
                f = _aligned(source_ft_raw[str(i)], t)
                q = q_by_position[j]
                if config.residual_target == "transported_delta":
                    desired[j] = [(fb - bb) @ q for bb, fb in zip(b, f, strict=True)]
                else:
                    mu_s = mu_s_by_position[j]
                    mu_t = mu_t_by_position[j]
                    desired[j] = [(fb - mu_s) @ q + mu_t - tb for fb, tb in zip(f, t, strict=True)]
            target_base_outputs_by_position = {j: target_raw[str(j)] for j in positions}

            realization = measure_direct_residual_realization(
                target_model,
                target_base_state,
                target_corrections,
                positions,
                tgt_batches,
                target_base_outputs_by_position,
                desired,
                device=device,
                components=components,
                family_adapter=family_adapter,
            )

            e_local_by_position: dict[int, dict[str, Any]] = {}
            e_mounted_by_position: dict[int, dict[str, Any]] = {}
            for j in positions:
                d_rows = _rows(desired[j]).double()
                d_norm = float(torch.linalg.norm(d_rows).item())
                mounted_ratio = float(realization[j]["block_realized_target_error"])
                e_mounted_by_position[j] = {
                    "e_mounted": mounted_ratio,
                    "desired_norm": d_norm,
                    "numerator": mounted_ratio * d_norm,
                }
                per_component: dict[str, Any] = {}
                for component in components:
                    key = shim.component_key(j, component, prefixed=True)
                    delta_w = target_corrections.get(key)
                    if delta_w is None:
                        continue
                    bias_key = _family_bias_key(key)
                    delta_b = target_corrections.get(bias_key) if bias_key is not None else None
                    h_rows = _rows(target_raw[f"{j}.{component}.h"]).double()
                    pred = h_rows @ delta_w.double().T
                    if delta_b is not None:
                        pred = pred + delta_b.double()
                    width = int(delta_w.shape[0])
                    effective_out = _component_effective_out(shim, target_model, j, component, width).double()
                    pred = pred @ effective_out
                    numerator = float(torch.linalg.norm(pred - d_rows).item())
                    per_component[component] = {
                        "e_local": numerator / d_norm if d_norm > 0 else 0.0,
                        "numerator": numerator,
                        "desired_norm": d_norm,
                    }
                e_local_by_position[j] = per_component

            out["splits"][split_name] = {
                "e_local_by_position": e_local_by_position,
                "e_mounted_by_position": e_mounted_by_position,
            }
    finally:
        target_model.load_state_dict(entry_state, strict=True)
    return out
