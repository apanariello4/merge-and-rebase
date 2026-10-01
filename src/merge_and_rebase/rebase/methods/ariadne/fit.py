"""Ariadne fits: ``fit_direct_residual`` and sequential source endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from ...discrete_layer_match import DiscreteLayerPairing
from .blockwise import _fit_block_boundary_backfit, _fit_block_boundary_joint
from .capture import capture_tokens
from .components import order_components
from .config import DirectResidualConfig
from .independent import _fit_all_positions_independent
from .layouts import _aligned, _rows
from .linalg import centered_rectangular_procrustes

Tensor = torch.Tensor


@torch.no_grad()
def fit_direct_residual(
    target_model,
    target_base_state: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Fit every target position's residual-writing components independently.

    A thin wrapper around
    `ariadne.independent._fit_all_positions_independent` (the shared
    independent-mode fast path also used by ``complete_residuals_direct``) --
    called once for every ``j in range(pairing.target_depth)`` at once, with
    ``source_coordinate = pairing.pairing[j]``, from a single shared target
    forward sweep rather than one sweep per position.

    Every call is forced to run with independent (no-cascade) semantics,
    regardless of ``config.cascade_order``: Direct Residual mounts nothing
    between fits, cross-position or intra-position, by construction (see the
    module docstring). ``cascade_order`` is accepted in `DirectResidualConfig`
    only for schema parity and is provably a no-op here -- see
    ``tests/test_direct_residual_cascade_order.py``.

    ``theta_j_corrected = theta_j_native + strength * correction_j`` is NOT
    applied by this function: it fits at unit strength (gamma=1), exactly as
    `complete_residuals_direct` does, and returns the raw correction. The
    caller applies `target_informed_runtime.scale_completion` afterwards, so
    ``strength=0`` is an exact native-target-base control by the same
    contract `complete_residuals_direct` already has.

    Returns ``(target_corrections, diagnostics)``. ``target_corrections``
    only contains the fitted ``attn.out_proj``/``mlp.c_proj`` weight+bias
    keys -- nothing outside `config.components` is touched.
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    components = order_components(config.components)
    batches = captured["target_batches"]
    target_outputs_by_position = captured["target_base_outputs_by_position"]
    if set(desired) != set(range(pairing.target_depth)):
        raise ValueError(
            "desired effects do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(desired)}"
        )
    # Force independent (no-mount) semantics for the shared solver regardless
    # of what cascade_order the caller's config happens to carry -- see the
    # docstring above and the module docstring for why this is always
    # correct for Direct Residual, not merely a default.
    solver_config = replace(config, cascade_order="independent") if config.cascade_order != "independent" else config
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        positions = list(range(pairing.target_depth))
        for j in positions:
            if j not in target_outputs_by_position:
                raise ValueError(f"Captured target reference is missing for position {j}")
        source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
        if config.block_split == "backfit":
            fitted = _fit_block_boundary_backfit(
                target_model,
                current_state,
                positions,
                source_coordinates,
                desired,
                target_outputs_by_position,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
            )
        elif config.block_split == "joint":
            fitted = _fit_block_boundary_joint(
                target_model,
                current_state,
                positions,
                source_coordinates,
                desired,
                target_outputs_by_position,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
            )
        else:
            # Every position is guaranteed pristine here (nothing is ever
            # mounted between fits, cross-position or intra-position), so all
            # of them share one target forward sweep instead of one per
            # position -- see _fit_all_positions_independent's docstring.
            fitted = _fit_all_positions_independent(
                target_model,
                current_state,
                positions,
                source_coordinates,
                desired,
                target_outputs_by_position,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
            )
        for j in positions:
            position_corrections, block_rows = fitted[j]
            target_corrections.update(position_corrections)
            for row in block_rows:
                row["source_position"] = row.pop("source_coordinate")
                # Analysis-only: which statistic Q_j was fit on. Never read by
                # any fit -- the row schema is otherwise unchanged, so this is
                # additive for every existing consumer (golden hashes are over
                # target_corrections, not these diagnostics rows).
                row["procrustes_source"] = config.procrustes_source
                diagnostics.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
    return target_corrections, diagnostics


def fit_sequential_source_endpoints(
    target_model,
    target_base_state: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Fit the source base, mount its synthesis, then fit the fine-tuned stage.

    Q and its means are fitted once on source base/native target boundaries.
    Both endpoint modes retain the mounted synthesized base as the stage-two
    design. ``sequential_source_endpoints`` fits the mapped FT endpoint against
    that network's outputs; ``sequential_delta_on_synthesized_base`` instead
    fits the mapped source update directly on that network's activations.
    """
    if config.endpoint_construction not in {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}:
        raise ValueError("sequential endpoint fit requires a sequential endpoint_construction")
    native_outputs = captured["target_base_outputs_by_position"]
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    mapped_base: dict[int, list[Tensor]] = {}
    mapped_ft: dict[int, list[Tensor]] = {}
    base_desired: dict[int, list[Tensor]] = {}
    alignment: dict[int, dict[str, float]] = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        native = native_outputs[j]
        s0 = _aligned(source_base[i], native)
        s1 = _aligned(source_ft[i], native)
        q, mu_s, mu_t = centered_rectangular_procrustes(_rows(s0).double(), _rows(native).double())
        q, mu_s, mu_t = q.float(), mu_s.float(), mu_t.float()
        mapped_base[j] = [(b - mu_s) @ q + mu_t for b in s0]
        mapped_ft[j] = [(f - mu_s) @ q + mu_t for f in s1]
        base_desired[j] = [m - t for m, t in zip(mapped_base[j], native, strict=True)]
        alignment[j] = {
            "pretrained_residual_norm": float(_rows(base_desired[j]).double().norm()),
            "source_update_norm": float(
                _rows([f - b for b, f in zip(mapped_base[j], mapped_ft[j], strict=True)]).double().norm()
            ),
        }
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        base_correction, base_rows = fit_direct_residual(
            target_model,
            target_base_state,
            captured,
            base_desired,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
        synthesized_base = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        for key, correction in base_correction.items():
            synthesized_base[key] = synthesized_base[key] + correction.to(synthesized_base[key])
        target_model.load_state_dict(synthesized_base, strict=True)
        requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
        synthesized_raw = capture_tokens(
            target_model, captured["target_batches"], requests, device, family_adapter=family_adapter
        )
        synthesized_outputs = {int(j): batches for j, batches in synthesized_raw.items()}
        if config.endpoint_construction == "sequential_delta_on_synthesized_base":
            # Preserve the stage-one synthesized pretrained model as the
            # stage-two design H_syn, while fitting only the mapped source
            # fine-tuning update.  The affine Procrustes means cancel between
            # these two mapped endpoints.  This gives the zero-update
            # invariant without changing the sequential hypothesis.
            ft_desired = {
                j: [ft - base for base, ft in zip(mapped_base[j], mapped_ft[j], strict=True)]
                for j in range(pairing.target_depth)
            }
        else:
            # Historical endpoint mode: fit the mapped FT endpoint against
            # the synthesized network's current output.
            ft_desired = {
                j: [m - t for m, t in zip(mapped_ft[j], synthesized_outputs[j], strict=True)]
                for j in range(pairing.target_depth)
            }
        ft_captured = dict(captured)
        ft_captured["target_base_outputs_by_position"] = synthesized_outputs
        task_vector, ft_rows = fit_direct_residual(
            target_model,
            synthesized_base,
            ft_captured,
            ft_desired,
            pairing,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
        # Confirm that subtracting the two *mounted* endpoints represents the
        # returned correction, subject to fp32 rounding of full model weights.
        max_endpoint_error = 0.0
        for key, correction in task_vector.items():
            ft_value = synthesized_base[key] + correction.to(synthesized_base[key])
            max_endpoint_error = max(
                max_endpoint_error,
                float(((ft_value - synthesized_base[key]).float() - correction.float()).abs().max()),
            )
        if not all(torch.isfinite(value).all() for value in task_vector.values()):
            raise RuntimeError("sequential endpoint fit produced a non-finite task vector")
        for row in base_rows:
            row["endpoint_stage"] = "pretrained"
        for row in ft_rows:
            row["endpoint_stage"] = "finetuned"
        diagnostics = {
            "alignment_by_position": alignment,
            "pretrained_correction_norm": float(
                sum(v.double().square().sum() for v in base_correction.values()).sqrt()
            ),
            "task_vector_norm": float(sum(v.double().square().sum() for v in task_vector.values()).sqrt()),
            "max_endpoint_subtraction_error": max_endpoint_error,
        }
        return task_vector, base_rows + ft_rows, diagnostics
    finally:
        target_model.load_state_dict(entry_state, strict=True)
