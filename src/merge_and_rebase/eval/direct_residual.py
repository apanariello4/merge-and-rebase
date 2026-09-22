"""Direct Residual: a standalone, transport-free rebase method.

Direct Residual answers the same local-functional-effect question ARIADNE's
Proposal 1 ``direct_target`` mode does -- can the desired effect a source
fine-tune had on its own boundary activations be written directly into a
*different* target model's residual-writing projections, with no parameter
transport at all -- but reaches it through a depth/width-alignment mechanism
that has nothing to do with ARIADNE's block-extension apparatus.

Where ARIADNE only ever aligns a target that is exactly ``2x`` a source's
depth (via ``validate_target_protocol``'s hard doubling gate, and the
bottom-top spread/duplicate insertion that produces an ``ancestry`` map), this
module pairs an arbitrary source depth with an arbitrary target depth through
the flat, closed-form ``DiscreteLayerPairing`` (``i(j) = round(j*(D_A-1)/
(D_B-1))``, `merge_and_rebase.rebase.discrete_layer_match`). That one
cardinality change -- one target position maps to exactly one source
position, never many -- is what lets this module drop ARIADNE's two-code-path
ancestry bookkeeping (one-to-many groups for extend, many-to-one span
tracking for shrink) entirely, and is why it works uniformly across extend,
shrink, and same-arch without any structural gate.

The actual per-position ridge solve is NOT reimplemented here. It is the
exact same code `target_informed_runtime.complete_residuals_direct` has
always executed for `cascade_order="independent"` -- shared, via the private
`target_informed_runtime._fit_all_positions_independent` helper, so this
module and ARIADNE's own `target_scope="all"` independent-mode path are
structurally guaranteed to agree whenever they are handed the same alignment
and the same captured banks (see `tests/test_direct_residual_extend_anchor.py`).

Direct Residual never mounts a correction before fitting the next one: it
never cascades, by construction, at either the cross-position or
intra-position (attn.out_proj -> mlp.c_proj) level. Every position is fit
against the pristine, untouched target base, so its `E_j == D_j` identically
-- there is no upstream state a later fit could see. `DirectResidualConfig
.cascade_order` is kept only for config-parity with
`ResidualCompletionConfig` (so a campaign generator can reuse one code path
to build both configs' JSON) and is a documented no-op here; see
`tests/test_direct_residual_cascade_order.py`. Because every position is
independent by construction, all of them are fit from one shared target
forward sweep rather than one sweep per position -- see
`_fit_all_positions_independent`'s docstring for why that is exact, not an
approximation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..rebase.discrete_layer_match import DiscreteLayerPairing
from .target_informed_runtime import (
    _aligned,
    _fit_all_positions_independent,
    _rows,
    capture_tokens,
    paired_calibration,
)
from .target_residual_completion import centered_rectangular_procrustes, order_components

Tensor = torch.Tensor


def capture_paired_boundary_activations(
    source_base_model,
    source_ft_model,
    target_base_model,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    *,
    num_batches: int,
    seed: int | None,
    device,
    family_adapter=None,
) -> dict[str, Any]:
    """Capture native boundary activations for every position Direct Residual fits.

    For every target position ``j`` in ``range(pairing.target_depth)``,
    captures ``source_base``/``source_ft`` boundary activations at
    ``pairing.pairing[j]`` -- deduplicated, since under extend many target
    positions share one source index and there is no reason to run the same
    forward pass twice -- and ``target_base`` boundary activations at ``j``
    itself. Uses `capture_tokens`/`paired_calibration` from
    `target_informed_runtime` unchanged: both already operate on
    ``(model, batches, requests)``/``(source_loader, target_loader)``, never
    on an ARIADNE realized-extension layout, so nothing here reaches into
    that machinery.

    No structural resize of any model happens (the three models passed in are
    used exactly as given, at their native depths) and no correction is
    fitted -- this function only captures activation banks.

    The returned dict also carries the replayed target-side calibration
    batches (under ``"target_batches"``) so `fit_direct_residual` can re-run
    forward passes against the (possibly partially-corrected) target model
    during the solve without needing the original `target_loader` again.
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if pairing.source_depth < 1:
        raise ValueError("pairing.source_depth must be positive")
    source_batches, target_batches, metadata = paired_calibration(
        source_loader, target_loader, num_batches=num_batches, seed=seed
    )
    # Deduplicate: under extend, many target positions share one source
    # index, and capturing it twice would be wasted compute and (worse) a
    # second, potentially non-identical activation bank for the same index
    # if anything upstream were ever non-deterministic.
    distinct_source_indices = sorted(set(pairing.pairing))
    source_requests = {str(i): (i, "boundary") for i in distinct_source_indices}
    target_requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    source_base_raw = capture_tokens(
        source_base_model, source_batches, source_requests, device, family_adapter=family_adapter
    )
    source_ft_raw = capture_tokens(
        source_ft_model, source_batches, source_requests, device, family_adapter=family_adapter
    )
    target_raw = capture_tokens(
        target_base_model, target_batches, target_requests, device, family_adapter=family_adapter
    )
    return {
        "source_base_outputs": {int(k): v for k, v in source_base_raw.items()},
        "source_ft_outputs": {int(k): v for k, v in source_ft_raw.items()},
        "target_base_outputs_by_position": {int(k): v for k, v in target_raw.items()},
        "target_batches": target_batches,
        "calibration": metadata,
    }


@dataclass(frozen=True)
class DirectResidualConfig:
    ridge_relative: float = 0.01
    # Mirrors ResidualCompletionConfig.ridge_estimator: the shared solver body
    # (_fit_direct_target_position / _fit_all_positions_independent) reads
    # this attribute unconditionally, so it has to exist here too even though
    # Direct Residual has never swept it. "fixed_relative" reproduces this
    # module's only-ever-exercised historical ridge.
    ridge_estimator: str = "fixed_relative"
    strength: float = 1.0
    num_batches: int = 10
    components: tuple[str, ...] = ("attn.out_proj", "mlp.c_proj")
    # Kept only for config-schema parity with ResidualCompletionConfig; Direct
    # Residual never mounts a correction before fitting the next one, so there
    # is no cascade for this field to order. See the module docstring and
    # tests/test_direct_residual_cascade_order.py.
    cascade_order: str = "independent"
    exact_form: bool = True
    missing_bias: str = "error"
    # "per_task_then_merge": fit one correction per source task, merge the
    # resulting target-space corrections afterwards (the ordinary path).
    # "merge_in_source_then_fit": merge task deltas on the native source base
    # first, then fit exactly once against the merged (source_base,
    # source_merged) pair. This module supports either uniformly -- neither
    # capture_paired_boundary_activations nor fit_direct_residual makes any
    # assumption about how many tasks contributed to source_ft_model, so the
    # merge-once orchestration lives entirely in the caller (vision_rebase.py).
    merge_mode: str = "per_task_then_merge"
    seed: int = 89


def parse_direct_residual_config(value: Mapping[str, Any] | None) -> DirectResidualConfig:
    """Parse and validate the Direct Residual configuration schema.

    Mirrors `target_residual_completion.parse_residual_completion_config`'s
    validation style: an explicit allowed-key set, unknown keys refused,
    type/range checks with clear messages. Direct Residual's config is
    deliberately narrower than `ResidualCompletionConfig` -- no
    `target_scope`, `mode`, `target_trajectory`, or `direct_passthrough`,
    since none of those ARIADNE-protocol concepts apply once there is no
    realized-extension layout to describe.
    """
    if value is None:
        return DirectResidualConfig()
    if not isinstance(value, Mapping):
        raise TypeError("direct_residual config must be a mapping")
    allowed = {
        "ridge_relative",
        "ridge_estimator",
        "strength",
        "num_batches",
        "components",
        "cascade_order",
        "exact_form",
        "missing_bias",
        "merge_mode",
        "seed",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown direct_residual fields: {sorted(unknown)}")
    payload = dict(value)
    if "components" in payload and isinstance(payload["components"], list):
        payload["components"] = tuple(payload["components"])
    cfg = DirectResidualConfig(**payload)
    if isinstance(cfg.num_batches, bool) or not isinstance(cfg.num_batches, int) or cfg.num_batches <= 0:
        raise ValueError("num_batches must be a positive integer")
    if isinstance(cfg.ridge_relative, bool) or not isinstance(cfg.ridge_relative, (int, float)):
        raise ValueError("ridge_relative must be a finite real number")
    if not math.isfinite(float(cfg.ridge_relative)):
        raise ValueError("ridge_relative must be finite")
    if cfg.ridge_relative <= 0:
        raise ValueError("ridge_relative must be > 0")
    if cfg.ridge_estimator not in {"fixed_relative", "empirical_bayes"}:
        raise ValueError("ridge_estimator must be 'fixed_relative' or 'empirical_bayes'")
    if isinstance(cfg.strength, bool) or not isinstance(cfg.strength, (int, float)):
        raise ValueError("strength must be a finite real number")
    if not math.isfinite(float(cfg.strength)):
        raise ValueError("strength must be finite")
    if cfg.strength < 0:
        raise ValueError("strength must be >= 0")
    if not isinstance(cfg.exact_form, bool):
        raise TypeError("exact_form must be bool")
    components = cfg.components
    if isinstance(components, str) or not isinstance(components, (list, tuple)):
        raise ValueError("components must be a list of projection names")
    components = tuple(components)
    if not components:
        raise ValueError("components must not be empty")
    if len(set(components)) != len(components):
        raise ValueError("components must not repeat a projection")
    from .target_residual_completion import COMPONENT_FORWARD_ORDER

    unsupported = set(components) - set(COMPONENT_FORWARD_ORDER)
    if unsupported:
        raise ValueError(f"unknown components: {sorted(unsupported)}; supported: {sorted(COMPONENT_FORWARD_ORDER)}")
    if "mlp.c_proj" not in components:
        raise ValueError("components must contain 'mlp.c_proj'")
    if cfg.cascade_order not in {"independent", "bottom_top", "top_bottom"}:
        raise ValueError("cascade_order must be 'independent', 'bottom_top' or 'top_bottom'")
    if cfg.missing_bias not in {"error", "materialize", "skip"}:
        raise ValueError("missing_bias must be 'error', 'materialize' or 'skip'")
    if cfg.missing_bias == "skip" and cfg.exact_form:
        raise ValueError(
            "missing_bias='skip' requires exact_form=false: the exact form fits a "
            "nonzero intercept, and dropping it would apply a centered-fit weight "
            "without the centering it assumes"
        )
    if cfg.merge_mode not in {"per_task_then_merge", "merge_in_source_then_fit"}:
        raise ValueError("merge_mode must be 'per_task_then_merge' or 'merge_in_source_then_fit'")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int):
        raise ValueError("seed must be an integer")
    return cfg


def compute_desired_effects(captured: Mapping[str, Any], pairing: DiscreteLayerPairing) -> dict[int, list[Tensor]]:
    """``D_j`` = Procrustes-aligned ``(source_ft_j - source_base_j)`` at ``pairing.pairing[j]``.

    Generalizes `target_informed_runtime._capture_residual_references`'s
    inner Procrustes computation from ARIADNE's two-code-path ancestry-group
    iteration (one-to-many groups for extend, many-to-one span tracking for
    shrink -- both needed because ARIADNE tracks *which* source block a
    position descends from, for provenance) to a single flat loop. Direct
    Residual doesn't need that bookkeeping: one cardinality (one target
    position <- exactly one source position, for every regime) handles
    extend, shrink, and same-arch uniformly.
    """
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    target_by_position = captured["target_base_outputs_by_position"]
    if set(target_by_position) != set(range(pairing.target_depth)):
        raise ValueError(
            "Captured target references do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(target_by_position)}"
        )
    desired: dict[int, list[Tensor]] = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        if i not in source_base or i not in source_ft:
            raise ValueError(f"Missing captured source reference for pairing index {i} at target position {j}")
        targets = target_by_position[j]
        source_base_batches = _aligned(source_base[i], targets)
        source_ft_batches = _aligned(source_ft[i], targets)
        q, _mu_s, _mu_t = centered_rectangular_procrustes(_rows(source_base_batches).double(), _rows(targets).double())
        q = q.float()
        desired[j] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
    return desired


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
    `target_informed_runtime._fit_all_positions_independent` (the shared
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
        # Every position is guaranteed pristine here (nothing is ever mounted
        # between fits, cross-position or intra-position), so all of them
        # share one target forward sweep instead of one per position -- see
        # _fit_all_positions_independent's docstring.
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
                diagnostics.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
    return target_corrections, diagnostics
