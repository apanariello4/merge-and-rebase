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

import contextlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..rebase.discrete_layer_match import DiscreteLayerPairing, discrete_layer_pairing
from .target_informed_runtime import (
    COMPONENT_INPUT_KIND,
    _aligned,
    _component_effective_out,
    _finalize_independent_component,
    _fit_all_positions_independent,
    _fit_block_boundary_backfit,
    _fit_component_outputs_from_contributions,
    _layout_for,
    _rows,
    capture_source_component_references,
    capture_tokens,
    # Re-exported for callers that assemble Direct Residual's realization
    # diagnostics (vision_rebase.py's _run_direct_residual_fit) and for
    # tests: both are standalone, post-hoc analyses over an already-fitted,
    # unit-strength task vector, gated entirely on
    # DirectResidualConfig.realization_diagnostics, and neither is called by
    # fit_direct_residual itself (its own 2-tuple return is unchanged).
    compute_direct_residual_task_vector_stats,
    iter_capture_tokens,
    measure_direct_residual_realization,
    paired_calibration,
)

__all__ = [
    "DirectResidualConfig",
    "capture_paired_boundary_activations",
    "compute_desired_effects",
    "compute_direct_residual_task_vector_stats",
    "fit_direct_residual",
    "fit_direct_residual_streaming",
    "measure_direct_residual_realization",
    "parse_direct_residual_config",
    "position_source_contributions",
    "prepare_direct_residual_streaming",
]
from .target_residual_completion import (
    COMPONENT_FORWARD_ORDER,
    INTERNAL_COMPONENTS,
    ResidualSufficientStatistics,
    _procrustes_from_cross,
    centered_rectangular_procrustes,
    order_components,
)

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
    component_inputs: tuple[str, ...] = (),
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

    ``component_inputs``, when non-empty (``component_target='output_local'``),
    additionally captures every source block's own component input banks and
    both endpoints' weight/bias slices for the named components, via
    ``target_informed_runtime.capture_source_component_references``. Unlike
    the deduplicated ``distinct_source_indices`` above, this always spans
    ``range(pairing.source_depth)``: a shrink layout's span partition
    (``position_source_contributions``) can reference source blocks that are
    not any position's *closest* pairing match, so every source block's
    references have to exist regardless of direction. When
    ``component_inputs=()`` (the default), this is a strict no-op -- the
    returned dict is unchanged from before this parameter existed.
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
    result = {
        "source_base_outputs": {int(k): v for k, v in source_base_raw.items()},
        "source_ft_outputs": {int(k): v for k, v in source_ft_raw.items()},
        "target_base_outputs_by_position": {int(k): v for k, v in target_raw.items()},
        "target_batches": target_batches,
        "calibration": metadata,
    }
    if component_inputs:
        source_component_inputs, source_component_weights = capture_source_component_references(
            source_base_model,
            source_ft_model,
            source_batches,
            list(range(pairing.source_depth)),
            component_inputs,
            device,
            family_adapter=family_adapter,
        )
        result["source_component_inputs"] = source_component_inputs
        result["source_component_weights"] = source_component_weights
    return result


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
    # Which target each component's fit is asked to reproduce.
    #   "block_boundary" -- the historical, default behaviour: every requested
    #                       component (out_proj and c_proj alike) is fit
    #                       against the SAME block-boundary target D_j.
    #                       Bit-identical to pre-ablation code, golden-hash
    #                       pinned (see tests/test_direct_residual_component_
    #                       coverage.py).
    #   "output_local"   -- component-specific target using only each
    #                       contributing source block's own (local) weight
    #                       change; see ``position_source_contributions`` for
    #                       how contributions and their weights are derived
    #                       per direction (extend/shrink/same_arch), and
    #                       ``target_informed_runtime._fit_component_outputs_
    #                       from_contributions`` for the fit itself. Requires
    #                       LayerScale to be nn.Identity, asserted at fit time.
    #                       Only this mode allows internal components
    #                       (q/k/v/c_fc) in ``components``.
    component_target: str = "block_boundary"
    # Analysis-only. Never read by any fit; only adds diagnostic fields to the
    # per-component rows. False reproduces the exact historical row schema.
    realization_diagnostics: bool = False
    # Only valid with component_target="block_boundary" and components subset
    # of {"attn.out_proj", "mlp.c_proj"}.
    #   "none"    -- historical behaviour: every requested component is fit
    #                independently against the SAME block-boundary target D_j
    #                (golden-hash pinned).
    #   "backfit" -- intra-block Gauss-Seidel: each component is refit against
    #                the residual left over once every OTHER component's
    #                current fit is mounted on a local (never the live target
    #                model) copy of the block and replayed on the pristine
    #                captured block input X_j^0. See
    #                target_informed_runtime._fit_block_boundary_backfit.
    block_split: str = "none"
    backfit_max_iters: int = 20
    # Stop when the change in ||E||/||D_j|| between consecutive sweeps drops
    # below this.
    backfit_tol: float = 1e-4
    # "resident" (default): capture_paired_boundary_activations/fit_direct_residual
    # hold full per-batch activation banks in host RAM for every position at
    # once (see the module docstring); bit-identical to pre-ablation code.
    # "streaming": prepare_direct_residual_streaming/fit_direct_residual_streaming
    # accumulate Procrustes and ridge sufficient statistics batch-by-batch,
    # keeping host RAM O(1) in num_batches, at the cost of requiring
    # component_target='block_boundary', block_split='none' and
    # realization_diagnostics=False (see parse_direct_residual_config).
    activation_storage: str = "resident"
    # Only meaningful with activation_storage="streaming": split
    # fit_direct_residual_streaming's target positions into chunks of this
    # size (each chunk gets its own capture sweep) to bound host RAM further
    # when num_positions * num_components is itself large. None (default)
    # fits every position in one chunk.
    streaming_position_chunk: int | None = None


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
        "component_target",
        "realization_diagnostics",
        "block_split",
        "backfit_max_iters",
        "backfit_tol",
        "activation_storage",
        "streaming_position_chunk",
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
    if cfg.component_target not in {"block_boundary", "output_local"}:
        raise ValueError("component_target must be 'block_boundary' or 'output_local'")
    if not isinstance(cfg.realization_diagnostics, bool):
        raise TypeError("realization_diagnostics must be bool")
    output_mode = cfg.component_target != "block_boundary"
    components = cfg.components
    if isinstance(components, str) or not isinstance(components, (list, tuple)):
        raise ValueError("components must be a list of projection names")
    components = tuple(components)
    if not components:
        raise ValueError("components must not be empty")
    if len(set(components)) != len(components):
        raise ValueError("components must not repeat a projection")
    if output_mode:
        all_names = frozenset(COMPONENT_FORWARD_ORDER) | frozenset(INTERNAL_COMPONENTS)
        unsupported = set(components) - all_names
        if unsupported:
            raise ValueError(f"unknown components: {sorted(unsupported)}; supported: {sorted(all_names)}")
        # Every requested component fits its own component-specific target, so
        # there is no "unanchored" fit the way a dangling out_proj-only
        # block_boundary fit would be; any non-empty, repeat-free subset of
        # the six names (CANONICAL_COMPONENT_ORDER) is legal.
    else:
        unsupported = set(components) - set(COMPONENT_FORWARD_ORDER)
        if unsupported:
            raise ValueError(f"unknown components: {sorted(unsupported)}; supported: {sorted(COMPONENT_FORWARD_ORDER)}")
        # Unlike target_residual_completion.parse_residual_completion_config
        # (P1), which requires 'mlp.c_proj' to anchor the sequential cascade,
        # Direct Residual never cascades -- every position is independently
        # fit against the pristine target base (see the module docstring) --
        # so there is no "unanchored" out_proj-only fit the way there would
        # be for a cascaded completion. Any non-empty, repeat-free subset of
        # COMPONENT_FORWARD_ORDER is legal here, including {'attn.out_proj'}
        # alone (DT-O).
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
    if cfg.block_split not in {"none", "backfit"}:
        raise ValueError("block_split must be 'none' or 'backfit'")
    if cfg.block_split == "backfit":
        if cfg.component_target != "block_boundary":
            raise ValueError("block_split='backfit' requires component_target='block_boundary'")
        if set(components) - set(COMPONENT_FORWARD_ORDER):
            raise ValueError(
                "block_split='backfit' only supports residual-writing components "
                f"{sorted(COMPONENT_FORWARD_ORDER)}"
            )
    if isinstance(cfg.backfit_max_iters, bool) or not isinstance(cfg.backfit_max_iters, int):
        raise ValueError("backfit_max_iters must be an integer")
    if cfg.backfit_max_iters <= 0:
        raise ValueError("backfit_max_iters must be positive")
    if isinstance(cfg.backfit_tol, bool) or not isinstance(cfg.backfit_tol, (int, float)):
        raise ValueError("backfit_tol must be a finite real number")
    if not math.isfinite(float(cfg.backfit_tol)) or cfg.backfit_tol <= 0:
        raise ValueError("backfit_tol must be finite and > 0")
    if cfg.activation_storage not in {"resident", "streaming"}:
        raise ValueError("activation_storage must be 'resident' or 'streaming'")
    if cfg.streaming_position_chunk is not None:
        if isinstance(cfg.streaming_position_chunk, bool) or not isinstance(cfg.streaming_position_chunk, int):
            raise ValueError("streaming_position_chunk must be None or a positive integer")
        if cfg.streaming_position_chunk <= 0:
            raise ValueError("streaming_position_chunk must be None or a positive integer")
    if cfg.activation_storage == "streaming":
        if cfg.component_target != "block_boundary":
            raise ValueError("activation_storage='streaming' requires component_target='block_boundary'")
        if cfg.block_split != "none":
            raise ValueError("activation_storage='streaming' requires block_split='none'")
        if cfg.realization_diagnostics:
            raise ValueError("activation_storage='streaming' requires realization_diagnostics=False")
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


def position_source_contributions(pairing: DiscreteLayerPairing) -> dict[int, list[tuple[int, float]]]:
    """Per-target-position weighted source contributions for
    ``component_target='output_local'``'s residual-writing components.

    The rule is span-aware and derived purely from ``pairing`` (never
    assumed), so it is the same code for extend, shrink and same-arch:

    * **Extend / same-arch** (``target_depth >= source_depth``): let
      ``m_i = #{j : pairing.pairing[j] == i}`` be how many target positions
      the discrete pairing sends to source block ``i``. Position ``j``'s only
      contribution is its own paired source block, at weight ``1/m_{i(j)}``:
      the source block's one true effect is split evenly across however many
      target positions realize it, mirroring BRACE's spread/duplicate
      insertion semantics (``m=1`` -- e.g. every position of a same-arch
      pairing -- is the plain, unweighted single-block case).
    * **Shrink** (``source_depth > target_depth``): partition every source
      block into spans ``S_j = {i : round(i*(target_depth-1)/(source_depth-1))
      == j}`` -- the mirror-image discrete pairing, from source depth down to
      target depth. Position ``j``'s contributions are every source block in
      its span, each at full weight 1.0: a shrunk target position absorbs the
      whole local effect of every source block that collapsed into it, and
      ``fit_direct_residual`` sums their (individually Procrustes-aligned)
      terms.

    Internal components (q/k/v/c_fc) use only the paired block
    ``pairing.pairing[j]``, but at the same weight that block carries in
    THIS function's own output for the position (``1/m_i`` on
    extend/same_arch, ``1.0`` on shrink) -- see
    ``_fit_component_outputs_from_contributions``, which looks up that
    weight from this function's return value rather than hardcoding 1.0.

    Raises ``AssertionError`` if the shrink partition does not cover every
    source block exactly once, or if the forward-pairing's own choice of
    source block for position ``j`` is not a member of ``j``'s span -- both
    would indicate the forward/reverse discrete formulas disagree, which
    should not happen for well-behaved depth pairs and is a bug to surface
    loudly rather than silently misattribute contributions.
    """
    source_depth, target_depth = pairing.source_depth, pairing.target_depth
    contributions: dict[int, list[tuple[int, float]]] = {j: [] for j in range(target_depth)}
    if target_depth >= source_depth:
        counts: dict[int, int] = {}
        for i in pairing.pairing:
            counts[i] = counts.get(i, 0) + 1
        for j in range(target_depth):
            i = pairing.pairing[j]
            contributions[j] = [(i, 1.0 / counts[i])]
        return contributions
    # Shrink: source_depth > target_depth. Partition every source block by the
    # mirror-image discrete pairing (source depth playing the "target depth"
    # role, target depth playing the "source depth" role).
    reverse = discrete_layer_pairing(target_depth, source_depth)
    spans: dict[int, list[int]] = {j: [] for j in range(target_depth)}
    for source_idx, j in enumerate(reverse):
        spans[j].append(source_idx)
    covered = sorted(idx for span in spans.values() for idx in span)
    if covered != list(range(source_depth)):
        raise AssertionError(
            "Shrink span partition does not cover every source block exactly once: "
            f"covered={covered}, expected={list(range(source_depth))}"
        )
    for j in range(target_depth):
        paired = pairing.pairing[j]
        if paired not in spans[j]:
            raise AssertionError(
                f"Forward pairing's source block for position {j} ({paired}) is not a member "
                f"of its own shrink span {spans[j]}; forward/reverse discrete pairings disagree"
            )
        contributions[j] = [(i, 1.0) for i in sorted(spans[j])]
    return contributions


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
        if config.component_target == "output_local":
            # Every component gets its own per-component, per-contribution
            # target -- not the shared block-boundary D_j the branch below
            # fits against -- so this is a genuinely different solve, not a
            # reuse of _fit_all_positions_independent.
            source_component_inputs = captured.get("source_component_inputs")
            source_component_weights = captured.get("source_component_weights")
            if source_component_inputs is None or source_component_weights is None:
                raise ValueError(
                    "component_target='output_local' requires 'source_component_inputs' and "
                    "'source_component_weights' in captured; call "
                    "capture_paired_boundary_activations with component_inputs set to the "
                    "requested components"
                )
            paired_source_index = {j: pairing.pairing[j] for j in positions}
            fitted = _fit_component_outputs_from_contributions(
                target_model,
                current_state,
                positions,
                position_source_contributions(pairing),
                paired_source_index,
                source_component_inputs,
                source_component_weights,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
            )
        elif config.block_split == "backfit":
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
                diagnostics.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
    return target_corrections, diagnostics


class _StreamingCrossCovariance:
    """Chan's pairwise online accumulator for a centered cross-covariance.

    Equivalent to accumulating every row into one bank and computing
    ``(X - mean_x).T @ (Y - mean_y)`` directly (what
    `centered_rectangular_procrustes` does), but in O(1) batches rather than
    O(num_batches) host memory. Kept on ``device`` (float64) throughout;
    inputs are expected already cast to float64 by the caller.
    """

    def __init__(self, device=None) -> None:
        self.device = device
        self.n = 0
        self.mean_x: Tensor | None = None
        self.mean_y: Tensor | None = None
        self.c: Tensor | None = None

    def update(self, x: Tensor, y: Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError("cross-covariance update requires matching row counts")
        n_b = int(x.shape[0])
        if n_b == 0:
            return
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
        mean_x_b = x.mean(dim=0)
        mean_y_b = y.mean(dim=0)
        c_b = (x - mean_x_b).T @ (y - mean_y_b)
        if self.n == 0:
            self.n, self.mean_x, self.mean_y, self.c = n_b, mean_x_b, mean_y_b, c_b
            return
        n_a = self.n
        n = n_a + n_b
        dx = mean_x_b - self.mean_x
        dy = mean_y_b - self.mean_y
        self.c = self.c + c_b + (n_a * n_b / n) * torch.outer(dx, dy)
        self.mean_x = self.mean_x + dx * (n_b / n)
        self.mean_y = self.mean_y + dy * (n_b / n)
        self.n = n

    def cross(self) -> Tensor:
        if self.c is None:
            raise ValueError("cannot compute cross-covariance of zero batches")
        return self.c


def prepare_direct_residual_streaming(
    source_base_model,
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
    """Streaming (Pass A) equivalent of ``capture_paired_boundary_activations`` +
    ``compute_desired_effects``: accumulates each position's Procrustes cross-
    covariance batch-by-batch instead of holding every batch's boundary bank
    resident, then solves the SVD once per position at the end.

    Uses the identical `paired_calibration` call as the resident path (same
    ``num_batches``/``seed`` -> identical sample IDs) and the identical
    `_aligned` token-interpolation helper, so the only difference from the
    resident path is *when* the Procrustes map is extracted from the
    accumulated cross-covariance (once at the end here, vs. from a fully
    materialized row bank there) -- both compute ``_procrustes_from_cross``
    over mathematically the same centered cross-covariance matrix.

    Returns a dict consumed by `fit_direct_residual_streaming`: the fitted
    per-position Procrustes maps (``"q_by_position"``), a per-(batch,
    position) determinism fingerprint of the target boundary activations
    (``"fingerprints"``) that Pass B uses to detect a target model mutated
    between passes, the replayed calibration batches for both sides, and the
    calibration metadata.
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if pairing.source_depth < 1:
        raise ValueError("pairing.source_depth must be positive")
    source_batches, target_batches, metadata = paired_calibration(
        source_loader, target_loader, num_batches=num_batches, seed=seed
    )
    distinct_source_indices = sorted(set(pairing.pairing))
    src_requests = {str(i): (i, "boundary") for i in distinct_source_indices}
    tgt_requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    accumulators = {j: _StreamingCrossCovariance(device=device) for j in range(pairing.target_depth)}
    fingerprints: dict[tuple[int, int], tuple[float, float]] = {}

    src_gen = iter_capture_tokens(
        source_base_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
    )
    tgt_gen = iter_capture_tokens(
        target_base_model, target_batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
    )
    with contextlib.closing(src_gen), contextlib.closing(tgt_gen):
        for k, (src, tgt) in enumerate(zip(src_gen, tgt_gen, strict=True)):
            for j in range(pairing.target_depth):
                i = pairing.pairing[j]
                t = tgt[str(j)]
                x = _aligned([src[str(i)]], [t])[0]
                x_rows = x.reshape(-1, x.shape[-1]).double()
                y_rows = t.reshape(-1, t.shape[-1]).double()
                accumulators[j].update(x_rows, y_rows)
                t64 = t.double()
                fingerprints[(k, j)] = (float(t64.sum().item()), float((t64**2).sum().item()))

    q_by_position = {j: _procrustes_from_cross(acc.cross()).float().cpu() for j, acc in accumulators.items()}
    return {
        "q_by_position": q_by_position,
        "fingerprints": fingerprints,
        "source_batches": source_batches,
        "target_batches": target_batches,
        "calibration": metadata,
        "distinct_source_indices": distinct_source_indices,
    }


@torch.no_grad()
def fit_direct_residual_streaming(
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
    """Streaming (Pass B) equivalent of ``fit_direct_residual``: fits every
    target position's residual-writing components from chunked capture
    sweeps that accumulate `ResidualSufficientStatistics` batch-by-batch,
    instead of ``_fit_all_positions_independent``'s single fully-materialized
    capture. Requires ``config.activation_storage == 'streaming'`` semantics
    to already be validated by `parse_direct_residual_config`
    (``component_target='block_boundary'``, ``block_split='none'``,
    ``realization_diagnostics=False``).

    Mirrors ``fit_direct_residual``'s wrapper exactly: same position checks,
    same forced ``cascade_order='independent'``, same
    save/load/restore-in-``finally`` of the target model's state, and the
    same final ``source_coordinate`` -> ``source_position`` diagnostic
    renaming, in position order.

    Each chunk of ``config.streaming_position_chunk`` positions (``None`` ->
    one chunk of all positions) gets its own lockstep sweep over three
    ``iter_capture_tokens`` generators (source_base, source_ft, target),
    re-deriving ``desired = (f - b) @ q_j`` per batch from `prepared`'s
    Procrustes maps instead of reading a resident ``desired`` bank, and
    checking the just-captured target boundary against `prepared`'s
    fingerprint before trusting it as ``base_out`` (there is no resident
    ``target_base_outputs_by_position`` bank to diff against directly).
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    components = order_components(config.components)
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    positions = list(range(pairing.target_depth))
    for j in positions:
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    solver_config = replace(config, cascade_order="independent") if config.cascade_order != "independent" else config
    shim = _layout_for(family_adapter)

    chunk_size = config.streaming_position_chunk or len(positions)
    chunks = [positions[start : start + chunk_size] for start in range(0, len(positions), chunk_size)]

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}

    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        for m in unique_models.values():
            m.to(device).eval()

        for chunk in chunks:
            distinct_i_needed = sorted({pairing.pairing[pos] for pos in chunk})
            src_requests = {str(i): (i, "boundary") for i in distinct_i_needed}
            tgt_requests: dict[str, tuple[int, str]] = {}
            for pos in chunk:
                tgt_requests[f"{pos}.out"] = (pos, "boundary")
                for component in components:
                    tgt_requests[f"{pos}.{component}.h"] = (pos, COMPONENT_INPUT_KIND[component])

            stats_by: dict[tuple[int, str], ResidualSufficientStatistics] = {}
            desired_sq: dict[tuple[int, str], float] = {}
            effect_sq: dict[tuple[int, str], float] = {}
            effective_out: dict[tuple[int, str], Tensor] = {}
            for pos in chunk:
                for component in components:
                    key = shim.component_key(pos, component, prefixed=True)
                    width = int(current_state[key].shape[0])
                    pair = (pos, component)
                    effective_out[pair] = _component_effective_out(shim, target_model, pos, component, width)
                    stats_by[pair] = ResidualSufficientStatistics(device=device)
                    desired_sq[pair] = 0.0
                    effect_sq[pair] = 0.0

            sb_gen = iter_capture_tokens(
                source_base_model, source_batches, src_requests, device, family_adapter=family_adapter,
                store_device=device,
            )
            sf_gen = iter_capture_tokens(
                source_ft_model, source_batches, src_requests, device, family_adapter=family_adapter,
                store_device=device,
            )
            tgt_gen = iter_capture_tokens(
                target_model, batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
            )
            with contextlib.closing(sb_gen), contextlib.closing(sf_gen), contextlib.closing(tgt_gen):
                for k, (sb, sf, tgt) in enumerate(zip(sb_gen, sf_gen, tgt_gen, strict=True)):
                    for pos in chunk:
                        i = pairing.pairing[pos]
                        out = tgt[f"{pos}.out"]
                        out64 = out.double()
                        fp_sum = float(out64.sum().item())
                        fp_sumsq = float((out64**2).sum().item())
                        exp_sum, exp_sumsq = fingerprints[(k, pos)]
                        scale = math.sqrt(exp_sumsq) if exp_sumsq > 0 else 1.0
                        sumsq_ok = math.isclose(fp_sumsq, exp_sumsq, rel_tol=1e-9, abs_tol=1e-9)
                        sum_ok = abs(fp_sum - exp_sum) <= 1e-9 * max(scale, 1.0)
                        if not (sumsq_ok and sum_ok):
                            raise RuntimeError(
                                "Direct completion started from a target model that is not the "
                                f"native base: target boundary fingerprint mismatch between "
                                f"pass A and pass B at position {pos} (batch {k})"
                            )
                        # Align on CPU, as the resident path does on its CPU banks, so D_j stays bit-comparable.
                        out_cpu = out.cpu()
                        b = _aligned([sb[str(i)].cpu()], [out_cpu])[0]
                        f = _aligned([sf[str(i)].cpu()], [out_cpu])[0]
                        base_out = out_cpu
                        q = q_by_position[pos]
                        desired = (f - b) @ q
                        effect = out_cpu - base_out
                        error = desired - effect
                        desired_sq_val = float((desired.double() ** 2).sum().item())
                        effect_sq_val = float((effect.double() ** 2).sum().item())
                        for component in components:
                            pair = (pos, component)
                            h = tgt[f"{pos}.{component}.h"].cpu()
                            desired_sq[pair] += desired_sq_val
                            effect_sq[pair] += effect_sq_val
                            stats_by[pair].update(
                                h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None,
                                effective_out[pair],
                            )

            for pos in chunk:
                position_corrections: dict[str, Tensor] = {}
                block_rows: list[dict[str, Any]] = []
                for component in components:
                    key = shim.component_key(pos, component, prefixed=True)
                    pair = (pos, component)
                    _finalize_independent_component(
                        stats_by[pair],
                        desired_sq[pair],
                        effect_sq[pair],
                        pos,
                        component,
                        key,
                        current_state,
                        effective_out[pair],
                        solver_config,
                        source_coordinates,
                        position_corrections,
                        block_rows,
                        h_batches=None,
                    )
                target_corrections.update(position_corrections)
                for row in block_rows:
                    row["source_position"] = row.pop("source_coordinate")
                    diagnostics.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    return target_corrections, diagnostics
