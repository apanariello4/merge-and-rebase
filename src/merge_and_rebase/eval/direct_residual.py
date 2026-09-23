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
from statistics import median
from typing import Any

import torch

from ..rebase.discrete_layer_match import DiscreteLayerPairing, discrete_layer_pairing
from .target_informed_runtime import (
    _aligned,
    _family_bias_key,
    _fit_all_positions_independent,
    _fit_block_boundary_backfit,
    _fit_block_boundary_joint,
    _fit_component_outputs_from_contributions,
    _layout_for,
    _rows,
    _task_vector_sha256,
    capture_block_gradients,
    capture_source_component_references,
    capture_tokens,
    # Re-exported for callers that assemble Direct Residual's realization
    # diagnostics (vision_rebase.py's _run_direct_residual_fit) and for
    # tests: both are standalone, post-hoc analyses over an already-fitted,
    # unit-strength task vector, gated entirely on
    # DirectResidualConfig.realization_diagnostics, and neither is called by
    # fit_direct_residual itself (its own 2-tuple return is unchanged).
    compute_direct_residual_task_vector_stats,
    measure_direct_residual_realization,
    paired_calibration,
)

__all__ = [
    "DirectResidualConfig",
    "apply_tv_scaling",
    "capture_paired_boundary_activations",
    "compute_alignment_diagnostics",
    "compute_desired_effects",
    "compute_direct_residual_task_vector_stats",
    "fit_direct_residual",
    "measure_direct_residual_realization",
    "parse_direct_residual_config",
    "position_source_contributions",
]
from .target_residual_completion import (
    COMPONENT_FORWARD_ORDER,
    INTERNAL_COMPONENTS,
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
    procrustes_source: str = "activation",
    source_recipe=None,
    target_recipe=None,
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

    ``procrustes_source="gradient"`` (``DirectResidualConfig.procrustes_source``)
    additionally captures block-boundary GRADIENTS -- ``dL/dT_i`` on
    ``source_base_model`` at every distinct paired source index, and
    ``dL/dT_j`` on ``target_base_model`` at every target position -- via
    ``target_informed_runtime.capture_block_gradients``, using
    ``source_recipe``/``target_recipe`` (each model's own
    ``models.grad_recipes.clip_contrastive_recipe``, exactly BiCo's
    recipe/statistic) on the SAME paired calibration batches the activation
    banks above use. ``source_ft_model`` is never used for gradients (BiCo
    only ever differentiates through base models). Stored under
    ``"source_base_gradients"``/``"target_base_gradients"``, keyed by index
    like the activation banks. Both recipes are required in gradient mode.
    Vision only: `capture_block_gradients` raises `NotImplementedError` for a
    non-``None`` ``family_adapter``.

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
    if procrustes_source not in {"activation", "gradient"}:
        raise ValueError(f"procrustes_source must be 'activation' or 'gradient', got {procrustes_source!r}")
    if procrustes_source == "gradient" and (source_recipe is None or target_recipe is None):
        raise ValueError("procrustes_source='gradient' requires both source_recipe and target_recipe")
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
    if procrustes_source == "gradient":
        source_grad_requests = {str(i): i for i in distinct_source_indices}
        target_grad_requests = {str(j): j for j in range(pairing.target_depth)}
        source_base_grad_raw = capture_block_gradients(
            source_base_model, source_batches, source_grad_requests, source_recipe, device, family_adapter=family_adapter
        )
        target_base_grad_raw = capture_block_gradients(
            target_base_model, target_batches, target_grad_requests, target_recipe, device, family_adapter=family_adapter
        )
        result["source_base_gradients"] = {int(k): v for k, v in source_base_grad_raw.items()}
        result["target_base_gradients"] = {int(k): v for k, v in target_base_grad_raw.items()}
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
    #   "joint"   -- closed-form joint ridge over the stacked (attn.out_proj,
    #                mlp.c_proj) features, solved in one linear-algebra step
    #                under the first-order approximation that the MLP does
    #                not respond to a change in attn.out_proj. Only valid for
    #                components subset of {"attn.out_proj", "mlp.c_proj"};
    #                with a single component this reduces to (is literally
    #                the same fit as) block_split="none". See
    #                target_informed_runtime._fit_block_boundary_joint.
    block_split: str = "none"
    backfit_max_iters: int = 20
    # Stop when the relative decrease of the safeguarded block objective J(Delta)
    # (data-fit term plus each component's own round-1-frozen ridge penalty; see
    # target_informed_runtime._fit_block_boundary_backfit) between consecutive
    # full sweeps drops below this. J is measured, not linearized, on every
    # sweep AND accepted/rejected at every Gauss-Seidel sub-step, so it is
    # non-increasing by construction -- see the same docstring.
    backfit_tol: float = 1e-4
    # What each position's target D_j is built from, out of the SAME centered
    # Procrustes fit Q_j: S_{j,0} -> T_j^0 (`centered_rectangular_procrustes`).
    #   "transported_delta"   -- historical, default behaviour: D_j = (S_1 -
    #                            S_0) Q_j, the fine-tuning delta transported
    #                            through Q_j. Bit-identical to pre-ablation
    #                            code, golden-hash pinned.
    #   "transported_endpoint" -- D_j = (S_1 - mu_s) Q_j + mu_t - T_j^0: apply
    #                            the centered source->target map to the
    #                            fine-tuned source endpoint, then subtract the
    #                            target zero-shot endpoint. Differs from the
    #                            delta target by exactly the Procrustes
    #                            residual E_j = (S_0 - mu_s) Q_j - (T_j^0 -
    #                            mu_t); see compute_alignment_diagnostics.
    #                            Requires component_target=
    #                            "block_boundary" (the endpoint form is only
    #                            defined against the block-boundary target).
    residual_target: str = "transported_delta"
    # Which statistic Q_j (the per-position Procrustes alignment map) is fit
    # on -- D_j = (S_ft - S_base) @ Q_j is unchanged in form either way; only
    # what Q_j is fit against changes.
    #   "activation" -- the historical, default behaviour: Q_j is fit on the
    #                   (source_base, target_base) block-boundary ACTIVATION
    #                   banks. Bit-identical to pre-ablation code, golden-hash
    #                   pinned (see tests/test_direct_residual_gradient_
    #                   procrustes.py).
    #   "gradient"  -- Q_j is fit on the block-boundary GRADIENT banks
    #                   dL/dT_i (source base) and dL/dT_j (target base),
    #                   L = BiCo's own zero-shot contrastive CE
    #                   (models.grad_recipes.clip_contrastive_recipe), on the
    #                   same paired calibration batches -- see
    #                   target_informed_runtime.capture_block_gradients and
    #                   capture_paired_boundary_activations's
    #                   procrustes_source parameter. Vision only. Only valid
    #                   with component_target="block_boundary": output_local
    #                   never calls compute_desired_effects (its own
    #                   per-component targets are fit directly from component
    #                   activation banks), so there is no Q_j for this field
    #                   to redirect there.
    procrustes_source: str = "activation"
    # Label-free rescaling of the unit-strength task vector, applied AFTER
    # fit_direct_residual assembles tau but BEFORE the caller's per-task
    # alpha-search. Motivation: block_boundary O+D realizes ||delta T_j|| far
    # from ||D_j|| (see measure_direct_residual_realization's
    # joint_delta_norm_over_desired), which pushes alpha-search onto a
    # badly-resolved region of its grid.
    #   "none"      -- historical behaviour: tau is untouched (golden-hash
    #                   pinned; see tests/test_direct_residual_tv_scaling.py).
    #   "global"    -- tau <- tau / c, c = median_j(||delta T_j|| / ||D_j||)
    #                   measured with all of tau mounted at unit strength in
    #                   one sweep. See apply_tv_scaling.
    #   "per_block" -- per-block-boundary-position scalars s_j, found by
    #                   tv_scaling_iters rounds of a simultaneous (Jacobi-
    #                   style) update s_j <- s_j / r_j, r_j measured from one
    #                   mounted sweep of the CURRENT scaled combination. See
    #                   apply_tv_scaling.
    # Only block_split="none" is supported; parse_direct_residual_config
    # rejects tv_scaling != "none" combined with block_split in
    # {"backfit", "joint"} rather than silently mis-scaling a per-block-split
    # fit whose interaction with this measurement has not been verified.
    tv_scaling: str = "none"
    # Number of simultaneous (Jacobi) update rounds for tv_scaling="per_block".
    # Unused (but still validated) for "none"/"global".
    tv_scaling_iters: int = 3


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
        "residual_target",
        "procrustes_source",
        "tv_scaling",
        "tv_scaling_iters",
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
    if cfg.block_split not in {"none", "backfit", "joint"}:
        raise ValueError("block_split must be 'none', 'backfit' or 'joint'")
    if cfg.block_split in {"backfit", "joint"}:
        if cfg.component_target != "block_boundary":
            raise ValueError(f"block_split={cfg.block_split!r} requires component_target='block_boundary'")
        if set(components) - set(COMPONENT_FORWARD_ORDER):
            raise ValueError(
                f"block_split={cfg.block_split!r} only supports residual-writing components "
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
    if cfg.residual_target not in {"transported_delta", "transported_endpoint"}:
        raise ValueError("residual_target must be 'transported_delta' or 'transported_endpoint'")
    if cfg.residual_target == "transported_endpoint" and cfg.component_target != "block_boundary":
        raise ValueError("residual_target='transported_endpoint' requires component_target='block_boundary'")
    if cfg.procrustes_source not in {"activation", "gradient"}:
        raise ValueError("procrustes_source must be 'activation' or 'gradient'")
    if cfg.procrustes_source == "gradient" and cfg.component_target != "block_boundary":
        raise ValueError(
            "procrustes_source='gradient' requires component_target='block_boundary': "
            f"component_target={cfg.component_target!r} never calls compute_desired_effects "
            "(its per-component targets are fit directly from component activation banks), "
            "so there is no block-boundary Q_j for procrustes_source to redirect"
        )
    if cfg.tv_scaling not in {"none", "global", "per_block"}:
        raise ValueError("tv_scaling must be 'none', 'global' or 'per_block'")
    if isinstance(cfg.tv_scaling_iters, bool) or not isinstance(cfg.tv_scaling_iters, int):
        raise ValueError("tv_scaling_iters must be an integer")
    if cfg.tv_scaling_iters <= 0:
        raise ValueError("tv_scaling_iters must be a positive integer")
    if cfg.tv_scaling != "none" and cfg.block_split != "none":
        raise ValueError(
            f"tv_scaling={cfg.tv_scaling!r} requires block_split='none' (got "
            f"block_split={cfg.block_split!r}): tv_scaling's mount-and-measure machinery "
            "(apply_tv_scaling) is only verified against the unsplit fit path; combining it "
            "with the backfit/joint intra-block solve is rejected rather than silently "
            "producing an unverified rescaling"
        )
    if cfg.residual_target == "transported_endpoint" and cfg.procrustes_source != "activation":
        raise ValueError(
            "residual_target='transported_endpoint' requires procrustes_source='activation': the endpoint "
            "target uses the activation-space means mu_s, mu_t of the same Procrustes fit, which a "
            "gradient-fitted Q_j does not provide"
        )
    return cfg


def compute_desired_effects(
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    residual_target: str = "transported_delta",
    procrustes_source: str = "activation",
    diagnostics_out: dict[int, dict[str, Any]] | None = None,
) -> dict[int, list[Tensor]]:
    """``D_j`` = Procrustes-aligned target effect at ``pairing.pairing[j]``.

    Generalizes `target_informed_runtime._capture_residual_references`'s
    inner Procrustes computation from ARIADNE's two-code-path ancestry-group
    iteration (one-to-many groups for extend, many-to-one span tracking for
    shrink -- both needed because ARIADNE tracks *which* source block a
    position descends from, for provenance) to a single flat loop. Direct
    Residual doesn't need that bookkeeping: one cardinality (one target
    position <- exactly one source position, for every regime) handles
    extend, shrink, and same-arch uniformly.

    ``residual_target`` selects what ``D_j`` is built from, given the SAME
    centered Procrustes fit ``Q_j, mu_s, mu_t = centered_rectangular_procrustes
    (S_{j,0} -> T_j^0)``:

      * ``"transported_delta"`` (default): ``D_j = (S_1 - S_0) Q_j``, exactly
        the historical expression and op order -- byte-identical to the
        pre-ablation code.
      * ``"transported_endpoint"``: ``D_j = (S_1 - mu_s) Q_j + mu_t - T_j^0``
        per batch, applying the centered map to the fine-tuned source
        endpoint and subtracting the target zero-shot endpoint.

    The two differ by exactly the Procrustes residual ``E_j = (S_0 - mu_s)
    Q_j - (T_j^0 - mu_t)`` -- the thing the fit minimizes; see
    `compute_alignment_diagnostics` for that residual's diagnostics, kept
    deliberately separate (and out of this function's cost) -- see its
    docstring for why.

    ``procrustes_source="activation"`` (default) is bit-identical to the
    pre-ablation code: ``Q_j`` is fit on the (source_base, target_base)
    ACTIVATION banks, exactly as before. ``procrustes_source="gradient"``
    changes only the statistic ``Q_j`` is fit on -- to the block-boundary
    GRADIENT banks `capture_paired_boundary_activations` captured under
    ``"source_base_gradients"``/``"target_base_gradients"`` -- using the same
    helper, the same centering and the same ``_aligned`` token interpolation.
    ``D_j = (source_ft_i - source_base_i) @ Q_j`` is unchanged in form in both
    modes: a gradient difference is never used as the regression target,
    only as the alignment statistic.

    When ``diagnostics_out`` is provided (a caller-owned, initially-empty
    dict), it is populated per position with analysis-only fields -- never
    read by any fit. In gradient mode this includes the activation-space
    ``Q`` computed purely for comparison (``"activation_gradient_procrustes_
    overlap"`` = :math:`\\lVert Q_{act}^\\top Q_{grad}\\rVert_F^2 / d_{\\min}`)
    and ``"procrustes_rank"`` (the numerical rank of the centered
    cross-covariance the gradient ``Q`` was solved from).
    """
    if procrustes_source not in {"activation", "gradient"}:
        raise ValueError(f"procrustes_source must be 'activation' or 'gradient', got {procrustes_source!r}")
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    target_by_position = captured["target_base_outputs_by_position"]
    if set(target_by_position) != set(range(pairing.target_depth)):
        raise ValueError(
            "Captured target references do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(target_by_position)}"
        )
    if residual_target not in {"transported_delta", "transported_endpoint"}:
        raise ValueError("residual_target must be 'transported_delta' or 'transported_endpoint'")
    source_base_grad = captured.get("source_base_gradients")
    target_base_grad = captured.get("target_base_gradients")
    if procrustes_source == "gradient" and (source_base_grad is None or target_base_grad is None):
        raise ValueError(
            "procrustes_source='gradient' requires 'source_base_gradients' and 'target_base_gradients' in "
            "captured; call capture_paired_boundary_activations with procrustes_source='gradient'"
        )
    if residual_target == "transported_endpoint" and procrustes_source != "activation":
        raise ValueError("residual_target='transported_endpoint' requires procrustes_source='activation'")
    desired: dict[int, list[Tensor]] = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        if i not in source_base or i not in source_ft:
            raise ValueError(f"Missing captured source reference for pairing index {i} at target position {j}")
        targets = target_by_position[j]
        source_base_batches = _aligned(source_base[i], targets)
        source_ft_batches = _aligned(source_ft[i], targets)
        if procrustes_source == "gradient":
            if i not in source_base_grad or j not in target_base_grad:
                raise ValueError(f"Missing captured gradient reference for pairing index {i} at target position {j}")
            aligned_source_grad = _aligned(source_base_grad[i], target_base_grad[j])
            grad_source_rows = _rows(aligned_source_grad).double()
            grad_target_rows = _rows(target_base_grad[j]).double()
            q, _mu_gs, _mu_gt = centered_rectangular_procrustes(grad_source_rows, grad_target_rows)
            if diagnostics_out is not None:
                q_act, _mu_s, _mu_t = centered_rectangular_procrustes(
                    _rows(source_base_batches).double(), _rows(targets).double()
                )
                d_min = min(q.shape)
                overlap = float(((q_act.T @ q).norm() ** 2) / d_min)
                gs_centered = grad_source_rows - grad_source_rows.mean(dim=0)
                gt_centered = grad_target_rows - grad_target_rows.mean(dim=0)
                cross = gs_centered.T @ gt_centered
                diagnostics_out[j] = {
                    "procrustes_source": "gradient",
                    "procrustes_rank": int(torch.linalg.matrix_rank(cross)),
                    "activation_gradient_procrustes_overlap": overlap,
                }
            q = q.float()
            desired[j] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
            continue
        q, mu_s, mu_t = centered_rectangular_procrustes(_rows(source_base_batches).double(), _rows(targets).double())
        if diagnostics_out is not None:
            diagnostics_out[j] = {"procrustes_source": "activation"}
        if residual_target == "transported_delta":
            q = q.float()
            desired[j] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
        else:
            q = q.float()
            mu_s = mu_s.float()
            mu_t = mu_t.float()
            desired[j] = [
                (f - mu_s) @ q + mu_t - t for f, t in zip(source_ft_batches, targets, strict=True)
            ]
    return desired


def compute_alignment_diagnostics(
    captured: Mapping[str, Any], pairing: DiscreteLayerPairing
) -> dict[int, dict[str, float]]:
    """Per-position centered-Procrustes alignment diagnostics. Analysis-only.

    Recomputes the SAME deterministic centered Procrustes fit ``compute_
    desired_effects`` computes internally (`centered_rectangular_procrustes`
    is a pure function of the captured rows, so this is a second, independent
    call, not a cached one), then reports the residual it minimizes and a few
    derived quantities -- never anything fed back into a fit.

    Deliberately NOT called from ``compute_desired_effects`` or folded into
    its cost: the caller (`vision_rebase._run_direct_residual_fit`) times and
    peak-memory-profiles the delta/endpoint construction as its own
    "alignment_calibration" bracket, and this function's float64 N x d_t
    temporaries (N in the tens of thousands of rows) would otherwise inflate
    that bracket's recorded seconds/peak-memory even in the default
    ``residual_target="transported_delta"`` path -- contaminating any
    cross-code-generation cost comparison for a number this function's own
    diagnostics never influence. Call it as a separate, untimed (or
    separately timed) step instead.

    Per position ``j``, all in float64 on the concatenated (all-batch) rows:
    ``procrustes_error_norm`` = ``||E_j||`` where ``E_j = (S_0 - mu_s) Q_j -
    (T^0 - mu_t)``; ``procrustes_relative_error`` = ``||E_j|| / ||T^0 -
    mu_t||`` (0 if the denominator is 0); ``delta_target_norm`` = ``||(S_1 -
    S_0) Q_j||``; ``endpoint_minus_delta_over_delta`` = ``||E_j|| /
    delta_target_norm`` (0 if the denominator is 0) -- how the transported-
    delta and transported-endpoint targets actually differ, scaled against
    the delta itself (not against ``T^0``, which is typically much larger
    than a fine-tuning delta); ``procrustes_error_in_range_norm`` /
    ``procrustes_error_out_of_range_norm`` = ``||E_j Q_j^T Q_j||`` /
    ``||E_j (I - Q_j^T Q_j)||`` (the latter is exactly 0 when ``d_t <=
    d_s``, since ``Q_j^T Q_j`` is only a proper projector -- rank ``d_s`` --
    when ``d_t > d_s``); ``mean_offset_norm`` = ``||mu_s Q_j - mu_t||``
    (documents what the literal, non-affine ``S Q`` form would have added).
    Also ``source_dim``, ``target_dim``.
    """
    source_base = captured["source_base_outputs"]
    source_ft = captured["source_ft_outputs"]
    target_by_position = captured["target_base_outputs_by_position"]
    if set(target_by_position) != set(range(pairing.target_depth)):
        raise ValueError(
            "Captured target references do not exactly match the pairing's target positions: "
            f"expected={sorted(range(pairing.target_depth))}, found={sorted(target_by_position)}"
        )
    alignment_diagnostics: dict[int, dict[str, float]] = {}
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        if i not in source_base or i not in source_ft:
            raise ValueError(f"Missing captured source reference for pairing index {i} at target position {j}")
        targets = target_by_position[j]
        source_base_batches = _aligned(source_base[i], targets)
        source_ft_batches = _aligned(source_ft[i], targets)
        s0_rows = _rows(source_base_batches).double()
        s1_rows = _rows(source_ft_batches).double()
        t0_rows = _rows(targets).double()
        q64, mu_s64, mu_t64 = centered_rectangular_procrustes(s0_rows, t0_rows)

        e = (s0_rows - mu_s64) @ q64 - (t0_rows - mu_t64)
        procrustes_error_norm = float(torch.linalg.norm(e))
        target_centered_norm = float(torch.linalg.norm(t0_rows - mu_t64))
        procrustes_relative_error = procrustes_error_norm / target_centered_norm if target_centered_norm > 0 else 0.0
        delta_target_norm = float(torch.linalg.norm((s1_rows - s0_rows) @ q64))
        endpoint_minus_delta_over_delta = (
            procrustes_error_norm / delta_target_norm if delta_target_norm > 0 else 0.0
        )
        # E @ Q^T @ Q rather than forming the d_t x d_t projector explicitly.
        e_in_range = (e @ q64.T) @ q64
        e_out_of_range = e - e_in_range
        procrustes_error_in_range_norm = float(torch.linalg.norm(e_in_range))
        procrustes_error_out_of_range_norm = float(torch.linalg.norm(e_out_of_range))
        mean_offset_norm = float(torch.linalg.norm(mu_s64 @ q64 - mu_t64))
        alignment_diagnostics[j] = {
            "procrustes_error_norm": procrustes_error_norm,
            "procrustes_relative_error": procrustes_relative_error,
            "delta_target_norm": delta_target_norm,
            "endpoint_minus_delta_over_delta": endpoint_minus_delta_over_delta,
            "procrustes_error_in_range_norm": procrustes_error_in_range_norm,
            "procrustes_error_out_of_range_norm": procrustes_error_out_of_range_norm,
            "mean_offset_norm": mean_offset_norm,
            "source_dim": int(s0_rows.shape[-1]),
            "target_dim": int(t0_rows.shape[-1]),
        }
    return alignment_diagnostics


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


# Below-epsilon ||D_j|| positions have no reliable ratio r_j = ||delta T_j|| /
# ||D_j||; per_block's guard keeps s_j frozen for that position/iteration
# rather than dividing by (near) zero.
_TV_SCALING_D_NORM_EPS = 1e-8


def _tau_frobenius_norm(sd: Mapping[str, Tensor]) -> float:
    total_sq = 0.0
    for value in sd.values():
        total_sq += float((value.detach().double() ** 2).sum().item())
    return total_sq**0.5


def _position_delta(
    shim, position: int, components: tuple[str, ...], target_corrections: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """The subset of ``target_corrections`` (weight + bias, unsliced) that
    belongs to block-boundary position ``position``.

    Each ``(position, component)`` pair maps to a distinct physical
    state-dict key (one set of projection matrices per block), so, unlike
    ``target_informed_runtime._family_delta_state`` (which slices packed
    q/k/v rows to isolate one COMPONENT out of several sharing one physical
    parameter), no row-slicing is needed here to isolate one POSITION: a
    key present in ``target_corrections`` belongs to exactly one position.
    """
    out: dict[str, Tensor] = {}
    for component in components:
        key = shim.component_key(position, component, prefixed=True)
        if key in target_corrections:
            out[key] = target_corrections[key]
        bias_key = _family_bias_key(key)
        if bias_key is not None and bias_key in target_corrections:
            out[bias_key] = target_corrections[bias_key]
    return out


def apply_tv_scaling(
    target_model,
    target_base_state: Mapping[str, Tensor],
    target_corrections: Mapping[str, Tensor],
    positions: list[int],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Label-free rescaling of a unit-strength Direct Residual task vector.

    Reuses ``measure_direct_residual_realization`` (mount, capture
    block-boundary outputs over the SAME calibration batches
    ``capture_paired_boundary_activations`` collected for the fit, compare to
    the pristine target base and to the block-boundary desired effect
    ``D_j``, restore-and-hash-verify) as the sole measurement primitive:
    ``r_j = joint_delta_norm_over_desired`` is ``||delta T_j||_F /
    (||D_j||_F + eps)`` for whatever delta is currently mounted.

    ``config.tv_scaling == "none"`` is a strict no-op (returns
    ``target_corrections`` unchanged, by value) -- callers must gate on this
    themselves for the golden-hash-pinned default path, but calling this
    function directly with ``tv_scaling="none"`` is also safe.

    Returns ``(scaled_corrections, diagnostics)``. ``diagnostics`` is
    intended to be recorded verbatim (or nested) under the Direct Residual
    run-summary section's additive ``tv_scaling_by_task`` key.
    """
    if config.tv_scaling == "none":
        return dict(target_corrections), {"mode": "none"}
    if config.block_split != "none":
        # parse_direct_residual_config should already have rejected this
        # combination; re-check here so a caller that builds a
        # DirectResidualConfig by hand (bypassing the parser) cannot reach
        # the unverified interaction either.
        raise ValueError(
            f"apply_tv_scaling: tv_scaling={config.tv_scaling!r} requires block_split='none' "
            f"(got block_split={config.block_split!r})"
        )
    shim = _layout_for(family_adapter)
    components = order_components(config.components)
    batches = captured["target_batches"]
    target_outputs_by_position = captured["target_base_outputs_by_position"]

    tau_stats_before = {
        "frobenius_norm": _tau_frobenius_norm(target_corrections),
        "sha256": _task_vector_sha256(target_corrections),
    }

    def measure(delta: Mapping[str, Tensor]) -> dict[int, dict[str, Any]]:
        return measure_direct_residual_realization(
            target_model,
            target_base_state,
            delta,
            positions,
            batches,
            target_outputs_by_position,
            desired,
            device=device,
            components=components,
            family_adapter=family_adapter,
        )

    if config.tv_scaling == "global":
        realization = measure(target_corrections)
        r_j = {j: float(realization[j]["joint_delta_norm_over_desired"]) for j in positions}
        c = float(median(r_j[j] for j in positions))
        if not math.isfinite(c) or c == 0.0:
            raise ValueError(
                f"tv_scaling='global' produced a degenerate scale c={c} "
                "(median of r_j across positions); cannot rescale tau"
            )
        final = {key: value / c for key, value in target_corrections.items()}
        post_realization = measure(final)
        diagnostics = {
            "mode": "global",
            "c": c,
            "r_j": r_j,
            "post_scaling_r_j": {
                j: float(post_realization[j]["joint_delta_norm_over_desired"]) for j in positions
            },
            "tau_stats_before": tau_stats_before,
            "tau_stats_after": {
                "frobenius_norm": _tau_frobenius_norm(final),
                "sha256": _task_vector_sha256(final),
            },
        }
        return final, diagnostics

    # config.tv_scaling == "per_block": per-position scalars s_j, found by a
    # simultaneous (Jacobi-style) fixed-point update -- every s_j is updated
    # from the SAME mounted-combination measurement sweep, never sequentially
    # (Gauss-Seidel) against a partially-updated combination.
    pos_delta = {j: _position_delta(shim, j, components, target_corrections) for j in positions}
    s = {j: 1.0 for j in positions}
    r_traces: list[dict[int, float | None]] = []
    s_traces: list[dict[int, float]] = []
    max_dev_trace: list[float] = []
    guard_log: list[dict[str, Any]] = []
    for iteration in range(int(config.tv_scaling_iters)):
        combined: dict[str, Tensor] = {}
        for j in positions:
            for key, value in pos_delta[j].items():
                combined[key] = value * s[j]
        realization = measure(combined)
        r_j: dict[int, float | None] = {}
        for j in positions:
            d_norm = float(realization[j]["desired_norm"])
            r = float(realization[j]["joint_delta_norm_over_desired"])
            if d_norm < _TV_SCALING_D_NORM_EPS or not math.isfinite(r):
                r_j[j] = None
                guard_log.append(
                    {
                        "iteration": iteration,
                        "position": j,
                        "reason": "desired_norm_near_zero" if d_norm < _TV_SCALING_D_NORM_EPS else "non_finite_r",
                        "desired_norm": d_norm,
                        "r_j": r,
                        "s_j_kept": s[j],
                    }
                )
            else:
                r_j[j] = r
        r_traces.append(dict(r_j))
        new_s = dict(s)
        for j in positions:
            if r_j[j] is not None:
                new_s[j] = s[j] / r_j[j]
        max_dev = max(abs((r_j[j] if r_j[j] is not None else 1.0) - 1.0) for j in positions)
        max_dev_trace.append(max_dev)
        s = new_s
        s_traces.append(dict(s))
    final = {}
    for j in positions:
        for key, value in pos_delta[j].items():
            final[key] = value * s[j]
    post_realization = measure(final)
    diagnostics = {
        "mode": "per_block",
        "iters": int(config.tv_scaling_iters),
        "r_traces": r_traces,
        "s_traces": s_traces,
        # max_j|r_j - 1| per iteration; "did it decrease vs the previous
        # iteration" is derivable from this trace directly (max_dev_trace[k]
        # < max_dev_trace[k-1]), reported as its own list rather than
        # collapsed to one bool so an oscillating trace is visible verbatim.
        "max_dev_trace": max_dev_trace,
        "guard_log": guard_log,
        "tau_stats_before": tau_stats_before,
        "tau_stats_after": {
            "frobenius_norm": _tau_frobenius_norm(final),
            "sha256": _task_vector_sha256(final),
        },
        "post_scaling_r_j": {j: float(post_realization[j]["joint_delta_norm_over_desired"]) for j in positions},
    }
    return final, diagnostics
