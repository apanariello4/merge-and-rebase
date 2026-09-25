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
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from statistics import median
from typing import Any

import torch

from ..rebase.discrete_layer_match import DiscreteLayerPairing, discrete_layer_pairing
from ..utils.cost_accounting import cost_phase_decorator
from .target_informed_runtime import (
    COMPONENT_INPUT_KIND,
    _aligned,
    _component_effective_out,
    _family_bias_key,
    _finalize_independent_component,
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
    iter_capture_block_gradients,
    iter_capture_tokens,
    measure_direct_residual_realization,
    measure_direct_residual_realization_streaming,
    paired_calibration,
)

__all__ = [
    "DirectResidualConfig",
    "apply_depth_pairing_override",
    "apply_tv_scaling",
    "capture_paired_boundary_activations",
    "compute_alignment_diagnostics",
    "compute_alignment_diagnostics_streaming",
    "compute_desired_effects",
    "compute_direct_residual_task_vector_stats",
    "compute_fidelity_holdout_diagnostics",
    "draw_fidelity_holdout_calibration",
    "fit_direct_residual",
    "fit_sequential_source_endpoints",
    "fit_direct_residual_streaming",
    "measure_direct_residual_realization",
    "measure_direct_residual_realization_streaming",
    "parse_direct_residual_config",
    "position_paired_only_contributions",
    "position_source_contributions",
    "prepare_direct_residual_streaming",
]
from .target_residual_completion import (
    COMPONENT_FORWARD_ORDER,
    INTERNAL_COMPONENTS,
    ResidualSufficientStatistics,
    _procrustes_from_cross,
    centered_rectangular_procrustes,
    centered_ridge_alignment,
    order_components,
)

Tensor = torch.Tensor


def apply_depth_pairing_override(pairing: DiscreteLayerPairing, depth_pairing: str) -> DiscreteLayerPairing:
    """Ablation: rewrite ``pairing.pairing`` per ``DirectResidualConfig.depth_pairing``.

    Called by the caller (``vision_rebase._direct_residual_fit_body`` and its
    ``merge_in_source_then_fit`` sibling) immediately after
    ``DiscreteLayerPairing.compute(source_depth, target_depth)``, before any
    capture. Every Direct Residual consumer -- ``capture_paired_boundary_
    activations``, ``compute_desired_effects``, ``fit_direct_residual``, and
    their streaming equivalents -- reads ``pairing.pairing[j]`` as the single
    source of truth for BOTH which source block's activations define
    ``D_j = (S_1 - S_0) Q_j`` and which source block ``Q_j`` itself aligns
    target position ``j`` with (``compute_desired_effects`` fits ``Q_j`` from
    ``source_base[pairing.pairing[j]]`` against ``target_base[j]``). So this
    one swap, applied once at construction, changes pi(j) consistently
    everywhere downstream without touching any of those call sites.

    ``depth_pairing="relative"`` returns ``pairing`` unchanged (identity,
    bit-for-bit -- the default, golden-hash-pinned path never calls this with
    anything else).  The other three modes derive a new pairing tuple from
    the ORIGINAL ``pairing.pairing`` (the closed-form relative pairing), not
    from each other:

      * ``"reversed"``:     ``pi_rev(j) = source_depth - 1 - pairing.pairing[j]``.
      * ``"shift_plus1"``:  ``min(source_depth - 1, pairing.pairing[j] + 1)``.
      * ``"shift_minus1"``: ``max(0, pairing.pairing[j] - 1)``.

    Only valid for ``component_target="block_boundary"`` -- validated by
    ``parse_direct_residual_config``, not here (this function has no config
    to check against and is usable standalone, e.g. by tests).
    """
    if depth_pairing == "relative":
        return pairing
    source_depth = pairing.source_depth
    if depth_pairing == "reversed":
        new_pairing = tuple(source_depth - 1 - i for i in pairing.pairing)
    elif depth_pairing == "shift_plus1":
        new_pairing = tuple(min(source_depth - 1, i + 1) for i in pairing.pairing)
    elif depth_pairing == "shift_minus1":
        new_pairing = tuple(max(0, i - 1) for i in pairing.pairing)
    else:
        raise ValueError(
            f"depth_pairing must be 'relative', 'reversed', 'shift_plus1' or 'shift_minus1', got {depth_pairing!r}"
        )
    return DiscreteLayerPairing(
        source_depth=pairing.source_depth, target_depth=pairing.target_depth, pairing=new_pairing
    )


def _derive_block_seed(alignment_seed: int, position: int) -> int:
    """Deterministic per-block seed for ``alignment_map='random_isometry'``.

    A plain ``alignment_seed + position`` would work too, but hashing keeps
    nearby positions' draws decorrelated (no shared low-order-bit structure
    across an entire depth sweep) and keeps the derivation obviously
    collision-free across the ``(alignment_seed, position)`` product space
    used across a sweep of many runs.
    """
    digest = hashlib.sha256(
        f"direct_residual_random_isometry:{int(alignment_seed)}:{int(position)}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _random_isometry_map(shape: tuple[int, int], *, seed: int) -> Tensor:
    """A random partial isometry of ``shape``, deterministic given ``seed``.

    Uses the exact same construction ``_procrustes_from_cross`` uses to turn
    a cross-covariance into the polar factor (SVD, then ``U @ Vh``) -- applied
    to a seeded standard-normal matrix instead of a cross-covariance -- so
    the result has the identical shape, orientation, and orthonormal-row/
    -column structure ``centered_rectangular_procrustes``'s polar map would
    have for the same (source, target) activation widths; only the direction
    it points in is randomized. Returned in float64, matching every other
    alignment map's fitting precision.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn(shape, generator=generator, dtype=torch.float64)
    return _procrustes_from_cross(gaussian)


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
    capture_source_ft_component_inputs: bool = False,
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

    ``component_inputs``, when non-empty (``component_target in
    {'output_local', 'output_total'}``), additionally captures every source
    block's own component input banks and both endpoints' weight/bias slices
    for the named components, via
    ``target_informed_runtime.capture_source_component_references``. Unlike
    the deduplicated ``distinct_source_indices`` above, this always spans
    ``range(pairing.source_depth)``: a shrink layout's span partition
    (``position_source_contributions``) can reference source blocks that are
    not any position's *closest* pairing match, so every source block's
    references have to exist regardless of direction. When
    ``component_inputs=()`` (the default), this is a strict no-op -- the
    returned dict is unchanged from before this parameter existed.

    ``capture_source_ft_component_inputs=True`` (``component_target=
    'output_total'`` only) additionally captures the source FT model's own
    component inputs ``X^{s1}`` for the same blocks/kinds, under
    ``"source_component_inputs_ft"``. It is a no-op unless ``component_inputs``
    is also non-empty, and the default (``False``) reproduces
    ``output_local``'s capture set exactly.
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
        source_component_inputs, source_component_weights, source_component_inputs_ft = (
            capture_source_component_references(
                source_base_model,
                source_ft_model,
                source_batches,
                list(range(pairing.source_depth)),
                component_inputs,
                device,
                family_adapter=family_adapter,
                capture_ft_inputs=capture_source_ft_component_inputs,
            )
        )
        result["source_component_inputs"] = source_component_inputs
        result["source_component_weights"] = source_component_weights
        if capture_source_ft_component_inputs:
            result["source_component_inputs_ft"] = source_component_inputs_ft
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
    # Fit a source-like base endpoint, then fit the FT endpoint on that
    # mounted target. The returned task vector is their target-space difference.
    endpoint_construction: str = "native_delta"
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
    #   "output_total"    -- component-specific target using each contributing
    #                       source block's FULL output change between its
    #                       fine-tuned and base endpoints, i.e.
    #                       A_c(X^{s1}_i, W^{ft}_i) - A_c(X^{s0}_i, W^{base}_i)
    #                       -- upstream-propagated input drift included, unlike
    #                       "output_local". Depth rule is deliberately matched
    #                       to "block_boundary" (every position uses only its
    #                       own paired source block, at weight 1.0), NOT to
    #                       "output_local"'s span/multiplicity-aware rule, so
    #                       that component_target is the only factor varied
    #                       against "block_boundary". See
    #                       ``position_paired_only_contributions``. Requires
    #                       LayerScale to be nn.Identity, same as
    #                       "output_local".
    #   Both "output_local" and "output_total" allow internal components
    #   (q/k/v/c_fc) in ``components``; "block_boundary" does not.
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
    # Activation-map family and calibration row weights. Defaults preserve
    # the historical centered polar factor exactly.
    alignment_map: str = "polar"
    alignment_row_weighting: str = "uniform"
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
    # Which images every Direct Residual fit collects its activations on.
    #   "task_local" (default) -- each per-task fit uses that task's own train
    #                             loaders; merge_in_source_then_fit uses the
    #                             first contributing task's. Bit-identical to
    #                             pre-field code.
    #   "tiny_imagenet"        -- one task-independent paired context
    #                             (zh-plus/tiny-imagenet, split "valid") shared
    #                             by every fit.
    #   "vision8_mix"          -- one exactly balanced Vision8 train context
    #                             (batch_size / 8 images per task per batch)
    #                             shared by every fit.
    # Only the fit's calibration changes: the per-task alpha search and the
    # evaluation stay on each task's own splits. Non-default values require
    # procrustes_source="activation" (gradient Procrustes needs task labels
    # and text features). The context is built in vision_rebase.py.
    calibration_data: str = "task_local"
    # Ablation: which source block index pi(j) each target position j is
    # paired with, overriding the DiscreteLayerPairing this module is handed
    # -- BOTH which source block's activations define D_j and which source
    # block Q_j aligns target position j with (see apply_depth_pairing_override;
    # threaded in by the caller at pairing-construction time, before capture).
    #   "relative"     (default) -- the pairing as computed, untouched
    #                   (DiscreteLayerPairing.compute's i(j) = round(j*(D_s-1)
    #                   /(D_t-1))). Bit-identical to pre-ablation code.
    #   "reversed"     -- pi_rev(j) = D_s - 1 - pi(j).
    #   "shift_plus1"  -- pi(j) + 1, clipped to [0, D_s - 1].
    #   "shift_minus1" -- pi(j) - 1, clipped to [0, D_s - 1].
    # Only valid with component_target="block_boundary" (see
    # apply_depth_pairing_override's docstring for why: the closed-form
    # position_source_contributions/position_paired_only_contributions span
    # rules the output_local/output_total targets rely on are keyed to the
    # untouched relative pairing).
    depth_pairing: str = "relative"
    # Ablation: alignment_map="random_isometry"'s per-position seed base.
    # Each position j draws its Gaussian generator from a seed derived
    # deterministically from (alignment_seed, j) -- see _derive_block_seed --
    # so the same seed reproduces the same random map for a given depth
    # pairing, and different positions never share a draw.
    alignment_seed: int = 0
    # Diagnostic (never read by any fit; see compute_fidelity_holdout_diagnostics).
    # A disjoint, unlabeled held-out slice of the SAME task-local calibration
    # split (drawn from the identical seeded permutation paired_calibration
    # uses for the fit, at the immediately-following, non-overlapping index
    # range) that measures, per target position, how well the fitted
    # (unit-strength) task vector reproduces D_j on images the fit never saw.
    fidelity_holdout: bool = False
    fidelity_holdout_batches: int = 10


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
        "endpoint_construction",
        "seed",
        "component_target",
        "realization_diagnostics",
        "block_split",
        "backfit_max_iters",
        "backfit_tol",
        "residual_target",
        "procrustes_source",
        "alignment_map",
        "alignment_row_weighting",
        "tv_scaling",
        "tv_scaling_iters",
        "activation_storage",
        "streaming_position_chunk",
        "calibration_data",
        "depth_pairing",
        "alignment_seed",
        "fidelity_holdout",
        "fidelity_holdout_batches",
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
    if cfg.ridge_estimator not in {"fixed_relative", "empirical_bayes", "none"}:
        raise ValueError("ridge_estimator must be 'fixed_relative', 'empirical_bayes' or 'none'")
    if isinstance(cfg.strength, bool) or not isinstance(cfg.strength, (int, float)):
        raise ValueError("strength must be a finite real number")
    if not math.isfinite(float(cfg.strength)):
        raise ValueError("strength must be finite")
    if cfg.strength < 0:
        raise ValueError("strength must be >= 0")
    if not isinstance(cfg.exact_form, bool):
        raise TypeError("exact_form must be bool")
    if cfg.component_target not in {"block_boundary", "output_local", "output_total"}:
        raise ValueError("component_target must be 'block_boundary', 'output_local' or 'output_total'")
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
    sequential_endpoint_modes = {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}
    if cfg.endpoint_construction not in {"native_delta", *sequential_endpoint_modes}:
        raise ValueError("endpoint_construction must be 'native_delta', 'sequential_source_endpoints' or 'sequential_delta_on_synthesized_base'")
    if cfg.endpoint_construction in sequential_endpoint_modes:
        if (tuple(cfg.components) != ("mlp.c_proj",) or cfg.component_target != "block_boundary"
                or cfg.block_split != "none" or cfg.procrustes_source != "activation"
                or cfg.tv_scaling != "none" or cfg.activation_storage != "resident"
                or cfg.realization_diagnostics or cfg.merge_mode != "per_task_then_merge"):
            raise ValueError("sequential endpoint construction requires D-only, block_boundary, no block split or TV scaling, activation Procrustes, resident storage, no realization diagnostics, and per-task fits")
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
    if cfg.alignment_map not in {"polar", "ridge", "random_isometry"}:
        raise ValueError("alignment_map must be 'polar', 'ridge' or 'random_isometry'")
    if cfg.alignment_row_weighting not in {"uniform", "cls_balanced", "delta_magnitude"}:
        raise ValueError("alignment_row_weighting must be 'uniform', 'cls_balanced' or 'delta_magnitude'")
    if (cfg.alignment_map != "polar" or cfg.alignment_row_weighting != "uniform") and (
        cfg.procrustes_source != "activation" or cfg.component_target != "block_boundary"
    ):
        raise ValueError("non-default alignment options require activation alignment and component_target='block_boundary'")
    if cfg.alignment_map in {"ridge", "random_isometry"} and cfg.alignment_row_weighting != "uniform":
        raise ValueError(f"alignment_map={cfg.alignment_map!r} requires alignment_row_weighting='uniform'")
    if isinstance(cfg.alignment_seed, bool) or not isinstance(cfg.alignment_seed, int):
        raise ValueError("alignment_seed must be an integer")
    if (cfg.alignment_map != "polar" or cfg.alignment_row_weighting != "uniform") and cfg.residual_target != "transported_delta":
        raise ValueError("non-default alignment options require residual_target='transported_delta'")
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
    if cfg.calibration_data not in {"task_local", "tiny_imagenet", "vision8_mix"}:
        raise ValueError("calibration_data must be 'task_local', 'tiny_imagenet' or 'vision8_mix'")
    if cfg.calibration_data != "task_local" and cfg.procrustes_source != "activation":
        raise ValueError(
            f"calibration_data={cfg.calibration_data!r} requires procrustes_source='activation': gradient "
            "Procrustes backpropagates the task's own labelled loss, which a task-independent calibration "
            "set does not provide"
        )
    if cfg.depth_pairing not in {"relative", "reversed", "shift_plus1", "shift_minus1"}:
        raise ValueError("depth_pairing must be 'relative', 'reversed', 'shift_plus1' or 'shift_minus1'")
    if cfg.depth_pairing != "relative" and cfg.component_target != "block_boundary":
        raise ValueError(
            f"depth_pairing={cfg.depth_pairing!r} requires component_target='block_boundary': the "
            "output_local/output_total span rules (position_source_contributions / "
            "position_paired_only_contributions) are keyed to the untouched relative pairing"
        )
    if not isinstance(cfg.fidelity_holdout, bool):
        raise TypeError("fidelity_holdout must be bool")
    if isinstance(cfg.fidelity_holdout_batches, bool) or not isinstance(cfg.fidelity_holdout_batches, int):
        raise ValueError("fidelity_holdout_batches must be a positive integer")
    if cfg.fidelity_holdout_batches <= 0:
        raise ValueError("fidelity_holdout_batches must be a positive integer")
    return cfg


@cost_phase_decorator("transformation")
def compute_desired_effects(
    captured: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    residual_target: str = "transported_delta",
    procrustes_source: str = "activation",
    alignment_map: str = "polar",
    alignment_row_weighting: str = "uniform",
    alignment_seed: int = 0,
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
    _validate_alignment_options(alignment_map, alignment_row_weighting, procrustes_source)
    if (alignment_map != "polar" or alignment_row_weighting != "uniform") and residual_target != "transported_delta":
        raise ValueError("non-default alignment options require residual_target='transported_delta'")
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
                map_distance = float(torch.linalg.norm(q_act - q) / (2.0 * d_min) ** 0.5)
                delta_act = torch.cat([(f - b).double() for b, f in zip(source_base_batches, source_ft_batches, strict=True)], 0) @ q_act
                delta_grad = torch.cat([(f - b).double() for b, f in zip(source_base_batches, source_ft_batches, strict=True)], 0) @ q
                delta_disagreement = float(torch.linalg.norm(delta_act - delta_grad) / (torch.linalg.norm(delta_act) + 1e-12))
                gs_centered = grad_source_rows - grad_source_rows.mean(dim=0)
                gt_centered = grad_target_rows - grad_target_rows.mean(dim=0)
                cross = gs_centered.T @ gt_centered
                diagnostics_out[j] = {
                    "procrustes_source": "gradient",
                    "procrustes_rank": int(torch.linalg.matrix_rank(cross)),
                    "activation_gradient_procrustes_overlap": overlap,
                    "activation_gradient_map_distance": map_distance,
                    "activation_gradient_delta_disagreement": delta_disagreement,
                    # See the activation branch below: kept for
                    # compute_fidelity_holdout_diagnostics to reuse the SAME
                    # fitted Q_j on held-out images without refitting.
                    "q": q.detach().float().clone(),
                }
            q = q.float()
            desired[j] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
            continue
        q, mu_s, mu_t, alignment_diag = _fit_activation_map(
            source_base_batches, targets, source_ft_batches,
            alignment_map=alignment_map, row_weighting=alignment_row_weighting,
            random_isometry_seed=_derive_block_seed(alignment_seed, j) if alignment_map == "random_isometry" else None,
        )
        if diagnostics_out is not None:
            diagnostics_out[j] = {
                "procrustes_source": "activation",
                "alignment_q_frobenius": float(torch.linalg.norm(q).item()),
                "alignment_q_rank": int(torch.linalg.matrix_rank(q)),
                # The fitted map/means themselves, kept for callers that need
                # to re-apply the SAME Q_j/mu without refitting (e.g.
                # compute_fidelity_holdout_diagnostics, which must evaluate
                # D_j on held-out images using the fit's own Q_j, never a
                # freshly refit one). Never read by any fit.
                "q": q.detach().float().clone(),
                "mu_s": mu_s.detach().float().clone(),
                "mu_t": mu_t.detach().float().clone(),
                **alignment_diag,
            }
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


def _validate_alignment_options(alignment_map, row_weighting, procrustes_source="activation"):
    if alignment_map not in {"polar", "ridge", "random_isometry"}:
        raise ValueError("alignment_map must be 'polar', 'ridge' or 'random_isometry'")
    if row_weighting not in {"uniform", "cls_balanced", "delta_magnitude"}:
        raise ValueError("alignment_row_weighting must be 'uniform', 'cls_balanced' or 'delta_magnitude'")
    if alignment_map != "polar" and row_weighting != "uniform":
        raise ValueError(f"alignment_map={alignment_map!r} requires uniform row weighting")
    if (alignment_map != "polar" or row_weighting != "uniform") and procrustes_source != "activation":
        raise ValueError("non-default alignment options require procrustes_source='activation'")


def _fit_activation_map(
    source_batches, target_batches, ft_batches, *,
    alignment_map="polar", row_weighting="uniform", random_isometry_seed: int | None = None,
):
    """Fit an activation map, optionally weighting each image's token rows."""
    xs, ys = [], []
    for x, y in zip(source_batches, target_batches, strict=True):
        xs.append(x.double())
        ys.append(y.double())
    # Keep the historical uniform polar call, including its exact reduction
    # order, so default configurations retain their golden hashes.
    if alignment_map == "polar" and row_weighting == "uniform":
        q, mx, my = centered_rectangular_procrustes(_rows(xs), _rows(ys))
        return q, mx, my, {"alignment_map": "polar", "alignment_row_weighting": "uniform"}
    if alignment_map == "random_isometry":
        # random_isometry always requires uniform row weighting (validated by
        # _validate_alignment_options), so the mean/centering is identical to
        # the plain uniform-polar path above -- only the map Q itself, whose
        # shape/orientation the fast path's centered_rectangular_procrustes
        # call also determines, is replaced by a random partial isometry of
        # that same shape.
        if random_isometry_seed is None:
            raise ValueError("alignment_map='random_isometry' requires random_isometry_seed")
        q_polar, mx, my = centered_rectangular_procrustes(_rows(xs), _rows(ys))
        q = _random_isometry_map(tuple(q_polar.shape), seed=random_isometry_seed)
        return q, mx, my, {
            "alignment_map": "random_isometry",
            "alignment_row_weighting": "uniform",
            "alignment_seed_used": int(random_isometry_seed),
        }
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    weights = torch.ones(x.shape[:2], dtype=torch.float64, device=x.device)
    if row_weighting == "cls_balanced":
        if x.shape[1] < 2:
            raise ValueError("cls_balanced alignment requires a CLS token and at least one patch token")
        weights[:, 0] = 0.5
        weights[:, 1:] = 0.5 / (x.shape[1] - 1)
    elif row_weighting == "delta_magnitude":
        if ft_batches is None:
            raise ValueError("delta_magnitude alignment requires source_ft_batches")
        ds = torch.cat([(f.double() - b.double()) for b, f in zip(source_batches, ft_batches, strict=True)], 0)
        norms = torch.linalg.vector_norm(ds, dim=-1)
        means = norms.mean(dim=1, keepdim=True)
        image_scale = torch.where(means > 0, (norms / means.clamp_min(torch.finfo(norms.dtype).tiny)).clamp(0.25, 4.0), torch.ones_like(norms))
        weights = image_scale
    # Each image receives equal total mass; within-image token weights sum to 1.
    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights / weights.sum()
    mu_x = (x * weights[..., None]).sum((0, 1))
    mu_y = (y * weights[..., None]).sum((0, 1))
    xc, yc = x - mu_x, y - mu_y
    xf, yf, wf = xc.reshape(-1, x.shape[-1]), yc.reshape(-1, y.shape[-1]), weights.reshape(-1)
    cross = xf.T @ (yf * wf[:, None])
    if alignment_map == "ridge":
        q, _, _, diag = centered_ridge_alignment(xf, yf)
        return q, mu_x, mu_y, {"alignment_map": "ridge", "alignment_row_weighting": "uniform", **diag}
    q = _procrustes_from_cross(cross)
    return q, mu_x, mu_y, {
        "alignment_map": "polar", "alignment_row_weighting": row_weighting,
        "weighted_cross_frobenius": float(torch.linalg.norm(cross).item()),
    }


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


def position_paired_only_contributions(pairing: DiscreteLayerPairing) -> dict[int, list[tuple[int, float]]]:
    """Per-target-position weighted source contributions for
    ``component_target='output_total'``.

    Deliberately different from ``position_source_contributions``
    (``output_local``'s span/multiplicity-aware rule): every position ``j``
    gets exactly one contribution, its own paired source block
    ``pairing.pairing[j]``, at weight 1.0 -- for every direction (extend,
    shrink, same_arch) and every component, matching ``block_boundary``'s own
    depth handling, where each position is fit against its paired block's
    full ``D_j`` alone. This is intentional, not an oversight:
    ``output_total``'s target already carries the block's full realized
    effect (including any upstream-propagated input drift, via ``X^{s1}``),
    so there is no local/global split left to represent through fractional
    weights or multi-source spans the way ``output_local``'s purely local
    target needs. Keeping this the ONLY behavioural difference from
    ``block_boundary``'s depth rule isolates ``component_target`` as the sole
    varied factor between a ``block_boundary`` and an ``output_total``
    ablation at the same pairing.
    """
    return {j: [(pairing.pairing[j], 1.0)] for j in range(pairing.target_depth)}


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
        if config.component_target in ("output_local", "output_total"):
            # Every component gets its own per-component, per-contribution
            # target -- not the shared block-boundary D_j the branch below
            # fits against -- so this is a genuinely different solve, not a
            # reuse of _fit_all_positions_independent.
            source_component_inputs = captured.get("source_component_inputs")
            source_component_weights = captured.get("source_component_weights")
            if source_component_inputs is None or source_component_weights is None:
                raise ValueError(
                    f"component_target={config.component_target!r} requires 'source_component_inputs' "
                    "and 'source_component_weights' in captured; call "
                    "capture_paired_boundary_activations with component_inputs set to the "
                    "requested components"
                )
            source_component_inputs_ft = None
            if config.component_target == "output_total":
                source_component_inputs_ft = captured.get("source_component_inputs_ft")
                if source_component_inputs_ft is None:
                    raise ValueError(
                        "component_target='output_total' requires 'source_component_inputs_ft' in "
                        "captured; call capture_paired_boundary_activations with component_inputs set "
                        "and capture_source_ft_component_inputs=True"
                    )
            paired_source_index = {j: pairing.pairing[j] for j in positions}
            contributions = (
                position_paired_only_contributions(pairing)
                if config.component_target == "output_total"
                else position_source_contributions(pairing)
            )
            fitted = _fit_component_outputs_from_contributions(
                target_model,
                current_state,
                positions,
                contributions,
                paired_source_index,
                source_component_inputs,
                source_component_weights,
                batches,
                components,
                solver_config,
                device,
                family_adapter=family_adapter,
                source_component_inputs_ft=source_component_inputs_ft,
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
            "source_update_norm": float(_rows([f - b for b, f in zip(mapped_base[j], mapped_ft[j], strict=True)]).double().norm()),
        }
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        base_correction, base_rows = fit_direct_residual(
            target_model, target_base_state, captured, base_desired, pairing,
            config=config, device=device, family_adapter=family_adapter,
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
            target_model, synthesized_base, ft_captured, ft_desired, pairing,
            config=config, device=device, family_adapter=family_adapter,
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
            "pretrained_correction_norm": float(sum(v.double().square().sum() for v in base_correction.values()).sqrt()),
            "task_vector_norm": float(sum(v.double().square().sum() for v in task_vector.values()).sqrt()),
            "max_endpoint_subtraction_error": max_endpoint_error,
        }
        return task_vector, base_rows + ft_rows, diagnostics
    finally:
        target_model.load_state_dict(entry_state, strict=True)


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


@cost_phase_decorator("transformation", exclusive=True)
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
    measure_fn=None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Label-free rescaling of a unit-strength Direct Residual task vector.

    ``measure_fn(delta) -> {j: realization row}`` replaces the resident measurement when
    given (the streaming path passes `measure_direct_residual_realization_streaming`
    bound to its own calibration sweep); ``captured``/``desired`` are then unused.

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

    tau_stats_before = {
        "frobenius_norm": _tau_frobenius_norm(target_corrections),
        "sha256": _task_vector_sha256(target_corrections),
    }

    def measure(delta: Mapping[str, Tensor]) -> dict[int, dict[str, Any]]:
        if measure_fn is not None:
            return measure_fn(delta)
        return measure_direct_residual_realization(
            target_model,
            target_base_state,
            delta,
            positions,
            captured["target_batches"],
            captured["target_base_outputs_by_position"],
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


class _StreamingCrossCovariance:
    """Chan's pairwise online accumulator for a centered cross-covariance.

    Equivalent to accumulating every row into one bank and computing
    ``(X - mean_x).T @ (Y - mean_y)`` directly (what
    `centered_rectangular_procrustes` does), but in O(1) batches rather than
    O(num_batches) host memory. Kept on ``device`` (float64) throughout;
    inputs are expected already cast to float64 by the caller.
    """

    def __init__(self, device=None, *, track_source_gram: bool = False) -> None:
        self.device = device
        self.track_source_gram = track_source_gram
        self.n = 0
        self.mean_x: Tensor | None = None
        self.mean_y: Tensor | None = None
        self.c: Tensor | None = None
        self.xx: Tensor | None = None
        self.weight = 0.0

    @cost_phase_decorator("transformation")
    def update(self, x: Tensor, y: Tensor, weights: Tensor | None = None) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError("cross-covariance update requires matching row counts")
        n_b = int(x.shape[0])
        if n_b == 0:
            return
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
        if weights is None:
            # Preserve historical uniform streaming arithmetic exactly.
            mean_x_b = x.mean(dim=0)
            mean_y_b = y.mean(dim=0)
            c_b = (x - mean_x_b).T @ (y - mean_y_b)
            if self.n == 0:
                self.n, self.weight, self.mean_x, self.mean_y, self.c = n_b, float(n_b), mean_x_b, mean_y_b, c_b
                if self.track_source_gram:
                    self.xx = (x - mean_x_b).T @ (x - mean_x_b)
                return
            n_a = self.n
            n = n_a + n_b
            dx = mean_x_b - self.mean_x
            dy = mean_y_b - self.mean_y
            self.c = self.c + c_b + (n_a * n_b / n) * torch.outer(dx, dy)
            if self.track_source_gram:
                xx_b = (x - mean_x_b).T @ (x - mean_x_b)
                self.xx = self.xx + xx_b + (n_a * n_b / n) * torch.outer(dx, dx)
            self.mean_x = self.mean_x + dx * (n_b / n)
            self.mean_y = self.mean_y + dy * (n_b / n)
            self.n = n
            self.weight = float(n)
            return
        w_b_rows = weights.to(device=x.device, dtype=x.dtype)
        if w_b_rows.shape != (n_b,) or (w_b_rows < 0).any() or not torch.isfinite(w_b_rows).all():
            raise ValueError("cross-covariance weights must be finite nonnegative row weights")
        w_b = float(w_b_rows.sum().item())
        if w_b <= 0:
            return
        mean_x_b = (x * w_b_rows[:, None]).sum(0) / w_b
        mean_y_b = (y * w_b_rows[:, None]).sum(0) / w_b
        c_b = (x - mean_x_b).T @ ((y - mean_y_b) * w_b_rows[:, None])
        xx_b = (x - mean_x_b).T @ ((x - mean_x_b) * w_b_rows[:, None]) if self.track_source_gram else None
        if self.weight == 0:
            self.n, self.weight, self.mean_x, self.mean_y, self.c, self.xx = n_b, w_b, mean_x_b, mean_y_b, c_b, xx_b
            return
        w_total = self.weight + w_b
        dx = mean_x_b - self.mean_x
        dy = mean_y_b - self.mean_y
        self.c = self.c + c_b + (self.weight * w_b / w_total) * torch.outer(dx, dy)
        if self.track_source_gram:
            self.xx = self.xx + xx_b + (self.weight * w_b / w_total) * torch.outer(dx, dx)
        self.mean_x = self.mean_x + dx * (w_b / w_total)
        self.mean_y = self.mean_y + dy * (w_b / w_total)
        self.weight = w_total
        self.n += n_b

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
    procrustes_source: str = "activation",
    source_recipe=None,
    target_recipe=None,
    source_ft_model=None,
    alignment_map: str = "polar",
    alignment_row_weighting: str = "uniform",
    alignment_seed: int = 0,
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

    ``procrustes_source="gradient"`` additionally runs the per-batch
    `iter_capture_block_gradients` generators (source base at the paired
    indices, target base at every position, each with its own recipe) in
    lockstep and fits ``Q_j`` on the Chan-accumulated centered GRADIENT
    cross-covariance, exactly the statistic the resident path fits it on. The
    activation accumulators always run: they provide the target fingerprints,
    the activation means ``mu_s``/``mu_t`` (``residual_target=
    "transported_endpoint"``) and the activation-space map used by the
    alignment diagnostics and the gradient-vs-activation overlap diagnostic.

    Returns a dict consumed by `fit_direct_residual_streaming`: the fitted
    per-position Procrustes maps (``"q_by_position"``), a per-(batch,
    position) determinism fingerprint of the target boundary activations
    (``"fingerprints"``) that Pass B uses to detect a target model mutated
    between passes, the activation-space maps and means, the replayed
    calibration batches for both sides, and the calibration metadata.
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if pairing.source_depth < 1:
        raise ValueError("pairing.source_depth must be positive")
    if procrustes_source not in {"activation", "gradient"}:
        raise ValueError("procrustes_source must be 'activation' or 'gradient'")
    _validate_alignment_options(alignment_map, alignment_row_weighting, procrustes_source)
    gradient_mode = procrustes_source == "gradient"
    if gradient_mode and (source_recipe is None or target_recipe is None):
        raise ValueError("procrustes_source='gradient' requires source_recipe and target_recipe")
    source_batches, target_batches, metadata = paired_calibration(
        source_loader, target_loader, num_batches=num_batches, seed=seed
    )
    distinct_source_indices = sorted(set(pairing.pairing))
    src_requests = {str(i): (i, "boundary") for i in distinct_source_indices}
    tgt_requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    accumulators = {
        j: _StreamingCrossCovariance(device=device, track_source_gram=alignment_map == "ridge")
        for j in range(pairing.target_depth)
    }
    grad_accumulators = (
        {j: _StreamingCrossCovariance(device=device) for j in range(pairing.target_depth)} if gradient_mode else {}
    )
    fingerprints: dict[tuple[int, int], tuple[float, float]] = {}

    if alignment_row_weighting == "delta_magnitude" and source_ft_model is None:
        raise ValueError("delta_magnitude streaming alignment requires source_ft_model")

    gens = {
        "src": iter_capture_tokens(
            source_base_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
        ),
        "tgt": iter_capture_tokens(
            target_base_model, target_batches, tgt_requests, device, family_adapter=family_adapter, store_device=device
        ),
    }
    if alignment_row_weighting == "delta_magnitude":
        gens["ft"] = iter_capture_tokens(
            source_ft_model, source_batches, src_requests, device, family_adapter=family_adapter, store_device=device
        )
    if gradient_mode:
        gens["src_grad"] = iter_capture_block_gradients(
            source_base_model,
            source_batches,
            {str(i): i for i in distinct_source_indices},
            source_recipe,
            device,
            family_adapter=family_adapter,
        )
        gens["tgt_grad"] = iter_capture_block_gradients(
            target_base_model,
            target_batches,
            {str(j): j for j in range(pairing.target_depth)},
            target_recipe,
            device,
            family_adapter=family_adapter,
        )
    names = list(gens)
    with contextlib.ExitStack() as stack:
        for gen in gens.values():
            stack.enter_context(contextlib.closing(gen))
        for k, values in enumerate(zip(*(gens[n] for n in names), strict=True)):
            by_name = dict(zip(names, values, strict=True))
            src, tgt = by_name["src"], by_name["tgt"]
            for j in range(pairing.target_depth):
                i = pairing.pairing[j]
                t = tgt[str(j)]
                x = _aligned([src[str(i)]], [t])[0]
                x_rows = x.reshape(-1, x.shape[-1]).double()
                y_rows = t.reshape(-1, t.shape[-1]).double()
                row_weights = None
                if alignment_row_weighting == "cls_balanced":
                    if x.shape[1] < 2:
                        raise ValueError("cls_balanced alignment requires a CLS token and at least one patch token")
                    row_weights = torch.full(x.shape[:2], 0.5 / (x.shape[1] - 1), dtype=torch.float64, device=x.device)
                    row_weights[:, 0] = 0.5
                elif alignment_row_weighting == "delta_magnitude":
                    sf = by_name["ft"][str(i)]
                    sf = _aligned([sf], [t])[0]
                    delta_norms = torch.linalg.vector_norm(sf.double() - x.double(), dim=-1)
                    means = delta_norms.mean(dim=1, keepdim=True)
                    scale = torch.where(
                        means > 0,
                        (delta_norms / means.clamp_min(torch.finfo(delta_norms.dtype).tiny)).clamp(0.25, 4.0),
                        torch.ones_like(delta_norms),
                    )
                    row_weights = scale / scale.sum(dim=1, keepdim=True)
                accumulators[j].update(x_rows, y_rows, None if row_weights is None else row_weights.reshape(-1))
                t64 = t.double()
                fingerprints[(k, j)] = (float(t64.sum().item()), float((t64**2).sum().item()))
                if gradient_mode:
                    g_t = by_name["tgt_grad"][str(j)]
                    g_s = _aligned([by_name["src_grad"][str(i)]], [g_t])[0]
                    grad_accumulators[j].update(
                        g_s.reshape(-1, g_s.shape[-1]).double(), g_t.reshape(-1, g_t.shape[-1]).double()
                    )

    def solve_map(acc, position):
        cross = acc.cross()
        if alignment_map == "polar":
            return _procrustes_from_cross(cross)
        if alignment_map == "random_isometry":
            # Same shape/orientation as the polar map above (both come from
            # `cross`'s shape); only the direction is randomized, from the
            # SAME per-block seed derivation the resident path uses, so
            # streaming and resident produce the identical random map for
            # the same alignment_seed (see test_direct_residual_random_
            # isometry.py's streaming-parity check).
            return _random_isometry_map(tuple(cross.shape), seed=_derive_block_seed(alignment_seed, position))
        assert acc.xx is not None
        trace = float(torch.trace(acc.xx).item())
        lam = trace / max(1, acc.n - 1)
        if trace == 0:
            return torch.zeros_like(cross)
        eye = torch.eye(acc.xx.shape[0], dtype=acc.xx.dtype, device=acc.xx.device)
        return torch.linalg.solve(acc.xx + lam * eye, cross)

    activation_q64 = {j: solve_map(acc, j).cpu() for j, acc in accumulators.items()}
    procrustes_diagnostics: dict[int, dict[str, Any]] = {}
    if gradient_mode:
        q_by_position = {}
        for j, acc in grad_accumulators.items():
            cross = acc.cross()
            q64 = _procrustes_from_cross(cross).cpu()
            q_by_position[j] = q64.float()
            d_min = min(q64.shape)
            procrustes_diagnostics[j] = {
                "procrustes_source": "gradient",
                "procrustes_rank": int(torch.linalg.matrix_rank(cross)),
                "activation_gradient_procrustes_overlap": float(((activation_q64[j].T @ q64).norm() ** 2) / d_min),
            }
    else:
        q_by_position = {j: q.float() for j, q in activation_q64.items()}
    return {
        "q_by_position": q_by_position,
        "procrustes_source": procrustes_source,
        "alignment_map": alignment_map,
        "alignment_row_weighting": alignment_row_weighting,
        "procrustes_diagnostics": procrustes_diagnostics,
        "activation_q64_by_position": activation_q64,
        "source_mean_by_position": {j: acc.mean_x.cpu() for j, acc in accumulators.items()},
        "target_mean_by_position": {j: acc.mean_y.cpu() for j, acc in accumulators.items()},
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
    residual_target = config.residual_target
    if residual_target == "transported_endpoint":
        if prepared.get("procrustes_source", "activation") != "activation":
            raise ValueError("residual_target='transported_endpoint' requires procrustes_source='activation'")
        mu_s_by_position = {j: m.float() for j, m in prepared["source_mean_by_position"].items()}
        mu_t_by_position = {j: m.float() for j, m in prepared["target_mean_by_position"].items()}
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
                        if residual_target == "transported_delta":
                            desired = (f - b) @ q
                        else:
                            desired = (f - mu_s_by_position[pos]) @ q + mu_t_by_position[pos] - out_cpu
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


def _streaming_source_iters(source_base_model, source_ft_model, prepared, pairing, device, family_adapter=None):
    """Fresh lockstep source generators (base + FT boundary at the paired indices) over the
    streaming path's replayed source calibration batches."""
    requests = {str(i): (i, "boundary") for i in sorted(set(pairing.pairing))}
    batches = prepared["source_batches"]
    return {
        "sb": iter_capture_tokens(
            source_base_model, batches, requests, device, family_adapter=family_adapter, store_device=device
        ),
        "sf": iter_capture_tokens(
            source_ft_model, batches, requests, device, family_adapter=family_adapter, store_device=device
        ),
    }


def _streaming_desired(prepared, pairing, residual_target, source_values, t0, positions):
    """Pass B's per-batch ``D_j`` (same arithmetic as `fit_direct_residual_streaming`)."""
    out: dict[int, Tensor] = {}
    for pos in positions:
        i = pairing.pairing[pos]
        out_cpu = t0[str(pos)].cpu()
        b = _aligned([source_values["sb"][str(i)].cpu()], [out_cpu])[0]
        f = _aligned([source_values["sf"][str(i)].cpu()], [out_cpu])[0]
        q = prepared["q_by_position"][pos]
        if residual_target == "transported_delta":
            out[pos] = (f - b) @ q
        else:
            mu_s = prepared["source_mean_by_position"][pos].float()
            mu_t = prepared["target_mean_by_position"][pos].float()
            out[pos] = (f - mu_s) @ q + mu_t - out_cpu
    return out


def measure_streaming_realization_for(
    target_model,
    target_base_state,
    source_base_model,
    source_ft_model,
    prepared,
    pairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
):
    """Bind `measure_direct_residual_realization_streaming` to one streaming run: returns
    ``measure(delta) -> {j: row}`` (the `apply_tv_scaling` ``measure_fn`` signature)."""
    positions = list(range(pairing.target_depth))
    components = order_components(config.components)

    def measure(delta):
        return measure_direct_residual_realization_streaming(
            target_model,
            target_base_state,
            delta,
            positions,
            prepared["target_batches"],
            lambda _k, source_values, t0: _streaming_desired(
                prepared, pairing, config.residual_target, source_values, t0, positions
            ),
            lambda: _streaming_source_iters(
                source_base_model, source_ft_model, prepared, pairing, device, family_adapter=family_adapter
            ),
            device=device,
            components=components,
            family_adapter=family_adapter,
        )

    return measure


@torch.no_grad()
def compute_alignment_diagnostics_streaming(
    source_base_model,
    source_ft_model,
    target_model,
    target_base_state: Mapping[str, Tensor],
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    device,
    family_adapter=None,
) -> dict[int, dict[str, float]]:
    """Streaming counterpart of `compute_alignment_diagnostics` (same keys, same meaning).

    One lockstep sweep (source base, source FT, pristine target) over the calibration
    batches, reusing Pass A's activation-space map and means (``Q_j``, ``mu_s``, ``mu_t``
    in float64), and accumulating every norm as a per-batch sum of squares: equal to the
    resident diagnostics up to floating-point summation order (and the Chan-accumulated
    Procrustes map). Analysis-only; like the resident function it must be called outside
    the timed brackets. The target model's entry state is restored.
    """
    positions = list(range(pairing.target_depth))
    q64 = prepared["activation_q64_by_position"]
    mu_s = prepared["source_mean_by_position"]
    mu_t = prepared["target_mean_by_position"]
    sums = {
        j: {"e": 0.0, "t": 0.0, "delta": 0.0, "in": 0.0, "out": 0.0, "src_dim": 0, "tgt_dim": 0} for j in positions
    }
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        requests = {str(j): (j, "boundary") for j in positions}
        gens = _streaming_source_iters(source_base_model, source_ft_model, prepared, pairing, device, family_adapter)
        gens["tgt"] = iter_capture_tokens(
            target_model, prepared["target_batches"], requests, device, family_adapter=family_adapter, store_device=device
        )
        names = list(gens)
        with contextlib.ExitStack() as stack:
            for gen in gens.values():
                stack.enter_context(contextlib.closing(gen))
            for values in zip(*(gens[n] for n in names), strict=True):
                by_name = dict(zip(names, values, strict=True))
                for j in positions:
                    i = pairing.pairing[j]
                    t_cpu = by_name["tgt"][str(j)].cpu()
                    b = _aligned([by_name["sb"][str(i)].cpu()], [t_cpu])[0]
                    f = _aligned([by_name["sf"][str(i)].cpu()], [t_cpu])[0]
                    s0 = b.reshape(-1, b.shape[-1]).double()
                    s1 = f.reshape(-1, f.shape[-1]).double()
                    t0 = t_cpu.reshape(-1, t_cpu.shape[-1]).double()
                    q = q64[j].double()
                    e = (s0 - mu_s[j]) @ q - (t0 - mu_t[j])
                    e_in = (e @ q.T) @ q
                    acc = sums[j]
                    acc["e"] += float((e**2).sum().item())
                    acc["t"] += float(((t0 - mu_t[j]) ** 2).sum().item())
                    acc["delta"] += float((((s1 - s0) @ q) ** 2).sum().item())
                    acc["in"] += float((e_in**2).sum().item())
                    acc["out"] += float(((e - e_in) ** 2).sum().item())
                    acc["src_dim"], acc["tgt_dim"] = int(s0.shape[-1]), int(t0.shape[-1])
    finally:
        target_model.load_state_dict(entry_state, strict=True)
    diagnostics: dict[int, dict[str, float]] = {}
    for j in positions:
        acc = sums[j]
        error_norm = acc["e"] ** 0.5
        target_norm = acc["t"] ** 0.5
        delta_norm = acc["delta"] ** 0.5
        diagnostics[j] = {
            "procrustes_error_norm": error_norm,
            "procrustes_relative_error": error_norm / target_norm if target_norm > 0 else 0.0,
            "delta_target_norm": delta_norm,
            "endpoint_minus_delta_over_delta": error_norm / delta_norm if delta_norm > 0 else 0.0,
            "procrustes_error_in_range_norm": acc["in"] ** 0.5,
            "procrustes_error_out_of_range_norm": acc["out"] ** 0.5,
            "mean_offset_norm": float(torch.linalg.norm(mu_s[j] @ q64[j].double() - mu_t[j])),
            "source_dim": acc["src_dim"],
            "target_dim": acc["tgt_dim"],
        }
    return diagnostics


def draw_fidelity_holdout_calibration(
    source_loader,
    target_loader,
    *,
    num_batches: int,
    holdout_batches: int,
    seed: int | None,
) -> tuple[list, list, dict[str, Any]]:
    """Draw a held-out batch set disjoint from the ``num_batches``-batch
    calibration set ``capture_paired_boundary_activations``/``prepare_direct_
    residual_streaming`` fit tau on, for ``DirectResidualConfig.fidelity_holdout``.

    Both sets are slices of the SAME seeded permutation `paired_calibration`
    draws (``num_batches`` requires a non-``None`` seed for this reason -- a
    ``None``-seeded, dataset-order calibration set has no "next" slice to draw
    a disjoint holdout from without risking overlap the dataset's own order
    could reintroduce): calling `paired_calibration` once with
    ``num_batches=num_batches + holdout_batches`` reproduces the identical
    leading ``num_batches`` slice `capture_paired_boundary_activations`/
    `prepare_direct_residual_streaming` already captured (same seed, same
    deterministic ``torch.randperm`` order), and the immediately-following
    ``holdout_batches`` slice is therefore guaranteed disjoint from it by
    construction -- verified explicitly below anyway, from the sample indices
    `paired_calibration` itself records, rather than merely assumed.

    Returns ``(holdout_source_batches, holdout_target_batches, holdout_metadata)``;
    ``holdout_metadata`` carries both slices' sample-index sha256 fingerprints.
    """
    if seed is None:
        raise ValueError("fidelity_holdout requires a deterministic (non-None) calibration seed")
    total_batches = int(num_batches) + int(holdout_batches)
    all_source, all_target, metadata = paired_calibration(
        source_loader, target_loader, num_batches=total_batches, seed=seed
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
    """``DirectResidualConfig.fidelity_holdout`` diagnostic: analysis-only,
    never read by any fit and never mutates ``target_corrections``.

    For BOTH the calibration split (the ``config.num_batches`` batches tau
    was fit on) and a disjoint held-out split
    (``draw_fidelity_holdout_calibration``, ``config.fidelity_holdout_batches``
    batches immediately following it in the same seeded permutation), computes
    per target position ``j`` (and, for ``e_local``, per fitted component):

      * ``e_local``  = ``||(H_j Delta_W_j + 1 beta_j^T) @ effective_out_j -
        D_j||_F / ||D_j||_F`` -- the component's OWN local linear-fit
        residual, ``H_j`` the base target's component-input activations at
        block ``j`` (``target_informed_runtime.COMPONENT_INPUT_KIND``),
        ``D_j`` recomputed on this split with the SAME fitted ``Q_j`` (and,
        for ``residual_target='transported_endpoint'``, the same ``mu_s``/
        ``mu_t``) passed in via ``q_by_position``/``mu_s_by_position``/
        ``mu_t_by_position`` -- never refit here.
      * ``e_mounted`` = ``block_realized_target_error`` from
        `measure_direct_residual_realization`, reused verbatim (mounts
        ``target_corrections`` at unit strength -- alpha=1, matching
        ``D_j`` being a unit-strength target -- runs the FULL nonlinear
        target forward pass, and compares the block-boundary delta to the
        same ``D_j``). With a single fitted component this is algebraically
        identical to that component's own ``e_local`` (no other component's
        effect to sum in); see
        ``tests/test_direct_residual_fidelity_holdout_20260925.py``.

    Every norm is reported alongside its ratio (``||D_j||`` and the raw
    numerator), not only the ratio, per the diagnostic's spec.
    """
    if not config.fidelity_holdout:
        raise ValueError("compute_fidelity_holdout_diagnostics called with fidelity_holdout=False")
    positions = list(range(pairing.target_depth))
    components = order_components(config.components)
    shim = _layout_for(family_adapter)
    distinct_source_indices = sorted(set(pairing.pairing))

    holdout_source, holdout_target, holdout_meta = draw_fidelity_holdout_calibration(
        source_loader, target_loader,
        num_batches=config.num_batches, holdout_batches=config.fidelity_holdout_batches, seed=config.seed,
    )
    calibration_source, calibration_target, _calib_meta = paired_calibration(
        source_loader, target_loader, num_batches=config.num_batches, seed=config.seed
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
        target_model.load_state_dict(
            {k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True
        )
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
                target_model, target_base_state, target_corrections, positions,
                tgt_batches, target_base_outputs_by_position, desired,
                device=device, components=components, family_adapter=family_adapter,
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
