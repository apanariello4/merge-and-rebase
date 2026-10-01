"""Configuration dataclass and parser for Ariadne (formerly Direct Residual)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Residual-writing projections, in the order a block executes them.
COMPONENT_FORWARD_ORDER: tuple[str, ...] = ("attn.out_proj", "mlp.c_proj")


#: Internal (non-residual-writing) components reachable only in output_* modes.
INTERNAL_COMPONENTS: tuple[str, ...] = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")


#: Canonical block-forward evaluation order across all six component names.
#: Relative order of attn.out_proj and mlp.c_proj is unchanged from
#: COMPONENT_FORWARD_ORDER, so any set drawn only from the historical two
#: names orders identically to before.
CANONICAL_COMPONENT_ORDER: tuple[str, ...] = (
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.out_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


def order_components(components) -> tuple[str, ...]:
    """Return ``components`` in block-forward order.

    The config names a *set* of write surfaces; the fit order is a property of
    the architecture, not of how the config happened to list them. Any name
    from ``CANONICAL_COMPONENT_ORDER`` (residual-writing or internal) is
    accepted; unknown names are silently dropped, matching the historical
    behaviour of filtering against a fixed order tuple.
    """
    selected = set(components)
    return tuple(name for name in CANONICAL_COMPONENT_ORDER if name in selected)


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
    # Which target each component's fit is asked to reproduce. Only
    # "block_boundary" is supported: every requested component (out_proj and
    # c_proj alike) is fit against the SAME block-boundary target D_j. The
    # former "output_local"/"output_total" options were retired (closed dead
    # ends) in the release cleanup and are rejected at parse time.
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
    #                ariadne.ablations._fit_block_boundary_backfit.
    #   "joint"   -- closed-form joint ridge over the stacked (attn.out_proj,
    #                mlp.c_proj) features, solved in one linear-algebra step
    #                under the first-order approximation that the MLP does
    #                not respond to a change in attn.out_proj. Only valid for
    #                components subset of {"attn.out_proj", "mlp.c_proj"};
    #                with a single component this reduces to (is literally
    #                the same fit as) block_split="none". See
    #                ariadne.ablations._fit_block_boundary_joint.
    block_split: str = "none"
    backfit_max_iters: int = 20
    # Stop when the relative decrease of the safeguarded block objective J(Delta)
    # (data-fit term plus each component's own round-1-frozen ridge penalty; see
    # ariadne.ablations._fit_block_boundary_backfit) between consecutive
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
    #                   ariadne.capture.capture_block_gradients and
    #                   capture_paired_boundary_activations's
    #                   procrustes_source parameter. Vision only. Only valid
    #                   with component_target="block_boundary".
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
    # component_target='block_boundary' and block_split='none' (see
    # parse_direct_residual_config); realization_diagnostics, tv_scaling and
    # fidelity_holdout are supported under streaming too.
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
    # apply_depth_pairing_override's docstring for why).
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


# Named bundles of config fields, selected with the optional ``"preset"`` key of the
# params mapping. A preset only sets the fields listed here (explicit keys in the same
# mapping override them); every other field keeps the dataclass default, and
# per-experiment budgets (``num_batches``, ``seed``) are never part of a preset.
_PRESETS: dict[str, dict[str, Any]] = {
    "ariadne": {
        "components": ("mlp.c_proj",),
        "activation_storage": "streaming",
        "ridge_estimator": "empirical_bayes",
    },
}


def resolve_direct_residual_preset(value: Mapping[str, Any] | None) -> str | None:
    """Return the validated ``"preset"`` name in a params mapping, or None when absent."""
    if value is None or not isinstance(value, Mapping) or "preset" not in value:
        return None
    name = value["preset"]
    if not isinstance(name, str) or name not in _PRESETS:
        raise ValueError(f"unknown direct_residual preset {name!r}; valid presets: {sorted(_PRESETS)}")
    return name


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
    preset = resolve_direct_residual_preset(value)
    if preset is not None:
        value = {k: v for k, v in value.items() if k != "preset"}
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
    payload = {**(_PRESETS[preset] if preset is not None else {}), **value}
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
    if cfg.component_target in {"output_local", "output_total"}:
        raise ValueError(
            f"component_target={cfg.component_target!r} was retired (closed dead end) in the release "
            "cleanup; only 'block_boundary' is supported"
        )
    if cfg.component_target != "block_boundary":
        raise ValueError("component_target must be 'block_boundary'")
    if not isinstance(cfg.realization_diagnostics, bool):
        raise TypeError("realization_diagnostics must be bool")
    components = cfg.components
    if isinstance(components, str) or not isinstance(components, (list, tuple)):
        raise ValueError("components must be a list of projection names")
    components = tuple(components)
    if not components:
        raise ValueError("components must not be empty")
    if len(set(components)) != len(components):
        raise ValueError("components must not repeat a projection")
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
        raise ValueError(
            "endpoint_construction must be 'native_delta', 'sequential_source_endpoints' or 'sequential_delta_on_synthesized_base'"
        )
    if cfg.endpoint_construction in sequential_endpoint_modes:
        if (
            tuple(cfg.components) != ("mlp.c_proj",)
            or cfg.component_target != "block_boundary"
            or cfg.block_split != "none"
            or cfg.procrustes_source != "activation"
            or cfg.tv_scaling != "none"
            or cfg.activation_storage != "resident"
            or cfg.realization_diagnostics
            or cfg.merge_mode != "per_task_then_merge"
        ):
            raise ValueError(
                "sequential endpoint construction requires D-only, block_boundary, no block split or TV scaling, activation Procrustes, resident storage, no realization diagnostics, and per-task fits"
            )
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
        raise ValueError(
            "non-default alignment options require activation alignment and component_target='block_boundary'"
        )
    if cfg.alignment_map in {"ridge", "random_isometry"} and cfg.alignment_row_weighting != "uniform":
        raise ValueError(f"alignment_map={cfg.alignment_map!r} requires alignment_row_weighting='uniform'")
    if isinstance(cfg.alignment_seed, bool) or not isinstance(cfg.alignment_seed, int):
        raise ValueError("alignment_seed must be an integer")
    if (
        cfg.alignment_map != "polar" or cfg.alignment_row_weighting != "uniform"
    ) and cfg.residual_target != "transported_delta":
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
            "the depth-pairing override is only defined for the block-boundary target"
        )
    if not isinstance(cfg.fidelity_holdout, bool):
        raise TypeError("fidelity_holdout must be bool")
    if isinstance(cfg.fidelity_holdout_batches, bool) or not isinstance(cfg.fidelity_holdout_batches, int):
        raise ValueError("fidelity_holdout_batches must be a positive integer")
    if cfg.fidelity_holdout_batches <= 0:
        raise ValueError("fidelity_holdout_batches must be a positive integer")
    return cfg
