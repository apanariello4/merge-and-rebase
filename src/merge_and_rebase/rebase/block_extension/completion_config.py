"""Validated configuration for target-residual completion and joint blockwise correction.

Lives in the ``rebase`` package so ``rebase.block_extension.config`` does not import
``merge_and_rebase.eval``; ``eval.target_residual_completion`` re-exports every name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..methods._ariadne.config import (
    COMPONENT_FORWARD_ORDER,
    INTERNAL_COMPONENTS,
    order_components,
)


@dataclass(frozen=True)
class ResidualCompletionConfig:
    enabled: bool = False
    added_blocks: str = "all"
    # ``inserted`` is the Proposal-1 protocol used by existing campaigns.
    # ``all`` is an opt-in ablation that completes every realized target block
    # position, including original descendants, using its source ancestry.
    target_scope: str = "inserted"
    component: str = "c_proj.weight"
    ridge_relative: float = 1e-3
    # ``fixed_relative`` preserves the historical trace-scaled ridge
    # ``ridge_relative * trace(S) / d_in``. ``empirical_bayes`` uses the
    # tuning-free trace ridge induced by the precision estimator of Wang et
    # al. (ICLR 2024), namely ``trace(S) / (N - 1)``; equivalently its
    # component-specific effective relative ridge is ``d_in / (N - 1)``.
    # ``none`` is the unregularized-ridge ablation (lambda = 0 exactly).
    ridge_estimator: str = "fixed_relative"
    # ``trace_normalized`` (default) scales whatever ``ridge_estimator``
    # produces by the mean centered feature and output-transport Gram traces,
    # same as always. ``absolute`` is a second, independent ablation: it
    # discards that trace-normalized value and uses one raw lambda for every
    # fitted block/task instead. It is intentionally opt-in (not scale
    # portable) and only meaningful alongside the default
    # ridge_estimator="fixed_relative" -- see the validation below, which
    # rejects pairing it with "empirical_bayes"/"none" rather than silently
    # discarding whichever estimator would otherwise be moot.
    ridge_mode: str = "trace_normalized"
    ridge_absolute: float | None = None
    strength: float = 1.0
    num_batches: int = 10
    # Exact affine fit (centered sufficient statistics + closed-form intercept,
    # ridge scaled by both the feature and transport Gram spectra) vs. the
    # historical reduced fit (no centering, no intercept, ridge scaled by the
    # feature Gram alone). The reduced form is valid only when the residual
    # (and source features) are zero-mean and t_out is exactly isometric; no
    # proposal-1 result has ever been produced under either setting, so there
    # is no historical default to preserve, and the exact form is the default.
    exact_form: bool = True
    # What to do when the residual-writing projection has no bias parameter to
    # receive the affine intercept. CLIP's mlp.c_proj always has one; HF decoder
    # MLPs are bias-free (Qwen2.5 sets mlp_bias=False), so the exact form has
    # nowhere to put it.
    #   "error"       -- refuse (the default, and vision's only reachable path)
    #   "materialize" -- add a zero bias to the target projection and write the
    #                    intercept there; exact, at the cost of a checkpoint
    #                    carrying a parameter stock Qwen does not have
    #   "skip"        -- drop the intercept, allowed ONLY when it is exactly
    #                    zero (exact_form=False). Dropping a fitted, nonzero
    #                    intercept is refused: W is fitted on centered banks, so
    #                    applying it without the intercept is not the same map.
    missing_bias: str = "error"
    # Which functional-transfer path Proposal 1 takes.
    #   "transport_residual" -- the historical path: complete the residual left
    #                           by an already-transported task vector, solving
    #                           in source coordinates through (t_in, t_out).
    #   "direct_target"      -- the transport-free ablation: start from the
    #                           native target base (no tau_t at all) and solve
    #                           for the desired effect directly in target
    #                           coordinates. Algebraically this is the same
    #                           affine ridge with t_in = I and t_out carrying
    #                           only the target's own LayerScale, so it reuses
    #                           the identical, tested solve rather than a
    #                           second implementation of it.
    mode: str = "transport_residual"
    # How a realized target position picks the source effect it is asked to
    # reproduce.
    #   "step"        -- every realized position takes its recorded ancestor's
    #                    full effect D = (B_i^1 - B_i^0) Q. Where one source
    #                    block is realized as several target blocks, each of
    #                    them is asked for the whole of that block's effect.
    #   "interpolate" -- blend neighbouring source effects by the position's
    #                    fractional depth in the realized chain, so a position
    #                    realizing "half of source block i" is asked for half
    #                    the step from i-1 to i. Agrees with "step" exactly at
    #                    the integer coordinates, i.e. at the last realized
    #                    member of every ancestry group.
    target_trajectory: str = "step"
    # Which residual-writing projections the completion is allowed to fit.  A
    # transformer block adds into the residual stream twice -- once from the
    # attention output projection, once from the MLP output projection -- and
    # the historical protocol corrected only the second.  Fitted in forward
    # order, cascaded (never as one joint linear system: out_proj's change
    # moves the MLP's own input through ln_2 and GELU, so the second fit has
    # to *measure* that rather than linearize it).
    components: tuple[str, ...] = ("mlp.c_proj",)
    # The order the sequential cascade visits realized target positions.
    #   "bottom_top" -- ascending position (the historical, default order). Each
    #                   block is fitted after every block upstream of it is
    #                   final, so its measured input H_j is the one the finished
    #                   model will actually feed it.
    #   "top_bottom" -- descending position. A block is then fitted before its
    #                   upstream neighbours change, so later fits invalidate the
    #                   input distribution earlier ones were fitted against.
    #   "independent" -- no cascade at all. Every block is fitted against the
    #                    pristine target base, so E_j == D_j for all j and the
    #                    corrections cannot interfere with each other's targets.
    #                    Order is irrelevant here by construction.
    # This is an ablation of how load-bearing the cascade coupling is: measured
    # `relative_residual_before` runs *above* 1.0 for most non-first blocks
    # (median ~1.06-1.08), i.e. upstream corrections leave downstream blocks a
    # harder target than the untouched base would, so the coupling is not
    # obviously helping and its direction is worth testing.
    cascade_order: str = "bottom_top"
    # Decoder/LLM path only. That path splits a task vector into a transportable
    # "body" and a ``passthrough`` remainder (embeddings, per-layer norms,
    # lm_head) which it folds in verbatim after completion. Vision has no such
    # split, so `direct_target` there is exactly ``theta_t^0 + gamma * dtau``.
    #   False (default) -- same contract on the decoder: the fitted correction
    #                      is the entire task vector, so gamma=0 is an exact
    #                      native-target-base control.
    #   True            -- also carry the shape-compatible passthrough keys, so
    #                      the arm differs from the transport arm only in how
    #                      the body is built. The arm is then not strictly
    #                      transport-free and gamma=0 no longer reproduces the
    #                      native base; both are recorded in the run summary.
    # For a width-changing pair (Qwen 0.5B->1.5B, 896->1536 hidden) almost every
    # passthrough key is shape-incompatible and already dropped, so this is
    # close to a no-op there; it bites on a same-width, depth-only rebase.
    direct_passthrough: bool = False


_DEFAULT_COMPONENTS: tuple[str, ...] = ("mlp.c_proj",)
_ALL_COMPONENT_NAMES: frozenset[str] = frozenset(COMPONENT_FORWARD_ORDER) | frozenset(INTERNAL_COMPONENTS)


def validate_residual_completion_depth_direction(
    config: ResidualCompletionConfig,
    *,
    source_depth: int,
    target_depth: int,
) -> None:
    """Fail early when direct-target shrink uses extension-only semantics.

    Reference capture happens before the realized reduction layout exists and
    can hold many gigabytes of activations.  Depth is already known at that
    point, so reject invalid shrink scopes and trajectories before doing that
    work.  The layout-level validator remains authoritative for span ancestry.
    """
    if not config.enabled or config.mode != "direct_target" or source_depth <= target_depth:
        return
    if config.target_scope != "all":
        raise ValueError(
            "Shrink direct_target completion requires target_scope='all': a reduction has no inserted blocks to address"
        )
    if config.target_trajectory != "step":
        raise ValueError(
            "Shrink direct_target completion currently requires target_trajectory='step'; "
            "interpolate has no span-aware reduction semantics"
        )


def parse_residual_completion_config(value: Mapping[str, Any] | None) -> ResidualCompletionConfig:
    """Parse and validate the narrow proposal-1 configuration schema."""
    if value is None:
        return ResidualCompletionConfig()
    if not isinstance(value, Mapping):
        raise TypeError("target_residual_completion must be a mapping")
    allowed = {
        "enabled",
        "added_blocks",
        "target_scope",
        "component",
        "ridge_relative",
        "ridge_estimator",
        "ridge_mode",
        "ridge_absolute",
        "strength",
        "num_batches",
        "exact_form",
        "missing_bias",
        "mode",
        "target_trajectory",
        "components",
        "cascade_order",
        "direct_passthrough",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown target_residual_completion fields: {sorted(unknown)}")
    payload = dict(value)
    # JSON gives a list; the dataclass is frozen and lands in asdict() output,
    # so normalize to a tuple up front rather than leaving two shapes around.
    if "components" in payload and isinstance(payload["components"], list):
        payload["components"] = tuple(payload["components"])
    cfg = ResidualCompletionConfig(**payload)
    if not isinstance(cfg.enabled, bool):
        raise TypeError("enabled must be bool")
    if cfg.added_blocks != "all":
        raise ValueError("added_blocks must be 'all' for the initial all-inserted-block protocol")
    if cfg.target_scope not in {"inserted", "all"}:
        raise ValueError("target_scope must be 'inserted' or 'all'")
    if cfg.component != "c_proj.weight":
        raise ValueError("component must be 'c_proj.weight'")
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
    if cfg.ridge_mode not in {"trace_normalized", "absolute"}:
        raise ValueError("ridge_mode must be 'trace_normalized' or 'absolute'")
    if cfg.ridge_mode == "absolute":
        if isinstance(cfg.ridge_absolute, bool) or not isinstance(cfg.ridge_absolute, (int, float)):
            raise ValueError("ridge_absolute must be a finite real number when ridge_mode='absolute'")
        if not math.isfinite(float(cfg.ridge_absolute)) or float(cfg.ridge_absolute) <= 0:
            raise ValueError("ridge_absolute must be finite and > 0 when ridge_mode='absolute'")
        if cfg.ridge_estimator != "fixed_relative":
            raise ValueError("ridge_mode='absolute' is only valid with the default ridge_estimator='fixed_relative'")
    elif cfg.ridge_absolute is not None:
        raise ValueError("ridge_absolute is only valid when ridge_mode='absolute'")
    if isinstance(cfg.strength, bool) or not isinstance(cfg.strength, (int, float)):
        raise ValueError("strength must be a finite real number")
    if not math.isfinite(float(cfg.strength)):
        raise ValueError("strength must be finite")
    if cfg.strength < 0:
        raise ValueError("strength must be >= 0")
    if not isinstance(cfg.exact_form, bool):
        raise TypeError("exact_form must be bool")
    if cfg.mode not in {"transport_residual", "direct_target"}:
        raise ValueError("mode must be 'transport_residual' or 'direct_target'")
    if cfg.target_trajectory not in {"step", "interpolate"}:
        raise ValueError("target_trajectory must be 'step' or 'interpolate'")
    if cfg.cascade_order not in {"bottom_top", "top_bottom", "independent"}:
        raise ValueError("cascade_order must be 'bottom_top', 'top_bottom' or 'independent'")
    if cfg.target_trajectory == "interpolate":
        # Both new write surfaces are direct-mode features; the transport arm's
        # solve is defined against fitted (t_in, t_out) maps and is deliberately
        # left exactly as it was.
        if cfg.mode != "direct_target":
            raise ValueError("target_trajectory='interpolate' requires mode='direct_target'")
        if cfg.target_scope != "all":
            # The inserted-only path consumes the precomputed references['desired']
            # banks, which are built without ancestry-group structure, so there is
            # no fractional depth to interpolate along. Refused rather than
            # silently stepping.
            raise ValueError("target_trajectory='interpolate' requires target_scope='all'")
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
    if "mlp.c_proj" not in components:
        # The MLP projection is the last write into the residual stream and the
        # only one whose solver objective is exactly the post-mount residual;
        # dropping it would leave the fit unanchored.
        raise ValueError("components must contain 'mlp.c_proj'")
    if order_components(components) != _DEFAULT_COMPONENTS and cfg.mode != "direct_target":
        raise ValueError(
            "components other than ['mlp.c_proj'] require mode='direct_target': the transport "
            "arm would need fitted t_in/t_out maps for attn.out_proj, which are not produced"
        )
    if not isinstance(cfg.direct_passthrough, bool):
        raise TypeError("direct_passthrough must be bool")
    if cfg.direct_passthrough and cfg.mode != "direct_target":
        raise ValueError("direct_passthrough=true requires mode='direct_target'")
    if cfg.missing_bias not in {"error", "materialize", "skip"}:
        raise ValueError("missing_bias must be 'error', 'materialize' or 'skip'")
    if cfg.missing_bias == "skip" and cfg.exact_form:
        # The exact form fits a centered map and recovers a nonzero intercept;
        # applying its weight without that intercept is a different map, not an
        # approximation of it. Refused up front rather than at the write.
        raise ValueError(
            "missing_bias='skip' requires exact_form=false: the exact form fits a "
            "nonzero intercept, and dropping it would apply a centered-fit weight "
            "without the centering it assumes"
        )
    return cfg


@dataclass(frozen=True)
class JointCorrectionConfig:
    """Configuration for the frozen-map Option 3 blockwise correction.

    ``source_weight`` weights the ordinary source ARIADNE reconstruction
    objective and ``target_weight`` weights the target-space transported-effect
    objective.  The transport maps are deliberately not configurable here:
    Option 3 fits them in a preceding, ordinary transport pass and freezes
    them for this solve.  ``enabled=False`` is the compatibility default and
    has no effect until a caller explicitly opts into the joint path.
    """

    enabled: bool = False
    source_weight: float = 1.0
    target_weight: float = 1.0
    ridge_relative: float = 1e-3


def parse_joint_correction_config(value: Mapping[str, Any] | None) -> JointCorrectionConfig:
    """Parse the narrow, explicit Option 3 configuration schema.

    This parser intentionally rejects transport/seeding controls.  Those
    belong to the transport method and changing them during the joint solve
    would violate the frozen-map protocol.
    """
    if value is None:
        return JointCorrectionConfig()
    if not isinstance(value, Mapping):
        raise TypeError("joint_blockwise_correction must be a mapping")
    allowed = {"enabled", "source_weight", "target_weight", "ridge_relative"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown joint_blockwise_correction fields: {sorted(unknown)}")
    cfg = JointCorrectionConfig(**dict(value))
    if not isinstance(cfg.enabled, bool):
        raise TypeError("enabled must be bool")
    for name in ("source_weight", "target_weight", "ridge_relative"):
        number = getattr(cfg, name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise ValueError(f"{name} must be a finite real number")
        if name == "ridge_relative" and number <= 0:
            raise ValueError("ridge_relative must be > 0")
        if name != "ridge_relative" and number < 0:
            raise ValueError(f"{name} must be >= 0")
    if cfg.source_weight == 0 and cfg.target_weight == 0:
        raise ValueError("at least one joint objective weight must be positive")
    return cfg
