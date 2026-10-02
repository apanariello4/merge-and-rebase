"""BRACE block-extension configuration: dataclasses, protocol, resolver and coercers.

Pure configuration/topology layer. It imports nothing from ``merge_and_rebase.eval``;
``eval.block_extension`` re-exports every public and private name defined here.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from .completion_config import (
    JointCorrectionConfig,
    ResidualCompletionConfig,
    parse_joint_correction_config,
    parse_residual_completion_config,
)


@dataclass(frozen=True)
class TargetSharedCorrection:
    """Blend the pretrained target block's activations into the correction target.

    ARIADNE fits each inserted block to reproduce its *source* ancestor's
    component outputs. That objective is source-internal: a zero-projection
    block already solves it exactly, which is why the residual-identity
    baseline is competitive. This option instead moves part of the target
    toward the representation the pretrained target model holds at the
    corresponding depth, so the inserted block occupies the representational
    slot that width transport actually has to align to.

    ``target_weight`` is the blend strength eta. Zero reproduces standard
    ARIADNE exactly and bypasses every target-side capture.
    """

    target_weight: float
    component: str = "c_proj"
    added_blocks: str = "all"
    num_batches: int | None = None

    @property
    def active(self) -> bool:
        return self.target_weight > 0.0


@dataclass(frozen=True)
class BlockExtensionConfig:
    blocks_to_add: int | None = None
    target_layers_total: int | None = None
    insertion_order: str = "bottom-top"
    extension_density: str = "spread"
    # Reduction-direction only. "cascade" is the historical schedule, where a
    # later anchor may fall inside an already-merged span and absorb another
    # block into it; it is the default so completed shrink campaigns stay
    # reproducible. "disjoint_spans" partitions the depth up front so every
    # collapsed span is disjoint and evenly sized.
    collapse_schedule: str = "cascade"
    extension_strategy: str = "interpolate_per_weight"
    dampening_factor: float = 1.0
    n_batches_act: int = 2
    calibration_split: str = "test"
    calibration_dataset: str | dict[str, Any] | None = None
    # Backward-compatible/ergonomic alias for a named calibration dataset.
    calibration_task: str | None = None
    skip_correction: bool = False
    skip_final_ln: bool = False
    eval_before_extension: bool = False
    first_n_eval_batches: int | None = None
    ridge_identity: float = 0.0
    # Paper Eq. (9): coefficient for the stabilizing ||W||_F^2 penalty.
    ridge_weight: float = 1e-6
    n_cascade_iters: int = 1
    share_ft_refs: bool = False
    component_ridge: dict[str, float] | None = None
    # ``shared`` fits the correction at the pretrained/base endpoint and
    # applies it to FT; ``shared_ft`` does the converse.
    lmc_mode: str = "independent"
    # ``lazy`` recaptures only the references each structural step needs;
    # ``eager`` captures every block for both endpoints once and reuses it,
    # reproducing the pre-refactor capture schedule for A/B comparison.
    reference_capture: str = "lazy"
    # Depth baselines against the ARIADNE inserted block. ``residual_identity``
    # zeroes the inserted block's output projections, so the expanded model is
    # an exact function-preserving copy of the original and the extra depth
    # carries no computation of its own. ``residual_identity_inert`` goes one
    # step further and gives both endpoints the same inserted block, so the
    # inserted position's task vector is zero on every parameter rather than
    # only on the projections.
    inserted_block_mode: str = "ariadne"
    # Opt-in target-informed correction target for inserted blocks.
    target_shared_correction: TargetSharedCorrection | None = None
    # Opt-in target-informed residual completion (ARIADNE proposal 1) for
    # inserted blocks' c_proj projections. ``enabled=False`` (the default) is
    # the standard ARIADNE path and does not touch any target-informed code.
    target_residual_completion: ResidualCompletionConfig = field(default_factory=ResidualCompletionConfig)
    # Opt-in Option 3 frozen-map joint source/target blockwise solve.  The
    # default is disabled so historical ARIADNE and Proposal-1 runs are
    # byte-compatible and do not allocate the additional activation banks.
    joint_blockwise_correction: JointCorrectionConfig = field(default_factory=JointCorrectionConfig)
    # Two-pass direct hybrid: fit frozen transport first, then refine the
    # shared ARIADNE c_proj affine map with a P1 residual objective.
    direct_p1_correction: JointCorrectionConfig = field(default_factory=JointCorrectionConfig)
    # Which blocks receive a component correction. ``inserted`` is the paper's
    # scope and the default. ``interleaved_once`` additionally repairs the
    # original block immediately above each insertion, which over the
    # bottom-to-top schedule corrects every original block exactly once, always
    # against an input that is already final. ``iterative_all`` repairs every
    # original block above each insertion, re-correcting the same block once
    # per later insertion.
    correction_scope: str = "inserted"
    # ``interpolate_neighbors`` tells the downstream width-transport method
    # (Theseus/BiCo) to read the inserted position's source activations as the
    # midpoint of the two original blocks that initialized it, instead of
    # forwarding the inserted block itself.
    transport_activation_mode: str = "model"
    # Text path: what the inserted block's ``out_proj``/``c_proj`` ridge fit
    # targets. ``direct`` pins the component output; ``residual`` pins the
    # residual stream at the block boundary.
    insertion_target_mode: str = "direct"
    verbose: bool = True
    show_progress: bool = True


def block_extension_protocol(config: BlockExtensionConfig) -> dict[str, Any]:
    """Describe the structural/P1 semantics of a resolved BRACE config.

    ``residual_identity`` is a structural initialization, not a baseline once
    target residual completion is enabled: Proposal 1 can populate the
    inserted positions' task-vector projections after transport.  Keep this
    distinction explicit in run summaries and diagnostics so the exploratory
    arm cannot be mistaken for the untouched identity control.
    """
    proposal_1 = bool(config.target_residual_completion.enabled)
    layout_only_direct_target = bool(
        proposal_1 and config.target_residual_completion.mode == "direct_target" and config.skip_correction
    )
    option_3 = bool(config.joint_blockwise_correction.enabled)
    direct_p1 = bool(config.direct_p1_correction.enabled)
    if direct_p1:
        return {
            "label": "ariadne_direct_p1_shared_correction",
            "initialization": "ariadne",
            "proposal": "direct_p1_shared_correction",
            "is_baseline": False,
            "interpretation": "second shared c_proj affine pass with frozen transport and P1 residual objective",
        }
    if option_3:
        return {
            "label": "ariadne_plus_joint_blockwise_option3",
            "initialization": "ariadne",
            "proposal": "joint_blockwise_option3",
            "is_baseline": False,
            "interpretation": (
                "frozen-map one-alternation additive source/target c_proj solve; "
                "transport maps remain fixed and corrections are mounted sequentially"
            ),
        }
    if layout_only_direct_target:
        return {
            "label": "direct_target_layout_only",
            "initialization": "layout_only",
            "proposal": "target_residual_completion",
            "is_baseline": False,
            "interpretation": (
                "transport-free P1 with structural correction skipped: the realized "
                "extension/reduction layout addresses native source references, while "
                "the fitted target-space correction is the entire task vector"
            ),
        }
    if proposal_1 and config.inserted_block_mode == "residual_identity":
        return {
            "label": "residual_identity_plus_proposal_1",
            "initialization": "residual_identity",
            "proposal": "target_residual_completion",
            "is_baseline": False,
            "interpretation": "identity initialization + P1; exploratory ablation, not a baseline",
        }
    if config.inserted_block_mode == "residual_identity":
        return {
            "label": "residual_identity_baseline",
            "initialization": "residual_identity",
            "proposal": None,
            "is_baseline": True,
            "interpretation": "zero-output residual-identity initialization without P1",
        }
    if proposal_1:
        return {
            "label": "ariadne_plus_proposal_1",
            "initialization": "ariadne",
            "proposal": "target_residual_completion",
            "is_baseline": False,
            "interpretation": "ARIADNE initialization + P1",
        }
    return {
        "label": "ariadne",
        "initialization": "ariadne",
        "proposal": None,
        "is_baseline": False,
        "interpretation": "canonical ARIADNE correction",
    }


# Keys the campaign generators write into ``block_extension_params`` purely to
# record provenance in the run summary. They are not knobs and are not read
# here, so they must not trigger the unknown-key warning below.
_ANNOTATION_PARAMS: frozenset[str] = frozenset(
    {
        "calibration_protocol",
        "depth_rule",
        "inserted_block_mode",
        "reference_capture",
        "ridge_weight",
        "transport_activation_mode",
    }
)


def _warn_unknown_block_extension_params(params: Mapping[str, Any]) -> None:
    """Warn about params that are silently dropped.

    A misspelled knob (``lambda_l2`` for ``ridge_identity``, say) otherwise
    resolves to the default without a trace, which makes the run look like an
    ablation it is not. Warn rather than raise: existing campaign configs carry
    the annotation keys above and must keep resolving.
    """
    known = {f.name for f in fields(BlockExtensionConfig)} | _ANNOTATION_PARAMS
    unknown = sorted(k for k in params if k not in known)
    if unknown:
        warnings.warn(
            "Ignoring unrecognized block_extension_params "
            f"{unknown}; these have no effect on the run. "
            f"Known fields: {sorted(f.name for f in fields(BlockExtensionConfig))}.",
            RuntimeWarning,
            stacklevel=3,
        )



# ---------------------------------------------------------------------------------------------
# Depth-rule schema (P6.11). Schema only: the per-method defaults are applied by
# ``rebase.run_config.resolve_depth_rule``; nothing here changes ``BlockExtensionConfig``.
# ---------------------------------------------------------------------------------------------

DEPTH_RULES: tuple[str, ...] = ("method_default", "brace", "discrete_index_match")
# top-level legacy ``depth_alignment`` value -> depth rule
_DEPTH_ALIGNMENT_ALIAS: dict[str, str] = {"ariadne": "brace", "discrete_index_match": "discrete_index_match"}

# Fields that only the BRACE structural step (insertion/collapse + correction) reads. Under
# ``discrete_index_match`` they are inert; naming them in a warning beats silently ignoring them.
_BRACE_ONLY_FIELDS: tuple[str, ...] = (
    "blocks_to_add",
    "target_layers_total",
    "insertion_order",
    "extension_density",
    "collapse_schedule",
    "extension_strategy",
    "dampening_factor",
    "skip_correction",
    "skip_final_ln",
    "ridge_identity",
    "ridge_weight",
    "n_cascade_iters",
    "share_ft_refs",
    "component_ridge",
    "lmc_mode",
    "reference_capture",
    "inserted_block_mode",
    "correction_scope",
    "transport_activation_mode",
    "insertion_target_mode",
    "target_shared_correction",
)


@dataclass(frozen=True)
class DepthRuleSchema:
    """Parsed depth-rule keys of a run config (nothing method-specific is resolved here).

    ``rule`` is ``"method_default"`` when neither ``block_extension_params.depth_rule`` nor the
    legacy top-level ``depth_alignment`` names one. ``source`` records where an explicit rule came
    from (``"config"``, ``"alias:depth_alignment"``) or ``None``. ``extension_strategy`` and
    ``skip_correction`` are ``None`` when absent (-> per-method default).
    """

    rule: str = "method_default"
    extension_strategy: str | None = None
    skip_correction: bool | None = None
    source: str | None = None
    depth_alignment_given: bool = False
    depth_rule_given: bool = False


def _is_default_raw(name: str, value: Any) -> bool:
    if value is None:
        return True
    if name == "target_shared_correction":
        return isinstance(value, Mapping) and (not value or not bool(value.get("enabled", True)))
    default = {f.name: f.default for f in fields(BlockExtensionConfig)}.get(name, None)
    return value == default


def brace_only_fields_set(params: Mapping[str, Any]) -> list[str]:
    """Explicitly given BRACE-only fields in ``params`` whose value differs from the default."""
    return sorted(k for k in _BRACE_ONLY_FIELDS if k in params and not _is_default_raw(k, params[k]))


def warn_brace_only_fields_under_discrete(params: Mapping[str, Any], *, stacklevel: int = 3) -> list[str]:
    """RuntimeWarning (not an error: such configs were valid and inert before) listing inert fields."""
    inert = brace_only_fields_set(params)
    if inert:
        warnings.warn(
            f"depth_rule='discrete_index_match' ignores the BRACE-only block_extension_params {inert}; "
            "they have no effect on this run.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return inert


def parse_depth_rule_schema(cfg: Mapping[str, Any], *, warn: bool = True) -> DepthRuleSchema:
    """Parse ``block_extension_params.{depth_rule, extension_strategy, skip_correction}`` + the alias.

    The top-level ``depth_alignment`` is a legacy alias (``ariadne`` -> ``brace``,
    ``discrete_index_match`` -> itself). Giving both with different rules is a ``ValueError``.
    BRACE-only fields under ``discrete_index_match`` raise a ``RuntimeWarning`` (when ``warn``).
    """
    raw_params = cfg.get("block_extension_params") or {}
    if not isinstance(raw_params, Mapping):
        raise ValueError("config['block_extension_params'] must be a dict when provided.")

    alias_rule: str | None = None
    if "depth_alignment" in cfg:
        mode = str(cfg.get("depth_alignment", "ariadne")).strip().lower()
        if mode not in _DEPTH_ALIGNMENT_ALIAS:
            raise ValueError("depth_alignment must be one of: ariadne, discrete_index_match")
        alias_rule = _DEPTH_ALIGNMENT_ALIAS[mode]

    explicit_rule: str | None = None
    if "depth_rule" in raw_params:
        explicit_rule = str(raw_params["depth_rule"]).strip().lower()
        if explicit_rule not in DEPTH_RULES:
            raise ValueError(f"block_extension_params.depth_rule must be one of: {', '.join(DEPTH_RULES)}.")
    if explicit_rule is not None and alias_rule is not None and explicit_rule != "method_default":
        if explicit_rule != alias_rule:
            raise ValueError(
                f"Conflicting depth rules: depth_alignment={cfg.get('depth_alignment')!r} maps to "
                f"depth_rule={alias_rule!r} but block_extension_params.depth_rule={explicit_rule!r}. "
                "Give only one (depth_alignment is a legacy alias)."
            )

    if explicit_rule is not None and explicit_rule != "method_default":
        rule, source = explicit_rule, "config"
    elif alias_rule is not None:
        rule, source = alias_rule, "alias:depth_alignment"
    else:
        rule, source = "method_default", None

    strategy: str | None = None
    if raw_params.get("extension_strategy") is not None:
        # Value validity stays with the extender (its messages are pinned); only presence matters here.
        strategy = str(raw_params["extension_strategy"])
    skip = raw_params.get("skip_correction", None)
    if warn and rule == "discrete_index_match":
        warn_brace_only_fields_under_discrete(raw_params, stacklevel=4)
    return DepthRuleSchema(
        rule=rule,
        extension_strategy=strategy,
        skip_correction=None if skip is None else bool(skip),
        source=source,
        depth_alignment_given="depth_alignment" in cfg,
        depth_rule_given=explicit_rule is not None,
    )


_MISPLACED_TOP_LEVEL_KEYS = (
    "target_shared_correction",
    "target_residual_completion",
    "joint_blockwise_correction",
    "direct_p1_correction",
    "capture_target_residual_reference",
)


def resolve_block_extension_config(cfg: Mapping[str, Any]) -> tuple[bool, BlockExtensionConfig]:
    misplaced = [key for key in _MISPLACED_TOP_LEVEL_KEYS if key in cfg]
    if misplaced:
        raise ValueError(
            f"Found {misplaced} at the top level of the run config; a top-level key there has no "
            "effect on block extension. Move each one under the nested "
            f"'block_extension_params.<key>' location instead (e.g. "
            f"'block_extension_params.{misplaced[0]}')."
        )

    raw_params = cfg.get("block_extension_params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raise ValueError("config['block_extension_params'] must be a dict when provided.")

    params = dict(raw_params)
    _warn_unknown_block_extension_params(params)
    parse_depth_rule_schema(cfg)
    enabled_raw = cfg.get("block_extension_enabled", None)
    enabled = bool(enabled_raw) if enabled_raw is not None else bool(params)

    n_batches_act = int(params.get("n_batches_act", 2))
    if n_batches_act <= 0:
        raise ValueError("block_extension_params.n_batches_act must be > 0.")
    ridge_identity = float(params.get("ridge_identity", 0.0))
    ridge_weight = float(params.get("ridge_weight", 1e-6))
    if ridge_identity < 0.0:
        raise ValueError("block_extension_params.ridge_identity must be >= 0.")
    if ridge_weight < 0.0:
        raise ValueError("block_extension_params.ridge_weight must be >= 0.")

    skip_correction = bool(params.get("skip_correction", False))
    inserted_block_mode = _as_inserted_block_mode(params.get("inserted_block_mode", "ariadne"))
    correction_scope = _as_correction_scope(params.get("correction_scope", "inserted"))
    if correction_scope != "inserted":
        # Repairing the original blocks presupposes that a correction runs at
        # all, and that the inserted block is the ARIADNE one that disturbs the
        # residual stream. The identity baselines deliberately disturb nothing,
        # so there is no damage for an original-block pass to repair.
        if skip_correction:
            raise ValueError(
                f"block_extension_params.correction_scope='{correction_scope}' requires "
                "skip_correction=false: there is no correction to extend to the original blocks."
            )
        if inserted_block_mode != "ariadne":
            raise ValueError(
                f"block_extension_params.correction_scope='{correction_scope}' requires "
                "inserted_block_mode='ariadne': the identity baselines leave the residual "
                "stream untouched, so the original blocks have nothing to repair."
            )
    transport_activation_mode = _as_transport_activation_mode(params.get("transport_activation_mode", "model"))
    # Both depth baselines replace ARIADNE's component correction rather than
    # composing with it: a fitted correction would immediately undo an identity
    # block, and interpolated transport activations describe an uncorrected
    # inserted block. Refuse the ambiguous combination instead of silently
    # picking an order.
    if inserted_block_mode != "ariadne" and not skip_correction:
        raise ValueError(
            f"block_extension_params.inserted_block_mode='{inserted_block_mode}' requires "
            "skip_correction=true: fitting the ARIADNE correction on a zero-projection block "
            "destroys the identity it is testing."
        )
    if transport_activation_mode != "model" and not skip_correction:
        raise ValueError(
            "block_extension_params.transport_activation_mode='interpolate_neighbors' requires "
            "skip_correction=true: the baseline replaces the inserted block's activations, so a "
            "correction fitted against them would not be interpretable."
        )

    target_shared_correction = _as_target_shared_correction(params.get("target_shared_correction", None))
    if target_shared_correction is not None and target_shared_correction.active:
        # The blend changes what the *shared* correction is fitted to. Fitting it
        # independently per endpoint, or skipping correction entirely, would not
        # be the method this option describes.
        if skip_correction:
            raise ValueError(
                "block_extension_params.target_shared_correction requires skip_correction=false: "
                "it modifies the correction's regression target."
            )
        if str(params.get("lmc_mode", "independent")) != "shared":
            raise ValueError("block_extension_params.target_shared_correction requires lmc_mode='shared'.")
        if inserted_block_mode != "ariadne":
            raise ValueError(
                "block_extension_params.target_shared_correction fits a real inserted block and "
                f"is undefined for inserted_block_mode='{inserted_block_mode}'."
            )
        if target_shared_correction.num_batches is not None and target_shared_correction.num_batches != n_batches_act:
            raise ValueError(
                "block_extension_params.target_shared_correction.num_batches="
                f"{target_shared_correction.num_batches} must equal n_batches_act={n_batches_act}: "
                "the source-side c_proj reference bank (captured over n_batches_act calibration "
                "batches) and the target-side reference bank (captured over "
                "target_shared_correction.num_batches) must come from the same calibration images "
                "for the per-image Procrustes pairing to be formable."
            )

    target_residual_completion = parse_residual_completion_config(params.get("target_residual_completion", None))
    joint_blockwise_correction = parse_joint_correction_config(params.get("joint_blockwise_correction", None))
    direct_p1_correction = parse_joint_correction_config(params.get("direct_p1_correction", None))
    enabled_target_methods = sum(
        int(flag)
        for flag in (
            target_residual_completion.enabled,
            joint_blockwise_correction.enabled,
            direct_p1_correction.enabled,
        )
    )
    if enabled_target_methods > 1:
        raise ValueError(
            "target_residual_completion, joint_blockwise_correction, and direct_p1_correction are mutually exclusive; "
            "enable one target-informed correction protocol per run."
        )
    for option_name, option_enabled in (
        ("joint_blockwise_correction", joint_blockwise_correction.enabled),
        ("direct_p1_correction", direct_p1_correction.enabled),
    ):
        if not option_enabled:
            continue
        if target_shared_correction is not None and target_shared_correction.active:
            raise ValueError(
                f"{option_name} and target_shared_correction are mutually exclusive; "
                "change one target-informed factor at a time."
            )
        if skip_correction:
            raise ValueError(f"{option_name} requires skip_correction=false.")
        if str(params.get("lmc_mode", "independent")) != "shared":
            raise ValueError(f"{option_name} requires lmc_mode='shared'.")
        if inserted_block_mode != "ariadne":
            raise ValueError(f"{option_name} requires inserted_block_mode='ariadne'.")
        if correction_scope != "inserted":
            raise ValueError(f"{option_name} currently requires correction_scope='inserted'.")
        if transport_activation_mode != "model":
            raise ValueError(f"{option_name} requires transport_activation_mode='model'.")
        if str(params.get("insertion_target_mode", "direct")) != "direct":
            raise ValueError(f"{option_name} requires insertion_target_mode='direct'.")
        if str(params.get("insertion_order", "bottom-top")) != "bottom-top":
            raise ValueError(f"{option_name} requires insertion_order='bottom-top'.")
        if str(params.get("collapse_schedule", "cascade")) not in {"cascade", "disjoint_spans"}:
            raise ValueError("collapse_schedule must be 'cascade' or 'disjoint_spans'.")
        if str(params.get("extension_density", "spread")) not in {"spread", "spread_mod"}:
            raise ValueError(f"{option_name} requires spread extension density.")
        if str(params.get("extension_strategy", "interpolate_per_weight")) != "duplicate_per_weight":
            raise ValueError(f"{option_name} requires extension_strategy='duplicate_per_weight'.")
        if str(params.get("calibration_split", "test")) != "val":
            raise ValueError(f"{option_name} requires calibration_split='val'.")
        if params.get("calibration_dataset") is not None or params.get("calibration_task") is not None:
            raise ValueError(f"{option_name} requires task-local calibration data.")
        if bool(params.get("share_ft_refs", False)):
            raise ValueError(f"{option_name} requires share_ft_refs=false.")
    if target_residual_completion.enabled:
        # The ordinary arm completes a corrected ARIADNE insertion.  The one
        # exploratory exception is explicit residual_identity + P1: the
        # inserted block starts as an exact identity, and P1 is then allowed to
        # populate its transported c_proj task-vector components.  The inert
        # identity mode remains rejected because it intentionally removes the
        # inserted task vector everywhere, making P1's interpretation
        # ambiguous rather than a test of activation of empty depth.
        direct_layout_only = skip_correction and target_residual_completion.mode == "direct_target"
        identity_p1 = skip_correction and inserted_block_mode == "residual_identity" and not direct_layout_only
        if skip_correction and not (identity_p1 or direct_layout_only):
            raise ValueError(
                "block_extension_params.target_residual_completion with skip_correction=true "
                "requires mode='direct_target' (layout-only P1) or "
                "inserted_block_mode='residual_identity' (the identity initialization + P1 ablation)."
            )
        if not direct_layout_only and str(params.get("lmc_mode", "independent")) != "shared":
            raise ValueError("block_extension_params.target_residual_completion requires lmc_mode='shared'.")
        if not skip_correction and inserted_block_mode != "ariadne":
            raise ValueError(
                "block_extension_params.target_residual_completion fits a real inserted block and "
                f"is undefined for inserted_block_mode='{inserted_block_mode}'."
            )
        if identity_p1 and transport_activation_mode != "model":
            raise ValueError(
                "residual_identity + Proposal 1 requires transport_activation_mode='model': "
                "interpolated-neighbor activations would confound the identity initialization ablation."
            )
        if identity_p1 and str(params.get("insertion_target_mode", "direct")) != "direct":
            raise ValueError("residual_identity + Proposal 1 requires insertion_target_mode='direct'.")

    return enabled, BlockExtensionConfig(
        blocks_to_add=_as_optional_int(params.get("blocks_to_add", None)),
        target_layers_total=_as_optional_int(params.get("target_layers_total", None)),
        insertion_order=str(params.get("insertion_order", "bottom-top")),
        extension_density=str(params.get("extension_density", "spread")),
        collapse_schedule=str(params.get("collapse_schedule", "cascade")),
        extension_strategy=str(params.get("extension_strategy", "interpolate_per_weight")),
        dampening_factor=float(params.get("dampening_factor", 1.0)),
        n_batches_act=n_batches_act,
        calibration_split=str(params.get("calibration_split", "test")),
        calibration_dataset=_as_optional_calibration_dataset(params.get("calibration_dataset", None)),
        calibration_task=_as_optional_str(params.get("calibration_task", None)),
        skip_correction=skip_correction,
        skip_final_ln=bool(params.get("skip_final_ln", False)),
        eval_before_extension=bool(params.get("eval_before_extension", False)),
        first_n_eval_batches=_as_optional_int(params.get("first_n_eval_batches", None)),
        ridge_identity=ridge_identity,
        ridge_weight=ridge_weight,
        n_cascade_iters=max(1, int(params.get("n_cascade_iters", 1))),
        share_ft_refs=bool(params.get("share_ft_refs", False)),
        component_ridge=_as_optional_dict_float(params.get("component_ridge", None)),
        lmc_mode=str(params.get("lmc_mode", "independent")),
        reference_capture=_as_reference_capture(params.get("reference_capture", "lazy")),
        inserted_block_mode=inserted_block_mode,
        target_shared_correction=target_shared_correction,
        target_residual_completion=target_residual_completion,
        joint_blockwise_correction=joint_blockwise_correction,
        direct_p1_correction=direct_p1_correction,
        correction_scope=correction_scope,
        transport_activation_mode=transport_activation_mode,
        insertion_target_mode=str(params.get("insertion_target_mode", "direct")),
        verbose=bool(params.get("verbose", True)),
        show_progress=bool(params.get("show_progress", True)),
    )


def calibration_dataset_spec(config: BlockExtensionConfig) -> str | dict[str, Any] | None:
    """Return the configured task-independent calibration dataset.

    ``calibration_dataset`` is the canonical field. ``calibration_task`` is
    accepted as a shorthand for named suite tasks such as ``ImageNet1K``.
    ``None`` preserves the historical task-local calibration behavior.
    """
    if config.calibration_dataset is not None and config.calibration_task is not None:
        raise ValueError("Set only one of block_extension_params.calibration_dataset or calibration_task.")
    if config.calibration_dataset is not None:
        return config.calibration_dataset
    if config.calibration_task is not None:
        return {"task": config.calibration_task}
    return None


def select_loader(
    split: str, train_loader: Iterable[Any], test_loader: Iterable[Any], val_loader: Iterable[Any] | None
):
    if split == "train":
        return train_loader
    if split == "val" and val_loader is not None:
        return val_loader
    return test_loader


def _as_correction_scope(value: Any) -> str:
    resolved = str(value).strip().lower()
    if resolved not in {"inserted", "interleaved_once", "iterative_all"}:
        raise ValueError(
            "block_extension_params.correction_scope must be 'inserted', 'interleaved_once', or 'iterative_all'."
        )
    return resolved


def _as_target_shared_correction(value: Any) -> TargetSharedCorrection | None:
    """Parse the ``target_shared_correction`` config group.

    ``None`` and ``enabled: false`` both return ``None`` so that the standard
    ARIADNE path is reached without evaluating any target-side option.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("block_extension_params.target_shared_correction must be an object.")
    params = dict(value)
    if not bool(params.pop("enabled", True)):
        return None

    component = str(params.pop("component", "c_proj")).strip()
    if component != "c_proj":
        raise ValueError(
            "block_extension_params.target_shared_correction.component must be 'c_proj' "
            f"in this campaign; got {component!r}."
        )
    added_blocks = str(params.pop("added_blocks", "all")).strip()
    if added_blocks != "all":
        raise ValueError(
            f"block_extension_params.target_shared_correction.added_blocks must be 'all'; got {added_blocks!r}."
        )
    target_weight = float(params.pop("target_weight", 0.0))
    if target_weight < 0.0:
        raise ValueError("block_extension_params.target_shared_correction.target_weight must be >= 0.")
    num_batches = _as_optional_int(params.pop("num_batches", None))
    if num_batches is not None and num_batches <= 0:
        raise ValueError("block_extension_params.target_shared_correction.num_batches must be > 0.")
    if params:
        raise ValueError(f"Unknown block_extension_params.target_shared_correction keys: {sorted(params)}.")
    return TargetSharedCorrection(
        target_weight=target_weight,
        component=component,
        added_blocks=added_blocks,
        num_batches=num_batches,
    )


def _as_inserted_block_mode(value: Any) -> str:
    resolved = str(value).strip().lower()
    if resolved not in {"ariadne", "residual_identity", "residual_identity_inert"}:
        raise ValueError(
            "block_extension_params.inserted_block_mode must be 'ariadne', "
            "'residual_identity', or 'residual_identity_inert'."
        )
    return resolved


def _as_transport_activation_mode(value: Any) -> str:
    resolved = str(value).strip().lower()
    if resolved not in {"model", "interpolate_neighbors"}:
        raise ValueError("block_extension_params.transport_activation_mode must be 'model' or 'interpolate_neighbors'.")
    return resolved


def _as_reference_capture(value: Any) -> str:
    resolved = str(value).strip().lower()
    if resolved not in {"lazy", "eager"}:
        raise ValueError("block_extension_params.reference_capture must be 'lazy' or 'eager'.")
    return resolved


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip()
    return resolved or None


def _as_optional_calibration_dataset(value: Any) -> str | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        resolved = value.strip()
        return resolved or None
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    raise ValueError("block_extension_params.calibration_dataset must be a string or dict.")


def _as_optional_dict_float(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Expected a dict for component_ridge.")
    return {str(k): float(v) for k, v in value.items()}
