"""Configuration dataclass and parser for Ariadne (formerly Direct Residual)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

#: Residual-writing projections, in the order a block executes them.
COMPONENT_FORWARD_ORDER: tuple[str, ...] = ("attn.out_proj", "mlp.c_proj")


#: Canonical block-forward order over all six names; the relative order of attn.out_proj and mlp.c_proj
#: matches COMPONENT_FORWARD_ORDER.
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

    The fit order is a property of the architecture, not of how the config listed them. Names outside
    ``CANONICAL_COMPONENT_ORDER`` are silently dropped.
    """
    selected = set(components)
    return tuple(name for name in CANONICAL_COMPONENT_ORDER if name in selected)


@dataclass(frozen=True)
class DirectResidualConfig:
    ridge_relative: float = 0.01
    # Read unconditionally by the shared solver (mirrors ResidualCompletionConfig); 'fixed_relative' = historical ridge.
    ridge_estimator: str = "fixed_relative"
    strength: float = 1.0
    num_batches: int = 10
    components: tuple[str, ...] = ("attn.out_proj", "mlp.c_proj")
    # Schema parity only: positions are fit independently (nothing cascades), so this orders nothing.
    cascade_order: str = "independent"
    exact_form: bool = True
    missing_bias: str = "error"
    # 'per_task_then_merge' (default): one correction per task, merged afterwards. 'merge_in_source_then_fit':
    # merge task deltas on the source base, fit once (orchestrated by the caller, vision_rebase.py).
    merge_mode: str = "per_task_then_merge"
    # 'native_delta' (default), or a 'sequential_*' mode: fit a source-like base endpoint, then the FT endpoint on it.
    endpoint_construction: str = "native_delta"
    seed: int = 89
    # Only 'block_boundary' is supported (every component fit against the same D_j); output_* options were retired.
    component_target: str = "block_boundary"
    # Analysis-only: adds diagnostic fields to the per-component rows; False keeps the historical row schema.
    realization_diagnostics: bool = False
    # Needs component_target='block_boundary' and components within {attn.out_proj, mlp.c_proj}.
    # 'none' (default): each component fit independently against the same D_j (golden-hash pinned).
    # 'backfit' (ablation, not used in current experiments): intra-block Gauss-Seidel, ablations._fit_block_boundary_backfit.
    # 'joint': closed-form joint ridge on the stacked features, assuming the MLP does not respond to attn.out_proj;
    #          equals 'none' for a single component (ablations._fit_block_boundary_joint).
    block_split: str = "none"
    backfit_max_iters: int = 20
    # Backfit stop: relative decrease of the safeguarded block objective J between sweeps (non-increasing by construction).
    backfit_tol: float = 1e-4
    # Target built from the centered Procrustes fit Q_j (S_{j,0} -> T_j^0):
    # 'transported_delta' (default): D_j = (S_1 - S_0) Q_j, golden-hash pinned.
    # 'transported_endpoint': D_j = (S_1 - mu_s) Q_j + mu_t - T_j^0; differs from the delta target by the Procrustes
    #     residual E_j (see compute_alignment_diagnostics). Needs block_boundary and procrustes_source='activation'.
    residual_target: str = "transported_delta"
    # Statistic Q_j is fit on (D_j = (S_ft - S_base) @ Q_j either way):
    # 'activation' (default): source/target base boundary activations, golden-hash pinned.
    # 'gradient': boundary gradients dL/dT of BiCo's zero-shot contrastive CE (capture_block_gradients);
    #     vision only, needs component_target='block_boundary'.
    procrustes_source: str = "activation"
    # Activation-map family and calibration row weights; defaults reproduce the historical centered polar factor.
    alignment_map: str = "polar"
    alignment_row_weighting: str = "uniform"
    # Label-free rescaling of tau after fit_direct_residual, before the caller's per-task alpha search (apply_tv_scaling):
    # 'none' (default, golden-hash pinned); 'global': tau / median_j(||dT_j|| / ||D_j||);
    # 'per_block': per-position scalars s_j from tv_scaling_iters simultaneous (Jacobi) rounds.
    # Rejected with block_split != 'none' (interaction with the measurement unverified).
    tv_scaling: str = "none"
    # Jacobi rounds for tv_scaling='per_block'; validated but unused otherwise.
    tv_scaling_iters: int = 3
    # 'resident' (default): full per-batch activation banks in host RAM, bit-identical to pre-ablation code.
    # 'streaming': sufficient statistics accumulated per batch (host RAM O(1) in num_batches); needs
    # component_target='block_boundary' and block_split='none'.
    activation_storage: str = "resident"
    # Streaming only: fit target positions in chunks of this size (one capture sweep each) to bound host RAM; None = one chunk.
    streaming_position_chunk: int | None = None
    # Images the fit collects activations on (alpha search and evaluation always use each task's own splits):
    # 'task_local' (default): each task's own train loaders (first contributing task's under merge_in_source_then_fit); bit-identical to pre-field code.
    # 'tiny_imagenet': one shared task-independent context (zh-plus/tiny-imagenet, "valid").
    # 'vision8_mix': one shared, exactly balanced Vision8 train context (batch_size / 8 per task per batch).
    # Non-default values require procrustes_source='activation'. The context is built in vision_rebase.py.
    calibration_data: str = "task_local"
    # Ablation overriding the source block pi(j) paired with each target position j (affects both D_j and Q_j;
    # apply_depth_pairing_override, applied at pairing-construction time). Needs component_target='block_boundary'.
    # 'relative' (default): pairing as computed, untouched (bit-identical to pre-ablation code). 'reversed': D_s - 1 - pi(j).
    # 'shift_plus1' / 'shift_minus1': pi(j) +/- 1, clipped to [0, D_s - 1].
    # 'spread_duplicate': pi(j) is the ancestor of target position j under BRACE's default schedule
    # (bottom-top, spread; shrink: the vision collapse, span -> first original index); see rebase.depth_pairing.
    depth_pairing: str = "relative"
    # alignment_map='random_isometry': per-position seed base; position j's generator is seeded from
    # (alignment_seed, j) via _derive_block_seed, so positions never share a draw and reruns reproduce.
    alignment_seed: int = 0
    # Diagnostic, never read by a fit (compute_fidelity_holdout_diagnostics): a disjoint, unlabeled held-out slice
    # of the same task-local calibration split (next index range of paired_calibration's seeded permutation)
    # measuring, per position, how well the unit-strength task vector reproduces D_j on unseen images.
    fidelity_holdout: bool = False
    fidelity_holdout_batches: int = 10
    # Streaming only: relative tolerance of the pass-A/pass-B target boundary fingerprint check (detects a target
    # model mutated between passes). 1e-9 = historical; loosen only if non-deterministic kernels trip it.
    streaming_fingerprint_tol: float = 1e-9
    # Ablation (default off): besides the fit, also copy source task-vector deltas whose shapes match the target's
    # (decoder/LLM runs; by default the Ariadne task vector is only its own fit). Serialized only when True.
    copy_shape_matching_source_deltas: bool = False


# Fields added after results were published: serialized only when non-default, so every historical summary,
# artifact and golden hash of a default run stays byte-identical.
_SERIALIZED_WHEN_NON_DEFAULT = {"streaming_fingerprint_tol": 1e-9, "copy_shape_matching_source_deltas": False}


def direct_residual_config_dict(cfg: DirectResidualConfig) -> dict[str, Any]:
    """`asdict(cfg)` minus `_SERIALIZED_WHEN_NON_DEFAULT` fields still at their default."""
    out = asdict(cfg)
    for name, default in _SERIALIZED_WHEN_NON_DEFAULT.items():
        if out.get(name) == default:
            del out[name]
    return out


def normalize_direct_residual_config_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Drop late-added fields that sit at their default, so full and compact serializations compare equal."""
    out = dict(raw)
    for name, default in _SERIALIZED_WHEN_NON_DEFAULT.items():
        if name in out and out[name] == default:
            del out[name]
    return out


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

    Explicit allowed-key set (unknown keys refused) and type/range checks, in the style of
    `target_residual_completion.parse_residual_completion_config`. Narrower than `ResidualCompletionConfig`:
    no `target_scope`, `mode`, `target_trajectory` or `direct_passthrough` (no realized-extension layout).
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
        "streaming_fingerprint_tol",
        "copy_shape_matching_source_deltas",
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
    if not isinstance(cfg.copy_shape_matching_source_deltas, bool):
        raise TypeError("copy_shape_matching_source_deltas must be bool")
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
    # Unlike parse_residual_completion_config (which needs 'mlp.c_proj' to anchor its cascade), nothing cascades
    # here (every position is fit independently against the pristine target base), so any non-empty,
    # repeat-free subset of COMPONENT_FORWARD_ORDER is legal, including {'attn.out_proj'} alone.
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
    if cfg.depth_pairing == "brace_ancestry":
        raise ValueError("depth_pairing 'brace_ancestry' was renamed to 'spread_duplicate'")
    if cfg.depth_pairing not in {"relative", "reversed", "shift_plus1", "shift_minus1", "spread_duplicate"}:
        raise ValueError(
            "depth_pairing must be 'relative', 'reversed', 'shift_plus1', 'shift_minus1' or 'spread_duplicate'"
        )
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
    tol = cfg.streaming_fingerprint_tol
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(float(tol)) or tol <= 0:
        raise ValueError("streaming_fingerprint_tol must be a positive finite number")
    return cfg


#: Defaults applied to decoder (LLM) runs when the key is absent (dataclass defaults are unchanged).
DECODER_DEFAULTS: dict[str, Any] = {
    "components": ("mlp.c_proj",),  # mapped to mlp.down_proj by the family adapter
    "activation_storage": "streaming",
    "ridge_estimator": "empirical_bayes",
    "missing_bias": "materialize",
    "exact_form": True,
}

#: (field, required value, why) for options that are vision-only.
_DECODER_REQUIRED: tuple[tuple[str, Any], ...] = (
    ("procrustes_source", "activation"),
    ("tv_scaling", "none"),
    ("fidelity_holdout", False),
    ("block_split", "none"),
    ("alignment_row_weighting", "uniform"),
    ("endpoint_construction", "native_delta"),
    ("calibration_data", "task_local"),
)


def resolve_ariadne_decoder_config(value: Mapping[str, Any] | None, family_adapter: Any) -> DirectResidualConfig:
    """Parse an Ariadne config for a decoder (LLM) run: fill decoder defaults for absent keys, reject vision-only options.

    ``family_adapter`` must be given (the family is what maps ``mlp.c_proj`` onto the decoder's down projection).
    """
    if family_adapter is None:
        raise ValueError("resolve_ariadne_decoder_config requires the model family adapter")
    raw = dict(value or {})
    preset = resolve_direct_residual_preset(raw)
    if preset is not None:
        raw.pop("preset")
        raw = {**_PRESETS[preset], **raw}
    cfg = parse_direct_residual_config({**DECODER_DEFAULTS, **raw})
    for name, required in _DECODER_REQUIRED:
        got = getattr(cfg, name)
        if got != required:
            raise ValueError(
                f"Ariadne on decoder models does not support {name}={got!r} (must be {required!r}): "
                "this option is vision-only"
            )
    return cfg
