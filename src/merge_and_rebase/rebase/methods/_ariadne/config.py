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
    # Ablation: how blocks are fitted across depth. 'independent' (default): every block fitted against the pristine
    # target base (golden-hash pinned). 'sequential': blocks fitted in order, block j against the target already
    # carrying the fitted updates of blocks < j (residual target D_j - U_j); streaming path only (see
    # streaming.fit_direct_residual_sequential_streaming). Serialized only when non-default.
    depth_fit: str = "independent"
    # How depth_fit='sequential' is computed (same estimator, same fitted task vector). 'reference' (default): the
    # original streaming code, four full forwards per position and batch. 'block_carry' (decoders only): every model
    # stream carries its exact block input forward, so each position runs only its own block (see
    # sequential_carry.py). Serialized only when non-default.
    sequential_impl: str = "reference"
    # Ablation (depth_fit='sequential', sequential_impl='reference', components attn.out_proj + mlp.c_proj): how many
    # correction points each block gets. 'single' (default): the one component fit as before. 'end_end': O
    # (attn.out_proj) is fitted first, then D (mlp.c_proj), both against the block-end target D_j (D fits what O left
    # missing at the block end). 'mid_end': O is fitted against a mid-block target M_j (residual stream after the
    # attention add) through its own polar Procrustes map, D against D_j (vision only). Serialized only when non-default.
    sequential_points: str = "single"
    # Ablation (polar map, procrustes_source='activation', streaming only; mutually exclusive, default None = full
    # map): rank-truncate the Procrustes map Q_j = U V^T to its top-k singular pairs, Q_j^k = U[:, :k] Vh[:k].
    # 'procrustes_rank': fixed k >= 1 (k >= min(d_source, d_target) is the full map). 'procrustes_energy': in (0, 1],
    # per block the smallest k with sum(s[:k]) / sum(s) >= procrustes_energy, s the PLAIN singular values of the
    # centered cross-covariance (NOT squared); 1.0 is the full map. Serialized only when set.
    procrustes_rank: int | None = None
    procrustes_energy: float | None = None
    # Decoder-only ablation (default off): besides padding, drop the first non-padded position of every sequence (the
    # attention sink, massive activations) from every statistic the fit consumes, for source and target alike.
    # Serialized only when True; rejected on the vision path (reject_decoder_only_options).
    exclude_first_token: bool = False
    # Diagnostic (vision only): with residual_target='native_oracle' the target is D*_j = T^ft_j - T^base_j, the block-j
    # output change of the NATIVE fine-tuned target model (own full forward) against the target base on the same target
    # calibration images; no Procrustes map is used. 'native_target_ckpt_template' ("...{task}...") names that checkpoint
    # (same loader as eval.diagnostics.ariadne_norm.readout). Streaming, depth_fit='independent', per-task fits only.
    native_target_ckpt_template: str | None = None
    # Ablation: target blocks to fit (sorted unique indices, None = all); every other block gets no update. Works for
    # depth_fit='independent' and depth_fit='sequential' (sequential_impl='reference', sequential_points='single'): in a
    # sequential fit an unfitted block is skipped (nothing mounted), later blocks still see the true drift U_j.
    fit_blocks: tuple[int, ...] | None = None
    # Ablation for extend (several target blocks share one source block pi(j)); target groups are the maximal runs of
    # consecutive target blocks with the same pi(j) (size 1 under shrink / same depth: no-op). 'none' (default).
    # 'per_block_share' (depth_fit='independent'): block j of a group of size m is fitted to D_j / m (each block's change
    # persists in the residual stream, so the group delivers D once, not m times). 'interpolate' (depth_fit='sequential',
    # reference, single point): the i-th block (i = 0..m-1) of a group starting at block j0 is fitted to
    # D_prev + (i + 1) / m * (D_j - D_prev), D_prev the unsplit transported change of block j0 - 1 (0 for j0 = 0), so the last
    # block of a group keeps D_j; the sequential error stays desired - U_j.
    pair_target_split: str = "none"
    # Ablation (vision only, depth_fit='independent', block_boundary, block_split='none'): how a component is fitted.
    # 'closed_form' (default): the ridge solve on the component's input features. 'gradient_exact': minimise
    # sum ||T_j(W + dW; x_j) - T_j(W; x_j) - D_j||^2 + lam ||dW||_F^2 (bias unpenalised) with the EXACT block forward T_j by
    # full-batch L-BFGS from 0. 'gradient_linearized': the same with T_j replaced by its first-order expansion (a convex
    # ridge problem, conjugate gradients on the normal equations, J never formed). Component groups (one per run):
    # attn.v_proj; attn.q_proj + attn.k_proj; attn.q_proj + attn.k_proj + attn.v_proj; mlp.c_fc; attn.out_proj; mlp.c_proj.
    # For mlp.c_proj both gradient arms equal the closed form; for attn.out_proj they include the MLP's reaction downstream.
    component_fit: str = "closed_form"
    gradient_fit_max_iters: int = 500
    # Stop when the relative decrease of the (scaled) objective between iterations is below this.
    gradient_fit_tol: float = 1e-9
    # 'eb_input': lam = empirical-Bayes value of the component's own centred input features (as the closed form).
    # 'match_norm': per block, lam found by bisection on log lam so ||dW_j||_F equals (1 %) that of the closed-form
    # mlp.c_proj fit of the same block and task (computed internally).
    gradient_ridge: str = "eb_input"
    # Ablation (vision only): closed-form correction of the attention V projection instead of the usual out_proj / c_proj
    # fits. None (default): off. 'preimage': per block and head h, with Z_h = A_h X (the attention-weighted ln_1 output), the
    # V slice change (dV_h, db_h) moves the pre-out_proj rows EXACTLY by Z_h dV_h^T + 1 db_h^T (rows of A_h sum to 1); the
    # block-end target D_j is pulled back through the (square) write map L = W_O^T diag(ls_1 gamma): T = D_j L^+, and each head
    # solves its own ridge regression Z_h -> T_h with unpenalised intercept and the empirical-Bayes ridge of the O/D path.
    # First order: the MLP is assumed not to react (as in the closed-form attn.out_proj fit). Requires components=['attn.v_proj'].
    v_fit: str | None = None


# Fields added after results were published: serialized only when non-default, so every historical summary,
# artifact and golden hash of a default run stays byte-identical.
_SERIALIZED_WHEN_NON_DEFAULT = {
    "streaming_fingerprint_tol": 1e-9,
    "copy_shape_matching_source_deltas": False,
    "depth_fit": "independent",
    "sequential_impl": "reference",
    "sequential_points": "single",
    "procrustes_rank": None,
    "procrustes_energy": None,
    "exclude_first_token": False,
    "native_target_ckpt_template": None,
    "fit_blocks": None,
    "pair_target_split": "none",
    "component_fit": "closed_form",
    "gradient_fit_max_iters": 500,
    "gradient_fit_tol": 1e-9,
    "gradient_ridge": "eb_input",
    "v_fit": None,
}


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
        "depth_fit",
        "sequential_impl",
        "sequential_points",
        "procrustes_rank",
        "procrustes_energy",
        "exclude_first_token",
        "native_target_ckpt_template",
        "fit_blocks",
        "pair_target_split",
        "component_fit",
        "gradient_fit_max_iters",
        "gradient_fit_tol",
        "gradient_ridge",
        "v_fit",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown direct_residual fields: {sorted(unknown)}")
    payload = {**(_PRESETS[preset] if preset is not None else {}), **value}
    if "components" in payload and isinstance(payload["components"], list):
        payload["components"] = tuple(payload["components"])
    if isinstance(payload.get("fit_blocks"), list):
        payload["fit_blocks"] = tuple(payload["fit_blocks"])
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
    if cfg.component_fit not in {"closed_form", "gradient_exact", "gradient_linearized"}:
        raise ValueError("component_fit must be 'closed_form', 'gradient_exact' or 'gradient_linearized'")
    supported = (
        CANONICAL_COMPONENT_ORDER
        if cfg.component_fit != "closed_form" or cfg.v_fit is not None
        else COMPONENT_FORWARD_ORDER
    )
    unsupported = set(components) - set(supported)
    if unsupported:
        raise ValueError(f"unknown components: {sorted(unsupported)}; supported: {sorted(supported)}")
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
    if cfg.residual_target not in {"transported_delta", "transported_endpoint", "native_oracle"}:
        raise ValueError("residual_target must be 'transported_delta', 'transported_endpoint' or 'native_oracle'")
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
        if cfg.block_split != "none" and not (cfg.block_split == "backfit" and cfg.depth_fit == "sequential"):
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
    rank, energy = cfg.procrustes_rank, cfg.procrustes_energy
    if rank is not None and energy is not None:
        raise ValueError("procrustes_rank and procrustes_energy are mutually exclusive")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
        raise ValueError("procrustes_rank must be an integer >= 1 or None")
    if energy is not None and (
        isinstance(energy, bool) or not isinstance(energy, (int, float)) or not 0.0 < float(energy) <= 1.0
    ):
        raise ValueError("procrustes_energy must be a real number in (0, 1] or None")
    if rank is not None or energy is not None:
        need = {"alignment_map": "polar", "procrustes_source": "activation", "activation_storage": "streaming"}
        bad = {name: getattr(cfg, name) for name, want in need.items() if getattr(cfg, name) != want}
        if bad:
            raise ValueError(f"procrustes_rank/procrustes_energy require {need}; got incompatible {bad}")
    if not isinstance(cfg.exclude_first_token, bool):
        raise TypeError("exclude_first_token must be bool")
    if cfg.depth_fit not in {"independent", "sequential"}:
        raise ValueError("depth_fit must be 'independent' or 'sequential'")
    if cfg.sequential_impl not in {"reference", "block_carry"}:
        raise ValueError("sequential_impl must be 'reference' or 'block_carry'")
    if cfg.sequential_impl != "reference" and cfg.depth_fit != "sequential":
        raise ValueError(f"sequential_impl={cfg.sequential_impl!r} requires depth_fit='sequential'")
    if cfg.sequential_points not in {"single", "end_end", "mid_end"}:
        raise ValueError("sequential_points must be 'single', 'end_end' or 'mid_end'")
    if cfg.sequential_points != "single" and cfg.depth_fit != "sequential":
        raise ValueError(f"sequential_points={cfg.sequential_points!r} requires depth_fit='sequential'")
    if cfg.depth_fit == "sequential":
        required = {
            "activation_storage": "streaming",
            "block_split": "none",
            "tv_scaling": "none",
            "endpoint_construction": "native_delta",
            "merge_mode": "per_task_then_merge",
            "procrustes_source": "activation",
            "residual_target": "transported_delta",
            "fidelity_holdout": False,
            "realization_diagnostics": False,
        }
        # Opt-in: intra-block Gauss-Seidel backfit of attn.out_proj + mlp.c_proj inside the sequential depth loop
        # (vision only; the shared full E_j target rules out any pair/two-point target split).
        seq_backfit = cfg.block_split == "backfit"
        if seq_backfit:
            required = {
                **required,
                "block_split": "backfit",
                "sequential_points": "single",
                "pair_target_split": "none",
                "component_fit": "closed_form",
            }
        bad = {name: getattr(cfg, name) for name, want in required.items() if getattr(cfg, name) != want}
        two_points = cfg.sequential_points != "single"
        if seq_backfit:
            if set(components) != {"attn.out_proj", "mlp.c_proj"} or len(components) != 2:
                bad["components"] = components
        elif two_points:
            if set(components) != {"attn.out_proj", "mlp.c_proj"} or len(components) != 2:
                bad["components"] = components
        elif len(components) != 1:
            bad["components"] = components
        if bad:
            what = (
                "exactly the components attn.out_proj and mlp.c_proj"
                if two_points or seq_backfit
                else "exactly one component"
            )
            raise ValueError(f"depth_fit='sequential' requires {required} and {what}; got incompatible {bad}")
        if two_points:
            if cfg.sequential_impl != "reference":
                raise ValueError(f"sequential_points={cfg.sequential_points!r} requires sequential_impl='reference'")
            if cfg.sequential_points == "mid_end":
                need = {
                    "alignment_map": "polar",
                    "alignment_row_weighting": "uniform",
                    "procrustes_source": "activation",
                    "procrustes_rank": None,
                    "procrustes_energy": None,
                }
                bad = {name: getattr(cfg, name) for name, want in need.items() if getattr(cfg, name) != want}
                if bad:
                    raise ValueError(f"sequential_points='mid_end' requires {need}; got incompatible {bad}")
    _validate_component_ablations(cfg, components)
    return cfg


#: Component groups a gradient fit takes as ONE joint fit (the q/k/v slices share the packed in_proj and the ln_1 input).
_GRADIENT_FIT_GROUPS: frozenset[frozenset[str]] = frozenset(
    frozenset(g)
    for g in (
        ("attn.v_proj",),
        ("attn.q_proj", "attn.k_proj"),
        ("attn.q_proj", "attn.k_proj", "attn.v_proj"),
        ("mlp.c_fc",),
        ("attn.out_proj",),
        ("mlp.c_proj",),
    )
)


def _validate_component_ablations(cfg: DirectResidualConfig, components: tuple[str, ...]) -> None:
    """Validation of ``native_oracle`` / ``fit_blocks`` / ``v_fit`` (all opt-in, default off)."""
    if cfg.v_fit is not None:
        if cfg.v_fit != "preimage":
            raise ValueError("v_fit must be None or 'preimage'")
        need = {
            "components": ("attn.v_proj",),
            "component_fit": "closed_form",
            "component_target": "block_boundary",
            "depth_fit": "independent",
            "block_split": "none",
            "activation_storage": "streaming",
            "residual_target": "transported_delta",
            "pair_target_split": "none",
            "fit_blocks": None,
            "tv_scaling": "none",
            "fidelity_holdout": False,
            "exact_form": True,
            "endpoint_construction": "native_delta",
        }
        got = {name: (tuple(components) if name == "components" else getattr(cfg, name)) for name in need}
        bad = {name: got[name] for name, want in need.items() if got[name] != want}
        if bad:
            raise ValueError(f"v_fit='preimage' requires {need}; got incompatible {bad}")
    if cfg.gradient_ridge not in {"eb_input", "match_norm"}:
        raise ValueError("gradient_ridge must be 'eb_input' or 'match_norm'")
    iters = cfg.gradient_fit_max_iters
    if isinstance(iters, bool) or not isinstance(iters, int) or iters <= 0:
        raise ValueError("gradient_fit_max_iters must be a positive integer")
    gtol = cfg.gradient_fit_tol
    if isinstance(gtol, bool) or not isinstance(gtol, (int, float)) or not math.isfinite(float(gtol)) or gtol <= 0:
        raise ValueError("gradient_fit_tol must be a positive finite number")
    if cfg.component_fit != "closed_form":
        if frozenset(components) not in _GRADIENT_FIT_GROUPS:
            raise ValueError(
                "component_fit gradient arms fit one component group per run: attn.v_proj; attn.q_proj + attn.k_proj; "
                "attn.q_proj + attn.k_proj + attn.v_proj; mlp.c_fc; attn.out_proj; or mlp.c_proj; "
                f"got {sorted(components)}"
            )
        need = {
            "depth_fit": "independent",
            "component_target": "block_boundary",
            "block_split": "none",
            "ridge_estimator": "empirical_bayes",
            "exact_form": True,
            "endpoint_construction": "native_delta",
            "tv_scaling": "none",
            "realization_diagnostics": False,
            "fidelity_holdout": False,
            "procrustes_source": "activation",
        }
        bad = {name: getattr(cfg, name) for name, want in need.items() if getattr(cfg, name) != want}
        if cfg.residual_target == "transported_endpoint":
            bad["residual_target"] = cfg.residual_target
        if bad:
            raise ValueError(f"component_fit={cfg.component_fit!r} requires {need}; got incompatible {bad}")
    elif cfg.gradient_ridge != "eb_input":
        raise ValueError("gradient_ridge='match_norm' needs component_fit='gradient_exact' or 'gradient_linearized'")
    template = cfg.native_target_ckpt_template
    if cfg.residual_target == "native_oracle":
        if not isinstance(template, str) or "{task}" not in template:
            raise ValueError("residual_target='native_oracle' requires native_target_ckpt_template containing '{task}'")
        need = {
            "depth_fit": "independent",
            "component_target": "block_boundary",
            "block_split": "none",
            "procrustes_source": "activation",
            "endpoint_construction": "native_delta",
            "merge_mode": "per_task_then_merge",
            "tv_scaling": "none",
            "realization_diagnostics": False,
            "fidelity_holdout": False,
        }
        bad = {name: getattr(cfg, name) for name, want in need.items() if getattr(cfg, name) != want}
        if bad:
            raise ValueError(f"residual_target='native_oracle' requires {need}; got incompatible {bad}")
    elif template is not None:
        raise ValueError("native_target_ckpt_template is only used with residual_target='native_oracle'")
    if cfg.pair_target_split not in {"none", "per_block_share", "interpolate"}:
        raise ValueError("pair_target_split must be 'none', 'per_block_share' or 'interpolate'")
    if cfg.pair_target_split != "none":
        need = {
            "depth_fit": "independent" if cfg.pair_target_split == "per_block_share" else "sequential",
            "sequential_impl": "reference",
            "sequential_points": "single",
            "residual_target": "transported_delta",
            "endpoint_construction": "native_delta",
            "block_split": "none",
            "tv_scaling": "none",
            "realization_diagnostics": False,
            "fidelity_holdout": False,
        }
        bad = {name: getattr(cfg, name) for name, want in need.items() if getattr(cfg, name) != want}
        if bad:
            raise ValueError(f"pair_target_split={cfg.pair_target_split!r} requires {need}; got incompatible {bad}")
    blocks = cfg.fit_blocks
    if blocks is not None:
        if isinstance(blocks, str) or not isinstance(blocks, (list, tuple)) or not blocks:
            raise ValueError("fit_blocks must be None or a non-empty list of block indices")
        if any(isinstance(b, bool) or not isinstance(b, int) or b < 0 for b in blocks):
            raise ValueError("fit_blocks must hold non-negative integers")
        if any(b >= c for b, c in zip(blocks, blocks[1:], strict=False)):
            raise ValueError("fit_blocks must be sorted and unique")
        need = {
            "endpoint_construction": "native_delta",
            "tv_scaling": "none",
            "sequential_impl": "reference",
            "sequential_points": "single",
        }
        bad = {name: getattr(cfg, name) for name, want in need.items() if getattr(cfg, name) != want}
        if bad:
            raise ValueError(f"fit_blocks requires {need}; got incompatible {bad}")


def resolve_fit_blocks(config: DirectResidualConfig, depth: int) -> list[int]:
    """Target blocks a fit covers: ``config.fit_blocks`` (range-checked against ``depth``) or every block."""
    if config.fit_blocks is None:
        return list(range(depth))
    blocks = [int(b) for b in config.fit_blocks]
    if blocks and blocks[-1] >= depth:
        raise ValueError(f"fit_blocks {blocks} out of range for a target with {depth} blocks")
    return blocks


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


def reject_decoder_only_options(cfg: DirectResidualConfig) -> None:
    """Vision path guard: refuse options that are only defined on decoder (text) calibration."""
    if cfg.exclude_first_token:
        raise ValueError(
            "Ariadne option exclude_first_token=true is decoder-only (it needs the attention mask of text "
            "calibration); the vision path does not support it"
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
    if cfg.residual_target == "native_oracle":
        raise NotImplementedError("residual_target=native_oracle is vision-only")
    if cfg.component_fit != "closed_form":
        raise NotImplementedError("component_fit=gradient is vision-only")
    if cfg.v_fit is not None:
        # GQA later: stack the Z_h rows of the query heads that share a key/value head.
        raise NotImplementedError("v_fit='preimage' is vision-only")
    if cfg.sequential_points == "mid_end":
        # HF decoder layers receive hidden_states as a keyword argument, so the block_input capture (positional
        # inputs[0]) does not see the block input there.
        raise NotImplementedError("mid_end is vision-only")
    return cfg


def target_groups(pairing) -> dict[int, tuple[int, int, int]]:
    """``{j: (group_size, index_in_group, group_start)}``: groups are the maximal runs of consecutive target blocks
    sharing one source block ``pairing[j]`` (size 1 when every target block has its own source block)."""
    out: dict[int, tuple[int, int, int]] = {}
    j = 0
    n = len(pairing)
    while j < n:
        k = j
        while k + 1 < n and pairing[k + 1] == pairing[j]:
            k += 1
        for t in range(j, k + 1):
            out[t] = (k - j + 1, t - j, j)
        j = k + 1
    return out


def annotate_target_split(rows, config: DirectResidualConfig, pairing) -> None:
    """Record the per-block target scale / interpolation weight of ``pair_target_split`` in the diagnostics rows."""
    if config.pair_target_split == "none":
        return
    groups = target_groups(pairing.pairing)
    for row in rows:
        size, index, _start = groups[int(row["position"])]
        row["pair_target_split"] = config.pair_target_split
        row["target_group_size"] = size
        row["target_group_index"] = index
        if config.pair_target_split == "per_block_share":
            row["target_scale"] = 1.0 / size
        else:
            row["target_interp_weight"] = (index + 1) / size
