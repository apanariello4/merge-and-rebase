"""Config resolution for ``eval.vision_rebase`` (Phase 5.6 / 5.7).

``resolve_run_config`` holds everything ``main()`` derives from the merged config dict *before* any
model exists, with the exact legacy semantics, error messages and validation order that the
``tests/golden/test_main_golden.py`` error table pins. ``ResolvedRunConfig.bind`` holds the
guards that can only fire once the source/target depths are known (they need the built models) and
returns a ``RunPlan`` with the prestep flags.

Nothing here builds a model or touches a dataset. The only environment dependence is the suite
registry, which the caller passes in (``suites=``) so that harnesses that patch ``SUITES`` on the
CLI module keep working; when omitted the real registry is imported lazily.

Depth rule: ``depth_defaults: "legacy"`` builds ``DepthRule`` from the top-level ``depth_alignment``
key only (default ``"ariadne"`` == BRACE interpolation), byte-identical to the pre-P5.12 code.
``"method"`` applies the per-method defaults (THESEUS-like: BRACE + ``skip_correction=True``;
BiCo-like: ``discrete_index_match``; Ariadne / others: none). When the key is absent the method
defaults apply too, but a configuration whose outcome would change raises
``ConfigMeaningChangedError`` naming both fixes (see ``resolve_depth_rule``).

Ariadne never reaches ``get_method`` or ``resolve_block_extension_config`` (see
``tests/test_vision_rebase_direct_residual_dispatch.py``): its branch builds a default
``BlockExtensionConfig`` and parses its own ``ariadne_params`` schema instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal

import torch

from merge_and_rebase.utils.helpers import parse_csv

from . import get_method
from .block_extension.config import (
    BlockExtensionConfig,
    parse_depth_rule_schema,
    resolve_block_extension_config,
    warn_brace_only_fields_under_discrete,
)
from .capabilities import MethodFamily, default_depth_prestep, depth_prestep_methods, method_family
from .config_schema import canonicalize
from .merge_modes import _SINGLE_TRANSPORT_MODES, _TRANSPORT_THEN_MERGE_MODES, _resolve_merge_mode_config
from .methods.ariadne import DirectResidualConfig, parse_direct_residual_config, resolve_direct_residual_preset
from .runtime import format_rebase_method_label, resolve_rebase_method_config

_BASE_CONSTRUCTION_MODES = ("per_task", "independent_endpoint_average")


@dataclass(frozen=True)
class DepthRule:
    """How a depth mismatch is resolved. Legacy: derived from ``depth_alignment`` only."""

    kind: Literal["none", "brace", "discrete_index_match"]
    extension_strategy: str | None = None
    skip_correction: bool | None = None
    source: Literal["config", "alias:depth_alignment", "method_default", "legacy_defaults"] = "legacy_defaults"

    @property
    def legacy_depth_alignment(self) -> str:
        """The legacy ``depth_alignment`` string (``ariadne`` means BRACE), as emitted in the summary."""
        return "discrete_index_match" if self.kind == "discrete_index_match" else "ariadne"


@dataclass(frozen=True)
class PrestepEvalSpec:
    """Source/target evaluation around the depth prestep (``eval_before_rebase``) and ``source_only`` runs."""

    block_extension_eval_requested: bool
    block_extension_eval_enabled: bool
    block_extension_eval_split: str
    block_extension_eval_first_n_batches: Any
    source_only: bool


#: Source-model linear-mode-connectivity (LMC) evaluation was removed. Configs that ask for it fail loudly; keys left
#: at their "off" value are ignored, since they never influenced a result.
_REMOVED_LMC_KEYS = (
    "source_lmc_eval",
    "source_lmc_eval_split",
    "source_lmc_first_n_batches",
    "source_lmc_alpha_min",
    "source_lmc_alpha_max",
    "source_lmc_alpha_step",
    "cross_task_lmc_pairs",
    "cross_task_lmc_eval_split",
    "all_task_lmc_tasks",
    "all_task_lmc_eval_split",
)


def _reject_removed_lmc_keys(cfg: Mapping[str, Any]) -> None:
    requested = [
        key
        for key in ("source_lmc_eval", "cross_task_lmc_pairs", "all_task_lmc_tasks")
        if cfg.get(key)  # true / a non-empty list asks for an evaluation that no longer exists
    ]
    if requested:
        raise ValueError(
            f"source-model LMC evaluation was removed; drop {requested} (and the other "
            "source_lmc_* / cross_task_lmc_* / all_task_lmc_* keys) from the config."
        )


@dataclass(frozen=True)
class AlphaSpec:
    search: bool
    patience: int
    search_split: str
    alphas: list[float]
    selection: str


@dataclass(frozen=True)
class MergeSpec:
    mode: str
    method_name: str
    params: Any
    global_alpha_search: Any
    base_construction: str


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Everything ``main()`` derives from the config before any model exists."""

    cfg: Mapping[str, Any]
    method: Any
    method_name: str
    method_params: dict
    method_label: str
    method_family: MethodFamily | None
    block_extension_enabled: bool
    block_extension_cfg: BlockExtensionConfig
    depth_rule: DepthRule
    ariadne_cfg: DirectResidualConfig | None
    ariadne_preset: str | None
    merge: MergeSpec
    alpha: AlphaSpec
    prestep_eval: PrestepEvalSpec
    strict_load: bool
    device: str
    grad_batch_size: int | None
    grad_imgs_per_class: int | None
    grad_num_batches: int | None
    suite_name: str
    suite: Any
    tasks: list[str]
    # Additive summary record of the resolved depth rule, and the post-model guard (P5.12).
    depth_rule_resolved: dict = field(default_factory=dict)
    depth_guard: str | None = None
    #: Entrypoint that resolved this config; selects the methods that run the depth prestep there.
    entrypoint: Literal["vision", "llm"] = "vision"

    # -- method dispatch predicates (from ``rebase.capabilities.METHOD_TRAITS``) ------------------
    @property
    def direct_fit(self) -> bool:
        """The method fits the target task vector itself (Ariadne) instead of transporting the source one."""
        return self.method_family is MethodFamily.DIRECT_FIT

    @property
    def theseus_mode(self) -> bool:
        return self.method_name == "theseus"

    @property
    def bico_mode(self) -> bool:
        return self.method_name == "bico"

    @property
    def depth_prestep_method(self) -> bool:
        """Depth-mismatched pairs of this method run the depth prestep (BRACE / discrete index match)."""
        return self.method_name in depth_prestep_methods(self.entrypoint)

    @property
    def transfusion_mode(self) -> bool:
        return self.method_name == "transfusion"

    @property
    def depth_alignment_mode(self) -> str:
        return self.depth_rule.legacy_depth_alignment

    def bind(self, source_depth: int, target_depth: int) -> RunPlan:
        """Post-model guards and prestep flags (needs the real source/target depths)."""
        if self.depth_guard is not None and source_depth != target_depth:
            raise ConfigMeaningChangedError(self.depth_guard)
        rule = "none"
        if self.depth_prestep_method and source_depth != target_depth:
            if self.depth_rule.kind == "brace" and self.block_extension_enabled:
                rule = "brace"
            elif self.depth_rule.kind == "discrete_index_match":
                rule = "discrete_index_match"
        # merge_then_brace_then_transport merges deltas on the native source base first and only
        # then runs its own once-only structural step, so the prestep never fires per task under it.
        timing = "once_on_merged_source" if self.merge.mode == "merge_then_brace_then_transport" else "per_task"
        if self.merge.mode == "merge_then_rebase" and rule == "brace":
            raise NotImplementedError(
                "merge_then_rebase does not support the block-extension prestep yet: "
                "per-task extended source bases live on different keyspaces and cannot be "
                "merged without a consensus-base step (see transport_then_merge). "
                "Use merge_mode='rebase_then_merge' for depth-mismatch pairs."
            )
        return RunPlan(
            source_depth=int(source_depth),
            target_depth=int(target_depth),
            depth_alignment=DepthAlignment(rule=rule, timing=timing),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class DepthAlignment:
    """How the source is brought to the target depth: decided once, in ``ResolvedRunConfig.bind``.

    ``rule``: ``brace`` (block extension / shrink), ``discrete_index_match`` (reindexed stack) or ``none`` (equal
    depths, a method that runs no depth prestep, or BRACE disabled). ``timing``: ``per_task`` (prestep before each
    task's transport) or ``once_on_merged_source`` (``merge_then_brace_then_transport``: once, on the merged source).
    """

    rule: Literal["none", "brace", "discrete_index_match"]
    timing: Literal["per_task", "once_on_merged_source"]


@dataclass(frozen=True)
class RunPlan:
    """Post-model plan: the source/target depths and the depth alignment they need."""

    source_depth: int
    target_depth: int
    depth_alignment: DepthAlignment

    # -- views of ``depth_alignment`` -----------------------------------------------------------
    @property
    def run_block_extension_prestep(self) -> bool:
        """BRACE runs (per task or once on the merged source)."""
        return self.depth_alignment.rule == "brace"

    @property
    def run_discrete_layer_match_prestep(self) -> bool:
        return self.depth_alignment.rule == "discrete_index_match"

    @property
    def task_block_extension_prestep(self) -> bool:
        """BRACE runs as the per-task prestep."""
        return self.run_block_extension_prestep and self.depth_alignment.timing == "per_task"

    @property
    def task_discrete_layer_match_prestep(self) -> bool:
        return self.run_discrete_layer_match_prestep and self.depth_alignment.timing == "per_task"


class ConfigMeaningChangedError(ValueError):
    """The config's outcome would differ under the per-method depth defaults (P5.12)."""


_DEPTH_DEFAULTS_MODES = ("legacy", "method")


def _parse_depth_defaults(cfg: Mapping[str, Any]) -> str | None:
    raw = cfg.get("depth_defaults")
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if mode not in _DEPTH_DEFAULTS_MODES:
        raise ValueError("depth_defaults must be one of: legacy, method")
    return mode


def _meaning_changed_message(method_name: str, reason: str, old_fix: str) -> str:
    return (
        f"{method_name}: {reason}. The per-method depth defaults would change this run's result. "
        f"Either keep the previous behaviour with {old_fix} (or \"depth_defaults\": \"legacy\"), "
        'or accept the new method default with "depth_defaults": "method".'
    )


def check_native_target_tasks(merge_mode: str, *, transfusion: bool) -> None:
    """Native target checkpoints (explicit or auto-detected) need a merge mode that merges on the target base."""
    if merge_mode == "none":
        raise ValueError(
            "Native target checkpoints require a merge mode; merge_mode='none' evaluates "
            "per-task transported deltas only. Use merge_mode='rebase_then_merge'."
        )
    if merge_mode in _SINGLE_TRANSPORT_MODES:
        raise ValueError(
            "Native target checkpoints cannot participate in merge_then_rebase: the merge "
            "happens on the source base, where native target deltas do not exist."
        )
    if transfusion:
        raise NotImplementedError(
            "Native target checkpoints with transfusion are not supported: the permutation "
            "prepare step swaps the target keyspace. Use a theseus/bico transport method."
        )


def reject_independent_endpoint_average(merge_mode: str, alpha_selection: str, *, has_native_tasks: bool) -> None:
    """``base_construction='independent_endpoint_average'`` is never valid; raise the most specific reason."""
    if merge_mode not in _TRANSPORT_THEN_MERGE_MODES:
        raise ValueError(
            "base_construction='independent_endpoint_average' requires merge_mode='brace_transport_then_merge'."
        )
    if alpha_selection != "shared":
        raise ValueError(
            "base_construction='independent_endpoint_average' requires "
            "alpha_selection='shared'; per-task alpha search is not part of this baseline."
        )
    if has_native_tasks:
        raise ValueError(
            "base_construction='independent_endpoint_average' requires every task to be "
            "an independently transformed source endpoint; native target tasks are not allowed."
        )
    # B6: in the accepted merge modes the independent base is computed for nobody -- it is consumed only by
    # brace_merge_then_transport, which this option cannot be combined with -- so the run would silently be
    # identical to base_construction='per_task'.
    raise ValueError(
        f"base_construction='independent_endpoint_average' has no effect with merge_mode='{merge_mode}': the "
        "independent endpoint base is only consumed by merge_mode='brace_merge_then_transport', which does "
        "not support it. Use base_construction='per_task' (the identical computation)."
    )


def resolve_depth_rule(
    default_rule: str | None,
    method_name: str,
    cfg: Mapping[str, Any],
    block_extension_enabled: bool,
    block_extension_cfg: BlockExtensionConfig,
) -> tuple[DepthRule, BlockExtensionConfig, str | None]:
    """Resolve the depth rule once. Returns ``(depth_rule, block_extension_cfg, depth_guard)``.

    ``depth_guard`` is a message that ``ResolvedRunConfig.bind`` raises as ``ConfigMeaningChangedError``
    when the source/target depths turn out to differ (the depths are unknown before the models exist).
    ``default_rule`` is the method's default depth prestep (``rebase.capabilities.default_depth_prestep``).
    """
    mode = _parse_depth_defaults(cfg)
    if mode == "legacy" or default_rule is None:
        return _resolve_depth_rule(cfg), block_extension_cfg, None
    schema = parse_depth_rule_schema(cfg, warn=False)
    raw_params = cfg.get("block_extension_params") or {}
    source = schema.source or "method_default"
    guard: str | None = None
    if default_rule == "discrete_index_match":
        rule = "discrete_index_match" if schema.rule == "method_default" else schema.rule
        if rule == "discrete_index_match":
            warn_brace_only_fields_under_discrete(raw_params, stacklevel=3)
        if mode is None and schema.rule == "method_default":
            guard = _meaning_changed_message(
                method_name,
                "depth-mismatched pair without depth_alignment / block_extension_params.depth_rule "
                "(the previous default ran the BRACE block-extension prestep with the block_extension_params "
                "settings; the new BiCo default is the discrete index match)",
                '"depth_alignment": "ariadne"',
            )
        return DepthRule(kind=rule, source=source), block_extension_cfg, guard  # type: ignore[arg-type]
    rule = "brace" if schema.rule == "method_default" else schema.rule
    if rule == "discrete_index_match":
        return DepthRule(kind=rule, source=source), block_extension_cfg, None  # type: ignore[arg-type]
    if schema.skip_correction is not None:
        return DepthRule(kind="brace", source=source), block_extension_cfg, None
    # skip_correction absent: the THESEUS default pair (interpolate_per_weight + skip_correction=True).
    old_fix = '"block_extension_params": {"skip_correction": false}'
    if mode is None:
        blocking = [
            name
            for name, active in (
                (
                    "target_shared_correction",
                    block_extension_cfg.target_shared_correction is not None
                    and block_extension_cfg.target_shared_correction.active,
                ),
                ("correction_scope!='inserted'", block_extension_cfg.correction_scope != "inserted"),
            )
            if active
        ]
        if blocking:
            raise ConfigMeaningChangedError(
                _meaning_changed_message(
                    method_name, f"{', '.join(blocking)} requires skip_correction=false but it is not given", old_fix
                )
            )
        if block_extension_enabled:
            guard = _meaning_changed_message(
                method_name, "depth-mismatched pair without an explicit skip_correction", old_fix
            )
    injected = dict(raw_params)
    injected["skip_correction"] = True
    new_cfg_raw = dict(cfg)
    new_cfg_raw["block_extension_params"] = injected
    _, new_cfg = resolve_block_extension_config(new_cfg_raw)
    return (
        DepthRule(
            kind="brace",
            extension_strategy=new_cfg.extension_strategy,
            skip_correction=True,
            source=source if schema.source else "method_default",
        ),
        new_cfg,
        guard,
    )


def _depth_rule_record(
    default_rule: str | None,
    depth_rule: DepthRule,
    block_extension_cfg: BlockExtensionConfig,
    ariadne_cfg: Any,
) -> dict:
    """Additive summary record ``depth_rule_resolved`` (rule, extension_strategy, skip_correction, ...)."""
    blockext = default_rule is not None
    brace = blockext and depth_rule.kind == "brace"
    return {
        "rule": depth_rule.kind if blockext else "none",
        "extension_strategy": block_extension_cfg.extension_strategy if brace else None,
        "skip_correction": bool(block_extension_cfg.skip_correction) if brace else None,
        "depth_pairing": getattr(ariadne_cfg, "depth_pairing", None) if ariadne_cfg is not None else None,
        "source": depth_rule.source if blockext else "method_default",
    }


def _resolve_depth_rule(cfg: Mapping[str, Any]) -> DepthRule:
    # "ariadne" (default) preserves every existing behavior: the block-extension prestep resizes
    # the source model's depth via ARIADNE's insertion/collapse machinery (BRACE).
    # "discrete_index_match" is the faithful BiCo/THESEUS structural-resize control: it reindexes
    # the source model to the target depth via the flat, closed-form `DiscreteLayerPairing`
    # instead, with no interpolation, no correction fit, and no ancestry bookkeeping. Resolved
    # once, so an unknown value fails fast rather than surfacing deep in the per-task loop.
    depth_alignment_mode = str(cfg.get("depth_alignment", "ariadne")).strip().lower()
    if depth_alignment_mode not in {"ariadne", "discrete_index_match"}:
        raise ValueError("depth_alignment must be one of: ariadne, discrete_index_match")
    return DepthRule(
        kind="brace" if depth_alignment_mode == "ariadne" else "discrete_index_match",
        source="alias:depth_alignment" if "depth_alignment" in cfg else "legacy_defaults",
    )


def resolve_run_config(cfg: Mapping[str, Any], *, suites: Mapping[str, Any] | None = None) -> ResolvedRunConfig:
    """Resolve and validate ``cfg`` into a ``ResolvedRunConfig`` (no models, no datasets).

    ``cfg`` must already be the merged config (file + CLI overrides) with the
    ``block_extension_enabled`` default applied by the caller. ``suites`` is the suite registry
    (defaults to the real one, imported lazily).
    """
    if suites is None:
        from ..eval.datasets.vision8_14_20 import SUITES as suites
    cfg = canonicalize(cfg)  # canonical or legacy key names; see rebase/config_schema.py

    method_name, method_params = resolve_rebase_method_config(cfg)
    # Ariadne (formerly Direct Residual; "direct_residual" is a registry alias of "ariadne" and
    # both spellings take exactly this path) is registered, but it needs whole model objects and
    # dataloaders for paired activation capture, not a state-dict-delta transport() call, so this
    # entrypoint never calls `get_method` for it (its `transport` raises NotImplementedError). It
    # also must never let `resolve_block_extension_config(cfg)` run over `cfg` -- BRACE's config
    # gates have no business accepting or rejecting an Ariadne config, since Ariadne never reaches
    # a single BRACE code path. See tests/test_vision_rebase_direct_residual_dispatch.py.
    # `method_name` stays exactly what the config said (it is recorded verbatim in the run
    # summary); only the dispatch decision goes through the canonical name.
    family = method_family(method_name)
    direct_fit = family is MethodFamily.DIRECT_FIT
    if direct_fit:
        method = SimpleNamespace(name=method_name)
        block_extension_enabled = False
        block_extension_cfg = BlockExtensionConfig()
        # Direct Residual's own config schema is narrower than, and independent of,
        # `method_params` (which historically carries per-transport-method kwargs consumed by
        # `method.transport()` -- a call Direct Residual never makes). A dedicated top-level key
        # keeps that separation explicit.
        if cfg.get("direct_residual_params") is not None and cfg.get("ariadne_params") is not None:
            raise ValueError("config has both 'direct_residual_params' and its alias 'ariadne_params'; use one")
        direct_residual_raw_params = (
            cfg["ariadne_params"] if cfg.get("ariadne_params") is not None else cfg.get("direct_residual_params")
        )
        direct_residual_cfg = parse_direct_residual_config(direct_residual_raw_params)
        direct_residual_preset = resolve_direct_residual_preset(direct_residual_raw_params)
        sequential_modes = {"sequential_source_endpoints", "sequential_delta_on_synthesized_base"}
        if (
            cfg.get("load_direct_residual_tvs_dir")
            and direct_residual_cfg.endpoint_construction not in sequential_modes
        ):
            raise ValueError("load_direct_residual_tvs_dir requires a sequential endpoint construction")
        if cfg.get("load_direct_residual_tvs_dir") and (
            cfg.get("save_transported_artifacts") or cfg.get("save_transported_tvs_dir")
        ):
            raise ValueError("cannot save transported artifacts while loading sequential DR vectors")
    else:
        method = get_method(method_name)
        block_extension_enabled, block_extension_cfg = resolve_block_extension_config(cfg)
        direct_residual_cfg = None
        direct_residual_preset = None
    method_label = format_rebase_method_label(method_name, method_params)
    depth_prestep_method = method_name in depth_prestep_methods("vision")
    default_rule = default_depth_prestep(method_name, "vision")
    depth_rule, block_extension_cfg, depth_guard = resolve_depth_rule(
        default_rule, method_name, cfg, block_extension_enabled, block_extension_cfg
    )
    depth_rule_resolved = _depth_rule_record(default_rule, depth_rule, block_extension_cfg, direct_residual_cfg)
    if (
        direct_fit
        and direct_residual_cfg.merge_mode == "merge_in_source_then_fit"
        and str(cfg.get("alpha_selection", "shared")).strip().lower() != "shared"
    ):
        # Mirrors the shared-alpha requirement _resolve_merge_mode_config already enforces for
        # merge_then_brace_then_transport: once every task's transported delta collapses to the
        # SAME once-fitted correction, a per-task alpha search is degenerate.
        raise ValueError(
            "direct_residual merge_mode='merge_in_source_then_fit' requires alpha_selection='shared': "
            "the fit is performed once, on the merged source pair, and produces one correction shared "
            "by every task -- a per-task alpha search over an identical delta is not meaningful."
        )
    eval_before_rebase = bool(cfg.get("eval_before_rebase", False))
    block_extension_eval_requested = bool(eval_before_rebase)
    block_extension_eval_enabled = bool(block_extension_eval_requested and depth_prestep_method)
    block_extension_eval_split = str(cfg.get("block_extension_eval_split", "test")).strip().lower()
    if block_extension_eval_split not in {"val", "test"}:
        raise ValueError("block_extension_eval_split must be one of: val, test")
    block_extension_eval_first_n_batches = block_extension_cfg.first_n_eval_batches
    _reject_removed_lmc_keys(cfg)
    source_only = bool(cfg.get("source_only", False))
    strict_load = bool(cfg.get("strict_load", False))
    device = str(cfg.get("device", "cuda"))

    if block_extension_eval_requested and not depth_prestep_method:
        print(
            "Block-extension target-dataset eval: requested but skipped "
            f"(method='{method_name}' does not support block-extension)."
        )

    grad_batch_size = int(cfg["grad_batch_size"]) if cfg.get("grad_batch_size") is not None else None
    grad_imgs_per_class = int(cfg["grad_imgs_per_class"]) if cfg.get("grad_imgs_per_class") is not None else None
    grad_num_batches = int(cfg["grad_num_batches"]) if cfg.get("grad_num_batches") is not None else None

    alpha_search = bool(cfg.get("alpha_search", False))
    alpha_patience_raw = cfg.get("alpha_patience", 0)
    alpha_patience = int(alpha_patience_raw) if alpha_patience_raw is not None else 0
    if alpha_patience < 0:
        raise ValueError("alpha_patience must be >= 0")

    alpha_search_split = str(cfg.get("alpha_search_split", "val")).strip().lower()
    if alpha_search_split not in {"val", "test"}:
        raise ValueError("alpha_search_split must be one of: val, test")

    if alpha_search:
        a_min = float(cfg.get("alpha_min", 0.0))
        a_max = float(cfg.get("alpha_max", 2.0))
        a_step = float(cfg.get("alpha_step", 0.1))
        if a_step <= 0.0:
            raise ValueError("alpha_step must be > 0")
        if a_max < a_min:
            raise ValueError("alpha_max must be >= alpha_min")
        alphas = torch.arange(a_min, a_max + 1e-9, a_step).tolist()
    else:
        alphas = [float(cfg.get("alpha", 1.0))]

    alpha_selection = str(cfg.get("alpha_selection", "shared")).strip().lower()
    if alpha_selection not in {"shared", "per_task"}:
        raise ValueError("alpha_selection must be one of: shared, per_task")

    merge_mode, merge_method_name, merge_params, global_alpha_search = _resolve_merge_mode_config(cfg, alpha_selection)
    if direct_fit and merge_mode in _SINGLE_TRANSPORT_MODES:
        # merge_then_rebase / brace_merge_then_transport / merge_then_brace_then_transport all end
        # by calling method.transport() once on a merged direction. Direct Residual has no such
        # method object to call, and its own once-only merge path is
        # `direct_residual_params.merge_mode='merge_in_source_then_fit'`, which is independent of
        # this top-level `merge_mode` key. Reject the combination early rather than failing later
        # with an unhelpful AttributeError.
        raise ValueError(
            f"method='direct_residual' does not support merge_mode='{merge_mode}': Direct Residual has no "
            "transport() call for the merge-mode dispatch to invoke. Use merge_mode='none' (optionally with "
            "direct_residual_params.merge_mode='merge_in_source_then_fit' for a once-only merged fit) or "
            "merge_mode='rebase_then_merge'/'brace_transport_then_merge' for per-task fits merged afterward."
        )

    base_construction = str(cfg.get("base_construction", "per_task")).strip().lower()
    if base_construction not in _BASE_CONSTRUCTION_MODES:
        raise ValueError("base_construction must be one of: " + ", ".join(_BASE_CONSTRUCTION_MODES))

    positive_alphas = [float(alpha) for alpha in alphas if float(alpha) > 0.0]
    if alpha_search and not positive_alphas:
        raise ValueError("alpha_search requires at least one alpha > 0.")

    suite_name = cfg.get("suite", "vision8")
    if suite_name not in suites:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(suites)}")
    suite = suites[suite_name]

    tasks_arg = cfg.get("tasks", "all")
    if tasks_arg == "all":
        tasks = list(suite.tasks)
    else:
        tasks = parse_csv(tasks_arg)
        bad = [t for t in tasks if t not in suite.tasks]
        if bad:
            raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite.tasks)}")

    # Config-only checks of the vision pre-model pass: they fail before any model or checkpoint is loaded.
    native_target_tasks = [str(t) for t in (cfg.get("native_target_tasks", []) or [])]
    unknown_native_tasks = [t for t in native_target_tasks if t not in tasks]
    if unknown_native_tasks:
        raise ValueError(f"native_target_tasks contains tasks not in the task list: {unknown_native_tasks}")
    if native_target_tasks:
        check_native_target_tasks(merge_mode, transfusion=method_name == "transfusion")
    if base_construction == "independent_endpoint_average":
        reject_independent_endpoint_average(merge_mode, alpha_selection, has_native_tasks=bool(native_target_tasks))

    return ResolvedRunConfig(
        cfg=cfg,
        method=method,
        method_name=method_name,
        method_params=method_params,
        method_label=method_label,
        method_family=family,
        block_extension_enabled=block_extension_enabled,
        block_extension_cfg=block_extension_cfg,
        depth_rule=depth_rule,
        ariadne_cfg=direct_residual_cfg,
        ariadne_preset=direct_residual_preset,
        merge=MergeSpec(
            mode=merge_mode,
            method_name=merge_method_name,
            params=merge_params,
            global_alpha_search=global_alpha_search,
            base_construction=base_construction,
        ),
        alpha=AlphaSpec(
            search=alpha_search,
            patience=alpha_patience,
            search_split=alpha_search_split,
            alphas=alphas,
            selection=alpha_selection,
        ),
        prestep_eval=PrestepEvalSpec(
            block_extension_eval_requested=block_extension_eval_requested,
            block_extension_eval_enabled=block_extension_eval_enabled,
            block_extension_eval_split=block_extension_eval_split,
            block_extension_eval_first_n_batches=block_extension_eval_first_n_batches,
            source_only=source_only,
        ),
        strict_load=strict_load,
        device=device,
        grad_batch_size=grad_batch_size,
        grad_imgs_per_class=grad_imgs_per_class,
        grad_num_batches=grad_num_batches,
        suite_name=suite_name,
        suite=suite,
        tasks=tasks,
        depth_rule_resolved=depth_rule_resolved,
        depth_guard=depth_guard,
    )
