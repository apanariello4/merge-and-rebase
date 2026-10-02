"""Config resolution for ``eval.vision_rebase`` (Phase 5.6 / 5.7).

``resolve_run_config`` holds everything ``main()`` derives from the merged config dict *before* any
model exists, with the exact legacy semantics, error messages and validation order that the
``tests/golden/test_main_golden.py`` error table pins. ``ResolvedRunConfig.bind`` holds the
guards that can only fire once the source/target depths are known (they need the built models) and
returns a ``RunPlan`` with the prestep flags.

Nothing here builds a model or touches a dataset. The only environment dependence is the suite
registry, which the caller passes in (``suites=``) so that harnesses that patch ``SUITES`` on the
CLI module keep working; when omitted the real registry is imported lazily.

Legacy depth semantics only: ``DepthRule`` is built from the top-level ``depth_alignment`` key
(default ``"ariadne"`` == BRACE interpolation). Per-method depth defaults are a later, separately
declared behaviour change and are deliberately NOT implemented here.

Ariadne never reaches ``get_method`` or ``resolve_block_extension_config`` (see
``tests/test_vision_rebase_direct_residual_dispatch.py``): its branch builds a default
``BlockExtensionConfig`` and parses its own ``ariadne_params`` schema instead.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import torch

from merge_and_rebase.utils.helpers import parse_csv

from ..eval.block_extension import BlockExtensionConfig, resolve_block_extension_config
from ..eval.target_residual_completion import validate_residual_completion_depth_direction
from ..eval.vision_rebase_merge import (
    _SINGLE_TRANSPORT_MODES,
    _resolve_merge_mode_config,
)
from . import get_method
from .methods.ariadne import DirectResidualConfig, parse_direct_residual_config, resolve_direct_residual_preset
from .registry import canonical_method_name
from .runtime import format_rebase_method_label, resolve_rebase_method_config

_THESEUS_LIKE = frozenset({"theseus", "theseus_reference"})
_BICO_LIKE = frozenset({"bico", "bico_gradin"})
_BASE_CONSTRUCTION_MODES = ("per_task", "independent_endpoint_average")


class MethodKind(enum.Enum):
    """Coarse dispatch class of the configured rebase method (replaces five booleans)."""

    THESEUS_LIKE = "theseus_like"
    BICO_LIKE = "bico_like"
    TRANSFUSION = "transfusion"
    ARIADNE = "ariadne"
    OTHER = "other"

    @classmethod
    def of(cls, method_name: str) -> MethodKind:
        if canonical_method_name(method_name) == "ariadne":
            return cls.ARIADNE
        if method_name in _THESEUS_LIKE:
            return cls.THESEUS_LIKE
        if method_name in _BICO_LIKE:
            return cls.BICO_LIKE
        if method_name == "transfusion":
            return cls.TRANSFUSION
        return cls.OTHER


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
class SourceLmcSpec:
    block_extension_eval_requested: bool
    block_extension_eval_enabled: bool
    block_extension_eval_split: str
    block_extension_eval_first_n_batches: Any
    eval: bool
    eval_split: str
    first_n_batches: int | None
    alphas: list[float]
    cross_task_pairs: list[tuple[str, str]]
    cross_task_split: str
    all_task_tasks: list[str]
    all_task_split: str
    source_only: bool


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
    method_kind: MethodKind
    block_extension_enabled: bool
    block_extension_cfg: BlockExtensionConfig
    depth_rule: DepthRule
    ariadne_cfg: DirectResidualConfig | None
    ariadne_preset: str | None
    merge: MergeSpec
    alpha: AlphaSpec
    lmc: SourceLmcSpec
    strict_load: bool
    device: str
    grad_batch_size: int | None
    grad_imgs_per_class: int | None
    grad_num_batches: int | None
    suite_name: str
    suite: Any
    tasks: list[str]

    # -- method-kind predicates (exactly the legacy boolean sets) ----------------------------
    @property
    def direct_residual_like(self) -> bool:
        return self.method_kind is MethodKind.ARIADNE

    @property
    def theseus_like_method(self) -> bool:
        return self.method_name in _THESEUS_LIKE

    @property
    def bico_mode(self) -> bool:
        return self.method_name in _BICO_LIKE

    @property
    def blockext_like_method(self) -> bool:
        return self.method_name in (_THESEUS_LIKE | _BICO_LIKE)

    @property
    def transfusion_mode(self) -> bool:
        return self.method_name == "transfusion"

    @property
    def depth_alignment_mode(self) -> str:
        return self.depth_rule.legacy_depth_alignment

    def bind(self, source_depth: int, target_depth: int) -> RunPlan:
        """Post-model guards and prestep flags (needs the real source/target depths)."""
        block_extension_cfg = self.block_extension_cfg
        blockext_like_method = self.blockext_like_method
        validate_residual_completion_depth_direction(
            block_extension_cfg.target_residual_completion,
            source_depth=source_depth,
            target_depth=target_depth,
        )
        run_block_extension_prestep = bool(
            blockext_like_method
            and self.block_extension_enabled
            and self.depth_rule.kind == "brace"
            and source_depth != target_depth
        )
        run_discrete_layer_match_prestep = bool(
            blockext_like_method and self.depth_rule.kind == "discrete_index_match" and source_depth != target_depth
        )
        # Direct-target P1 can write into a native target model at equal depth;
        # in that case it uses an identity layout instead of an ARIADNE resize.
        run_same_depth_direct_target = bool(
            blockext_like_method
            and self.block_extension_enabled
            and source_depth == target_depth
            and block_extension_cfg.target_residual_completion.enabled
            and block_extension_cfg.target_residual_completion.mode == "direct_target"
        )
        if self.depth_rule.kind == "discrete_index_match" and (
            block_extension_cfg.target_residual_completion.enabled
            or block_extension_cfg.joint_blockwise_correction.enabled
            or block_extension_cfg.direct_p1_correction.enabled
        ):
            raise ValueError(
                "depth_alignment='discrete_index_match' is incompatible with target_residual_completion, "
                "joint_blockwise_correction, and direct_p1_correction."
            )
        if block_extension_cfg.joint_blockwise_correction.enabled or block_extension_cfg.direct_p1_correction.enabled:
            if not blockext_like_method:
                raise ValueError("Joint/direct P1 correction requires a Theseus- or BiCo-like transport method")
            if not run_block_extension_prestep:
                raise ValueError(
                    "Joint/direct P1 correction requires a depth-mismatched source/target pair "
                    "so that ARIADNE realizes inserted blocks"
                )
        if self.merge.mode == "merge_then_rebase" and run_block_extension_prestep:
            raise NotImplementedError(
                "merge_then_rebase does not support the block-extension prestep yet: "
                "per-task extended source bases live on different keyspaces and cannot be "
                "merged without a consensus-base step (see transport_then_merge). "
                "Use merge_mode='rebase_then_merge' for depth-mismatch pairs."
            )
        # merge_then_brace_then_transport merges deltas on the native source base first and only
        # then runs its own once-only structural step, so neither prestep fires per-task under it.
        per_task_gate = self.merge.mode != "merge_then_brace_then_transport"
        return RunPlan(
            source_depth=int(source_depth),
            target_depth=int(target_depth),
            run_block_extension_prestep=run_block_extension_prestep,
            run_discrete_layer_match_prestep=run_discrete_layer_match_prestep,
            run_same_depth_direct_target=run_same_depth_direct_target,
            task_block_extension_prestep=bool(run_block_extension_prestep and per_task_gate),
            task_discrete_layer_match_prestep=bool(run_discrete_layer_match_prestep and per_task_gate),
        )


@dataclass(frozen=True)
class RunPlan:
    """Post-model plan: which depth prestep runs, globally and per task."""

    source_depth: int
    target_depth: int
    run_block_extension_prestep: bool
    run_discrete_layer_match_prestep: bool
    run_same_depth_direct_target: bool
    task_block_extension_prestep: bool
    task_discrete_layer_match_prestep: bool


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
    method_kind = MethodKind.of(method_name)
    direct_residual_like = method_kind is MethodKind.ARIADNE
    if direct_residual_like:
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
    blockext_like_method = method_name in (_THESEUS_LIKE | _BICO_LIKE)
    depth_rule = _resolve_depth_rule(cfg)
    if (
        direct_residual_like
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
    block_extension_eval_enabled = bool(block_extension_eval_requested and blockext_like_method)
    block_extension_eval_split = str(cfg.get("block_extension_eval_split", "test")).strip().lower()
    if block_extension_eval_split not in {"val", "test"}:
        raise ValueError("block_extension_eval_split must be one of: val, test")
    block_extension_eval_first_n_batches = block_extension_cfg.first_n_eval_batches
    source_lmc_eval = bool(cfg.get("source_lmc_eval", False))
    source_lmc_eval_split = str(cfg.get("source_lmc_eval_split", "val")).strip().lower()
    if source_lmc_eval_split not in {"val", "test"}:
        raise ValueError("source_lmc_eval_split must be one of: val, test")
    source_lmc_first_n_batches_raw = cfg.get("source_lmc_first_n_batches", None)
    source_lmc_first_n_batches = (
        int(source_lmc_first_n_batches_raw) if source_lmc_first_n_batches_raw is not None else None
    )
    source_lmc_alpha_min = float(cfg.get("source_lmc_alpha_min", 0.0))
    source_lmc_alpha_max = float(cfg.get("source_lmc_alpha_max", 1.0))
    source_lmc_alpha_step = float(cfg.get("source_lmc_alpha_step", 0.05))
    if source_lmc_alpha_step <= 0:
        raise ValueError("source_lmc_alpha_step must be > 0")
    source_lmc_alphas = torch.arange(
        source_lmc_alpha_min,
        source_lmc_alpha_max + source_lmc_alpha_step * 0.5,
        source_lmc_alpha_step,
    ).tolist()
    cross_task_lmc_pairs_raw = cfg.get("cross_task_lmc_pairs", [])
    if cross_task_lmc_pairs_raw is None:
        cross_task_lmc_pairs_raw = []
    if not isinstance(cross_task_lmc_pairs_raw, (list, tuple)):
        raise ValueError("cross_task_lmc_pairs must be a list of two-task lists.")
    cross_task_lmc_pairs: list[tuple[str, str]] = []
    for pair in cross_task_lmc_pairs_raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each cross_task_lmc_pairs item must contain exactly two task names.")
        task_a, task_b = str(pair[0]), str(pair[1])
        if task_a == task_b:
            raise ValueError("cross_task_lmc_pairs cannot interpolate a task with itself.")
        cross_task_lmc_pairs.append((task_a, task_b))
    cross_task_lmc_split = str(cfg.get("cross_task_lmc_eval_split", source_lmc_eval_split)).strip().lower()
    if cross_task_lmc_split not in {"val", "test"}:
        raise ValueError("cross_task_lmc_eval_split must be one of: val, test")
    all_task_lmc_tasks_raw = cfg.get("all_task_lmc_tasks", [])
    if all_task_lmc_tasks_raw is None:
        all_task_lmc_tasks_raw = []
    if not isinstance(all_task_lmc_tasks_raw, (list, tuple)):
        raise ValueError("all_task_lmc_tasks must be a list of task names.")
    all_task_lmc_tasks = [str(task) for task in all_task_lmc_tasks_raw]
    if all_task_lmc_tasks and (len(all_task_lmc_tasks) < 2 or len(set(all_task_lmc_tasks)) != len(all_task_lmc_tasks)):
        raise ValueError("all_task_lmc_tasks must contain at least two distinct task names.")
    all_task_lmc_split = str(cfg.get("all_task_lmc_eval_split", cross_task_lmc_split)).strip().lower()
    if all_task_lmc_split not in {"val", "test"}:
        raise ValueError("all_task_lmc_eval_split must be one of: val, test")
    source_only = bool(cfg.get("source_only", False))
    strict_load = bool(cfg.get("strict_load", False))
    device = str(cfg.get("device", "cuda"))

    if block_extension_eval_requested and not blockext_like_method:
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
    if direct_residual_like and merge_mode in _SINGLE_TRANSPORT_MODES:
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

    return ResolvedRunConfig(
        cfg=cfg,
        method=method,
        method_name=method_name,
        method_params=method_params,
        method_label=method_label,
        method_kind=method_kind,
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
        lmc=SourceLmcSpec(
            block_extension_eval_requested=block_extension_eval_requested,
            block_extension_eval_enabled=block_extension_eval_enabled,
            block_extension_eval_split=block_extension_eval_split,
            block_extension_eval_first_n_batches=block_extension_eval_first_n_batches,
            eval=source_lmc_eval,
            eval_split=source_lmc_eval_split,
            first_n_batches=source_lmc_first_n_batches,
            alphas=source_lmc_alphas,
            cross_task_pairs=cross_task_lmc_pairs,
            cross_task_split=cross_task_lmc_split,
            all_task_tasks=all_task_lmc_tasks,
            all_task_split=all_task_lmc_split,
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
    )
