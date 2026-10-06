"""LLM entry into the shared run-config contract (``rebase.run_config.ResolvedRunConfig`` / ``RunPlan``).

The LLM entrypoint validates in a fixed order that the golden error table pins (some checks fire before any model is
built, the block-extension ones after), so resolution is split the same way: ``resolve_llm_method`` runs before the
models exist, ``resolve_llm_run_config`` right after the block-extension config is resolved. Vision-only fields of
``ResolvedRunConfig`` (suite, alpha grid, source-LMC, grad batches) carry neutral values here: the LLM alpha search,
task list and harness settings stay on ``LlmRuntime``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...merge.methods._common import get_method_params
from ...rebase import get_method
from ...rebase.block_extension.config import BlockExtensionConfig
from ...rebase.capabilities import default_depth_prestep, method_family
from ...rebase.config_schema import canonicalize
from ...rebase.run_config import (
    AlphaSpec,
    DepthRule,
    MergeSpec,
    ResolvedRunConfig,
    RunPlan,
    SourceLmcSpec,
    resolve_depth_rule,
)


def resolve_llm_method(cfg: dict[str, Any]) -> tuple[str, Any, dict[str, Any]]:
    """Pre-model validation: method lookup, the Ariadne stop and the ``method_params.n_batches`` rename."""
    cfg = canonicalize(cfg)
    method_name = str(cfg.get("method", "theseus"))
    method = get_method(method_name)
    method_params = dict(get_method_params({"method_params": cfg.get("method_params", {})}))
    if "n_batches" in method_params:
        raise ValueError(
            "config['method_params'].n_batches is deprecated: it silently "
            "raced with method_params.num_batches (whichever the resolver "
            "checked first won, so the other was ignored without warning). "
            "Rename it to 'num_batches' in the config."
        )
    return method_name, method, method_params


def resolve_llm_run_config(
    cfg: dict[str, Any],
    *,
    method: Any,
    method_name: str,
    method_params: dict[str, Any],
    block_extension_enabled: bool,
    block_extension_cfg: BlockExtensionConfig,
    device: str,
    eval_before_rebase_only: bool,
) -> ResolvedRunConfig:
    """Wrap the already-resolved LLM method / block-extension settings into the shared ``ResolvedRunConfig``.

    The depth rule is resolved by the shared ``resolve_depth_rule`` (per-method defaults behind ``depth_defaults``,
    with the same meaning-changed guard as vision); ``theseus_gqa`` counts as THESEUS for the depth rule.
    """
    default_rule = default_depth_prestep(method_name, "llm")
    if default_rule is not None:
        depth_rule, block_extension_cfg, depth_guard = resolve_depth_rule(
            default_rule, method_name, cfg, block_extension_enabled, block_extension_cfg
        )
    else:
        depth_rule, depth_guard = DepthRule(kind="none"), None
    return ResolvedRunConfig(
        cfg=cfg,
        method=method,
        method_name=method_name,
        method_params=method_params,
        method_label=method_name,
        method_family=method_family(method_name),
        block_extension_enabled=block_extension_enabled,
        block_extension_cfg=block_extension_cfg,
        depth_rule=depth_rule,
        depth_guard=depth_guard,
        ariadne_cfg=None,
        ariadne_preset=None,
        merge=MergeSpec(
            mode="none", method_name=method_name, params=method_params, global_alpha_search=None,
            base_construction="per_task",
        ),
        alpha=AlphaSpec(search=bool(cfg.get("alpha_search", False)), patience=0, search_split="val", alphas=[],
                        selection="shared"),
        lmc=SourceLmcSpec(
            block_extension_eval_requested=False,
            block_extension_eval_enabled=False,
            block_extension_eval_split="test",
            block_extension_eval_first_n_batches=None,
            eval=False,
            eval_split="val",
            first_n_batches=None,
            alphas=[],
            cross_task_pairs=[],
            cross_task_split="val",
            all_task_tasks=[],
            all_task_split="val",
            # eval_before_rebase_only stops each task after its prestep: the one thing ``source_only`` means.
            source_only=eval_before_rebase_only,
        ),
        strict_load=False,
        device=device,
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        suite_name="",
        suite=None,
        tasks=[],
        entrypoint="llm",
    )


def bind_llm_plan(
    resolved: ResolvedRunConfig, *, source_meta: Any, target_meta: Any, source_depth: int, target_depth: int
) -> RunPlan:
    """``resolved.bind`` plus the LLM rule that the prestep needs both family metadata records."""
    plan = resolved.bind(source_depth, target_depth)
    if source_meta is None or target_meta is None:
        if plan.depth_alignment.rule == "brace":
            plan = replace(plan, depth_alignment=replace(plan.depth_alignment, rule="none"))
    return plan
