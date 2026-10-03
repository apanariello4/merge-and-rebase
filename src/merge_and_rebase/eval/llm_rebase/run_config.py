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
from ...rebase.registry import canonical_method_name
from ...rebase.run_config import (
    AlphaSpec,
    DepthRule,
    MergeSpec,
    MethodKind,
    ResolvedRunConfig,
    RunPlan,
    SourceLmcSpec,
)

LLM_BLOCKEXT_METHODS = frozenset({"theseus", "theseus_gqa", "bico"})


def resolve_llm_method(cfg: dict[str, Any]) -> tuple[str, Any, dict[str, Any]]:
    """Pre-model validation: method lookup, the Ariadne stop and the ``method_params.n_batches`` rename."""
    method_name = str(cfg.get("method", "theseus"))
    method = get_method(method_name)
    if canonical_method_name(method_name) == "ariadne":
        # Capability-supported, but llm_rebase has no Ariadne branch until S10: fail before loading any model.
        raise ValueError("Ariadne LLM entrypoint lands in S10; llm_rebase does not run Ariadne yet.")
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
    """Wrap the already-resolved LLM method / block-extension settings into the shared ``ResolvedRunConfig``."""
    blockext_like = method_name in LLM_BLOCKEXT_METHODS
    return ResolvedRunConfig(
        cfg=cfg,
        method=method,
        method_name=method_name,
        method_params=method_params,
        method_label=method_name,
        method_kind=MethodKind.of(method_name),
        block_extension_enabled=block_extension_enabled,
        block_extension_cfg=block_extension_cfg,
        depth_rule=DepthRule(kind="brace" if blockext_like and block_extension_enabled else "none"),
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
        blockext_methods=LLM_BLOCKEXT_METHODS,
    )


def bind_llm_plan(
    resolved: ResolvedRunConfig, *, source_meta: Any, target_meta: Any, source_depth: int, target_depth: int
) -> RunPlan:
    """``resolved.bind`` plus the LLM rule that the prestep needs both family metadata records."""
    plan = resolved.bind(source_depth, target_depth)
    if source_meta is None or target_meta is None:
        plan = replace(plan, run_block_extension_prestep=False, task_block_extension_prestep=False)
    return plan
