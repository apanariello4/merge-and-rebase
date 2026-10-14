"""Text calibration corpus for the LLM rebase run (resolved lazily, cached, recorded in the summary)."""

from __future__ import annotations

from copy import deepcopy  # noqa: F401
from dataclasses import replace
from typing import Any

import torch

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ...data.llm_calibration import (
    build_text_calibration_loader as _build_text_calibration_loader,
)
from ...data.llm_calibration import resolve_calibration_texts, tokenization_stats
from ...hyperparam_search import build_search_planner
from ...io.ckpt import load_ckpt, load_into_model
from ...io.text_checkpoints import load_aligned_tuned_from_ref
from ...merge.runtime import to_cpu_fp32
from ...merge.task_vectors import default_key_filter
from ...models.text_lm import TextBuildConfig, TextLM
from ...rebase.base import FitMethod, TransportMethod
from ...rebase.block_extension.config import resolve_block_extension_config, warn_decoder_ignored_fields
from ...rebase.capabilities import check_pair, depth_prestep_methods, uses_calibration
from ...rebase.depth_report import describe_depth_handling
from ...rebase.methods._ariadne.config import resolve_ariadne_decoder_config
from ...rebase.model_families import infer_family
from ...rebase.run_config import ResolvedRunConfig, RunPlan
from .common import resolve_eval_mode, resolve_suite_name, resolve_tasks
from .run_config import bind_llm_plan, resolve_llm_run_config
from .runtime import CalibrationSettings, LlmCalibration, LlmEvaluation, LlmModels, LlmRuntime
from .splits import resolve_harness_slices
from .stages import LlmTaskContext


def _owned_cpu_fp32(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep base snapshots independent of later loads into a CPU fp32 model."""
    out = to_cpu_fp32(sd)
    return {key: value.clone() if value.data_ptr() == sd[key].data_ptr() else value for key, value in out.items()}


class TextCalibrationCache:
    """Resolved on first use: building it from an lm-harness task has to index the task registry, which is far
    too expensive to pay for on a run that never collects activations at all."""

    def __init__(
        self,
        *,
        prompts: Any,
        calibration_dataset_cfg: Any,
        block_extension_cfg: Any,
        harness_tasks: Any,
        n_sequences_cfg: Any,
        n_calib_batches: int,
        calib_batch_size: int,
        calib_max_length: int,
        seed: int,
        include_target: bool,
        tokenizer: Any,
    ) -> None:
        self._prompts = prompts
        self._calibration_dataset_cfg = calibration_dataset_cfg
        self._block_extension_cfg = block_extension_cfg
        self._harness_tasks = harness_tasks
        self._n_sequences_cfg = n_sequences_cfg
        self._n_calib_batches = n_calib_batches
        self._calib_batch_size = calib_batch_size
        self._calib_max_length = calib_max_length
        self._seed = seed
        self._include_target = include_target
        self._tokenizer = tokenizer
        self._cache: list[Any] = []

    def get(self) -> Any:
        if not self._cache:
            block_extension_cfg = self._block_extension_cfg
            resolved = resolve_calibration_texts(
                prompts=self._prompts,
                calibration_dataset=(
                    self._calibration_dataset_cfg
                    or block_extension_cfg.calibration_dataset
                    or block_extension_cfg.calibration_task
                ),
                calibration_split=str(block_extension_cfg.calibration_split),
                harness_tasks=list(self._harness_tasks),
                n_sequences=(
                    int(self._n_sequences_cfg)
                    if self._n_sequences_cfg is not None
                    else max(1, self._n_calib_batches) * self._calib_batch_size
                ),
                seed=self._seed,
                include_target=self._include_target,
            )
            for note in resolved.notes:
                print(f"Calibration note: {note}")
            print(f"Calibration corpus: {resolved.describe()}")
            self._cache.append(resolved)
        return self._cache[0]

    def provenance(self) -> dict[str, Any] | None:
        """Additive summary record (None when no calibration corpus was ever resolved)."""
        if not self._cache:
            return None
        record = self._cache[0].provenance()
        record["calibration_batch_size"] = self._calib_batch_size
        record["calibration_max_length"] = self._calib_max_length
        try:
            record["tokenization"] = tokenization_stats(self._tokenizer, self._cache[0].texts, self._calib_max_length)
        except Exception as exc:  # noqa: BLE001 - provenance must never fail a finished run
            record["tokenization"] = {"error": f"{type(exc).__name__}: {exc}"}
        return record


def _resolve_evaluation(cfg: dict[str, Any], method_params: dict[str, Any]) -> LlmEvaluation:
    """Validate task/checkpoint and evaluation settings without loading either model."""
    eval_before_rebase_only = bool(cfg.get("eval_before_rebase_only", False))
    harness_tasks_raw = cfg.get("harness_tasks", None)
    is_harness_only = harness_tasks_raw is not None and cfg.get("tasks") is None and cfg.get("suite") is None
    harness_tasks_resolved = (
        parse_csv(harness_tasks_raw) if isinstance(harness_tasks_raw, str) else (harness_tasks_raw or [])
    )
    harness_num_fewshot = cfg.get("harness_num_fewshot", 0)
    harness_batch_size = str(cfg.get("harness_batch_size", "auto"))
    harness_limit = cfg.get("harness_limit", None)

    # The "before rebase" reference is the SOURCE model exactly as transport
    # sees it: resized (and LMC-corrected) to the target depth by block
    # extension, before any task vector is transported. The target base is
    # not a useful reference here -- block extension never touches it, so
    # its score says nothing about how much the extension cost us.
    eval_before_rebase = bool(cfg.get("eval_before_rebase", False)) or eval_before_rebase_only
    if eval_before_rebase_only and not harness_tasks_resolved:
        raise ValueError("eval_before_rebase_only needs harness_tasks: there is nothing else to run.")
    run_before_rebase_eval = eval_before_rebase and bool(harness_tasks_resolved)
    if eval_before_rebase and not harness_tasks_resolved:
        print("eval_before_rebase requested but no harness_tasks configured; skipping.")
    if is_harness_only:
        tasks = []
        suite_name = None
    else:
        suite_name = resolve_suite_name(cfg.get("suite", None))
        tasks = resolve_tasks(cfg.get("tasks", None), suite_name=suite_name)

    tuned_bodies_raw = cfg.get("tuned_bodies", None)
    if not tuned_bodies_raw:
        raise ValueError("tuned_bodies config is required (dict task->path or list).")

    if isinstance(tuned_bodies_raw, dict):
        tuned_ref_dict = {str(k).strip().lower(): str(v) for k, v in tuned_bodies_raw.items()}
        if not is_harness_only:
            missing = [t for t in tasks if t not in tuned_ref_dict]
            if missing:
                raise ValueError(f"tuned_bodies missing task keys: {missing}. Provided: {sorted(tuned_ref_dict)}")
            tuned_ref_list = [tuned_ref_dict[t] for t in tasks]
        else:
            # Harness-only: use first tuned body
            tuned_ref_list = [next(iter(tuned_ref_dict.values()))]
    elif isinstance(tuned_bodies_raw, (list, tuple)):
        tuned_ref_list = [str(x) for x in tuned_bodies_raw]
    else:
        raise ValueError("tuned_bodies must be a dict or list.")

    task_heads_path = cfg.get("task_heads", None)
    eval_mode = resolve_eval_mode(str(cfg.get("eval_mode", "auto")), task_heads_path)
    head_key_pattern = str(cfg.get("head_key_pattern", "modules_to_save"))

    allow_prompt_eval = bool(cfg.get("allow_prompt_eval", False))
    if eval_mode == "prompt" and not allow_prompt_eval and not is_harness_only:
        raise ValueError(
            "Prompt evaluation is disabled unless explicitly enabled. "
            "Set --allow-prompt-eval (or config['allow_prompt_eval']=true)."
        )

    _task_contexts(tasks, tuned_ref_list)
    return LlmEvaluation(
        tasks=tasks,
        tuned_ref_list=tuned_ref_list,
        suite_name=suite_name,
        is_harness_only=is_harness_only,
        harness_tasks=list(harness_tasks_resolved),
        harness_num_fewshot=harness_num_fewshot,
        harness_batch_size=harness_batch_size,
        harness_limit=harness_limit,
        slices=resolve_harness_slices(cfg, harness_tasks_resolved),
        run_before_rebase_eval=run_before_rebase_eval,
        eval_before_rebase_only=eval_before_rebase_only,
        eval_mode=eval_mode,
        head_key_pattern=head_key_pattern,
        task_heads_path=task_heads_path,
        search_planner=build_search_planner(cfg=cfg, base_method_params=method_params),
    )


def _resolve_calibration_settings(cfg: dict[str, Any]) -> CalibrationSettings:
    """Validate the calibration inputs independently of model construction."""
    calibration_prompts_cfg = cfg.get("calibration_prompts", None)
    if isinstance(calibration_prompts_cfg, str):
        # Allow pointing at a JSON file shaped {"prompts": [...]}, so a large
        # domain-specific calibration bank doesn't have to be inlined into
        # (and duplicated across) every config.
        calibration_prompts_cfg = load_json(calibration_prompts_cfg).get("prompts", None)
    if calibration_prompts_cfg is not None and not isinstance(calibration_prompts_cfg, list):
        raise ValueError(
            "config['calibration_prompts'] must be a list of strings, or a path to a JSON file holding one."
        )

    batch_size = int(cfg.get("calibration_batch_size", cfg.get("batch_size", 2) or 2))
    max_length = int(cfg.get("calibration_max_length", 128))
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("calibration_batch_size and calibration_max_length must be > 0.")
    n_sequences = cfg.get("calibration_n_sequences")
    if n_sequences is not None and int(n_sequences) <= 0:
        raise ValueError("config['calibration_n_sequences'] must be > 0 when given.")
    return CalibrationSettings(
        prompts=calibration_prompts_cfg,
        dataset=cfg.get("calibration_dataset"),
        n_sequences=int(n_sequences) if n_sequences is not None else None,
        include_target=bool(cfg.get("calibration_include_target", False)),
        batch_size=batch_size,
        max_length=max_length,
    )


def _load_models(
    cfg: dict[str, Any],
    source_build_cfg: TextBuildConfig,
    target_build_cfg: TextBuildConfig,
    device: str,
    num_labels: int,
    method_name: str,
) -> LlmModels:
    source_model_name = cfg.get("source_model_name_or_path")
    target_model_name = cfg.get("target_model_name_or_path")
    print(f"Source model: {source_model_name}")
    print(f"Target model: {target_model_name}")
    print(f"Method: {method_name}")
    print("Building source model...")

    source_llm = TextLM.build(source_build_cfg)
    print("Building target model...")
    target_llm = TextLM.build(target_build_cfg)
    print("Built source and target models.")

    source_base_sd = _owned_cpu_fp32(source_llm.model.state_dict())
    target_base_sd = _owned_cpu_fp32(target_llm.model.state_dict())

    source_base_ckpt = cfg.get("source_base_ckpt", None)
    if source_base_ckpt:
        sd0 = load_ckpt(str(source_base_ckpt))
        load_into_model(source_llm.model, sd0, strict=False)
        source_base_sd = _owned_cpu_fp32(source_llm.model.state_dict())
        print(f"Loaded source base checkpoint from {source_base_ckpt}")

    target_base_ckpt = cfg.get("target_base_ckpt", None)
    if target_base_ckpt:
        sd0 = load_ckpt(str(target_base_ckpt))
        load_into_model(target_llm.model, sd0, strict=False)
        target_base_sd = _owned_cpu_fp32(target_llm.model.state_dict())
        print(f"Loaded target base checkpoint from {target_base_ckpt}")

    source_family = infer_family(source_llm.model)
    target_family = infer_family(target_llm.model)

    tp_keys = source_family.transportable_keys(source_base_sd) if source_family else None
    full_fp_keys = {k for k, v in source_base_sd.items() if default_key_filter(k, v)} if source_family else None
    return LlmModels(
        source_llm=source_llm,
        target_llm=target_llm,
        source_build_cfg=source_build_cfg,
        source_base_sd=source_base_sd,
        target_base_sd=target_base_sd,
        source_family=source_family,
        target_family=target_family,
        device=device,
        num_labels=num_labels,
        tp_keys=tp_keys,
        full_fp_keys=full_fp_keys,
        load_tuned=load_aligned_tuned_from_ref,
    )


def _resolve_plan(
    cfg: dict[str, Any],
    models: LlmModels,
    method_name: str,
    method: TransportMethod | FitMethod,
    method_params: dict[str, Any],
) -> tuple[ResolvedRunConfig, RunPlan, list[str]]:
    source_family, target_family = models.source_family, models.target_family
    source_meta = source_family.metadata(models.source_llm.model) if source_family else None
    target_meta = target_family.metadata(models.target_llm.model) if target_family else None
    device = models.device
    # Block extension config
    depth_prestep_method = method_name in depth_prestep_methods("llm")
    if "block_extension_enabled" not in cfg:
        cfg["block_extension_enabled"] = True
    block_extension_enabled, block_extension_cfg = resolve_block_extension_config(cfg)
    ignored_block_extension_fields = warn_decoder_ignored_fields(cfg.get("block_extension_params"))

    source_depth = source_meta.num_hidden_layers if source_meta else 0
    target_depth = target_meta.num_hidden_layers if target_meta else 0
    depth_mismatch = source_depth != target_depth
    check_pair(
        method_name,
        source_meta,
        target_meta,
        allow_depth_mismatch=bool(depth_prestep_method and block_extension_enabled and depth_mismatch),
    )
    print(f"Capability check passed for {method_name}")

    if source_depth > target_depth and depth_prestep_method and block_extension_enabled:
        shrink_strategies = {
            "per_weight",
            "per-weight",
            "shrink",
            "interpolate_per_weight",
            "interpolate-per-weight",
            "duplicate_per_weight",
            "duplicate-per-weight",
        }
        if block_extension_cfg.extension_strategy not in shrink_strategies:
            raise ValueError(
                "LLM depth shrinking requires a per-weight block-extension strategy. "
                f"Got '{block_extension_cfg.extension_strategy}'."
            )

    eval_before_rebase_only = bool(cfg.get("eval_before_rebase_only", False))
    resolved = resolve_llm_run_config(
        cfg,
        method=method,
        method_name=method_name,
        method_params=method_params,
        block_extension_enabled=block_extension_enabled,
        block_extension_cfg=block_extension_cfg,
        device=device,
        eval_before_rebase_only=eval_before_rebase_only,
    )
    # The per-method depth defaults may have rewritten the block-extension config (THESEUS skip_correction).
    block_extension_cfg = resolved.block_extension_cfg
    if resolved.direct_fit:
        resolved = replace(
            resolved,
            ariadne_cfg=resolve_ariadne_decoder_config(_ariadne_params(cfg), target_family or source_family),
        )
    plan = bind_llm_plan(
        resolved,
        source_meta=source_meta,
        target_meta=target_meta,
        source_depth=source_depth,
        target_depth=target_depth,
    )
    print(describe_depth_handling(resolved, plan, source_depth, target_depth))
    run_block_extension_prestep = plan.run_block_extension_prestep
    if depth_prestep_method:
        if run_block_extension_prestep:
            print(
                f"Block extension preprocess: enabled "
                f"(source_depth={source_depth} -> target_depth={target_depth}, "
                f"strategy={block_extension_cfg.extension_strategy}, "
                f"n_batches_act={block_extension_cfg.n_batches_act})."
            )
        else:
            reason = "disabled by config"
            if not block_extension_enabled:
                reason = "disabled by config"
            elif source_depth == target_depth:
                reason = "source/target depth already match"
            print(
                f"Block extension preprocess: skipped "
                f"({reason}, source_depth={source_depth}, target_depth={target_depth})."
            )

    return resolved, plan, ignored_block_extension_fields


def _build_calibration(
    cfg: dict[str, Any],
    settings: CalibrationSettings,
    models: LlmModels,
    evaluation: LlmEvaluation,
    resolved: ResolvedRunConfig,
    plan: RunPlan,
) -> LlmCalibration:
    """Bind the calibration budgets and document selections to the loaded model pair."""
    batch_budget = resolved.method_params.get("num_batches")
    num_batches = int(batch_budget) if batch_budget is not None else None
    ariadne_batches = int(resolved.ariadne_cfg.num_batches) if resolved.ariadne_cfg is not None else 0
    cache = TextCalibrationCache(
        prompts=settings.prompts,
        calibration_dataset_cfg=settings.dataset,
        block_extension_cfg=resolved.block_extension_cfg,
        harness_tasks=evaluation.harness_tasks,
        n_sequences_cfg=settings.n_sequences,
        n_calib_batches=max(int(resolved.block_extension_cfg.n_batches_act), int(num_batches or 0), ariadne_batches),
        calib_batch_size=settings.batch_size,
        calib_max_length=settings.max_length,
        seed=int(cfg.get("seed", 0)),
        include_target=settings.include_target,
        tokenizer=models.source_llm.tokenizer,
    )
    needs_calibration = plan.run_block_extension_prestep or uses_calibration(resolved.method_name)
    holdout = cache.get().eval_samples or None if needs_calibration else None
    evaluation.slices = resolve_harness_slices(cfg, evaluation.harness_tasks, calibration_holdout=holdout)
    loader = None
    if plan.run_block_extension_prestep:
        loader = _build_text_calibration_loader(
            tokenizer=models.source_llm.tokenizer,
            texts=cache.get().texts,
            batch_size=settings.batch_size,
            max_length=settings.max_length,
        )
    return LlmCalibration(cache, settings.batch_size, settings.max_length, num_batches, loader)


def build_runtime(
    cfg: dict[str, Any],
    *,
    method_name: str,
    method: TransportMethod | FitMethod,
    method_params: dict[str, Any],
    source_build_cfg: TextBuildConfig,
    target_build_cfg: TextBuildConfig,
    device: str,
    num_labels: int,
) -> LlmRuntime:
    """Validate inputs, load models, resolve geometry, then bind calibration and evaluation."""
    evaluation = _resolve_evaluation(cfg, method_params)
    settings = _resolve_calibration_settings(cfg)
    models = _load_models(cfg, source_build_cfg, target_build_cfg, device, num_labels, method_name)
    resolved, plan, ignored = _resolve_plan(cfg, models, method_name, method, method_params)
    print(f"Using eval_mode={evaluation.eval_mode}")
    print("\nComputing task deltas...")
    if models.tp_keys:
        print(f"Transportable body keys: {len(models.tp_keys)} / {len(models.full_fp_keys)} total FP keys")
    calibration = _build_calibration(cfg, settings, models, evaluation, resolved, plan)
    return LlmRuntime(
        cfg=cfg,
        models=models,
        calibration=calibration,
        evaluation=evaluation,
        resolved=resolved,
        plan=plan,
        ignored_block_extension_fields=ignored,
        task_contexts=_task_contexts(evaluation.tasks, evaluation.tuned_ref_list),
    )


def _task_contexts(tasks, tuned_ref_list) -> dict[str, LlmTaskContext]:
    """One context per tuned body; with named tasks the two lists must have equal length."""
    if len(tasks) != len(set(tasks)):
        duplicates = sorted({task for task in tasks if tasks.count(task) > 1})
        raise ValueError(f"Task names must be unique; duplicate labels: {duplicates}.")
    if tasks and len(tasks) != len(tuned_ref_list):
        raise ValueError(
            f"{len(tuned_ref_list)} tuned bodies for {len(tasks)} tasks: every tuned body needs exactly one task name"
        )
    return {
        (tasks[i] if tasks else f"task_{i}"): LlmTaskContext(ckpt_ref=ref, index=i)
        for i, ref in enumerate(tuned_ref_list)
    }


def _ariadne_params(cfg) -> dict:
    """``ariadne_params`` (or its legacy alias ``direct_residual_params``); both at once is an error."""
    if cfg.get("ariadne_params") is not None and cfg.get("direct_residual_params") is not None:
        raise ValueError("Set either ariadne_params or direct_residual_params, not both.")
    raw = cfg.get("ariadne_params")
    if raw is None:
        raw = cfg.get("direct_residual_params")
    return dict(raw or {})
