"""Text calibration corpus for the LLM rebase run (resolved lazily, cached, recorded in the summary)."""

from __future__ import annotations

from copy import deepcopy  # noqa: F401
from typing import Any

import torch

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ...data.llm_calibration import (
    build_text_calibration_loader as _build_text_calibration_loader,
)
from ...data.llm_calibration import resolve_calibration_texts, tokenization_stats
from ...io.ckpt import load_ckpt, load_into_model
from ...io.text_checkpoints import load_aligned_tuned_from_ref
from ...merge.runtime import to_cpu_fp32
from ...merge.task_vectors import default_key_filter
from ...models.text_lm import TextBuildConfig, TextLM
from ...rebase.block_extension.config import resolve_block_extension_config, warn_decoder_ignored_fields
from ...rebase.capabilities import check_pair
from ...rebase.model_families import infer_family
from .common import resolve_eval_mode, resolve_suite_name, resolve_tasks
from .pipeline import LlmRuntime
from .run_config import bind_llm_plan, resolve_llm_run_config
from .stages import LlmTaskContext


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


def build_runtime(
    cfg: dict[str, Any],
    *,
    method_name: str,
    method: Any,
    method_params: dict[str, Any],
    source_build_cfg: TextBuildConfig,
    target_build_cfg: TextBuildConfig,
    device: str,
    num_labels: int,
) -> LlmRuntime:
    """Build both models, resolve block extension / harness / tuned-body / calibration settings (former main body)."""
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

    source_base_sd = to_cpu_fp32(source_llm.model.state_dict())
    target_base_sd = to_cpu_fp32(target_llm.model.state_dict())

    source_base_ckpt = cfg.get("source_base_ckpt", None)
    if source_base_ckpt:
        sd0 = load_ckpt(str(source_base_ckpt))
        load_into_model(source_llm.model, sd0, strict=False)
        source_base_sd = to_cpu_fp32(source_llm.model.state_dict())
        print(f"Loaded source base checkpoint from {source_base_ckpt}")

    target_base_ckpt = cfg.get("target_base_ckpt", None)
    if target_base_ckpt:
        sd0 = load_ckpt(str(target_base_ckpt))
        load_into_model(target_llm.model, sd0, strict=False)
        target_base_sd = to_cpu_fp32(target_llm.model.state_dict())
        print(f"Loaded target base checkpoint from {target_base_ckpt}")

    source_family = infer_family(source_llm.model)
    target_family = infer_family(target_llm.model)
    source_meta = source_family.metadata(source_llm.model) if source_family else None
    target_meta = target_family.metadata(target_llm.model) if target_family else None

    # Block extension config
    blockext_like_method = method_name in {"theseus", "theseus_gqa", "bico"}
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
        allow_depth_mismatch=bool(blockext_like_method and block_extension_enabled and depth_mismatch),
    )
    print(f"Capability check passed for {method_name}")

    if source_depth > target_depth and blockext_like_method and block_extension_enabled:
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
    plan = bind_llm_plan(
        resolved,
        source_meta=source_meta,
        target_meta=target_meta,
        source_depth=source_depth,
        target_depth=target_depth,
    )
    run_block_extension_prestep = plan.run_block_extension_prestep
    if blockext_like_method:
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

    harness_tasks_raw = cfg.get("harness_tasks", None)
    is_harness_only = harness_tasks_raw is not None and cfg.get("tasks") is None and cfg.get("suite") is None
    harness_tasks_resolved = (
        parse_csv(harness_tasks_raw)
        if isinstance(harness_tasks_raw, str)
        else (harness_tasks_raw or [])
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
        raise ValueError(
            "eval_before_rebase_only needs harness_tasks: there is nothing else to run."
        )
    run_before_rebase_eval = eval_before_rebase and bool(harness_tasks_resolved)
    if eval_before_rebase and not harness_tasks_resolved:
        print("eval_before_rebase requested but no harness_tasks configured; skipping.")
    baseline_harness_results_by_task: dict[str, dict[str, float]] = {}

    def _baseline_summary() -> dict[str, Any] | None:
        """Flat for a single task vector, label -> results for several."""
        if not baseline_harness_results_by_task:
            return None
        if len(baseline_harness_results_by_task) == 1:
            return next(iter(baseline_harness_results_by_task.values()))
        return dict(baseline_harness_results_by_task)

    def _eval_before_rebase(model: torch.nn.Module, label: str) -> None:
        from .harness import run as run_harness

        print(f"\nEvaluating source model ({label}) with lm-harness (before rebase)...")
        results = run_harness(
            tasks=list(harness_tasks_resolved),
            model=model,
            tokenizer=source_llm.tokenizer,
            device=device,
            num_fewshot=harness_num_fewshot,
            batch_size=harness_batch_size,
            limit=harness_limit,
            samples=harness_samples,
        )
        for task_name, acc in results.items():
            print(f"  [before rebase / {label}] {task_name}: {acc:.4f}")
        baseline_harness_results_by_task[label] = results

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
        tuned_ref_dict = {
            str(k).strip().lower(): str(v)
            for k, v in tuned_bodies_raw.items()
        }
        if not is_harness_only:
            missing = [t for t in tasks if t not in tuned_ref_dict]
            if missing:
                raise ValueError(
                    f"tuned_bodies missing task keys: {missing}. "
                    f"Provided: {sorted(tuned_ref_dict)}"
                )
            tuned_ref_list = [tuned_ref_dict[t] for t in tasks]
        else:
            # Harness-only: use first tuned body
            tuned_ref_list = [next(iter(tuned_ref_dict.values()))]
    elif isinstance(tuned_bodies_raw, (list, tuple)):
        tuned_ref_list = [str(x) for x in tuned_bodies_raw]
    else:
        raise ValueError("tuned_bodies must be a dict or list.")

    task_heads_path = cfg.get("task_heads", None)
    eval_mode = resolve_eval_mode(
        str(cfg.get("eval_mode", "auto")), task_heads_path
    )
    print(f"Using eval_mode={eval_mode}")
    head_key_pattern = str(cfg.get("head_key_pattern", "modules_to_save"))

    allow_prompt_eval = bool(cfg.get("allow_prompt_eval", False))
    if eval_mode == "prompt" and not allow_prompt_eval and not is_harness_only:
        raise ValueError(
            "Prompt evaluation is disabled unless explicitly enabled. "
            "Set --allow-prompt-eval (or config['allow_prompt_eval']=true)."
        )

    # Compute per-task source deltas
    print("\nComputing task deltas...")

    tp_keys = None
    full_fp_keys = None
    if source_family is not None:
        tp_keys = source_family.transportable_keys(source_base_sd)
        full_fp_keys = {
            k for k, v in source_base_sd.items()
            if isinstance(v, torch.Tensor) and default_key_filter(k, v)
        }
        if tp_keys:
            print(f"Transportable body keys: {len(tp_keys)} / {len(full_fp_keys)} total FP keys")

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

    # Calibration text comes from the dataset the run is actually scored
    # on (or an explicitly configured one), not from a fixed prompt bank:
    # size the slice to what the run will consume so num_batches is real.
    calib_batch_size = int(cfg.get("calibration_batch_size", cfg.get("batch_size", 2) or 2))
    calib_max_length = int(cfg.get("calibration_max_length", 128))
    # method_params.num_batches is the one calibration-budget knob theseus/bico
    # read from config (method_params.n_batches is rejected above). Resolve it
    # once here so it's counted when sizing the corpus and can beat the
    # _DEFAULT_CALIB_BATCHES default injected at the transport call below.
    # Vision configs express the calibration budget as
    # method_params.num_batches, and theseus/bico accept either name
    # (preferring n_batches). Resolve it once here so a vision-style config
    # means the same thing on this path: without this, num_batches was
    # neither counted when sizing the corpus nor able to beat the n_batches
    # default injected at the transport call, so it silently did nothing.
    calib_n_batches_cfg = method_params.get("n_batches", method_params.get("num_batches"))
    calib_n_batches = int(calib_n_batches_cfg) if calib_n_batches_cfg is not None else None
    # These are two separate budgets over one shared text pool, not one
    # knob: block extension consumes n_batches_act batches for its
    # activation capture, theseus/bico consume num_batches for theirs, and
    # neither is derived from the other. The max only sizes the pool, so
    # whichever consumer asks for more still finds enough text.
    n_calib_batches = max(
        int(block_extension_cfg.n_batches_act),
        int(calib_n_batches or 0),
    )
    # Opt-in calibration knobs (all default to the historical behaviour):
    #   calibration_n_sequences      explicit pool size instead of max(batches) * batch_size
    #   calibration_dataset          explicit HF corpus, decoupled from the evaluated task (no eval hold-out);
    #                                beats block_extension_params.calibration_dataset
    #   calibration_include_target   harness source: append the gold target to each rendered prompt
    calibration_n_sequences_cfg = cfg.get("calibration_n_sequences", None)
    if calibration_n_sequences_cfg is not None and int(calibration_n_sequences_cfg) <= 0:
        raise ValueError("config['calibration_n_sequences'] must be > 0 when given.")
    calibration_dataset_cfg = cfg.get("calibration_dataset", None)
    calibration_include_target = bool(cfg.get("calibration_include_target", False))
    _calibration_cache = TextCalibrationCache(
        prompts=calibration_prompts_cfg,
        calibration_dataset_cfg=calibration_dataset_cfg,
        block_extension_cfg=block_extension_cfg,
        harness_tasks=harness_tasks_resolved,
        n_sequences_cfg=calibration_n_sequences_cfg,
        n_calib_batches=n_calib_batches,
        calib_batch_size=calib_batch_size,
        calib_max_length=calib_max_length,
        seed=int(cfg.get("seed", 0)),
        include_target=calibration_include_target,
        tokenizer=source_llm.tokenizer,
    )
    _calibration = _calibration_cache.get
    _calibration_provenance = _calibration_cache.provenance

    configured_harness_samples_raw = cfg.get("harness_samples", None)
    configured_harness_samples: dict[str, list[int]] | None = None
    if configured_harness_samples_raw is not None:
        if not isinstance(configured_harness_samples_raw, dict):
            raise ValueError("config['harness_samples'] must map task names to document-index lists.")
        configured_harness_samples = {}
        for task_name, indices in configured_harness_samples_raw.items():
            if not isinstance(indices, list) or not all(isinstance(i, int) and i >= 0 for i in indices):
                raise ValueError(
                    "config['harness_samples'] values must be lists of non-negative document indices."
                )
            configured_harness_samples[str(task_name)] = list(indices)

    needs_calibration = run_block_extension_prestep or method_name in (
        "theseus",
        "theseus_gqa",
        "bico",
    )
    # The eval slice must be known before the first before-rebase eval, so
    # resolve up front whenever this run will calibrate at all.
    calibration_eval_samples = _calibration().eval_samples or None if needs_calibration else None
    if (
        configured_harness_samples is not None
        and calibration_eval_samples is not None
        and configured_harness_samples != calibration_eval_samples
    ):
        raise ValueError(
            "config['harness_samples'] disagrees with the IFEval hold-out derived from calibration; "
            "use the derived samples or an independent calibration corpus."
        )
    harness_samples = configured_harness_samples or calibration_eval_samples

    # Block extension: build calibration loader once if needed
    blockext_calib_loader = None
    if run_block_extension_prestep:
        blockext_calib_loader = _build_text_calibration_loader(
            tokenizer=source_llm.tokenizer,
            texts=_calibration().texts,
            batch_size=calib_batch_size,
            max_length=calib_max_length,
        )
    family_adapter = target_family or source_family
    return LlmRuntime(
        cfg=cfg,
        method_name=method_name,
        method=method,
        method_params=method_params,
        source_llm=source_llm,
        target_llm=target_llm,
        source_build_cfg=source_build_cfg,
        source_base_sd=source_base_sd,
        target_base_sd=target_base_sd,
        source_family=source_family,
        target_family=target_family,
        family_adapter=family_adapter,
        device=device,
        num_labels=num_labels,
        block_extension_cfg=block_extension_cfg,
        run_block_extension_prestep=run_block_extension_prestep,
        source_depth=source_depth,
        target_depth=target_depth,
        tuned_ref_list=tuned_ref_list,
        tasks=tasks,
        suite_name=suite_name,
        is_harness_only=is_harness_only,
        harness_tasks_resolved=harness_tasks_resolved,
        harness_num_fewshot=harness_num_fewshot,
        harness_batch_size=harness_batch_size,
        harness_limit=harness_limit,
        harness_samples=harness_samples,
        run_before_rebase_eval=run_before_rebase_eval,
        eval_before_rebase_only=eval_before_rebase_only,
        ignored_block_extension_fields=ignored_block_extension_fields,
        _calibration=_calibration,
        _calibration_provenance=_calibration_provenance,
        calib_batch_size=calib_batch_size,
        calib_max_length=calib_max_length,
        calib_n_batches=calib_n_batches,
        blockext_calib_loader=blockext_calib_loader,
        tp_keys=tp_keys,
        full_fp_keys=full_fp_keys,
        eval_mode=eval_mode,
        head_key_pattern=head_key_pattern,
        task_heads_path=task_heads_path,
        _eval_before_rebase=_eval_before_rebase,
        _baseline_summary=_baseline_summary,
        load_tuned=load_aligned_tuned_from_ref,
        resolved=resolved,
        plan=plan,
        task_contexts={
            (tasks[i] if i < len(tasks) else f"task_{i}"): LlmTaskContext(ckpt_ref=ref, index=i)
            for i, ref in enumerate(tuned_ref_list)
        },
    )
