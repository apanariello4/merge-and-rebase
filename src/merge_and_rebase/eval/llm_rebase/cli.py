from __future__ import annotations

import argparse
import json
import time
from copy import copy, deepcopy
from dataclasses import dataclass, is_dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.hyperparam_search import (
    build_search_planner,
)
from merge_and_rebase.utils.helpers import load_json, parse_csv

from ...cli_args import (
    add_alpha_args,
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_suite_arg,
    add_tasks_arg,
    build_logging_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ...data.llm_calibration import (
    TokenizedPromptDataset,
    build_text_calibration_loader,
)
from ...data.text_loaders import (
    NLI_TASKS,
    NLITaskData,
    NLITokenizedData,
    build_nli_task_data,
    build_nli_tokenized_loader,
)
from ...io.ckpt import load_ckpt, load_into_model
from ...io.text_checkpoints import load_aligned_tuned_from_ref
from ...merge.methods._common import get_method_params
from ...merge.runtime import (
    compose_weighted_deltas,
    to_cpu_fp32,
)
from ...merge.task_vectors import TaskVector, default_key_filter
from ...models.text_lm import TextBuildConfig, TextLM
from ...rebase import get_method
from ...rebase.block_extension.config import resolve_block_extension_config, warn_decoder_ignored_fields
from ...rebase.block_extension.decoder import run_block_extension_llm
from ...rebase.capabilities import check_pair
from ...rebase.model_families import infer_family
from ...rebase.registry import canonical_method_name
from ...run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ..print_utils import pretty_print_task_accuracies
from .alpha_search import harness_alpha_search, nli_alpha_search, score_harness_test_slice
from .artifacts import save_merged_state
from .common import (
    head_class_ids_for_task,
    load_task_heads,
    resolve_eval_mode,
    resolve_fine_tuned_acc,
    resolve_suite_name,
    resolve_task_mask_class,
    resolve_tasks,
    to_unit_acc,
)
from .context import TextCalibrationCache
from .merge import (
    _summarize_merged_delta,
    norm_match_transported,
    report_merged_delta,
    resolve_delta_source,
    resolve_norm_match,
)
from .summary import assemble_harness_summary, assemble_nli_summary


@dataclass
class _PreparedTaskDelta:
    """Delta together with the exact source context it was prepared from."""

    delta: dict[str, torch.Tensor]
    source_base: dict[str, torch.Tensor]
    transport_keys: set[str]
    source_model: torch.nn.Module
    # The same task vector resized without correction. Under lmc_mode="shared"
    # the fitted correction W is applied to base and ft alike, so the corrected
    # delta is W @ delta: correction changes the task vector's scale as well as
    # its direction. Keeping the uncorrected delta lets a run transport one and
    # normalize to the other, separating those two effects.
    uncorrected_delta: dict[str, torch.Tensor] | None = None
    # Realized block chain from the resize, needed by residual completion to
    # address inserted positions by ancestry instead of a depth pattern.
    extension_layout: dict[str, Any] | None = None
    # Proposal-1 native reference banks, captured before the resize.



# Calibration batches used by theseus/bico when a config names neither
# method_params.num_batches nor method_params.n_batches.
_DEFAULT_CALIB_BATCHES = 2


def _config_with_correction_disabled(config: Any) -> Any | None:
    """Same block-extension config with correction off, or None if not derivable.

    Real runs pass a BlockExtensionConfig dataclass. Tests pass lightweight
    stubs, and a stub that cannot express skip_correction simply means no
    uncorrected reference vector is available for that call.
    """
    if is_dataclass(config) and not isinstance(config, type):
        return dataclass_replace(config, skip_correction=True)
    clone = copy(config)
    try:
        clone.skip_correction = True
    except (AttributeError, TypeError):
        return None
    return clone


def _prepare_resized_task_delta(
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    calibration_loader: Any,
    target_layers_total: int,
    config: Any,
    family_adapter: Any,
    device: str,
) -> _PreparedTaskDelta:
    """Resize one task pair and retain the exact source context for transport."""
    # Reference resize with correction disabled, kept so the caller can compare
    # or substitute the uncorrected task vector. This costs a deepcopy and no
    # forward passes: with skip_correction the extension only duplicates blocks,
    # it never captures reference or component activations.
    uncorrected_delta: dict[str, torch.Tensor] | None = None
    reference_config = _config_with_correction_disabled(config)
    if reference_config is not None and not bool(getattr(config, "skip_correction", False)):
        ref_base = deepcopy(source_base_model)
        ref_ft = deepcopy(source_ft_model)
        run_block_extension_llm(
            source_base_model=ref_base,
            source_ft_model=ref_ft,
            calibration_loader=calibration_loader,
            target_layers_total=target_layers_total,
            config=reference_config,
            family_adapter=family_adapter,
            device=device,
        )
        uncorrected_delta = TaskVector.from_checkpoints(
            to_cpu_fp32(ref_base.state_dict()), to_cpu_fp32(ref_ft.state_dict()), strict=False
        ).delta
        del ref_base, ref_ft

    extension_layout: dict[str, Any] = {}
    final_depth = run_block_extension_llm(
        source_base_model=source_base_model,
        source_ft_model=source_ft_model,
        calibration_loader=calibration_loader,
        target_layers_total=target_layers_total,
        config=config,
        family_adapter=family_adapter,
        device=device,
        layout_out=extension_layout,
    )
    if final_depth != target_layers_total:
        raise RuntimeError(
            "Block extension preprocess failed: "
            f"final_depth={final_depth}, expected={target_layers_total}."
        )

    source_base = to_cpu_fp32(source_base_model.state_dict())
    source_ft = to_cpu_fp32(source_ft_model.state_dict())
    task_vector = TaskVector.from_checkpoints(source_base, source_ft, strict=False)
    if uncorrected_delta is None:
        # skip_correction: the corrected and uncorrected resizes are the same run.
        uncorrected_delta = task_vector.delta
    return _PreparedTaskDelta(
        delta=task_vector.delta,
        source_base=source_base,
        transport_keys=set(family_adapter.transportable_keys(source_base)),
        source_model=source_base_model,
        uncorrected_delta=uncorrected_delta,
        extension_layout=extension_layout or None,
    )



# Promoted to data/llm_calibration (P7.S5); the old names stay importable (golden tests import them).
_build_text_calibration_loader = build_text_calibration_loader
_TokenizedPromptDataset = TokenizedPromptDataset


def main() -> None:
    run_logger = None
    try:
        p = argparse.ArgumentParser("Rebase LLM task vectors and evaluate.")
        add_config_arg(p)
        add_suite_arg(p, choices=["nli6"], default=None)

        # Source / target model
        p.add_argument("--source-model-name-or-path", type=str, default=None)
        p.add_argument("--target-model-name-or-path", type=str, default=None)
        p.add_argument("--source-base-ckpt", type=str, default=None)
        p.add_argument("--target-base-ckpt", type=str, default=None)
        p.add_argument("--model-arch", type=str, default=None, choices=["llama", "qwen", "t5", "auto"])
        p.add_argument("--model-kind", type=str, default=None, choices=["causal_lm", "sequence_classification"])
        p.add_argument("--num-labels", type=int, default=None)
        add_device_dtype_args(p, device_default=None, dtype_default=None)
        p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--use-fast-tokenizer", action=argparse.BooleanOptionalAction, default=None)

        # Tuned checkpoints
        p.add_argument("--tuned-bodies", type=str, nargs="+", default=None)
        p.add_argument("--task-heads", type=str, default=None)
        p.add_argument("--head-key-pattern", type=str, default=None)

        # Method
        p.add_argument("--method", type=str, default=None)
        p.add_argument("--method-params", type=str, default=None)

        # Prealign
        p.add_argument("--prealign-enabled", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--prealign-strategy", type=str, default=None)

        # Alpha
        add_alpha_args(
            p,
            alpha_default=None,
            alpha_min_default=None,
            alpha_max_default=None,
            alpha_step_default=None,
            alpha_search_default=None,
        )

        # Eval
        add_tasks_arg(p, default=None, help_text=f"NLI tasks: {', '.join(NLI_TASKS)} or 'all'")
        p.add_argument("--eval-mode", type=str, default=None, choices=["auto", "prompt", "head_logits"])
        p.add_argument("--split", type=str, default=None, choices=["train", "validation", "test"])
        p.add_argument("--max-samples-per-task", type=int, default=None)
        p.add_argument("--prompt-template", type=str, default=None)
        p.add_argument("--max-prompt-tokens", type=int, default=None)
        p.add_argument("--print-every", type=int, default=None)
        p.add_argument("--allow-prompt-eval", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--max-length", type=int, default=None)
        p.add_argument("--eval-single-task-tuned", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--fine-tuned-acc-json", type=str, default=None)
        p.add_argument("--save-merged", type=str, default=None)

        # Harness
        p.add_argument("--harness-tasks", type=str, default=None)
        # None, not 0/"auto": merge_non_none only lets a CLI value win over the
        # config when the flag was actually passed. A concrete default here
        # would silently clobber the config's harness_num_fewshot/
        # harness_batch_size on every run, whether or not the flag was given.
        p.add_argument("--harness-num-fewshot", type=int, default=None)
        p.add_argument("--harness-batch-size", type=str, default=None)
        p.add_argument("--harness-limit", type=int, default=None)

        # Block extension
        p.add_argument(
            "--block-extension-enabled",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable block-extension preprocess before task-vector transport when depth mismatch is found.",
        )
        p.add_argument(
            "--block-extension-params",
            type=str,
            default=None,
            help="JSON object for block-extension preprocess kwargs.",
        )
        p.add_argument(
            "--eval-before-rebase",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Optionally evaluate source model before rebase for pre/post comparison.",
        )
        p.add_argument(
            "--eval-before-rebase-only",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Run block extension and the before-rebase eval, then stop before "
                "transport. Implies --eval-before-rebase."
            ),
        )

        add_logging_args(p)
        args = p.parse_args()

        method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")
        block_extension_params_cli = parse_json_object_arg(
            args.block_extension_params, arg_name="--block-extension-params"
        )

        cfg: dict[str, Any] = {}
        if args.config is not None:
            cfg = load_json(args.config)

        cli_overrides = {
            "source_model_name_or_path": args.source_model_name_or_path,
            "target_model_name_or_path": args.target_model_name_or_path,
            "source_base_ckpt": args.source_base_ckpt,
            "target_base_ckpt": args.target_base_ckpt,
            "model_arch": args.model_arch,
            "model_kind": args.model_kind,
            "num_labels": args.num_labels,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "use_fast_tokenizer": args.use_fast_tokenizer,
            "tuned_bodies": args.tuned_bodies,
            "task_heads": args.task_heads,
            "head_key_pattern": args.head_key_pattern,
            "method": args.method,
            "method_params": method_params_cli,
            "prealign_enabled": args.prealign_enabled,
            "prealign_strategy": args.prealign_strategy,
            "alpha_search": args.alpha_search,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_step": args.alpha_step,
            "alpha": args.alpha,
            "eval_mode": args.eval_mode,
            "tasks": args.tasks,
            "suite": args.suite,
            "split": args.split,
            "max_samples_per_task": args.max_samples_per_task,
            "prompt_template": args.prompt_template,
            "max_prompt_tokens": args.max_prompt_tokens,
            "print_every": args.print_every,
            "allow_prompt_eval": args.allow_prompt_eval,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_length": args.max_length,
            "eval_single_task_tuned": args.eval_single_task_tuned,
            "fine_tuned_acc": (
                json.loads(args.fine_tuned_acc_json) if args.fine_tuned_acc_json else None
            ),
            "save_merged": args.save_merged,
            "harness_tasks": args.harness_tasks,
            "harness_num_fewshot": args.harness_num_fewshot,
            "harness_batch_size": args.harness_batch_size,
            "harness_limit": args.harness_limit,
            "block_extension_enabled": args.block_extension_enabled,
            "block_extension_params": block_extension_params_cli,
            "eval_before_rebase": args.eval_before_rebase,
            "eval_before_rebase_only": args.eval_before_rebase_only,
        }
        cfg = merge_non_none(cfg, cli_overrides)

        logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
        cfg["logging"] = logging_cfg

        source_model_name = cfg.get("source_model_name_or_path")
        target_model_name = cfg.get("target_model_name_or_path")
        if not source_model_name or not target_model_name:
            raise ValueError(
                "Both source_model_name_or_path and target_model_name_or_path are required."
            )

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

        model_arch = str(cfg.get("model_arch", "auto"))
        model_kind = str(cfg.get("model_kind", "causal_lm"))
        num_labels = int(cfg.get("num_labels", 3))
        device = str(cfg.get("device", "cuda"))
        dtype = cfg.get("dtype", None)

        source_build_cfg = TextBuildConfig(
            model_name_or_path=str(source_model_name),
            model_arch=model_arch,
            device=device,
            dtype=dtype,
            model_kind=model_kind,
            num_labels=num_labels,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            use_fast_tokenizer=bool(cfg.get("use_fast_tokenizer", True)),
        )
        target_build_cfg = TextBuildConfig(
            model_name_or_path=str(target_model_name),
            model_arch=model_arch,
            device=device,
            dtype=dtype,
            model_kind=model_kind,
            num_labels=num_labels,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            use_fast_tokenizer=bool(cfg.get("use_fast_tokenizer", True)),
        )

        run_logger = start_run(
            entrypoint="eval.llm_rebase",
            logging_cfg=logging_cfg,
            summary_path=default_summary_path(
                entrypoint="eval.llm_rebase",
                logging_cfg=logging_cfg,
                default_parent=(
                    Path(str(cfg["save_merged"])).parent if cfg.get("save_merged") else None
                ),
            ),
            metadata={"config_path": args.config, "resolved_config": cfg},
        )

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

        run_block_extension_prestep = bool(
            blockext_like_method
            and block_extension_enabled
            and source_meta is not None
            and target_meta is not None
            and depth_mismatch
        )
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
        eval_before_rebase_only = bool(cfg.get("eval_before_rebase_only", False))
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
        prepared_tasks: list[_PreparedTaskDelta] = []

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

        if run_before_rebase_eval and not run_block_extension_prestep:
            # No depth change: the model transport starts from is the plain
            # source base, so one pass is enough for every task.
            load_into_model(source_llm.model, source_base_sd, strict=False)
            _eval_before_rebase(source_llm.model, "source_base")
            source_llm.model.to("cpu")
        elif run_before_rebase_eval and bool(cfg.get("eval_source_before_extension", False)):
            # The unextended source, scored on the same eval slice. Without it
            # the only "before" number is the extended source base, so there is
            # nothing to say whether correction restores the original model or
            # improves on it -- the two are indistinguishable from the extended
            # score alone.
            load_into_model(source_llm.model, source_base_sd, strict=False)
            _eval_before_rebase(source_llm.model, "source_base_unextended")
            source_llm.model.to("cpu")

        for task_idx, ckpt_ref in enumerate(tuned_ref_list):
            task_label = tasks[task_idx] if task_idx < len(tasks) else f"task_{task_idx}"
            if run_block_extension_prestep:
                # Each task starts from an immutable source template, then its
                # own copy is resized to the target depth before transport.
                # Keep that depth-matched source model alive below.
                source_base_model_task = deepcopy(source_llm.model)
                source_ft_model_task = deepcopy(source_llm.model)

                # Load tuned checkpoint into ft model
                aligned = load_aligned_tuned_from_ref(
                    ckpt_ref=ckpt_ref,
                    base_sd=source_base_sd,
                    build_cfg=source_build_cfg,
                    model=source_ft_model_task,
                    prefer_lora_view=False,
                )
                tuned_sd = to_cpu_fp32(aligned) if isinstance(aligned, dict) else {k: v.cpu() for k, v in aligned.items()}
                load_into_model(source_ft_model_task, tuned_sd, strict=False)

                family_adapter_for_ext = target_family or source_family
                if family_adapter_for_ext is None:
                    raise ValueError("Block extension requires a family adapter but none was inferred.")

                prepared_task = _prepare_resized_task_delta(
                    source_base_model=source_base_model_task,
                    source_ft_model=source_ft_model_task,
                    calibration_loader=blockext_calib_loader,
                    target_layers_total=int(target_depth),
                    config=block_extension_cfg,
                    family_adapter=family_adapter_for_ext,
                    device=device,
                )
                print(f"  block extension completed (source_depth={source_depth} -> {target_depth})")
                # The resized ft model has already been absorbed into the delta;
                # drop it before the eval below so it is not holding device
                # memory while lm-harness runs.
                del source_ft_model_task
                if run_before_rebase_eval:
                    # Scored here, after extension and before transport: this is
                    # the extended source base that BiCo/Theseus will read from.
                    _eval_before_rebase(
                        source_base_model_task, f"extended_source_base:{task_label}"
                    )
                prepared_tasks.append(prepared_task)
                source_base_model_task.to("cpu")
            else:
                aligned = load_aligned_tuned_from_ref(
                    ckpt_ref=ckpt_ref,
                    base_sd=source_base_sd,
                    build_cfg=source_build_cfg,
                    model=source_llm.model,
                    prefer_lora_view=False,
                )
                tuned_cpu = to_cpu_fp32(aligned) if isinstance(aligned, dict) else {k: v.cpu() for k, v in aligned.items()}

                # Full-model same-size delta
                if full_fp_keys is not None:
                    tuned_cpu = {k: v for k, v in tuned_cpu.items() if k in full_fp_keys and k in source_base_sd}
                    # Validate shapes
                    for k in tuned_cpu:
                        if tuple(tuned_cpu[k].shape) != tuple(source_base_sd[k].shape):
                            raise ValueError(
                                f"Source base vs tuned shape mismatch for '{k}': "
                                f"base {tuple(source_base_sd[k].shape)} vs "
                                f"tuned {tuple(tuned_cpu[k].shape)}"
                            )

                tv = TaskVector.from_checkpoints(
                    source_base_sd, tuned_cpu, strict=False
                )
                prepared_tasks.append(
                    _PreparedTaskDelta(
                        delta=tv.delta,
                        source_base=source_base_sd,
                        transport_keys=set(tp_keys or ()),
                        source_model=source_llm.model,
                    )
                )

        if eval_before_rebase_only:
            # Everything the before-rebase reference needs is done: block
            # extension has run and the extended source base has been scored.
            # Transport is the expensive half and contributes nothing here.
            print("\nStopping after the before-rebase eval (eval_before_rebase_only).")
            if run_logger is not None:
                run_logger.log_summary({
                    "ignored_block_extension_fields": ignored_block_extension_fields,
                    "calibration_provenance": _calibration_provenance(),
                    "method": method_name,
                    "backend": "lm_harness",
                    "stopped_after": "before_rebase_eval",
                    "harness_results_before_rebase": _baseline_summary(),
                    "before_rebase_model": (
                        "extended_source_base" if run_block_extension_prestep else "source_base"
                    ),
                    "source_depth": source_depth,
                    "target_depth": target_depth,
                })
                run_logger.finish("success")
            return

        weights_raw = cfg.get("weights", None)
        if weights_raw is None:
            weights = [1.0] * len(tuned_ref_list)
        else:
            w = weights_raw if isinstance(weights_raw, (list, tuple)) else [float(weights_raw)]
            if len(w) < len(tuned_ref_list):
                w = w * len(tuned_ref_list)
            weights = [float(x) for x in w[: len(tuned_ref_list)]]

        # Transport each task delta
        print(f"\n=== Transporting {len(tasks) if tasks else 1} task vectors with {method_name} ===")
        transported_deltas: list[dict[str, torch.Tensor]] = []

        family_adapter = target_family or source_family
        delta_source = resolve_delta_source(cfg)
        norm_match = resolve_norm_match(cfg)
        task_vector_norms: list[dict[str, float]] = []
        for idx, prepared_task in enumerate(prepared_tasks):
            corrected_delta = prepared_task.delta
            reference_delta = prepared_task.uncorrected_delta or corrected_delta
            delta = reference_delta if delta_source == "uncorrected" else corrected_delta
            transport_keys = prepared_task.transport_keys
            if tasks:
                label = tasks[idx]
            else:
                label = f"task_{idx}"
            print(f"\n--- '{label}' ({idx + 1}/{len(prepared_tasks)}) ---")
            t0 = time.time()

            if run_block_extension_prestep:
                prepared_task.source_model.to(device)

            if method_name in ("theseus", "theseus_gqa", "bico") and transport_keys:
                # Hybrid: transport body keys, identity-pass the rest
                body_delta = {k: v for k, v in delta.items() if k in transport_keys}
                passthrough_delta = {k: v for k, v in delta.items() if k not in transport_keys}

                transport_kwargs = dict(method_params)
                source_calib = _build_text_calibration_loader(
                    tokenizer=source_llm.tokenizer,
                    texts=_calibration().texts,
                    batch_size=calib_batch_size,
                    max_length=calib_max_length,
                )
                target_calib = _build_text_calibration_loader(
                    tokenizer=target_llm.tokenizer,
                    texts=_calibration().texts,
                    batch_size=calib_batch_size,
                    max_length=calib_max_length,
                )

                if method_name in ("theseus", "theseus_gqa"):
                    transport_kwargs.setdefault("seq_align", "interpolate")
                    if calib_n_batches is None:
                        transport_kwargs.setdefault("n_batches", _DEFAULT_CALIB_BATCHES)
                    shared_kwargs = dict(
                        source_model=prepared_task.source_model,
                        target_model=target_llm.model,
                        source_dataloader=source_calib,
                        target_dataloader=target_calib,
                        family_adapter=family_adapter,
                        device=device,
                    )
                    transported_body = method.transport(
                        source_base=prepared_task.source_base,
                        target_base=target_base_sd,
                        delta=body_delta,
                        strict=False,
                        **shared_kwargs,
                        **transport_kwargs,
                    )
                else:
                    from ...models.grad_recipes import causal_lm_recipe

                    transport_kwargs.setdefault("seq_align", "interpolate")
                    if calib_n_batches is None:
                        transport_kwargs.setdefault("n_batches", _DEFAULT_CALIB_BATCHES)
                    shared_kwargs = dict(
                        source_model=prepared_task.source_model,
                        target_model=target_llm.model,
                        source_dataloader=source_calib,
                        target_dataloader=target_calib,
                        source_recipe=causal_lm_recipe(device=device),
                        target_recipe=causal_lm_recipe(device=device),
                        family_adapter=family_adapter,
                        device=device,
                    )
                    transported_body = method.transport(
                        source_base=prepared_task.source_base,
                        target_base=target_base_sd,
                        delta=body_delta,
                        strict=False,
                        curvature_dataloader=None,
                        **shared_kwargs,
                        **transport_kwargs,
                    )
                out = dict(transported_body)
                skipped_passthrough: list[str] = []
                for k, v in passthrough_delta.items():
                    if k in target_base_sd and tuple(v.shape) == tuple(target_base_sd[k].shape):
                        out[k] = v.to(dtype=target_base_sd[k].dtype, device="cpu")
                    else:
                        skipped_passthrough.append(k)
                transported = out
                if skipped_passthrough:
                    print(
                        f"  skipped {len(skipped_passthrough)} passthrough keys with incompatible target shape "
                        f"(sample={skipped_passthrough[:5]})"
                    )
            else:
                transported = method.transport(
                    source_base=prepared_task.source_base,
                    target_base=target_base_sd,
                    delta=delta,
                    strict=False,
                    **method_params,
                )
            elapsed = time.time() - t0
            print(f"  transported {len(transported)} keys in {elapsed:.1f}s")

            transported, norms = norm_match_transported(
                transported,
                corrected_delta=corrected_delta,
                reference_delta=reference_delta,
                transport_keys=transport_keys,
                norm_match=norm_match,
            )
            task_vector_norms.append(norms)
            transported_deltas.append(transported)
            if run_block_extension_prestep:
                # Release each task-local resized model immediately after its
                # matching transport completes.
                prepared_task.source_model.to("cpu")
                del prepared_task.source_model

        # Merge transported deltas
        merged_delta = compose_weighted_deltas(transported_deltas, weights)
        delta_stats = _summarize_merged_delta(merged_delta, target_base_sd)
        task_vector_report = {
            "transport_delta_source": delta_source,
            "delta_norm_match": norm_match or "none",
            "per_task": task_vector_norms,
        }
        report_merged_delta(delta_stats)

        search_planner = build_search_planner(
            cfg=cfg, base_method_params=method_params
        )

        # ---- Dispatch evaluation backend ----
        if is_harness_only or harness_tasks_resolved:

            baseline_harness_results = _baseline_summary()

            best_harness_eval, harness_results_by_alpha, harness_search_results = harness_alpha_search(
                search_planner=search_planner,
                merged_delta=merged_delta,
                target_base_sd=target_base_sd,
                target_llm=target_llm,
                harness_tasks_resolved=harness_tasks_resolved,
                device=device,
                harness_num_fewshot=harness_num_fewshot,
                harness_batch_size=harness_batch_size,
                harness_limit=harness_limit,
                harness_samples=harness_samples,
            )

            best_alpha = float(best_harness_eval.candidate.alpha)
            best_harness_results = harness_results_by_alpha[best_alpha]
            print(f"\nBest alpha={best_alpha:.3f} -> avg_score={best_harness_eval.avg_acc:.6f}")
            print("\n=== Harness results (best alpha) ===")
            for task_name, acc in best_harness_results.items():
                print(f"  {task_name}: {acc:.4f}")

            harness_test_results, test_samples = score_harness_test_slice(
                cfg=cfg,
                merged_delta=merged_delta,
                target_base_sd=target_base_sd,
                target_llm=target_llm,
                harness_tasks_resolved=harness_tasks_resolved,
                device=device,
                harness_num_fewshot=harness_num_fewshot,
                harness_batch_size=harness_batch_size,
                harness_samples=harness_samples,
                best_alpha=best_alpha,
            )

            if cfg.get("save_merged", None) is not None:
                save_merged_state(
                    cfg["save_merged"], merged_delta, best_alpha, target_base_sd, message="Saved rebased state to"
                )

            if run_logger is not None:
                run_logger.log_summary(
                    assemble_harness_summary(
                        ignored_block_extension_fields=ignored_block_extension_fields,
                        calibration_provenance=_calibration_provenance(),
                        method_name=method_name,
                        best_alpha=best_alpha,
                        best_harness_results=best_harness_results,
                        harness_test_results=harness_test_results,
                        test_samples=test_samples,
                        baseline_harness_results=baseline_harness_results,
                        run_block_extension_prestep=run_block_extension_prestep,
                        harness_results_by_alpha=harness_results_by_alpha,
                        delta_stats=delta_stats,
                        task_vector_report=task_vector_report,
                        search_planner=search_planner,
                        harness_search_results=harness_search_results,
                        saved_merged_path=cfg.get("save_merged"),
                    )
                )
                run_logger.finish("success")
            return

        # ---- NLI eval path (existing) ----
        task_heads: dict[str, Any] | None = None
        if task_heads_path is not None:
            task_heads = load_task_heads(str(task_heads_path))

        user_prompt_template = cfg.get("prompt_template", None)
        split = str(cfg.get("split", "validation"))
        max_samples_per_task = cfg.get("max_samples_per_task", None)
        if max_samples_per_task is not None:
            max_samples_per_task = int(max_samples_per_task)
        max_prompt_tokens = cfg.get("max_prompt_tokens", None)
        if max_prompt_tokens is not None:
            max_prompt_tokens = int(max_prompt_tokens)
        print_every = cfg.get("print_every", None)
        if print_every is not None:
            print_every = int(print_every)

        task_data: list[NLITaskData] = []
        for t in tasks:
            td = build_nli_task_data(task=t, split=split, max_samples=max_samples_per_task)
            task_data.append(td)
            print(f"Loaded task {t}: {td.meta}")

        external_ref_acc = resolve_fine_tuned_acc(cfg=cfg, tasks=tasks)
        if external_ref_acc is not None:
            print(f"External ref accs: {external_ref_acc}")

        tokenized_task_data: list[NLITokenizedData] = []
        if eval_mode == "head_logits" and task_heads is not None:
            batch_size = int(cfg.get("batch_size", 8))
            num_workers = int(cfg.get("num_workers", 0))
            max_length = int(cfg.get("max_length", 512))
            task_mask_class = resolve_task_mask_class(cfg.get("task_mask_class", {}))
            head_num_labels = int(getattr(target_llm.model.config, "num_labels", num_labels))

            for td in task_data:
                masked_class = task_mask_class.get(td.task, None)
                class_ids = head_class_ids_for_task(
                    task=td.task,
                    task_num_labels=len(td.labels),
                    head_num_labels=head_num_labels,
                    masked_class=masked_class,
                )
                tk = build_nli_tokenized_loader(
                    task_data=td,
                    tokenizer=target_llm.tokenizer,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_length=max_length,
                    head_class_ids=class_ids,
                )
                tokenized_task_data.append(tk)
                print(f"Tokenized {td.task}: {tk.meta}")

        best_result, search_results = nli_alpha_search(
            search_planner=search_planner,
            merged_delta=merged_delta,
            target_base_sd=target_base_sd,
            target_llm=target_llm,
            task_data=task_data,
            tokenized_task_data=tokenized_task_data,
            task_heads=task_heads,
            eval_mode=eval_mode,
            head_key_pattern=head_key_pattern,
            user_prompt_template=user_prompt_template,
            max_prompt_tokens=max_prompt_tokens,
            print_every=print_every,
            external_ref_acc=external_ref_acc,
            device=device,
        )

        best_alpha = float(best_result.candidate.alpha)
        best_vals = list(best_result.per_task_acc)
        print(f"\nBest alpha={best_alpha:.2f} -> avg_acc={best_result.avg_acc:.6f}")

        if external_ref_acc is not None:
            per_task_rows = [{"task": td.task} for td in task_data]
            single_accs = [
                to_unit_acc(external_ref_acc[td.task]) if td.task in external_ref_acc else 0.0
                for td in task_data
            ]
            norm_ratio = [
                (best_vals[i] / single_accs[i]) if single_accs[i] > 0 else 0.0
                for i in range(len(best_vals))
            ]
            pretty_print_task_accuracies(
                suite_name or "nli6",
                method_name,
                "full",
                per_task_rows,
                best_vals,
                norm_ratio,
                single_accs=single_accs,
            )

        if cfg.get("save_merged", None) is not None:
            save_merged_state(
                cfg["save_merged"], merged_delta, best_alpha, target_base_sd, message="Saved best-alpha rebased state to"
            )

        if run_logger is not None:
            run_logger.log_summary(
                assemble_nli_summary(
                    ignored_block_extension_fields=ignored_block_extension_fields,
                    calibration_provenance=_calibration_provenance(),
                    method_name=method_name,
                    best_alpha=best_alpha,
                    task_data=task_data,
                    delta_stats=delta_stats,
                    task_vector_report=task_vector_report,
                    search_planner=search_planner,
                    search_results=search_results,
                    best_vals=best_vals,
                    saved_merged_path=cfg.get("save_merged"),
                )
            )
            run_logger.finish("success")

    except Exception as exc:
        if run_logger is not None:
            finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
