from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.hyperparam_search import (
    SearchEvaluation,
    build_search_planner,
    describe_candidate,
    summarize_search_results,
)
from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..cli_args import (
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
from ..data.text_loaders import (
    NLI_TASKS,
    NLITaskData,
    NLITokenizedData,
    build_nli_task_data,
    build_nli_tokenized_loader,
)
from ..io.ckpt import load_ckpt, load_into_model
from ..io.text_checkpoints import load_aligned_tuned_from_ref
from ..merge.methods._common import get_method_params
from ..merge.runtime import (
    apply_delta,
    compose_weighted_deltas,
    to_cpu_fp32,
)
from ..merge.task_vectors import TaskVector, default_key_filter
from ..models.text_lm import TextBuildConfig, TextLM
from ..rebase import get_method
from ..rebase.capabilities import check_pair
from ..rebase.model_families import infer_family
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from .block_extension import resolve_block_extension_config
from .block_extension_llm import run_block_extension_llm
from .llm_common import (
    default_prompt_for_task,
    head_class_ids_for_task,
    inject_task_head,
    load_task_heads,
    normalized_acc,
    resolve_eval_mode,
    resolve_fine_tuned_acc,
    resolve_suite_name,
    resolve_task_mask_class,
    resolve_tasks,
    to_unit_acc,
)
from .print_utils import pretty_print_task_accuracies

_DEFAULT_CALIBRATION_PROMPTS = [
    "Write a short summary of the moon landing.",
    "Explain why the sky looks blue.",
    "Translate 'good morning' into French.",
    "Give three healthy breakfast ideas.",
    "What is the capital of Japan?",
    "Write a polite email asking for a meeting.",
    "List two differences between cats and dogs.",
    "Solve: 17 plus 26.",
    "What is the derivative of x^3 + 2x with respect to x?",
    "Simplify the fraction 24/36.",
    "Solve for x: 3x - 7 = 14.",
    "What is the area of a circle with radius 5?",
    "Factor the polynomial x^2 - 9.",
    "What is the least common multiple of 8 and 12?",
    "Convert 0.75 into a fraction in lowest terms.",
    "Explain the Pythagorean theorem in one sentence.",
    "What is the sum of the first 10 positive integers?",
    "Solve the inequality 2x + 3 > 11.",
    "What is 15% of 240?",
    "Find the slope of the line passing through (2, 3) and (4, 9).",
    "Explain what a prime number is.",
    "What is the value of 7! (7 factorial)?",
    "Solve the system: x + y = 10, x - y = 2.",
    "What is the perimeter of a rectangle with sides 4 and 9?",
    "Write a short poem about autumn.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What are the main causes of the French Revolution?",
    "Explain how photosynthesis works.",
    "Describe the water cycle briefly.",
    "What is the boiling point of water at sea level?",
    "Give a brief definition of inflation in economics.",
    "Name three renewable energy sources.",
]


class _TokenizedPromptDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        feat = dict(self.features[int(idx)])
        feat["labels"] = list(feat["input_ids"])
        return feat


@dataclass
class _PreparedTaskDelta:
    """Delta together with the exact source context it was prepared from."""

    delta: dict[str, torch.Tensor]
    source_base: dict[str, torch.Tensor]
    transport_keys: set[str]
    source_model: torch.nn.Module


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
    final_depth = run_block_extension_llm(
        source_base_model=source_base_model,
        source_ft_model=source_ft_model,
        calibration_loader=calibration_loader,
        target_layers_total=target_layers_total,
        config=config,
        family_adapter=family_adapter,
        device=device,
    )
    if final_depth != target_layers_total:
        raise RuntimeError(
            "Block extension preprocess failed: "
            f"final_depth={final_depth}, expected={target_layers_total}."
        )

    source_base = to_cpu_fp32(source_base_model.state_dict())
    source_ft = to_cpu_fp32(source_ft_model.state_dict())
    task_vector = TaskVector.from_checkpoints(source_base, source_ft, strict=False)
    return _PreparedTaskDelta(
        delta=task_vector.delta,
        source_base=source_base,
        transport_keys=set(family_adapter.transportable_keys(source_base)),
        source_model=source_base_model,
    )


def _build_text_calibration_loader(
    *,
    tokenizer: Any,
    prompts: list[str] | None = None,
    batch_size: int = 2,
    max_length: int = 128,
) -> DataLoader:
    prompt_list = list(prompts or _DEFAULT_CALIBRATION_PROMPTS)
    enc = tokenizer(
        prompt_list,
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
    )
    features: list[dict[str, Any]] = []
    for i in range(len(prompt_list)):
        features.append({k: v[i] for k, v in enc.items()})

    dataset = _TokenizedPromptDataset(features)

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        feats = [{k: v for k, v in row.items() if k != "labels"} for row in batch]
        padded = tokenizer.pad(feats, return_tensors="pt", padding="max_length", max_length=int(max_length))
        padded["labels"] = padded["input_ids"].clone()
        if "attention_mask" in padded:
            padded["labels"][padded["attention_mask"] == 0] = -100
        return padded

    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
    )


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
        p.add_argument("--harness-num-fewshot", type=int, default=0)
        p.add_argument("--harness-batch-size", type=str, default="auto")
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
            "block_extension_enabled": args.block_extension_enabled,
            "block_extension_params": block_extension_params_cli,
            "eval_before_rebase": args.eval_before_rebase,
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
        method_params = dict(get_method_params({"method_params": cfg.get("method_params", {})}))

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

        # Block extension: build calibration loader once if needed
        blockext_calib_loader = None
        if run_block_extension_prestep:
            calib_batch_size = int(cfg.get("calibration_batch_size", cfg.get("batch_size", 2) or 2))
            calib_max_length = int(cfg.get("calibration_max_length", 128))
            blockext_calib_loader = _build_text_calibration_loader(
                tokenizer=source_llm.tokenizer,
                prompts=calibration_prompts_cfg,
                batch_size=calib_batch_size,
                max_length=calib_max_length,
            )

        for ckpt_ref in tuned_ref_list:
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
                prepared_tasks.append(prepared_task)
                source_base_model_task.to("cpu")
                del source_ft_model_task
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
        for idx, prepared_task in enumerate(prepared_tasks):
            delta = prepared_task.delta
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
                calib_batch_size = int(cfg.get("calibration_batch_size", cfg.get("batch_size", 2) or 2))
                calib_max_length = int(cfg.get("calibration_max_length", 128))
                source_calib = _build_text_calibration_loader(
                    tokenizer=source_llm.tokenizer,
                    prompts=calibration_prompts_cfg,
                    batch_size=calib_batch_size,
                    max_length=calib_max_length,
                )
                target_calib = _build_text_calibration_loader(
                    tokenizer=target_llm.tokenizer,
                    prompts=calibration_prompts_cfg,
                    batch_size=calib_batch_size,
                    max_length=calib_max_length,
                )

                if method_name in ("theseus", "theseus_gqa"):
                    transport_kwargs.setdefault("seq_align", "interpolate")
                    transport_kwargs.setdefault("n_batches", 2)
                    transported_body = method.transport(
                        source_base=prepared_task.source_base,
                        target_base=target_base_sd,
                        delta=body_delta,
                        strict=False,
                        source_model=prepared_task.source_model,
                        target_model=target_llm.model,
                        source_dataloader=source_calib,
                        target_dataloader=target_calib,
                        family_adapter=family_adapter,
                        device=device,
                        **transport_kwargs,
                    )
                else:
                    from ..models.grad_recipes import causal_lm_recipe

                    transport_kwargs.setdefault("seq_align", "interpolate")
                    transport_kwargs.setdefault("n_batches", 2)
                    transported_body = method.transport(
                        source_base=prepared_task.source_base,
                        target_base=target_base_sd,
                        delta=body_delta,
                        strict=False,
                        source_model=prepared_task.source_model,
                        target_model=target_llm.model,
                        source_dataloader=source_calib,
                        target_dataloader=target_calib,
                        source_recipe=causal_lm_recipe(device=device),
                        target_recipe=causal_lm_recipe(device=device),
                        curvature_dataloader=None,
                        family_adapter=family_adapter,
                        device=device,
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
            transported_deltas.append(transported)
            if run_block_extension_prestep:
                # Release each task-local resized model immediately after its
                # matching transport completes.
                prepared_task.source_model.to("cpu")
                del prepared_task.source_model

        # Merge transported deltas
        merged_delta = compose_weighted_deltas(transported_deltas, weights)

        search_planner = build_search_planner(
            cfg=cfg, base_method_params=method_params
        )

        # ---- Dispatch evaluation backend ----
        harness_tasks_resolved = parse_csv(harness_tasks_raw) if isinstance(harness_tasks_raw, str) else (harness_tasks_raw or [])

        if is_harness_only or harness_tasks_resolved:
            from .lm_harness_runner import run as run_harness

            harness_num_fewshot = int(cfg.get("harness_num_fewshot", 0))
            harness_batch_size = str(cfg.get("harness_batch_size", "auto"))
            harness_limit = cfg.get("harness_limit", None)

            best_harness_eval: SearchEvaluation | None = None
            harness_results_by_alpha: dict[float, dict[str, float]] = {}
            harness_search_results: list[SearchEvaluation] = []

            baseline_harness_results: dict[str, float] | None = None
            if bool(cfg.get("eval_before_rebase", False)):
                load_into_model(target_llm.model, target_base_sd, strict=False)
                print("\nEvaluating untransported target with lm-harness (before rebase)...")
                baseline_harness_results = run_harness(
                    tasks=list(harness_tasks_resolved),
                    model=target_llm.model,
                    tokenizer=target_llm.tokenizer,
                    device=device,
                    num_fewshot=harness_num_fewshot,
                    batch_size=harness_batch_size,
                    limit=harness_limit,
                )
                for task_name, acc in baseline_harness_results.items():
                    print(f"  [before rebase] {task_name}: {acc:.4f}")

            while True:
                batch = search_planner.next_batch()
                if batch is None:
                    break
                batch_results: list[SearchEvaluation] = []

                for candidate in batch:
                    alpha = float(candidate.alpha)
                    scaled = {k: v * alpha for k, v in merged_delta.items()}
                    merged_sd = apply_delta(target_base_sd, scaled)
                    load_into_model(target_llm.model, merged_sd, strict=False)

                    print(f"\nEvaluating with lm-harness (alpha={alpha:.3f})...")
                    harness_results = run_harness(
                        tasks=list(harness_tasks_resolved),
                        model=target_llm.model,
                        tokenizer=target_llm.tokenizer,
                        device=device,
                        num_fewshot=harness_num_fewshot,
                        batch_size=harness_batch_size,
                        limit=harness_limit,
                    )
                    for task_name, acc in harness_results.items():
                        print(f"  {task_name}: {acc:.4f}")

                    score = sum(harness_results.values()) / max(1, len(harness_results))
                    result = SearchEvaluation(
                        candidate=candidate,
                        score=float(score),
                        avg_acc=float(score),
                        avg_norm_acc=0.0,
                        per_task_acc=[float(v) for v in harness_results.values()],
                        per_task_norm_acc=[],
                    )
                    batch_results.append(result)
                    harness_search_results.append(result)
                    harness_results_by_alpha[alpha] = harness_results

                    if best_harness_eval is None or result.score > best_harness_eval.score:
                        best_harness_eval = result

                    print(f"  alpha={alpha:.3f}  avg_score={score:.6f}")
                    del merged_sd

                search_planner.observe(batch_results)

            if best_harness_eval is None:
                raise RuntimeError("Harness alpha search produced no results.")

            if len(harness_search_results) > 1:
                print("\n=== Harness alpha search summary ===")
                for r in harness_search_results:
                    print(f"{describe_candidate(r.candidate)}  avg_score={r.avg_acc:.6f}")

            best_alpha = float(best_harness_eval.candidate.alpha)
            best_harness_results = harness_results_by_alpha[best_alpha]
            print(f"\nBest alpha={best_alpha:.3f} -> avg_score={best_harness_eval.avg_acc:.6f}")
            print("\n=== Harness results (best alpha) ===")
            for task_name, acc in best_harness_results.items():
                print(f"  {task_name}: {acc:.4f}")

            if cfg.get("save_merged", None) is not None:
                scaled = {k: v * best_alpha for k, v in merged_delta.items()}
                best_sd = apply_delta(target_base_sd, scaled)
                outp = Path(str(cfg["save_merged"]))
                outp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(to_cpu_fp32(best_sd), str(outp))
                print(f"Saved rebased state to {outp}")

            if run_logger is not None:
                run_logger.log_summary({
                    "method": method_name,
                    "best_alpha": best_alpha,
                    "backend": "lm_harness",
                    "harness_results": best_harness_results,
                    "harness_results_before_rebase": baseline_harness_results,
                    "search_strategy": search_planner.search_summary(),
                    "search_results": summarize_search_results(harness_search_results),
                    "saved_merged_path": cfg.get("save_merged"),
                })
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

        best_result: SearchEvaluation | None = None
        search_results: list[SearchEvaluation] = []
        alpha_to_task_accs: dict[float, list[float]] = {}
        alpha_to_task_norm_accs: dict[float, list[float]] = {}

        while True:
            batch = search_planner.next_batch()
            if batch is None:
                break
            batch_results: list[SearchEvaluation] = []

            for candidate in batch:
                alpha = float(candidate.alpha)

                scaled = {k: v * alpha for k, v in merged_delta.items()}
                merged_sd = apply_delta(target_base_sd, scaled)
                load_into_model(target_llm.model, merged_sd, strict=False)

                accs: list[float] = []
                norm_accs: list[float] = []
                for i, td in enumerate(task_data):
                    if eval_mode == "head_logits" and task_heads is not None:
                        tk = tokenized_task_data[i]
                        inject_task_head(
                            model=target_llm.model,
                            task=td.task,
                            task_heads=task_heads,
                            head_key_pattern=head_key_pattern,
                            head_class_ids=list(tk.meta.get("head_class_ids", [])),
                        )
                        acc = target_llm.sequence_classification_accuracy(
                            tk.loader,
                            device=device,
                            mask_class=tk.mask_class,
                            print_every=print_every,
                        )
                    else:
                        tpl = user_prompt_template if user_prompt_template else default_prompt_for_task(td)
                        acc = target_llm.nli_accuracy(
                            examples=td.examples,
                            label_texts=td.label_texts,
                            prompt_template=tpl,
                            device=device,
                            max_prompt_tokens=max_prompt_tokens,
                            print_every=print_every,
                        )
                    accs.append(acc)
                    if external_ref_acc is not None and td.task in external_ref_acc:
                        n = normalized_acc(acc, external_ref_acc[td.task])
                        norm_accs.append(n)
                        print(f"  {td.task}: acc={acc:.6f}  norm_acc={n:.3f}")
                    else:
                        print(f"  {td.task}: acc={acc:.6f}")

                avg_acc = sum(accs) / max(1, len(accs))
                avg_norm_acc = sum(norm_accs) / max(1, len(norm_accs)) if norm_accs else 0.0
                score = avg_norm_acc if norm_accs else avg_acc
                result = SearchEvaluation(
                    candidate=candidate,
                    score=float(score),
                    avg_acc=float(avg_acc),
                    avg_norm_acc=float(avg_norm_acc),
                    per_task_acc=[float(v) for v in accs],
                    per_task_norm_acc=[float(v) for v in norm_accs],
                )
                batch_results.append(result)
                search_results.append(result)
                alpha_to_task_accs[alpha] = [float(v) for v in accs]
                alpha_to_task_norm_accs[alpha] = [float(v) for v in norm_accs]

                if best_result is None or result.score > best_result.score:
                    best_result = result

                print(f"  alpha={alpha:.2f}  avg_acc={avg_acc:.6f}  avg_norm_acc={avg_norm_acc:.3f}" if norm_accs else f"  alpha={alpha:.2f}  avg_acc={avg_acc:.6f}")

                del merged_sd

            search_planner.observe(batch_results)

        if best_result is None:
            raise RuntimeError("Alpha search produced no results.")

        print("\n=== Alpha search summary ===")
        for r in search_results:
            if r.per_task_norm_acc:
                print(
                    f"{describe_candidate(r.candidate)}  "
                    f"avg_acc={r.avg_acc:.6f}  avg_norm_acc={r.avg_norm_acc:.3f}"
                )
            else:
                print(f"{describe_candidate(r.candidate)}  avg_acc={r.avg_acc:.6f}")

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
            scaled = {k: v * best_alpha for k, v in merged_delta.items()}
            best_sd = apply_delta(target_base_sd, scaled)
            outp = Path(str(cfg["save_merged"]))
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(to_cpu_fp32(best_sd), str(outp))
            print(f"Saved best-alpha rebased state to {outp}")

        if run_logger is not None:
            run_logger.log_summary({
                "method": method_name,
                "best_alpha": best_alpha,
                "tasks": [td.task for td in task_data],
                "search_strategy": search_planner.search_summary(),
                "search_results": summarize_search_results(search_results),
                "best_per_task_acc": {td.task: float(best_vals[i]) for i, td in enumerate(task_data)},
                "saved_merged_path": cfg.get("save_merged"),
            })
            run_logger.finish("success")

    except Exception as exc:
        if run_logger is not None:
            finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
