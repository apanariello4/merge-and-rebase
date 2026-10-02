from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from merge_and_rebase.utils.helpers import load_json

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
from ...data.llm_calibration import (  # noqa: F401  (old names stay importable from the package)
    TokenizedPromptDataset,
    build_text_calibration_loader,
)
from ...data.text_loaders import NLI_TASKS
from ...merge.methods._common import get_method_params
from ...models.text_lm import TextBuildConfig
from ...rebase import get_method
from ...rebase.registry import canonical_method_name
from ...run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from .alpha_search import run_alpha_search
from .context import build_runtime
from .pipeline import run_rebase
from .stages import _prepare_resized_task_delta  # noqa: F401
from .summary import assemble_summary

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

        rt = build_runtime(
            cfg,
            method_name=method_name,
            method=method,
            method_params=method_params,
            source_build_cfg=source_build_cfg,
            target_build_cfg=target_build_cfg,
            device=device,
            num_labels=num_labels,
        )
        outputs = run_rebase(rt, run_logger)
        if outputs is None:
            return
        record = run_alpha_search(rt, outputs)
        if run_logger is not None:
            run_logger.log_summary(assemble_summary(record))
            run_logger.finish("success")

    except Exception as exc:
        if run_logger is not None:
            finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
