from __future__ import annotations

import argparse
from typing import Any

from ..cli_args import (
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    build_logging_overrides,
    merge_non_none,
)
from ..models.text_lm import TextBuildConfig, TextLM
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ..utils.helpers import load_json, parse_csv
from .lm_harness_runner import run as run_harness


def main() -> None:
    """
    Load a single HF causal-LM and evaluate it on lm-eval harness tasks.

    Unlike llm_rebase.py's method="identity" trick, this does not build a
    source+target pair or run any transport/merge machinery -- it just loads
    one model and evaluates it.
    """
    run_logger = None
    try:
        p = argparse.ArgumentParser(
            "Evaluate a single HF causal-LM on lm-eval harness tasks (no transport/merge)."
        )
        add_config_arg(p)
        p.add_argument("--model-name-or-path", type=str, default=None)
        p.add_argument("--model-arch", type=str, default=None, choices=["llama", "qwen", "t5", "auto"])
        p.add_argument("--model-kind", type=str, default=None, choices=["causal_lm", "sequence_classification"])
        p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--use-fast-tokenizer", action=argparse.BooleanOptionalAction, default=None)
        add_device_dtype_args(p, device_default=None, dtype_default=None)
        p.add_argument("--harness-tasks", type=str, default=None)
        p.add_argument("--harness-num-fewshot", type=int, default=None)
        p.add_argument("--harness-batch-size", type=str, default=None)
        p.add_argument("--harness-limit", type=int, default=None)
        p.add_argument(
            "--exclude-calib-batches",
            type=int,
            default=None,
            help=(
                "Hold out the same doc slice a merge job's calibration would have "
                "consumed (exclude_calib_batches * calibration_batch_size docs, "
                "picked the same way as llm_rebase.py's calibration/eval split) "
                "so this baseline is scored on exactly the docs the merge run was."
            ),
        )
        p.add_argument("--calibration-batch-size", type=int, default=None)
        p.add_argument("--calibration-split", type=str, default=None)
        add_logging_args(p)
        args = p.parse_args()

        cfg: dict[str, Any] = load_json(args.config) if args.config is not None else {}
        cfg = merge_non_none(
            cfg,
            {
                "model_name_or_path": args.model_name_or_path,
                "model_arch": args.model_arch,
                "model_kind": args.model_kind,
                "trust_remote_code": args.trust_remote_code,
                "use_fast_tokenizer": args.use_fast_tokenizer,
                "device": args.device,
                "dtype": args.dtype,
                "harness_tasks": args.harness_tasks,
                "harness_num_fewshot": args.harness_num_fewshot,
                "harness_batch_size": args.harness_batch_size,
                "harness_limit": args.harness_limit,
                "exclude_calib_batches": args.exclude_calib_batches,
                "calibration_batch_size": args.calibration_batch_size,
                "calibration_split": args.calibration_split,
            },
        )
        cfg["logging"] = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))

        model_name = cfg.get("model_name_or_path")
        if not model_name:
            raise ValueError("model_name_or_path is required.")

        harness_tasks_raw = cfg.get("harness_tasks")
        if not harness_tasks_raw:
            raise ValueError("harness_tasks is required (this entrypoint only runs lm-eval harness tasks).")
        harness_tasks = (
            parse_csv(harness_tasks_raw) if isinstance(harness_tasks_raw, str) else list(harness_tasks_raw)
        )

        exclude_calib_batches = int(cfg.get("exclude_calib_batches", 0) or 0)
        harness_samples: dict[str, list[int]] | None = None
        if exclude_calib_batches > 0:
            # Mirror llm_rebase.py's calibration/eval doc split exactly (same
            # harness_tasks/calibration_split/seed) so this baseline is scored
            # on precisely the docs a merge job with the same
            # n_batches_act/calibration_batch_size never used for calibration.
            from ..data.llm_calibration import resolve_calibration_texts

            calib_batch_size = int(cfg.get("calibration_batch_size", cfg.get("batch_size", 4) or 4))
            calibration = resolve_calibration_texts(
                calibration_split=str(cfg.get("calibration_split", "val")),
                harness_tasks=harness_tasks,
                n_sequences=exclude_calib_batches * calib_batch_size,
                seed=int(cfg.get("seed", 0)),
            )
            print(f"Calibration hold-out: {calibration.describe()}")
            harness_samples = calibration.eval_samples or None

        run_logger = start_run(
            entrypoint="eval.llm_eval_only",
            logging_cfg=cfg["logging"],
            summary_path=default_summary_path(entrypoint="eval.llm_eval_only", logging_cfg=cfg["logging"]),
            metadata={"config_path": args.config, "resolved_config": cfg},
        )

        device = str(cfg.get("device", "cuda"))
        build_cfg = TextBuildConfig(
            model_name_or_path=str(model_name),
            model_arch=str(cfg.get("model_arch", "auto")),
            device=device,
            dtype=cfg.get("dtype", None),
            model_kind=str(cfg.get("model_kind", "causal_lm")),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            use_fast_tokenizer=bool(cfg.get("use_fast_tokenizer", True)),
        )
        print(f"Model: {model_name}")
        llm = TextLM.build(build_cfg)

        print(f"\nEvaluating with lm-harness: {harness_tasks}")
        harness_results = run_harness(
            tasks=harness_tasks,
            model=llm.model,
            tokenizer=llm.tokenizer,
            device=device,
            num_fewshot=cfg.get("harness_num_fewshot", 0),
            batch_size=str(cfg.get("harness_batch_size", "auto")),
            limit=cfg.get("harness_limit", None),
            samples=harness_samples,
        )
        print("\n=== Harness results ===")
        for name, value in harness_results.items():
            print(f"  {name}: {value:.4f}")

        run_logger.log_summary(
            {
                "model_name_or_path": model_name,
                "backend": "lm_harness",
                "harness_results": harness_results,
            }
        )
        run_logger.finish("success")

    except Exception as exc:
        if run_logger is not None:
            finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
