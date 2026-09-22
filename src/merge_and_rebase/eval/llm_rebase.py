from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, is_dataclass
from dataclasses import replace as dataclass_replace
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
from ..data.llm_calibration import resolve_calibration_texts
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
from .target_informed_runtime import (
    capture_residual_references,
    complete_residuals,
    complete_residuals_direct,
    materialize_missing_projection_biases,
    projection_transforms,
    scale_completion,
)
from .target_residual_completion import ResidualCompletionConfig


class _TokenizedPromptDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]], sample_ids: list[str] | None = None) -> None:
        self.features = features
        # Identity of the underlying examples, not of this tokenization. The
        # source and target calibration loaders tokenize the SAME texts with
        # different tokenizers, so they are different objects holding different
        # token ids; paired calibration has to recognise them as the same
        # examples replayed under two preprocessors, which is exactly what it
        # falls back to sample_ids for.
        self.sample_ids = list(sample_ids) if sample_ids is not None else None

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
    residual_references: dict[str, Any] | None = None


def _delta_norm(delta: Mapping[str, torch.Tensor], keys: Iterable[str] | None = None) -> float:
    """Frobenius norm of a task vector, optionally restricted to `keys`."""
    total = 0.0
    for key, value in delta.items():
        if keys is not None and key not in keys:
            continue
        total += float(value.float().pow(2).sum())
    return total ** 0.5


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



def _maybe_capture_target_residual_references(
    *,
    config: ResidualCompletionConfig,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_loader: Any,
    target_loader: Any,
    family_adapter: Any,
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    """Capture proposal-1 native reference banks, or no-op when disabled.

    Must run before the resize: these are the un-resized source model's own
    boundary activations, paired against the pretrained target. Returns None
    when disabled so callers can thread the result through unconditionally and
    still get a byte-identical no-op.
    """
    if not config.enabled:
        return None
    return capture_residual_references(
        source_base_model,
        source_ft_model,
        target_model,
        source_loader,
        target_loader,
        num_batches=config.num_batches,
        seed=seed,
        device=device,
        target_scope=config.target_scope,
        family_adapter=family_adapter,
    )


def _maybe_complete_target_residual_task_vector(
    *,
    config: ResidualCompletionConfig,
    references: dict[str, Any] | None,
    prepared: Any,
    layout: Mapping[str, Any] | None,
    target_model: torch.nn.Module,
    target_base_sd: Mapping[str, torch.Tensor],
    transported_delta: dict[str, torch.Tensor],
    target_loader: Any,
    family_adapter: Any,
    device: str,
    materialized_bias_keys: set[str] | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
    """Complete the transported task vector, or return it untouched.

    Runs after transport is fitted and only ever adds to the task vector, never
    to the target base weights. Disabled, or missing references/layout, returns
    the same dict object so a caller hashing the delta sees no change.

    ``config.mode`` selects the arm, exactly as on the vision path:

    - ``transport_residual`` completes the residual an already-transported task
      vector left behind, solving through the fitted ``(t_in, t_out)`` maps;
    - ``direct_target`` is transport-free. The caller must already have skipped
      the transport fit and apply, so ``transported_delta`` is empty and the
      fitted correction is the whole task vector. It is scaled against an
      explicit zero baseline, which keeps ``strength=0`` an exact
      native-target-base control.
    """
    if not config.enabled or references is None or not layout:
        return transported_delta, None
    if config.mode == "direct_target":
        # Transport-free arm. Checked before the materialized-bias seeding
        # below, because that seeding would put keys into transported_delta and
        # make an empty, genuinely transport-free vector look populated.
        if transported_delta:
            raise ValueError(
                "mode='direct_target' requires an empty transported task vector: the "
                f"caller passed {len(transported_delta)} transported keys, so the arm would "
                "not be transport-free"
            )
        target_corrections, diagnostics = complete_residuals_direct(
            target_model,
            target_base_sd,
            references,
            layout,
            target_loader,
            config=config,
            device=device,
            family_adapter=family_adapter,
        )
        # projection_transforms() is deliberately not called: it demands fitted
        # t_in/t_out, and this arm has neither.
        zero_baseline = {key: torch.zeros_like(value) for key, value in target_corrections.items()}
        return scale_completion(zero_baseline, target_corrections, config.strength), diagnostics
    # A materialized bias is a target parameter that did not exist before this
    # run and is zero in the base, so the task vector has no entry for it --
    # transport only produced the body weights. scale_completion requires every
    # completion key to have a baseline to add to, so seed those zeros here.
    # Without this the intercept has nowhere to land and the run dies at
    # strength>0 (strength=0 returns early and never notices).
    transported_delta = dict(transported_delta)
    for bias_key in materialized_bias_keys or ():
        if bias_key not in transported_delta and bias_key in target_base_sd:
            transported_delta[bias_key] = torch.zeros_like(target_base_sd[bias_key])

    transforms = projection_transforms(
        prepared, layout, target_scope=config.target_scope, family_adapter=family_adapter
    )
    _source_corrections, target_corrections, diagnostics = complete_residuals(
        target_model,
        target_base_sd,
        transported_delta,
        references,
        transforms,
        layout,
        target_loader,
        config=config,
        device=device,
        family_adapter=family_adapter,
    )
    completed = scale_completion(transported_delta, target_corrections, config.strength)
    return completed, diagnostics


def _apply_passthrough_delta(
    out: dict[str, torch.Tensor],
    passthrough_delta: Mapping[str, torch.Tensor],
    target_base_sd: Mapping[str, torch.Tensor],
    *,
    carry: bool,
) -> tuple[dict[str, torch.Tensor], list[str], set[str]]:
    """Fold the non-transportable remainder into the task vector, or drop it.

    The decoder path splits a task vector into a transportable body and a
    remainder -- embeddings, per-layer norms, lm_head -- that no transport
    touches. Vision has no such split, so this policy is decoder-only.

    ``carry=True`` folds in every shape-compatible key verbatim; that is what
    the transport arm has always done, and it is what ``direct_passthrough``
    opts back into. ``carry=False`` drops the remainder entirely, which is the
    transport-free arm's default: those keys are raw *source* parameters, so
    folding them in would put source weights into the target, the fitted
    correction would stop being the whole task vector, and ``strength=0`` would
    stop being an exact native-target-base control.

    Returns ``(out, skipped, dropped)`` -- ``skipped`` naming the keys that were
    carried but whose target shape did not match, and ``dropped`` the keys the
    drop policy discarded. Exactly one of the two is ever non-empty.
    """
    if not carry:
        return out, [], set(passthrough_delta)
    skipped: list[str] = []
    for key, value in passthrough_delta.items():
        if key in target_base_sd and tuple(value.shape) == tuple(target_base_sd[key].shape):
            out[key] = value.to(dtype=target_base_sd[key].dtype, device="cpu")
        else:
            skipped.append(key)
    return out, skipped, set()


def _summarize_merged_delta(
    merged_delta: dict[str, torch.Tensor],
    target_base: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Measure how much of the target model the transported delta actually moves.

    A transport that silently zeroes every key still returns a full set of
    correctly shaped tensors, so the alpha sweep looks healthy while every
    candidate evaluates the same untouched base model. These numbers go into the
    run summary so that failure mode is visible in the JSON, not only in stdout.
    """
    sq_delta = 0.0
    sq_base = 0.0
    nonzero = 0
    for key, value in merged_delta.items():
        val = value.float()
        sq_delta += float(val.pow(2).sum())
        if float(val.abs().sum()) > 0.0:
            nonzero += 1
        base_ref = target_base.get(key)
        if base_ref is not None:
            sq_base += float(base_ref.float().pow(2).sum())

    delta_norm = sq_delta ** 0.5
    base_norm = sq_base ** 0.5
    return {
        "key_count": float(len(merged_delta)),
        "nonzero_key_count": float(nonzero),
        "merged_delta_norm": delta_norm,
        "merged_delta_rel_norm": (delta_norm / base_norm) if base_norm > 0.0 else 0.0,
    }


def _build_text_calibration_loader(
    *,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 2,
    max_length: int = 128,
) -> DataLoader:
    prompt_list = list(texts)
    if not prompt_list:
        raise ValueError("Calibration loader needs at least one text sequence.")
    enc = tokenizer(
        prompt_list,
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
    )
    features: list[dict[str, Any]] = []
    for i in range(len(prompt_list)):
        features.append({k: v[i] for k, v in enc.items()})

    # Stable across processes: str.__hash__ is salted per interpreter, which
    # would make these ids non-reproducible if they were ever persisted.
    sample_ids = [f"{i}:{hashlib.sha1(t.encode()).hexdigest()[:12]}" for i, t in enumerate(prompt_list)]
    dataset = _TokenizedPromptDataset(features, sample_ids=sample_ids)

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
        # ARIADNE proposal 1 (target residual completion). Disabled by default,
        # and resolve_block_extension_config already rejects it alongside
        # skip_correction=true, so an enabled run always has a correction to
        # complete.
        residual_completion_cfg = block_extension_cfg.target_residual_completion

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
        if residual_completion_cfg.enabled and not run_block_extension_prestep:
            # Proposal 1 is a silent no-op without the pre-step: the native
            # reference banks are only captured inside it, and completion with
            # references=None returns the delta untouched. The run would then
            # "succeed" having measured the plain baseline while its config and
            # its summary both claim a P1 arm. Fail here, in seconds, instead.
            reasons = []
            if not blockext_like_method:
                reasons.append(f"method={method_name!r} is not one of theseus/theseus_gqa/bico")
            if not block_extension_enabled:
                reasons.append("block_extension_enabled=false")
            if source_meta is None or target_meta is None:
                reasons.append("no family adapter metadata for the source/target pair")
            elif not depth_mismatch:
                reasons.append(f"source and target depths match ({source_depth})")
            raise ValueError(
                "target_residual_completion.enabled=true but the block-extension pre-step will "
                "not run, so proposal 1 would silently measure the uncorrected baseline: "
                + "; ".join(reasons)
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
            from .lm_harness_runner import run as run_harness

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
        calib_n_batches_cfg = method_params.get("num_batches")
        calib_n_batches = int(calib_n_batches_cfg) if calib_n_batches_cfg is not None else None
        # These are two separate budgets over one shared text pool, not one
        # knob: block extension consumes n_batches_act batches for its
        # activation capture, theseus/bico consume num_batches for theirs, and
        # neither is derived from the other. The max only sizes the pool, so
        # whichever consumer asks for more still finds enough text.
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
            int(calib_n_batches or 0),
        )
        # Resolved on first use: building it from an lm-harness task has to
        # index the task registry, which is far too expensive to pay for on a
        # run that never collects activations at all.
        _calibration_cache: list[Any] = []

        def _calibration() -> Any:
            if not _calibration_cache:
                resolved = resolve_calibration_texts(
                    prompts=calibration_prompts_cfg,
                    calibration_dataset=(
                        block_extension_cfg.calibration_dataset
                        or block_extension_cfg.calibration_task
                    ),
                    calibration_split=str(block_extension_cfg.calibration_split),
                    harness_tasks=list(harness_tasks_resolved),
                    n_sequences=max(1, n_calib_batches) * calib_batch_size,
                    seed=int(cfg.get("seed", 0)),
                )
                print(f"Calibration corpus: {resolved.describe()}")
                _calibration_cache.append(resolved)
            return _calibration_cache[0]

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

                # Proposal 1: capture the native reference banks BEFORE the
                # resize below mutates source_base_model_task/source_ft_model_task
                # in place. These are the un-resized source's own boundary
                # activations paired against the pretrained target, which is the
                # information the completion later regresses against.
                task_residual_references = _maybe_capture_target_residual_references(
                    config=residual_completion_cfg,
                    source_base_model=source_base_model_task,
                    source_ft_model=source_ft_model_task,
                    target_model=target_llm.model,
                    source_loader=blockext_calib_loader,
                    target_loader=_build_text_calibration_loader(
                        tokenizer=target_llm.tokenizer,
                        texts=_calibration().texts,
                        batch_size=calib_batch_size,
                        max_length=calib_max_length,
                    ) if residual_completion_cfg.enabled else None,
                    family_adapter=family_adapter_for_ext,
                    seed=int(cfg.get("seed", 0)),
                    device=device,
                )

                prepared_task = _prepare_resized_task_delta(
                    source_base_model=source_base_model_task,
                    source_ft_model=source_ft_model_task,
                    calibration_loader=blockext_calib_loader,
                    target_layers_total=int(target_depth),
                    config=block_extension_cfg,
                    family_adapter=family_adapter_for_ext,
                    device=device,
                )
                prepared_task.residual_references = task_residual_references
                if residual_completion_cfg.enabled:
                    layout_desc = prepared_task.extension_layout or {}
                    print(
                        f"  residual completion armed: scope={residual_completion_cfg.target_scope} "
                        f"strength={residual_completion_cfg.strength} "
                        f"inserted={len(layout_desc.get('inserted_blocks', ()))} "
                        f"final_blocks={len(layout_desc.get('final_blocks', ()))}"
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
        # Which task vector gets transported. "corrected" is the status quo:
        # activations and delta both come from the corrected resize.
        # "uncorrected" keeps the corrected model for activation capture -- so
        # the fitted alignment map is unchanged -- but transports the delta from
        # the uncorrected resize, isolating whether correction helps the map or
        # only distorts the vector.
        delta_source = str(cfg.get("transport_delta_source", "corrected")).strip().lower()
        if delta_source not in {"corrected", "uncorrected"}:
            raise ValueError(
                f"transport_delta_source must be 'corrected' or 'uncorrected'. Got: {delta_source!r}"
            )
        # Rescale the transported delta to the uncorrected task vector's norm.
        # Procrustes transport is orthogonal and norm-preserving, so without
        # this the correction's effect on scale reaches the target model in full
        # and a fixed alpha cannot distinguish scale from direction.
        norm_match = cfg.get("delta_norm_match", None)
        norm_match = str(norm_match).strip().lower() if norm_match is not None else None
        # Proposal-1 transport-free ablation. THESEUS/BiCo are neither fitted
        # nor applied: the question this arm asks is whether the desired local
        # functional effect can be written into the target with no parameter
        # transport at all, so fitting a transport and discarding it would both
        # burn the GPU hours the arm exists to save and blur the claim.
        direct_target_p1 = bool(
            residual_completion_cfg.enabled and residual_completion_cfg.mode == "direct_target"
        )
        if residual_completion_cfg.enabled and residual_completion_cfg.mode not in {
            "transport_residual", "direct_target"
        }:
            # complete_residuals() does not inspect config.mode; only the direct
            # solver does. An unhandled mode would silently run a different
            # method and report a plausible number for it.
            raise ValueError(
                f"target_residual_completion.mode={residual_completion_cfg.mode!r} is not "
                "implemented on the LLM path."
            )
        if norm_match not in {None, "none", "uncorrected"}:
            raise ValueError(
                f"delta_norm_match must be null or 'uncorrected'. Got: {norm_match!r}"
            )
        if direct_target_p1 and norm_match == "uncorrected":
            # The rescale multiplies the whole delta by
            # ||reference_delta restricted to transport_keys|| / ||transported||.
            # In direct mode the delta is the fitted correction, which is not a
            # transported image of the source task vector and is far smaller, so
            # the ratio is a large arbitrary number that would silently blow the
            # correction up. Refused rather than special-cased.
            raise ValueError(
                "delta_norm_match='uncorrected' is incompatible with "
                "target_residual_completion.mode='direct_target': the direct arm's task vector is "
                "the fitted correction, not a transported source vector, so matching its norm to "
                "the source delta has no meaning and would rescale it by an arbitrary factor"
            )
        task_vector_norms: list[dict[str, float]] = []
        residual_completion_diagnostics: dict[str, list[dict[str, Any]]] = {}
        materialized_bias_keys: set[str] = set()
        dropped_passthrough_keys: set[str] = set()
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
                # The direct arm never reads the source model's activations, so
                # its calibration loader is not built either.
                source_calib = None if direct_target_p1 else _build_text_calibration_loader(
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

                if direct_target_p1:
                    # Neither method.prepare nor method.transport runs: the
                    # transported task vector is empty by construction and the
                    # fitted correction below becomes the whole of it.
                    fitted_prepared = None
                    transported_body = {}
                    print(
                        "  transport skipped: target_residual_completion.mode='direct_target' "
                        "(transport-free arm)"
                    )
                elif method_name in ("theseus", "theseus_gqa"):
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
                    # Residual completion needs the fitted transforms, so the
                    # payload is built explicitly and handed to transport. When
                    # completion is off, prepared stays None and transport fits
                    # it internally exactly as before.
                    fitted_prepared = (
                        method.prepare(target_base=target_base_sd, delta=body_delta, **shared_kwargs, **transport_kwargs)
                        if residual_completion_cfg.enabled
                        else None
                    )
                    transported_body = method.transport(
                        source_base=prepared_task.source_base,
                        target_base=target_base_sd,
                        delta=body_delta,
                        strict=False,
                        prepared=fitted_prepared,
                        **shared_kwargs,
                        **transport_kwargs,
                    )
                else:
                    from ..models.grad_recipes import causal_lm_recipe

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
                    fitted_prepared = (
                        method.prepare(target_base=target_base_sd, delta=body_delta, **shared_kwargs, **transport_kwargs)
                        if residual_completion_cfg.enabled
                        else None
                    )
                    transported_body = method.transport(
                        source_base=prepared_task.source_base,
                        target_base=target_base_sd,
                        delta=body_delta,
                        strict=False,
                        prepared=fitted_prepared,
                        curvature_dataloader=None,
                        **shared_kwargs,
                        **transport_kwargs,
                    )
                # missing_bias="materialize": give the target's down_proj a zero
                # bias before completion runs, in the model and the base state
                # together. A decoder has none, and the exact form needs one for
                # its intercept; doing it here (not inside the solver) keeps the
                # merged state, completion's strict restore, and the eval load
                # all agreeing on the model's shape.
                if residual_completion_cfg.enabled and residual_completion_cfg.missing_bias == "materialize":
                    added_bias_keys = materialize_missing_projection_biases(
                        target_llm.model, target_base_sd, prepared_task.extension_layout or {},
                        family_adapter=family_adapter,
                        # Every projection the fit will write to, not just the
                        # MLP one: the two-component direct arm also fits
                        # self_attn.o_proj, which is bias-free on Qwen too.
                        components=residual_completion_cfg.components,
                    )
                    if added_bias_keys:
                        materialized_bias_keys.update(added_bias_keys)
                        print(f"  materialized {len(added_bias_keys)} zero projection bias(es) for the intercept")

                # Proposal 1: complete the transported task vector before the
                # passthrough keys are folded in. Only ever adds to the task
                # vector, never to the target base weights; a disabled run gets
                # the same object back.
                transported_body, completion_diagnostics = _maybe_complete_target_residual_task_vector(
                    config=residual_completion_cfg,
                    references=prepared_task.residual_references,
                    prepared=fitted_prepared,
                    layout=prepared_task.extension_layout,
                    target_model=target_llm.model,
                    target_base_sd=target_base_sd,
                    transported_delta=dict(transported_body),
                    target_loader=target_calib,
                    family_adapter=family_adapter,
                    device=device,
                    materialized_bias_keys=materialized_bias_keys,
                )
                if completion_diagnostics is not None:
                    residual_completion_diagnostics[str(label)] = completion_diagnostics
                    print(f"  residual completion applied to {len(completion_diagnostics)} block(s)")
                elif direct_target_p1:
                    # On the transport arm a skipped completion still leaves a
                    # transported vector behind, so the run remains meaningful.
                    # Here it leaves an empty delta: the pre-step ran but
                    # captured no reference banks or no realized layout, so the
                    # arm measured nothing. Say so, rather than failing later on
                    # the zero-delta gate with a message about transport.
                    raise RuntimeError(
                        "mode='direct_target' produced no completion: the block-extension "
                        "pre-step captured no reference banks or no realized layout. The run "
                        "would measure the untouched target base."
                    )

                # The transport-free arm drops the passthrough remainder by
                # default; direct_passthrough=true opts back into folding it in,
                # at the cost of the arm no longer being strictly transport-free
                # and gamma=0 no longer reproducing the native base. Both facts
                # reach the run summary rather than being left to be inferred.
                out, skipped_passthrough, dropped = _apply_passthrough_delta(
                    dict(transported_body),
                    passthrough_delta,
                    target_base_sd,
                    carry=not direct_target_p1 or residual_completion_cfg.direct_passthrough,
                )
                dropped_passthrough_keys.update(dropped)
                if dropped:
                    print(
                        f"  dropped {len(dropped)} passthrough keys (direct_target is "
                        "transport-free: nothing but the fitted correction reaches the target)"
                    )
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

            # Norms are reported for every run, not only when matching is on, so
            # the scale effect of correction is visible in the summary.
            n_corrected = _delta_norm(corrected_delta, transport_keys)
            n_uncorrected = _delta_norm(reference_delta, transport_keys)
            n_transported = _delta_norm(transported)
            scale = 1.0
            if norm_match == "uncorrected" and n_transported > 0.0:
                scale = n_uncorrected / n_transported
                transported = {k: v * scale for k, v in transported.items()}
            norms = {
                "source_corrected": n_corrected,
                "source_uncorrected": n_uncorrected,
                "transported_before_match": n_transported,
                "norm_match_scale": scale,
                "transported_after_match": n_transported * scale,
            }
            task_vector_norms.append(norms)
            print(
                f"  ||tv|| source corrected={n_corrected:.2f} uncorrected={n_uncorrected:.2f}"
                f" transported={n_transported:.2f}"
                + (f" -> rescaled x{scale:.3e}" if norm_match == "uncorrected" else "")
            )
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
            "residual_completion": {
                "enabled": bool(residual_completion_cfg.enabled),
                # Which arm actually ran. Recorded because the two answer
                # different questions, because a config alone no longer tells
                # them apart at read time, and because the summary otherwise
                # advertises `method` (theseus/bico) even when direct mode never
                # fitted or applied it, which would read as a transport run.
                "mode": residual_completion_cfg.mode,
                "parameter_transport": "none" if direct_target_p1 else method_name,
                "transport_fitted": not direct_target_p1,
                "target_scope": residual_completion_cfg.target_scope,
                "strength": float(residual_completion_cfg.strength),
                "exact_form": bool(residual_completion_cfg.exact_form),
                "missing_bias": residual_completion_cfg.missing_bias,
                "target_trajectory": residual_completion_cfg.target_trajectory,
                "components": list(residual_completion_cfg.components),
                "cascade_order": residual_completion_cfg.cascade_order,
                # Whether the direct arm carried the non-transportable source
                # keys or dropped them; see the comment at the drop site. None on
                # the transport arm, which always folds them in.
                "direct_passthrough": (
                    bool(residual_completion_cfg.direct_passthrough) if direct_target_p1 else None
                ),
                "passthrough_policy": (
                    ("folded" if residual_completion_cfg.direct_passthrough else "dropped")
                    if direct_target_p1
                    else "folded"
                ),
                "dropped_passthrough_key_count": len(dropped_passthrough_keys),
                "diagnostics": residual_completion_diagnostics,
                # Parameters this run added that stock Qwen does not have.
                "materialized_bias_keys": sorted(materialized_bias_keys),
            },
        }
        print(
            f"\nMerged delta: keys={delta_stats['key_count']} "
            f"nonzero_keys={delta_stats['nonzero_key_count']} "
            f"norm={delta_stats['merged_delta_norm']:.4f} "
            f"rel_norm={delta_stats['merged_delta_rel_norm']:.6f}"
        )
        if delta_stats["nonzero_key_count"] == 0:
            if direct_target_p1 and float(residual_completion_cfg.strength) == 0.0:
                # gamma=0 in direct mode is the exact native-target-base control:
                # the fit still runs and reports diagnostics, and an identically
                # zero delta is the intended result, not the silent-zero-transport
                # failure this gate exists to catch.
                print(
                    "  merged delta is identically zero, as expected for the direct-mode "
                    "gamma=0 native-target-base control"
                )
            else:
                raise RuntimeError(
                    "Merged transported delta is identically zero: every alpha would evaluate "
                    "the untouched target base model. Check the transport diagnostics above."
                )

        search_planner = build_search_planner(
            cfg=cfg, base_method_params=method_params
        )

        # ---- Dispatch evaluation backend ----
        if is_harness_only or harness_tasks_resolved:
            from .lm_harness_runner import run as run_harness
            from .lm_harness_runner import score_by_task

            best_harness_eval: SearchEvaluation | None = None
            harness_results_by_alpha: dict[float, dict[str, float]] = {}
            harness_search_results: list[SearchEvaluation] = []

            baseline_harness_results = _baseline_summary()

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
                        samples=harness_samples,
                    )
                    for task_name, acc in harness_results.items():
                        print(f"  {task_name}: {acc:.4f}")

                    score = score_by_task(harness_results, list(harness_tasks_resolved))
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
                    "before_rebase_model": (
                        "extended_source_base" if run_block_extension_prestep else "source_base"
                    ),
                    # Named per-alpha metrics: search_results only keeps a flat
                    # per_task_acc list, which loses which task each number is.
                    "harness_results_by_alpha": {
                        f"{a:g}": r for a, r in sorted(harness_results_by_alpha.items())
                    },
                    "merged_delta": delta_stats,
                    "task_vectors": task_vector_report,
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
                "merged_delta": delta_stats,
                "task_vectors": task_vector_report,
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
