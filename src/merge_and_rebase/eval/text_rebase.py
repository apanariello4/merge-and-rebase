from __future__ import annotations

import argparse
import os
import random
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from merge_and_rebase.utils.helpers import load_json

from ..cli_args import (
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    build_logging_overrides,
    merge_non_none,
)
from ..data.text_loaders import build_nli_task_data, build_nli_tokenized_loader
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..merge.methods._common import axpy_state_dict
from ..models.text_lm import TextBuildConfig, TextLM
from ..rebase.methods.theseus import ActivationStore
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run


_HEAD_PATTERNS = ("classification_head", "classifier", "score")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_output_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    for attr in ("last_hidden_state", "logits"):
        value = getattr(output, attr, None)
        if torch.is_tensor(value):
            return value
    if isinstance(output, (tuple, list)):
        return next((x for x in output if torch.is_tensor(x)), None)
    return None


class ActivationHook:
    """Capture parameterized-module inputs and outputs, as in GradientSigns."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.handles.append(model.register_forward_hook(self._make_hook("")))
        for name, module in model.named_modules():
            if name and list(module.parameters(recurse=False)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            inp = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else inputs
            if torch.is_tensor(inp):
                self.inputs[name] = inp.detach().cpu()
            out = _extract_output_tensor(output)
            if out is not None:
                self.outputs[name] = out.detach().cpu()

        return hook

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def process_tokens(
    source: torch.Tensor,
    target: torch.Tensor,
    strategy: str = "interpolate",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn aligned model activations into row matrices for cross-covariance."""
    mode = str(strategy).strip().lower()
    if mode not in {"cls", "mean", "interpolate"}:
        raise ValueError("token_strategy must be one of: cls, mean, interpolate")
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"Activation batch mismatch: {source.shape[0]} != {target.shape[0]}")

    if mode == "cls":
        if source.ndim == 3:
            source = source[:, 0, :]
        if target.ndim == 3:
            target = target[:, 0, :]
    elif mode == "mean":
        if source.ndim == 3:
            source = source.mean(dim=1)
        if target.ndim == 3:
            target = target.mean(dim=1)
    elif source.ndim == 3 and target.ndim == 3:
        if source.shape[1] != target.shape[1]:
            source = F.interpolate(
                source.permute(0, 2, 1), size=target.shape[1], mode="linear", align_corners=False
            ).permute(0, 2, 1)
        source = source.reshape(-1, source.shape[-1])
        target = target.reshape(-1, target.shape[-1])

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Unsupported aligned activation shapes: {tuple(source.shape)}, {tuple(target.shape)}")
    return source, target


def _move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    # Labels are irrelevant for activation calibration and can trigger an avoidable loss path.
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "labels"}


@torch.inference_mode()
def collect_activations(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_dataloader: Iterable[Mapping[str, Any]],
    target_dataloader: Iterable[Mapping[str, Any]],
    *,
    device: str,
    n_batches: int,
    token_strategy: str = "interpolate",
) -> dict[str, ActivationStore]:
    """Collect paired T5 activation cross-products on the same NLI examples."""
    registry: dict[str, ActivationStore] = {}
    source_hook = ActivationHook(source_model)
    target_hook = ActivationHook(target_model)
    try:
        for batch_idx, (source_batch, target_batch) in enumerate(zip(source_dataloader, target_dataloader, strict=True)):
            if batch_idx >= int(n_batches):
                break
            source_model(**_move_batch(source_batch, device))
            target_model(**_move_batch(target_batch, device))

            for name in set(source_hook.inputs) & set(target_hook.inputs):
                try:
                    src, tgt = process_tokens(source_hook.inputs[name], target_hook.inputs[name], token_strategy)
                except ValueError:
                    continue
                registry.setdefault(f"{name}.in", ActivationStore()).update(src, tgt)
            for name in set(source_hook.outputs) & set(target_hook.outputs):
                try:
                    src, tgt = process_tokens(source_hook.outputs[name], target_hook.outputs[name], token_strategy)
                except ValueError:
                    continue
                registry.setdefault(f"{name}.out", ActivationStore()).update(src, tgt)
            source_hook.clear()
            target_hook.clear()
    finally:
        source_hook.remove()
        target_hook.remove()
    return registry


def svd_transport(
    cov_in: torch.Tensor,
    cov_out: torch.Tensor,
    weight: torch.Tensor,
    *,
    device: str = "cpu",
    double_precision: bool = False,
) -> torch.Tensor:
    """GradientSigns transport: ``T_out.T @ weight @ T_in``."""
    dtype = torch.float64 if double_precision else torch.float32
    u_out, _, vh_out = torch.linalg.svd(cov_out.to(device=device, dtype=dtype), full_matrices=False)
    u_in, _, vh_in = torch.linalg.svd(cov_in.to(device=device, dtype=dtype), full_matrices=False)
    t_out = u_out @ vh_out
    t_in = u_in @ vh_in
    return (t_out.T @ weight.to(device=device, dtype=dtype) @ t_in).float()


def align_task_vector(
    target_model: torch.nn.Module,
    source_delta: Mapping[str, torch.Tensor],
    stats: Mapping[str, ActivationStore],
    *,
    source_base: Mapping[str, torch.Tensor] | None = None,
    center_acts: bool = False,
    norm_matching: bool = False,
    double_precision: bool = False,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Transport a source T5 task vector into a target T5 parameter space."""
    target_sd = target_model.state_dict()
    param_to_module: dict[str, str] = {}
    for module_name, module in target_model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            key = f"{module_name}.{param_name}" if module_name else param_name
            param_to_module[key] = module_name

    aligned: dict[str, torch.Tensor] = {}
    for key, delta_source in source_delta.items():
        target_ref = target_sd.get(key)
        if target_ref is None:
            continue
        module_name = param_to_module.get(key, key.rsplit(".", 1)[0] if "." in key else "")
        transported = torch.zeros_like(target_ref, device=device, dtype=torch.float32)

        if delta_source.ndim == 2:
            in_store = stats.get(f"{module_name}.in")
            out_store = stats.get(f"{module_name}.out")
            if in_store is not None and out_store is not None:
                cov_in = in_store.get_covariance(center=center_acts)
                cov_out = out_store.get_covariance(center=center_acts)
                if cov_in is not None and cov_out is not None:
                    try:
                        transported = svd_transport(
                            cov_in,
                            cov_out,
                            delta_source,
                            device=device,
                            double_precision=double_precision,
                        )
                    except (RuntimeError, ValueError):
                        pass
        elif delta_source.ndim == 1:
            out_store = stats.get(f"{module_name}.out")
            cov_out = out_store.get_covariance(center=center_acts) if out_store is not None else None
            if cov_out is not None:
                dtype = torch.float64 if double_precision else torch.float32
                u, _, vh = torch.linalg.svd(cov_out.to(device=device, dtype=dtype), full_matrices=False)
                transported = (delta_source.to(device=device, dtype=dtype) @ (u @ vh)).float()

        if transported.shape != target_ref.shape:
            transported = torch.zeros_like(target_ref, device=device, dtype=torch.float32)
        if norm_matching and source_base is not None and key in source_base and transported.norm() > 1e-8:
            source_norm = source_base[key].float().norm()
            if source_norm > 1e-8:
                scale = (delta_source.float().norm() / source_norm) * (target_ref.float().norm() / transported.norm())
                transported.mul_(scale)
        aligned[key] = transported.to(device="cpu", dtype=target_ref.dtype)
    return aligned


def _task_vector(base: Mapping[str, torch.Tensor], tuned: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tuned = align_to_base_keys(tuned, base)
    return {
        key: tuned[key].detach().cpu().float() - value.detach().cpu().float()
        for key, value in base.items()
        if key in tuned and value.is_floating_point() and tuned[key].shape == value.shape
    }


def _is_head_key(key: str, patterns: Iterable[str]) -> bool:
    return any(pattern and pattern in key for pattern in patterns)


def _copy_selected_parameters(
    destination: dict[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    patterns: Iterable[str],
) -> list[str]:
    copied: list[str] = []
    for key, value in source.items():
        if key in destination and destination[key].shape == value.shape and _is_head_key(key, patterns):
            destination[key] = value.detach().cpu().to(dtype=destination[key].dtype)
            copied.append(key)
    return copied


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Transport a T5 task vector from source base A to target base B")
    add_config_arg(parser)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--source-model-name-or-path", type=str, default=None)
    parser.add_argument("--target-model-name-or-path", type=str, default=None)
    parser.add_argument("--source-tuned-ckpt", type=str, default=None)
    parser.add_argument("--target-tuned-ckpt", type=str, default=None)
    add_device_dtype_args(parser, device_default=None, dtype_default=None)
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--n-batches-act", type=int, default=None)
    parser.add_argument("--token-strategy", choices=["cls", "mean", "interpolate"], default=None)
    parser.add_argument("--center-acts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--norm-matching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--double-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--layers-to-skip", type=str, nargs="*", default=None)
    parser.add_argument(
        "--eval-mode",
        choices=["A_ft_head_on_b", "B_ft_head_on_b", "base_head_on_b"],
        default=None,
    )
    parser.add_argument("--head-patterns", type=str, nargs="+", default=None)
    parser.add_argument("--test-base-models", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--aligned-tv-cache", type=str, default=None)
    parser.add_argument("--save-transported-tv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-fast-tokenizer", action=argparse.BooleanOptionalAction, default=None)
    add_logging_args(parser)
    return parser


def main() -> None:
    run_logger = None
    try:
        parser = _build_parser()
        args = parser.parse_args()
        cfg: dict[str, Any] = load_json(args.config) if args.config else {}
        cli = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
        cfg = merge_non_none(cfg, cli)
        logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
        cfg["logging"] = logging_cfg

        required = ("task", "source_model_name_or_path", "target_model_name_or_path", "source_tuned_ckpt")
        missing = [key for key in required if not cfg.get(key)]
        if missing:
            raise ValueError(f"Missing required configuration fields: {missing}")

        seed = int(cfg.get("seed", 42))
        _set_seed(seed)
        device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        task = str(cfg["task"]).strip().lower()
        train_data = build_nli_task_data(task=task, split="train", max_samples=cfg.get("max_train_samples"))
        eval_data = build_nli_task_data(task=task, split="test", max_samples=cfg.get("max_eval_samples"))
        num_labels = int(cfg.get("num_labels", max(3, len(train_data.labels))))
        common_build = {
            "model_arch": "t5",
            "device": device,
            "dtype": cfg.get("dtype"),
            "model_kind": "sequence_classification",
            "num_labels": num_labels,
            "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
            "use_fast_tokenizer": bool(cfg.get("use_fast_tokenizer", True)),
        }
        source = TextLM.build(TextBuildConfig(model_name_or_path=str(cfg["source_model_name_or_path"]), **common_build))
        target = TextLM.build(TextBuildConfig(model_name_or_path=str(cfg["target_model_name_or_path"]), **common_build))
        source_base = {k: v.detach().cpu() for k, v in source.model.state_dict().items()}
        target_base = {k: v.detach().cpu() for k, v in target.model.state_dict().items()}
        source_tuned = align_to_base_keys(load_ckpt(str(cfg["source_tuned_ckpt"])), source_base)
        source_delta = _task_vector(source_base, source_tuned)
        target_tuned = None
        target_tuned_ref = cfg.get("target_tuned_ckpt")
        if target_tuned_ref:
            target_tuned = align_to_base_keys(load_ckpt(str(target_tuned_ref)), target_base)

        batch_size = int(cfg.get("batch_size", 32))
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": int(cfg.get("num_workers", 0)),
            "max_length": int(cfg.get("max_length", 256)),
        }
        source_train = build_nli_tokenized_loader(task_data=train_data, tokenizer=source.tokenizer, **loader_kwargs)
        target_train = build_nli_tokenized_loader(task_data=train_data, tokenizer=target.tokenizer, **loader_kwargs)
        target_eval = build_nli_tokenized_loader(task_data=eval_data, tokenizer=target.tokenizer, **loader_kwargs)

        summary_path = default_summary_path(
            entrypoint="eval.text_rebase",
            logging_cfg=logging_cfg,
            default_parent=Path(str(cfg["output_dir"])) if cfg.get("output_dir") else None,
        )
        run_logger = start_run(
            entrypoint="eval.text_rebase",
            logging_cfg=logging_cfg,
            summary_path=summary_path,
            metadata={"config_path": args.config, "resolved_config": cfg, "summary_path": str(summary_path)},
        )

        cache_path = cfg.get("aligned_tv_cache")
        if cache_path and Path(str(cache_path)).exists():
            aligned = load_ckpt(str(cache_path))
        else:
            stats = collect_activations(
                source.model,
                target.model,
                source_train.loader,
                target_train.loader,
                device=device,
                n_batches=int(cfg.get("n_batches_act", 5)),
                token_strategy=str(cfg.get("token_strategy", "interpolate")),
            )
            aligned = align_task_vector(
                target.model,
                source_delta,
                stats,
                source_base=source_base,
                center_acts=bool(cfg.get("center_acts", False)),
                norm_matching=bool(cfg.get("norm_matching", False)),
                double_precision=bool(cfg.get("double_precision", False)),
                device=device,
            )
            if cache_path:
                Path(str(cache_path)).parent.mkdir(parents=True, exist_ok=True)
                torch.save(aligned, str(cache_path))

        save_tv = cfg.get("save_transported_tv")
        if save_tv:
            Path(str(save_tv)).parent.mkdir(parents=True, exist_ok=True)
            torch.save(aligned, str(save_tv))

        eval_base = deepcopy(target_base)
        head_patterns = tuple(cfg.get("head_patterns", _HEAD_PATTERNS))
        eval_mode = str(cfg.get("eval_mode", "A_ft_head_on_b"))
        if eval_mode == "A_ft_head_on_b":
            _copy_selected_parameters(eval_base, source_tuned, head_patterns)
        elif eval_mode == "B_ft_head_on_b":
            if target_tuned is None:
                raise ValueError("B_ft_head_on_b requires target_tuned_ckpt")
            _copy_selected_parameters(eval_base, target_tuned, head_patterns)

        skipped = tuple(cfg.get("layers_to_skip", ())) + head_patterns
        aligned_for_eval = {k: v for k, v in aligned.items() if not _is_head_key(k, skipped)}

        def evaluate(state: Mapping[str, torch.Tensor]) -> float:
            load_into_model(target.model, state, strict=bool(cfg.get("strict_load", False)))
            return target.sequence_classification_accuracy(
                target_eval.loader,
                device=device,
                mask_class=target_eval.mask_class,
            )

        results: dict[str, float] = {}
        if bool(cfg.get("test_base_models", False)):
            results["baseline_model_b"] = evaluate(eval_base)
            if target_tuned is not None:
                native_state = dict(target_base)
                native_state.update(target_tuned)
                results["native_task_vector_b"] = evaluate(native_state)

        alphas = [float(x) for x in cfg.get("alphas", [0.2, 0.4, 0.6, 0.8, 1.0])]
        for alpha in alphas:
            score = evaluate(axpy_state_dict(eval_base, aligned_for_eval, alpha=alpha))
            results[f"aligned_tv_a_alpha_{alpha:g}"] = score
            run_logger.log_event(
                "alpha_eval_end",
                metrics={"alpha/value": alpha, "alpha/accuracy": score},
                context={"task": task, "eval_mode": eval_mode},
            )
            print(f"{task}: alpha={alpha:g}, accuracy={score:.6f}")

        final_summary = {
            "task": task,
            "source_model": str(cfg["source_model_name_or_path"]),
            "target_model": str(cfg["target_model_name_or_path"]),
            "eval_mode": eval_mode,
            "results": results,
            "transported_parameter_count": len(aligned),
        }
        run_logger.log_summary(final_summary)
        run_logger.finish("success")

        if cfg.get("output_dir"):
            output_dir = Path(str(cfg["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(final_summary, output_dir / f"{task}_text_rebase_results.pt")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
