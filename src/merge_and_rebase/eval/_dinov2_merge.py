"""Backbone-only DINOv2 merging with task-specific learned heads."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers

from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.finetune._dinov2 import make_loaders, top1
from merge_and_rebase.io.ckpt import resolve_ckpt_path
from merge_and_rebase.io.utils import atomic_write_json
from merge_and_rebase.merge.registry import get_method
from merge_and_rebase.merge.runtime import validate_prepared_merge
from merge_and_rebase.models.dinov2_classifier import DinoV2Classifier
from merge_and_rebase.run_logging import default_summary_path, finish_with_error, start_run
from merge_and_rebase.utils.helpers import parse_csv


def load_checkpoint(path):
    return torch.load(resolve_ckpt_path(str(path)), map_location="cpu", weights_only=True)


def validate_checkpoint(base, tuned, task):
    if tuned.get("format") != "dinov2_classifier_v1" or tuned.get("task") != task:
        raise ValueError(f"Expected DINOv2 classification checkpoint for {task}.")
    if tuned["identity"] != base["identity"]:
        raise ValueError(f"DINOv2 base/model/processor mismatch for {task}.")
    state = tuned["backbone_state"]
    reference = base["backbone_state"]
    if state.keys() != reference.keys() or any(state[k].shape != reference[k].shape for k in reference):
        raise ValueError(f"Incompatible backbone tensors for {task}.")
    original = load_checkpoint(tuned["base_checkpoint"])
    if original["identity"] != base["identity"] or any(
        not torch.equal(original["backbone_state"][k], v) for k, v in reference.items()
    ):
        raise ValueError(f"Task {task} was trained from different base weights.")


def prepare_merge(base, tuned, method, weights=None, params=None):
    merger = get_method(method)
    if not callable(getattr(merger, "prepare", None)) or not callable(getattr(merger, "apply", None)):
        raise ValueError(f"Merge method {method} does not support dense prepared merging.")
    prepared = merger.prepare(base=base, tuned=tuned, weights=weights, strict=True, method_params=params or {})
    validate_prepared_merge(prepared, expected_base=base, method_name=method)
    return merger, prepared


@contextmanager
def _merge_preparation_slot(cfg):
    """Serialize CPU merge preparation while allowing GPU evaluation to overlap."""
    lock_path = cfg.get("exclusive_merge_lock")
    if not lock_path:
        yield
        return

    import fcntl

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        print(f"Waiting for CPU merge preparation slot: {path}", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print(f"Acquired CPU merge preparation slot: {cfg.get('method')}", flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            print("Released CPU merge preparation slot", flush=True)


def run_merge(cfg):
    from merge_and_rebase.finetune.train_vision import _device

    # These options require CLIP-specific runtime integrations.
    for key in ("postmerge", "zero_shot_only", "single_acc_zero_shot", "source_base_ckpt", "target_base_ckpt"):
        if cfg.get(key):
            raise ValueError(f"DINOv2 merging does not support {key}.")
    if cfg.get("forward_mode", "standard") not in {None, "auto", "standard"} or cfg.get(
        "peft_subspace", "full"
    ) not in {None, "full"}:
        raise ValueError("DINOv2 merging requires standard forward mode and full dense subspace.")
    if cfg.get("dtype") not in {None, "fp32", "float32"}:
        raise ValueError("DINOv2 currently supports dtype=fp32.")
    suite = SUITES[cfg.get("suite", "vision8")]
    tasks = cfg.get("tasks", "all")
    tasks = list(suite.tasks) if tasks == "all" else (parse_csv(tasks) if isinstance(tasks, str) else tasks)
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(suite.tasks):
        raise ValueError("Invalid tasks for selected vision suite.")
    tuned_paths = cfg.get("tuned_ckpts", {})
    if set(tasks) - tuned_paths.keys():
        raise ValueError("DINOv2 merging requires a tuned_ckpts entry for every task.")
    tuned = {task: load_checkpoint(tuned_paths[task]) for task in tasks}
    base = load_checkpoint(cfg.get("base_ckpt") or tuned[tasks[0]]["base_checkpoint"])
    for task, payload in tuned.items():
        validate_checkpoint(base, payload, task)
    backbone_cfg = cfg.get("backbone", {})
    if backbone_cfg.get("model_id", base["identity"]["model_id"]) != base["identity"]["model_id"]:
        raise ValueError("Configured model_id differs from saved DINOv2 backbone.")
    method = cfg.get("method", "task_arithmetic")
    weights = cfg.get("weights")
    if isinstance(weights, dict):
        weights = [weights[task] for task in tasks]
    print(f"Preparing {method} for {len(tasks)} task backbones (heads excluded)", flush=True)
    with _merge_preparation_slot(cfg):
        merger, prepared = prepare_merge(
            base["backbone_state"],
            [tuned[t]["backbone_state"] for t in tasks],
            method,
            weights,
            cfg.get("method_params"),
        )
    device = _device(cfg.get("device", "cuda"))
    processor_cfg = base["identity"]["processor"]
    processor = getattr(transformers, processor_cfg["image_processor_type"]).from_dict(processor_cfg)
    loaders = {}
    for task, payload in tuned.items():
        data = dict(payload["data"])
        for key in ("batch_size", "num_workers", "val_fraction"):
            if cfg.get(key) is not None:
                data[key] = cfg[key]
        if "val_fraction" in data and float(data["val_fraction"]) != float(payload["data"].get("val_fraction", 0.1)):
            raise ValueError("Evaluation val_fraction must match training to preserve held-out test data.")
        loaders[task] = make_loaders(task, processor, data, payload["seed"])
        if list(loaders[task].classnames) != payload["classnames"]:
            raise ValueError(f"Class order mismatch for {task}.")
    references = {}
    if cfg.get("single_task_summaries"):
        for task in tasks:
            with Path(cfg["single_task_summaries"][task]).open() as handle:
                summary = json.load(handle)
            references[task] = {"val": float(summary["best_val_top1"]), "test": float(summary["test_top1"])}
            if any(value <= 0 for value in references[task].values()):
                raise ValueError(f"Single-task reference accuracies must be positive for {task}.")
    selection_metric = cfg.get("selection_metric", "mean_top1")
    if selection_metric not in {"mean_top1", "mean_normalized_top1"}:
        raise ValueError("Unknown alpha selection_metric.")
    if selection_metric == "mean_normalized_top1" and not references:
        raise ValueError("Normalized alpha selection requires single_task_summaries.")
    model = None

    def evaluate(state, split):
        nonlocal model
        scores = {}
        for task in tasks:
            if model is None or model.head.out_features != len(tuned[task]["classnames"]):
                model = DinoV2Classifier.from_checkpoint(tuned[task]).to(device)
            model.backbone.load_state_dict(state, strict=True)
            model.head.load_state_dict(tuned[task]["head_state"], strict=True)
            scores[task] = top1(model, getattr(loaders[task], split), device)
            for key, value in model.head.state_dict().items():
                if not torch.equal(value.detach().cpu(), tuned[task]["head_state"][key]):
                    raise RuntimeError(f"Task head changed during inference for {task}.")
            print(f"{split} {task}: top1={scores[task]:.6f}", flush=True)
        result = {"per_task_top1": scores, "mean_top1": sum(scores.values()) / len(scores)}
        if references:
            normalized = {task: scores[task] / references[task][split] for task in tasks}
            result.update(
                per_task_normalized_top1=normalized, mean_normalized_top1=sum(normalized.values()) / len(tasks)
            )
        return result

    alphas = cfg.get("alphas", [cfg.get("alpha", 1.0)])
    if isinstance(alphas, str):
        alphas = [float(x) for x in alphas.split(",")]
    if cfg.get("alpha_search"):
        import numpy as np

        step = float(cfg.get("alpha_step", 0.1))
        if step <= 0:
            raise ValueError("alpha_step must be positive.")
        alphas = np.arange(float(cfg.get("alpha_min", 0)), float(cfg.get("alpha_max", 2)) + step / 2, step).tolist()
    if not alphas:
        raise ValueError("At least one alpha is required.")
    if cfg.get("fixed_eval_split") and len(alphas) != 1:
        raise ValueError("fixed_eval_split requires a single alpha.")
    logging_cfg = cfg.get("logging", {})
    logger = start_run(
        entrypoint="eval.vision_merge",
        logging_cfg=logging_cfg,
        summary_path=default_summary_path(
            entrypoint="eval.vision_merge",
            logging_cfg=logging_cfg,
            default_parent=Path(cfg.get("save_merged", "results/dinov2/merged.pt")).parent,
        ),
        metadata={"resolved_config": cfg},
    )
    try:
        candidates = []
        for alpha in alphas:
            print(f"Evaluating {method} alpha={float(alpha):.2f}", flush=True)
            state = merger.apply(prepared, alpha=float(alpha))
            # Some algorithms intentionally omit tensors outside their merge key filter.
            state = {**base["backbone_state"], **state}
            split = cfg.get("fixed_eval_split") or "val"
            if split not in {"val", "test"}:
                raise ValueError("fixed_eval_split must be val or test.")
            scores = evaluate(state, split)
            candidates.append({"alpha": float(alpha), **scores})
            print(
                f"alpha={float(alpha):.2f}: mean_top1={scores['mean_top1']:.6f} selection_score={scores[selection_metric]:.6f}",
                flush=True,
            )
            logger.log_event(
                "alpha",
                metrics={
                    "alpha": float(alpha),
                    **{key: value for key, value in scores.items() if key.startswith("mean_")},
                },
                context={"method": method},
            )
            if cfg.get("results"):
                atomic_write_json(
                    Path(cfg["results"]).with_suffix(".progress.json"),
                    {
                        "status": "searching",
                        "method": method,
                        "selection_metric": selection_metric,
                        "candidates": candidates,
                    },
                )
            if (
                cfg.get("alpha_early_stop")
                and len(candidates) > 1
                and candidates[-1][selection_metric] < candidates[-2][selection_metric]
            ):
                break
        best = max(candidates, key=lambda c: c[selection_metric])
        merged = {**base["backbone_state"], **merger.apply(prepared, alpha=best["alpha"])}
        result = {
            "method": method,
            "alpha": best["alpha"],
            "selection_metric": selection_metric,
            "candidates": candidates,
        }
        if cfg.get("fixed_eval_split"):
            result[cfg["fixed_eval_split"]] = {key: value for key, value in best.items() if key != "alpha"}
        else:
            result["val"] = {key: value for key, value in best.items() if key != "alpha"}
            result["test"] = evaluate(merged, "test")
        if cfg.get("save_merged"):
            destination = Path(cfg["save_merged"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": "dinov2_merged_v1",
                    "identity": base["identity"],
                    "model_config": base["identity"]["model_config"],
                    "processor": base["identity"]["processor"],
                    "backbone_state": merged,
                    "tasks": {
                        task: {key: tuned[task][key] for key in ("head_state", "classnames", "data", "seed")}
                        for task in tasks
                    },
                    "merge": result,
                },
                destination,
            )
        if cfg.get("results"):
            atomic_write_json(Path(cfg["results"]), result)
            atomic_write_json(Path(cfg["results"]).with_suffix(".progress.json"), {"status": "completed", **result})
        print(result)
        logger.log_summary(result)
        logger.finish("success")
    except Exception as exc:
        finish_with_error(logger, exc)
        raise
