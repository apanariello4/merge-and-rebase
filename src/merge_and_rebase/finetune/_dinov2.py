"""DINOv2 runtime dispatched by the standard vision training command."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from merge_and_rebase.cli_args import build_logging_overrides
from merge_and_rebase.data.vision_loaders import build_vision_loaders, load_hf_splits
from merge_and_rebase.eval.datasets.vision8_14_20 import _vision_spec
from merge_and_rebase.io.utils import atomic_write_json
from merge_and_rebase.models.dinov2_classifier import DinoPreprocess, DinoV2Classifier
from merge_and_rebase.run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run

from .schedulers import build_lr_scheduler
from .strategies.base import build_optimizer


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def validate_options(cfg):
    from .train_vision import _get

    for key in (
        "strategy.attention",
        "strategy.peft",
        "strategy.text_prompt_tuning",
        "strategy.text_embeddings_finetune",
        "regularization",
        "initialization",
    ):
        value = _get(cfg, key, None)
        if value and value != {"name": "none"}:
            raise ValueError(f"DINOv2 does not support {key}.")
    if _get(cfg, "strategy.forward_mode", "standard") != "standard" or _get(cfg, "train.text_only", False):
        raise ValueError("DINOv2 requires standard image classification forward mode.")
    if _get(cfg, "strategy.params", {}) or _get(cfg, "strategy.name", "full") not in {
        "full",
        "linear_probe",
        "classifier_then_backbone",
    }:
        raise ValueError("DINOv2 supports full, linear_probe, or classifier_then_backbone with empty strategy.params.")
    if _get(cfg, "output.save_format", "full") != "full":
        raise ValueError("DINOv2 requires output.save_format=full.")
    if _get(cfg, "dtype", "fp32") not in {None, "fp32", "float32"}:
        raise ValueError("DINOv2 currently supports dtype=fp32.")


def make_loaders(task, processor, data, seed):
    path, config, split_map = _vision_spec(task)
    dataset = load_hf_splits(path, config=config, requested_splits=list(split_map.values()))
    return build_vision_loaders(
        dataset,
        hf_path=path,
        preprocess=DinoPreprocess(processor),
        ft_epochs=1,
        split_map=split_map,
        batch_size=int(data.get("batch_size", 64)),
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=bool(data.get("pin_memory", True)),
        val_fraction=float(data.get("val_fraction", 0.1)),
        seed=int(data.get("split_seed", seed)),
    )


@torch.no_grad()
def top1(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        labels = labels.to(device)
        correct += (model(images.to(device)).argmax(-1) == labels).sum().item()
        total += labels.numel()
    if not total:
        raise ValueError("Cannot evaluate an empty dataset.")
    return correct / total


def train_epochs(model, loaders, cfg, device, on_epoch=None, on_checkpoint=None):
    from .train_vision import _get

    train = cfg["train"]
    epochs = int(train["epochs"])
    accumulation = int(train.get("accumulate_grad_batches", 1))
    max_batches = train.get("max_train_batches")
    batches = min(len(loaders.train), int(max_batches)) if max_batches is not None else len(loaders.train)
    if epochs < 1 or accumulation < 1 or batches < 1:
        raise ValueError("epochs, accumulation, and training batch count must be positive.")
    mode = _get(cfg, "strategy.name", "full")
    model.set_training_mode(mode)
    opt = build_optimizer(
        (p for p in model.parameters() if p.requires_grad),
        _get(train, "optimizer.name", "adamw"),
        float(train.get("lr", 1e-5)),
        float(train.get("weight_decay", 0.1)),
    )
    schedule = build_lr_scheduler(
        opt,
        name=_get(train, "lr_scheduler.name", "cosine"),
        base_lrs=float(train.get("lr", 1e-5)),
        warmup_length=int(_get(train, "lr_scheduler.warmup_steps", 0)),
        steps=epochs * math.ceil(batches / accumulation),
    )
    best = None
    best_acc = -1.0
    best_epoch = -1
    stale = step = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for index, (images, labels) in enumerate(loaders.train):
            if index >= batches:
                break
            # The final, possibly short accumulation window needs its actual size.
            window = min(accumulation, batches - (index // accumulation) * accumulation)
            loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
            (loss / window).backward()
            loss_sum += loss.item()
            if (index + 1) % accumulation == 0 or index + 1 == batches:
                clip = train.get("grad_clip_norm")
                if clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                schedule(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        accuracy = top1(model, loaders.val, device)
        if accuracy > best_acc:
            best_acc, best_epoch, best, stale = accuracy, epoch, cpu_state(model), 0
        else:
            stale += 1
        metrics = {"epoch": epoch, "loss": loss_sum / batches, "val_top1": accuracy, "lr": opt.param_groups[0]["lr"]}
        print(metrics)
        if on_epoch:
            on_epoch(metrics)
        if on_checkpoint:
            on_checkpoint(best, cpu_state(model), metrics)
        if train.get("early_stopping", False) and stale >= int(train.get("early_stopping_patience", 5)):
            break
    last = cpu_state(model)
    model.load_state_dict(best, strict=True)
    return (
        best,
        last,
        {
            "best_val_top1": best_acc,
            "best_epoch": best_epoch,
            "last_epoch": epoch,
            "test_top1": top1(model, loaders.test, device),
        },
    )


def train_variant(model, loaders, cfg, device, on_epoch=None, on_checkpoint=None):
    """Train a task jointly, or fit its head before updating only the backbone."""
    mode = cfg.get("strategy", {}).get("name", "full")
    stages = [(mode, cfg)]
    if mode == "classifier_then_backbone":
        classifier_cfg = deepcopy(cfg)
        classifier_cfg["strategy"]["name"] = "linear_probe"
        classifier_cfg["train"].update({"lr": 1e-3, "weight_decay": 0.0})
        classifier_cfg["train"].update(cfg["train"].get("classifier_stage", {}))
        backbone_cfg = deepcopy(cfg)
        backbone_cfg["strategy"]["name"] = "backbone_only"
        stages = [("classifier", classifier_cfg), ("backbone", backbone_cfg)]
    stage_summaries = {}
    for stage, stage_cfg in stages:
        print(f"Starting stage {stage}", flush=True)
        best, last, summary = train_epochs(
            model,
            loaders,
            stage_cfg,
            device,
            (lambda metrics, stage=stage: on_epoch(stage, metrics)) if on_epoch else None,
            (lambda best, last, metrics, stage=stage: on_checkpoint(stage, best, last, metrics))
            if on_checkpoint
            else None,
        )
        # train_epochs restores the validation-selected model, including the fitted
        # classifier. A new optimizer in the next stage excludes its frozen weights.
        stage_summaries[stage] = summary
    if mode == "classifier_then_backbone":
        summary = {**summary, "stages": stage_summaries}
    return best, last, summary


def run_training(args, cfg_file, tasks):
    from .train_vision import _deep_update, _device, _get, _set_seed

    common = cfg_file["common"]
    # Validate all tasks before downloading model weights or datasets.
    configs = {}
    if not tasks:
        raise ValueError("At least one training task is required.")
    for task in tasks:
        cfg = _deep_update(deepcopy(common), cfg_file.get("datasets", {}).get(task, {}))
        validate_options(cfg)
        if (
            _get(cfg, "data.train_preprocess", "eval") != "eval"
            or _get(cfg, "data.loader_profile", "hf") != "hf"
            or _get(cfg, "data.data_root")
        ):
            raise ValueError("DINOv2 uses HF loaders and checkpoint preprocessing.")
        if _get(cfg, "output.out_dir") != _get(common, "output.out_dir"):
            raise ValueError("DINOv2 tasks must share output.out_dir; each receives its own task subdirectory.")
        if cfg["backbone"] != common["backbone"]:
            raise ValueError("All DINOv2 tasks must share a backbone configuration.")
        if not _get(cfg, "train.epochs"):
            raise ValueError(f"Missing train.epochs for {task}.")
        configs[task] = cfg
    requested_device = args.device or common.get("device", "cuda")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing to start a full run on CPU.")
    device = _device(requested_device)
    out = Path(_get(common, "output.out_dir", "checkpoints/dinov2"))
    backbone_cfg = common["backbone"]
    model_id = backbone_cfg.get("model_id", "facebook/dinov2-base")
    backbone, processor = DinoV2Classifier.pretrained(model_id, backbone_cfg.get("revision"))
    base = cpu_state(backbone)
    identity = {
        "model_id": model_id,
        "revision": getattr(backbone.config, "_commit_hash", None) or backbone_cfg.get("revision"),
        "model_config": backbone.config.to_dict(),
        "processor": json.loads(processor.to_json_string()),
    }
    out.mkdir(parents=True, exist_ok=True)
    base_path = out / "base.pt"
    if base_path.exists():
        old = torch.load(base_path, map_location="cpu", weights_only=True)
        if old["identity"] != identity or any(not torch.equal(old["backbone_state"][k], v) for k, v in base.items()):
            raise ValueError(f"Existing base checkpoint differs: {base_path}. Choose a new output directory.")
    else:
        torch.save({"format": "dinov2_base_v1", "identity": identity, "backbone_state": base}, base_path)
    logging_cfg = merge_logging_config(common.get("logging", {}), build_logging_overrides(args))
    logger = start_run(
        entrypoint="finetune.train_vision",
        logging_cfg=logging_cfg,
        summary_path=default_summary_path(
            entrypoint="finetune.train_vision", logging_cfg=logging_cfg, default_parent=out
        ),
        metadata={"resolved_config": cfg_file, "tasks": tasks},
    )
    results = {}
    try:
        for task, cfg in configs.items():
            seed = int(cfg.get("seed", 42))
            _set_seed(seed)
            data = deepcopy(cfg.get("data", {}))
            if (
                args.train_preprocess not in {None, "eval"}
                or args.vision_loader_profile not in {None, "hf"}
                or args.vision_data_root
            ):
                raise ValueError("DINOv2 currently uses HF loaders and the checkpoint image processor.")
            loaders = make_loaders(task, processor, data, seed)
            model = DinoV2Classifier(deepcopy(backbone), len(loaders.classnames)).to(device)
            mode = _get(cfg, "strategy.name", "full")
            task_dir = out / task
            task_dir.mkdir(parents=True, exist_ok=True)
            best_path = task_dir / f"{mode}_best.pt"
            if args.skip_existing_task_vectors and best_path.exists():
                continue
            payload = {
                "format": "dinov2_classifier_v1",
                "identity": identity,
                "model_config": identity["model_config"],
                "processor": identity["processor"],
                "base_checkpoint": str(base_path.resolve()),
                "task": task,
                "classnames": list(loaders.classnames),
                "strategy": mode,
                "data": data,
                "seed": seed,
                "config": cfg,
            }

            def save(state, path, payload=payload):
                torch.save(
                    {
                        **payload,
                        "backbone_state": {
                            k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")
                        },
                        "head_state": {k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")},
                    },
                    path,
                )

            def checkpoint(stage, best, last, metrics, task_dir=task_dir, mode=mode, cfg=cfg, save=save):
                stem = mode + "_classifier" if stage == "classifier" else mode
                atomic_write_json(task_dir / "progress.json", {"stage": stage, **metrics})
                if _get(cfg, "output.save_checkpoints", True):
                    save(best, task_dir / f"{stem}_best.pt")
                    if _get(cfg, "output.save_last_epoch", False):
                        save(last, task_dir / f"{stem}_last.pt")

            best, last, summary = train_variant(
                model,
                loaders,
                cfg,
                device,
                lambda stage, metrics, task=task: logger.log_event(
                    "epoch", metrics=metrics, context={"task": task, "stage": stage}
                ),
                checkpoint,
            )
            if _get(cfg, "output.save_checkpoints", True):
                save(best, best_path)
                if _get(cfg, "output.save_last_epoch", False):
                    save(last, task_dir / f"{mode}_last.pt")
            results[task] = summary
            atomic_write_json(task_dir / f"{mode}.json", summary)
        logger.log_summary({"results": results})
        logger.finish("success")
    except Exception as exc:
        finish_with_error(logger, exc)
        raise
