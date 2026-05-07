# src/merge_and_rebase/finetune/train_vision.py
from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml  # type: ignore
from tqdm import tqdm

from merge_and_rebase.cli_args import add_logging_args, build_logging_overrides
from merge_and_rebase.data.templates import get_templates
from merge_and_rebase.io.peft_helpers import state_dict_looks_patched_attn
from merge_and_rebase.run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from merge_and_rebase.utils.helpers import parse_csv

from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.datasets.vision8_14_20 import SUITES, VISION20_TASKS, _vision_spec
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..models.patch_openclip_attention import set_linear_attention_ramp_step, split_openclip_vit_attn
from .regularizers.registry import get_regularizer, list_regularizers
from .strategies.registry import get_strategy, list_strategies
from .text_prestages import (
    _resolve_text_embeddings_finetune_cfg,
    _resolve_text_prompt_tuning_cfg,
    _run_text_embeddings_finetune_stage,
    _run_text_prompt_tuning_stage,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """
    Recursive dict merge: src overwrites dst. Returns dst (mutated).
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _load_config(path: str) -> dict[str, Any]:
    """
    Load a single config file (YAML preferred, JSON supported).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    if p.suffix.lower() in [".yaml", ".yml"]:
        if yaml is None:
            raise RuntimeError("PyYAML not available. Install pyyaml or use a .json config.")
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("YAML config must be a mapping at the top-level.")
        return cfg

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("JSON config must be an object at the top-level.")
        return cfg

    raise ValueError(f"Unsupported config extension: {p.suffix} (use .yaml/.yml or .json)")


def _get_common_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    common = cfg.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("config['common'] must be a dict.")
    return common


def _get_dataset_override(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    ds = cfg.get("datasets", {})
    if ds is None:
        return {}
    if not isinstance(ds, dict):
        raise ValueError("config['datasets'] must be a dict mapping dataset_name -> overrides.")
    ov = ds.get(task, {})
    if ov is None:
        return {}
    if not isinstance(ov, dict):
        raise ValueError(f"config['datasets']['{task}'] must be a dict.")
    return ov


def _resolve_tasks_from_cfg(cfg: dict[str, Any]) -> list[str] | None:
    order = cfg.get("datasets_order", None)
    if order is None:
        return None
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise ValueError("config['datasets_order'] must be a list of strings.")
    return list(order)


def _resolve_attention_patch_cfg(strategy_cfg: dict[str, Any] | None, *, total_steps: int) -> dict[str, Any] | None:
    if not isinstance(strategy_cfg, dict):
        return None
    attention_cfg = strategy_cfg.get("attention", None)
    if attention_cfg is None:
        return None
    if not isinstance(attention_cfg, dict):
        raise ValueError("strategy.attention must be a dict when provided.")

    attn_impl = str(attention_cfg.get("attn_impl", "softmax")).strip().lower()
    if attn_impl not in {"softmax", "linear"}:
        raise ValueError("attention.attn_impl must be one of: softmax, linear")
    ramp_fraction_default = 0.2 if attn_impl == "linear" else 0.0
    ramp_fraction = float(attention_cfg.get("ramp_fraction", ramp_fraction_default))
    if ramp_fraction < 0.0 or ramp_fraction > 1.0:
        raise ValueError("attention.ramp_fraction must be in [0, 1].")

    linear_rule = str(attention_cfg.get("linear_rule", "kernel")).strip().lower()
    if linear_rule not in {"kernel", "delta"}:
        raise ValueError("attention.linear_rule must be one of: kernel, delta")

    ramp_steps = int(round(ramp_fraction * max(1, int(total_steps))))
    return {
        "attn_impl": attn_impl,
        "kernel": str(attention_cfg.get("kernel", "elu_plus_one")),
        "eps": float(attention_cfg.get("eps", 1e-6)),
        "ramp_fraction": ramp_fraction,
        "ramp_steps": ramp_steps,
        "linear_rule": linear_rule,
        "delta_eta": float(attention_cfg.get("delta_eta", 1.0)),
        "delta_exclude_cls_from_store": bool(attention_cfg.get("delta_exclude_cls_from_store", True)),
        "delta_cls_only_readout": bool(attention_cfg.get("delta_cls_only_readout", False)),
        "delta_learn_w0": bool(attention_cfg.get("delta_learn_w0", False)),
        "delta_w0_rank": int(attention_cfg.get("delta_w0_rank", 0)),
    }


def _save_peft_visual_adapter(
    *,
    model: nn.Module,
    task_dir: Path,
    strategy: str,
    suffix: str | None,
    peft_cfg: dict[str, Any] | None,
    patched_attn: bool,
    attn_patch_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Save PEFT adapter using PEFT's native API on model.clip_model.model.visual.

    Returns a dict to be inserted into the checkpoint payload.
    """
    # We expect PEFT to wrap ONLY the visual module:
    visual = model.clip_model.model.visual  # type: ignore[attr-defined]
    if not hasattr(visual, "save_pretrained"):
        raise ValueError(
            "save_format='peft' expects model.clip_model.model.visual to be a PEFT-wrapped module "
            "(must have .save_pretrained())."
        )

    adapter_name = f"{strategy}_adapter" if suffix is None else f"{strategy}_{suffix}_adapter"
    adapter_dir = task_dir / adapter_name
    _ensure_dir(adapter_dir)
    visual.save_pretrained(adapter_dir)

    meta = {
        "format": "peft",
        "peft_target": "visual",
        "peft_adapter_dir": str(adapter_dir),
        "peft_cfg": peft_cfg if peft_cfg is not None else {},
        "patched_attn": bool(patched_attn),
        "attn_patch_cfg": dict(attn_patch_cfg or {}),
    }
    _save_json(adapter_dir / "merge_and_rebase_meta.json", meta)
    return meta


# ---------------------------
# A simple classifier wrapper for finetuning
# ---------------------------


class ImageEncoder(nn.Module):
    """
    Wraps an OpenCLIP image encoder + linear head.
    Forward: images -> logits [B,C]
    """

    def __init__(self, classifier: OpenClipClassifier) -> None:
        super().__init__()
        self.clip_model = classifier
        for param in self.clip_model.model.transformer.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        img_feats = self.clip_model.model.visual(images)
        if self.clip_model.normalize:
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)

        if self.clip_model._zs_text_features.numel() == 0:
            raise RuntimeError("Call build_zeroshot_text_features() before forward in zero-shot mode.")

        return self.clip_model.logit_scale * (img_feats @ self.clip_model._zs_text_features.t())

    @torch.no_grad()
    def top1(self, loader, device: str) -> float:
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.to(dev)
        self.eval()

        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = self(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return float(correct / max(1, total))


# ---------------------------
# Training loop
# ---------------------------


def train_task(
    *,
    task: str,
    hf_path: str,
    hf_config: str | None,
    split_map: dict[str, str],
    build_cfg: OpenClipBuildConfig,
    strategy: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_length: int,
    clip_grad_norm: float,
    accumulate_grad_batches: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    early_stopping: bool,
    early_stopping_patience: int,
    seed: int,
    device: str,
    out_dir: Path,
    save_format: str,  # "full"|"head"|"peft"
    save_last_epoch: bool = False,
    peft_cfg: dict[str, Any] | None = None,
    strategy_cfg: dict[str, Any] | None = None,
    regularization_cfg: dict[str, Any] | None = None,
    log_every_n_steps: int = 50,
    run_logger: Any | None = None,
) -> dict[str, Any]:
    dev = _device(device)
    _set_seed(seed)
    if accumulate_grad_batches <= 0:
        raise ValueError("accumulate_grad_batches must be >= 1.")

    # datasets + loaders
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    clf = OpenClipClassifier.build(build_cfg)

    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=clf.preprocess,
        ft_epochs=1,
        split_map=split_map,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        val_fraction=val_fraction,
        seed=seed,
    )

    num_classes = len(loaders.classnames)

    task_dir = out_dir / build_cfg.model_name / build_cfg.pretrained / task
    _ensure_dir(task_dir)

    if run_logger is not None:
        run_logger.log_event(
            "task_start",
            metrics={},
            context={
                "task": task,
                "strategy": strategy,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "effective_batch_size": int(batch_size * accumulate_grad_batches),
                "task_dir": str(task_dir),
            },
        )

    # model = encoder + head
    model = ImageEncoder(clf).to(dev)
    model.clip_model.build_zeroshot_text_features(list(loaders.classnames), build_cfg)
    zero_shot_val = (
        model.top1(loaders.val, str(dev)) if hasattr(loaders, "val") and loaders.val is not None else float("nan")
    )
    zero_shot_test = model.top1(loaders.test, str(dev))
    print(f"[{task}] zero-shot before finetuning  val={zero_shot_val:.4f}  test={zero_shot_test:.4f}")
    if run_logger is not None:
        run_logger.log_event(
            "zero_shot_eval",
            metrics={
                f"zero_shot/{task}/val_top1": float(zero_shot_val),
                f"zero_shot/{task}/test_top1": float(zero_shot_test),
            },
            context={"task": task},
        )

    text_emb_ft_cfg = _resolve_text_embeddings_finetune_cfg(
        strategy_cfg,
        default_epochs=epochs,
        default_lr=lr,
        default_weight_decay=weight_decay,
        default_warmup_length=warmup_length,
        default_clip_grad_norm=clip_grad_norm,
        default_accumulate_grad_batches=accumulate_grad_batches,
    )
    text_prompt_ft_cfg = _resolve_text_prompt_tuning_cfg(
        strategy_cfg,
        default_epochs=epochs,
        default_lr=lr,
        default_weight_decay=weight_decay,
        default_warmup_length=warmup_length,
        default_clip_grad_norm=clip_grad_norm,
        default_accumulate_grad_batches=accumulate_grad_batches,
    )
    if text_emb_ft_cfg is not None and text_prompt_ft_cfg is not None:
        raise ValueError(
            "strategy.text_embeddings_finetune and strategy.text_prompt_tuning are mutually exclusive. "
            "Enable only one text pre-stage."
        )

    text_emb_ft_summary: dict[str, Any] | None = None
    text_prompt_ft_summary: dict[str, Any] | None = None
    if text_prompt_ft_cfg is not None:
        print(
            f"[{task}] Running text prompt-tuning pre-stage "
            f"(epochs={text_prompt_ft_cfg['epochs']}, ctx_len={text_prompt_ft_cfg['context_length']}, lr={text_prompt_ft_cfg['lr']:.2e})."
        )
        text_prompt_ft_summary = _run_text_prompt_tuning_stage(
            task=task,
            model=model,
            loaders=loaders,
            device=dev,
            cfg=text_prompt_ft_cfg,
        )
        if run_logger is not None:
            run_logger.log_event(
                "text_prestage_end",
                metrics={},
                context={
                    "task": task,
                    "stage": "text_prompt_tuning",
                    "summary": text_prompt_ft_summary,
                },
            )
    elif text_emb_ft_cfg is not None:
        print(
            f"[{task}] Running text-embedding pre-stage "
            f"(epochs={text_emb_ft_cfg['epochs']}, lr={text_emb_ft_cfg['lr']:.2e})."
        )
        text_emb_ft_summary = _run_text_embeddings_finetune_stage(
            task=task,
            model=model,
            loaders=loaders,
            device=dev,
            cfg=text_emb_ft_cfg,
        )
        if run_logger is not None:
            run_logger.log_event(
                "text_prestage_end",
                metrics={},
                context={
                    "task": task,
                    "stage": "text_embeddings_finetune",
                    "summary": text_emb_ft_summary,
                },
            )

    loss_fn = nn.CrossEntropyLoss()
    steps_per_epoch = math.ceil(len(loaders.train) / accumulate_grad_batches)
    total_steps = epochs * steps_per_epoch

    # For non-PEFT strategies, optional strategy.attention patching is applied here.
    # PEFT handles its own attention patching inside PeftLoraVision.configure().
    if strategy != "peft_lora":
        attn_patch_cfg = _resolve_attention_patch_cfg(strategy_cfg, total_steps=total_steps)
        if attn_patch_cfg is not None:
            patched = split_openclip_vit_attn(
                model.clip_model.model.visual,
                proj_dropout=0.0,
                attn_impl=str(attn_patch_cfg.get("attn_impl", "softmax")),
                kernel=str(attn_patch_cfg.get("kernel", "elu_plus_one")),
                eps=float(attn_patch_cfg.get("eps", 1e-6)),
                ramp_steps=int(attn_patch_cfg.get("ramp_steps", 0)),
                linear_rule=str(attn_patch_cfg.get("linear_rule", "kernel")),
                delta_eta=float(attn_patch_cfg.get("delta_eta", 1.0)),
                delta_exclude_cls_from_store=bool(attn_patch_cfg.get("delta_exclude_cls_from_store", True)),
                delta_cls_only_readout=bool(attn_patch_cfg.get("delta_cls_only_readout", False)),
                delta_learn_w0=bool(attn_patch_cfg.get("delta_learn_w0", False)),
                delta_w0_rank=int(attn_patch_cfg.get("delta_w0_rank", 0)),
            )
            if patched == 0:
                raise RuntimeError("Requested strategy.attention patching but patched 0 blocks.")
            model.peft_patched_attn = True  # type: ignore[attr-defined]
            model.peft_attn_patch_cfg = dict(attn_patch_cfg)  # type: ignore[attr-defined]
            print(f"[{task}] Patched {patched} attention blocks (attn_impl={attn_patch_cfg['attn_impl']}).")

    regularizer_cfg = dict(regularization_cfg or {})
    regularizer_name = str(regularizer_cfg.get("name", "")).strip()
    regularizer_impl = None
    regularizer_info: dict[str, int] = {}
    regularizer_fns: list[callable] = []
    if regularizer_name:
        regularizer_impl = get_regularizer(regularizer_name)
        regularizer_impl.prepare_model(
            model=model,
            device=dev,
            regularization_cfg=regularizer_cfg,
            task=task,
            strategy_cfg=strategy_cfg,
        )

    strategy_impl = get_strategy(strategy)
    configured = strategy_impl.configure(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        warmup_length=warmup_length,
        steps=total_steps,
        device=dev,
        peft_cfg=peft_cfg,
        strategy_cfg=strategy_cfg,
        task=task,
    )
    if len(configured) != 3:
        raise ValueError("Strategy.configure() must return (opt, scheduler, info).")
    opt, scheduler, trainable_info = configured

    if regularizer_impl is not None:
        regularizer_fn, regularizer_info = regularizer_impl.configure(
            model=model,
            device=dev,
            regularization_cfg=regularizer_cfg,
            task=task,
            strategy_cfg=strategy_cfg,
            strategy=strategy,
        )
        regularizer_fns.append(regularizer_fn)

    best_val = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    last_epoch = 0
    last_val = float("nan")
    last_test = float("nan")
    early_stopping_patience_current = early_stopping_patience
    model.to(dev)

    t_start = time.time()
    global_update_step = 0

    ckpt_stem = strategy if not regularizer_name else f"{strategy}__{regularizer_name}"

    def _build_checkpoint_payload(
        *,
        epoch_i: int,
        val_acc_i: float,
        test_acc_i: float,
        kind: str,  # "best_ep" | "last_ep"
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": task,
            "strategy": strategy,
            "backbone": {
                "kind": "openclip",
                "model_name": build_cfg.model_name,
                "pretrained": build_cfg.pretrained,
                "dtype": build_cfg.dtype,
            },
            "num_classes": num_classes,
            "classnames": list(loaders.classnames),
            "metrics": {"val_top1": float(val_acc_i), "test_top1": float(test_acc_i)},
            "zero_shot_metrics": {
                "val_top1": float(zero_shot_val),
                "test_top1": float(zero_shot_test),
            },
        }
        if text_emb_ft_summary is not None:
            payload["text_embeddings_finetune"] = dict(text_emb_ft_summary)
        if text_prompt_ft_summary is not None:
            payload["text_prompt_tuning"] = dict(text_prompt_ft_summary)
        if text_emb_ft_summary is not None or text_prompt_ft_summary is not None:
            payload["tuned_text_features"] = model.clip_model._zs_text_features.detach().cpu()
        tuned_prompt_context = getattr(model.clip_model, "_tuned_prompt_context", None)
        if text_prompt_ft_summary is not None and isinstance(tuned_prompt_context, torch.Tensor):
            payload["tuned_prompt_context"] = tuned_prompt_context.detach().cpu()
        if kind == "best_ep":
            payload["best_epoch"] = int(epoch_i)
        elif kind == "last_ep":
            payload["last_epoch"] = int(epoch_i)
            payload["best_epoch"] = int(best_epoch)
        else:
            raise ValueError("kind must be 'best_ep' or 'last_ep'")

        model_sd = model.state_dict()
        patched_attn = bool(getattr(model, "peft_patched_attn", False)) or state_dict_looks_patched_attn(model_sd)
        attn_patch_cfg_raw = getattr(model, "peft_attn_patch_cfg", None)
        attn_patch_cfg = dict(attn_patch_cfg_raw) if isinstance(attn_patch_cfg_raw, dict) else None
        if patched_attn and attn_patch_cfg is None:
            # Fallback for non-PEFT paths that patched q/k/v attention without explicit cfg metadata.
            attn_patch_cfg = {
                "attn_impl": "softmax",
                "kernel": "elu_plus_one",
                "eps": 1e-6,
                "linear_rule": "kernel",
                "delta_eta": 1.0,
                "delta_exclude_cls_from_store": True,
                "delta_cls_only_readout": False,
                "delta_learn_w0": False,
                "delta_w0_rank": 0,
            }
        payload["patched_attn"] = patched_attn
        if attn_patch_cfg is not None:
            payload["attn_patch_cfg"] = attn_patch_cfg

        if save_format == "full":
            payload["state_dict"] = {k: v.detach().cpu() for k, v in model_sd.items()}
            payload["format"] = "full"
        elif save_format == "head":
            payload["head"] = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
            payload["format"] = "head"
        elif save_format == "peft":
            payload.update(
                _save_peft_visual_adapter(
                    model=model,
                    task_dir=task_dir,
                    strategy=ckpt_stem,
                    suffix=kind,
                    peft_cfg=peft_cfg,
                    patched_attn=patched_attn,
                    attn_patch_cfg=attn_patch_cfg,
                )
            )
        else:
            raise ValueError("save_format must be 'full', 'head', or 'peft'")
        return payload

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        window_batch_count = 0
        window_size = 1

        with tqdm(total=len(loaders.train), desc=f"[{task}] Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for i, (x, y) in enumerate(loaders.train):
                if window_batch_count == 0:
                    remaining = len(loaders.train) - i
                    window_size = min(accumulate_grad_batches, remaining)
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)

                # Blend softmax -> linear attention during warmup ramp (if enabled).
                set_linear_attention_ramp_step(model, step=global_update_step)
                logits = model(x)
                raw_loss = loss_fn(logits, y)
                reg_loss = raw_loss.new_zeros(())
                for reg_fn in regularizer_fns:
                    reg_loss = reg_loss + reg_fn(model=model, step=global_update_step, batch_index=i)
                total_loss = raw_loss + reg_loss
                loss = total_loss / window_size

                loss.backward()

                window_batch_count += 1
                should_step = window_batch_count == window_size
                if should_step:
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    scheduler(global_update_step)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    global_update_step += 1
                    window_batch_count = 0

                bs = int(y.numel())
                running_loss += float(total_loss.item()) * bs
                n_seen += bs

                train_loss = running_loss / max(1, n_seen)
                if (
                    run_logger is not None
                    and should_step
                    and log_every_n_steps > 0
                    and global_update_step > 0
                    and global_update_step % log_every_n_steps == 0
                ):
                    run_logger.log_event(
                        "train_step",
                        metrics={
                            f"train/{task}/loss": float(train_loss),
                            f"train/{task}/lr": float(opt.param_groups[0]["lr"]),
                            f"train/{task}/reg_loss": float(reg_loss.item()) if regularizer_fns else 0.0,
                        },
                        step=int(global_update_step),
                        context={
                            "task": task,
                            "epoch": int(epoch),
                        },
                    )
                pbar.update(1)
                postfix = {"loss": f"{train_loss:.4f}", "lr": f"{opt.param_groups[0]['lr']:.6f}"}
                if regularizer_fns:
                    postfix["reg"] = f"{float(reg_loss.item()):.2e}"
                pbar.set_postfix(postfix)

        # val/test
        set_linear_attention_ramp_step(model, step=global_update_step)
        val_acc = (
            model.top1(loaders.val, str(dev)) if hasattr(loaders, "val") and loaders.val is not None else float("nan")
        )
        test_acc = model.top1(loaders.test, str(dev))

        last_epoch = epoch
        last_val = float(val_acc)
        last_test = float(test_acc)

        if not math.isnan(val_acc) and val_acc > best_val:
            early_stopping_patience_current = early_stopping_patience
            best_epoch = epoch
            best_val = val_acc
            best_state = _build_checkpoint_payload(
                epoch_i=best_epoch,
                val_acc_i=float(val_acc),
                test_acc_i=float(test_acc),
                kind="best_ep",
            )
            torch.save(best_state, task_dir / f"{ckpt_stem}_best_ep.pt")
        else:
            early_stopping_patience_current -= 1
            if early_stopping_patience_current <= 0 and early_stopping:
                print(f"[{task}] Early stopping triggered. No improvement in validation for several epochs.")
                break

        print(
            f"[{task}] epoch {epoch:03d}/{epochs}  loss={train_loss:.4f}  val={val_acc:.4f}  test={test_acc:.4f} patience={early_stopping_patience_current}/{early_stopping_patience}"
        )
        if run_logger is not None:
            run_logger.log_event(
                "epoch_end",
                metrics={
                    f"train/{task}/loss": float(train_loss),
                    f"train/{task}/lr": float(opt.param_groups[0]["lr"]),
                    f"val/{task}/top1": float(val_acc),
                    f"test/{task}/top1": float(test_acc),
                    f"train/{task}/seconds": float(time.time() - t_start),
                },
                step=int(epoch),
                context={
                    "task": task,
                    "epoch": int(epoch),
                    "patience_left": int(early_stopping_patience_current),
                },
            )

    seconds = time.time() - t_start

    if best_state is None:
        fallback_best_epoch = best_epoch if best_epoch > 0 else last_epoch
        fallback_test = last_test if not math.isnan(last_test) else float(model.top1(loaders.test, str(dev)))
        best_state = _build_checkpoint_payload(
            epoch_i=fallback_best_epoch,
            val_acc_i=last_val,
            test_acc_i=fallback_test,
            kind="best_ep",
        )
        best_state["regularization"] = {"name": regularizer_name, "info": regularizer_info}

    best_ckpt_path = task_dir / f"{ckpt_stem}_best_ep.pt"
    torch.save(best_state, best_ckpt_path)

    last_ckpt_path: Path | None = None
    if save_last_epoch:
        if last_epoch <= 0:
            last_epoch = epochs
        last_state = _build_checkpoint_payload(
            epoch_i=last_epoch,
            val_acc_i=last_val,
            test_acc_i=last_test,
            kind="last_ep",
        )
        last_state["regularization"] = {"name": regularizer_name, "info": regularizer_info}
        last_ckpt_path = task_dir / f"{ckpt_stem}_last_ep.pt"
        torch.save(last_state, last_ckpt_path)

    summary = {
        "task": task,
        "strategy": strategy,
        "save_format": save_format,
        "save_last_epoch": bool(save_last_epoch),
        "ckpt_path": str(best_ckpt_path),
        "best_ckpt_path": str(best_ckpt_path),
        "last_ckpt_path": str(last_ckpt_path) if last_ckpt_path is not None else None,
        "metrics": best_state.get("metrics", {}),
        "zero_shot_metrics": best_state.get("zero_shot_metrics", {}),
        "seconds": float(seconds),
        "trainable": trainable_info,
        "text_embeddings_finetune": text_emb_ft_summary,
        "text_prompt_tuning": text_prompt_ft_summary,
        "regularization": {"name": regularizer_name, "info": regularizer_info},
        "best_epoch": best_state.get("best_epoch", -1),
        "last_epoch": int(last_epoch),
        "hparams": {
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            # "scheduler": strategy_impl.__name__,
            "warmup_length": int(warmup_length),
            "clip_grad_norm": float(clip_grad_norm),
            "accumulate_grad_batches": int(accumulate_grad_batches),
            "batch_size": int(batch_size),
            "effective_batch_size": int(batch_size * accumulate_grad_batches),
            "num_workers": int(num_workers),
            "val_fraction": float(val_fraction),
            "seed": int(seed),
        },
    }
    _save_json(task_dir / f"{strategy}.json", summary)

    print(f"[{task}] saved best: {best_ckpt_path}")
    if last_ckpt_path is not None:
        print(f"[{task}] saved last: {last_ckpt_path}")
    if run_logger is not None:
        run_logger.log_event(
            "task_end",
            metrics={
                f"val/{task}/top1": float(summary["metrics"].get("val_top1", float("nan"))),
                f"test/{task}/top1": float(summary["metrics"].get("test_top1", float("nan"))),
                f"train/{task}/seconds": float(summary["seconds"]),
            },
            context={
                "task": task,
                "summary": summary,
            },
        )
    return summary


# ---------------------------
# Main
# ---------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fine-tune from a vision config file (YAML/JSON).")

    g = p.add_argument_group("Config")
    g.add_argument("--vision-config", type=str, required=True, help="Path to vision config (.yaml/.yml/.json).")

    g = p.add_argument_group("Task selection overrides (optional)")
    g.add_argument("--suite", type=str, default=None, choices=sorted(SUITES.keys()))
    g.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names (overrides suite/order).")

    g = p.add_argument_group("Runtime overrides (optional)")
    g.add_argument("--device", type=str, default=None, help="Override config device, e.g. cuda, cuda:0, cpu, mps.")
    add_logging_args(p)

    return p


def resolve_tasks(args, cfg_file: dict[str, Any]) -> list[str]:
    if args.datasets and args.datasets.strip():
        return parse_csv(args.datasets)
    if args.suite is not None:
        return list(SUITES[args.suite].tasks)

    tasks = _resolve_tasks_from_cfg(cfg_file)
    return tasks if tasks is not None else list(SUITES["vision8"].tasks)


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Tiny helper to read nested dicts with dot paths."""
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def main() -> None:
    run_logger = None
    try:
        parser = build_parser()
        args = parser.parse_args()

        cfg_file = _load_config(args.vision_config)
        common = _get_common_cfg(cfg_file)

        # Resolve tasks (datasets/suite override config order)
        tasks = resolve_tasks(args, cfg_file)

        # Global cfg is common with selected CLI overrides.
        global_cfg = deepcopy(common)

        backbone_name = _get(global_cfg, "backbone.name", "openclip")
        if backbone_name != "openclip":
            raise ValueError(f"Unsupported backbone '{backbone_name}' (only openclip for now).")

        clip_model = _get(global_cfg, "backbone.clip_model", "ViT-B-32")
        clip_pretrained = _get(global_cfg, "backbone.clip_pretrained", "openai")
        device = str(args.device) if args.device is not None else _get(global_cfg, "device", "cuda")
        dtype = _get(global_cfg, "dtype", None)

        out_dir = Path(_get(global_cfg, "output.out_dir", "src/checkpoints/finetune"))
        save_format_default = str(_get(global_cfg, "output.save_format", "full"))
        save_last_epoch_default = bool(_get(global_cfg, "output.save_last_epoch", False))
        logging_cfg = merge_logging_config(_get(global_cfg, "logging", {}), build_logging_overrides(args))
        run_ts = int(time.time())
        run_path = default_summary_path(
            entrypoint="finetune.train_vision",
            logging_cfg=logging_cfg,
            default_parent=out_dir / str(clip_model) / str(clip_pretrained),
            timestamp=run_ts,
        )

        all_summaries: dict[str, Any] = {
            "config_path": args.vision_config,
            "common": common,
            "cli": {
                "suite": args.suite,
                "datasets": args.datasets,
                "device": args.device,
                "logging": build_logging_overrides(args),
            },
            "resolved": {
                "tasks": tasks,
                "build_cfg": {
                    "backbone": backbone_name,
                    "clip_model": clip_model,
                    "clip_pretrained": clip_pretrained,
                    "dtype": dtype,
                    "device": device,
                },
                "run_path": str(run_path),
            },
            "results": {},
        }
        run_logger = start_run(
            entrypoint="finetune.train_vision",
            logging_cfg=logging_cfg,
            summary_path=run_path,
            metadata={
                "config_path": args.vision_config,
                "cli": all_summaries["cli"],
                "resolved": all_summaries["resolved"],
                "logging": logging_cfg,
            },
        )

        for task in tasks:
            if task not in VISION20_TASKS:
                raise ValueError(f"Unknown task '{task}'. Supported: {VISION20_TASKS}")

            # task_cfg = common -> per-dataset override
            task_cfg = deepcopy(common)
            _deep_update(task_cfg, _get_dataset_override(cfg_file, task))
            task_logging_cfg = merge_logging_config(_get(task_cfg, "logging", {}), build_logging_overrides(args))

            epochs = _get(task_cfg, "train.epochs", None)
            if epochs is None:
                raise ValueError(f"[{task}] train.epochs missing. Set common.train.epochs or datasets.{task}.train.epochs.")
            epochs = int(epochs)

            strategy = str(_get(task_cfg, "strategy.name", "full"))
            if strategy not in list_strategies():
                raise ValueError(f"[{task}] Unknown strategy '{strategy}'. Available: {list_strategies()}")
            strategy_cfg = _get(task_cfg, "strategy", {})
            if not isinstance(strategy_cfg, dict):
                raise ValueError(f"[{task}] strategy must be a dict.")
            regularization_cfg = _get(task_cfg, "regularization", {})
            if regularization_cfg is None:
                regularization_cfg = {}
            if not isinstance(regularization_cfg, dict):
                raise ValueError(f"[{task}] regularization must be a dict when provided.")
            regularization_name = str(regularization_cfg.get("name", "")).strip()
            if regularization_name and regularization_name not in list_regularizers():
                raise ValueError(f"[{task}] Unknown regularizer '{regularization_name}'. Available: {list_regularizers()}")

            lr = float(_get(task_cfg, "train.lr", 1e-4))
            weight_decay = float(_get(task_cfg, "train.weight_decay", 0.0))
            clip_grad_norm = float(_get(task_cfg, "train.grad_clip_norm", 1.0))
            accumulate_grad_batches = int(_get(task_cfg, "train.accumulate_grad_batches", 1))
            if accumulate_grad_batches <= 0:
                raise ValueError(f"[{task}] train.accumulate_grad_batches must be >= 1.")

            batch_size = int(_get(task_cfg, "data.batch_size", 64))
            num_workers = int(_get(task_cfg, "data.num_workers", 6))
            val_fraction = float(_get(task_cfg, "data.val_fraction", 0.1))
            seed = int(_get(task_cfg, "seed", 42))
            early_stopping = bool(_get(task_cfg, "train.early_stopping", False))
            early_stopping_patience = int(_get(task_cfg, "train.early_stopping_patience", 5))

            task_out_dir = Path(_get(task_cfg, "output.out_dir", str(out_dir)))
            save_format = str(_get(task_cfg, "output.save_format", save_format_default))
            save_last_epoch = bool(_get(task_cfg, "output.save_last_epoch", save_last_epoch_default))

            hf_path, hf_config, split_map = _vision_spec(task)

            build_cfg = OpenClipBuildConfig(
                model_name=str(clip_model),
                pretrained=str(clip_pretrained),
                device=str(device),
                dtype=dtype,
                prompt_templates=get_templates(task),
            )

            summary = train_task(
                task=task,
                hf_path=hf_path,
                hf_config=hf_config,
                split_map=split_map,
                build_cfg=build_cfg,
                strategy=strategy,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                warmup_length=int(_get(task_cfg, "train.lr_scheduler.warmup_steps", 500)),
                clip_grad_norm=clip_grad_norm,
                accumulate_grad_batches=accumulate_grad_batches,
                batch_size=batch_size,
                num_workers=num_workers,
                val_fraction=val_fraction,
                seed=seed,
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
                device=str(device),
                out_dir=task_out_dir,
                save_format=save_format,
                save_last_epoch=save_last_epoch,
                peft_cfg=strategy_cfg.get("peft") if strategy_cfg else None,
                strategy_cfg=strategy_cfg,
                regularization_cfg=regularization_cfg,
                log_every_n_steps=int(task_logging_cfg.get("log_every_n_steps", 50)),
                run_logger=run_logger,
            )
            all_summaries["results"][task] = summary

        _save_json(run_path, all_summaries)
        run_logger.log_summary(all_summaries)
        run_logger.finish("success")
        print(f"\nSaved run summary: {run_path}")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
