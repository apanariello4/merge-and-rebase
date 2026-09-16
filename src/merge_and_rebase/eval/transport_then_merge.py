from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from merge_and_rebase.cli_args import add_logging_args, build_logging_overrides
from merge_and_rebase.data.templates import get_templates
from merge_and_rebase.data.vision_loaders import (
    build_vision_calibration_loader,
    build_vision_loaders,
    load_hf_splits,
)
from merge_and_rebase.eval.block_extension import (
    calibration_dataset_spec,
    resolve_block_extension_config,
    run_block_extension,
)
from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.utils import to_cpu_fp32
from merge_and_rebase.io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from merge_and_rebase.merge.methods._common import axpy_state_dict
from merge_and_rebase.merge.registry import get_method as get_merge_method
from merge_and_rebase.merge.task_vectors import TaskVector
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from merge_and_rebase.rebase.methods.theseus import InterpolatedBlockActivations
from merge_and_rebase.rebase.registry import get_method as get_rebase_method
from merge_and_rebase.run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from merge_and_rebase.utils.helpers import load_json


def _visual_only_filter(k: str, v: torch.Tensor) -> bool:
    if not v.is_floating_point():
        return False
    if ".aligner." in k:
        return False
    return k.startswith("visual.")


def _build_cfg(model_name: str, pretrained: str, device: str, dtype: str | None, task: str) -> OpenClipBuildConfig:
    templates = get_templates(task)
    if not templates:
        raise ValueError(f"get_templates('{task}') returned empty list")
    return OpenClipBuildConfig(
        model_name=model_name, pretrained=pretrained, device=device, dtype=dtype, prompt_templates=templates,
    )


def _load_task_delta(base_sd: dict[str, torch.Tensor], tuned_path: str) -> dict[str, torch.Tensor]:
    tuned_sd = load_ckpt(tuned_path)
    aligned = align_to_base_keys(tuned_sd, base_sd)
    if not aligned:
        raise ValueError(f"No tensors aligned from {tuned_path}")
    tuned_cpu = to_cpu_fp32(aligned)
    tv = TaskVector.from_checkpoints(base_sd, tuned_cpu, strict=False, key_filter=_visual_only_filter)
    return tv.delta


@dataclass
class TransportContext:
    source_model_template: torch.nn.Module
    target_model_template: torch.nn.Module
    clf_source: OpenClipClassifier
    clf_target: OpenClipClassifier
    source_base_sd: dict[str, torch.Tensor]
    target_base_sd: dict[str, torch.Tensor]
    target_depth: int


def _extend_and_prepare(
    ctx: TransportContext,
    *,
    task: str,
    source_task_cfg: OpenClipBuildConfig,
    target_task_cfg: OpenClipBuildConfig,
    tuned_path: str,
    method_name: str,
    method_params: dict[str, Any],
    block_ext_cfg: Any,
    block_extension_calibration_loader: Any | None,
    source_loaders: Any,
    target_loaders: Any,
    source_classnames: list[str],
    target_classnames: list[str],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], Any, torch.nn.Module, torch.nn.Module]:
    """Extend base+ft, compute task delta, and prepare transport maps.

    Returns:
        (task_delta, task_source_base_sd, prepared, source_base_model, source_ft_model)
        Models are returned (not deleted) so the caller can keep them alive
        for the transport.prepare() activations if needed.
    """
    print(f"  [transport] {task}: loading tuned checkpoint")
    tuned_sd = load_ckpt(tuned_path)
    aligned = align_to_base_keys(tuned_sd, ctx.source_base_sd)
    tuned_cpu = to_cpu_fp32(aligned)

    source_base_model = deepcopy(ctx.source_model_template)
    source_ft_model = deepcopy(ctx.source_model_template)
    load_into_model(source_base_model, ctx.source_base_sd, strict=False)
    load_into_model(source_ft_model, ctx.source_base_sd, strict=False)
    load_into_model(source_ft_model, tuned_cpu, strict=False)

    print(f"  [transport] {task}: extending source depth {len(source_base_model.visual.transformer.resblocks)} -> {ctx.target_depth}")

    extension_layout: dict[str, Any] = {}
    final_depth = run_block_extension(
        source_base_model=source_base_model,
        source_ft_model=source_ft_model,
        calibration_loader=(
            block_extension_calibration_loader
            if block_extension_calibration_loader is not None
            else source_loaders.train
        ),
        target_layers_total=ctx.target_depth,
        config=block_ext_cfg,
        device=device,
        layout_out=extension_layout,
    )
    source_activation_plan = (
        None
        if str(getattr(block_ext_cfg, "transport_activation_mode", "model")) == "model"
        else InterpolatedBlockActivations.from_extension_layout(extension_layout)
    )
    print(f"  [transport] {task}: block extension done, final_depth={final_depth}")

    task_source_base_sd = to_cpu_fp32({k: v for k, v in source_base_model.state_dict().items()})
    task_source_ft_sd = to_cpu_fp32({k: v for k, v in source_ft_model.state_dict().items()})
    tv = TaskVector.from_checkpoints(task_source_base_sd, task_source_ft_sd, strict=False, key_filter=_visual_only_filter)
    task_delta = tv.delta

    method = get_rebase_method(method_name)
    params = dict(method_params)

    if method_name in ("theseus", "theseus_reference"):
        source_model = source_ft_model
        target_model = deepcopy(ctx.target_model_template)
        load_into_model(target_model, ctx.target_base_sd, strict=False)
        prepared = method.prepare(
            source_model=source_model, target_model=target_model,
            source_dataloader=source_loaders.train, target_dataloader=target_loaders.train,
            target_base=ctx.target_base_sd, delta=task_delta, device=device,
            source_activation_plan=source_activation_plan, **params,
        )
    elif method_name == "bico":
        from merge_and_rebase.models.grad_recipes import clip_contrastive_recipe

        source_model = source_base_model
        target_model = deepcopy(ctx.target_model_template)
        load_into_model(target_model, ctx.target_base_sd, strict=False)
        source_recipe = clip_contrastive_recipe(ctx.clf_source, source_classnames, source_task_cfg, device=device)
        target_recipe = clip_contrastive_recipe(ctx.clf_target, target_classnames, target_task_cfg, device=device)
        prepared = method.prepare(
            source_model=source_model, target_model=target_model,
            source_dataloader=source_loaders.train, target_dataloader=target_loaders.train,
            source_recipe=source_recipe, target_recipe=target_recipe,
            target_base=ctx.target_base_sd, delta=task_delta, device=device,
            source_activation_plan=source_activation_plan, **params,
        )
    else:
        if source_activation_plan is not None:
            raise ValueError(
                "The interpolated-activation baseline only applies to activation-aligned "
                f"transport; method '{method_name}' does not consume activations."
            )
        prepared = None

    return task_delta, task_source_base_sd, prepared, source_base_model, source_ft_model


def _transport_delta(
    ctx: TransportContext,
    *,
    task: str,
    task_source_base_sd: dict[str, torch.Tensor],
    task_delta: dict[str, torch.Tensor],
    method_name: str,
    method_params: dict[str, Any],
    prepared: Any,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    device: str,
    strict_load: bool,
) -> dict[str, torch.Tensor]:
    """Transport a (possibly corrected) delta using precomputed prepared maps."""
    method = get_rebase_method(method_name)
    params = dict(method_params)

    transported_delta = method.transport(
        source_base=task_source_base_sd, target_base=ctx.target_base_sd,
        delta=task_delta, strict=strict_load, prepared=prepared, **params,
    )

    del source_base_model, source_ft_model
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.empty_cache()

    return transported_delta


def _transport_source_task(
    ctx: TransportContext,
    *,
    task: str,
    source_task_cfg: OpenClipBuildConfig,
    target_task_cfg: OpenClipBuildConfig,
    tuned_path: str,
    method_name: str,
    method_params: dict[str, Any],
    block_ext_cfg: Any,
    block_extension_calibration_loader: Any | None,
    source_loaders: Any,
    target_loaders: Any,
    source_classnames: list[str],
    target_classnames: list[str],
    device: str,
    strict_load: bool,
) -> dict[str, torch.Tensor]:
    """Legacy single-call wrapper: extend+prepare+transport in one shot."""
    task_delta, task_source_base_sd, prepared, source_base_model, source_ft_model = _extend_and_prepare(
        ctx,
        task=task,
        source_task_cfg=source_task_cfg,
        target_task_cfg=target_task_cfg,
        tuned_path=tuned_path,
        method_name=method_name,
        method_params=method_params,
        block_ext_cfg=block_ext_cfg,
        block_extension_calibration_loader=block_extension_calibration_loader,
        source_loaders=source_loaders,
        target_loaders=target_loaders,
        source_classnames=source_classnames,
        target_classnames=target_classnames,
        device=device,
    )
    return _transport_delta(
        ctx,
        task=task,
        task_source_base_sd=task_source_base_sd,
        task_delta=task_delta,
        method_name=method_name,
        method_params=method_params,
        prepared=prepared,
        source_base_model=source_base_model,
        source_ft_model=source_ft_model,
        device=device,
        strict_load=strict_load,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser("Transport source task vectors to target, then merge with native target tasks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-transported-dir", type=str, default=None)
    parser.add_argument("--merge-method", type=str, default=None, help="Merge method (task_arithmetic, ties_merge, tsv_merge, isoc_merge, ...). Overrides config.")
    parser.add_argument("--merge-params", type=str, default=None, help="JSON dict of merge method params. Overrides config.")
    add_logging_args(parser)
    args = parser.parse_args()

    run_logger = None
    try:
        cfg = load_json(args.config)
        device = str(args.device or cfg.get("device", "cuda"))

        logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
        cfg["logging"] = logging_cfg
        run_logger = start_run(
            entrypoint="scripts.transport_then_merge",
            logging_cfg=logging_cfg,
            summary_path=default_summary_path(entrypoint="scripts.transport_then_merge", logging_cfg=logging_cfg),
            metadata={
                "config_path": args.config,
                "resolved_config": cfg,
                "suite": cfg.get("suite", "vision8"),
                "tasks_to_transport": cfg.get("tasks_to_transport", []),
                "native_target_tasks": cfg.get("native_target_tasks", []),
            },
        )

        suite = SUITES[cfg.get("suite", "vision8")]
        tasks_to_transport = cfg.get("tasks_to_transport", [])
        native_target_tasks = cfg.get("native_target_tasks", [])
        all_tasks = list(tasks_to_transport) + list(native_target_tasks)

        source_cfg = OpenClipBuildConfig(
            model_name=cfg["source_clip_model"], pretrained=cfg["source_clip_pretrained"],
            device=device, dtype=cfg.get("dtype", None),
        )
        target_cfg = OpenClipBuildConfig(
            model_name=cfg["target_clip_model"], pretrained=cfg["target_clip_pretrained"],
            device=device, dtype=cfg.get("dtype", None),
        )

        print(f"Source: {source_cfg.model_name} / {source_cfg.pretrained}")
        print(f"Target: {target_cfg.model_name} / {target_cfg.pretrained}")
        print(f"Tasks to transport: {tasks_to_transport}")
        print(f"Native target tasks: {native_target_tasks}")

        clf_source = OpenClipClassifier.build(source_cfg)
        clf_target = OpenClipClassifier.build(target_cfg)

        ctx = TransportContext(
            source_model_template=deepcopy(clf_source.model),
            target_model_template=deepcopy(clf_target.model),
            clf_source=clf_source,
            clf_target=clf_target,
            source_base_sd=to_cpu_fp32({k: v for k, v in clf_source.model.state_dict().items()}),
            target_base_sd=to_cpu_fp32({k: v for k, v in clf_target.model.state_dict().items()}),
            target_depth=len(clf_target.model.visual.transformer.resblocks),
        )

        _, block_ext_cfg = resolve_block_extension_config(cfg)
        method_name = cfg["transport_method"]
        method_params = cfg.get("transport_params", {})
        tuned_ckpts = cfg["tuned_ckpts"]
        merge_method = args.merge_method or cfg.get("merge_method", "task_arithmetic")
        merge_params = json.loads(args.merge_params) if args.merge_params else cfg.get("merge_params", {})
        alpha_search = cfg.get("alpha_search", True)
        alpha_min = float(cfg.get("alpha_min", 0.0))
        alpha_max = float(cfg.get("alpha_max", 2.0))
        alpha_step = float(cfg.get("alpha_step", 0.1))
        alpha_search_split = cfg.get("alpha_search_split", "val")
        batch_size = int(cfg.get("batch_size", 32))
        num_workers = int(cfg.get("num_workers", 0))
        val_fraction = float(cfg.get("val_fraction", 0.1))
        seed = int(cfg.get("seed", 42))
        strict_load = bool(cfg.get("strict_load", False))
        base_construction = str(cfg.get("base_construction", "per_task"))
        alpha_mode = str(cfg.get("alpha_mode", "shared"))

        block_extension_calibration_loader = None
        calibration_dataset = calibration_dataset_spec(block_ext_cfg)
        if calibration_dataset is not None:
            block_extension_calibration_loader = build_vision_calibration_loader(
                calibration_dataset,
                resolver=suite.resolver,
                preprocess=clf_source.preprocess,
                calibration_split=str(block_ext_cfg.calibration_split),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                val_fraction=val_fraction,
                seed=seed,
            )
            print(
                "Using one task-independent block-extension calibration loader "
                f"from {calibration_dataset!r}."
            )

        save_dir = args.save_transported_dir or cfg.get("save_transported_dir", "results/transported_tvs")
        os.makedirs(save_dir, exist_ok=True)

        all_deltas: list[dict[str, torch.Tensor]] = []

        if base_construction == "averaged" and len(tasks_to_transport) > 0:
            print(f"\n=== Base construction: AVERAGED CONSENSUS ({len(tasks_to_transport)} tasks) ===")

            # Phase 1: Extend + prepare for all tasks (keep models alive for prepare)
            per_task_data: list[dict[str, Any]] = []
            for task in tasks_to_transport:
                print(f"\n=== Extending + preparing {task} ===")
                hf_path, hf_config, split_map = suite.resolver(task)
                hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

                source_loaders = build_vision_loaders(
                    hf_ds=hf_ds, hf_path=hf_path, preprocess=clf_source.preprocess, ft_epochs=1,
                    split_map=split_map, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, val_fraction=val_fraction, seed=seed,
                )
                target_loaders = build_vision_loaders(
                    hf_ds=hf_ds, hf_path=hf_path, preprocess=clf_target.preprocess, ft_epochs=1,
                    split_map=split_map, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, val_fraction=val_fraction, seed=seed,
                )

                task_delta, task_source_base_sd, prepared, source_base_model, source_ft_model = _extend_and_prepare(
                    ctx,
                    task=task,
                    source_task_cfg=_build_cfg(source_cfg.model_name, source_cfg.pretrained, device, source_cfg.dtype, task),
                    target_task_cfg=_build_cfg(target_cfg.model_name, target_cfg.pretrained, device, target_cfg.dtype, task),
                    tuned_path=tuned_ckpts[task],
                    method_name=method_name,
                    method_params=method_params,
                    block_ext_cfg=block_ext_cfg,
                    block_extension_calibration_loader=block_extension_calibration_loader,
                    source_loaders=source_loaders,
                    target_loaders=target_loaders,
                    source_classnames=list(source_loaders.classnames),
                    target_classnames=list(target_loaders.classnames),
                    device=device,
                )

                per_task_data.append({
                    "task": task,
                    "task_delta": task_delta,
                    "task_source_base_sd": task_source_base_sd,
                    "prepared": prepared,
                    "source_base_model": source_base_model,
                    "source_ft_model": source_ft_model,
                })

            # Phase 2: Compute averaged base (visual-only filtered)
            print("\n=== Computing averaged consensus base ===")
            avg_base_sd: dict[str, torch.Tensor] = {}
            visual_keys = set(per_task_data[0]["task_source_base_sd"].keys())
            for d in per_task_data:
                visual_keys &= {k for k in d["task_source_base_sd"] if _visual_only_filter(k, d["task_source_base_sd"][k])}
            for k in visual_keys:
                stacked = torch.stack([d["task_source_base_sd"][k].float() for d in per_task_data])
                avg_base_sd[k] = (stacked.mean(dim=0)).to(dtype=per_task_data[0]["task_source_base_sd"][k].dtype)
            print(f"  Averaged {len(avg_base_sd)} visual keys over {len(per_task_data)} tasks")

            # Phase 2b: Diagnostic — evaluate avg_base + delta_t on source side
            print("\n=== Diagnostic: avg_base + delta_t accuracy (source side, extended B16) ===")
            diag_model = per_task_data[-1]["source_base_model"]
            diag_clf = OpenClipClassifier(
                model=deepcopy(diag_model),
                tokenizer=clf_source.tokenizer, preprocess=clf_source.preprocess,
                normalize=clf_source.normalize, logit_scale=clf_source.logit_scale,
            )
            diag_results: dict[str, float] = {}
            for td in per_task_data:
                task = td["task"]
                avg_plus_delta = axpy_state_dict(avg_base_sd, td["task_delta"], alpha=1.0)
                full_sd = to_cpu_fp32({k: v for k, v in diag_model.state_dict().items()})
                for k, v in avg_plus_delta.items():
                    full_sd[k] = v
                load_into_model(diag_clf.model, full_sd, strict=False)
                hf_path, hf_config, split_map = suite.resolver(task)
                hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
                diag_loaders = build_vision_loaders(
                    hf_ds=hf_ds, hf_path=hf_path, preprocess=clf_source.preprocess, ft_epochs=1,
                    split_map=split_map, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, val_fraction=val_fraction, seed=seed,
                )
                classnames = list(diag_loaders.classnames)
                diag_task_cfg = _build_cfg(source_cfg.model_name, source_cfg.pretrained, device, source_cfg.dtype, task)
                diag_clf.build_zeroshot_text_features(classnames, diag_task_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
                acc = float(diag_clf.top1(diag_loaders.test, device=device))
                diag_results[task] = acc
                print(f"  [diag] {task}: avg_base+delta = {acc:.4f}")
            diag_avg = sum(diag_results.values()) / max(1, len(diag_results))
            print(f"  [diag] avg = {diag_avg:.4f}")
            del diag_clf
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.empty_cache()

            # Phase 3: Correct deltas and transport
            for td in per_task_data:
                task = td["task"]
                print(f"\n=== Transporting {task} (averaged base) ===")
                corrected_delta: dict[str, torch.Tensor] = {}
                for k, v in td["task_delta"].items():
                    if k in avg_base_sd:
                        correction = td["task_source_base_sd"][k].float() - avg_base_sd[k].float()
                        corrected_delta[k] = (v.float() + correction).to(dtype=v.dtype)
                    else:
                        corrected_delta[k] = v

                transported_delta = _transport_delta(
                    ctx,
                    task=task,
                    task_source_base_sd=td["task_source_base_sd"],
                    task_delta=corrected_delta,
                    method_name=method_name,
                    method_params=method_params,
                    prepared=td["prepared"],
                    source_base_model=td["source_base_model"],
                    source_ft_model=td["source_ft_model"],
                    device=device,
                    strict_load=strict_load,
                )

                transported_tuned = axpy_state_dict(ctx.target_base_sd, transported_delta, alpha=1.0)
                save_path = os.path.join(save_dir, f"{task}_{method_name}_transported.pt")
                torch.save(to_cpu_fp32(transported_tuned), save_path)
                print(f"  Saved: {save_path}")
                all_deltas.append(transported_delta)

        else:
            # Default: per-task base construction (original behavior)
            for task in tasks_to_transport:
                print(f"\n=== Transporting {task} ===")
                hf_path, hf_config, split_map = suite.resolver(task)
                hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

                source_loaders = build_vision_loaders(
                    hf_ds=hf_ds, hf_path=hf_path, preprocess=clf_source.preprocess, ft_epochs=1,
                    split_map=split_map, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, val_fraction=val_fraction, seed=seed,
                )
                target_loaders = build_vision_loaders(
                    hf_ds=hf_ds, hf_path=hf_path, preprocess=clf_target.preprocess, ft_epochs=1,
                    split_map=split_map, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, val_fraction=val_fraction, seed=seed,
                )

                transported_delta = _transport_source_task(
                    ctx,
                    task=task,
                    source_task_cfg=_build_cfg(source_cfg.model_name, source_cfg.pretrained, device, source_cfg.dtype, task),
                    target_task_cfg=_build_cfg(target_cfg.model_name, target_cfg.pretrained, device, target_cfg.dtype, task),
                    tuned_path=tuned_ckpts[task],
                    method_name=method_name,
                    method_params=method_params,
                    block_ext_cfg=block_ext_cfg,
                    block_extension_calibration_loader=block_extension_calibration_loader,
                    source_loaders=source_loaders,
                    target_loaders=target_loaders,
                    source_classnames=list(source_loaders.classnames),
                    target_classnames=list(target_loaders.classnames),
                    device=device,
                    strict_load=strict_load,
                )

                transported_tuned = axpy_state_dict(ctx.target_base_sd, transported_delta, alpha=1.0)
                save_path = os.path.join(save_dir, f"{task}_{method_name}_transported.pt")
                torch.save(to_cpu_fp32(transported_tuned), save_path)
                print(f"  Saved: {save_path}")
                all_deltas.append(transported_delta)

        for task in native_target_tasks:
            print(f"\n=== Native target delta: {task} ===")
            delta = _load_task_delta(ctx.target_base_sd, tuned_ckpts[task])
            all_deltas.append(delta)
            print(f"  {task}: {len(delta)} params")

        print(f"\n=== Merging {len(all_deltas)} task vectors with {merge_method} (params={merge_params}) ===")
        print(f"  base_construction={base_construction}, alpha_mode={alpha_mode}")

        alphas = (
            [float(x) for x in torch.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)]
            if alpha_search else [float(cfg.get("alpha", 1.0))]
        )
        print(f"  Alpha sweep: {alphas[0]:.2f} to {alphas[-1]:.2f}, step={alpha_step}")

        eval_clf = OpenClipClassifier(
            model=deepcopy(clf_target.model),
            tokenizer=clf_target.tokenizer, preprocess=clf_target.preprocess,
            normalize=clf_target.normalize, logit_scale=clf_target.logit_scale,
        )

        task_data: dict[str, Any] = {}
        for task in all_tasks:
            hf_path, hf_config, split_map = suite.resolver(task)
            hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
            loaders = build_vision_loaders(
                hf_ds=hf_ds, hf_path=hf_path, preprocess=clf_target.preprocess, ft_epochs=1,
                split_map=split_map, batch_size=batch_size, num_workers=num_workers,
                pin_memory=True, val_fraction=val_fraction, seed=seed,
            )
            classnames = list(loaders.classnames)
            task_build_cfg = _build_cfg(target_cfg.model_name, target_cfg.pretrained, device, target_cfg.dtype, task)
            eval_clf.build_zeroshot_text_features(classnames, task_build_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
            task_data[task] = (loaders, classnames, task_build_cfg)

        if alpha_mode == "per_task":
            print("\n=== Per-task alpha search ===")
            best_alphas_per_task: dict[str, float] = {}
            for task, delta in zip(all_tasks, all_deltas, strict=False):
                print(f"  Searching alpha for {task}...")
                best_a = alphas[0]
                best_acc = -1.0
                for alpha in alphas:
                    merged_sd = axpy_state_dict(ctx.target_base_sd, delta, alpha=alpha)
                    load_into_model(eval_clf.model, merged_sd, strict=False)
                    loaders, classnames, task_build_cfg = task_data[task]
                    eval_clf.build_zeroshot_text_features(classnames, task_build_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
                    loader = loaders.val if alpha_search_split == "val" and loaders.val is not None else loaders.test
                    acc = float(eval_clf.top1(loader, device=device))
                    if acc > best_acc:
                        best_acc = acc
                        best_a = alpha
                best_alphas_per_task[task] = best_a
                print(f"    {task}: best_alpha={best_a:.2f} (val={best_acc:.4f})")

            print("\n  Per-task alphas: " + "  ".join(f"{t}={best_alphas_per_task[t]:.2f}" for t in all_tasks))

            # Scale each delta by its per-task alpha
            scaled_deltas = []
            for task, delta in zip(all_tasks, all_deltas, strict=False):
                a = best_alphas_per_task[task]
                scaled = {k: v * a for k, v in delta.items()}
                scaled_deltas.append(scaled)

            # Merge scaled deltas (apply at alpha=1.0 since scaling is baked in)
            merge_method_obj = get_merge_method(merge_method)
            tuned_sds = [axpy_state_dict(ctx.target_base_sd, d, alpha=1.0) for d in scaled_deltas]
            prepared = merge_method_obj.prepare(
                base=ctx.target_base_sd,
                tuned=tuned_sds,
                **merge_params,
            )

            # Single evaluation at alpha=1.0
            best_alpha = 1.0
            best_avg = -1.0
            results: dict[float, dict[str, float]] = {}

            merged_sd = merge_method_obj.apply(prepared, alpha=1.0)
            load_into_model(eval_clf.model, merged_sd, strict=False)
            per_task: dict[str, float] = {}
            for task in all_tasks:
                loaders, classnames, task_build_cfg = task_data[task]
                eval_clf.build_zeroshot_text_features(classnames, task_build_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
                loader = loaders.val if alpha_search_split == "val" and loaders.val is not None else loaders.test
                per_task[task] = float(eval_clf.top1(loader, device=device))
            avg = sum(per_task.values()) / max(1, len(per_task))
            results[1.0] = per_task
            best_avg = avg
            print(f"  merged (per-task alpha)  avg={avg:.4f}  " + "  ".join(f"{t}={per_task[t]:.4f}" for t in all_tasks))
            if run_logger is not None:
                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={"alpha/value": 1.0, "alpha/avg_val": float(avg)},
                    context={"per_task_acc": per_task, "per_task_alphas": best_alphas_per_task},
                )

        else:
            # Default: shared alpha search
            merge_method_obj = get_merge_method(merge_method)
            tuned_sds = [axpy_state_dict(ctx.target_base_sd, delta, alpha=1.0) for delta in all_deltas]
            prepared = merge_method_obj.prepare(
                base=ctx.target_base_sd,
                tuned=tuned_sds,
                **merge_params,
            )

            best_alpha = alphas[0]
            best_avg = -1.0
            results: dict[float, dict[str, float]] = {}

            for alpha in alphas:
                merged_sd = merge_method_obj.apply(prepared, alpha=alpha)
                load_into_model(eval_clf.model, merged_sd, strict=False)
                per_task: dict[str, float] = {}
                for task in all_tasks:
                    loaders, classnames, task_build_cfg = task_data[task]
                    eval_clf.build_zeroshot_text_features(classnames, task_build_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
                    loader = loaders.val if alpha_search_split == "val" and loaders.val is not None else loaders.test
                    per_task[task] = float(eval_clf.top1(loader, device=device))
                avg = sum(per_task.values()) / max(1, len(per_task))
                results[alpha] = per_task
                print(f"  alpha={alpha:.2f}  avg={avg:.4f}  " + "  ".join(f"{t}={per_task[t]:.4f}" for t in all_tasks))
                if run_logger is not None:
                    run_logger.log_event(
                        "alpha_eval_end",
                        metrics={
                            "alpha/value": float(alpha),
                            "alpha/avg_val": float(avg),
                        },
                        context={"per_task_acc": per_task},
                    )
                if avg > best_avg:
                    best_avg = avg
                    best_alpha = alpha

        print(f"\n=== Best alpha={best_alpha:.2f} (val avg={best_avg:.4f}) ===")
        if alpha_mode == "per_task":
            load_into_model(eval_clf.model, merge_method_obj.apply(prepared, alpha=1.0), strict=False)
        else:
            load_into_model(eval_clf.model, merge_method_obj.apply(prepared, alpha=best_alpha), strict=False)

        test_results: dict[str, float] = {}
        for task in all_tasks:
            loaders, classnames, task_build_cfg = task_data[task]
            eval_clf.build_zeroshot_text_features(classnames, task_build_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
            test_results[task] = float(eval_clf.top1(loaders.test, device=device))
            print(f"  [test] {task}: {test_results[task]:.4f}")

        avg_test = sum(test_results.values()) / max(1, len(test_results))
        print(f"\n=== TEST avg={avg_test:.4f} ===")

        summary = {
            "method": method_name, "merge_method": merge_method,
            "merge_params": merge_params,
            "base_construction": base_construction,
            "alpha_mode": alpha_mode,
            "best_alpha": best_alpha, "best_val_avg": best_avg,
            "best_alphas_per_task": best_alphas_per_task if alpha_mode == "per_task" else None,
            "test_avg": avg_test, "test_per_task": test_results,
            "all_alpha_results": {f"{a:.2f}": v for a, v in results.items()},
            "tasks_to_transport": tasks_to_transport,
            "native_target_tasks": native_target_tasks,
        }
        summary_path = os.path.join(save_dir, f"transport_merge_{method_name}_{merge_method}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

        if run_logger is not None:
            run_logger.log_summary({
                "method": method_name,
                "merge_method": merge_method,
                "merge_params": merge_params,
                "base_construction": base_construction,
                "alpha_mode": alpha_mode,
                "suite": cfg.get("suite", "vision8"),
                "tasks_to_transport": tasks_to_transport,
                "native_target_tasks": native_target_tasks,
                "best_alpha": best_alpha,
                "best_val_avg": best_avg,
                "best_alphas_per_task": best_alphas_per_task if alpha_mode == "per_task" else None,
                "test_avg": avg_test,
                "test_per_task": test_results,
            })
            run_logger.finish("success")

        print("\nDone.")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
