"""Calibration-data and per-task context builders for the vision rebase entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ...data.balanced_calibration import Vision8TaskContext, build_balanced_vision8_calibration_loaders
from ...data.templates import get_templates
from ...data.vision_loaders import (
    build_vision_calibration_loader,
    build_vision_loaders,
    extract_classnames,
    load_hf_splits,
)
from ...eval.utils import humanize
from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..block_extension import select_loader


@dataclass(frozen=True)
class _TaskContext:
    loaders: Any
    source_loaders: Any | None
    classnames: list[str]
    build_cfg_task: OpenClipBuildConfig
    source_build_cfg_task: OpenClipBuildConfig
    target_text_features: torch.Tensor | None = None
    source_text_features: torch.Tensor | None = None


@dataclass(frozen=True)
class _CalibrationLoaders:
    """Minimal loader bundle accepted by the transport preparation helpers."""

    train: Any
    val: Any | None = None
    test: Any | None = None


def _select_dedicated_brace_loader(
    *,
    brace_loader: Any | None,
    transport_loader: Any,
    correction_enabled: bool,
) -> Any | None:
    """Validate that BRACE does not consume the transport calibration loader."""
    if correction_enabled and brace_loader is None:
        raise RuntimeError(
            "merge_then_brace_then_transport requires a dedicated BRACE calibration loader when correction is enabled."
        )
    if brace_loader is transport_loader:
        raise RuntimeError("BRACE and transport calibration must use separate loader instances.")
    return brace_loader


def _build_direct_paired_calibration_context(
    dataset_spec: Mapping[str, Any],
    *,
    suite: Any,
    cfg: dict[str, Any],
    clf_source: OpenClipClassifier,
    clf_target: OpenClipClassifier,
    source_cfg: OpenClipBuildConfig,
    target_cfg: OpenClipBuildConfig,
) -> _TaskContext:
    """Build label-aligned source/target views of one direct HF dataset."""
    source_loader = build_vision_calibration_loader(
        dataset_spec,
        resolver=suite.resolver,
        preprocess=clf_source.preprocess,
        calibration_split=str(dataset_spec.get("split", "valid")),
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 6)),
        pin_memory=True,
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    target_loader = build_vision_calibration_loader(
        dataset_spec,
        resolver=suite.resolver,
        preprocess=clf_target.preprocess,
        calibration_split=str(dataset_spec.get("split", "valid")),
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 6)),
        pin_memory=True,
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    path = str(dataset_spec.get("path", dataset_spec.get("hf_path", dataset_spec.get("dataset", ""))))
    split = str(dataset_spec.get("split", "valid"))
    hf_ds = load_hf_splits(
        path,
        config=(str(dataset_spec["config"]) if dataset_spec.get("config") is not None else None),
        requested_splits=(split,),
    )
    classnames = list(
        extract_classnames(
            hf_ds,
            label_key=str(dataset_spec.get("label_key", "label")),
            strict=False,
        )
    )
    templates = get_templates("ImageNet1K")

    def _with_templates(build: OpenClipBuildConfig) -> OpenClipBuildConfig:
        return OpenClipBuildConfig(
            model_name=build.model_name,
            pretrained=build.pretrained,
            device=build.device,
            dtype=build.dtype,
            prompt_templates=templates,
        )

    return _TaskContext(
        loaders=_CalibrationLoaders(train=target_loader),
        source_loaders=_CalibrationLoaders(train=source_loader),
        classnames=classnames,
        build_cfg_task=_with_templates(target_cfg),
        source_build_cfg_task=_with_templates(source_cfg),
    )


def _build_balanced_calibration_context(
    per_task: Sequence[Mapping[str, Any]],
    *,
    cfg: dict[str, Any],
    clf_source: OpenClipClassifier,
    clf_target: OpenClipClassifier,
    n_batches: int,
    split: str = "val",
) -> tuple[_TaskContext, dict[str, Any]]:
    """Build the paired, exactly balanced Vision8 transport context."""
    contexts: dict[str, Vision8TaskContext] = {}
    by_task = {str(item["task"]): item for item in per_task}
    for task, item in by_task.items():
        source_loaders = item["source_loaders"]
        if source_loaders is None:
            raise RuntimeError(f"Balanced transport calibration requires source loaders for {task}.")
        source_loader = select_loader(
            split,
            train_loader=source_loaders.train,
            val_loader=source_loaders.val,
            test_loader=source_loaders.test,
        )
        target_loader = select_loader(
            split,
            train_loader=item["loaders"].train,
            val_loader=item["loaders"].val,
            test_loader=item["loaders"].test,
        )
        contexts[task] = Vision8TaskContext(
            source_dataset=source_loader.dataset,
            target_dataset=target_loader.dataset,
            classnames=list(item["classnames"]),
        )

    balanced = build_balanced_vision8_calibration_loaders(
        contexts,
        n_batches=int(n_batches),
        batch_size=int(cfg.get("batch_size", 128)),
        seed=int(cfg.get("seed", 42)),
        num_workers=int(cfg.get("num_workers", 6)),
        pin_memory=True,
    )

    source_features: list[torch.Tensor] = []
    target_features: list[torch.Tensor] = []
    for task in sorted(by_task):
        item = by_task[task]
        names = list(item["classnames"])
        source_features.append(
            clf_source._compute_zeroshot_text_features(names, item["source_build_cfg_task"]).detach().cpu()
        )
        target_features.append(clf_target._compute_zeroshot_text_features(names, item["build_cfg_task"]).detach().cpu())

    first = by_task[sorted(by_task)[0]]
    context = _TaskContext(
        loaders=_CalibrationLoaders(train=balanced.target_loaders),
        source_loaders=_CalibrationLoaders(train=balanced.source_loaders),
        classnames=list(balanced.union_classnames),
        build_cfg_task=first["build_cfg_task"],
        source_build_cfg_task=first["source_build_cfg_task"],
        target_text_features=torch.cat(target_features, dim=0),
        source_text_features=torch.cat(source_features, dim=0),
    )
    return context, {"plan": balanced.plan, "fingerprint": balanced.fingerprint}


DIRECT_RESIDUAL_TINY_IMAGENET_SPEC = {"path": "zh-plus/tiny-imagenet", "split": "valid"}


TRANSPORT_CALIBRATION_DATA = ("task_local", "tiny_imagenet")


def _resolve_transport_calibration_data(cfg: Mapping[str, Any], *, theseus_like_method: bool, bico_mode: bool) -> str:
    """Validate the top-level ``transport_calibration_data`` key.

    ``"task_local"`` (default) keeps THESEUS/BiCo calibrating on each task's own
    train loaders. ``"tiny_imagenet"`` makes every per-task prepare use one
    task-independent paired Tiny-ImageNet context (the same split Direct
    Residual's ``calibration_data="tiny_imagenet"`` uses); BiCo's gradient
    recipe then scores Tiny-ImageNet's own labels and class names. Only the
    per-task transport path of THESEUS/BiCo reads it.
    """
    value = str(cfg.get("transport_calibration_data", "task_local")).strip().lower()
    if value not in TRANSPORT_CALIBRATION_DATA:
        raise ValueError(f"transport_calibration_data must be one of {TRANSPORT_CALIBRATION_DATA}, got {value!r}")
    if value != "task_local" and not (theseus_like_method or bico_mode):
        raise ValueError(
            "transport_calibration_data applies to THESEUS/BiCo only (Direct Residual uses calibration_data)"
        )
    return value


def _build_direct_residual_calibration(
    calibration_data: str,
    *,
    per_task: Sequence[Mapping[str, Any]],
    suite: Any,
    cfg: dict[str, Any],
    clf_source: OpenClipClassifier,
    clf_target: OpenClipClassifier,
    source_cfg: OpenClipBuildConfig,
    target_cfg: OpenClipBuildConfig,
    num_batches: int,
    calibration_seed: int,
) -> tuple[_TaskContext, dict[str, Any]]:
    """The one task-independent calibration context of ``calibration_data``.

    ``"tiny_imagenet"`` reuses `_build_direct_paired_calibration_context` on the
    whole Tiny-ImageNet ``valid`` split (10,000 images); Direct Residual's own
    seeded ``paired_calibration`` then draws ``num_batches * batch_size`` of
    them, exactly as it does from a task's train split. ``"vision8_mix"`` reuses
    `_build_balanced_calibration_context` on the tasks' train splits with
    ``num_batches`` complete balanced batches (``batch_size / n_tasks`` images
    per task per batch), so the fit sees every balanced image once; its
    per-task sample draw uses ``calibration_seed`` (Direct Residual's own
    seed), so a calibration-seed replicate changes the balanced images too.
    """
    if calibration_data == "tiny_imagenet":
        context = _build_direct_paired_calibration_context(
            DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
            suite=suite,
            cfg=cfg,
            clf_source=clf_source,
            clf_target=clf_target,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
        )
        return context, {
            "calibration_data": calibration_data,
            **DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
            "num_samples": len(context.loaders.train.dataset),
        }
    if calibration_data == "vision8_mix":
        batch_size = int(cfg.get("batch_size", 128))
        if not per_task or batch_size % len(per_task):
            raise ValueError(
                f"calibration_data='vision8_mix' needs batch_size divisible by the {len(per_task)} "
                f"calibrated tasks (got batch_size={batch_size})."
            )
        context, meta = _build_balanced_calibration_context(
            per_task,
            cfg={**cfg, "seed": int(calibration_seed)},
            clf_source=clf_source,
            clf_target=clf_target,
            n_batches=num_batches,
            split="train",
        )
        return context, {
            "calibration_data": calibration_data,
            "split": "train",
            "n_batches": int(num_batches),
            "seed": int(calibration_seed),
            "samples_per_task": meta["plan"]["samples_per_task"],
            "fingerprint": meta["fingerprint"],
        }
    raise ValueError(f"no task-independent calibration context for calibration_data={calibration_data!r}")


def _build_task_context(
    task: str,
    *,
    suite: Any,
    cfg: dict[str, Any],
    clf_target: OpenClipClassifier,
    clf_source: OpenClipClassifier,
    source_cfg: OpenClipBuildConfig,
    target_cfg: OpenClipBuildConfig,
    use_humanized_classnames: bool,
    need_source_loaders: bool,
) -> _TaskContext:
    hf_path, hf_config, split_map = suite.resolver(task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

    loader_kwargs = dict(
        hf_ds=hf_ds,
        hf_path=hf_path,
        ft_epochs=1,
        split_map=split_map,
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 6)),
        pin_memory=True,
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    loaders = build_vision_loaders(preprocess=clf_target.preprocess, **loader_kwargs)

    classnames = list(loaders.classnames)
    if use_humanized_classnames:
        classnames = [humanize(c) for c in classnames]

    templates = get_templates(task)
    if not templates:
        raise ValueError(f"get_templates('{task}') returned empty list")

    def _build_cfg_for(build: OpenClipBuildConfig) -> OpenClipBuildConfig:
        return OpenClipBuildConfig(
            model_name=build.model_name,
            pretrained=build.pretrained,
            device=build.device,
            dtype=build.dtype,
            prompt_templates=templates,
        )

    return _TaskContext(
        loaders=loaders,
        source_loaders=(
            build_vision_loaders(preprocess=clf_source.preprocess, **loader_kwargs) if need_source_loaders else None
        ),
        classnames=classnames,
        build_cfg_task=_build_cfg_for(target_cfg),
        source_build_cfg_task=_build_cfg_for(source_cfg),
    )


@dataclass
class RunCalibration:
    """Per-task contexts and the task-independent calibration contexts a run builds before any transport."""

    task_context_by_name: dict[str, _TaskContext]
    #: One dict per task, in task order (loaders, classnames, build configs); consumed by the alpha search.
    per_task: list[dict[str, Any]]
    ariadne_calibration_ctx: _TaskContext | None
    ariadne_calibration_meta: dict[str, Any]
    transport_calibration_ctx: _TaskContext | None
    transport_calibration_meta: dict[str, Any]
    #: The dedicated BRACE loader (replaced by the balanced vision8 mix loader under that protocol).
    block_extension_calibration_loader: Any
    brace_calibration_metadata: dict[str, Any] | None


def build_run_calibration(
    resolved: Any,
    plan: Any,
    *,
    cfg: Mapping[str, Any],
    clf_source: Any,
    clf_target: Any,
    source_cfg: Any,
    target_cfg: Any,
    native_tasks: set[str],
    use_humanized_classnames: bool,
    block_extension_calibration_loader: Any,
    run_logger: Any,
) -> RunCalibration:
    """Build every dataset context of the run once, in the legacy order (the global RNG makes it observable).

    Order: per-task contexts, the task-independent Ariadne calibration, the THESEUS/BiCo transport
    calibration, then the balanced BRACE calibration mix.
    """
    suite = resolved.suite
    tasks = resolved.tasks
    theseus_like_method = resolved.theseus_like_method
    transfusion_mode = resolved.transfusion_mode
    bico_mode = resolved.bico_mode
    direct_residual_like = resolved.direct_residual_like
    direct_residual_cfg = resolved.ariadne_cfg
    block_extension_cfg = resolved.block_extension_cfg
    run_block_extension_prestep = plan.run_block_extension_prestep
    task_context_by_name: dict[str, _TaskContext] = {}
    per_task: list[dict[str, Any]] = []
    brace_calibration_metadata: dict[str, Any] | None = None

    for task in tasks:
        task_ctx = _build_task_context(
            task,
            suite=suite,
            cfg=cfg,
            clf_target=clf_target,
            clf_source=clf_source,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            use_humanized_classnames=use_humanized_classnames,
            need_source_loaders=bool(
                (theseus_like_method or transfusion_mode or bico_mode or direct_residual_like)
                and task not in native_tasks
            ),
        )
        task_context_by_name[task] = task_ctx
        per_task.append(
            {
                "task": task,
                "loaders": task_ctx.loaders,
                "classnames": task_ctx.classnames,
                "build_cfg_task": task_ctx.build_cfg_task,
                "source_loaders": task_ctx.source_loaders,
                "source_build_cfg_task": task_ctx.source_build_cfg_task,
            }
        )

    # Task-independent Direct Residual calibration (direct_residual_params.
    # calibration_data != "task_local"): ONE paired context, built here once
    # and passed to every Direct Residual fit below (per task and
    # merge_in_source_then_fit alike). Only the fit's calibration images
    # change; each task's alpha search and evaluation keep its own splits.
    direct_residual_calibration_ctx: _TaskContext | None = None
    direct_residual_calibration_meta: dict[str, Any] = {"calibration_data": "task_local"}
    if direct_residual_like and direct_residual_cfg.calibration_data != "task_local":
        direct_residual_calibration_ctx, direct_residual_calibration_meta = _build_direct_residual_calibration(
            direct_residual_cfg.calibration_data,
            per_task=[item for item in per_task if item["task"] not in native_tasks],
            suite=suite,
            cfg=cfg,
            clf_source=clf_source,
            clf_target=clf_target,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            num_batches=int(direct_residual_cfg.num_batches),
            calibration_seed=int(direct_residual_cfg.seed),
        )
        print(f"Direct Residual calibration: {direct_residual_calibration_meta}")

    # Task-independent THESEUS/BiCo calibration (transport_calibration_data):
    # one paired context for every task's prepare; alpha search and
    # evaluation keep each task's own splits.
    transport_calibration_data = _resolve_transport_calibration_data(
        cfg, theseus_like_method=theseus_like_method, bico_mode=bico_mode
    )
    transport_calibration_ctx: _TaskContext | None = None
    transport_calibration_meta: dict[str, Any] = {"transport_calibration_data": transport_calibration_data}
    if transport_calibration_data == "tiny_imagenet":
        transport_calibration_ctx = _build_direct_paired_calibration_context(
            DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
            suite=suite,
            cfg=cfg,
            clf_source=clf_source,
            clf_target=clf_target,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
        )
        transport_calibration_meta.update(
            **DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
            num_samples=len(transport_calibration_ctx.loaders.train.dataset),
            num_classes=len(transport_calibration_ctx.classnames),
        )
        print(f"Transport calibration: {transport_calibration_meta}")

    brace_protocol = str(
        (cfg.get("block_extension_params", {}) or {}).get("calibration_protocol", "task_local")
    ).lower()
    if (
        run_block_extension_prestep
        and not block_extension_cfg.skip_correction
        and brace_protocol.startswith("vision8_mix")
    ):
        brace_mix_context, brace_mix_metadata = _build_balanced_calibration_context(
            per_task,
            cfg=cfg,
            clf_source=clf_source,
            clf_target=clf_target,
            n_batches=block_extension_cfg.n_batches_act,
            split=block_extension_cfg.calibration_split,
        )
        block_extension_calibration_loader = brace_mix_context.source_loaders.train
        brace_calibration_metadata = brace_mix_metadata
        run_logger.log_event("brace_calibration_plan", context=brace_mix_metadata)

    return RunCalibration(
        task_context_by_name=task_context_by_name,
        per_task=per_task,
        ariadne_calibration_ctx=direct_residual_calibration_ctx,
        ariadne_calibration_meta=direct_residual_calibration_meta,
        transport_calibration_ctx=transport_calibration_ctx,
        transport_calibration_meta=transport_calibration_meta,
        block_extension_calibration_loader=block_extension_calibration_loader,
        brace_calibration_metadata=brace_calibration_metadata,
    )
