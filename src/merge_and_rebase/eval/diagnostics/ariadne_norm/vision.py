"""Vision (OpenCLIP) adapter of the ``ariadne_norm`` diagnostics.

``build_vision_inputs(cfg, device)`` re-creates, task by task, exactly what a reference ``eval.vision_rebase`` Ariadne
run handed to ``AriadneRebase().prepare`` (``AriadneStage.run``, ``eval/vision_rebase/method_stages.py``), so that a
re-fit with the same code is bit-identical to the run's saved task vector. Everything is derived from the reference
summary JSON: its ``run_logging.metadata.resolved_config`` is the merged config the run resolved, and it is resolved
here through the same ``resolve_run_config`` call.

Config keys (on top of those documented in ``cli.py``):

- ``reference_summary``: path of the reference run's summary JSON (required). The run directory is its parent; the saved
  unit task vectors are read from ``<run_dir>/transported_tvs`` when present.
- ``tasks``: ``"all"`` (the reference run's task list) or a list of task names.
- ``generic_num_batches`` (default 10) / ``generic_seed`` (default 0): the N6 off-task batches, drawn from the
  Tiny-ImageNet ``valid`` split under the TARGET preprocessing with ``paired_calibration``'s seeded draw.
- ``num_batches_override``: must stay ``None``; this adapter only reproduces runs.

Reproduction caveats (recorded per task in ``provenance``): the calibration images of ``task_local`` runs are the
``num_batches * batch_size`` images ``paired_calibration`` draws with ``config.seed`` from each task's train split; the
reference code fingerprint is recorded next to the current one because numbers from different code generations are
only comparable after byte-level equality has been shown (the runner's N0 check does that for the task vector).
The job script must keep ``HF_HUB_OFFLINE=1`` / ``HF_DATASETS_OFFLINE=1`` set (nodes have no network).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.data.vision_loaders import build_vision_calibration_loader
from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.utils import patch_base_for_attn
from merge_and_rebase.eval.vision_rebase.artifacts import _state_dict_sha256
from merge_and_rebase.eval.vision_rebase.cli import _set_deterministic_seed
from merge_and_rebase.eval.vision_rebase.context import (
    DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
    _build_direct_residual_calibration,
    _build_task_context,
    _TaskContext,
)
from merge_and_rebase.eval.vision_rebase.merge import _infer_ckpt_base
from merge_and_rebase.eval.vision_rebase.method_stages import AriadneStage
from merge_and_rebase.eval.vision_rebase.pipeline import _owned_cpu_fp32
from merge_and_rebase.io.ckpt import load_ckpt, load_into_model, resolve_ckpt_path
from merge_and_rebase.io.peft_helpers import normalize_attn_patch_cfg
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from merge_and_rebase.rebase.block_extension.config import select_loader
from merge_and_rebase.rebase.methods._ariadne.capture import _stable_dataset_identity, paired_calibration
from merge_and_rebase.rebase.methods._ariadne.fit import _task_vector_sha256
from merge_and_rebase.rebase.methods._ariadne.layouts import _layout_for
from merge_and_rebase.rebase.prestep import StageEnv
from merge_and_rebase.rebase.run_config import resolve_run_config
from merge_and_rebase.run_logging import _code_fingerprint

from .inputs import DiagnosticInputs

#: The fitted projection of the vision layout (the reference runs fit ``components == ("mlp.c_proj",)`` only).
_COMPONENT = "mlp.c_proj"
#: Tensor-valued (per-channel) field of a summary row; the runner compares scalars only.
_TENSOR_ROW_FIELDS = ("bias_correction",)


# ---- pure helpers (unit-tested without models) --------------------------------------------------------------------


def token_index_for_batch(batch_size: int, n_rows: int) -> torch.Tensor:
    """Within-image token index (0 = CLS) of each row of ``[B, T, D].reshape(-1, D)``: ``arange(T).repeat(B)``."""
    if batch_size <= 0 or n_rows % batch_size:
        raise ValueError(f"{n_rows} rows are not a whole number of tokens for {batch_size} images")
    return torch.arange(n_rows // batch_size).repeat(batch_size)


def vision_token_index(batch: Any, n_rows: int) -> torch.Tensor:
    """``token_index`` of `DiagnosticInputs` for an ``(images, labels)`` vision batch (B from ``len(batch[0])``)."""
    return token_index_for_batch(_layout_for(None).batch_size(batch), n_rows)


def reference_vector_path(run_dir: Path | str, task: str) -> Path | None:
    """The saved transported vector of ``task`` under ``<run_dir>/transported_tvs`` or ``None`` when absent.

    The saver names it ``{task}_{method.name}_transported_native.pt`` with the run's spelling of the method (the
    canonical ``ariadne`` or the legacy alias ``direct_residual``); finding both is ambiguous and refused.
    """
    root = Path(run_dir) / "transported_tvs"
    found = [
        root / f"{task}_{name}_transported_native.pt"
        for name in ("direct_residual", "ariadne")
        if (root / f"{task}_{name}_transported_native.pt").exists()
    ]
    if len(found) > 1:
        raise ValueError(f"ambiguous saved vectors for {task!r}: {[str(p) for p in found]}")
    return found[0] if found else None


def strip_tensor_fields(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summary rows without their tensor-like list fields (``bias_correction``)."""
    return [{k: v for k, v in row.items() if k not in _TENSOR_ROW_FIELDS} for row in rows]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_loader_identity(loader: Any, *, seed: int, num_batches: int) -> dict[str, Any]:
    """Dataset identity of a calibration loader and the draw ``paired_calibration`` will make from it."""
    n = len(loader.dataset)
    take = int(num_batches) * int(loader.batch_size)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()[:take]
    return {
        "dataset_identity": _stable_dataset_identity(loader.dataset),
        "n_dataset": n,
        "batch_size": int(loader.batch_size),
        "sampling_seed": int(seed),
        "num_batches": int(num_batches),
        "indices_sha256": hashlib.sha256(repr(order).encode()).hexdigest(),
        "first_indices": order[:8],
    }


# ---- adapter ------------------------------------------------------------------------------------------------------


def _generic_target_batches(
    *, suite: Any, clf_target: OpenClipClassifier, ref_cfg: Mapping[str, Any], num_batches: int, seed: int
) -> tuple[list[Any], dict[str, Any]]:
    """N6 batches: Tiny-ImageNet ``valid`` under the TARGET preprocessing, in ``paired_calibration``'s batch format.

    The loader is built exactly like the ``calibration_data="tiny_imagenet"`` target view
    (`_build_direct_paired_calibration_context`); the seeded draw is ``paired_calibration``'s own (``randperm`` with a
    fresh generator), so batches are ``[images, labels]`` lists of CPU tensors like the calibration target batches.
    """
    loader = build_vision_calibration_loader(
        DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
        resolver=suite.resolver,
        preprocess=clf_target.preprocess,
        calibration_split=str(DIRECT_RESIDUAL_TINY_IMAGENET_SPEC["split"]),
        batch_size=int(ref_cfg.get("batch_size", 128)),
        num_workers=0,
        pin_memory=False,
        val_fraction=float(ref_cfg.get("val_fraction", 0.1)),
        seed=int(ref_cfg.get("seed", 42)),
    )
    _, batches, meta = paired_calibration(loader, loader, num_batches=int(num_batches), seed=int(seed))
    provenance = {
        "dataset": dict(DIRECT_RESIDUAL_TINY_IMAGENET_SPEC),
        "preprocess": "target",
        "generator_seed": int(seed),
        "num_batches": int(num_batches),
        "dataset_identity": meta["dataset_identity"],
        "indices": list(meta["indices"]),
        "n_dataset": len(loader.dataset),
    }
    return batches, provenance


def _build_clf(
    model_name: str, pretrained: str, device: str, dtype: Any
) -> tuple[OpenClipClassifier, OpenClipBuildConfig]:
    build = OpenClipBuildConfig(model_name=model_name, pretrained=pretrained, device=device, dtype=dtype)
    return OpenClipClassifier.build(build), build


def build_vision_inputs(cfg: dict, device: str) -> Iterator[DiagnosticInputs]:
    """Yield one `DiagnosticInputs` per requested task, reproducing the reference run's Ariadne fit inputs."""
    if cfg.get("num_batches_override") is not None:
        raise NotImplementedError("num_batches_override: the diagnostics only reproduce the reference run's fit")
    summary_path = Path(cfg["reference_summary"]).resolve()
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    metadata = summary["run_logging"]["metadata"]
    ref_cfg = copy.deepcopy(metadata["resolved_config"])
    run_dir = summary_path.parent

    # Same call as vision_rebase.cli.main (cfg is the merged config with the block_extension_enabled default applied).
    resolved = resolve_run_config(ref_cfg, suites=SUITES)
    if not resolved.direct_residual_like or resolved.ariadne_cfg is None:
        raise ValueError(f"reference run is not an Ariadne run (method {resolved.method_name!r})")
    config = resolved.ariadne_cfg
    if config.merge_mode != "per_task_then_merge":
        raise NotImplementedError(f"merge_mode={config.merge_mode!r}: only per-task fits are reproduced")
    if ref_cfg.get("load_direct_residual_tvs_dir"):
        raise NotImplementedError("the reference run loaded saved vectors instead of fitting")
    if list(summary["tasks"]) != list(resolved.tasks):
        raise ValueError(f"summary tasks {summary['tasks']} != resolved tasks {resolved.tasks}")
    requested = resolved.tasks if cfg.get("tasks", "all") == "all" else [str(t) for t in cfg["tasks"]]
    unknown = [t for t in requested if t not in resolved.tasks]
    if unknown:
        raise ValueError(f"tasks {unknown} are not tasks of the reference run {resolved.tasks}")

    # vision_rebase.cli.main seeds before anything is built. Loader construction here uses explicit seeds only, but the
    # global state is reproduced anyway.
    _set_deterministic_seed(int(ref_cfg.get("seed", 42)))

    tuned_by_task = {t: resolve_ckpt_path(str(p)) for t, p in ref_cfg["tuned_ckpts"].items()}
    source_clf, source_cfg = _build_clf(
        ref_cfg.get("source_clip_model", "ViT-B-32"),
        ref_cfg.get("source_clip_pretrained", "openai"),
        device,
        ref_cfg.get("dtype", None),
    )
    target_clf, target_cfg = _build_clf(
        ref_cfg.get("target_clip_model", "ViT-B-32"),
        ref_cfg.get("target_clip_pretrained", "laion2b_s34b_b79k"),
        device,
        ref_cfg.get("dtype", None),
    )
    source_depth = len(source_clf.model.visual.transformer.resblocks)
    target_depth = len(target_clf.model.visual.transformer.resblocks)
    plan = resolved.bind(source_depth, target_depth)

    # pipeline.run_rebase: optional attention patch (mutates clf.model in place), then owned CPU fp32 base states.
    patched = False
    attn_patch_cfg_raw = ref_cfg.get("attn_patch_cfg", None)
    if attn_patch_cfg_raw is not None and not isinstance(attn_patch_cfg_raw, dict):
        raise ValueError("config['attn_patch_cfg'] must be a dict when provided.")
    if bool(ref_cfg.get("patched_attn", attn_patch_cfg_raw is not None)):
        patched = True
        attn_patch_cfg = normalize_attn_patch_cfg(attn_patch_cfg_raw)
        source_base_sd = _owned_cpu_fp32(
            patch_base_for_attn(
                clf=source_clf, base_ckpt=None, strict_load=resolved.strict_load, attn_patch_cfg=attn_patch_cfg
            )
        )
        target_base_sd = _owned_cpu_fp32(
            patch_base_for_attn(
                clf=target_clf, base_ckpt=None, strict_load=resolved.strict_load, attn_patch_cfg=attn_patch_cfg
            )
        )
    else:
        source_base_sd = _owned_cpu_fp32({k: v for k, v in source_clf.model.state_dict().items()})
        target_base_sd = _owned_cpu_fp32({k: v for k, v in target_clf.model.state_dict().items()})
    target_hash = _state_dict_sha256(target_base_sd)
    reference_target_hash = summary.get("target_hash_before")
    if reference_target_hash is not None and reference_target_hash != target_hash:
        print(
            f"[ariadne_norm] WARNING: target base hash {target_hash} != reference {reference_target_hash}; "
            "the target weights differ from the reference run's"
        )

    env = StageEnv(
        resolved=resolved,
        plan=plan,
        cfg=ref_cfg,
        device=device,
        clf_source=source_clf,
        clf_target=target_clf,
        tuned_by_task=tuned_by_task,
        patch_attn_before_rebase=patched,
        source_base_sd=source_base_sd,
        target_base_sd=target_base_sd,
        target_hash_before=target_hash,
        run_logger=None,
    )
    # The stage object only supplies `.pairing(env)` here (computed from the same config/depths as in the run).
    stage = AriadneStage(
        env, task_contexts={}, tasks=list(resolved.tasks), merge_weights=[], calibration_ctx=None, calibration_meta={}
    )
    pairing = stage.pairing(env)

    # Native-target checkpoints are skipped by the run (pipeline.run_rebase pre-pass + TaskPipeline.run).
    explicit_native = {str(t) for t in (ref_cfg.get("native_target_tasks", []) or [])}
    auto_detect = bool(ref_cfg.get("auto_detect_ckpt_base", True))

    def is_native(task: str, raw_sd: Mapping[str, torch.Tensor]) -> bool:
        if task in explicit_native:
            return True
        if not auto_detect:
            return False
        inferred = _infer_ckpt_base(raw_sd, source_base_sd=source_base_sd, target_base_sd=target_base_sd)
        if inferred is None:
            raise ValueError(f"tuned checkpoint of {task!r} matches neither the source nor the target backbone")
        return inferred == "target"

    use_humanized = not bool(ref_cfg.get("no_humanize", True))

    def task_context(task: str) -> _TaskContext:
        return _build_task_context(
            task,
            suite=resolved.suite,
            cfg=ref_cfg,
            clf_target=target_clf,
            clf_source=source_clf,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            use_humanized_classnames=use_humanized,
            need_source_loaders=True,
        )

    # Task-independent calibration (build_run_calibration): one shared context for every task.
    shared_ctx: _TaskContext | None = None
    shared_meta: dict[str, Any] = {"calibration_data": "task_local"}
    if config.calibration_data != "task_local":
        per_task: list[dict[str, Any]] = []
        if config.calibration_data == "vision8_mix":
            # The balanced mix is built from every non-native task of the run, not only the requested ones.
            for t in resolved.tasks:
                if is_native(t, load_ckpt(str(tuned_by_task[t]))):
                    continue
                ctx = task_context(t)
                per_task.append(
                    {
                        "task": t,
                        "loaders": ctx.loaders,
                        "classnames": ctx.classnames,
                        "build_cfg_task": ctx.build_cfg_task,
                        "source_loaders": ctx.source_loaders,
                        "source_build_cfg_task": ctx.source_build_cfg_task,
                    }
                )
        shared_ctx, shared_meta = _build_direct_residual_calibration(
            config.calibration_data,
            per_task=per_task,
            suite=resolved.suite,
            cfg=ref_cfg,
            clf_source=source_clf,
            clf_target=target_clf,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            num_batches=int(config.num_batches),
            calibration_seed=int(config.seed),
        )
        del per_task

    generic_batches, generic_provenance = _generic_target_batches(
        suite=resolved.suite,
        clf_target=target_clf,
        ref_cfg=ref_cfg,
        num_batches=int(cfg.get("generic_num_batches", 10)),
        seed=int(cfg.get("generic_seed", 0)),
    )

    component_key = lambda j: _layout_for(None).component_key(int(j), _COMPONENT, prefixed=True)  # noqa: E731
    unit_vector_ok = float(config.strength) == 1.0 and config.tv_scaling == "none"
    selected_alpha = summary.get("selected_alpha_by_task", {}) or {}
    rows_by_task = (summary.get("direct_residual", {}) or {}).get("diagnostics_by_task", {}) or {}
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    current_fingerprint = _code_fingerprint()

    for task in requested:
        # AriadneStage.run: native copies of the source models, never a reference from elsewhere in the pipeline.
        ckpt_path = str(tuned_by_task[task])
        raw_sd = load_ckpt(ckpt_path)
        if is_native(task, raw_sd):
            print(f"[ariadne_norm] {task}: native target checkpoint -- the reference run skipped transport")
            del raw_sd
            continue
        source_base_model = copy.deepcopy(source_clf.model)
        source_ft_model = copy.deepcopy(source_clf.model)
        load_into_model(source_base_model, source_base_sd, strict=True)
        load_into_model(source_ft_model, source_base_sd, strict=True)
        load_into_model(source_ft_model, raw_sd, strict=False)
        del raw_sd

        if shared_ctx is not None:
            source_loader = shared_ctx.source_loaders.train
            target_loader = shared_ctx.loaders.train
            calibration_source = {"kind": shared_meta.get("calibration_data"), "meta": shared_meta}
        else:
            ctx = task_context(task)
            source_loader = select_loader(
                "train",
                train_loader=ctx.source_loaders.train,
                test_loader=ctx.source_loaders.test,
                val_loader=ctx.source_loaders.val,
            )
            target_loader = select_loader(
                "train",
                train_loader=ctx.loaders.train,
                test_loader=ctx.loaders.test,
                val_loader=ctx.loaders.val,
            )
            calibration_source = {"kind": "task_local", "split": "train"}
            del ctx

        vector_path = reference_vector_path(run_dir, task)
        reference_tv = None
        reference_note = "no saved vector under transported_tvs"
        if vector_path is not None and unit_vector_ok:
            reference_tv = torch.load(vector_path, map_location="cpu")
            reference_note = "saved vector is the unit vector (strength 1, tv_scaling none)"
        elif vector_path is not None:
            reference_note = (
                f"saved vector is scaled (strength={config.strength}, tv_scaling={config.tv_scaling!r}); not comparable"
            )

        rows = rows_by_task.get(task)
        provenance = {
            "reference_summary": str(summary_path),
            "reference_summary_sha256": summary_sha256,
            "reference_config_path": metadata.get("config_path"),
            "reference_code_fingerprint": metadata.get("code_fingerprint"),
            "current_code_fingerprint": current_fingerprint,
            "reference_target_hash_before": reference_target_hash,
            "target_base_sha256": target_hash,
            "target_hash_matches_reference": (reference_target_hash == target_hash)
            if reference_target_hash is not None
            else None,
            "tuned_checkpoint": ckpt_path,
            "source_clip": [source_cfg.model_name, source_cfg.pretrained],
            "target_clip": [target_cfg.model_name, target_cfg.pretrained],
            "run_seed": int(ref_cfg.get("seed", 42)),
            "calibration": {
                **calibration_source,
                "source_loader": _calibration_loader_identity(
                    source_loader, seed=config.seed, num_batches=config.num_batches
                ),
                "target_loader": _calibration_loader_identity(
                    target_loader, seed=config.seed, num_batches=config.num_batches
                ),
            },
            "generic": generic_provenance,
            "reference_task_vector": {
                "path": str(vector_path) if vector_path is not None else None,
                "note": reference_note,
                "sha256_of_tensors": _task_vector_sha256(reference_tv) if reference_tv is not None else None,
                "file_sha256": _file_sha256(vector_path) if vector_path is not None else None,
            },
            "patched_attn": patched,
            "pairing": {
                "depth_pairing": config.depth_pairing,
                "source_depth": source_depth,
                "target_depth": target_depth,
            },
        }
        yield DiagnosticInputs(
            family="vision",
            task=task,
            device=device,
            config=config,
            pairing=pairing,
            source_base_model=source_base_model,
            source_ft_model=source_ft_model,
            target_model=target_clf.model,
            target_base_state=target_base_sd,
            source_loader=source_loader,
            target_loader=target_loader,
            generic_target_batches=generic_batches,
            token_index=vision_token_index,
            component_key=component_key,
            source_component_key=component_key,
            alpha_star=(float(selected_alpha[task]) if task in selected_alpha else None),
            reference_task_vector=reference_tv,
            reference_task_vector_tolerance="exact",
            reference_rows=(strip_tensor_fields(rows) if rows is not None else None),
            provenance=provenance,
        )
        # Drop this task's references before the next ones are built (peak memory).
        del source_base_model, source_ft_model, source_loader, target_loader, reference_tv
