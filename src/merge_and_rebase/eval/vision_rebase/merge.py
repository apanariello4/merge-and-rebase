"""Merge-mode configuration and delta-space merge helpers for the vision rebase entrypoint."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from ...io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ...merge.base import PreparedMergeMethod
from ...merge.methods._common import axpy_state_dict
from ...merge.registry import get_method as get_merge_method
from ...merge.task_vectors import TaskVector
from ...rebase.block_extension.vision import run_block_extension
from ...rebase.merge_modes import (  # noqa: F401  (re-exported: same objects)
    _SINGLE_TRANSPORT_MODES,
    _TRANSPORT_THEN_MERGE_MODES,
    _VALID_MERGE_MODES,
    _resolve_merge_mode_config,
)
from ...rebase.prestep import StageEnv
from ..utils import to_cpu_fp32
from .context import (
    _build_balanced_calibration_context,
    _build_direct_paired_calibration_context,
    _select_dedicated_brace_loader,
    _TaskContext,
)
from .stages import _resolve_source_activation_plan, _visual_only_filter


def _scale_delta(delta_sd: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
    w = float(weight)
    if w == 1.0:
        return delta_sd
    return {k: (v * w) for k, v in delta_sd.items()}


def _check_untransported_compatibility(
    base_sd: dict[str, torch.Tensor],
    delta_sd: dict[str, torch.Tensor],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for k, d in delta_sd.items():
        b = base_sd.get(k)
        if b is None:
            issues.append(f"{k}: missing in target base")
            continue
        if tuple(d.shape) != tuple(b.shape):
            issues.append(f"{k}: delta shape {tuple(d.shape)} != target shape {tuple(b.shape)}")
    return (len(issues) == 0), issues


def _average_visual_state_dicts(
    states_by_task: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Average independently transformed visual endpoint bases.

    Only floating-point visual tensors present with identical shapes in every
    task endpoint participate. This is deliberately separate from task-vector
    construction: callers must form each ``ft_ind_t - base_ind_t`` first.
    """
    if not states_by_task:
        raise ValueError("Cannot average independent endpoint bases for an empty task set.")

    task_items = list(states_by_task.items())
    common_keys = set(task_items[0][1])
    for _, state in task_items[1:]:
        common_keys &= set(state)

    usable_keys: list[str] = []
    for key in sorted(common_keys):
        values = [state[key] for _, state in task_items]
        if not key.startswith("visual.") or any(not value.is_floating_point() for value in values):
            continue
        shape = tuple(values[0].shape)
        if any(tuple(value.shape) != shape for value in values[1:]):
            continue
        usable_keys.append(key)

    if not usable_keys:
        raise ValueError("Independent endpoint bases have no common floating-point visual tensors to average.")

    average: dict[str, torch.Tensor] = {}
    for key in usable_keys:
        values = [state[key].detach().to(device="cpu", dtype=torch.float32) for _, state in task_items]
        average[key] = torch.stack(values, dim=0).mean(dim=0)
    return average, usable_keys


def _relative_visual_state_distance(
    state: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    keys: Sequence[str],
) -> float:
    """Return ||state-reference||_2 / ||reference||_2 over selected tensors."""
    distance_sq = torch.zeros((), dtype=torch.float64)
    reference_sq = torch.zeros((), dtype=torch.float64)
    for key in keys:
        value = state[key].detach().to(device="cpu", dtype=torch.float64)
        ref = reference[key].detach().to(device="cpu", dtype=torch.float64)
        distance_sq += torch.sum((value - ref) ** 2)
        reference_sq += torch.sum(ref**2)
    denominator = float(torch.sqrt(reference_sq))
    if denominator == 0.0:
        return 0.0 if float(torch.sqrt(distance_sq)) == 0.0 else float("inf")
    return float(torch.sqrt(distance_sq)) / denominator


def _pseudo_tuned(
    base_sd: dict[str, torch.Tensor],
    deltas: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Synthesize tuned states ``base + delta_i`` as expected by MergeMethod APIs."""
    return [axpy_state_dict(base_sd, d, alpha=1.0) for d in deltas]


def _scale_deltas_by(
    deltas: list[dict[str, torch.Tensor]],
    alphas: Sequence[float],
) -> list[dict[str, torch.Tensor]]:
    """Scale each delta by its own alpha (bakes per-task alphas into the deltas)."""
    if len(deltas) != len(alphas):
        raise ValueError("deltas and alphas must have the same length.")
    out: list[dict[str, torch.Tensor]] = []
    for delta, alpha in zip(deltas, alphas, strict=True):
        a = float(alpha)
        if a == 1.0:
            out.append(delta)
        else:
            out.append({k: v * a for k, v in delta.items()})
    return out


def _visual_key_fingerprint(sd: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compact shape fingerprint of a checkpoint's visual backbone for diagnostics."""
    depths = [k for k in sd if ".resblocks." in k]
    block_ids = {int(k.split(".resblocks.")[1].split(".")[0]) for k in depths if ".resblocks." in k}
    visual_widths = sorted({int(v.shape[0]) for k, v in sd.items() if k.startswith("visual.") and v.dim() >= 1})
    return {
        "n_visual_keys": sum(1 for k in sd if k.startswith("visual.")),
        "max_block_id": max(block_ids) if block_ids else None,
        "n_blocks": len(block_ids),
        "visual_out_dims_sample": visual_widths[:4],
    }


def _ckpt_visual_base_coverage(
    sd: Mapping[str, torch.Tensor],
    base_sd: Mapping[str, torch.Tensor],
) -> float:
    """Fraction of the base's visual keys covered (key + shape) by sd after conservative alignment."""
    visual_base_keys = [k for k, v in base_sd.items() if k.startswith("visual.") and isinstance(v, torch.Tensor)]
    if not visual_base_keys:
        return 0.0
    aligned = align_to_base_keys(sd, base_sd)
    covered = sum(1 for k in visual_base_keys if k in aligned)
    return covered / len(visual_base_keys)


def _infer_ckpt_base(
    sd: Mapping[str, torch.Tensor],
    *,
    source_base_sd: Mapping[str, torch.Tensor],
    target_base_sd: Mapping[str, torch.Tensor],
    native_coverage_threshold: float = 0.5,
) -> str | None:
    """Classify a raw tuned checkpoint as 'source' or 'target' by visual-key coverage.

    Returns 'target' when the checkpoint matches the target visual backbone
    (a native target checkpoint), 'source' otherwise (ties prefer source so
    transport stays the default), or None when it matches neither base.
    """
    coverage_source = _ckpt_visual_base_coverage(sd, source_base_sd)
    coverage_target = _ckpt_visual_base_coverage(sd, target_base_sd)
    if coverage_target >= native_coverage_threshold and coverage_target > coverage_source:
        return "target"
    if coverage_source > 0.0:
        return "source"
    return None


def _merge_direction(
    *,
    base_sd: dict[str, torch.Tensor],
    deltas: list[dict[str, torch.Tensor]],
    merge_method_name: str,
    weights: Sequence[float],
    merge_params: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Merge task deltas into one direction in delta-space via the public merge registry.

    For PreparedMergeMethod implementations ``prepare`` already yields the
    weighted direction; plain methods are merged at alpha=1.0 and the base is
    subtracted back out. The result feeds ``rebased_deltas=[direction] * n`` so
    the existing per-task sweep machinery evaluates the same merged model.
    """
    merge_method = get_merge_method(merge_method_name)
    tuned = _pseudo_tuned(base_sd, deltas)
    method_kwargs = dict(merge_params)

    if isinstance(merge_method, PreparedMergeMethod):
        _prepared_base, direction = merge_method.prepare(
            base=base_sd,
            tuned=tuned,
            weights=list(weights),
            **method_kwargs,
        )
        return dict(direction)

    merged = merge_method.merge(base=base_sd, tuned=tuned, weights=list(weights), **method_kwargs)
    return {k: v - base_sd[k] for k, v in merged.items() if k in base_sd}


@dataclass
class MergePlan:
    """Deltas the alpha search evaluates, produced once per run by :func:`compose_rebased_deltas`."""

    rebased_deltas: list[dict[str, torch.Tensor]]
    untransported_deltas: list[dict[str, torch.Tensor]]
    #: Per-task untransported baseline feasibility (all False for every merge mode but ``none``).
    can_eval_untransported: list[bool]
    #: Individual (transported + native) deltas of the transport-then-merge modes, else ``None``.
    single_tv_deltas: list[dict[str, torch.Tensor]] | None
    #: Calibration provenance of the single-transport modes, else ``None``.
    calibration_metadata: dict[str, Any] | None
    #: The merged-then-transported delta of the single-transport modes (additive; saved under ``save_transported_tvs="auto"``).
    single_transport_delta: dict[str, torch.Tensor] | None = None


def compose_rebased_deltas(
    env: StageEnv,
    *,
    per_task: list[dict[str, Any]],
    transported_deltas: list[dict[str, torch.Tensor]],
    original_deltas: list[dict[str, torch.Tensor]],
    merge_weights: list[float],
    source_cfg: Any,
    target_cfg: Any,
) -> MergePlan:
    """Merge-mode dispatch: turn the per-task deltas into the per-task evaluation deltas.

    ``none`` scales the transported deltas by the task weights, the transport-then-merge modes compose
    them (or defer composition to the hierarchical pass 2), and the single-transport modes merge on the
    source side and transport once.
    """
    from .method_stages import _build_rebase_prepared  # lazy: method_stages imports this module

    resolved = env.resolved
    cfg = env.cfg
    tasks = resolved.tasks
    suite = resolved.suite
    merge_mode = resolved.merge.mode
    merge_method_name = resolved.merge.method_name
    merge_params = resolved.merge.params
    alpha_selection = resolved.alpha.selection
    method_name = resolved.method_name
    method = resolved.method
    method_params = resolved.method_params
    strict_load = resolved.strict_load
    grad_batch_size = resolved.grad_batch_size
    grad_imgs_per_class = resolved.grad_imgs_per_class
    grad_num_batches = resolved.grad_num_batches
    theseus_like_method = resolved.theseus_like_method
    bico_mode = resolved.bico_mode
    device = env.device
    clf_source = env.clf_source
    clf_target = env.clf_target
    tuned_by_task = env.tuned_by_task
    native_tasks = env.native_tasks
    source_base_sd = env.source_base_sd
    target_base_sd = env.target_base_sd
    transfusion_prepared = env.transfusion_prepared
    block_extension_cfg = resolved.block_extension_cfg
    block_extension_calibration_loader = env.block_extension_calibration_loader
    target_depth = env.target_depth
    run_logger = env.run_logger

    can_eval_untransported_by_task: list[bool] = []
    single_tv_deltas_for_diagnostic: list[dict[str, torch.Tensor]] | None = None
    single_transport_calibration_metadata: dict[str, Any] | None = None
    single_transport_delta: dict[str, torch.Tensor] | None = None
    if merge_mode == "none":
        rebased_deltas = [_scale_delta(d, w) for d, w in zip(transported_deltas, merge_weights, strict=True)]
        untransported_deltas = [_scale_delta(d, w) for d, w in zip(original_deltas, merge_weights, strict=True)]
        print(f"Prepared {len(tasks)} transported deltas (task-independent alpha mode)")

        for task_name, delta_sd in zip(tasks, untransported_deltas, strict=True):
            enabled, issues = _check_untransported_compatibility(target_base_sd, delta_sd)
            can_eval_untransported_by_task.append(enabled)
            if enabled:
                print(f"Untransported baseline for '{task_name}': enabled.")
            else:
                print(f"Untransported baseline for '{task_name}': skipped (incompatible with target model).")
                for msg in issues[:3]:
                    print(f"  - {msg}")
                if len(issues) > 3:
                    print(f"  - ... and {len(issues) - 3} more incompatibilities")
    elif merge_mode in _TRANSPORT_THEN_MERGE_MODES:
        native_delta_by_task: dict[str, dict[str, torch.Tensor]] = {}
        for task in sorted(native_tasks):
            path = str(tuned_by_task[task])
            sd = load_ckpt(path)
            aligned = align_to_base_keys(sd, target_base_sd)
            if not aligned:
                raise ValueError(
                    f"No tensors from native target checkpoint aligned to target base keys for task '{task}': {path}."
                )
            native_delta_by_task[task] = TaskVector.from_checkpoints(
                target_base_sd,
                to_cpu_fp32(aligned),
                strict=False,
                key_filter=_visual_only_filter,
            ).delta
            print(f"  {task}: native target delta computed ({len(native_delta_by_task[task])} params)")
            run_logger.log_event(
                "native_delta_end",
                metrics={f"rebase/{task}/native_param_count": float(len(native_delta_by_task[task]))},
                context={"task": task},
            )

        transported_iter = iter(transported_deltas)
        merge_input_deltas = []
        for t in tasks:
            if t in native_delta_by_task:
                merge_input_deltas.append(native_delta_by_task[t])
            else:
                merge_input_deltas.append(next(transported_iter))
        single_tv_deltas_for_diagnostic = list(merge_input_deltas)

        if alpha_selection == "per_task":
            # Hierarchical: per-task alphas are searched on the individual
            # deltas (transported + native) first; composition happens after pass 1.
            rebased_deltas = list(merge_input_deltas)
            untransported_deltas = original_deltas
            print(
                f"Hierarchical mode '{merge_mode}' ({merge_method_name}): per-task alpha "
                f"search on {len(merge_input_deltas)} deltas before merge"
            )
        else:
            merged_direction = _merge_direction(
                base_sd=target_base_sd,
                deltas=merge_input_deltas,
                merge_method_name=merge_method_name,
                weights=merge_weights,
                merge_params=merge_params,
            )
            print(
                f"Merge mode '{merge_mode}' ({merge_method_name}): composed {len(merge_input_deltas)} "
                f"deltas (transported + native) -> merged direction with {len(merged_direction)} params"
            )
            run_logger.log_event(
                "merge_composition_end",
                metrics={"merge/param_count": float(len(merged_direction))},
                context={
                    "mode": merge_mode,
                    "merge_method": merge_method_name,
                    "merge_params": merge_params,
                    "n_tasks": len(merge_input_deltas),
                    "n_native": len(native_delta_by_task),
                },
            )
            # One shared merged model: every task evaluates the same direction at alpha.
            rebased_deltas = [merged_direction] * len(tasks)
            untransported_deltas = original_deltas
    else:
        first_item = per_task[0]
        transport_protocol = str(cfg.get("transport_calibration_protocol", "task_local")).lower()
        calibration_metadata: dict[str, Any] = {"protocol": transport_protocol}
        single_transport_calibration_metadata = calibration_metadata
        if transport_protocol.startswith("tiny"):
            direct_spec = {
                "path": "zh-plus/tiny-imagenet",
                "split": "valid",
                "max_samples": int(cfg.get("transport_calibration_max_samples", 2048)),
            }
            transport_ctx = _build_direct_paired_calibration_context(
                direct_spec,
                suite=suite,
                cfg=cfg,
                clf_source=clf_source,
                clf_target=clf_target,
                source_cfg=source_cfg,
                target_cfg=target_cfg,
            )
        elif transport_protocol in {"vision8_mix", "vision8_mix_10", "task_local", "task_local_10"}:
            transport_ctx, balanced_meta = _build_balanced_calibration_context(
                per_task,
                cfg=cfg,
                clf_source=clf_source,
                clf_target=clf_target,
                n_batches=int(cfg.get("transport_calibration_batches", 10)),
                split=str(cfg.get("transport_calibration_split", "val")),
            )
            calibration_metadata.update(balanced_meta)
        else:
            raise ValueError(f"Unsupported transport_calibration_protocol: {transport_protocol!r}")

        if merge_mode == "merge_then_rebase":
            # Historical same-depth behavior is intentionally unchanged.
            transport_ctx = _TaskContext(
                loaders=first_item["loaders"],
                source_loaders=first_item["source_loaders"],
                classnames=list(first_item["classnames"]),
                build_cfg_task=first_item["build_cfg_task"],
                source_build_cfg_task=first_item["source_build_cfg_task"],
            )
            merged_source_base = source_base_sd
            merged_source_direction = _merge_direction(
                base_sd=source_base_sd,
                deltas=original_deltas,
                merge_method_name=merge_method_name,
                weights=merge_weights,
                merge_params=merge_params,
            )
            source_template_once = None
            prepared_has_brace = False
            merged_source_activation_plan = None
        elif merge_mode == "brace_merge_then_transport":
            if env.endpoints.corrected_source_template is None or not env.endpoints.base_by_task:
                raise RuntimeError("BRACE-then-merge requires corrected source endpoints for every task.")
            average_visual, average_keys = _average_visual_state_dicts(env.endpoints.base_by_task)
            first_base = env.endpoints.base_by_task[sorted(env.endpoints.base_by_task)[0]]
            merged_source_base = dict(first_base)
            merged_source_base.update(average_visual)
            merged_source_direction = _merge_direction(
                base_sd=merged_source_base,
                deltas=original_deltas,
                merge_method_name=merge_method_name,
                weights=merge_weights,
                merge_params=merge_params,
            )
            distances = {
                task: _relative_visual_state_distance(state, average_visual, average_keys)
                for task, state in env.endpoints.base_by_task.items()
            }
            calibration_metadata.update(
                {
                    "consensus_source_base": "mean_corrected_source_base",
                    "consensus_visual_key_count": len(average_keys),
                    "source_base_relative_distance_by_task": distances,
                    "source_base_max_relative_distance": max(distances.values()),
                }
            )
            source_template_once = env.endpoints.corrected_source_template
            prepared_has_brace = True
            merged_source_activation_plan = _resolve_source_activation_plan(
                block_extension_cfg, env.recorded_extension_layout
            )
        elif merge_mode == "merge_then_brace_then_transport":
            native_merged_direction = _merge_direction(
                base_sd=source_base_sd,
                deltas=original_deltas,
                merge_method_name=merge_method_name,
                weights=merge_weights,
                merge_params=merge_params,
            )
            source_base_model_once = deepcopy(clf_source.model)
            source_ft_model_once = deepcopy(clf_source.model)
            load_into_model(source_base_model_once, source_base_sd, strict=True)
            load_into_model(
                source_ft_model_once,
                axpy_state_dict(source_base_sd, native_merged_direction, alpha=1.0),
                strict=True,
            )
            # BRACE and transport have distinct calibration contracts and
            # must not share a bounded loader.  In particular, campaign
            # rows may request 40 BRACE batches but only 10 transport
            # batches.  The dedicated BRACE loader was constructed above
            # from block_extension_cfg; transport_ctx remains exclusively
            # owned by the subsequent transport preparation.
            brace_loader = _select_dedicated_brace_loader(
                brace_loader=block_extension_calibration_loader,
                transport_loader=transport_ctx.source_loaders.train,
                correction_enabled=not block_extension_cfg.skip_correction,
            )
            merged_extension_layout = {}
            final_depth = run_block_extension(
                source_base_model=source_base_model_once,
                source_ft_model=source_ft_model_once,
                calibration_loader=brace_loader,
                target_layers_total=target_depth,
                config=block_extension_cfg,
                device=device,
                layout_out=merged_extension_layout,
            )
            if final_depth != target_depth:
                raise RuntimeError(
                    f"Merged-pair BRACE depth mismatch: final_depth={final_depth}, target_depth={target_depth}."
                )
            merged_source_base = to_cpu_fp32(dict(source_base_model_once.state_dict()))
            merged_source_ft = to_cpu_fp32(dict(source_ft_model_once.state_dict()))
            merged_source_direction = TaskVector.from_checkpoints(
                merged_source_base,
                merged_source_ft,
                strict=True,
                key_filter=_visual_only_filter,
            ).delta
            source_template_once = deepcopy(source_base_model_once).cpu()
            prepared_has_brace = True
            merged_source_activation_plan = _resolve_source_activation_plan(
                block_extension_cfg, merged_extension_layout
            )
        else:  # pragma: no cover - validated by _resolve_merge_mode_config
            raise AssertionError(f"Unhandled merge mode: {merge_mode}")

        prepared_once = _build_rebase_prepared(
            method_name=method_name,
            method=method,
            method_params=method_params,
            cfg=cfg,
            device=device,
            grad_batch_size=grad_batch_size,
            grad_imgs_per_class=grad_imgs_per_class,
            grad_num_batches=grad_num_batches,
            theseus_like_method=theseus_like_method,
            bico_mode=bico_mode,
            run_block_extension_prestep=prepared_has_brace,
            clf_source=clf_source,
            clf_target=clf_target,
            classnames=list(transport_ctx.classnames),
            loaders=transport_ctx.loaders,
            source_loaders=transport_ctx.source_loaders,
            build_cfg_task=transport_ctx.build_cfg_task,
            source_build_cfg_task=transport_ctx.source_build_cfg_task,
            task_source_base_sd=merged_source_base,
            target_base_sd=target_base_sd,
            task_delta=merged_source_direction,
            source_base_model_task=source_template_once,
            transfusion_prepared=transfusion_prepared,
            source_text_features=transport_ctx.source_text_features,
            target_text_features=transport_ctx.target_text_features,
            source_activation_plan=merged_source_activation_plan,
        )
        transport_started = time.perf_counter()
        transported_merged_delta = method.transport(
            source_base=merged_source_base,
            target_base=target_base_sd,
            delta=merged_source_direction,
            strict=strict_load,
            prepared=prepared_once,
            **method_params,
        )
        transport_seconds = time.perf_counter() - transport_started
        print(
            f"Merge mode '{merge_mode}' ({merge_method_name}): merged on source -> single transport in "
            f"{transport_seconds:.2f}s ({len(transported_merged_delta)} params)"
        )
        run_logger.log_event(
            "merge_composition_end",
            metrics={
                "merge/param_count": float(len(transported_merged_delta)),
                "merge/single_transport_seconds": float(transport_seconds),
            },
            context={
                "mode": merge_mode,
                "merge_method": merge_method_name,
                "merge_params": merge_params,
                "n_tasks": len(original_deltas),
                "calibration": calibration_metadata,
            },
        )
        single_transport_delta = transported_merged_delta
        rebased_deltas = [transported_merged_delta] * len(tasks)
        untransported_deltas = original_deltas

    if merge_mode != "none":
        # The per-task "untransported" baseline is meaningless for a single
        # merged model; fall back to the (alpha-independent, cached) target
        # zero-shot baseline so normalized ratios remain defined.
        can_eval_untransported_by_task = [False] * len(tasks)

    return MergePlan(
        rebased_deltas=rebased_deltas,
        untransported_deltas=untransported_deltas,
        can_eval_untransported=can_eval_untransported_by_task,
        single_tv_deltas=single_tv_deltas_for_diagnostic,
        calibration_metadata=single_transport_calibration_metadata,
        single_transport_delta=single_transport_delta,
    )
