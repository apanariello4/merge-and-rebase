from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import resource
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

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
from ..data.balanced_calibration import (
    Vision8TaskContext,
    build_balanced_vision8_calibration_loaders,
)
from ..data.templates import get_templates
from ..data.vision_loaders import (
    build_vision_calibration_loader,
    build_vision_loaders,
    extract_classnames,
    load_hf_splits,
)
from ..eval.utils import (
    eval_task_top1,
    humanize,
    patch_base_for_attn,
    resolve_eval_split_loader,
    to_cpu_fp32,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model, resolve_ckpt_path
from ..io.peft_helpers import normalize_attn_patch_cfg
from ..merge.base import PreparedMergeMethod
from ..merge.methods._common import axpy_state_dict
from ..merge.registry import get_method as get_merge_method
from ..merge.registry import list_methods as list_merge_methods
from ..merge.task_vectors import TaskVector
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..rebase import get_method, list_methods
from ..rebase.discrete_layer_match import DiscreteLayerPairing, build_discrete_indexed_model
from ..rebase.methods.theseus import InterpolatedBlockActivations
from ..rebase.runtime import (
    format_rebase_method_label,
    resolve_rebase_method_config,
)
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ..utils.alpha_search import PerTaskAlphaTracker, average_scores
from .block_extension import (
    BlockExtensionConfig,
    block_extension_protocol,
    calibration_dataset_spec,
    resolve_block_extension_config,
    run_block_extension,
    select_loader,
)
from .datasets.vision8_14_20 import SUITES
from .direct_residual import (
    DirectResidualConfig,
    apply_tv_scaling,
    capture_paired_boundary_activations,
    compute_alignment_diagnostics,
    compute_alignment_diagnostics_streaming,
    compute_desired_effects,
    fit_direct_residual,
    fit_direct_residual_streaming,
    measure_streaming_realization_for,
    parse_direct_residual_config,
    prepare_direct_residual_streaming,
)
from .print_utils import pretty_print_task_accuracies
from .rebase_metrics import normalized_accuracy_ratio
from .target_informed_runtime import (
    capture_residual_references,
    capture_resized_joint_source_inputs,
    complete_direct_p1_shared_correction,
    complete_joint_blockwise,
    complete_residuals,
    complete_residuals_direct,
    compute_direct_residual_task_vector_stats,
    measure_direct_residual_realization,
    projection_transforms,
    scale_completion,
)
from .target_residual_completion import (
    JointCorrectionConfig,
    ResidualCompletionConfig,
    order_components,
    validate_residual_completion_depth_direction,
)

_ZERO_SHOT_CACHE_DIR = os.environ.get("BRACE_ZS_CACHE_DIR", "src/.cache/zs_cache")


def _set_deterministic_seed(seed: int) -> None:
    """Seed torch/cuda and force deterministic kernels for this run.

    BRACE's per-weight reference capture forwards the same pristine model
    through the same frozen calibration batches once per structural step
    (rather than once, globally) to bound memory. Without
    `cudnn.deterministic`/`use_deterministic_algorithms`, repeated forward
    passes over identical weights and inputs are not guaranteed bit-identical
    on GPU, and with a small ridge_weight the resulting ridge fit can amplify
    that noise into visible downstream accuracy drift between otherwise
    identical runs.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _legacy_visual_key(key: str) -> str | None:
    if not key.startswith("visual."):
        return None
    out = key[len("visual.") :]
    replacements = (
        (".attn.q_proj.", ".attn.q."),
        (".attn.k_proj.", ".attn.k."),
        (".attn.v_proj.", ".attn.v."),
        (".attn.out_proj.", ".attn.proj."),
        (".mlp.c_fc.", ".mlp.fc1."),
        (".mlp.c_proj.", ".mlp.fc2."),
    )
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def _legacy_visual_delta(delta: dict[str, torch.Tensor], *, drop_conv1: bool = False) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in delta.items():
        legacy_key = _legacy_visual_key(key)
        if legacy_key is None:
            continue
        if drop_conv1 and legacy_key == "conv1.weight":
            continue
        out[legacy_key] = value.detach().to(device="cpu", dtype=torch.float32)
    return out


def _visual_only_filter(k: str, v: torch.Tensor) -> bool:
    if not v.is_floating_point():
        return False
    if ".aligner." in k:
        return False
    return k.startswith("visual.")


def _evaluate_source_model_top1(
    *,
    model: torch.nn.Module,
    clf_source: OpenClipClassifier,
    loaders_obj: Any,
    classnames_task: list[str],
    source_build_cfg_task: OpenClipBuildConfig,
    split: str,
    first_n_batches: int | None,
    device: str,
) -> float:
    eval_clf = OpenClipClassifier(
        model=model,
        tokenizer=clf_source.tokenizer,
        preprocess=clf_source.preprocess,
        normalize=clf_source.normalize,
        logit_scale=clf_source.logit_scale,
    )
    eval_loader = resolve_eval_split_loader(loaders_obj, split)
    if first_n_batches is not None:
        eval_loader = itertools.islice(iter(eval_loader), max(1, int(first_n_batches)))

    eval_clf.build_zeroshot_text_features(
        classnames_task,
        source_build_cfg_task,
        cache_dir=_ZERO_SHOT_CACHE_DIR,
        force_rebuild=False,
    )
    return float(eval_clf.top1(eval_loader, device=device))


@torch.no_grad()
def _evaluate_source_lmc(
    *,
    model: torch.nn.Module,
    restore_sd: dict[str, torch.Tensor],
    endpoint_a_sd: dict[str, torch.Tensor],
    endpoint_b_sd: dict[str, torch.Tensor],
    clf_source: OpenClipClassifier,
    loaders_obj: Any,
    classnames_task: list[str],
    source_build_cfg_task: OpenClipBuildConfig,
    split: str,
    first_n_batches: int | None,
    alphas: list[float],
    device: str,
) -> dict[str, Any]:
    """Evaluate the parameter chord between two source endpoints.

    This is deliberately evaluated in the source architecture, before width
    transport, so a low barrier can be compared with downstream transport
    quality. The caller supplies ``restore_sd`` because the interpolation is
    loaded into a live model in order to avoid another full model copy.
    """
    if not alphas or not any(abs(float(a)) < 1e-8 for a in alphas) or not any(
        abs(float(a) - 1.0) < 1e-8 for a in alphas
    ):
        raise ValueError("source LMC alphas must include both 0 and 1")
    if set(endpoint_a_sd) != set(endpoint_b_sd):
        raise ValueError("source LMC endpoint keyspaces differ")

    eval_clf = OpenClipClassifier(
        model=model,
        tokenizer=clf_source.tokenizer,
        preprocess=clf_source.preprocess,
        normalize=clf_source.normalize,
        logit_scale=clf_source.logit_scale,
    )
    eval_clf.build_zeroshot_text_features(
        classnames_task,
        source_build_cfg_task,
        cache_dir=_ZERO_SHOT_CACHE_DIR,
        force_rebuild=False,
    )
    eval_loader = resolve_eval_split_loader(loaders_obj, split)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    eval_clf.to(dev)
    eval_clf.eval()

    accuracies: list[float] = []
    losses: list[float] = []
    try:
        for alpha in alphas:
            interpolated: dict[str, torch.Tensor] = {}
            for key, value_a in endpoint_a_sd.items():
                value_b = endpoint_b_sd[key]
                if torch.is_floating_point(value_a):
                    interpolated[key] = torch.lerp(value_a, value_b, float(alpha))
                else:
                    interpolated[key] = value_a
            load_into_model(model, interpolated, strict=False)
            del interpolated

            loader = eval_loader
            if first_n_batches is not None:
                loader = itertools.islice(iter(eval_loader), max(1, int(first_n_batches)))
            total = 0
            correct = 0
            loss_sum = 0.0
            for images, labels in loader:
                images = images.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True)
                logits = eval_clf(images)
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
                correct += int((logits.argmax(dim=-1) == labels).sum().item())
                total += int(labels.numel())
            accuracies.append(float(correct / max(1, total)))
            losses.append(float(loss_sum / max(1, total)))
            print(
                f"    source LMC alpha={float(alpha):.3f} "
                f"acc={accuracies[-1]:.6f} loss={losses[-1]:.6f}"
            )
    finally:
        load_into_model(model, restore_sd, strict=False)

    idx0 = min(range(len(alphas)), key=lambda i: abs(float(alphas[i])))
    idx1 = min(range(len(alphas)), key=lambda i: abs(float(alphas[i]) - 1.0))
    loss_chord = [
        (1.0 - float(alpha)) * losses[idx0] + float(alpha) * losses[idx1]
        for alpha in alphas
    ]
    errors = [1.0 - acc for acc in accuracies]
    error_chord = [
        (1.0 - float(alpha)) * errors[idx0] + float(alpha) * errors[idx1]
        for alpha in alphas
    ]
    in_unit = [i for i, alpha in enumerate(alphas) if 0.0 <= float(alpha) <= 1.0]
    loss_barriers = [losses[i] - loss_chord[i] for i in range(len(alphas))]
    error_barriers = [errors[i] - error_chord[i] for i in range(len(alphas))]
    max_loss_idx = max(in_unit, key=lambda i: loss_barriers[i])
    max_error_idx = max(in_unit, key=lambda i: error_barriers[i])
    min_loss_idx = min(in_unit, key=lambda i: loss_barriers[i])
    min_error_idx = min(in_unit, key=lambda i: error_barriers[i])

    def _area_below_chord(gaps: list[float]) -> float:
        """Trapezoidal integral of the amount the path lies below its chord."""
        return float(
            sum(
                max(0.0, -0.5 * (gaps[left] + gaps[right])) * (float(alphas[right]) - float(alphas[left]))
                for left, right in zip(in_unit, in_unit[1:], strict=False)
            )
        )

    return {
        "alphas": [float(a) for a in alphas],
        "accuracy": accuracies,
        "loss": losses,
        "loss_barrier_curve": loss_barriers,
        "error_barrier_curve": error_barriers,
        "max_loss_barrier": float(loss_barriers[max_loss_idx]),
        "max_loss_barrier_alpha": float(alphas[max_loss_idx]),
        "min_loss_chord_gap": float(loss_barriers[min_loss_idx]),
        "min_loss_chord_gap_alpha": float(alphas[min_loss_idx]),
        "mean_loss_chord_gap": float(sum(loss_barriers[i] for i in in_unit) / len(in_unit)),
        "area_below_loss_chord": _area_below_chord(loss_barriers),
        "max_error_barrier": float(error_barriers[max_error_idx]),
        "max_error_barrier_alpha": float(alphas[max_error_idx]),
        "min_error_chord_gap": float(error_barriers[min_error_idx]),
        "min_error_chord_gap_alpha": float(alphas[min_error_idx]),
        "mean_error_chord_gap": float(sum(error_barriers[i] for i in in_unit) / len(in_unit)),
        "area_below_error_chord": _area_below_chord(error_barriers),
        "split": split,
        "first_n_batches": first_n_batches,
    }


@torch.no_grad()
def _evaluate_cross_task_source_lmc(
    *,
    model: torch.nn.Module,
    restore_sd: dict[str, torch.Tensor],
    endpoint_a_sd: dict[str, torch.Tensor],
    endpoint_b_sd: dict[str, torch.Tensor],
    clf_source: OpenClipClassifier,
    task_contexts: list[dict[str, Any]],
    split: str,
    first_n_batches: int | None,
    alphas: list[float],
    device: str,
) -> dict[str, Any]:
    """Measure the source-space chord between two BRACE-corrected task endpoints."""
    if not alphas or not any(abs(float(a)) < 1e-8 for a in alphas) or not any(
        abs(float(a) - 1.0) < 1e-8 for a in alphas
    ):
        raise ValueError("cross-task source LMC alphas must include both 0 and 1")
    if set(endpoint_a_sd) != set(endpoint_b_sd):
        raise ValueError("cross-task source LMC endpoint keyspaces differ")
    if not task_contexts:
        raise ValueError("cross-task source LMC requires at least one evaluation task")

    eval_clf = OpenClipClassifier(
        model=model,
        tokenizer=clf_source.tokenizer,
        preprocess=clf_source.preprocess,
        normalize=clf_source.normalize,
        logit_scale=clf_source.logit_scale,
    )
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    eval_clf.to(dev)
    eval_clf.eval()

    text_features_by_task: dict[str, torch.Tensor] = {}
    for ctx in task_contexts:
        task = str(ctx["task"])
        eval_clf.build_zeroshot_text_features(
            ctx["classnames"],
            ctx["source_build_cfg_task"],
            cache_dir=_ZERO_SHOT_CACHE_DIR,
            force_rebuild=False,
        )
        if eval_clf._zs_text_features is None:
            raise RuntimeError(f"Could not build source text features for cross-task LMC task '{task}'.")
        text_features_by_task[task] = eval_clf._zs_text_features.detach().cpu()

    average_accuracy: list[float] = []
    average_loss: list[float] = []
    per_task_accuracy: dict[str, list[float]] = {str(ctx["task"]): [] for ctx in task_contexts}
    per_task_loss: dict[str, list[float]] = {str(ctx["task"]): [] for ctx in task_contexts}
    try:
        for alpha in alphas:
            interpolated = {
                key: (torch.lerp(value_a, endpoint_b_sd[key], float(alpha)) if torch.is_floating_point(value_a) else value_a)
                for key, value_a in endpoint_a_sd.items()
            }
            load_into_model(model, interpolated, strict=True)
            del interpolated

            alpha_accuracies: list[float] = []
            alpha_losses: list[float] = []
            for ctx in task_contexts:
                task = str(ctx["task"])
                eval_clf._zs_text_features = text_features_by_task[task].to(dev)
                loader = resolve_eval_split_loader(ctx["source_loaders"], split)
                if first_n_batches is not None:
                    loader = itertools.islice(iter(loader), max(1, int(first_n_batches)))
                total = 0
                correct = 0
                loss_sum = 0.0
                for images, labels in loader:
                    images = images.to(dev, non_blocking=True)
                    labels = labels.to(dev, non_blocking=True)
                    logits = eval_clf(images)
                    loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
                    correct += int((logits.argmax(dim=-1) == labels).sum().item())
                    total += int(labels.numel())
                acc = float(correct / max(1, total))
                loss = float(loss_sum / max(1, total))
                per_task_accuracy[task].append(acc)
                per_task_loss[task].append(loss)
                alpha_accuracies.append(acc)
                alpha_losses.append(loss)
            average_accuracy.append(float(sum(alpha_accuracies) / len(alpha_accuracies)))
            average_loss.append(float(sum(alpha_losses) / len(alpha_losses)))
            print(
                f"    cross-task source LMC alpha={float(alpha):.3f} "
                f"avg_acc={average_accuracy[-1]:.6f} avg_loss={average_loss[-1]:.6f}"
            )
    finally:
        load_into_model(model, restore_sd, strict=True)

    idx0 = min(range(len(alphas)), key=lambda i: abs(float(alphas[i])))
    idx1 = min(range(len(alphas)), key=lambda i: abs(float(alphas[i]) - 1.0))
    loss_chord = [
        (1.0 - float(alpha)) * average_loss[idx0] + float(alpha) * average_loss[idx1]
        for alpha in alphas
    ]
    loss_gaps = [average_loss[i] - loss_chord[i] for i in range(len(alphas))]
    in_unit = [i for i, alpha in enumerate(alphas) if 0.0 <= float(alpha) <= 1.0]
    per_task_loss_chord_gap = {
        task: [
            loss - ((1.0 - float(alpha)) * losses[idx0] + float(alpha) * losses[idx1])
            for alpha, loss in zip(alphas, losses, strict=True)
        ]
        for task, losses in per_task_loss.items()
    }
    per_task_max_loss_barrier = {
        task: float(max(gaps[i] for i in in_unit)) for task, gaps in per_task_loss_chord_gap.items()
    }
    max_gap_idx = max(in_unit, key=lambda i: loss_gaps[i])
    min_gap_idx = min(in_unit, key=lambda i: loss_gaps[i])
    area_below = sum(
        max(0.0, -0.5 * (loss_gaps[left] + loss_gaps[right])) * (float(alphas[right]) - float(alphas[left]))
        for left, right in zip(in_unit, in_unit[1:], strict=False)
    )
    return {
        "alphas": [float(alpha) for alpha in alphas],
        "average_accuracy": average_accuracy,
        "average_loss": average_loss,
        "per_task_accuracy": per_task_accuracy,
        "per_task_loss": per_task_loss,
        "per_task_loss_chord_gap": per_task_loss_chord_gap,
        "per_task_max_loss_barrier": per_task_max_loss_barrier,
        "max_per_task_loss_barrier": float(max(per_task_max_loss_barrier.values())),
        "loss_chord_gap": loss_gaps,
        "max_loss_barrier": float(loss_gaps[max_gap_idx]),
        "max_loss_barrier_alpha": float(alphas[max_gap_idx]),
        "min_loss_chord_gap": float(loss_gaps[min_gap_idx]),
        "min_loss_chord_gap_alpha": float(alphas[min_gap_idx]),
        "area_below_loss_chord": float(area_below),
        "split": split,
        "first_n_batches": first_n_batches,
    }


@torch.no_grad()
def _evaluate_all_task_star_lmc(
    *,
    model: torch.nn.Module,
    restore_sd: dict[str, torch.Tensor],
    endpoint_states: dict[str, dict[str, torch.Tensor]],
    clf_source: OpenClipClassifier,
    task_contexts: list[dict[str, Any]],
    split: str,
    first_n_batches: int | None,
    alphas: list[float],
    device: str,
) -> dict[str, Any]:
    """Sample every endpoint-to-uniform-barycenter ray of a task simplex."""
    tasks = list(endpoint_states)
    if len(tasks) < 2:
        raise ValueError("All-task simplex LMC requires at least two task endpoints.")
    keyspace = set(endpoint_states[tasks[0]])
    if any(set(endpoint_states[task]) != keyspace for task in tasks[1:]):
        raise ValueError("All-task simplex LMC endpoint keyspaces differ.")
    barycenter = {
        key: sum((endpoint_states[task][key].float() for task in tasks), start=torch.zeros_like(endpoint_states[tasks[0]][key], dtype=torch.float32))
        / len(tasks)
        if torch.is_floating_point(endpoint_states[tasks[0]][key])
        else endpoint_states[tasks[0]][key]
        for key in endpoint_states[tasks[0]]
    }
    rays = {
        task: _evaluate_cross_task_source_lmc(
            model=model,
            restore_sd=restore_sd,
            endpoint_a_sd=endpoint_states[task],
            endpoint_b_sd=barycenter,
            clf_source=clf_source,
            task_contexts=task_contexts,
            split=split,
            first_n_batches=first_n_batches,
            alphas=alphas,
            device=device,
        )
        for task in tasks
    }
    per_task = {
        task: max(ray["per_task_max_loss_barrier"][task] for ray in rays.values())
        for task in tasks
    }
    return {
        "tasks": tasks,
        "geometry": "all endpoint-to-uniform-barycenter simplex rays",
        "rays": rays,
        "per_task_max_loss_barrier": per_task,
        "max_per_task_loss_barrier": max(per_task.values()),
        "max_joint_loss_barrier": max(ray["max_loss_barrier"] for ray in rays.values()),
        "split": split,
        "first_n_batches": first_n_batches,
    }


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


def _norm_acc(result_acc: float, baseline_acc: float) -> float:
    return normalized_accuracy_ratio(result_acc, baseline_acc)


def _average_defined(values: list[float]) -> float:
    defined = [float(v) for v in values if float(v) == float(v)]
    return average_scores(defined) if defined else float("nan")


_BASE_CONSTRUCTION_MODES = ("per_task", "independent_endpoint_average")


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
        reference_sq += torch.sum(ref ** 2)
    denominator = float(torch.sqrt(reference_sq))
    if denominator == 0.0:
        return 0.0 if float(torch.sqrt(distance_sq)) == 0.0 else float("inf")
    return float(torch.sqrt(distance_sq)) / denominator


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
            "merge_then_brace_then_transport requires a dedicated BRACE "
            "calibration loader when correction is enabled."
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
        target_features.append(
            clf_target._compute_zeroshot_text_features(names, item["build_cfg_task"]).detach().cpu()
        )

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
            build_vision_loaders(preprocess=clf_source.preprocess, **loader_kwargs)
            if need_source_loaders
            else None
        ),
        classnames=classnames,
        build_cfg_task=_build_cfg_for(target_cfg),
        source_build_cfg_task=_build_cfg_for(source_cfg),
    )


_VALID_MERGE_MODES = (
    "none",
    "rebase_then_merge",
    "merge_then_rebase",
    "brace_transport_then_merge",
    "brace_merge_then_transport",
    "merge_then_brace_then_transport",
)
_TRANSPORT_THEN_MERGE_MODES = {"rebase_then_merge", "brace_transport_then_merge"}
_SINGLE_TRANSPORT_MODES = {
    "merge_then_rebase",
    "brace_merge_then_transport",
    "merge_then_brace_then_transport",
}


def _resolve_merge_mode_config(
    cfg: dict[str, Any],
    alpha_selection: str,
) -> tuple[str, str, dict[str, Any], bool]:
    """Resolve and validate the merge-mode knobs.

    Returns (merge_mode, merge_method_name, merge_params, global_alpha_search).
    ``merge_mode="none"`` keeps the historical per-task transfer evaluation;
    ``rebase_then_merge`` and its explicit campaign alias
    ``brace_transport_then_merge`` support hierarchical search (per-task alphas
    followed by a global merge alpha) when ``alpha_selection="per_task"``.
    """
    merge_mode = str(cfg.get("merge_mode", "none")).strip().lower()
    if merge_mode not in _VALID_MERGE_MODES:
        raise ValueError(f"merge_mode must be one of: {', '.join(_VALID_MERGE_MODES)}")

    if merge_mode not in _TRANSPORT_THEN_MERGE_MODES and merge_mode != "none" and alpha_selection == "per_task":
        raise ValueError(
            f"{merge_mode} requires alpha_selection='shared': per-task alpha search "
            "is only defined for individually transported deltas on the target base. "
            "Use merge_mode='brace_transport_then_merge' for hierarchical per-task alphas."
        )

    merge_method_name = str(cfg.get("merge_method", "task_arithmetic"))
    get_merge_method(merge_method_name)  # validate early for clearer UX

    raw_params = cfg.get("merge_params", {}) or {}
    if not isinstance(raw_params, dict):
        raise ValueError("merge_params must be a JSON object / mapping when provided.")

    global_alpha_search = cfg.get("global_alpha_search", True)
    if not isinstance(global_alpha_search, bool):
        raise ValueError("global_alpha_search must be a boolean (true/false).")
    return merge_mode, merge_method_name, dict(raw_params), global_alpha_search


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


def _state_dict_sha256(sd: Mapping[str, torch.Tensor]) -> str:
    """Stable CPU hash used to prove that the native target base was not mutated."""
    digest = hashlib.sha256()
    for key in sorted(sd):
        value = sd[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()))
    return digest.hexdigest()


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


def _resolve_source_activation_plan(
    block_extension_cfg: Any,
    extension_layout: Mapping[str, Any] | None,
) -> InterpolatedBlockActivations | None:
    """Build the interpolated-activation baseline plan, or ``None`` for ARIADNE.

    The plan is derived from the layout the extender actually realized rather
    than re-derived from the schedule, so it stays correct for insertion orders
    that draw source blocks at random.
    """
    if str(getattr(block_extension_cfg, "transport_activation_mode", "model")) == "model":
        return None
    if not extension_layout:
        raise RuntimeError(
            "transport_activation_mode='interpolate_neighbors' requires a recorded block "
            "extension layout; the extension prestep did not run."
        )
    return InterpolatedBlockActivations.from_extension_layout(extension_layout)


def _build_rebase_prepared(
    *,
    method_name: str,
    method: Any,
    method_params: dict[str, Any],
    cfg: dict[str, Any],
    device: str,
    grad_batch_size: int | None,
    grad_imgs_per_class: int | None,
    grad_num_batches: int | None,
    theseus_like_method: bool,
    bico_mode: bool,
    run_block_extension_prestep: bool,
    clf_source: OpenClipClassifier,
    clf_target: OpenClipClassifier,
    classnames: list[str],
    loaders: Any,
    source_loaders: Any,
    build_cfg_task: OpenClipBuildConfig,
    source_build_cfg_task: OpenClipBuildConfig,
    task_source_base_sd: dict[str, torch.Tensor],
    target_base_sd: dict[str, torch.Tensor],
    task_delta: dict[str, torch.Tensor],
    source_base_model_task: torch.nn.Module | None,
    transfusion_prepared: dict[str, Any] | None,
    source_text_features: torch.Tensor | None = None,
    target_text_features: torch.Tensor | None = None,
    source_activation_plan: Any | None = None,
) -> Any:
    """Compute the rebase method's prepared state for one task context.

    Shared by the per-task transport path and by merge_then_rebase's single
    post-composition transport. Note that theseus/bico use ``task_delta`` for
    key filtering and shape handling only — values never affect the prepared
    transforms.

    For Theseus and BiCo, ``seed`` controls the deterministic sampling of
    calibration batches used to fit the transport maps.  Default it to the
    run-level seed so a seed sweep actually changes those maps, while allowing
    ``method_params.seed`` to override it when map sampling must be decoupled
    from the validation/test split seed.
    """
    if method_name == "gradfix":
        from ..eval.utils import build_grad_dataloader
        from ..models.grad_recipes import clip_contrastive_recipe

        grad_loader = build_grad_dataloader(
            loaders.train,
            loaders.train.dataset,
            grad_batch_size=grad_batch_size,
            grad_imgs_per_class=grad_imgs_per_class,
            grad_num_batches=grad_num_batches,
            num_workers=int(cfg.get("num_workers", 6)),
            seed=int(cfg.get("seed", 42)),
        )
        recipe = clip_contrastive_recipe(
            clf_target,
            classnames,
            build_cfg_task,
            device=device,
            reduction="none" if str(method_params.get("vote", "mean")) in {"majority", "max"} else "mean",
        )
        return method.prepare(
            target_model=clf_target.model,
            target_dataloader=grad_loader,
            recipe=recipe,
            device=device,
            **method_params,
        )

    if source_activation_plan is not None and not (theseus_like_method or bico_mode):
        raise ValueError(
            "The interpolated-activation baseline only applies to the activation-aligned "
            f"transport methods; method '{method_name}' does not consume activations."
        )

    if theseus_like_method:
        theseus_params = dict(method_params)
        transport_seed = int(theseus_params.pop("seed", cfg.get("seed", 42)))
        if run_block_extension_prestep:
            if source_base_model_task is None:
                raise RuntimeError("Theseus block-extension preprocess requires the corrected source base model.")
            # BRACE changes the source depth.  Loading this state into the raw
            # source architecture non-strictly drops every inserted block.
            source_model_for_theseus = deepcopy(source_base_model_task)
            load_into_model(source_model_for_theseus, task_source_base_sd, strict=True)
        else:
            source_model_for_theseus = deepcopy(clf_source.model)
            load_into_model(source_model_for_theseus, task_source_base_sd, strict=True)
        target_model_for_theseus = deepcopy(clf_target.model)
        load_into_model(target_model_for_theseus, target_base_sd, strict=True)

        source_depth = len(source_model_for_theseus.visual.transformer.resblocks)
        target_depth = len(target_model_for_theseus.visual.transformer.resblocks)
        if source_depth != target_depth:
            raise ValueError(
                "Theseus calibration models must be depth-matched: "
                f"source_depth={source_depth}, target_depth={target_depth}"
            )

        # Corrected BRACE templates are intentionally retained on CPU to keep
        # the multi-task campaign's resident memory bounded.  Theseus sends
        # calibration inputs to ``device`` but does not own model placement,
        # so place only these isolated working copies immediately before
        # activation collection.
        source_model_for_theseus.to(device).eval()
        target_model_for_theseus.to(device).eval()

        # ``covariance_source`` other than the default fits the alignment on the
        # fine-tuned source endpoint as well as the base one.  That endpoint is
        # the corrected base plus the corrected task vector, which is the same
        # theta_ft_bar the transported task vector is defined against.
        source_ft_model_for_theseus: torch.nn.Module | None = None
        if str(theseus_params.get("covariance_source", "base")).strip().lower() not in {"base", "source_base", "source-base"}:
            if task_delta is None:
                raise RuntimeError("A non-default Theseus covariance_source requires the task delta.")
            source_ft_model_for_theseus = deepcopy(source_model_for_theseus)
            load_into_model(
                source_ft_model_for_theseus,
                axpy_state_dict(task_source_base_sd, task_delta, alpha=1.0),
                strict=True,
            )
            source_ft_model_for_theseus.to(device).eval()

        return method.prepare(
            source_model=source_model_for_theseus,
            target_model=target_model_for_theseus,
            source_model_ft=source_ft_model_for_theseus,
            source_dataloader=source_loaders.train,
            target_dataloader=loaders.train,
            target_base=target_base_sd,
            delta=task_delta,
            device=device,
            seed=transport_seed,
            source_activation_plan=source_activation_plan,
            **theseus_params,
        )

    if bico_mode:
        from ..models.grad_recipes import clip_contrastive_recipe

        bico_params = dict(method_params)
        transport_seed = int(bico_params.pop("seed", cfg.get("seed", 42)))

        if run_block_extension_prestep:
            if source_base_model_task is None:
                raise RuntimeError("BiCo block-extension preprocess requires the corrected source base model.")
            # BiCo moves models across devices and performs backward passes
            # while collecting statistics; use a strict-loaded copy so the
            # task's corrected endpoint remains immutable for later steps.
            source_model_for_bico = deepcopy(source_base_model_task)
            load_into_model(source_model_for_bico, task_source_base_sd, strict=True)
        else:
            source_model_for_bico = deepcopy(clf_source.model)
            load_into_model(source_model_for_bico, task_source_base_sd, strict=True)
        target_model_for_bico = deepcopy(clf_target.model)
        load_into_model(target_model_for_bico, target_base_sd, strict=True)

        source_depth = len(source_model_for_bico.visual.transformer.resblocks)
        target_depth = len(target_model_for_bico.visual.transformer.resblocks)
        if source_depth != target_depth:
            raise ValueError(
                "BiCo calibration models must be depth-matched: "
                f"source_depth={source_depth}, target_depth={target_depth}"
            )

        source_recipe = clip_contrastive_recipe(
            clf_source,
            classnames,
            source_build_cfg_task,
            device=device,
            text_features=source_text_features,
        )
        target_recipe = clip_contrastive_recipe(
            clf_target,
            classnames,
            build_cfg_task,
            device=device,
            text_features=target_text_features,
        )

        prepared = method.prepare(
            source_model=source_model_for_bico,
            target_model=target_model_for_bico,
            source_dataloader=source_loaders.train,
            target_dataloader=loaders.train,
            source_recipe=source_recipe,
            target_recipe=target_recipe,
            target_base=target_base_sd,
            delta=task_delta,
            device=device,
            seed=transport_seed,
            source_activation_plan=source_activation_plan,
            **bico_params,
        )
        del source_model_for_bico, target_model_for_bico, source_recipe, target_recipe
        return prepared

    return transfusion_prepared


def _maybe_capture_target_residual_references(
    *,
    config: ResidualCompletionConfig,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_loader: Any,
    target_loader: Any,
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    """Capture ARIADNE proposal-1 native reference banks, or no-op when disabled.

    Must be called before ``run_block_extension`` structurally resizes
    ``source_base_model``/``source_ft_model``: the native references are the
    un-resized source model's own boundary activations, paired against the
    pretrained target model at the doubled positions those source blocks will
    be inserted at. Returns ``None`` when the option is disabled, so callers
    that thread the result through unconditionally get a byte-identical no-op.
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
    )


def _maybe_complete_target_residual_task_vector(
    *,
    config: ResidualCompletionConfig,
    references: dict[str, Any] | None,
    prepared: Any,
    layout: Mapping[str, Any],
    target_model: torch.nn.Module,
    target_base_sd: Mapping[str, torch.Tensor],
    transported_delta: dict[str, torch.Tensor],
    target_loader: Any,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
    """Complete the transported task vector's residual-writing keys, or no-op.

    Which projections those are is ``config.components``: ``mlp.c_proj`` alone
    by default, optionally ``attn.out_proj`` as well in ``direct_target`` mode.

    Runs after transport is fitted, using the already-fitted ``prepared``
    transforms; it only ever adds to the *task vector*, never the target base
    weights. ``config.enabled=False`` or missing ``references`` (the option
    was disabled when references would have been captured) returns
    ``transported_delta`` completely unchanged -- same dict object, so a
    caller comparing state-dict hashes sees byte-identical output. Fitting
    always happens at gamma=1 (see ``complete_residuals``); ``config.strength``
    is applied afterwards by ``scale_completion``, and ``strength=0.0`` is a
    true null ablation because ``scale_completion`` short-circuits to the
    baseline in that case.
    """
    if not config.enabled or references is None:
        return transported_delta, None
    if config.mode == "direct_target":
        # Transport-free arm. There is no tau_t to complete: the caller has
        # already skipped the transport fit, so ``transported_delta`` must be
        # empty and the fitted correction is the entire task vector. The
        # baseline it is scaled against is therefore an explicit zero delta,
        # which keeps gamma=0 an exact native-target-base control (identical
        # semantics to the transport-aware arm's gamma=0).
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
        )
        zero_baseline = {key: torch.zeros_like(value) for key, value in target_corrections.items()}
        return scale_completion(zero_baseline, target_corrections, config.strength), diagnostics
    transforms = projection_transforms(prepared, layout, target_scope=config.target_scope)
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
    )
    completed_delta = scale_completion(transported_delta, target_corrections, config.strength)
    return completed_delta, diagnostics


def _maybe_complete_joint_blockwise_task_vector(
    *,
    config: JointCorrectionConfig,
    references: dict[str, Any] | None,
    prepared: Any,
    layout: Mapping[str, Any],
    target_model: torch.nn.Module,
    target_base_sd: Mapping[str, torch.Tensor],
    transported_delta: dict[str, torch.Tensor],
    target_loader: Any,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
    """Run the opt-in frozen-map Option 3 blockwise solve."""
    if not config.enabled or references is None:
        return transported_delta, None
    transforms = projection_transforms(prepared, layout, target_scope="inserted")
    _source, target_corrections, diagnostics = complete_joint_blockwise(
        target_model,
        target_base_sd,
        transported_delta,
        references,
        transforms,
        layout,
        target_loader,
        config=config,
        device=device,
    )
    completed = dict(transported_delta)
    for key, correction in target_corrections.items():
        if key in completed:
            completed[key] = completed[key] + correction.to(completed[key])
        else:
            completed[key] = correction
    return completed, diagnostics


def _maybe_complete_direct_p1_task_vector(
    *,
    config: JointCorrectionConfig,
    references: dict[str, Any] | None,
    prepared: Any,
    layout: Mapping[str, Any],
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    target_base_sd: Mapping[str, torch.Tensor],
    transported_delta: dict[str, torch.Tensor],
    source_loader: Any,
    target_loader: Any,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
    """Fit the direct shared-geometry/P1 correction with frozen maps."""
    if not config.enabled or references is None:
        return transported_delta, None
    transforms = projection_transforms(prepared, layout, target_scope="inserted")
    target_corrections, diagnostics = complete_direct_p1_shared_correction(
        source_base_model,
        source_ft_model,
        target_model,
        target_base_sd,
        transported_delta,
        references,
        transforms,
        layout,
        source_loader,
        target_loader,
        config=config,
        device=device,
    )
    completed = dict(transported_delta)
    for key, correction in target_corrections.items():
        if key in completed:
            completed[key] = completed[key] + correction.to(completed[key])
        else:
            completed[key] = correction
    return completed, diagnostics


def _run_direct_residual_fit(
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    target_base_sd: dict[str, torch.Tensor],
    source_loader: Any,
    target_loader: Any,
    pairing: DiscreteLayerPairing,
    config: DirectResidualConfig,
    device: str,
    clf_source: Any = None,
    clf_target: Any = None,
    classnames: list[str] | None = None,
    source_build_cfg_task: Any = None,
    build_cfg_task: Any = None,
    source_text_features: torch.Tensor | None = None,
    target_text_features: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Run Direct Residual's capture -> desired-effect -> fit -> scale pipeline once.

    Mirrors the exact ``torch.cuda.reset_peak_memory_stats()`` /
    ``torch.cuda.synchronize()`` / ``torch.cuda.max_memory_allocated()`` /
    ``time.perf_counter()`` idiom the existing ``transport_timings`` bracket
    uses, split into two brackets: one around alignment/capture
    (``capture_paired_boundary_activations`` + ``compute_desired_effects``,
    the "calibration" half) and one around the ridge solve itself
    (``fit_direct_residual``). ``theta_j_corrected = theta_j_native +
    strength * correction_j`` is applied here (not inside
    ``fit_direct_residual``, which always fits at unit strength) so that
    ``strength=0`` is an exact native-target-base control, matching
    ``target_informed_runtime.scale_completion``'s ``gamma=0`` contract --
    replicated directly rather than called through ``scale_completion``
    itself, since that helper requires every corrected key to already be
    present in its ``baseline`` argument, and Direct Residual's baseline is
    the empty ``transported_delta={}`` (there is no transport step to have
    populated it).

    Returns ``(scaled_delta, timing, diagnostics, extra)`` where ``timing`` has
    ``"alignment_calibration"``/``"correction_fit"`` sub-dicts, each shaped
    like a ``transport_timings[task]`` entry, and ``extra`` is
    ``{"realization_by_position": ..., "task_vector_stats": ...,
    "alignment_diagnostics": ...}``. The first two are ``None`` unless
    ``config.realization_diagnostics`` is set, and are computed from the
    unscaled, unit-strength ``target_corrections`` ``fit_direct_residual``
    returns -- i.e. before ``strength`` is applied -- matching the plan's
    "AFTER the task vector tau (unit strength, as returned by the fit) is
    assembled" requirement. ``alignment_diagnostics`` is always present (keyed
    by target position): it comes from a separate, untimed call to
    ``compute_alignment_diagnostics`` -- deliberately outside both the
    ``alignment_calibration`` and ``correction_fit`` timing/peak-memory
    brackets, so its own float64 recomputation cost never contaminates either
    (see the inline comment at the call site) -- is analysis-only, and never
    feeds any fit regardless of ``config.residual_target``. Neither
    realization-diagnostics call mutates ``target_model``'s entry state (both
    restore it internally and assert so via a state-dict hash).

    When ``config.procrustes_source == "gradient"``, builds source/target
    ``clip_contrastive_recipe`` gradient recipes exactly like the BiCo branch
    above (same classifier/classnames/build-cfg/text-features arguments) and
    passes them into ``capture_paired_boundary_activations`` so ``Q_j`` is
    fit on block-boundary gradients instead of activations; the six extra
    kwargs (``clf_source``, ``clf_target``, ``classnames``,
    ``source_build_cfg_task``, ``build_cfg_task``, ``source_text_features``/
    ``target_text_features``) are required only in that mode. The
    activation-vs-gradient Procrustes overlap diagnostic is merged into each
    position's diagnostics row by position.

    ``config.activation_storage`` branches between the resident path above
    (full per-batch banks, unchanged) and the streaming path
    (``prepare_direct_residual_streaming`` + ``fit_direct_residual_streaming``,
    O(1)-in-``num_batches`` host RAM); ``parse_direct_residual_config`` has
    already rejected any streaming config for which realization diagnostics
    would be reachable, so that block below only ever runs for the resident
    path. Both paths additionally record each bracket's peak host RSS
    (``resource.getrusage(resource.RUSAGE_SELF).ru_maxrss``, KiB on Linux) so
    campaigns can see streaming's host memory stay flat as ``num_batches``
    grows while resident's does not.
    """
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    alignment_started = time.perf_counter()
    streaming = config.activation_storage == "streaming"
    captured = None
    desired = None
    prepared = None
    gradient_mode = config.procrustes_source == "gradient"
    procrustes_diagnostics: dict[int, dict[str, Any]] = {}
    source_recipe = target_recipe = None
    if gradient_mode:
        missing = [
            name
            for name, value in (
                ("clf_source", clf_source),
                ("clf_target", clf_target),
                ("classnames", classnames),
                ("source_build_cfg_task", source_build_cfg_task),
                ("build_cfg_task", build_cfg_task),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"procrustes_source='gradient' requires {missing} to be provided")
        from ..models.grad_recipes import clip_contrastive_recipe

        source_recipe = clip_contrastive_recipe(
            clf_source,
            classnames,
            source_build_cfg_task,
            device=device,
            text_features=source_text_features,
        )
        target_recipe = clip_contrastive_recipe(
            clf_target,
            classnames,
            build_cfg_task,
            device=device,
            text_features=target_text_features,
        )
    if streaming:
        prepared = prepare_direct_residual_streaming(
            source_base_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            procrustes_source=config.procrustes_source,
            source_recipe=source_recipe,
            target_recipe=target_recipe,
        )
        procrustes_diagnostics.update(prepared["procrustes_diagnostics"])
    else:
        component_inputs = (
            order_components(config.components) if config.component_target != "block_boundary" else ()
        )
        captured = capture_paired_boundary_activations(
            source_base_model,
            source_ft_model,
            target_model,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            component_inputs=component_inputs,
            procrustes_source=config.procrustes_source,
            source_recipe=source_recipe,
            target_recipe=target_recipe,
            capture_source_ft_component_inputs=config.component_target == "output_total",
        )
        desired = compute_desired_effects(
            captured,
            pairing,
            residual_target=config.residual_target,
            procrustes_source=config.procrustes_source,
            diagnostics_out=procrustes_diagnostics if gradient_mode else None,
        )
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
        alignment_peak_memory_bytes = float(torch.cuda.max_memory_allocated())
    else:
        alignment_peak_memory_bytes = 0.0
    alignment_timing = {
        "alignment_calibration_seconds": time.perf_counter() - alignment_started,
        "alignment_calibration_peak_memory_bytes": alignment_peak_memory_bytes,
        "alignment_calibration_process_peak_host_rss_bytes": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }

    # Deliberately outside BOTH the alignment_calibration bracket above (just
    # closed) and the correction_fit bracket below (not yet opened):
    # compute_alignment_diagnostics recomputes the same centered Procrustes
    # fit a second time purely for analysis, with several float64 N x d_t
    # temporaries (N in the tens of thousands of rows). Folding it into
    # either bracket would inflate that bracket's recorded seconds/peak-
    # memory bytes -- even in the default residual_target="transported_delta"
    # path -- contaminating any cross-code-generation cost comparison for a
    # quantity these diagnostics never feed into.
    # compute_alignment_diagnostics describes the ACTIVATION-space Procrustes fit;
    # under procrustes_source="gradient" that is not the Q_j the fit used, so it
    # is not reported there (None) rather than reported for the wrong map.
    alignment_diagnostics = None
    if config.procrustes_source == "activation":
        if streaming:
            alignment_diagnostics = compute_alignment_diagnostics_streaming(
                source_base_model, source_ft_model, target_model, target_base_sd, prepared, pairing, device=device
            )
        else:
            alignment_diagnostics = compute_alignment_diagnostics(captured, pairing)

    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    fit_started = time.perf_counter()
    if streaming:
        target_corrections, diagnostics = fit_direct_residual_streaming(
            target_model,
            target_base_sd,
            source_base_model,
            source_ft_model,
            prepared,
            pairing,
            config=config,
            device=device,
        )
    else:
        target_corrections, diagnostics = fit_direct_residual(
            target_model,
            target_base_sd,
            captured,
            desired,
            pairing,
            config=config,
            device=device,
        )
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
        fit_peak_memory_bytes = float(torch.cuda.max_memory_allocated())
    else:
        fit_peak_memory_bytes = 0.0
    fit_timing = {
        "correction_fit_seconds": time.perf_counter() - fit_started,
        "correction_fit_peak_memory_bytes": fit_peak_memory_bytes,
        "correction_fit_process_peak_host_rss_bytes": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    if procrustes_diagnostics:
        for row in diagnostics:
            extra = procrustes_diagnostics.get(int(row.get("position", -1)))
            if extra:
                row.update(extra)

    streaming_measure = (
        measure_streaming_realization_for(
            target_model, target_base_sd, source_base_model, source_ft_model, prepared, pairing,
            config=config, device=device,
        )
        if streaming
        else None
    )
    tv_scaling_diagnostics = None
    if config.tv_scaling != "none":
        # Label-free, applied AFTER the unit-strength tau is assembled but
        # BEFORE the caller's per-task alpha-search (and therefore before the
        # realization_diagnostics/task_vector_stats block below, so both
        # report the FINAL tau that alpha-search actually sees). tv_scaling
        # defaults to "none" (a strict no-op, see apply_tv_scaling), so this
        # branch never executes for the historical, golden-hash-pinned path.
        target_corrections, tv_scaling_diagnostics = apply_tv_scaling(
            target_model,
            target_base_sd,
            target_corrections,
            list(range(pairing.target_depth)),
            captured,
            desired,
            config=config,
            device=device,
            measure_fn=streaming_measure,
        )

    realization_by_position = None
    task_vector_stats = None
    if bool(config.realization_diagnostics):
        positions = list(range(pairing.target_depth))
        # Explicit, never the broad CANONICAL_COMPONENT_ORDER default: packed
        # q/k/v share ONE physical state-dict key (attn.in_proj_weight), so a
        # presence check keyed only off "is this key in target_corrections"
        # cannot tell which of q/k/v were actually fit -- e.g. a v-only
        # output_local run's in_proj_weight key exists in target_corrections
        # with only its v-rows nonzero, and checking q/k against that same
        # key would falsely report them "present" too. Passing exactly
        # order_components(config.components) sidesteps this: only names the
        # caller actually asked to fit are ever checked.
        fitted_components = order_components(config.components)
        if streaming:
            realization_by_position = streaming_measure(target_corrections)
        else:
            realization_by_position = measure_direct_residual_realization(
                target_model,
                target_base_sd,
                target_corrections,
                positions,
                captured["target_batches"],
                captured["target_base_outputs_by_position"],
                desired,
                device=device,
                components=fitted_components,
            )
        task_vector_stats = compute_direct_residual_task_vector_stats(
            target_corrections,
            target_base_sd,
            positions,
            components=fitted_components,
        )

    strength = float(config.strength)
    scaled_delta = (
        {} if strength == 0.0 else {key: strength * correction for key, correction in target_corrections.items()}
    )
    extra = {
        "realization_by_position": realization_by_position,
        "task_vector_stats": task_vector_stats,
        "alignment_diagnostics": alignment_diagnostics,
        "tv_scaling": tv_scaling_diagnostics,
    }
    return scaled_delta, {"alignment_calibration": alignment_timing, "correction_fit": fit_timing}, diagnostics, extra


def main() -> None:
    run_logger = None
    try:
        p = argparse.ArgumentParser("Rebase task vectors from source base A to target base B and evaluate")

        add_config_arg(p)
        add_suite_arg(p, choices=sorted(SUITES.keys()))
        add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

        p.add_argument("--source-clip-model", type=str, default=None)
        p.add_argument("--source-clip-pretrained", type=str, default=None)
        p.add_argument("--target-clip-model", type=str, default=None)
        p.add_argument("--target-clip-pretrained", type=str, default=None)

        add_device_dtype_args(p, device_default=None, dtype_default=None)

        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--val-fraction", type=float, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--no-humanize", action="store_true", default=None, help="Use raw classnames.")

        p.add_argument("--tuned-ckpts", type=str, nargs="+", default=None)
        p.add_argument("--weights", type=float, nargs="*", default=None)
        p.add_argument("--strict-load", action="store_true", default=None)
        p.add_argument("--method", type=str, choices=list_methods(), default=None)
        p.add_argument("--method-params", type=str, default=None, help="JSON object for rebase-method kwargs.")
        p.add_argument(
            "--merge-mode",
            type=str,
            choices=list(_VALID_MERGE_MODES),
            default=None,
            help="Compose task deltas into one merged model instead of (or after) per-task transport.",
        )
        p.add_argument("--merge-method", type=str, choices=list_merge_methods(), default=None)
        p.add_argument("--merge-params", type=str, default=None, help="JSON object for merge-method kwargs.")

        p.add_argument("--mask-mode", type=str, default=None, choices=["normal", "force"])
        p.add_argument("--vote", type=str, default=None, choices=["mean", "majority", "max"])
        p.add_argument("--grad-batch-size", type=int, default=None)
        p.add_argument("--grad-imgs-per-class", type=int, default=None)
        p.add_argument("--grad-num-batches", type=int, default=None)

        add_alpha_args(
            p,
            alpha_default=None,
            alpha_min_default=None,
            alpha_max_default=None,
            alpha_step_default=None,
            alpha_search_default=None,
            alpha_search_help="Enable linear search over alpha.",
        )
        p.add_argument("--alpha-selection", type=str, choices=["shared", "per_task"], default=None)
        p.add_argument("--alpha-patience", type=int, default=None)
        p.add_argument("--alpha-search-split", type=str, default=None, choices=["val", "test"])
        p.add_argument("--save-merged", type=str, default=None)
        p.add_argument("--save-transported-tvs-dir", type=str, default=None)
        p.add_argument(
            "--eval-before-rebase",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Optionally evaluate source zero-shot/FT on target task dataset before rebase.",
        )
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
        add_logging_args(p)

        args = p.parse_args()
        method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")
        merge_params_cli = parse_json_object_arg(args.merge_params, arg_name="--merge-params")
        block_extension_params_cli = parse_json_object_arg(
            args.block_extension_params,
            arg_name="--block-extension-params",
        )

        cfg: dict[str, Any] = {}
        if args.config is not None:
            cfg = load_json(args.config)

        cli: dict[str, Any] = {
            "source_clip_model": args.source_clip_model,
            "source_clip_pretrained": args.source_clip_pretrained,
            "target_clip_model": args.target_clip_model,
            "target_clip_pretrained": args.target_clip_pretrained,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "no_humanize": args.no_humanize,
            "suite": getattr(args, "suite", None),
            "tasks": getattr(args, "tasks", None),
            "device": args.device,
            "dtype": args.dtype,
            "tuned_ckpts": args.tuned_ckpts,
            "weights": args.weights,
            "strict_load": args.strict_load,
            "method": args.method,
            "method_params": method_params_cli,
            "mask_mode": args.mask_mode,
            "vote": args.vote,
            "grad_batch_size": args.grad_batch_size,
            "grad_imgs_per_class": args.grad_imgs_per_class,
            "grad_num_batches": args.grad_num_batches,
            "alpha_search": getattr(args, "alpha_search", None),
            "alpha_selection": getattr(args, "alpha_selection", None),
            "merge_mode": args.merge_mode,
            "merge_method": args.merge_method,
            "merge_params": merge_params_cli,
            "alpha_patience": args.alpha_patience,
            "alpha_search_split": args.alpha_search_split,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_step": args.alpha_step,
            "alpha": args.alpha,
            "save_merged": args.save_merged,
            "save_transported_tvs_dir": args.save_transported_tvs_dir,
            "eval_before_rebase": args.eval_before_rebase,
            "block_extension_enabled": args.block_extension_enabled,
            "block_extension_params": block_extension_params_cli,
        }
        cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
        logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
        cfg["logging"] = logging_cfg
        _set_deterministic_seed(int(cfg.get("seed", 42)))

        if "block_extension_enabled" not in cfg:
            cfg["block_extension_enabled"] = True

        method_name, method_params = resolve_rebase_method_config(cfg)
        # Direct Residual is deliberately NOT registered in the rebase method
        # registry (rebase/registry.py): it needs whole model objects and
        # dataloaders for paired activation capture, not a state-dict-delta
        # transport() call, so `get_method` would raise KeyError for it. It
        # also must never let `resolve_block_extension_config(cfg)` run over
        # `cfg` -- ARIADNE's config gates have no business accepting or
        # rejecting a Direct Residual config, since Direct Residual never
        # reaches a single ARIADNE code path. See the module docstring of
        # `direct_residual.py` and tests/test_vision_rebase_direct_residual_dispatch.py.
        direct_residual_like = method_name == "direct_residual"
        if direct_residual_like:
            method = SimpleNamespace(name=method_name)
            block_extension_enabled = False
            block_extension_cfg = BlockExtensionConfig()
            # Direct Residual's own config schema is narrower than, and
            # independent of, `method_params` (which historically carries
            # per-transport-method kwargs consumed by `method.transport()` --
            # a call Direct Residual never makes). A dedicated top-level key
            # keeps that separation explicit rather than overloading
            # `method_params`'s existing per-method dispatch conventions.
            direct_residual_cfg = parse_direct_residual_config(cfg.get("direct_residual_params"))
        else:
            method = get_method(method_name)
            block_extension_enabled, block_extension_cfg = resolve_block_extension_config(cfg)
            direct_residual_cfg = None
        method_label = format_rebase_method_label(method_name, method_params)
        theseus_like_method = method_name in {"theseus", "theseus_reference"}
        blockext_like_method = method_name in {"theseus", "theseus_reference", "bico", "bico_gradin"}
        transfusion_mode = method_name == "transfusion"
        bico_mode = method_name in ("bico", "bico_gradin")
        # "ariadne" (default) preserves every existing behavior: the
        # block-extension prestep resizes the source model's depth via
        # ARIADNE's insertion/collapse machinery. "discrete_index_match" is
        # the faithful BiCo/THESEUS structural-resize control: it reindexes
        # the source model to the target depth via the flat, closed-form
        # `DiscreteLayerPairing` instead, with no interpolation, no
        # correction fit, and no ancestry bookkeeping. Resolved here, once,
        # so an unknown value fails fast rather than surfacing deep in the
        # per-task loop.
        depth_alignment_mode = str(cfg.get("depth_alignment", "ariadne")).strip().lower()
        if depth_alignment_mode not in {"ariadne", "discrete_index_match"}:
            raise ValueError("depth_alignment must be one of: ariadne, discrete_index_match")
        if (
            direct_residual_like
            and direct_residual_cfg.merge_mode == "merge_in_source_then_fit"
            and str(cfg.get("alpha_selection", "shared")).strip().lower() != "shared"
        ):
            # Mirrors the shared-alpha requirement _resolve_merge_mode_config
            # already enforces for merge_then_brace_then_transport: once every
            # task's transported delta collapses to the SAME once-fitted
            # correction, a per-task alpha search is degenerate (it would just
            # search the same objective under a different name per task).
            raise ValueError(
                "direct_residual merge_mode='merge_in_source_then_fit' requires alpha_selection='shared': "
                "the fit is performed once, on the merged source pair, and produces one correction shared "
                "by every task -- a per-task alpha search over an identical delta is not meaningful."
            )
        eval_before_rebase = bool(cfg.get("eval_before_rebase", False))
        block_extension_eval_requested = bool(eval_before_rebase)
        block_extension_eval_enabled = bool(block_extension_eval_requested and blockext_like_method)
        block_extension_eval_split = str(cfg.get("block_extension_eval_split", "test")).strip().lower()
        if block_extension_eval_split not in {"val", "test"}:
            raise ValueError("block_extension_eval_split must be one of: val, test")
        block_extension_eval_first_n_batches = block_extension_cfg.first_n_eval_batches
        source_lmc_eval = bool(cfg.get("source_lmc_eval", False))
        source_lmc_eval_split = str(cfg.get("source_lmc_eval_split", "val")).strip().lower()
        if source_lmc_eval_split not in {"val", "test"}:
            raise ValueError("source_lmc_eval_split must be one of: val, test")
        source_lmc_first_n_batches_raw = cfg.get("source_lmc_first_n_batches", None)
        source_lmc_first_n_batches = (
            int(source_lmc_first_n_batches_raw) if source_lmc_first_n_batches_raw is not None else None
        )
        source_lmc_alpha_min = float(cfg.get("source_lmc_alpha_min", 0.0))
        source_lmc_alpha_max = float(cfg.get("source_lmc_alpha_max", 1.0))
        source_lmc_alpha_step = float(cfg.get("source_lmc_alpha_step", 0.05))
        if source_lmc_alpha_step <= 0:
            raise ValueError("source_lmc_alpha_step must be > 0")
        source_lmc_alphas = torch.arange(
            source_lmc_alpha_min,
            source_lmc_alpha_max + source_lmc_alpha_step * 0.5,
            source_lmc_alpha_step,
        ).tolist()
        cross_task_lmc_pairs_raw = cfg.get("cross_task_lmc_pairs", [])
        if cross_task_lmc_pairs_raw is None:
            cross_task_lmc_pairs_raw = []
        if not isinstance(cross_task_lmc_pairs_raw, (list, tuple)):
            raise ValueError("cross_task_lmc_pairs must be a list of two-task lists.")
        cross_task_lmc_pairs: list[tuple[str, str]] = []
        for pair in cross_task_lmc_pairs_raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each cross_task_lmc_pairs item must contain exactly two task names.")
            task_a, task_b = str(pair[0]), str(pair[1])
            if task_a == task_b:
                raise ValueError("cross_task_lmc_pairs cannot interpolate a task with itself.")
            cross_task_lmc_pairs.append((task_a, task_b))
        cross_task_lmc_split = str(cfg.get("cross_task_lmc_eval_split", source_lmc_eval_split)).strip().lower()
        if cross_task_lmc_split not in {"val", "test"}:
            raise ValueError("cross_task_lmc_eval_split must be one of: val, test")
        all_task_lmc_tasks_raw = cfg.get("all_task_lmc_tasks", [])
        if all_task_lmc_tasks_raw is None:
            all_task_lmc_tasks_raw = []
        if not isinstance(all_task_lmc_tasks_raw, (list, tuple)):
            raise ValueError("all_task_lmc_tasks must be a list of task names.")
        all_task_lmc_tasks = [str(task) for task in all_task_lmc_tasks_raw]
        if all_task_lmc_tasks and (len(all_task_lmc_tasks) < 2 or len(set(all_task_lmc_tasks)) != len(all_task_lmc_tasks)):
            raise ValueError("all_task_lmc_tasks must contain at least two distinct task names.")
        all_task_lmc_split = str(cfg.get("all_task_lmc_eval_split", cross_task_lmc_split)).strip().lower()
        if all_task_lmc_split not in {"val", "test"}:
            raise ValueError("all_task_lmc_eval_split must be one of: val, test")
        source_only = bool(cfg.get("source_only", False))
        strict_load = bool(cfg.get("strict_load", False))
        device = str(cfg.get("device", "cuda"))

        if block_extension_eval_requested and not blockext_like_method:
            print(
                "Block-extension target-dataset eval: requested but skipped "
                f"(method='{method_name}' does not support block-extension)."
            )

        grad_batch_size = int(cfg["grad_batch_size"]) if cfg.get("grad_batch_size") is not None else None
        grad_imgs_per_class = int(cfg["grad_imgs_per_class"]) if cfg.get("grad_imgs_per_class") is not None else None
        grad_num_batches = int(cfg["grad_num_batches"]) if cfg.get("grad_num_batches") is not None else None

        alpha_search = bool(cfg.get("alpha_search", False))
        alpha_patience_raw = cfg.get("alpha_patience", 0)
        alpha_patience = int(alpha_patience_raw) if alpha_patience_raw is not None else 0
        if alpha_patience < 0:
            raise ValueError("alpha_patience must be >= 0")

        alpha_search_split = str(cfg.get("alpha_search_split", "val")).strip().lower()
        if alpha_search_split not in {"val", "test"}:
            raise ValueError("alpha_search_split must be one of: val, test")

        if alpha_search:
            a_min = float(cfg.get("alpha_min", 0.0))
            a_max = float(cfg.get("alpha_max", 2.0))
            a_step = float(cfg.get("alpha_step", 0.1))
            if a_step <= 0.0:
                raise ValueError("alpha_step must be > 0")
            if a_max < a_min:
                raise ValueError("alpha_max must be >= alpha_min")
            alphas = torch.arange(a_min, a_max + 1e-9, a_step).tolist()
        else:
            alphas = [float(cfg.get("alpha", 1.0))]

        alpha_selection = str(cfg.get("alpha_selection", "shared")).strip().lower()
        if alpha_selection not in {"shared", "per_task"}:
            raise ValueError("alpha_selection must be one of: shared, per_task")

        merge_mode, merge_method_name, merge_params, global_alpha_search = _resolve_merge_mode_config(cfg, alpha_selection)
        if direct_residual_like and merge_mode in _SINGLE_TRANSPORT_MODES:
            # merge_then_rebase / brace_merge_then_transport / merge_then_brace_then_transport
            # all end by calling method.transport() once on a merged direction
            # (see the merge-mode dispatch after the per-task loop). Direct
            # Residual has no such method object to call -- it is not in the
            # rebase method registry at all (see the method-dispatch comment
            # above) -- and its own once-only merge path is
            # `direct_residual_params.merge_mode='merge_in_source_then_fit'`,
            # which is independent of this top-level `merge_mode` key. Reject
            # the combination early rather than failing later with an
            # unhelpful AttributeError.
            raise ValueError(
                f"method='direct_residual' does not support merge_mode='{merge_mode}': Direct Residual has no "
                "transport() call for the merge-mode dispatch to invoke. Use merge_mode='none' (optionally with "
                "direct_residual_params.merge_mode='merge_in_source_then_fit' for a once-only merged fit) or "
                "merge_mode='rebase_then_merge'/'brace_transport_then_merge' for per-task fits merged afterward."
            )

        base_construction = str(cfg.get("base_construction", "per_task")).strip().lower()
        if base_construction not in _BASE_CONSTRUCTION_MODES:
            raise ValueError(
                "base_construction must be one of: " + ", ".join(_BASE_CONSTRUCTION_MODES)
            )

        positive_alphas = [float(alpha) for alpha in alphas if float(alpha) > 0.0]
        if alpha_search and not positive_alphas:
            raise ValueError("alpha_search requires at least one alpha > 0.")

        suite_name = cfg.get("suite", "vision8")
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
        suite = SUITES[suite_name]

        tasks_arg = cfg.get("tasks", "all")
        if tasks_arg == "all":
            tasks = list(suite.tasks)
        else:
            tasks = parse_csv(tasks_arg)
            bad = [t for t in tasks if t not in suite.tasks]
            if bad:
                raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite.tasks)}")

        run_summary_path = default_summary_path(
            entrypoint="eval.vision_rebase",
            logging_cfg=logging_cfg,
            default_parent=(Path(str(cfg["save_merged"])).parent if cfg.get("save_merged") else None),
        )
        run_logger = start_run(
            entrypoint="eval.vision_rebase",
            logging_cfg=logging_cfg,
            summary_path=run_summary_path,
            metadata={
                "config_path": args.config,
                "resolved_config": cfg,
                "suite": suite_name,
                "tasks": tasks,
                "summary_path": str(run_summary_path),
            },
        )

        tuned_by_task = cfg.get("tuned_ckpts", None)
        if tuned_by_task is not None:
            tuned_by_task = {t: resolve_ckpt_path(str(p)) for t, p in tuned_by_task.items()}
        if not tuned_by_task:
            raise ValueError("Provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")

        merge_weights = cfg.get("weights", None)
        if merge_weights is None:
            merge_weights = [1.0] * len(tasks)
        merge_weights = [float(w) for w in merge_weights]

        source_cfg = OpenClipBuildConfig(
            model_name=cfg.get("source_clip_model", "ViT-B-32"),
            pretrained=cfg.get("source_clip_pretrained", "openai"),
            device=device,
            dtype=cfg.get("dtype", None),
        )
        target_cfg = OpenClipBuildConfig(
            model_name=cfg.get("target_clip_model", "ViT-B-32"),
            pretrained=cfg.get("target_clip_pretrained", "laion2b_s34b_b79k"),
            device=device,
            dtype=cfg.get("dtype", None),
        )

        print(f"Source model (A): {source_cfg.model_name} / {source_cfg.pretrained}")
        print(f"Target model (B): {target_cfg.model_name} / {target_cfg.pretrained}")

        clf_source = OpenClipClassifier.build(source_cfg)
        clf_target = OpenClipClassifier.build(target_cfg)

        source_depth = int(len(clf_source.model.visual.transformer.resblocks))
        target_depth = int(len(clf_target.model.visual.transformer.resblocks))
        validate_residual_completion_depth_direction(
            block_extension_cfg.target_residual_completion,
            source_depth=source_depth,
            target_depth=target_depth,
        )
        run_block_extension_prestep = bool(
            blockext_like_method
            and block_extension_enabled
            and depth_alignment_mode == "ariadne"
            and source_depth != target_depth
        )
        run_discrete_layer_match_prestep = bool(
            blockext_like_method and depth_alignment_mode == "discrete_index_match" and source_depth != target_depth
        )
        # Direct-target P1 can write into a native target model at equal depth;
        # in that case it uses an identity layout instead of an ARIADNE resize.
        run_same_depth_direct_target = bool(
            blockext_like_method
            and block_extension_enabled
            and source_depth == target_depth
            and block_extension_cfg.target_residual_completion.enabled
            and block_extension_cfg.target_residual_completion.mode == "direct_target"
        )
        if depth_alignment_mode == "discrete_index_match" and (
            block_extension_cfg.target_residual_completion.enabled
            or block_extension_cfg.joint_blockwise_correction.enabled
            or block_extension_cfg.direct_p1_correction.enabled
        ):
            raise ValueError(
                "depth_alignment='discrete_index_match' is incompatible with target_residual_completion, "
                "joint_blockwise_correction, and direct_p1_correction."
            )
        if (
            block_extension_cfg.joint_blockwise_correction.enabled
            or block_extension_cfg.direct_p1_correction.enabled
        ):
            if not blockext_like_method:
                raise ValueError("Joint/direct P1 correction requires a Theseus- or BiCo-like transport method")
            if not run_block_extension_prestep:
                raise ValueError(
                    "Joint/direct P1 correction requires a depth-mismatched source/target pair "
                    "so that ARIADNE realizes inserted blocks"
                )
        if merge_mode == "merge_then_rebase" and run_block_extension_prestep:
            raise NotImplementedError(
                "merge_then_rebase does not support the block-extension prestep yet: "
                "per-task extended source bases live on different keyspaces and cannot be "
                "merged without a consensus-base step (see transport_then_merge). "
                "Use merge_mode='rebase_then_merge' for depth-mismatch pairs."
            )
        if blockext_like_method:
            calibration_dataset = calibration_dataset_spec(block_extension_cfg)
            if run_block_extension_prestep:
                print(
                    "Block extension preprocess: enabled "
                    f"(source_depth={source_depth} -> target_depth={target_depth}, "
                    f"split={block_extension_cfg.calibration_split}, "
                    f"dataset={calibration_dataset!r}, "
                    f"n_batches_act={block_extension_cfg.n_batches_act})."
                )
            else:
                reason = "disabled by config"
                if not block_extension_enabled:
                    reason = "disabled by config"
                elif source_depth == target_depth:
                    reason = "source/target depth already match"
                print(
                    "Block extension preprocess: skipped "
                    f"({reason}, source_depth={source_depth}, target_depth={target_depth})."
                )

        block_extension_calibration_loader = None
        calibration_dataset = calibration_dataset_spec(block_extension_cfg)
        if (
            run_block_extension_prestep
            and not block_extension_cfg.skip_correction
            and calibration_dataset is not None
        ):
            block_extension_calibration_loader = build_vision_calibration_loader(
                calibration_dataset,
                resolver=suite.resolver,
                preprocess=clf_source.preprocess,
                calibration_split=block_extension_cfg.calibration_split,
                batch_size=int(cfg.get("batch_size", 128)),
                num_workers=int(cfg.get("num_workers", 6)),
                pin_memory=True,
                val_fraction=float(cfg.get("val_fraction", 0.1)),
                seed=int(cfg.get("seed", 42)),
            )
            print(
                "Block extension preprocess: using one task-independent calibration loader "
                f"from {calibration_dataset!r}."
            )

        attn_patch_cfg_raw = cfg.get("attn_patch_cfg", None)
        if attn_patch_cfg_raw is not None and not isinstance(attn_patch_cfg_raw, dict):
            raise ValueError("config['attn_patch_cfg'] must be a dict when provided.")
        patch_attn_before_rebase = bool(cfg.get("patched_attn", attn_patch_cfg_raw is not None))
        attn_patch_cfg = normalize_attn_patch_cfg(attn_patch_cfg_raw) if patch_attn_before_rebase else None

        if patch_attn_before_rebase:
            print(f"Patching source/target attention before rebase: {attn_patch_cfg}")
            source_base_sd = to_cpu_fp32(
                patch_base_for_attn(
                    clf=clf_source,
                    base_ckpt=None,
                    strict_load=strict_load,
                    attn_patch_cfg=attn_patch_cfg,
                )
            )
            target_base_sd = to_cpu_fp32(
                patch_base_for_attn(
                    clf=clf_target,
                    base_ckpt=None,
                    strict_load=strict_load,
                    attn_patch_cfg=attn_patch_cfg,
                )
            )
        else:
            source_base_sd = to_cpu_fp32({k: v for k, v in clf_source.model.state_dict().items()})
            target_base_sd = to_cpu_fp32({k: v for k, v in clf_target.model.state_dict().items()})
        target_hash_before = _state_dict_sha256(target_base_sd)

        use_humanized_classnames = not bool(cfg.get("no_humanize", True))
        print(f"Classname mode: {'humanized' if use_humanized_classnames else 'raw'}")
        print(f"Rebase method: {method_label}")

        # ---- Mixed-merging pre-pass: classify each tuned checkpoint by its base ----
        native_tasks_requested = [str(t) for t in (cfg.get("native_target_tasks", []) or [])]
        unknown_native_tasks = [t for t in native_tasks_requested if t not in tasks]
        if unknown_native_tasks:
            raise ValueError(f"native_target_tasks contains tasks not in the task list: {unknown_native_tasks}")
        auto_detect_ckpt_base = bool(cfg.get("auto_detect_ckpt_base", True))
        native_tasks: set[str] = set(native_tasks_requested)

        if native_tasks or auto_detect_ckpt_base:
            print("Checkpoint base classification (visual-key coverage vs source/target):")
            for task in tasks:
                if task in native_tasks:
                    print(f"  {task}: native target checkpoint (explicit)")
                    continue
                raw_sd = load_ckpt(str(tuned_by_task[task]))
                inferred = _infer_ckpt_base(raw_sd, source_base_sd=source_base_sd, target_base_sd=target_base_sd)
                if inferred is None:
                    fingerprint = _visual_key_fingerprint(raw_sd)
                    raise ValueError(
                        f"Tuned checkpoint for task '{task}' matches neither the source nor the target "
                        f"visual backbone ({tuned_by_task[task]}). Checkpoint fingerprint: {fingerprint}. "
                        f"Source fingerprint: {_visual_key_fingerprint(source_base_sd)}. "
                        f"Target fingerprint: {_visual_key_fingerprint(target_base_sd)}."
                    )
                if inferred == "target":
                    if not auto_detect_ckpt_base:
                        raise ValueError(
                            f"Tuned checkpoint for task '{task}' matches the target architecture; "
                            "add it to native_target_tasks or set auto_detect_ckpt_base=true."
                        )
                    native_tasks.add(task)
                    print(f"  {task}: native target checkpoint (auto-detected)")
                else:
                    if strict_load:
                        coverage = _ckpt_visual_base_coverage(raw_sd, source_base_sd)
                        if coverage != 1.0:
                            raise ValueError(
                                f"Strict visual checkpoint coverage failed for task '{task}': "
                                f"coverage={coverage:.6f}, expected=1.0 ({tuned_by_task[task]})."
                            )
                    print(f"  {task}: source checkpoint (transport required)")
                del raw_sd

        if native_tasks:
            if merge_mode == "none":
                raise ValueError(
                    "Native target checkpoints require a merge mode; merge_mode='none' evaluates "
                    "per-task transported deltas only. Use merge_mode='rebase_then_merge'."
                )
            if merge_mode in _SINGLE_TRANSPORT_MODES:
                raise ValueError(
                    "Native target checkpoints cannot participate in merge_then_rebase: the merge "
                    "happens on the source base, where native target deltas do not exist."
                )
            if transfusion_mode:
                raise NotImplementedError(
                    "Native target checkpoints with transfusion are not supported: the permutation "
                    "prepare step swaps the target keyspace. Use a theseus/bico transport method."
                )

        if base_construction == "independent_endpoint_average":
            if merge_mode not in _TRANSPORT_THEN_MERGE_MODES:
                raise ValueError(
                    "base_construction='independent_endpoint_average' requires "
                    "merge_mode='brace_transport_then_merge'."
                )
            if alpha_selection != "shared":
                raise ValueError(
                    "base_construction='independent_endpoint_average' requires "
                    "alpha_selection='shared'; per-task alpha search is not part of this baseline."
                )
            if native_tasks:
                raise ValueError(
                    "base_construction='independent_endpoint_average' requires every task to be "
                    "an independently transformed source endpoint; native target tasks are not allowed."
                )

        task_context_by_name: dict[str, _TaskContext] = {}
        per_task: list[dict[str, Any]] = []
        transported_deltas: list[dict[str, torch.Tensor]] = []
        original_deltas: list[dict[str, torch.Tensor]] = []
        transport_timings: dict[str, dict[str, float]] = {}
        residual_completion_diagnostics: dict[str, list[dict[str, Any]]] = {}
        joint_blockwise_diagnostics: dict[str, list[dict[str, Any]]] = {}
        direct_p1_diagnostics: dict[str, list[dict[str, Any]]] = {}
        direct_residual_diagnostics: dict[str, list[dict[str, Any]]] = {}
        direct_residual_realization: dict[str, dict[int, dict[str, Any]]] = {}
        direct_residual_task_vector_stats: dict[str, dict[str, Any]] = {}
        direct_residual_alignment_diagnostics: dict[str, dict[int, dict[str, float]]] = {}
        direct_residual_tv_scaling: dict[str, dict[str, Any] | None] = {}
        transported_artifacts: dict[str, list[str]] = {}
        block_extension_eval_rows: list[dict[str, Any]] = []
        source_lmc_rows: list[dict[str, Any]] = []
        cross_task_lmc_rows: list[dict[str, Any]] = []
        all_task_lmc_rows: list[dict[str, Any]] = []
        corrected_ft_states: dict[str, dict[str, torch.Tensor]] = {}
        corrected_ft_templates: dict[str, torch.nn.Module] = {}
        independent_base_by_task: dict[str, dict[str, torch.Tensor]] = {}
        # Last realized per-task extension layout. The insertion schedule
        # depends only on the source/target depths, so every task shares it;
        # the merged-pair transport paths reuse it to address inserted blocks.
        recorded_extension_layout: dict[str, Any] = {}
        independent_ft_by_task: dict[str, dict[str, torch.Tensor]] = {}
        independent_base_average: dict[str, torch.Tensor] | None = None
        independent_base_distance_by_task: dict[str, float] = {}
        independent_base_dispersion: float | None = None
        independent_base_diagnostics_path: str | None = None
        independent_source_merge_param_count: int | None = None
        independent_direct_delta_key_count: dict[str, int] = {}
        corrected_source_template: torch.nn.Module | None = None
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
        transfusion_prepared: dict[str, Any] | None = None
        task_block_extension_prestep = bool(
            run_block_extension_prestep and merge_mode != "merge_then_brace_then_transport"
        )
        # Same merge_mode-aware guard as task_block_extension_prestep above:
        # merge_then_brace_then_transport merges deltas on the native source
        # base first and only then runs its own once-only structural step
        # (see the merge_then_brace_then_transport branch further below), so
        # neither prestep fires per-task under that merge mode.
        task_discrete_layer_match_prestep = bool(
            run_discrete_layer_match_prestep and merge_mode != "merge_then_brace_then_transport"
        )
        # Timing/memory brackets (wandb-visible), parallel to transport_timings:
        # alignment_calibration_timings covers whatever depth/width-alignment
        # step runs before any correction is fitted (build_discrete_indexed_model
        # for the discrete-index-match control, or capture_paired_boundary_activations
        # for Direct Residual); correction_fit_timings covers fit_direct_residual
        # only. Both dicts default to {} and are always present in final_summary,
        # even for methods/paths that never populate them, so downstream JSON
        # parsing is uniform across every method.
        alignment_calibration_timings: dict[str, dict[str, float]] = {}
        correction_fit_timings: dict[str, dict[str, float]] = {}

        # merge_in_source_then_fit (a DirectResidualConfig field, distinct
        # from the top-level `merge_mode` cfg key): merge every task's native
        # source-base-relative delta ONCE, before the per-task loop, then
        # capture/fit Direct Residual's correction ONCE against that merged
        # source pair. The per-task loop below reuses this single cached
        # correction for every task instead of re-fitting -- it is the same
        # transported_delta for every task, exactly as
        # merge_then_brace_then_transport reuses one merged/transported model
        # across tasks (see that branch further below for the analogous
        # native-source merge-then-structural-step pattern this mirrors).
        direct_residual_merged_correction: dict[str, torch.Tensor] | None = None
        direct_residual_merged_timing: dict[str, dict[str, float]] | None = None
        if direct_residual_like and direct_residual_cfg.merge_mode == "merge_in_source_then_fit":
            merge_in_source_tasks = [t for t in tasks if t not in native_tasks]
            merge_in_source_deltas: list[dict[str, torch.Tensor]] = []
            merge_in_source_weights: list[float] = []
            for idx, t in enumerate(tasks):
                if t in native_tasks:
                    continue
                ckpt_path = str(tuned_by_task[t])
                sd = load_ckpt(ckpt_path)
                aligned = align_to_base_keys(sd, source_base_sd)
                if not aligned:
                    raise ValueError(
                        f"No tensors from tuned checkpoint aligned to source base keys for task '{t}': {ckpt_path}."
                    )
                tuned_sd = to_cpu_fp32(aligned)
                delta = TaskVector.from_checkpoints(
                    source_base_sd, tuned_sd, strict=False, key_filter=_visual_only_filter
                ).delta
                merge_in_source_deltas.append(delta)
                merge_in_source_weights.append(merge_weights[idx])
            if not merge_in_source_tasks:
                raise ValueError("direct_residual merge_in_source_then_fit requires at least one non-native task.")
            merged_direction = _merge_direction(
                base_sd=source_base_sd,
                deltas=merge_in_source_deltas,
                merge_method_name=merge_method_name,
                weights=merge_in_source_weights,
                merge_params=merge_params,
            )
            source_base_model_merged = deepcopy(clf_source.model)
            source_ft_model_merged = deepcopy(clf_source.model)
            load_into_model(source_base_model_merged, source_base_sd, strict=True)
            load_into_model(
                source_ft_model_merged,
                axpy_state_dict(source_base_sd, merged_direction, alpha=1.0),
                strict=True,
            )
            # Direct Residual has no dedicated calibration-dataset config
            # field of its own (unlike block_extension_cfg's calibration_dataset):
            # the once-only merged fit has no single "task" to draw loaders
            # from, so it falls back to the first contributing task's train
            # loaders, mirroring the existing calibration-loader fallback
            # pattern (`calibration_loader = ...; if None: select_loader(...)`)
            # used elsewhere in this function when no dedicated loader is set.
            merged_calibration_task_ctx = task_context_by_name[merge_in_source_tasks[0]]
            direct_residual_pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
            (
                direct_residual_merged_correction,
                direct_residual_merged_timing,
                direct_residual_merged_diag,
                direct_residual_merged_extra,
            ) = _run_direct_residual_fit(
                source_base_model=source_base_model_merged,
                source_ft_model=source_ft_model_merged,
                target_model=clf_target.model,
                target_base_sd=target_base_sd,
                source_loader=merged_calibration_task_ctx.source_loaders.train,
                target_loader=merged_calibration_task_ctx.loaders.train,
                pairing=direct_residual_pairing,
                config=direct_residual_cfg,
                device=device,
                clf_source=clf_source,
                clf_target=clf_target,
                classnames=merged_calibration_task_ctx.classnames,
                source_build_cfg_task=merged_calibration_task_ctx.source_build_cfg_task,
                build_cfg_task=merged_calibration_task_ctx.build_cfg_task,
                source_text_features=merged_calibration_task_ctx.source_text_features,
                target_text_features=merged_calibration_task_ctx.target_text_features,
            )
            for t in merge_in_source_tasks:
                direct_residual_diagnostics[t] = direct_residual_merged_diag
                direct_residual_realization[t] = direct_residual_merged_extra["realization_by_position"]
                direct_residual_task_vector_stats[t] = direct_residual_merged_extra["task_vector_stats"]
                direct_residual_alignment_diagnostics[t] = direct_residual_merged_extra["alignment_diagnostics"]
                direct_residual_tv_scaling[t] = direct_residual_merged_extra["tv_scaling"]

        for task in tasks:
            task_ctx = task_context_by_name[task]
            loaders = task_ctx.loaders
            classnames = task_ctx.classnames
            build_cfg_task = task_ctx.build_cfg_task
            source_build_cfg_task = task_ctx.source_build_cfg_task
            source_loaders = task_ctx.source_loaders

            if task in native_tasks:
                print(f"  {task}: native target checkpoint — skipping transport")
                continue

            task_source_base_sd = source_base_sd
            task_residual_references: dict[str, Any] | None = None
            task_residual_target_loader: Any = None
            task_joint_references: dict[str, Any] | None = None
            task_joint_target_loader: Any = None
            task_direct_p1_references: dict[str, Any] | None = None
            task_direct_p1_target_loader: Any = None
            task_source_activation_plan: InterpolatedBlockActivations | None = None
            task_extension_layout: dict[str, Any] = {}

            source_base_model_task: torch.nn.Module | None = None
            source_ft_model_task: torch.nn.Module | None = None
            if blockext_like_method and (
                task_block_extension_prestep
                or task_discrete_layer_match_prestep
                or run_same_depth_direct_target
                or block_extension_eval_enabled
            ):
                source_base_model_task = deepcopy(clf_source.model)
                source_ft_model_task = deepcopy(clf_source.model)
                load_into_model(source_base_model_task, source_base_sd, strict=True)
                load_into_model(source_ft_model_task, source_base_sd, strict=True)
                load_into_model(source_ft_model_task, load_ckpt(str(tuned_by_task[task])), strict=False)

            if block_extension_eval_enabled and source_loaders is not None:
                eval_row: dict[str, Any] = {
                    "task": task,
                    "split": block_extension_eval_split,
                    "first_n_batches": (
                        int(block_extension_eval_first_n_batches)
                        if block_extension_eval_first_n_batches is not None
                        else None
                    ),
                    "extension_applied": bool(task_block_extension_prestep),
                }
                if source_base_model_task is None or source_ft_model_task is None:
                    raise RuntimeError("Block-extension eval requested but source task models were not initialized.")

                if task_block_extension_prestep:
                    zero_pre = _evaluate_source_model_top1(
                        model=source_base_model_task,
                        clf_source=clf_source,
                        loaders_obj=source_loaders,
                        classnames_task=classnames,
                        source_build_cfg_task=source_build_cfg_task,
                        split=block_extension_eval_split,
                        first_n_batches=block_extension_eval_first_n_batches,
                        device=device,
                    )
                    ft_pre = _evaluate_source_model_top1(
                        model=source_ft_model_task,
                        clf_source=clf_source,
                        loaders_obj=source_loaders,
                        classnames_task=classnames,
                        source_build_cfg_task=source_build_cfg_task,
                        split=block_extension_eval_split,
                        first_n_batches=block_extension_eval_first_n_batches,
                        device=device,
                    )
                    eval_row["zero_shot_pre"] = float(zero_pre)
                    eval_row["ft_pre"] = float(ft_pre)
                else:
                    zero_curr = _evaluate_source_model_top1(
                        model=source_base_model_task,
                        clf_source=clf_source,
                        loaders_obj=source_loaders,
                        classnames_task=classnames,
                        source_build_cfg_task=source_build_cfg_task,
                        split=block_extension_eval_split,
                        first_n_batches=block_extension_eval_first_n_batches,
                        device=device,
                    )
                    ft_curr = _evaluate_source_model_top1(
                        model=source_ft_model_task,
                        clf_source=clf_source,
                        loaders_obj=source_loaders,
                        classnames_task=classnames,
                        source_build_cfg_task=source_build_cfg_task,
                        split=block_extension_eval_split,
                        first_n_batches=block_extension_eval_first_n_batches,
                        device=device,
                    )
                    eval_row["zero_shot"] = float(zero_curr)
                    eval_row["ft"] = float(ft_curr)

                block_extension_eval_rows.append(eval_row)

            source_lmc_row: dict[str, Any] | None = None
            if source_lmc_eval and task_block_extension_prestep:
                if source_loaders is None or source_base_model_task is None or source_ft_model_task is None:
                    raise RuntimeError("Source LMC evaluation requires initialized source models and loaders.")
                source_lmc_row = {
                    "task": task,
                    "lmc_mode": block_extension_cfg.lmc_mode,
                }
                source_pre_base_sd = to_cpu_fp32(
                    {key: value for key, value in source_base_model_task.state_dict().items()}
                )
                source_pre_ft_sd = to_cpu_fp32(
                    {key: value for key, value in source_ft_model_task.state_dict().items()}
                )
                print(f"  {task}: evaluating source LMC before block extension")
                source_lmc_row["before_brace"] = _evaluate_source_lmc(
                    model=source_base_model_task,
                    restore_sd=source_pre_base_sd,
                    endpoint_a_sd=source_pre_base_sd,
                    endpoint_b_sd=source_pre_ft_sd,
                    clf_source=clf_source,
                    loaders_obj=source_loaders,
                    classnames_task=classnames,
                    source_build_cfg_task=source_build_cfg_task,
                    split=source_lmc_eval_split,
                    first_n_batches=source_lmc_first_n_batches,
                    alphas=source_lmc_alphas,
                    device=device,
                )

            if task_block_extension_prestep:
                if source_loaders is None:
                    raise ValueError("Block extension preprocess requires source_loaders for calibration.")
                if source_base_model_task is None or source_ft_model_task is None:
                    raise RuntimeError("Block extension preprocess expected initialized source task models.")

                calibration_loader = block_extension_calibration_loader
                if calibration_loader is None:
                    calibration_loader = select_loader(
                        block_extension_cfg.calibration_split,
                        train_loader=source_loaders.train,
                        test_loader=source_loaders.test,
                        val_loader=source_loaders.val,
                    )
                if block_extension_cfg.target_residual_completion.enabled:
                    # ARIADNE proposal 1: capture the native reference banks
                    # (source base/ft boundary activations, paired against the
                    # pretrained target model) BEFORE block extension resizes
                    # source_base_model_task/source_ft_model_task in place.
                    # These are the un-transported, un-inserted references the
                    # completion step later regresses each inserted block's
                    # c_proj projection against.
                    task_residual_target_loader = select_loader(
                        block_extension_cfg.calibration_split,
                        train_loader=loaders.train,
                        test_loader=loaders.test,
                        val_loader=loaders.val,
                    )
                    task_residual_references = _maybe_capture_target_residual_references(
                        config=block_extension_cfg.target_residual_completion,
                        source_base_model=source_base_model_task,
                        source_ft_model=source_ft_model_task,
                        target_model=clf_target.model,
                        source_loader=calibration_loader,
                        target_loader=task_residual_target_loader,
                        seed=int(cfg.get("seed", 42)),
                        device=device,
                    )
                if block_extension_cfg.joint_blockwise_correction.enabled:
                    task_joint_target_loader = select_loader(
                        block_extension_cfg.calibration_split,
                        train_loader=loaders.train,
                        test_loader=loaders.test,
                        val_loader=loaders.val,
                    )
                    task_joint_references = capture_residual_references(
                        source_base_model_task,
                        source_ft_model_task,
                        clf_target.model,
                        calibration_loader,
                        task_joint_target_loader,
                        num_batches=block_extension_cfg.n_batches_act,
                        seed=int(cfg.get("seed", 42)),
                        device=device,
                        capture_joint=True,
                    )
                if block_extension_cfg.direct_p1_correction.enabled:
                    task_direct_p1_target_loader = select_loader(
                        block_extension_cfg.calibration_split,
                        train_loader=loaders.train,
                        test_loader=loaders.test,
                        val_loader=loaders.val,
                    )
                    task_direct_p1_references = capture_residual_references(
                        source_base_model_task,
                        source_ft_model_task,
                        clf_target.model,
                        calibration_loader,
                        task_direct_p1_target_loader,
                        num_batches=block_extension_cfg.n_batches_act,
                        seed=int(cfg.get("seed", 42)),
                        device=device,
                        capture_joint=True,
                    )

                task_extension_layout = {}
                final_depth = run_block_extension(
                    source_base_model=source_base_model_task,
                    source_ft_model=source_ft_model_task,
                    calibration_loader=calibration_loader,
                    target_layers_total=target_depth,
                    config=block_extension_cfg,
                    device=device,
                    layout_out=task_extension_layout,
                    # Only the target-informed correction option reads this; every
                    # standard ARIADNE path leaves the target backbone untouched.
                    target_model=(
                        clf_target.model
                        if block_extension_cfg.target_shared_correction is not None
                        else None
                    ),
                )
                recorded_extension_layout = dict(task_extension_layout)
                task_source_activation_plan = _resolve_source_activation_plan(
                    block_extension_cfg, task_extension_layout
                )
                if final_depth != target_depth:
                    raise RuntimeError(
                        f"Block extension preprocess failed for task '{task}': "
                        f"final_depth={final_depth}, expected={target_depth}."
                    )
                if block_extension_cfg.joint_blockwise_correction.enabled:
                    task_joint_references = capture_resized_joint_source_inputs(
                        source_base_model_task,
                        calibration_loader,
                        task_joint_references,
                        task_extension_layout,
                        device=device,
                    )

                task_source_base_sd = to_cpu_fp32({k: v for k, v in source_base_model_task.state_dict().items()})
                task_source_ft_sd = to_cpu_fp32({k: v for k, v in source_ft_model_task.state_dict().items()})
                task_delta = TaskVector.from_checkpoints(
                    task_source_base_sd,
                    task_source_ft_sd,
                    strict=True,
                    key_filter=_visual_only_filter,
                ).delta
                if merge_mode == "brace_merge_then_transport" or base_construction == "independent_endpoint_average":
                    independent_base_by_task[task] = task_source_base_sd
                if base_construction == "independent_endpoint_average":
                    independent_ft_by_task[task] = task_source_ft_sd
                if merge_mode == "brace_merge_then_transport" and corrected_source_template is None:
                    corrected_source_template = deepcopy(source_base_model_task).cpu()

                if cross_task_lmc_pairs or all_task_lmc_tasks:
                    corrected_ft_states[task] = task_source_ft_sd
                    corrected_ft_templates[task] = deepcopy(source_ft_model_task).cpu()

                if source_lmc_row is not None:
                    print(f"  {task}: evaluating source LMC after block extension")
                    source_lmc_row["after_brace"] = _evaluate_source_lmc(
                        model=source_base_model_task,
                        restore_sd=task_source_base_sd,
                        endpoint_a_sd=task_source_base_sd,
                        endpoint_b_sd=task_source_ft_sd,
                        clf_source=clf_source,
                        loaders_obj=source_loaders,
                        classnames_task=classnames,
                        source_build_cfg_task=source_build_cfg_task,
                        split=source_lmc_eval_split,
                        first_n_batches=source_lmc_first_n_batches,
                        alphas=source_lmc_alphas,
                        device=device,
                    )
                    source_lmc_rows.append(source_lmc_row)
                    run_logger.log_event(
                        "source_lmc",
                        metrics={
                            f"source_lmc/{task}/before/max_loss_barrier": source_lmc_row["before_brace"][
                                "max_loss_barrier"
                            ],
                            f"source_lmc/{task}/after/max_loss_barrier": source_lmc_row["after_brace"][
                                "max_loss_barrier"
                            ],
                            f"source_lmc/{task}/before/max_error_barrier": source_lmc_row["before_brace"][
                                "max_error_barrier"
                            ],
                            f"source_lmc/{task}/after/max_error_barrier": source_lmc_row["after_brace"][
                                "max_error_barrier"
                            ],
                        },
                        context={"task": task, "lmc_mode": block_extension_cfg.lmc_mode},
                    )

                if block_extension_eval_enabled:
                    zero_post = _evaluate_source_model_top1(
                        model=source_base_model_task,
                        clf_source=clf_source,
                        loaders_obj=source_loaders,
                        classnames_task=classnames,
                        source_build_cfg_task=source_build_cfg_task,
                        split=block_extension_eval_split,
                        first_n_batches=block_extension_eval_first_n_batches,
                        device=device,
                    )
                    ft_post = _evaluate_source_model_top1(
                        model=source_ft_model_task,
                        clf_source=clf_source,
                        loaders_obj=source_loaders,
                        classnames_task=classnames,
                        source_build_cfg_task=source_build_cfg_task,
                        split=block_extension_eval_split,
                        first_n_batches=block_extension_eval_first_n_batches,
                        device=device,
                    )
                    last_row = block_extension_eval_rows[-1]
                    last_row["zero_shot_post"] = float(zero_post)
                    last_row["ft_post"] = float(ft_post)
                    print(
                        f"  {task}: source target-dataset eval "
                        f"zero_shot {last_row['zero_shot_pre']:.6f}->{zero_post:.6f} "
                        f"ft {last_row['ft_pre']:.6f}->{ft_post:.6f}"
                    )
                    run_logger.log_event(
                        "block_extension_eval",
                        metrics={
                            f"block_extension/eval/{task}/zero_shot_pre": float(last_row["zero_shot_pre"]),
                            f"block_extension/eval/{task}/zero_shot_post": float(zero_post),
                            f"block_extension/eval/{task}/ft_pre": float(last_row["ft_pre"]),
                            f"block_extension/eval/{task}/ft_post": float(ft_post),
                        },
                        context=last_row,
                    )
                print(
                    f"  {task}: block extension preprocess completed "
                    f"(source_depth={source_depth} -> {final_depth}, delta_keys={len(task_delta)})."
                )
            elif run_same_depth_direct_target:
                if source_loaders is None or source_base_model_task is None or source_ft_model_task is None:
                    raise RuntimeError("Same-depth direct-target P1 requires source models and calibration loaders.")
                calibration_loader = select_loader(
                    block_extension_cfg.calibration_split,
                    train_loader=source_loaders.train,
                    test_loader=source_loaders.test,
                    val_loader=source_loaders.val,
                )
                task_residual_target_loader = select_loader(
                    block_extension_cfg.calibration_split,
                    train_loader=loaders.train,
                    test_loader=loaders.test,
                    val_loader=loaders.val,
                )
                task_residual_references = _maybe_capture_target_residual_references(
                    config=block_extension_cfg.target_residual_completion,
                    source_base_model=source_base_model_task,
                    source_ft_model=source_ft_model_task,
                    target_model=clf_target.model,
                    source_loader=calibration_loader,
                    target_loader=task_residual_target_loader,
                    seed=int(cfg.get("seed", 42)),
                    device=device,
                )
                task_extension_layout = {
                    "direction": "extend",
                    "final_blocks": [
                        {
                            "position": pos,
                            "source_orig_idx": pos,
                            "span_orig_idxs": [pos],
                            "block_kind": "original",
                        }
                        for pos in range(target_depth)
                    ],
                    "inserted_blocks": [],
                }
                recorded_extension_layout = dict(task_extension_layout)
            elif task_discrete_layer_match_prestep:
                if source_base_model_task is None or source_ft_model_task is None:
                    raise RuntimeError("Discrete layer match expected initialized source task models.")
                if torch.cuda.is_available() and device != "cpu":
                    torch.cuda.reset_peak_memory_stats()
                alignment_started = time.perf_counter()
                pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
                source_base_model_task = build_discrete_indexed_model(source_base_model_task, pairing)
                source_ft_model_task = build_discrete_indexed_model(source_ft_model_task, pairing)
                if torch.cuda.is_available() and device != "cpu":
                    torch.cuda.synchronize()
                    alignment_peak_memory_bytes = float(torch.cuda.max_memory_allocated())
                else:
                    alignment_peak_memory_bytes = 0.0
                alignment_calibration_timings[task] = {
                    "alignment_calibration_seconds": time.perf_counter() - alignment_started,
                    "alignment_calibration_peak_memory_bytes": alignment_peak_memory_bytes,
                }
                task_source_base_sd = to_cpu_fp32(source_base_model_task.state_dict())
                task_source_ft_sd = to_cpu_fp32(source_ft_model_task.state_dict())
                task_delta = TaskVector.from_checkpoints(
                    task_source_base_sd, task_source_ft_sd, strict=True, key_filter=_visual_only_filter
                ).delta
                print(
                    f"  {task}: discrete layer match reindex completed "
                    f"(source_depth={source_depth} -> {target_depth}, delta_keys={len(task_delta)})."
                )
            elif block_extension_eval_enabled and block_extension_eval_rows:
                last_row = block_extension_eval_rows[-1]
                print(
                    f"  {task}: source target-dataset eval "
                    f"zero_shot={last_row['zero_shot']:.6f} ft={last_row['ft']:.6f}"
                )
                run_logger.log_event(
                    "block_extension_eval",
                    metrics={
                        f"block_extension/eval/{task}/zero_shot": float(last_row["zero_shot"]),
                        f"block_extension/eval/{task}/ft": float(last_row["ft"]),
                    },
                    context=last_row,
                )

            if source_only:
                continue

            if not task_block_extension_prestep and not task_discrete_layer_match_prestep:
                if transfusion_mode:
                    if transfusion_prepared is None:
                        transfusion_prepared = method.prepare(
                            clf_source=clf_source,
                            clf_target=clf_target,
                            source_loaders=source_loaders,
                            classnames=classnames,
                            source_build_cfg=source_build_cfg_task,
                            device=device,
                            seed=int(cfg.get("seed", 42)),
                            **method_params,
                        )
                        source_base_sd = transfusion_prepared["source_base_sd"]
                        target_base_sd = transfusion_prepared["target_base_sd"]
                        target_hash_before = _state_dict_sha256(target_base_sd)
                        clf_target.model = transfusion_prepared["target_model_patched"]
                        if transfusion_prepared.get("sanity_check_pre") is not None:
                            print(
                                f"  TransFusion perm sanity (once): "
                                f"{transfusion_prepared['sanity_check_pre']:.6f} -> "
                                f"{transfusion_prepared['sanity_check_post']:.6f} "
                                f"(delta={transfusion_prepared['sanity_check_post'] - transfusion_prepared['sanity_check_pre']:+.6f})"
                            )
                            run_logger.log_event(
                                "transfusion_perm_sanity",
                                metrics={
                                    f"transfusion/{task}/source_zeroshot": float(transfusion_prepared["sanity_check_pre"]),
                                    f"transfusion/{task}/permuted_zeroshot": float(transfusion_prepared["sanity_check_post"]),
                                    f"transfusion/{task}/perm_delta": float(
                                        transfusion_prepared["sanity_check_post"] - transfusion_prepared["sanity_check_pre"]
                                    ),
                                },
                                context={"task": task},
                            )

                    tuned_sd = method.load_task_checkpoint(
                        str(tuned_by_task[task]),
                        transfusion_prepared["source_model_unpatched"],
                    )
                    task_delta = method.compute_task_delta(tuned_sd, source_base_sd)
                else:
                    ckpt_path = str(tuned_by_task[task])
                    sd = load_ckpt(ckpt_path)
                    aligned = align_to_base_keys(sd, source_base_sd)
                    if not aligned:
                        raise ValueError(
                            f"No tensors from tuned checkpoint aligned to source base keys for task '{task}': {ckpt_path}. "
                            f"{'The base model was attention-patched before rebase, so the checkpoint must use the same patched keyspace.' if patch_attn_before_rebase else ''}"
                        )
                    tuned_sd = to_cpu_fp32(aligned)
                    task_delta = TaskVector.from_checkpoints(
                        source_base_sd,
                        tuned_sd,
                        strict=False,
                        key_filter=_visual_only_filter,
                    ).delta
                    if base_construction == "independent_endpoint_average":
                        independent_base_by_task[task] = task_source_base_sd
                        independent_ft_by_task[task] = tuned_sd

                n_keys = len(tuned_sd)
                print(f"Loaded tuned checkpoint for '{task}' ({n_keys} keys)")

            direct_target_p1 = bool(
                (task_block_extension_prestep or run_same_depth_direct_target)
                and block_extension_cfg.target_residual_completion.enabled
                and block_extension_cfg.target_residual_completion.mode == "direct_target"
            )
            if direct_target_p1:
                print(
                    f"\n--- Direct-target P1 for '{task}' "
                    "(parameter transport skipped) ---"
                )
            else:
                print(f"\n--- Transporting '{task}' with method '{method.name}' ---")
            if merge_mode not in _SINGLE_TRANSPORT_MODES:
                if torch.cuda.is_available() and device != "cpu":
                    torch.cuda.reset_peak_memory_stats()
                prepare_started = time.perf_counter()

                # Proposal-1 transport-free ablation: THESEUS/BiCo are neither
                # fitted nor applied. The question this arm asks is whether the
                # desired functional effect can be written into the target at
                # all without parameter transport, so invoking the transport
                # fit and then discarding its output would only burn GPU hours
                # and blur the claim.
                bypass_ordinary_transport = direct_target_p1 or direct_residual_like
                prepared = None if bypass_ordinary_transport else _build_rebase_prepared(
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
                    run_block_extension_prestep=task_block_extension_prestep or task_discrete_layer_match_prestep,
                    clf_source=clf_source,
                    clf_target=clf_target,
                    classnames=classnames,
                    loaders=loaders,
                    source_loaders=source_loaders,
                    build_cfg_task=build_cfg_task,
                    source_build_cfg_task=source_build_cfg_task,
                    task_source_base_sd=task_source_base_sd,
                    target_base_sd=target_base_sd,
                    task_delta=task_delta,
                    source_base_model_task=source_base_model_task,
                    transfusion_prepared=transfusion_prepared,
                    source_activation_plan=task_source_activation_plan,
                )

                prepare_seconds = time.perf_counter() - prepare_started
                if torch.cuda.is_available() and device != "cpu":
                    torch.cuda.synchronize()
                    peak_memory_bytes = float(torch.cuda.max_memory_allocated())
                else:
                    peak_memory_bytes = 0.0

                transport_started = time.perf_counter()
                transported_delta = {} if bypass_ordinary_transport else method.transport(
                    source_base=task_source_base_sd,
                    target_base=target_base_sd,
                    delta=task_delta,
                    strict=strict_load,
                    prepared=prepared,
                    **method_params,
                )
                if torch.cuda.is_available() and device != "cpu":
                    torch.cuda.synchronize()
                transport_timings[task] = {
                    "prepare_seconds": prepare_seconds,
                    "transport_seconds": time.perf_counter() - transport_started,
                    "peak_memory_allocated_bytes": peak_memory_bytes,
                }

                if direct_residual_like:
                    # Direct Residual never resizes anything: capture must run
                    # against the NATIVE, un-resized source models, never a
                    # block-extended reference from elsewhere in this function
                    # (source_base_model_task/source_ft_model_task are only
                    # ever populated when blockext_like_method is True, which
                    # is never the case for direct_residual_like -- see the
                    # method-dispatch resolution above -- so freshly building
                    # native copies here, rather than reusing those variables,
                    # is both correct and the only option).
                    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
                    if direct_residual_cfg.merge_mode == "merge_in_source_then_fit":
                        if direct_residual_merged_correction is None or direct_residual_merged_timing is None:
                            raise RuntimeError(
                                "Direct Residual merge_in_source_then_fit correction was not precomputed "
                                "before the per-task loop."
                            )
                        transported_delta = dict(direct_residual_merged_correction)
                        alignment_calibration_timings[task] = dict(direct_residual_merged_timing["alignment_calibration"])
                        correction_fit_timings[task] = dict(direct_residual_merged_timing["correction_fit"])
                    else:
                        source_base_model_native = deepcopy(clf_source.model)
                        source_ft_model_native = deepcopy(clf_source.model)
                        load_into_model(source_base_model_native, source_base_sd, strict=True)
                        load_into_model(source_ft_model_native, source_base_sd, strict=True)
                        load_into_model(source_ft_model_native, load_ckpt(str(tuned_by_task[task])), strict=False)
                        direct_residual_source_loader = select_loader(
                            "train",
                            train_loader=source_loaders.train,
                            test_loader=source_loaders.test,
                            val_loader=source_loaders.val,
                        )
                        direct_residual_target_loader = select_loader(
                            "train",
                            train_loader=loaders.train,
                            test_loader=loaders.test,
                            val_loader=loaders.val,
                        )
                        (
                            transported_delta,
                            direct_residual_timing,
                            task_direct_residual_diag,
                            task_direct_residual_extra,
                        ) = _run_direct_residual_fit(
                            source_base_model=source_base_model_native,
                            source_ft_model=source_ft_model_native,
                            target_model=clf_target.model,
                            target_base_sd=target_base_sd,
                            source_loader=direct_residual_source_loader,
                            target_loader=direct_residual_target_loader,
                            pairing=pairing,
                            config=direct_residual_cfg,
                            device=device,
                            clf_source=clf_source,
                            clf_target=clf_target,
                            classnames=classnames,
                            source_build_cfg_task=source_build_cfg_task,
                            build_cfg_task=build_cfg_task,
                            source_text_features=task_ctx.source_text_features,
                            target_text_features=task_ctx.target_text_features,
                        )
                        alignment_calibration_timings[task] = direct_residual_timing["alignment_calibration"]
                        correction_fit_timings[task] = direct_residual_timing["correction_fit"]
                        direct_residual_diagnostics[task] = task_direct_residual_diag
                        direct_residual_realization[task] = task_direct_residual_extra["realization_by_position"]
                        direct_residual_task_vector_stats[task] = task_direct_residual_extra["task_vector_stats"]
                        direct_residual_alignment_diagnostics[task] = task_direct_residual_extra["alignment_diagnostics"]
                        direct_residual_tv_scaling[task] = task_direct_residual_extra["tv_scaling"]

                if (task_block_extension_prestep or run_same_depth_direct_target) and block_extension_cfg.target_residual_completion.enabled:
                    # Proposal 1 completes the transported task vector's
                    # residual projections. In the same-depth direct-target
                    # path, references and the identity layout are prepared
                    # separately; both cases leave the target base unchanged.
                    transported_delta, task_residual_diagnostics = _maybe_complete_target_residual_task_vector(
                        config=block_extension_cfg.target_residual_completion,
                        references=task_residual_references,
                        prepared=prepared,
                        layout=task_extension_layout,
                        target_model=clf_target.model,
                        target_base_sd=target_base_sd,
                        transported_delta=transported_delta,
                        target_loader=task_residual_target_loader,
                        device=device,
                    )
                    if task_residual_diagnostics is not None:
                        residual_completion_diagnostics[task] = task_residual_diagnostics
                if task_block_extension_prestep and block_extension_cfg.joint_blockwise_correction.enabled:
                    transported_delta, task_joint_diagnostics = _maybe_complete_joint_blockwise_task_vector(
                        config=block_extension_cfg.joint_blockwise_correction,
                        references=task_joint_references,
                        prepared=prepared,
                        layout=task_extension_layout,
                        target_model=clf_target.model,
                        target_base_sd=target_base_sd,
                        transported_delta=transported_delta,
                        target_loader=task_joint_target_loader,
                        device=device,
                    )
                    if task_joint_diagnostics is not None:
                        joint_blockwise_diagnostics[task] = task_joint_diagnostics
                        run_logger.log_event(
                            "joint_blockwise_correction",
                            metrics={
                                f"rebase/{task}/joint_blocks": float(len(task_joint_diagnostics)),
                                f"rebase/{task}/joint_objective_after": float(
                                    sum(row["objective_after"] for row in task_joint_diagnostics)
                                ),
                            },
                            context={"task": task, "method": method.name},
                        )
                if task_block_extension_prestep and block_extension_cfg.direct_p1_correction.enabled:
                    transported_delta, task_direct_p1_diagnostics = _maybe_complete_direct_p1_task_vector(
                        config=block_extension_cfg.direct_p1_correction,
                        references=task_direct_p1_references,
                        prepared=prepared,
                        layout=task_extension_layout,
                        source_base_model=source_base_model_task,
                        source_ft_model=source_ft_model_task,
                        target_model=clf_target.model,
                        target_base_sd=target_base_sd,
                        transported_delta=transported_delta,
                        source_loader=calibration_loader,
                        target_loader=task_direct_p1_target_loader,
                        device=device,
                    )
                    if task_direct_p1_diagnostics is not None:
                        direct_p1_diagnostics[task] = task_direct_p1_diagnostics
                        run_logger.log_event(
                            "direct_p1_correction",
                            metrics={
                                f"rebase/{task}/direct_p1_blocks": float(len(task_direct_p1_diagnostics)),
                                f"rebase/{task}/direct_p1_objective_after": float(
                                    sum(row["objective_after"] for row in task_direct_p1_diagnostics)
                                ),
                            },
                            context={"task": task, "method": method.name},
                        )

                transported_deltas.append(transported_delta)
                original_deltas.append(task_delta)
                print(f"  {task}: transported delta computed for {len(transported_delta)} params")
                run_logger.log_event(
                    "transport_task_end",
                    metrics={f"rebase/{task}/transported_param_count": float(len(transported_delta))},
                    context={"task": task, "method": method.name},
                )

                save_transport_dir = cfg.get("save_transported_tvs_dir", None)
                save_transported_artifacts = bool(cfg.get("save_transported_artifacts", bool(save_transport_dir)))
                if save_transported_artifacts and not save_transport_dir:
                    raise ValueError("save_transported_artifacts=true requires save_transported_tvs_dir.")
                if save_transported_artifacts and save_transport_dir:
                    os.makedirs(save_transport_dir, exist_ok=True)
                    native_path = os.path.join(save_transport_dir, f"{task}_{method.name}_transported_native.pt")
                    torch.save(to_cpu_fp32(transported_delta), native_path)
                    print(f"  {task}: saved transported TV -> {native_path}")
                    transported_artifacts[task] = [native_path]
                    if bool(cfg.get("save_transported_tvs_legacy", False)):
                        legacy_path = os.path.join(save_transport_dir, f"{task}_{method.name}_transported_legacy_visual.pt")
                        legacy_no_conv1_path = os.path.join(
                            save_transport_dir, f"{task}_{method.name}_transported_legacy_visual_no_conv1.pt"
                        )
                        torch.save(_legacy_visual_delta(transported_delta), legacy_path)
                        torch.save(_legacy_visual_delta(transported_delta, drop_conv1=True), legacy_no_conv1_path)
                        print(f"  {task}: saved legacy visual TV -> {legacy_path}")
                        print(f"  {task}: saved legacy visual TV without conv1 -> {legacy_no_conv1_path}")
                        transported_artifacts[task] += [legacy_path, legacy_no_conv1_path]
            else:
                original_deltas.append(task_delta)
                print(f"  {task}: delta collected for merge_then_rebase ({len(task_delta)} params)")

        can_eval_untransported_by_task: list[bool] = []
        single_tv_deltas_for_diagnostic: list[dict[str, torch.Tensor]] | None = None
        single_transport_calibration_metadata: dict[str, Any] | None = None
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
                        f"No tensors from native target checkpoint aligned to target base keys "
                        f"for task '{task}': {path}."
                    )
                native_delta_by_task[task] = TaskVector.from_checkpoints(
                    target_base_sd,
                    to_cpu_fp32(aligned),
                    strict=False,
                    key_filter=_visual_only_filter,
                ).delta
                print(
                    f"  {task}: native target delta computed ({len(native_delta_by_task[task])} params)"
                )
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
                if corrected_source_template is None or not independent_base_by_task:
                    raise RuntimeError("BRACE-then-merge requires corrected source endpoints for every task.")
                average_visual, average_keys = _average_visual_state_dicts(independent_base_by_task)
                first_base = independent_base_by_task[sorted(independent_base_by_task)[0]]
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
                    for task, state in independent_base_by_task.items()
                }
                calibration_metadata.update(
                    {
                        "consensus_source_base": "mean_corrected_source_base",
                        "consensus_visual_key_count": len(average_keys),
                        "source_base_relative_distance_by_task": distances,
                        "source_base_max_relative_distance": max(distances.values()),
                    }
                )
                source_template_once = corrected_source_template
                prepared_has_brace = True
                merged_source_activation_plan = _resolve_source_activation_plan(
                    block_extension_cfg, recorded_extension_layout
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
            rebased_deltas = [transported_merged_delta] * len(tasks)
            untransported_deltas = original_deltas

        if merge_mode != "none":
            # The per-task "untransported" baseline is meaningless for a single
            # merged model; fall back to the (alpha-independent, cached) target
            # zero-shot baseline so normalized ratios remain defined.
            can_eval_untransported_by_task = [False] * len(tasks)

        def _eval_task(item: dict[str, Any], split: str) -> float:
            return float(
                eval_task_top1(
                    clf=clf_target,
                    loaders=item["loaders"],
                    classnames=list(item["classnames"]),
                    build_cfg_task=item["build_cfg_task"],
                    device=device,
                    split=split,
                )
            )

        def _eval_all_tasks(split: str) -> list[float]:
            return [_eval_task(item, split) for item in per_task]

        if all(can_eval_untransported_by_task):
            baseline_label = "untransported"
        elif any(can_eval_untransported_by_task):
            baseline_label = "mixed_baseline"
        else:
            baseline_label = "target_zeroshot"
        result_label = "rebased"
        task_col = max(max((len(str(item["task"])) for item in per_task), default=4), len("task"), len("avg"))
        metric_col = max(12, len(baseline_label) + 2, len(result_label) + 2, len("norm") + 2)

        baseline_cache_zeroshot: dict[str, list[float]] = {}

        if baseline_label == "untransported":
            print("Using untransported baseline evaluation for all tasks.")
        elif baseline_label == "mixed_baseline":
            print("Using mixed baseline evaluation: untransported where compatible, target zeroshot otherwise.")
        else:
            print("Using target zeroshot baseline for all tasks.")

        def _load_into_target_model(sd: dict[str, torch.Tensor]) -> None:
            if transfusion_mode:
                method.load_into_target_visual(clf_target, sd, strict=False)
            else:
                load_into_model(clf_target.model, sd, strict=strict_load)

        def _eval_zeroshot_all_tasks(split: str) -> list[float]:
            if split not in baseline_cache_zeroshot:
                _load_into_target_model(target_base_sd)
                baseline_cache_zeroshot[split] = _eval_all_tasks(split)
            return list(baseline_cache_zeroshot[split])

        def _eval_baseline_task(split: str, idx: int, alpha: float) -> float:
            if can_eval_untransported_by_task[idx]:
                baseline_sd = axpy_state_dict(target_base_sd, untransported_deltas[idx], alpha=float(alpha))
                _load_into_target_model(baseline_sd)
                del baseline_sd
                return _eval_task(per_task[idx], split)
            return _eval_zeroshot_all_tasks(split)[idx]

        def _eval_baseline_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            return {idx: _eval_baseline_task(split, idx, alpha) for idx in indices}

        def _eval_rebased_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            out: dict[int, float] = {}
            for idx in indices:
                rebase_sd_task = axpy_state_dict(target_base_sd, rebased_deltas[idx], alpha=float(alpha))
                _load_into_target_model(rebase_sd_task)
                del rebase_sd_task
                out[idx] = _eval_task(per_task[idx], split)
            return out

        def _eval_single_tv_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            if single_tv_deltas_for_diagnostic is None:
                raise RuntimeError("Single-TV diagnostic requires merge_mode='rebase_then_merge'.")
            out: dict[int, float] = {}
            for idx in indices:
                single_sd = axpy_state_dict(
                    target_base_sd,
                    single_tv_deltas_for_diagnostic[idx],
                    alpha=float(alpha),
                )
                _load_into_target_model(single_sd)
                del single_sd
                out[idx] = _eval_task(per_task[idx], split)
            return out

        hierarchical = bool(merge_mode in _TRANSPORT_THEN_MERGE_MODES and alpha_selection == "per_task")
        single_tv_diagnostic_enabled = single_tv_deltas_for_diagnostic is not None
        single_tv_val_best_acc: list[float] | None = (
            [float("-inf")] * len(per_task) if single_tv_diagnostic_enabled else None
        )
        single_tv_val_best_alpha: list[float] | None = (
            [float(alphas[0])] * len(per_task) if single_tv_diagnostic_enabled else None
        )
        single_tv_alpha_protocol = (
            "per_task_premerge_alpha" if hierarchical else "single_tv_validation_oracle"
        )
        per_task_premerge_alphas: list[float] | None = None
        hierarchical_premerge_alpha_curve: list[dict[str, Any]] | None = None
        global_alpha_curve: list[dict[str, Any]] | None = None

        if alpha_selection == "shared" or hierarchical:
            if hierarchical:
                # ---------------- PASS 1: per-task alpha on individual deltas ----------------
                hierarchical_premerge_alpha_curve = []
                tracker = PerTaskAlphaTracker(
                    task_names=[str(item["task"]) for item in per_task],
                    initial_alpha=float(alphas[0]),
                    patience=alpha_patience,
                )
                # Merge-mode baselines are the alpha-independent target zero-shot:
                # pre-seed the secondary stream inactive for every task.
                for idx in range(len(per_task)):
                    baseline_val = _eval_baseline_task(alpha_search_split, idx, 0.0)
                    tracker.best_secondary_alpha[idx] = 0.0
                    tracker.best_secondary_acc[idx] = baseline_val
                    tracker.secondary_active[idx] = False

                for alpha in alphas:
                    eval_indices = tracker.eval_active_indices()
                    if not eval_indices:
                        print("\nAll tasks have early-stopped; ending hierarchical pass-1 alpha sweep.")
                        break

                    primary_indices = set(tracker.primary_active_indices())
                    print(
                        f"\n=== alpha {alpha:.3f} — {method_label} "
                        f"(split: {alpha_search_split}, mode: per_task pass 1/2, hierarchical) ==="
                    )

                    baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, eval_indices, float(alpha))
                    rebase_eval_indices = [idx for idx in eval_indices if idx in primary_indices]
                    rebase_by_idx_active = _eval_rebased_task_indices(alpha_search_split, rebase_eval_indices, float(alpha))
                    rebase_by_idx: dict[int, float] = {}
                    for idx in eval_indices:
                        rebase_by_idx[idx] = rebase_by_idx_active[idx] if idx in primary_indices else float("-inf")
                    baseline_accs = [baseline_by_idx[idx] for idx in eval_indices]
                    rebase_accs = [rebase_by_idx[idx] for idx in eval_indices]

                    for idx, baseline_acc, rebase_acc in zip(eval_indices, baseline_accs, rebase_accs, strict=True):
                        task_name = per_task[idx]["task"]
                        if idx in primary_indices:
                            display_rebase = rebase_acc
                            marker = " "
                        else:
                            frozen = float(tracker.best_primary_acc[idx])
                            display_rebase = frozen if frozen != float("-inf") else 0.0
                            marker = "*"
                        norm = _norm_acc(display_rebase, baseline_acc)
                        print(
                            f" {marker}{task_name}: {baseline_label}={baseline_acc:.6f}  "
                            f"per-task={display_rebase:.6f}  norm={norm:.6f}"
                        )

                    hierarchical_premerge_alpha_curve.append(
                        {
                            "alpha": float(alpha),
                            "per_task_rebased": {
                                per_task[idx]["task"]: float(rebase_by_idx[idx])
                                for idx in rebase_eval_indices
                            },
                            "per_task_baseline": {
                                per_task[idx]["task"]: float(baseline_by_idx[idx])
                                for idx in eval_indices
                            },
                        }
                    )
                    run_logger.log_event(
                        "hierarchical_premerge_alpha_eval_end",
                        metrics={"alpha/value": float(alpha)},
                        context=hierarchical_premerge_alpha_curve[-1],
                    )

                    stopped_primary, _ = tracker.update(
                        alpha=float(alpha),
                        indices=eval_indices,
                        primary_accs=rebase_accs,
                        secondary_accs=baseline_accs,
                    )
                    if stopped_primary:
                        stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_primary)
                        print(f"  Early-stopping per-task alphas at alpha={alpha:.3f}: {stopped_names}")

                per_task_premerge_alphas = [float(tracker.best_primary_alpha[idx]) for idx in range(len(per_task))]
                single_tv_val_best_acc = [float(tracker.best_primary_acc[idx]) for idx in range(len(per_task))]
                single_tv_val_best_alpha = list(per_task_premerge_alphas)
                print("\n=== Hierarchical pass-1 summary (per-task alphas) ===")
                for item, a in zip(per_task, per_task_premerge_alphas, strict=True):
                    print(f"  {item['task']}: premerge_alpha={a:.3f}")
                run_logger.log_event(
                    "hierarchical_pass1_end",
                    metrics={
                        "hierarchical/avg_premerge_alpha": float(
                            sum(per_task_premerge_alphas) / max(1, len(per_task_premerge_alphas))
                        )
                    },
                    context={
                        "per_task_premerge_alphas": {
                            item["task"]: float(per_task_premerge_alphas[i]) for i, item in enumerate(per_task)
                        }
                    },
                )

                # ---------------- PASS 2: scale by per-task alphas, compose once ----------------
                scaled_input = _scale_deltas_by(merge_input_deltas, per_task_premerge_alphas)
                merged_direction = _merge_direction(
                    base_sd=target_base_sd,
                    deltas=scaled_input,
                    merge_method_name=merge_method_name,
                    weights=merge_weights,
                    merge_params=merge_params,
                )
                rebased_deltas = [merged_direction] * len(per_task)
                print(
                    f"Hierarchical merge ({merge_method_name}): composed {len(scaled_input)} scaled deltas "
                    f"-> merged direction with {len(merged_direction)} params"
                )
                run_logger.log_event(
                    "merge_composition_end",
                    metrics={"merge/param_count": float(len(merged_direction))},
                    context={
                        "mode": merge_mode,
                        "merge_method": merge_method_name,
                        "merge_params": merge_params,
                        "n_tasks": len(scaled_input),
                        "per_task_premerge_alphas": {
                            item["task"]: float(per_task_premerge_alphas[i]) for i, item in enumerate(per_task)
                        },
                    },
                )

            sweep_alphas = list(alphas)
            if hierarchical and not global_alpha_search:
                sweep_alphas = [1.0]
                print("\nglobal_alpha_search=false: evaluating the merged model at gamma=1.0 only.")
            sweep_positive_alphas = [float(a) for a in sweep_alphas if float(a) > 0.0]
            sweep_mode_label = "global (gamma)" if hierarchical else "shared"

            best_rebase_avg = float("-inf")
            best_baseline_avg = float("-inf")
            best_alpha = float(sweep_alphas[0])
            # When every task's baseline is target_zeroshot (untransported infeasible),
            # the baseline is alpha-independent — keep best_baseline_alpha at 0.0 so
            # the summary does not report a spurious non-zero value.
            has_untransported = any(can_eval_untransported_by_task)
            best_baseline_alpha = (
                float(sweep_positive_alphas[0] if sweep_positive_alphas else sweep_alphas[0])
                if has_untransported
                else 0.0
            )
            sweep_results: list[dict[str, Any]] = []
            shared_bad_steps = 0

            for alpha in sweep_alphas:
                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: {sweep_mode_label}) ===")

                idxs = list(range(len(per_task)))
                baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, idxs, float(alpha))
                rebase_by_idx = _eval_rebased_task_indices(alpha_search_split, idxs, float(alpha))
                if (
                    single_tv_diagnostic_enabled
                    and single_tv_val_best_acc is not None
                    and single_tv_val_best_alpha is not None
                ):
                    single_tv_by_idx = _eval_single_tv_task_indices(alpha_search_split, idxs, float(alpha))
                    for idx in idxs:
                        if single_tv_by_idx[idx] > single_tv_val_best_acc[idx]:
                            single_tv_val_best_acc[idx] = float(single_tv_by_idx[idx])
                            single_tv_val_best_alpha[idx] = float(alpha)
                baseline_accs = [baseline_by_idx[i] for i in idxs]
                rebase_accs = [rebase_by_idx[i] for i in idxs]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for i, item in enumerate(per_task):
                    task_name = item["task"]
                    baseline_acc = baseline_accs[i]
                    rebase_acc = rebase_accs[i]
                    norm = _norm_acc(rebase_acc, baseline_acc)
                    print(
                        f"  {task_name:<{task_col}}  {baseline_acc:>{metric_col}.6f}  {rebase_acc:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                    )

                avg_rebase = average_scores(rebase_accs)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined([_norm_acc(r, b) for r, b in zip(rebase_accs, baseline_accs, strict=True)])
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                print(
                    f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
                )

                sweep_results.append(
                    {
                        "alpha": float(alpha),
                        "baseline_accs": baseline_accs,
                        "rebase_accs": rebase_accs,
                    }
                )
                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(alpha),
                        "alpha/avg_acc": float(avg_rebase),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "baseline_label": baseline_label,
                        "per_task_baseline": {item["task"]: float(baseline_accs[i]) for i, item in enumerate(per_task)},
                        "per_task_rebased": {item["task"]: float(rebase_accs[i]) for i, item in enumerate(per_task)},
                    },
                )

                eps = 1e-12
                # Track baseline best alpha independently of rebased best alpha,
                # but only when there is at least one untransported baseline task
                # (otherwise the baseline is target_zeroshot and alpha-independent).
                if has_untransported:
                    if avg_baseline != avg_baseline:  # NaN guard
                        avg_baseline_for_track = float("-inf")
                    else:
                        avg_baseline_for_track = float(avg_baseline)
                    if avg_baseline_for_track > best_baseline_avg + eps:
                        best_baseline_avg = avg_baseline_for_track
                        best_baseline_alpha = float(alpha)

                if avg_rebase > best_rebase_avg + eps:
                    best_rebase_avg = avg_rebase
                    best_alpha = float(alpha)
                    shared_bad_steps = 0
                elif avg_rebase + eps >= best_rebase_avg:
                    shared_bad_steps = 0
                elif len(sweep_alphas) > 1:
                    shared_bad_steps += 1
                    print(
                        f"  (alpha={alpha:.3f} fell below best shared avg {best_rebase_avg:.6f}; "
                        f"bad_steps={shared_bad_steps}/{alpha_patience + 1})"
                    )
                    if shared_bad_steps > alpha_patience:
                        break

            print("\n=== Alpha search summary (shared) ===")
            for r in sweep_results:
                a = r["alpha"]
                avg_r = average_scores(r["rebase_accs"])
                avg_b = _average_defined(r["baseline_accs"])
                print(f"  alpha={a:.3f}  {baseline_label}={avg_b:.6f}  {result_label}={avg_r:.6f}")
            global_alpha_curve = [
                {
                    "alpha": float(row["alpha"]),
                    "avg_rebased": float(average_scores(row["rebase_accs"])),
                    "avg_baseline": float(_average_defined(row["baseline_accs"])),
                    "per_task_rebased": {
                        item["task"]: float(row["rebase_accs"][idx]) for idx, item in enumerate(per_task)
                    },
                    "per_task_baseline": {
                        item["task"]: float(row["baseline_accs"][idx]) for idx, item in enumerate(per_task)
                    },
                }
                for row in sweep_results
            ]
            print(
                f"\nBest alpha: rebase={best_alpha:.3f} (avg rebased val acc={best_rebase_avg:.6f}) | "
                f"baseline={best_baseline_alpha:.3f} (avg baseline val acc={best_baseline_avg:.6f})"
            )

            print(
                f"\n(Re-running on test split: rebase at alpha={best_alpha:.3f}, baseline at alpha={best_baseline_alpha:.3f})"
            )
            all_indices = list(range(len(per_task)))
            baseline_test_by_idx = _eval_baseline_task_indices("test", all_indices, float(best_baseline_alpha))
            rebase_test_by_idx = _eval_rebased_task_indices("test", all_indices, float(best_alpha))
            baseline_test_accs = [baseline_test_by_idx[i] for i in all_indices]
            rebase_test_accs = [rebase_test_by_idx[i] for i in all_indices]
            selected_alpha_by_task = [float(best_alpha)] * len(per_task)
            selected_baseline_alpha_by_task = [float(best_baseline_alpha)] * len(per_task)

        else:
            tracker = PerTaskAlphaTracker(
                task_names=[str(item["task"]) for item in per_task],
                initial_alpha=float(alphas[0]),
                patience=alpha_patience,
            )
            # For tasks where the untransported baseline is infeasible (different
            # architecture / shape), the baseline is target_zeroshot and
            # alpha-independent. Pre-seed the secondary tracker so
            # best_secondary_alpha stays at 0.0 and the secondary stream never
            # participates in alpha optimization for these tasks.
            for idx in range(len(per_task)):
                if not can_eval_untransported_by_task[idx]:
                    baseline_val = _eval_baseline_task(alpha_search_split, idx, 0.0)
                    tracker.best_secondary_alpha[idx] = 0.0
                    tracker.best_secondary_acc[idx] = baseline_val
                    tracker.secondary_active[idx] = False
            sweep_results = []

            for alpha in alphas:
                # The baseline may keep being swept after the rebased has early-stopped
                # a task, so we evaluate the union of primary- and secondary-active tasks
                # and decouple the two streams' early stopping.
                eval_indices = tracker.eval_active_indices()
                if not eval_indices:
                    print("\nAll tasks have early-stopped on both streams; ending per-task alpha sweep.")
                    break

                primary_indices = set(tracker.primary_active_indices())
                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: per_task) ===")

                # Baseline must be evaluated for every index in the union (baseline may
                # still be active for tasks where the rebased already early-stopped).
                baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, eval_indices, float(alpha))
                # Rebased is only evaluated for primary-active tasks; for tasks where it
                # has already stopped, we pass -inf as a no-op placeholder.
                rebase_eval_indices = [idx for idx in eval_indices if idx in primary_indices]
                rebase_by_idx_active = _eval_rebased_task_indices(alpha_search_split, rebase_eval_indices, float(alpha))
                rebase_by_idx: dict[int, float] = {}
                for idx in eval_indices:
                    if idx in primary_indices:
                        rebase_by_idx[idx] = rebase_by_idx_active[idx]
                    else:
                        rebase_by_idx[idx] = float("-inf")

                baseline_accs = [baseline_by_idx[idx] for idx in eval_indices]
                rebase_accs = [rebase_by_idx[idx] for idx in eval_indices]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for idx, baseline_acc, rebase_acc in zip(eval_indices, baseline_accs, rebase_accs, strict=True):
                    task_name = per_task[idx]["task"]
                    # During the val sweep, norm uses the baseline's own best-so-far
                    # accuracy (not the same-alpha baseline), so the per-step ratio
                    # already reflects the final normalization semantics.
                    best_secondary = float(tracker.best_secondary_acc[idx])
                    norm_baseline = best_secondary if best_secondary != float("-inf") else baseline_acc
                    # For tasks whose rebased stream already early-stopped, rebase_acc
                    # is -inf (not evaluated this step); display the frozen best rebased
                    # accuracy instead so the reported number tracks the rebased peak.
                    if idx in primary_indices:
                        display_rebase = rebase_acc
                    else:
                        frozen = float(tracker.best_primary_acc[idx])
                        display_rebase = frozen if frozen != float("-inf") else 0.0
                    norm = _norm_acc(display_rebase, norm_baseline)
                    marker = " " if idx in primary_indices else "*"
                    print(
                        f" {marker}{task_name:<{task_col - 1}}  {baseline_acc:>{metric_col}.6f}  {display_rebase:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                    )

                # Average rebased across ALL tasks: use the current value for active tasks
                # and the frozen best for stopped tasks, so the avg does not
                # collapse to 0.0 once every rebased task has early-stopped.
                all_rebase_vals: list[float] = []
                for idx in range(len(per_task)):
                    if idx in primary_indices:
                        all_rebase_vals.append(float(rebase_by_idx[idx]))
                    else:
                        frozen = float(tracker.best_primary_acc[idx])
                        all_rebase_vals.append(frozen if frozen != float("-inf") else 0.0)
                avg_rebase = average_scores(all_rebase_vals)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined(
                    [_norm_acc(
                        float(rebase_by_idx[idx]) if idx in primary_indices else max(float(tracker.best_primary_acc[idx]), 0.0),
                        baseline_accs[i],
                    ) for i, idx in enumerate(eval_indices)]
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                print(
                    f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
                )

                stopped_primary: list[int] = []
                stopped_secondary: list[int] = []
                stopped_primary, stopped_secondary = tracker.update(
                    alpha=float(alpha),
                    indices=eval_indices,
                    primary_accs=rebase_accs,
                    secondary_accs=baseline_accs,
                )
                if stopped_primary:
                    stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_primary)
                    print(f"  Early-stopping REBASED tasks at alpha={alpha:.3f}: {stopped_names}")
                if stopped_secondary:
                    stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_secondary)
                    print(f"  Early-stopping BASELINE tasks at alpha={alpha:.3f}: {stopped_names}")

                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(alpha),
                        "alpha/avg_acc": float(avg_rebase),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "active_tasks": [per_task[idx]["task"] for idx in eval_indices],
                        "per_task_baseline": {per_task[idx]["task"]: float(baseline_by_idx[idx]) for idx in eval_indices},
                        "per_task_rebased": {per_task[idx]["task"]: float(rebase_by_idx[idx]) for idx in eval_indices},
                        "stopped_primary": [int(idx) for idx in stopped_primary],
                        "stopped_secondary": [int(idx) for idx in stopped_secondary],
                    },
                )

                sweep_results.append(
                    {
                        "alpha": float(alpha),
                        "active_indices": list(eval_indices),
                        "baseline_accs": baseline_accs,
                        "rebase_accs": rebase_accs,
                        "primary_active": [int(idx) for idx in eval_indices if idx in primary_indices],
                    }
                )

            print("\n=== Alpha search summary (per-task) ===")
            for idx, item in enumerate(per_task):
                print(
                    f"  {item['task']}: rebase_alpha={tracker.best_primary_alpha[idx]:.3f}  "
                    f"rebase_val={tracker.best_primary_acc[idx]:.6f} | "
                    f"baseline_alpha={tracker.best_secondary_alpha[idx]:.3f}  "
                    f"baseline_val={tracker.best_secondary_acc[idx]:.6f}"
                )
            print(f"\nAvg per-task best rebase val acc: {tracker.best_avg():.6f}")
            best_baseline_vals = [float(v) for v in tracker.best_secondary_acc if v != float("-inf")]
            if best_baseline_vals:
                print(f"Avg per-task best baseline val acc: {sum(best_baseline_vals) / len(best_baseline_vals):.6f}")

            print("\n(Re-running per-task best alphas on test split — decoupled per stream)")
            baseline_test_accs: list[float] = []
            rebase_test_accs: list[float] = []
            selected_alpha_by_task: list[float] = []
            selected_baseline_alpha_by_task: list[float] = []
            for idx, item in enumerate(per_task):
                rebase_alpha = float(tracker.best_primary_alpha[idx])
                baseline_alpha = float(tracker.best_secondary_alpha[idx])
                selected_alpha_by_task.append(rebase_alpha)
                selected_baseline_alpha_by_task.append(baseline_alpha)
                print(f"  {item['task']}: rebase_alpha={rebase_alpha:.3f}  baseline_alpha={baseline_alpha:.3f}")

                baseline_test_accs.append(_eval_baseline_task("test", idx, baseline_alpha))

                rebase_sd_task = axpy_state_dict(target_base_sd, rebased_deltas[idx], alpha=rebase_alpha)
                _load_into_target_model(rebase_sd_task)
                del rebase_sd_task
                rebase_test_accs.append(_eval_task(item, "test"))
            best_alpha = float(sum(selected_alpha_by_task) / max(1, len(selected_alpha_by_task)))
            best_baseline_alpha = float(sum(selected_baseline_alpha_by_task) / max(1, len(selected_baseline_alpha_by_task)))

        norm_accs = [_norm_acc(r, b) for r, b in zip(rebase_test_accs, baseline_test_accs, strict=True)]

        single_tv_test_accs: list[float] | None = None
        single_tv_test_alpha_by_task: list[float] | None = None
        if single_tv_diagnostic_enabled and single_tv_val_best_alpha is not None:
            single_tv_test_alpha_by_task = [float(a) for a in single_tv_val_best_alpha]
            single_tv_test_accs = []
            print("\nSingle transported task-vector test diagnostic:")
            for idx, item in enumerate(per_task):
                alpha = single_tv_test_alpha_by_task[idx]
                single_acc = _eval_single_tv_task_indices("test", [idx], alpha)[idx]
                single_tv_test_accs.append(float(single_acc))
                print(f"  {item['task']}: alpha={alpha:.3f}  single_tv_test={single_acc:.6f}")
            single_avg = sum(single_tv_test_accs) / len(single_tv_test_accs)
            merged_avg = sum(rebase_test_accs) / len(rebase_test_accs)
            print(
                f"  avg single_tv_test={single_avg:.6f}  merged_test={merged_avg:.6f} "
                f"merge_gap={single_avg - merged_avg:+.6f}"
            )
            run_logger.log_event(
                "single_tv_test_diagnostic_end",
                metrics={
                    "single_tv/avg_test_accuracy": float(single_avg),
                    "single_tv/merged_test_accuracy": float(merged_avg),
                    "single_tv/merge_gap": float(single_avg - merged_avg),
                },
                context={
                    "alpha_protocol": single_tv_alpha_protocol,
                    "per_task_alpha": {
                        item["task"]: float(single_tv_test_alpha_by_task[i])
                        for i, item in enumerate(per_task)
                    },
                    "per_task_test_accuracy": {
                        item["task"]: float(single_tv_test_accs[i])
                        for i, item in enumerate(per_task)
                    },
                },
            )

        alpha_display_label = "hierarchical(per_task+global)" if hierarchical else alpha_selection
        pretty_print_task_accuracies(
            suite_name,
            f"{method_label}, alpha={alpha_display_label}",
            f"A={source_cfg.pretrained} → B={target_cfg.pretrained}",
            per_task,
            rebase_test_accs,
            norm_accs,
            single_accs=baseline_test_accs,
            baseline_label=baseline_label,
            result_label=result_label,
        )

        if hierarchical and per_task_premerge_alphas is not None:
            print("\nHierarchical alphas (per-task premerge + global merge alpha):")
            for item, a in zip(per_task, per_task_premerge_alphas, strict=True):
                print(f"  {item['task']}: premerge={a:.3f}  global={best_alpha:.3f}")
        elif alpha_selection == "per_task":
            print("\nSelected test-time alpha by task:")
            for item, r_a, b_a in zip(per_task, selected_alpha_by_task, selected_baseline_alpha_by_task, strict=True):
                print(f"  {item['task']}: rebase={r_a:.3f}  baseline={b_a:.3f}")

        saved_merged_path: str | None = None
        if cfg.get("save_merged"):
            if merge_mode == "none":
                print(
                    "save_merged was requested, but task-independent alpha mode does not produce a single merged checkpoint; skipping save."
                )
            else:
                best_merged_state = axpy_state_dict(
                    target_base_sd,
                    rebased_deltas[0],
                    alpha=float(best_alpha),
                )
                out_path = str(cfg["save_merged"])
                out_parent = Path(out_path).parent
                if str(out_parent):
                    out_parent.mkdir(parents=True, exist_ok=True)
                torch.save(to_cpu_fp32(best_merged_state), out_path)
                saved_merged_path = out_path
                print(f"Saved merged model (alpha={best_alpha:.3f}) -> {out_path}")

        target_hash_after = _state_dict_sha256(target_base_sd)
        if target_hash_after != target_hash_before:
            raise RuntimeError(
                "Native target base was mutated during merge/transport preparation: "
                f"before={target_hash_before}, after={target_hash_after}."
            )

        final_summary = {
            "suite": suite_name,
            "tasks": tasks,
            "method": method.name,
            "method_label": method_label,
            "merge_mode": merge_mode,
            "merge_method": merge_method_name if merge_mode != "none" else None,
            "merge_params": merge_params if merge_mode != "none" else None,
            "target_hash_before": target_hash_before,
            "target_hash_after": target_hash_after,
            "strict_diagnostics": {"missing": 0, "failures": 0, "wrong_shape": 0},
            "single_transport_calibration": single_transport_calibration_metadata,
            "brace_calibration": brace_calibration_metadata,
            "block_extension_protocol": block_extension_protocol(block_extension_cfg),
            "base_construction": base_construction,
            "independent_endpoint_baseline": (
                {
                    "task_vector_definition": "tau_ind_t = ft_ind_t - base_ind_t",
                    "base_average_definition": "base_ind_avg = mean_t(base_ind_t)",
                    "base_average_key_scope": "common floating-point visual tensors",
                    "n_common_visual_keys": len(independent_base_average) if independent_base_average is not None else None,
                    "base_dispersion_ind": independent_base_dispersion,
                    "per_task_distance_to_mean_base": independent_base_distance_by_task,
                    "source_merge_direction_param_count": independent_source_merge_param_count,
                    "diagnostics_path": independent_base_diagnostics_path,
                    "direct_delta_key_count": independent_direct_delta_key_count,
                    "direct_endpoint_difference_used": base_construction == "independent_endpoint_average",
                }
                if base_construction == "independent_endpoint_average"
                else None
            ),
            "native_target_tasks": sorted(native_tasks) if native_tasks else [],
            "global_alpha_search": global_alpha_search if hierarchical else None,
            "per_task_premerge_alphas": (
                {item["task"]: float(per_task_premerge_alphas[i]) for i, item in enumerate(per_task)}
                if hierarchical and per_task_premerge_alphas is not None
                else None
            ),
            "hierarchical_premerge_alpha_curve": hierarchical_premerge_alpha_curve,
            "global_alpha_curve": global_alpha_curve,
            "alpha_selection": alpha_selection,
            "best_alpha": float(best_alpha),
            "best_baseline_alpha": float(best_baseline_alpha),
            "baseline_label": baseline_label,
            "metric_definitions": {
                "absolute_accuracy": "top-1 accuracy in [0, 1] (rebased/transported at the rebased's own best alpha)",
                "baseline_accuracy": "untransported baseline top-1 at the baseline's own best alpha",
                "normalized_accuracy_ratio": (
                    "absolute_accuracy (at rebased best alpha) / baseline_accuracy (at baseline best alpha); "
                    "each stream independently optimizes alpha on the alpha-search split"
                ),
                "normalized_accuracy_ratio_display": (
                    "ratio (decimal, not a percentage); values above 1.0 indicate the rebased/transported "
                    "model exceeds the untransported baseline; report as a decimal ratio, never multiplied by 100"
                ),
            },
            "test_results": {
                # Explicit names prevent a table exporter from treating a ratio as raw accuracy.
                "per_task_baseline_accuracy": {
                    item["task"]: float(baseline_test_accs[i]) for i, item in enumerate(per_task)
                },
                "per_task_absolute_accuracy": {
                    item["task"]: float(rebase_test_accs[i]) for i, item in enumerate(per_task)
                },
                "per_task_normalized_accuracy_ratio": {
                    item["task"]: float(norm_accs[i]) for i, item in enumerate(per_task)
                },
                "per_task_baseline": {item["task"]: float(baseline_test_accs[i]) for i, item in enumerate(per_task)},
                "per_task_rebased": {item["task"]: float(rebase_test_accs[i]) for i, item in enumerate(per_task)},
                "per_task_norm": {item["task"]: float(norm_accs[i]) for i, item in enumerate(per_task)},
                "avg_rebased": float(sum(rebase_test_accs) / len(rebase_test_accs)),
                "avg_norm": float(sum(norm_accs) / len(norm_accs)),
            },
            "single_tv_diagnostic": (
                {
                    "definition": (
                        "For each task t, evaluate target_base + alpha_t * transported/native task_vector_t "
                        "on task t's test set; alpha_t is selected on validation only."
                    ),
                    "alpha_protocol": single_tv_alpha_protocol,
                    "per_task_validation_alpha": {
                        item["task"]: float(single_tv_val_best_alpha[i])
                        for i, item in enumerate(per_task)
                    },
                    "per_task_validation_accuracy": {
                        item["task"]: float(single_tv_val_best_acc[i])
                        for i, item in enumerate(per_task)
                    },
                    "per_task_test_accuracy": {
                        item["task"]: float(single_tv_test_accs[i])
                        for i, item in enumerate(per_task)
                    },
                    "avg_test_accuracy": float(sum(single_tv_test_accs) / len(single_tv_test_accs)),
                    "merged_avg_test_accuracy": float(sum(rebase_test_accs) / len(rebase_test_accs)),
                    "merge_gap_single_minus_merged": float(
                        sum(single_tv_test_accs) / len(single_tv_test_accs)
                        - sum(rebase_test_accs) / len(rebase_test_accs)
                    ),
                }
                if single_tv_test_accs is not None
                else None
            ),
            "selected_alpha_by_task": {item["task"]: float(selected_alpha_by_task[i]) for i, item in enumerate(per_task)},
            "selected_baseline_alpha_by_task": {
                item["task"]: float(selected_baseline_alpha_by_task[i]) for i, item in enumerate(per_task)
            },
            "block_extension_target_dataset_eval": block_extension_eval_rows,
            "source_lmc": source_lmc_rows,
            "cross_task_source_lmc": cross_task_lmc_rows,
            "all_task_source_lmc": all_task_lmc_rows,
            "transported_artifacts": transported_artifacts,
            "transport_timings": transport_timings,
            # Always present (default {}) regardless of method/path, so a
            # downstream summary-JSON parser can read these keys uniformly
            # across every method, not only depth_alignment='discrete_index_match'
            # or method='direct_residual' runs.
            "alignment_calibration_timings": alignment_calibration_timings,
            "correction_fit_timings": correction_fit_timings,
            "direct_residual": (
                {
                    "config": asdict(direct_residual_cfg),
                    "diagnostics_by_task": direct_residual_diagnostics,
                    # Additive, analysis-only: both are None per task unless
                    # direct_residual_cfg.realization_diagnostics is set (see
                    # _run_direct_residual_fit / measure_direct_residual_realization
                    # / compute_direct_residual_task_vector_stats).
                    "realization_by_task": direct_residual_realization,
                    "task_vector_stats_by_task": direct_residual_task_vector_stats,
                    # Analysis-only Procrustes-alignment diagnostics (never
                    # fed to any fit), always populated regardless of
                    # config.residual_target or realization_diagnostics; see
                    # compute_alignment_diagnostics.
                    "alignment_diagnostics_by_task": direct_residual_alignment_diagnostics,
                    # Additive, analysis-only: None per task unless
                    # direct_residual_cfg.tv_scaling != "none" (see
                    # apply_tv_scaling / _run_direct_residual_fit). tv_scaling
                    # mode + iters are already carried by "config" above
                    # (asdict(direct_residual_cfg)); this key carries the
                    # per-task measurement (r_j traces, s_j / c, tau stats
                    # before/after).
                    "tv_scaling_by_task": direct_residual_tv_scaling,
                }
                if direct_residual_like
                else None
            ),
            # Reports whichever depth-alignment mode was active for a Theseus-/
            # BiCo-like method; always present so a downstream parser can rely
            # on the key, even though the default "ariadne" path never touches
            # anything new added by this change.
            "depth_alignment": depth_alignment_mode,
            "target_residual_completion": (
                {
                    "config": asdict(block_extension_cfg.target_residual_completion),
                    "diagnostics_by_task": residual_completion_diagnostics,
                }
                if block_extension_cfg.target_residual_completion.enabled
                else None
            ),
            "joint_blockwise_correction": (
                {
                    "config": asdict(block_extension_cfg.joint_blockwise_correction),
                    "diagnostics_by_task": joint_blockwise_diagnostics,
                }
                if block_extension_cfg.joint_blockwise_correction.enabled
                else None
            ),
            "direct_p1_correction": (
                {
                    "config": asdict(block_extension_cfg.direct_p1_correction),
                    "diagnostics_by_task": direct_p1_diagnostics,
                }
                if block_extension_cfg.direct_p1_correction.enabled
                else None
            ),
            "saved_merged_path": saved_merged_path,
        }
        run_logger.log_summary(final_summary)
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
