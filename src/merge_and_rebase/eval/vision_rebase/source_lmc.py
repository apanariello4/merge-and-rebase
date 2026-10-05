"""Source-model linear-mode-connectivity evaluation helpers for the vision rebase entrypoint."""

from __future__ import annotations

import itertools
import os
from typing import Any

import torch
import torch.nn.functional as F

from ...eval.utils import resolve_eval_split_loader
from ...io.ckpt import load_into_model
from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

_ZERO_SHOT_CACHE_DIR = os.environ.get("BRACE_ZS_CACHE_DIR", "src/.cache/zs_cache")


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
    if (
        not alphas
        or not any(abs(float(a)) < 1e-8 for a in alphas)
        or not any(abs(float(a) - 1.0) < 1e-8 for a in alphas)
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
            print(f"    source LMC alpha={float(alpha):.3f} acc={accuracies[-1]:.6f} loss={losses[-1]:.6f}")
    finally:
        load_into_model(model, restore_sd, strict=False)

    idx0 = min(range(len(alphas)), key=lambda i: abs(float(alphas[i])))
    idx1 = min(range(len(alphas)), key=lambda i: abs(float(alphas[i]) - 1.0))
    loss_chord = [(1.0 - float(alpha)) * losses[idx0] + float(alpha) * losses[idx1] for alpha in alphas]
    errors = [1.0 - acc for acc in accuracies]
    error_chord = [(1.0 - float(alpha)) * errors[idx0] + float(alpha) * errors[idx1] for alpha in alphas]
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
    if (
        not alphas
        or not any(abs(float(a)) < 1e-8 for a in alphas)
        or not any(abs(float(a) - 1.0) < 1e-8 for a in alphas)
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
                key: (
                    torch.lerp(value_a, endpoint_b_sd[key], float(alpha))
                    if torch.is_floating_point(value_a)
                    else value_a
                )
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
    loss_chord = [(1.0 - float(alpha)) * average_loss[idx0] + float(alpha) * average_loss[idx1] for alpha in alphas]
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
        key: sum(
            (endpoint_states[task][key].float() for task in tasks),
            start=torch.zeros_like(endpoint_states[tasks[0]][key], dtype=torch.float32),
        )
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
    per_task = {task: max(ray["per_task_max_loss_barrier"][task] for ray in rays.values()) for task in tasks}
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
