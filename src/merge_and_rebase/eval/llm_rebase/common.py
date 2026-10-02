from __future__ import annotations

from typing import Any

import torch

from merge_and_rebase.utils.helpers import parse_csv

from ...data.text_loaders import (
    NLI_TASKS,
    NLITaskData,
    default_head_class_ids_for_task,
)

NLI_SUITES: dict[str, tuple[str, ...]] = {
    "nli6": tuple(NLI_TASKS),
}


def resolve_tasks(tasks_raw: Any, *, suite_name: str | None = None) -> list[str]:
    allowed = list(NLI_TASKS) if suite_name is None else list(NLI_SUITES[suite_name])
    if tasks_raw is None:
        return allowed
    if isinstance(tasks_raw, str):
        if tasks_raw.strip().lower() == "all":
            return allowed
        tasks = [t.strip().lower() for t in parse_csv(tasks_raw)]
    elif isinstance(tasks_raw, (list, tuple)):
        tasks = [str(t).strip().lower() for t in tasks_raw]
    else:
        raise ValueError("tasks must be 'all', a CSV string, or a list.")

    bad = [t for t in tasks if t not in allowed]
    if bad:
        if suite_name is None:
            raise ValueError(f"Unknown tasks: {bad}. Supported: {list(NLI_TASKS)}")
        raise ValueError(f"Unknown tasks for suite '{suite_name}': {bad}. Allowed: {allowed}")
    return tasks


def resolve_suite_name(raw: Any) -> str | None:
    if raw is None:
        return None
    name = str(raw).strip().lower()
    if not name:
        return None
    if name not in NLI_SUITES:
        raise ValueError(f"Unknown suite '{name}'. Available: {sorted(NLI_SUITES)}")
    return name


def resolve_eval_mode(eval_mode: str, task_heads_path: str | None) -> str:
    mode = str(eval_mode).strip().lower()
    if mode == "auto":
        return "head_logits" if task_heads_path else "prompt"
    if mode not in {"prompt", "head_logits"}:
        raise ValueError("eval_mode must be one of: auto, prompt, head_logits")
    return mode


def default_prompt_for_task(task_data: NLITaskData) -> str:
    label_space = ", ".join(task_data.label_texts)
    return (
        "You are an NLI classifier.\n"
        f"Given a premise and a hypothesis, predict one label from: {label_space}.\n"
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n"
        "Label:"
    )


def resolve_task_mask_class(raw: Any) -> dict[str, int | None]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("task_mask_class must be a dict task->masked_class (or null).")
    out: dict[str, int | None] = {}
    for k, v in raw.items():
        key = str(k).strip().lower()
        if not key:
            continue
        out[key] = None if v is None else int(v)
    return out


def head_class_ids_for_task(
    *,
    task: str,
    task_num_labels: int,
    head_num_labels: int,
    masked_class: int | None,
) -> list[int]:
    if int(task_num_labels) <= 0:
        raise ValueError(f"Invalid task_num_labels for '{task}': {task_num_labels}")
    if int(head_num_labels) <= 0:
        raise ValueError(f"Invalid head_num_labels for '{task}': {head_num_labels}")

    if masked_class is None:
        if int(head_num_labels) == int(task_num_labels):
            return list(range(int(task_num_labels)))

        t = str(task).strip().lower()
        if int(task_num_labels) == 2 and int(head_num_labels) >= 3:
            if t in {"qnli", "rte"}:
                return [0, 2]
            if t == "scitail":
                return [0, 1]

        out = default_head_class_ids_for_task(task, num_labels=int(head_num_labels))
        if len(out) == int(task_num_labels):
            return out
        raise ValueError(
            f"Could not infer head_class_ids for task '{task}': "
            f"task_num_labels={task_num_labels}, head_num_labels={head_num_labels}. "
            "Set config['task_mask_class'] explicitly."
        )

    masked = int(masked_class)
    if masked < 0 or masked >= int(head_num_labels):
        raise ValueError(
            f"Invalid masked class for task '{task}': {masked}. Allowed range is [0, {int(head_num_labels) - 1}]"
        )
    keep = [c for c in range(int(head_num_labels)) if c != masked]
    if len(keep) != int(task_num_labels):
        raise ValueError(
            f"Mask-derived class ids for task '{task}' are incompatible: keep={keep}, "
            f"task_num_labels={task_num_labels}, head_num_labels={head_num_labels}."
        )
    return keep


def load_task_heads(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"task_heads file must contain a dict. Got: {type(obj)}")
    out: dict[str, Any] = {}
    for k, v in obj.items():
        out[str(k).strip().lower()] = v
    return out


def task_head_tensor_for_param(
    *,
    task_key: str,
    name: str,
    param: torch.Tensor,
    value: torch.Tensor,
    head_class_ids: list[int] | None = None,
) -> torch.Tensor:
    tgt = param.detach().clone()
    src = value.to(device=tgt.device, dtype=tgt.dtype)
    if tuple(src.shape) == tuple(tgt.shape):
        return src

    mapped_class_ids: list[int] | None = None
    if head_class_ids is not None:
        mapped_class_ids = [int(x) for x in head_class_ids]
        if len(set(mapped_class_ids)) != len(mapped_class_ids):
            raise ValueError(f"head_class_ids for task '{task_key}' must be unique. Got: {mapped_class_ids}")

    if mapped_class_ids is not None:
        if (
            name.endswith("classification_head.out_proj.weight")
            and src.ndim == 2
            and tgt.ndim == 2
            and src.shape[0] == len(mapped_class_ids)
            and src.shape[1] == tgt.shape[1]
        ):
            if min(mapped_class_ids) < 0 or max(mapped_class_ids) >= tgt.shape[0]:
                raise ValueError(
                    f"head_class_ids out of range for task '{task_key}', param '{name}': "
                    f"ids={mapped_class_ids}, target_rows={tgt.shape[0]}"
                )
            for i, cls_id in enumerate(mapped_class_ids):
                tgt[int(cls_id)].copy_(src[i])
            return tgt
        if (
            name.endswith("classification_head.out_proj.bias")
            and src.ndim == 1
            and tgt.ndim == 1
            and src.shape[0] == len(mapped_class_ids)
        ):
            if min(mapped_class_ids) < 0 or max(mapped_class_ids) >= tgt.shape[0]:
                raise ValueError(
                    f"head_class_ids out of range for task '{task_key}', param '{name}': "
                    f"ids={mapped_class_ids}, target_rows={tgt.shape[0]}"
                )
            for i, cls_id in enumerate(mapped_class_ids):
                tgt[int(cls_id)].copy_(src[i])
            return tgt

    if name.endswith("classification_head.out_proj.weight"):
        if src.ndim == 2 and tgt.ndim == 2 and src.shape[1] == tgt.shape[1] and src.shape[0] < tgt.shape[0]:
            tgt[: src.shape[0]].copy_(src)
            return tgt
    if name.endswith("classification_head.out_proj.bias"):
        if src.ndim == 1 and tgt.ndim == 1 and src.shape[0] < tgt.shape[0]:
            tgt[: src.shape[0]].copy_(src)
            return tgt

    raise ValueError(
        f"Head shape mismatch for task '{task_key}', param '{name}': "
        f"model={tuple(tgt.shape)} payload={tuple(src.shape)}"
    )


def task_head_param_overrides(
    *,
    model: Any,
    task: str,
    task_heads: dict[str, Any],
    head_key_pattern: str,
    head_class_ids: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    task_key = str(task).strip().lower()
    if task_key not in task_heads:
        raise KeyError(f"Task '{task_key}' not found in task_heads.")
    payload = task_heads[task_key]
    named_params = {n: p for n, p in model.named_parameters()}
    pattern = str(head_key_pattern)
    out: dict[str, torch.Tensor] = {}

    def _add_override(name: str, param: torch.Tensor, value: torch.Tensor) -> None:
        out[name] = task_head_tensor_for_param(
            task_key=task_key,
            name=name,
            param=param,
            value=value,
            head_class_ids=head_class_ids,
        )

    if isinstance(payload, torch.Tensor):
        cands = [(n, p) for n, p in named_params.items() if pattern in n and tuple(p.shape) == tuple(payload.shape)]
        if not cands:
            by_shape = [(n, p) for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
            preferred = [
                (n, p)
                for n, p in by_shape
                if n.endswith("score.weight")
                or n.endswith("classifier.weight")
                or n.endswith("classification_head.weight")
            ]
            if len(preferred) == 1:
                cands = preferred
            elif len(by_shape) == 1:
                cands = by_shape
        if len(cands) != 1:
            names = [n for n, _ in cands]
            shape_only = [n for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
            raise ValueError(
                f"Could not uniquely match tensor head for task '{task_key}'. "
                f"pattern='{pattern}', shape={tuple(payload.shape)}, candidates={names}, "
                f"shape_only_matches={shape_only[:8]}"
            )
        name, param = cands[0]
        _add_override(name, param, payload)
        return out

    if not isinstance(payload, dict):
        raise ValueError(f"task_heads['{task_key}'] must be a Tensor or dict. Got: {type(payload)}")

    for hk, hv in payload.items():
        if not isinstance(hv, torch.Tensor):
            continue
        key = str(hk)
        if key in named_params:
            _add_override(key, named_params[key], hv)
            continue

        suffix_matches = [(n, p) for n, p in named_params.items() if pattern in n and n.endswith(key)]
        if len(suffix_matches) == 1:
            n, p = suffix_matches[0]
            _add_override(n, p, hv)
            continue
        if len(suffix_matches) == 0:
            any_suffix_matches = [(n, p) for n, p in named_params.items() if n.endswith(key)]
            if len(any_suffix_matches) == 1:
                n, p = any_suffix_matches[0]
                _add_override(n, p, hv)
                continue
        if len(suffix_matches) > 1:
            raise ValueError(
                f"Ambiguous suffix match for task '{task_key}', key='{key}', "
                f"matches={[n for n, _ in suffix_matches]}"
            )
        raise KeyError(f"No parameter match for task '{task_key}' head key '{key}'.")
    return out


def inject_task_head(
    *,
    model: Any,
    task: str,
    task_heads: dict[str, Any],
    head_key_pattern: str,
    head_class_ids: list[int] | None = None,
) -> None:
    task_key = str(task).strip().lower()
    if task_key not in task_heads:
        raise KeyError(f"Task '{task_key}' not found in task_heads.")
    payload = task_heads[task_key]

    named_params = {n: p for n, p in model.named_parameters()}
    pattern = str(head_key_pattern)

    def _copy_param(name: str, param: torch.Tensor, value: torch.Tensor) -> None:
        param.copy_(
            task_head_tensor_for_param(
                task_key=task_key,
                name=name,
                param=param,
                value=value,
                head_class_ids=head_class_ids,
            )
        )

    with torch.no_grad():
        if isinstance(payload, torch.Tensor):
            cands = [(n, p) for n, p in named_params.items() if pattern in n and tuple(p.shape) == tuple(payload.shape)]
            if not cands:
                by_shape = [(n, p) for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
                preferred = [
                    (n, p)
                    for n, p in by_shape
                    if n.endswith("score.weight")
                    or n.endswith("classifier.weight")
                    or n.endswith("classification_head.weight")
                ]
                if len(preferred) == 1:
                    cands = preferred
                elif len(by_shape) == 1:
                    cands = by_shape
            if len(cands) != 1:
                names = [n for n, _ in cands]
                shape_only = [n for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
                raise ValueError(
                    f"Could not uniquely match tensor head for task '{task_key}'. "
                    f"pattern='{pattern}', shape={tuple(payload.shape)}, candidates={names}, "
                    f"shape_only_matches={shape_only[:8]}"
                )
            name, param = cands[0]
            _copy_param(name, param, payload)
            return

        if not isinstance(payload, dict):
            raise ValueError(f"task_heads['{task_key}'] must be a Tensor or dict. Got: {type(payload)}")

        for hk, hv in payload.items():
            if not isinstance(hv, torch.Tensor):
                continue
            key = str(hk)
            if key in named_params:
                p = named_params[key]
                _copy_param(key, p, hv)
                continue

            suffix_matches = [(n, p) for n, p in named_params.items() if pattern in n and n.endswith(key)]
            if len(suffix_matches) == 1:
                n, p = suffix_matches[0]
                _copy_param(n, p, hv)
                continue
            if len(suffix_matches) == 0:
                any_suffix_matches = [(n, p) for n, p in named_params.items() if n.endswith(key)]
                if len(any_suffix_matches) == 1:
                    n, p = any_suffix_matches[0]
                    _copy_param(n, p, hv)
                    continue
            if len(suffix_matches) > 1:
                raise ValueError(
                    f"Ambiguous suffix match for task '{task_key}', key='{key}', "
                    f"matches={[n for n, _ in suffix_matches]}"
                )
            raise KeyError(f"No parameter match for task '{task_key}' head key '{key}'.")


def to_unit_acc(ref_acc: float) -> float:
    v = float(ref_acc)
    if v > 1.0:
        v = v / 100.0
    return v


def normalized_acc(acc: float, fine_tuned_acc_ref: float) -> float:
    denom = to_unit_acc(fine_tuned_acc_ref)
    if denom <= 0:
        return 0.0
    return float(acc) / denom


def resolve_fine_tuned_acc(
    *,
    cfg: dict[str, Any],
    tasks: list[str],
) -> dict[str, float] | None:
    raw = cfg.get("fine_tuned_acc", None)
    if isinstance(raw, dict):
        out: dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k).strip().lower()] = float(v)
            except Exception:
                continue
        missing = [t for t in tasks if t not in out]
        if missing:
            print(
                f"[warn] fine_tuned_acc missing tasks {missing}. Normalized accuracy will be skipped for missing tasks."
            )
        return out
    if raw is None:
        return None
    raise ValueError("fine_tuned_acc must be a dict when provided.")
