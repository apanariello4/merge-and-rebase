from __future__ import annotations

import dataclasses
from typing import Any

import torch.nn as nn


def _patched_task_config_to_dict(self, keep_callable: bool = False) -> dict:
    """
    Drop-in replacement for lm_eval's TaskConfig.to_dict that avoids
    dataclasses.asdict()'s recursive copy.deepcopy of every field.

    For group tasks with many subtasks (e.g. "mmlu", 57 subjects) that deep
    copy ends up cloning the live model object passed in via model_args once
    per subtask (observed as a torch.nn.Parameter.__deepcopy__ crash deep
    inside dataclasses.asdict), exhausting GPU memory even for a 1.5B model
    on a 64GB A100. This builds the same dict shape via a shallow field copy
    instead, which is all lm_eval actually needs here -- the result is only
    used for the "configs" provenance section of simple_evaluate's return
    value, never for computing scores, and this repo's run() below never
    reads that section.
    """
    cfg_dict = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    for k, v in list(cfg_dict.items()):
        if v is None:
            cfg_dict.pop(k)
        elif k == "metric_list":
            v = [dict(d) for d in v]
            for metric_dict in v:
                for metric_key, metric_value in metric_dict.items():
                    if callable(metric_value):
                        metric_dict[metric_key] = self.serialize_function(metric_value, keep_callable=keep_callable)
            cfg_dict[k] = v
        elif callable(v):
            cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
    return cfg_dict


# Fixed bookkeeping keys lm-eval attaches to every task's result dict (see
# lm_eval.result_schema._TaskMetrics) that are not accuracy metrics and must
# never be averaged in with real scores.
_NON_METRIC_KEYS = frozenset({"name", "alias", "sample_len", "sample_count"})


def _resolve_fewshot_by_task(
    tasks: list[str], num_fewshot: int | list[int] | dict[str, int]
) -> dict[str, int]:
    """Normalize the various `num_fewshot` shapes to one value per task."""
    if isinstance(num_fewshot, dict):
        missing = [t for t in tasks if t not in num_fewshot]
        if missing:
            raise ValueError(f"num_fewshot dict is missing entries for tasks: {missing}")
        return {t: int(num_fewshot[t]) for t in tasks}
    if isinstance(num_fewshot, (list, tuple)):
        if len(num_fewshot) != len(tasks):
            raise ValueError(
                f"num_fewshot list has {len(num_fewshot)} entries but there are "
                f"{len(tasks)} tasks; they must be parallel lists."
            )
        return {t: int(n) for t, n in zip(tasks, num_fewshot, strict=True)}
    return {t: int(num_fewshot) for t in tasks}


def _extract_metrics(results: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not results:
        return out
    for task_name, task_results in results.get("results", {}).items():
        for key, value in task_results.items():
            # keys look like "exact_match,none", "acc_norm,none", "exact_match_stderr,none", ...
            metric, _, _filter_name = str(key).partition(",")
            if metric in _NON_METRIC_KEYS or metric.endswith("_stderr") or not isinstance(value, int | float):
                continue
            out[f"{task_name}_{metric}"] = float(value)
    return out


def run(
    tasks: list[str],
    model: nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    num_fewshot: int | list[int] | dict[str, int] = 0,
    batch_size: str = "auto",
    limit: int | None = None,
) -> dict[str, float]:
    """
    Evaluate a causal-LM model on lm-eval harness tasks.

    Parameters
    ----------
    tasks : List of task names (e.g. "hellaswag", "piqa", etc.)
    model : A HuggingFace causal-LM model.
    tokenizer : Corresponding tokenizer.
    device : Target device string.
    num_fewshot : Number of few-shot examples. Either a single int applied to
        every task, or a list/dict giving a per-task few-shot count (e.g.
        arc_easy=0, arc_challenge=25, mmlu=5, ...). Tasks sharing the same
        count are batched into a single `simple_evaluate` call.
    batch_size : Batch size ("auto" or int).
    limit : Optional max eval examples per task.

    Returns
    -------
    dict[str, float] : Flat "{task}_{metric}" -> value dict, for every
        non-stderr metric lm-eval reports for each task (e.g. "acc", "acc_norm",
        "exact_match", "math_verify" ...), whichever apply to the given tasks.
    """
    try:
        from lm_eval import simple_evaluate
        from lm_eval.config.task import TaskConfig
    except ImportError:
        raise ImportError(
            "lm-eval harness requires `lm-eval` to be installed. "
            "Install with: pip install 'merge-and-rebase[harness]'"
        ) from None

    model.eval()
    if hasattr(model, "to"):
        model.to(device)

    fewshot_by_task = _resolve_fewshot_by_task(list(tasks), num_fewshot)
    groups: dict[int, list[str]] = {}
    for t in tasks:
        groups.setdefault(fewshot_by_task[t], []).append(t)

    # See _patched_task_config_to_dict docstring: avoids an OOM on group
    # tasks (e.g. "mmlu") caused by lm_eval deep-copying the live model once
    # per subtask while dumping config provenance we never read.
    _original_to_dict = TaskConfig.to_dict
    TaskConfig.to_dict = _patched_task_config_to_dict
    try:
        out: dict[str, float] = {}
        for n_shot, group_tasks in groups.items():
            try:
                results = simple_evaluate(
                    model="hf",
                    model_args={
                        "pretrained": model,
                        "tokenizer": tokenizer,
                    },
                    tasks=list(group_tasks),
                    num_fewshot=n_shot,
                    batch_size=batch_size,
                    device=device,
                    limit=limit,
                )
            except TypeError:
                results = simple_evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    tasks=list(group_tasks),
                    num_fewshot=n_shot,
                    batch_size=batch_size,
                    device=device,
                    limit=limit,
                )
            out.update(_extract_metrics(results))
    finally:
        TaskConfig.to_dict = _original_to_dict

    return out


def score_by_task(results: dict[str, float], tasks: list[str]) -> float:
    """
    Average harness metrics into one score, weighting each requested task
    equally regardless of how many raw metric keys it expands into.

    Without this, a group task like "mmlu" (57 subjects + 5 aggregates, all
    keyed "mmlu_*") would dominate a flat average over every "{task}_{metric}"
    key, swamping single-metric tasks like "gsm8k". Each task's own metrics
    (e.g. "acc" and "acc_norm" for arc_*) are still averaged together first.
    """
    task_scores: list[float] = []
    for t in tasks:
        prefix = f"{t}_"
        vals = [v for k, v in results.items() if k.startswith(prefix)]
        if vals:
            task_scores.append(sum(vals) / len(vals))
    return sum(task_scores) / len(task_scores) if task_scores else 0.0
