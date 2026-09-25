from __future__ import annotations

import dataclasses
import os
from pathlib import Path
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
            metric, _, filter_name = str(key).partition(",")
            if metric in _NON_METRIC_KEYS or metric.endswith("_stderr") or not isinstance(value, int | float):
                continue
            # A task with several filters (gsm8k: "strict-match" and
            # "flexible-extract") reports the same metric once per filter. Keying
            # on the metric alone let the last filter silently overwrite the
            # others, so gsm8k_exact_match was flexible-extract only. Name the
            # filter unless it is the default "none", which keeps every
            # single-filter task's key unchanged.
            suffix = "" if filter_name in ("", "none") else f"_{filter_name}"
            out[f"{task_name}_{metric}{suffix}"] = float(value)
    return out


#: Opt-in: when set, every `run()` writes the generations lm-eval already
#: collected to this directory. Off by default, since nothing here read them and
#: a sweep calls `run()` once per alpha. It exists because a string-matched
#: metric (harmbench_refusal) cannot be audited from its aggregate alone.
SAMPLES_DIR_ENV = "MR_HARNESS_SAMPLES_DIR"

_samples_call_index = 0


def _dump_generation_samples(results: dict[str, Any] | None, out_dir: str) -> None:
    """Write one JSONL per generate_until task: prompt, response, metrics.

    Files are numbered by call order within the process, so in a sweep file
    NNN matches the NNN-th entry of `search_results` (the final best-alpha
    re-eval comes last). Each record carries its own metric values, so the
    mapping can be checked against the aggregates. Loglikelihood tasks (arc,
    mmlu) are skipped: their "responses" are scores, not text.
    """
    global _samples_call_index
    import json

    idx = _samples_call_index
    _samples_call_index += 1
    if not results or not results.get("samples"):
        return
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    for task_name, rows in results["samples"].items():
        if not rows or not isinstance((rows[0].get("filtered_resps") or [None])[0], str):
            continue
        with open(path / f"{idx:03d}_{task_name}.jsonl", "w") as f:
            for row in rows:
                record = {
                    "doc_id": row.get("doc_id"),
                    "prompt": (row.get("arguments") or [[None]])[0][0],
                    "response": row["filtered_resps"][0],
                }
                record.update({m: row.get(m) for m in row.get("metrics", [])})
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def _samples_for(
    samples: dict[str, list[int]] | None, group_tasks: list[str]
) -> dict[str, dict[str, list[int]]]:
    """Restrict `samples` to the tasks in this few-shot group.

    lm-eval rejects a `samples` mapping naming a task it was not asked to run,
    and tasks are dispatched one few-shot group at a time, so the mapping has
    to be narrowed per call. Returns {} when nothing applies, which keeps the
    kwarg off the call entirely for older lm-eval versions.
    """
    if not samples:
        return {}
    scoped = {t: samples[t] for t in group_tasks if t in samples}
    return {"samples": scoped} if scoped else {}


def _check_samples_conflict(limit: int | None, samples: dict[str, list[int]] | None) -> None:
    """lm-eval rejects `limit` and `samples` together; fail with the reason why."""
    if limit is not None and samples:
        raise ValueError(
            "harness_limit cannot be combined with a held-out calibration slice: "
            "lm-eval accepts either 'limit' or explicit 'samples', not both. "
            "Set harness_limit to null, or point "
            "block_extension_params.calibration_dataset at a separate corpus so "
            "no eval docs are held out."
        )


def run(
    tasks: list[str],
    model: nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    num_fewshot: int | list[int] | dict[str, int] = 0,
    batch_size: str = "auto",
    limit: int | None = None,
    samples: dict[str, list[int]] | None = None,
    apply_chat_template: bool = False,
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
    samples : Optional explicit doc indices to score per task. Used to keep
        the scored docs disjoint from the calibration slice carved out of
        the same task (see `data.llm_calibration`).
    apply_chat_template : Wrap each prompt in the tokenizer's chat template.
        Off by default, because turning it on changes the prompt every task
        sees and so is not comparable with any run made without it. It exists
        for the `*_instruct` task variants, whose `gen_prefix` assumes an
        assistant turn to continue and which score 0 without one. Both the
        source and the target tokenizer must carry a chat template; Qwen2.5
        base checkpoints do.

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

    _check_samples_conflict(limit, samples)

    # humaneval/mbpp (and similar) set `unsafe_code: true` and lm-eval refuses
    # to run them unless explicitly confirmed, since scoring executes
    # model-generated code. Opt in via HF_ALLOW_CODE_EVAL=1 (the standard
    # lm-eval-harness/human-eval convention) rather than defaulting this on.
    confirm_run_unsafe_code = os.environ.get("HF_ALLOW_CODE_EVAL") == "1"

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
                    confirm_run_unsafe_code=confirm_run_unsafe_code,
                    apply_chat_template=apply_chat_template,
                    **_samples_for(samples, group_tasks),
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
                    confirm_run_unsafe_code=confirm_run_unsafe_code,
                    apply_chat_template=apply_chat_template,
                    **_samples_for(samples, group_tasks),
                )
            out.update(_extract_metrics(results))
            if os.environ.get(SAMPLES_DIR_ENV):
                _dump_generation_samples(results, os.environ[SAMPLES_DIR_ENV])
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
