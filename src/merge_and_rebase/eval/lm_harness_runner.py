from __future__ import annotations

from typing import Any

import torch.nn as nn


def run(
    tasks: list[str],
    model: nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    num_fewshot: int = 0,
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
    num_fewshot : Number of few-shot examples.
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
    except ImportError:
        raise ImportError(
            "lm-eval harness requires `lm-eval` to be installed. "
            "Install with: pip install 'merge-and-rebase[harness]'"
        ) from None

    model.eval()
    if hasattr(model, "to"):
        model.to(device)

    try:
        results = simple_evaluate(
            model="hf",
            model_args={
                "pretrained": model,
                "tokenizer": tokenizer,
            },
            tasks=list(tasks),
            num_fewshot=int(num_fewshot),
            batch_size=batch_size,
            device=device,
            limit=limit,
        )
    except TypeError:
        results = simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=list(tasks),
            num_fewshot=int(num_fewshot),
            batch_size=batch_size,
            device=device,
            limit=limit,
        )

    out: dict[str, float] = {}
    if results is None:
        return out

    for task_name, task_results in results.get("results", {}).items():
        for key, value in task_results.items():
            # keys look like "exact_match,none", "acc_norm,none", "exact_match_stderr,none", ...
            metric, _, _filter_name = str(key).partition(",")
            if metric.endswith("_stderr") or not isinstance(value, int | float):
                continue
            out[f"{task_name}_{metric}"] = float(value)

    return out
