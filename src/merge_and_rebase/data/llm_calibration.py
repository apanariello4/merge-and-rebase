"""Dataset-backed calibration text for LLM block extension and transport.

Block extension and the Theseus/BiCo transports fit their maps on activations
collected from a calibration corpus. The vision path has always drawn that
corpus from a real dataset (`build_vision_calibration_loader`, driven by
`block_extension_params.calibration_dataset` / `calibration_split`); 
this module is the LLM counterpart. Calibration text is resolved, in order, from:

1. explicit `calibration_prompts` in the config (escape hatch for a curated
   bank),
2. `block_extension_params.calibration_dataset` -- an HF dataset path or spec,
   mirroring the vision path,
3. the evaluated lm-harness task(s), rendered through the task's own
   `doc_to_text` so calibration sees exactly the prompt format eval scores.

For (3) the docs are split deterministically into a calibration slice and an
eval slice. The eval slice is handed back as explicit doc indices so the
harness can be told to score only those, keeping the two sets disjoint even
for a single-split benchmark like ifeval.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

_TEXT_COLUMN_CANDIDATES = (
    "text",
    "prompt",
    "question",
    "content",
    "sentence",
    "instruction",
    "input",
    "article",
    "document",
)


class CalibrationTexts:
    """Resolved calibration corpus plus the eval doc indices it leaves free.

    `eval_samples` maps an lm-harness task name to the doc indices that were
    NOT used for calibration. It is empty whenever calibration came from a
    source independent of the evaluated task, in which case the harness should
    score the task in full.
    """

    def __init__(
        self,
        texts: list[str],
        *,
        source: str,
        eval_samples: dict[str, list[int]] | None = None,
    ) -> None:
        if not texts:
            raise ValueError(f"Calibration source {source!r} produced no text.")
        self.texts = texts
        self.source = source
        self.eval_samples = dict(eval_samples or {})

    def __len__(self) -> int:
        return len(self.texts)

    def describe(self) -> str:
        held_out = ", ".join(
            f"{task}: {len(idx)} eval docs" for task, idx in sorted(self.eval_samples.items())
        )
        suffix = f" (held out -> {held_out})" if held_out else ""
        return f"{len(self.texts)} sequences from {self.source}{suffix}"


def resolve_calibration_texts(
    *,
    prompts: Sequence[str] | None = None,
    calibration_dataset: str | Mapping[str, Any] | None = None,
    calibration_split: str = "val",
    harness_tasks: Sequence[str] | None = None,
    n_sequences: int,
    seed: int = 0,
) -> CalibrationTexts:
    """Resolve the calibration corpus for an LLM run.

    Parameters
    ----------
    prompts : Explicit prompt bank from `config['calibration_prompts']`.
    calibration_dataset : HF dataset path, or a spec mapping with `path` and
        optional `name`/`split`/`text_column`.
    calibration_split : Split to calibrate on. For an lm-harness source this
        selects the held-out slice ("val"/"validation") rather than a split
        that has to exist upstream, so it also works for single-split tasks.
    harness_tasks : Evaluated lm-harness tasks, used as the default source.
    n_sequences : How many sequences the run will actually consume
        (`n_batches * batch_size`). The calibration slice is sized to cover
        this where the source allows it.
    seed : Seed for the deterministic calibration/eval partition.
    """
    if n_sequences <= 0:
        raise ValueError("n_sequences must be > 0.")

    if prompts:
        return CalibrationTexts(
            [str(p) for p in prompts], source="config['calibration_prompts']"
        )

    if calibration_dataset is not None:
        return _from_hf_dataset(
            calibration_dataset,
            calibration_split=calibration_split,
            n_sequences=n_sequences,
        )

    if harness_tasks:
        return _from_harness_tasks(
            list(harness_tasks),
            calibration_split=calibration_split,
            n_sequences=n_sequences,
            seed=seed,
        )

    raise ValueError(
        "No calibration source configured. Set one of: "
        "config['calibration_prompts'], "
        "config['block_extension_params']['calibration_dataset'], "
        "or config['harness_tasks']."
    )


def _from_hf_dataset(
    spec: str | Mapping[str, Any],
    *,
    calibration_split: str,
    n_sequences: int,
) -> CalibrationTexts:
    import datasets

    if isinstance(spec, str):
        spec = {"path": spec}
    if not isinstance(spec, Mapping):
        raise TypeError("calibration_dataset must be a dataset path or a mapping spec.")

    path = spec.get("path", spec.get("dataset", None))
    if not path:
        raise ValueError("calibration_dataset spec needs a 'path'.")
    name = spec.get("name", spec.get("config", None))
    split = str(spec.get("split", calibration_split))
    text_column = spec.get("text_column", None)

    ds = (
        datasets.load_dataset(str(path), str(name), split=split)
        if name
        else datasets.load_dataset(str(path), split=split)
    )

    column = text_column or _pick_text_column(ds.column_names)
    if column is None:
        raise ValueError(
            f"Could not infer a text column for calibration_dataset {path!r} "
            f"(columns: {ds.column_names}). Set 'text_column' explicitly."
        )

    texts: list[str] = []
    for row in ds:
        value = row.get(column, None)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        if len(texts) >= n_sequences:
            break

    return CalibrationTexts(texts, source=f"{path}[{split}].{column}")


def _pick_text_column(columns: Sequence[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in _TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _from_harness_tasks(
    tasks: list[str],
    *,
    calibration_split: str,
    n_sequences: int,
    seed: int,
) -> CalibrationTexts:
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    wants_holdout = str(calibration_split).lower() in {"val", "validation", "dev"}

    texts: list[str] = []
    eval_samples: dict[str, list[int]] = {}
    per_task = max(1, -(-n_sequences // max(1, len(tasks))))

    for task_name in tasks:
        loaded = manager.load_task_or_group([task_name])
        task = loaded.get(task_name, None)
        if task is None or not hasattr(task, "doc_to_text"):
            # A group expands to several subtasks; calibrating on a group's
            # aggregate is ambiguous, so skip rather than guess.
            continue

        docs = list(_task_docs(task))
        if not docs:
            continue

        order = list(range(len(docs)))
        random.Random(seed).shuffle(order)

        if wants_holdout:
            n_calib = min(per_task, max(1, len(order) - 1))
            calib_idx = sorted(order[:n_calib])
            eval_idx = sorted(order[n_calib:])
            eval_samples[task_name] = eval_idx
        else:
            calib_idx = sorted(order[:per_task])

        for i in calib_idx:
            rendered = task.doc_to_text(docs[i])
            if isinstance(rendered, str) and rendered.strip():
                texts.append(rendered)

    if not texts:
        raise ValueError(
            f"lm-harness tasks {tasks} yielded no calibration text. "
            "Point block_extension_params.calibration_dataset at a dataset instead."
        )

    split_label = "holdout" if wants_holdout else str(calibration_split)
    return CalibrationTexts(
        texts,
        source=f"lm-harness {'+'.join(tasks)}[{split_label}]",
        eval_samples=eval_samples,
    )


def _task_docs(task: Any) -> Sequence[Any]:
    """Return exactly the docs lm-eval would score, in lm-eval's own order.

    The indices handed back as `eval_samples` are resolved by lm-eval against
    `Task.eval_docs` (`build_all_requests` filters `enumerate(self.eval_docs)`),
    so the calibration slice has to be carved from that same sequence. Reading
    `eval_docs` rather than reimplementing its test/validation preference keeps
    the two in lockstep if lm-eval ever changes that preference.
    """
    try:
        return list(task.eval_docs)
    except (AttributeError, ValueError, KeyError):
        # Task has neither test nor validation docs; lm-eval could not score it
        # either, so there is no index space to stay disjoint from.
        return []
