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

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

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
        notes: list[str] | None = None,
        decoupled: bool = False,
    ) -> None:
        if not texts:
            raise ValueError(f"Calibration source {source!r} produced no text.")
        self.texts = texts
        self.source = source
        self.eval_samples = dict(eval_samples or {})
        #: Recorded (not raised) deviations, e.g. a harness task without a gold target under ``include_target``.
        self.notes = list(notes or [])
        #: True when the corpus was chosen independently of the evaluated task (so no eval hold-out exists).
        self.decoupled = bool(decoupled)

    def provenance(self) -> dict[str, Any]:
        """Additive run-summary record: corpus fingerprint, size, hold-out hash and any recorded deviations."""
        holdout = json.dumps({k: list(v) for k, v in sorted(self.eval_samples.items())}, sort_keys=True)
        return {
            "source": self.source,
            "n_sequences": len(self.texts),
            "corpus_sha256": hashlib.sha256("\x00".join(self.texts).encode()).hexdigest(),
            "holdout_sha256": hashlib.sha256(holdout.encode()).hexdigest() if self.eval_samples else None,
            "holdout_sizes": {k: len(v) for k, v in sorted(self.eval_samples.items())},
            "decoupled_from_eval": self.decoupled,
            "notes": list(self.notes),
        }

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
    include_target: bool = False,
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
    include_target : Harness source only (default off): append each doc's gold target (``doc_to_target``; an int
        index is mapped through ``doc_to_choice``) to its rendered prompt. Tasks without a gold target fall back to
        the prompt alone and the fallback is recorded in ``CalibrationTexts.notes``.
    """
    if n_sequences <= 0:
        raise ValueError("n_sequences must be > 0.")

    if prompts:
        bad = [i for i, p in enumerate(prompts) if p is None or not str(p).strip()]
        if bad:
            raise ValueError(f"config['calibration_prompts'] has empty entries at indices {bad[:10]}.")
        return CalibrationTexts(
            [str(p) for p in prompts], source="config['calibration_prompts']", decoupled=True
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
            include_target=include_target,
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
    text_columns = spec.get("text_columns", None)
    text_template = spec.get("text_template", None)
    if text_columns is not None and text_column is not None:
        raise ValueError("calibration_dataset: set only one of 'text_column' and 'text_columns'.")

    ds = (
        datasets.load_dataset(str(path), str(name), split=split)
        if name
        else datasets.load_dataset(str(path), split=split)
    )

    if text_columns is not None:
        # Opt-in multi-column rendering (e.g. question + answer). Default (single text column) is unchanged.
        columns = [str(c) for c in text_columns]
        missing = [c for c in columns if c not in ds.column_names]
        if not columns or missing:
            raise ValueError(f"calibration_dataset text_columns {columns} not all in {ds.column_names}.")
        template = str(text_template) if text_template is not None else "\n".join("{" + c + "}" for c in columns)
        texts = []
        for row in ds:
            values = {c: row.get(c, None) for c in columns}
            if all(isinstance(v, str) and v.strip() for v in values.values()):
                texts.append(template.format(**values))
            if len(texts) >= n_sequences:
                break
        return CalibrationTexts(texts, source=f"{path}[{split}].{'+'.join(columns)}", decoupled=True)

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

    return CalibrationTexts(texts, source=f"{path}[{split}].{column}", decoupled=True)


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
    include_target: bool = False,
) -> CalibrationTexts:
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    wants_holdout = str(calibration_split).lower() in {"val", "validation", "dev"}

    texts: list[str] = []
    notes: list[str] = []
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

        n_without_target = 0
        for i in calib_idx:
            rendered = task.doc_to_text(docs[i])
            if isinstance(rendered, str) and rendered.strip():
                if include_target:
                    target = _gold_target_text(task, docs[i])
                    if target is None:
                        n_without_target += 1
                    else:
                        rendered = f"{rendered}{getattr(getattr(task, 'config', None), 'target_delimiter', ' ')}{target}"
                texts.append(rendered)
        if include_target and n_without_target:
            notes.append(
                f"include_target: {n_without_target}/{len(calib_idx)} docs of {task_name} have no gold target; "
                "their prompt was used alone."
            )

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
        notes=notes,
    )


def _gold_target_text(task: Any, doc: Any) -> str | None:
    """The doc's gold target as text, or ``None`` when the task has none (e.g. generative tasks like ifeval)."""
    try:
        target = task.doc_to_target(doc)
    except Exception:  # noqa: BLE001 - tasks without a target raise task-specific errors
        return None
    if isinstance(target, bool):
        return None
    if isinstance(target, int):
        try:
            choices = task.doc_to_choice(doc)
            return str(choices[target])
        except Exception:  # noqa: BLE001
            return None
    if isinstance(target, str) and target.strip():
        return target
    return None


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


class TokenizedPromptDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]], sample_ids: list[str] | None = None) -> None:
        self.features = features
        # Identity of the underlying examples, not of this tokenization. The
        # source and target calibration loaders tokenize the SAME texts with
        # different tokenizers, so they are different objects holding different
        # token ids; paired calibration has to recognise them as the same
        # examples replayed under two preprocessors, which is exactly what it
        # falls back to sample_ids for.
        self.sample_ids = list(sample_ids) if sample_ids is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        feat = dict(self.features[int(idx)])
        feat["labels"] = list(feat["input_ids"])
        return feat


def build_text_calibration_loader(
    *,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 2,
    max_length: int = 128,
) -> DataLoader:
    prompt_list = list(texts)
    if not prompt_list:
        raise ValueError("Calibration loader needs at least one text sequence.")
    enc = tokenizer(
        prompt_list,
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
    )
    features: list[dict[str, Any]] = []
    for i in range(len(prompt_list)):
        features.append({k: v[i] for k, v in enc.items()})

    # Stable across processes: str.__hash__ is salted per interpreter, which
    # would make these ids non-reproducible if they were ever persisted.
    sample_ids = [f"{i}:{hashlib.sha1(t.encode()).hexdigest()[:12]}" for i, t in enumerate(prompt_list)]
    dataset = TokenizedPromptDataset(features, sample_ids=sample_ids)

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        feats = [{k: v for k, v in row.items() if k != "labels"} for row in batch]
        padded = tokenizer.pad(feats, return_tensors="pt", padding="max_length", max_length=int(max_length))
        padded["labels"] = padded["input_ids"].clone()
        if "attention_mask" in padded:
            padded["labels"][padded["attention_mask"] == 0] = -100
        return padded

    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
    )


def tokenization_stats(tokenizer: Any, texts: Sequence[str], max_length: int) -> dict[str, Any]:
    """Token-level provenance of a calibration corpus under ``max_length`` padding (summary record)."""
    lengths = [len(ids) for ids in tokenizer(list(texts), truncation=False, add_special_tokens=True)["input_ids"]]
    n = len(lengths)
    kept = [min(length, int(max_length)) for length in lengths]
    total_slots = n * int(max_length)
    return {
        "n_sequences": n,
        "max_length": int(max_length),
        "mean_length": (sum(kept) / n) if n else 0.0,
        "n_truncated": sum(1 for length in lengths if length > int(max_length)),
        "pad_fraction": (1.0 - sum(kept) / total_slots) if total_slots else 0.0,
    }
