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
    include_target: bool = False,
) -> CalibrationTexts:
    """Resolve the calibration corpus for an LLM run.

    Parameters
    ----------
    prompts : Explicit prompt bank from `config['calibration_prompts']`.
    calibration_dataset : HF dataset path, or a spec mapping with `path` and
        optional `name`/`split`/`text_column`/`text_template`. A
        `text_template` such as ``"{question}\n{answer}"`` is formatted per row
        with the row's columns, so a dataset's reference responses can join the
        prompt; it replaces `text_column`. `shuffle: true` draws the rows in a
        `seed`-shuffled order instead of taking the first ones, so different
        seeds calibrate on different subsets (without it the seed has no
        effect on an HF corpus that is used in full).
    calibration_split : Split to calibrate on. For an lm-harness source this
        selects the held-out slice ("val"/"validation") rather than a split
        that has to exist upstream, so it also works for single-split tasks.
    harness_tasks : Evaluated lm-harness tasks, used as the default source.
    n_sequences : How many sequences the run will actually consume
        (`n_batches * batch_size`). The calibration slice is sized to cover
        this where the source allows it.
    seed : Seed for the deterministic calibration/eval partition.
    include_target : lm-harness source only. Append each doc's reference
        target (`doc_to_target`, joined with the task's `target_delimiter`) to
        its rendered prompt, so the banks include the positions where the task
        answer is written. Refused for tasks whose target is not text.
    """
    if n_sequences <= 0:
        raise ValueError("n_sequences must be > 0.")

    if include_target and (prompts or calibration_dataset is not None or not harness_tasks):
        raise ValueError(
            "calibration_include_target applies only to lm-harness calibration; for an HF "
            "calibration_dataset use its text_template to add the response column."
        )

    if prompts:
        return CalibrationTexts(
            [str(p) for p in prompts], source="config['calibration_prompts']"
        )

    if calibration_dataset is not None:
        return _from_hf_dataset(
            calibration_dataset,
            calibration_split=calibration_split,
            n_sequences=n_sequences,
            seed=seed,
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
    seed: int = 0,
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
    text_template = spec.get("text_template", None)
    shuffle = bool(spec.get("shuffle", False))
    if text_template is not None and text_column is not None:
        raise ValueError("calibration_dataset takes text_column or text_template, not both.")

    ds = (
        datasets.load_dataset(str(path), str(name), split=split)
        if name
        else datasets.load_dataset(str(path), split=split)
    )

    rows = ds
    order_label = ""
    if shuffle:
        order = list(range(len(ds)))
        random.Random(seed).shuffle(order)
        rows = (ds[i] for i in order)
        order_label = f" shuffled(seed={seed})"

    if text_template is not None:
        import string

        fields = {name for _, name, _, _ in string.Formatter().parse(str(text_template)) if name}
        missing = sorted(fields - set(ds.column_names))
        if missing:
            raise ValueError(
                f"text_template references columns {missing} not in calibration_dataset {path!r} "
                f"(columns: {ds.column_names})."
            )
        texts = []
        for row in rows:
            value = str(text_template).format(**{k: row[k] for k in fields})
            if value.strip():
                texts.append(value)
            if len(texts) >= n_sequences:
                break
        return CalibrationTexts(texts, source=f"{path}[{split}]{order_label} template {text_template!r}")

    column = text_column or _pick_text_column(ds.column_names)
    if column is None:
        raise ValueError(
            f"Could not infer a text column for calibration_dataset {path!r} "
            f"(columns: {ds.column_names}). Set 'text_column' explicitly."
        )

    texts: list[str] = []
    for row in rows:
        value = row.get(column, None)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        if len(texts) >= n_sequences:
            break

    return CalibrationTexts(texts, source=f"{path}[{split}]{order_label}.{column}")


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
            if not (isinstance(rendered, str) and rendered.strip()):
                continue
            if include_target:
                target = task.doc_to_target(docs[i])
                if isinstance(target, list) and len(target) == 1:
                    target = target[0]
                if not isinstance(target, str) or not target.strip():
                    # IFEval has no reference response at all; multiple-choice
                    # tasks return a choice index. Neither is text to append.
                    raise ValueError(
                        f"lm-harness task {task_name!r} has no text reference target "
                        f"(doc_to_target returned {type(target).__name__}); use "
                        "align_with_gen_response or a calibration_dataset instead."
                    )
                delimiter = getattr(getattr(task, "config", None), "target_delimiter", " ")
                rendered = f"{rendered}{delimiter if delimiter is not None else ' '}{target}"
            texts.append(rendered)

    if not texts:
        raise ValueError(
            f"lm-harness tasks {tasks} yielded no calibration text. "
            "Point block_extension_params.calibration_dataset at a dataset instead."
        )

    split_label = "holdout" if wants_holdout else str(calibration_split)
    return CalibrationTexts(
        texts,
        source=f"lm-harness {'+'.join(tasks)}[{split_label}]" + ("+target" if include_target else ""),
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


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def append_generated_responses(
    texts: Sequence[str],
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    max_prompt_length: int | None = None,
    apply_chat_template: bool = False,
    device: str = "cuda",
) -> tuple[list[str], dict[str, Any]]:
    """Extend each calibration prompt with the model's own greedy response.

    The source fine-tune's effect on an instruct or reasoning task shows up
    while it *writes* the answer, and a prompt-only bank reaches that only at
    the last prompt token. Teacher-forcing the generated response puts those
    positions in the bank; D_j = B^1 - B^0 still compares both source models on
    the same text, so the text only has to be representative, not correct.

    Generation is greedy and left-padded (the pad side the fit never sees:
    the returned strings are re-tokenized by the calibration loader). With
    ``apply_chat_template`` the prompt is wrapped in the tokenizer's chat
    template both for generation and in the returned text, so it must match
    how the harness renders prompts at eval time.

    Returns ``(texts, stats)``; ``stats`` carries response-length figures for
    the run summary.
    """
    import torch

    if max_new_tokens <= 0:
        raise ValueError("align_with_gen_response.max_new_tokens must be > 0.")
    prompts = [_chat_prompt(tokenizer, t) if apply_chat_template else str(t) for t in texts]
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    training = model.training
    original_device = next(model.parameters()).device
    out_texts: list[str] = []
    lengths: list[int] = []
    hit_limit = 0
    try:
        model.to(device).eval()
        for start in range(0, len(prompts), max(1, int(batch_size))):
            chunk = prompts[start : start + max(1, int(batch_size))]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=max_prompt_length is not None,
                max_length=max_prompt_length,
                # A chat template already carries its own BOS/role tokens.
                add_special_tokens=not apply_chat_template,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            for prompt, row in zip(chunk, new_tokens):
                real = row[row != tokenizer.pad_token_id]
                lengths.append(int(real.numel()))
                hit_limit += int(real.numel() >= int(max_new_tokens))
                out_texts.append(prompt + tokenizer.decode(real, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = original_side
        model.to(original_device).train(training)
    stats = {
        "n": len(out_texts),
        "max_new_tokens": int(max_new_tokens),
        "mean_response_tokens": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "hit_max_new_tokens": hit_limit,
        "apply_chat_template": bool(apply_chat_template),
    }
    return out_texts, stats
