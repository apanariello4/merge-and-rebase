from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

NLI_TASKS = ("snli", "mnli", "sick", "qnli", "rte", "scitail")


@dataclass(frozen=True)
class NLIExample:
    premise: str
    hypothesis: str
    label: int


@dataclass(frozen=True)
class NLITaskData:
    task: str
    examples: list[NLIExample]
    labels: list[str]
    label_texts: list[str]
    meta: dict[str, Any]


@dataclass(frozen=True)
class NLITokenizedData:
    task: str
    loader: DataLoader
    mask_class: list[int]
    meta: dict[str, Any]


@dataclass(frozen=True)
class _TaskSpec:
    hf_path: str
    hf_configs: tuple[str | None, ...]
    split_map: dict[str, tuple[str, ...]]
    premise_keys: tuple[str, ...]
    hypothesis_keys: tuple[str, ...]
    label_keys: tuple[str, ...]
    labels: tuple[str, ...]
    label_texts: tuple[str, ...]
    label_int_map: dict[int, str] | None = None
    label_str_map: dict[str, str] | None = None


_TASK_SPECS: dict[str, _TaskSpec] = {
    "snli": _TaskSpec(
        hf_path="snli",
        hf_configs=(None,),
        split_map={"train": ("train",), "validation": ("validation",), "test": ("test",)},
        premise_keys=("premise",),
        hypothesis_keys=("hypothesis",),
        label_keys=("label",),
        labels=("entailment", "neutral", "contradiction"),
        label_texts=("entailment", "neutral", "contradiction"),
        label_int_map={0: "entailment", 1: "neutral", 2: "contradiction"},
    ),
    "mnli": _TaskSpec(
        hf_path="glue",
        hf_configs=("mnli",),
        split_map={
            "train": ("train",),
            "validation": ("validation_matched", "validation_mismatched", "validation"),
            # GLUE test labels are unavailable; use labeled validation splits for eval.
            "test": ("validation_matched", "validation_mismatched", "validation"),
        },
        premise_keys=("premise",),
        hypothesis_keys=("hypothesis",),
        label_keys=("label",),
        labels=("entailment", "neutral", "contradiction"),
        label_texts=("entailment", "neutral", "contradiction"),
        label_int_map={0: "entailment", 1: "neutral", 2: "contradiction"},
    ),
    "sick": _TaskSpec(
        hf_path="yangwang825/sick",
        hf_configs=(None,),
        split_map={
            "train": ("train",),
            "validation": ("validation", "dev", "trial"),
            "test": ("test",),
        },
        premise_keys=("text1", "sentence_A", "sentence1", "premise"),
        hypothesis_keys=("text2", "sentence_B", "sentence2", "hypothesis"),
        label_keys=("label", "entailment_label"),
        labels=("entailment", "neutral", "contradiction"),
        label_texts=("entailment", "neutral", "contradiction"),
        label_int_map={0: "entailment", 1: "neutral", 2: "contradiction"},
        label_str_map={
            "entailment": "entailment",
            "neutral": "neutral",
            "contradiction": "contradiction",
            "entails": "entailment",
        },
    ),
    "qnli": _TaskSpec(
        hf_path="glue",
        hf_configs=("qnli",),
        # GLUE test labels are unavailable; use validation for eval.
        split_map={"train": ("train",), "validation": ("validation",), "test": ("validation",)},
        premise_keys=("question", "premise"),
        hypothesis_keys=("sentence", "hypothesis"),
        label_keys=("label",),
        labels=("entailment", "contradiction"),
        label_texts=("entailment", "contradiction"),
        label_int_map={0: "entailment", 1: "contradiction"},
        label_str_map={
            "entailment": "entailment",
            "not_entailment": "contradiction",
            "contradiction": "contradiction",
        },
    ),
    "rte": _TaskSpec(
        hf_path="glue",
        hf_configs=("rte",),
        # GLUE test labels are unavailable; use validation for eval.
        split_map={"train": ("train",), "validation": ("validation",), "test": ("validation",)},
        premise_keys=("sentence1", "premise"),
        hypothesis_keys=("sentence2", "hypothesis"),
        label_keys=("label",),
        labels=("entailment", "contradiction"),
        label_texts=("entailment", "contradiction"),
        label_int_map={0: "entailment", 1: "contradiction"},
        label_str_map={
            "entailment": "entailment",
            "not_entailment": "contradiction",
            "contradiction": "contradiction",
        },
    ),
    "scitail": _TaskSpec(
        hf_path="scitail",
        hf_configs=("tsv_format", None),
        split_map={"train": ("train",), "validation": ("validation", "dev"), "test": ("test",)},
        premise_keys=("sentence1", "premise"),
        hypothesis_keys=("sentence2", "hypothesis"),
        label_keys=("label",),
        labels=("entailment", "neutral"),
        label_texts=("entailment", "neutral"),
        label_int_map={0: "entailment", 1: "neutral"},
        label_str_map={
            "entails": "entailment",
            "entailment": "entailment",
            "neutral": "neutral",
            "not_entails": "neutral",
        },
    ),
}


def _first_non_empty(ex: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in ex and isinstance(ex[k], str):
            s = ex[k].strip()
            if s:
                return s
    return None


def _norm_label_str(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def _to_label_name(raw: Any, spec: _TaskSpec) -> str | None:
    if isinstance(raw, bool):
        raw = int(raw)

    if isinstance(raw, int):
        if spec.label_int_map is not None:
            return spec.label_int_map.get(int(raw), None)
        if 0 <= int(raw) < len(spec.labels):
            return spec.labels[int(raw)]
        return None

    if isinstance(raw, str):
        k = _norm_label_str(raw)
        if spec.label_str_map is not None and k in spec.label_str_map:
            return spec.label_str_map[k]
        if k in spec.labels:
            return k
        return None

    return None


def _load_dataset_with_fallbacks(
    *,
    task: str,
    spec: _TaskSpec,
    split: str,
):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("Text task loading requires `datasets` (install with `.[data]`).") from e

    if split not in spec.split_map:
        raise ValueError(f"Unsupported split '{split}' for task '{task}'.")

    errors: list[str] = []
    for cfg in spec.hf_configs:
        for hf_split in spec.split_map[split]:
            try:
                if cfg is None:
                    ds = load_dataset(spec.hf_path, split=hf_split)
                else:
                    ds = load_dataset(spec.hf_path, cfg, split=hf_split)
                return ds, cfg, hf_split
            except Exception as e:  # pragma: no cover - best-effort fallback path
                errors.append(f"path={spec.hf_path}, config={cfg}, split={hf_split}: {type(e).__name__}: {e}")
                continue

    joined = "\n".join(errors[:8])
    raise RuntimeError(f"Failed to load dataset for task '{task}'. Tried:\n{joined}")


def build_nli_task_data(
    *,
    task: str,
    split: str = "validation",
    max_samples: int | None = None,
) -> NLITaskData:
    task_key = str(task).strip().lower()
    if task_key not in _TASK_SPECS:
        raise ValueError(f"Unknown task '{task}'. Supported tasks: {list(NLI_TASKS)}")
    spec = _TASK_SPECS[task_key]

    ds, used_cfg, used_split = _load_dataset_with_fallbacks(task=task_key, spec=spec, split=split)

    label_index = {n: i for i, n in enumerate(spec.labels)}
    rows: list[NLIExample] = []
    skipped = 0

    for ex in ds:
        premise = _first_non_empty(ex, spec.premise_keys)
        hypothesis = _first_non_empty(ex, spec.hypothesis_keys)
        if premise is None or hypothesis is None:
            skipped += 1
            continue

        raw_label = None
        for lk in spec.label_keys:
            if lk in ex:
                raw_label = ex[lk]
                break
        if raw_label is None:
            skipped += 1
            continue

        label_name = _to_label_name(raw_label, spec)
        if label_name is None or label_name not in label_index:
            skipped += 1
            continue

        rows.append(NLIExample(premise=premise, hypothesis=hypothesis, label=label_index[label_name]))

    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]

    if not rows:
        raise ValueError(
            f"No usable examples loaded for task '{task_key}' (split='{split}'). "
            "Check dataset availability/mapping."
        )

    meta = {
        "task": task_key,
        "hf_path": spec.hf_path,
        "hf_config": used_cfg,
        "hf_split": used_split,
        "num_examples": len(rows),
        "num_skipped": skipped,
        "labels": list(spec.labels),
        "label_texts": list(spec.label_texts),
    }
    return NLITaskData(
        task=task_key,
        examples=rows,
        labels=list(spec.labels),
        label_texts=list(spec.label_texts),
        meta=meta,
    )


class _TokenizedNLIDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]], labels: list[int]) -> None:
        if len(features) != len(labels):
            raise ValueError(f"features/labels length mismatch: {len(features)} vs {len(labels)}")
        self.features = features
        self.labels = [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = int(idx)
        out = dict(self.features[i])
        out["labels"] = int(self.labels[i])
        return out


def default_head_class_ids_for_task(task: str, num_labels: int) -> list[int]:
    t = str(task).strip().lower()
    if t in {"qnli", "rte"} and int(num_labels) >= 3:
        # binary entailment tasks often use class ids {0,2} in a 3-way head space.
        return [0, 2]
    if t == "scitail" and int(num_labels) >= 2:
        return [0, 1]
    if num_labels == 3:
        return [0, 1, 2]
    return list(range(num_labels))


def build_nli_tokenized_loader(
    *,
    task_data: NLITaskData,
    tokenizer: Any,
    batch_size: int = 8,
    num_workers: int = 0,
    max_length: int = 512,
    shuffle: bool = False,
    head_class_ids: list[int] | None = None,
) -> NLITokenizedData:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0.")
    if int(max_length) <= 4:
        raise ValueError("max_length must be > 4.")

    mapped_class_ids = (
        list(head_class_ids)
        if head_class_ids is not None
        else default_head_class_ids_for_task(task_data.task, num_labels=len(task_data.labels))
    )
    if len(mapped_class_ids) != len(task_data.labels):
        raise ValueError(
            f"head_class_ids length mismatch for task '{task_data.task}': "
            f"{len(mapped_class_ids)} vs {len(task_data.labels)}"
        )

    premises = [ex.premise for ex in task_data.examples]
    hypotheses = [ex.hypothesis for ex in task_data.examples]
    local_labels = [int(ex.label) for ex in task_data.examples]
    labels = [int(mapped_class_ids[y]) for y in local_labels]

    enc = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        max_length=int(max_length),
        padding=False,
    )
    features: list[dict[str, Any]] = []
    n = len(labels)
    for i in range(n):
        feat = {}
        for k, v in enc.items():
            feat[k] = v[i]
        features.append(feat)
    dataset = _TokenizedNLIDataset(features=features, labels=labels)

    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        feats = [{k: v for k, v in x.items() if k != "labels"} for x in batch]
        ys = torch.tensor([int(x["labels"]) for x in batch], dtype=torch.long)
        padded = tokenizer.pad(feats, return_tensors="pt")
        padded["labels"] = ys
        return padded

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    mask_class = sorted(set(mapped_class_ids))
    meta = dict(task_data.meta)
    meta.update(
        {
            "num_examples_tokenized": len(dataset),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "shuffle": bool(shuffle),
            "mask_class": list(mask_class),
            "head_class_ids": list(mapped_class_ids),
        }
    )
    return NLITokenizedData(task=task_data.task, loader=loader, mask_class=mask_class, meta=meta)
