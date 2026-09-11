"""Deterministic paired calibration loaders for a collection of vision tasks.

The builder in this module deliberately does not depend on the repository's
benchmark-specific dataset loaders.  It accepts ordinary map-style PyTorch
datasets and task-specific transforms, which makes the resulting calibration
plan usable for both model endpoints in a rebase/merge experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

Transform = Callable[[Any], Any] | None


@dataclass(frozen=True)
class Vision8TaskContext:
    """Source/target datasets and preprocessing for one Vision8 task.

    ``classnames`` is the local class order shared by both endpoints.  The
    optional endpoint-specific fields are accepted to make mismatches
    explicit: if either is supplied, both must describe the same class order.
    """

    source_dataset: Dataset
    target_dataset: Dataset
    source_transform: Transform = None
    target_transform: Transform = None
    classnames: Sequence[str] | None = None
    source_classnames: Sequence[str] | None = None
    target_classnames: Sequence[str] | None = None


@dataclass(frozen=True)
class BalancedVision8CalibrationLoaders:
    """Result returned by :func:`build_balanced_vision8_calibration_loaders`."""

    source_loaders: DataLoader
    target_loaders: DataLoader
    union_classnames: tuple[str, ...]
    label_offsets: dict[str, int]
    plan: dict[str, Any]
    fingerprint: str

    @property
    def source_loader(self) -> DataLoader:
        return self.source_loaders

    @property
    def target_loader(self) -> DataLoader:
        return self.target_loaders

    @property
    def classnames(self) -> tuple[str, ...]:
        """Compatibility alias for callers that use ``classnames``."""

        return self.union_classnames

    def __iter__(self):
        yield self.source_loaders
        yield self.target_loaders


def _as_int_label(value: Any) -> int:
    """Convert scalar tensor/NumPy-like labels to a Python integer."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar label, got tensor with shape {tuple(value.shape)}.")
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Dataset labels must be integer-like, got {value!r}.") from exc


def _split_sample(sample: Any) -> tuple[Any, int]:
    """Extract an input and label from the common dataset sample formats."""

    if isinstance(sample, Mapping):
        for input_key in ("image", "x", "input", "pixel_values"):
            if input_key in sample:
                break
        else:
            raise ValueError("Dataset mapping sample has no image/x/input/pixel_values field.")
        for label_key in ("label", "labels", "y", "target"):
            if label_key in sample:
                return sample[input_key], _as_int_label(sample[label_key])
        raise ValueError("Dataset mapping sample has no label/labels/y/target field.")

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], _as_int_label(sample[1])
    raise ValueError("Dataset samples must be (input, label) tuples or mappings.")


def _label_values(dataset: Dataset) -> list[int]:
    """Read raw labels without invoking a dataset transform."""

    if isinstance(dataset, Subset):
        parent = _label_values(dataset.dataset)
        values = [parent[int(index)] for index in dataset.indices]
        if not values:
            raise ValueError("Calibration datasets must not be empty.")
        return values

    split = getattr(dataset, "split", None)
    label_key = getattr(dataset, "label_key", "label")
    if split is not None and hasattr(split, "column_names") and label_key in split.column_names:
        values = [_as_int_label(label) for label in split[label_key]]
        if not values:
            raise ValueError("Calibration datasets must not be empty.")
        return values

    values: list[int] = []
    for index in range(len(dataset)):
        _, label = _split_sample(dataset[index])
        values.append(label)
    if not values:
        raise ValueError("Calibration datasets must not be empty.")
    return values


def _normalise_transforms(
    transforms: Mapping[str, Transform] | None,
    task_names: Sequence[str],
    side: str,
) -> dict[str, Transform]:
    if transforms is None:
        return {task: None for task in task_names}
    missing = [task for task in task_names if task not in transforms]
    extra = [task for task in transforms if task not in task_names]
    if missing or extra:
        raise ValueError(f"{side} transforms must match datasets; missing={missing}, extra={extra}.")
    return {task: transforms[task] for task in task_names}


def _jsonable_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a plan through JSON to guarantee that it is serialization-safe."""

    return json.loads(json.dumps(plan, sort_keys=True, separators=(",", ":")))


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _PlannedTaskDataset(Dataset):
    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        transforms: Mapping[str, Transform],
        records: Sequence[tuple[str, int]],
        label_maps: Mapping[str, Mapping[int, int]],
        label_offsets: Mapping[str, int],
    ) -> None:
        self._datasets = datasets
        self._transforms = transforms
        self._records = tuple(records)
        self._label_maps = label_maps
        self._label_offsets = label_offsets

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        task, dataset_index = self._records[index]
        value, raw_label = _split_sample(self._datasets[task][dataset_index])
        transform = self._transforms[task]
        if transform is not None:
            value = transform(value)
        try:
            local_label = self._label_maps[task][raw_label]
        except KeyError as exc:
            raise ValueError(f"Unexpected label {raw_label} for task {task!r}.") from exc
        label = self._label_offsets[task] + local_label
        return value, torch.tensor(label, dtype=torch.long)


@dataclass(frozen=True)
class BalancedVisionCalibrationLoaders:
    """Paired, aligned source and target calibration loaders."""

    source_loader: DataLoader
    target_loader: DataLoader
    plan: dict[str, Any]
    fingerprint: str

    @property
    def source(self) -> DataLoader:
        return self.source_loader

    @property
    def target(self) -> DataLoader:
        return self.target_loader

    def __iter__(self):
        """Allow ``source_loader, target_loader = build(...)`` unpacking."""

        yield self.source_loader
        yield self.target_loader


def build_balanced_vision_calibration_loaders(
    source_datasets: Mapping[str, Dataset],
    target_datasets: Mapping[str, Dataset],
    *,
    source_transforms: Mapping[str, Transform] | None = None,
    target_transforms: Mapping[str, Transform] | None = None,
    batch_size: int,
    n_batches: int,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    class_counts_by_task: Mapping[str, int] | None = None,
) -> BalancedVisionCalibrationLoaders:
    """Build exactly aligned balanced calibration loaders.

    A batch contains ``batch_size // n_tasks`` examples from every task.  The
    same task/index records are used by both loaders, while source and target
    transforms are applied independently.  Sampling is without replacement;
    insufficient data therefore raises during construction rather than causing
    an implicit cycling or silently repeated calibration example.
    """

    if not isinstance(source_datasets, Mapping) or not isinstance(target_datasets, Mapping):
        raise TypeError("source_datasets and target_datasets must be mappings.")
    if not source_datasets:
        raise ValueError("At least one calibration task is required.")
    if set(source_datasets) != set(target_datasets):
        raise ValueError(
            "Source and target task mappings must have identical keys; "
            f"source={sorted(source_datasets)}, target={sorted(target_datasets)}."
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(n_batches, int) or n_batches <= 0:
        raise ValueError("n_batches must be a positive integer.")

    task_names = tuple(sorted(str(task) for task in source_datasets))
    if len(set(task_names)) != len(task_names):
        raise ValueError("Task names must be unique after string conversion.")
    n_tasks = len(task_names)
    if batch_size < n_tasks or batch_size % n_tasks:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by the {n_tasks} calibration tasks."
        )
    samples_per_task = batch_size // n_tasks
    required_per_task = samples_per_task * n_batches

    # Re-key after normalising names so the plan and offsets are deterministic.
    source_by_task = {str(task): source_datasets[task] for task in source_datasets}
    target_by_task = {str(task): target_datasets[task] for task in target_datasets}
    source_tf = _normalise_transforms(source_transforms, task_names, "source")
    target_tf = _normalise_transforms(target_transforms, task_names, "target")

    source_raw_labels: dict[str, list[int]] = {}
    target_raw_labels: dict[str, list[int]] = {}
    source_label_maps: dict[str, dict[int, int]] = {}
    target_label_maps: dict[str, dict[int, int]] = {}
    class_counts: dict[str, int] = {}
    selected_indices: dict[str, tuple[int, ...]] = {}
    records: list[tuple[str, int]] = []

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    offset = 0
    label_offsets: dict[str, int] = {}
    dataset_lengths: dict[str, dict[str, int]] = {}

    for task in task_names:
        source_dataset = source_by_task[task]
        target_dataset = target_by_task[task]
        source_len = len(source_dataset)
        target_len = len(target_dataset)
        if source_len != target_len:
            raise ValueError(
                f"Paired datasets for task {task!r} must have equal lengths; "
                f"source={source_len}, target={target_len}."
            )
        if source_len < required_per_task:
            raise ValueError(
                f"Calibration dataset for task {task!r} is exhausted: need "
                f"{required_per_task} samples, source has {source_len}, target has {target_len}."
            )

        source_labels = _label_values(source_dataset)
        target_labels = _label_values(target_dataset)
        source_unique = sorted(set(source_labels))
        target_unique = sorted(set(target_labels))
        if len(source_unique) != len(target_unique):
            raise ValueError(
                f"Source/target class counts differ for task {task!r}: "
                f"{len(source_unique)} != {len(target_unique)}."
            )
        if source_labels != target_labels:
            raise ValueError(f"Source and target labels are not aligned for task {task!r}.")

        declared_count = (
            int(class_counts_by_task[task])
            if class_counts_by_task is not None and task in class_counts_by_task
            else None
        )
        if declared_count is not None:
            if declared_count <= 0:
                raise ValueError(f"Declared class count for task {task!r} must be positive.")
            invalid = [label for label in source_unique if label < 0 or label >= declared_count]
            if invalid:
                raise ValueError(
                    f"Observed labels for task {task!r} exceed its declared class range "
                    f"0..{declared_count - 1}: {invalid[:8]}."
                )
            source_map = {label: label for label in range(declared_count)}
            target_map = dict(source_map)
            count = declared_count
        else:
            source_map = {label: local for local, label in enumerate(source_unique)}
            target_map = {label: local for local, label in enumerate(target_unique)}
            count = len(source_unique)

        permutation = torch.randperm(source_len, generator=generator)[:required_per_task].tolist()
        selected_indices[task] = tuple(int(index) for index in permutation)

        class_counts[task] = count
        label_offsets[task] = offset
        offset += count
        source_raw_labels[task] = source_labels
        target_raw_labels[task] = target_labels
        source_label_maps[task] = source_map
        target_label_maps[task] = target_map
        dataset_lengths[task] = {"source": source_len, "target": target_len}

    # Interleave task slices batch-by-batch.  Keeping the task dimension in
    # the record order makes the equal-per-task invariant visible to callers
    # and avoids relying on a custom collate function.
    for batch_index in range(n_batches):
        start = batch_index * samples_per_task
        end = start + samples_per_task
        for task in task_names:
            records.extend((task, index) for index in selected_indices[task][start:end])

    plan = _jsonable_plan(
        {
            "version": 1,
            "task_names": list(task_names),
            "batch_size": batch_size,
            "n_batches": n_batches,
            "samples_per_task": samples_per_task,
            "required_samples_per_task": required_per_task,
            "seed": int(seed),
            "class_counts": class_counts,
            "label_offsets": label_offsets,
            "dataset_lengths": dataset_lengths,
            "selected_indices": {task: list(indices) for task, indices in selected_indices.items()},
        }
    )
    fingerprint = _plan_fingerprint(plan)

    source_dataset = _PlannedTaskDataset(
        source_by_task, source_tf, records, source_label_maps, label_offsets
    )
    target_dataset = _PlannedTaskDataset(
        target_by_task, target_tf, records, target_label_maps, label_offsets
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    return BalancedVisionCalibrationLoaders(
        source_loader=DataLoader(source_dataset, **loader_kwargs),
        target_loader=DataLoader(target_dataset, **loader_kwargs),
        plan=plan,
        fingerprint=fingerprint,
    )


def _context_value(context: Vision8TaskContext | Mapping[str, Any], keys: Sequence[str]) -> Any:
    if isinstance(context, Vision8TaskContext):
        for key in keys:
            if hasattr(context, key):
                value = getattr(context, key)
                if value is not None:
                    return value
        return None
    if isinstance(context, Mapping):
        for key in keys:
            if key in context and context[key] is not None:
                return context[key]
        return None
    raise TypeError("Each task context must be Vision8TaskContext or a mapping.")


def _context_classnames(
    task: str,
    context: Vision8TaskContext | Mapping[str, Any],
    class_count: int,
) -> tuple[str, ...]:
    common = _context_value(context, ("classnames", "class_names"))
    source = _context_value(context, ("source_classnames", "source_class_names"))
    target = _context_value(context, ("target_classnames", "target_class_names"))
    if common is not None and (source is not None or target is not None):
        if source is not None and list(source) != list(common):
            raise ValueError(f"Source classnames do not match classnames for task {task!r}.")
        if target is not None and list(target) != list(common):
            raise ValueError(f"Target classnames do not match classnames for task {task!r}.")
    if source is not None and target is not None and list(source) != list(target):
        raise ValueError(f"Source and target classnames are not aligned for task {task!r}.")
    names = common if common is not None else source if source is not None else target
    if names is None:
        return tuple(f"{task}:{index}" for index in range(class_count))
    if isinstance(names, (str, bytes)):
        raise TypeError(f"Classnames for task {task!r} must be a sequence of names.")
    names_tuple = tuple(str(name) for name in names)
    if len(names_tuple) != class_count:
        raise ValueError(
            f"Task {task!r} has {class_count} observed classes but "
            f"{len(names_tuple)} classnames were provided."
        )
    return names_tuple


def build_balanced_vision8_calibration_loaders(
    task_contexts: Mapping[str, Vision8TaskContext | Mapping[str, Any]],
    n_batches: int,
    batch_size: int,
    seed: int = 0,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> BalancedVision8CalibrationLoaders:
    """Build paired balanced Vision8 calibration loaders.

    Args:
        task_contexts: Mapping ``task_name -> Vision8TaskContext``.  A mapping
            with the same fields is also accepted.  Required mapping keys are
            ``source_dataset`` and ``target_dataset``; optional keys are
            ``source_transform``, ``target_transform``, and ``classnames``.
        n_batches: Number of complete balanced batches to produce.
        batch_size: Complete batch size; it must be divisible by the number of
            tasks.
        seed: Seed for deterministic, without-replacement per-task sampling.
        num_workers: DataLoader worker count for both loaders.
        pin_memory: Whether both DataLoaders pin returned tensors.

    Returns:
        ``BalancedVision8CalibrationLoaders`` with ``source_loaders`` and
        ``target_loaders`` (one DataLoader each), the concatenated
        ``union_classnames``, task ``label_offsets``, a JSON-ready ``plan``,
        and its SHA-256 ``fingerprint``.

    Source and target datasets must have equal lengths and exactly equal raw
    label sequences for every task.  The shared index plan then guarantees
    that every paired sample has the same remapped label.
    """

    if not isinstance(task_contexts, Mapping) or not task_contexts:
        raise ValueError("task_contexts must be a non-empty task-name mapping.")
    task_names = tuple(sorted(str(task) for task in task_contexts))
    if len(set(task_names)) != len(task_names):
        raise ValueError("Task names must be unique after string conversion.")

    source_datasets: dict[str, Dataset] = {}
    target_datasets: dict[str, Dataset] = {}
    source_transforms: dict[str, Transform] = {}
    target_transforms: dict[str, Transform] = {}
    contexts_by_task: dict[str, Vision8TaskContext | Mapping[str, Any]] = {}
    declared_class_counts: dict[str, int] = {}
    for original_task, context in task_contexts.items():
        task = str(original_task)
        source_dataset = _context_value(context, ("source_dataset", "source"))
        target_dataset = _context_value(context, ("target_dataset", "target"))
        if source_dataset is None or target_dataset is None:
            raise ValueError(
                f"Task context {task!r} must provide source_dataset and target_dataset."
            )
        source_datasets[task] = source_dataset
        target_datasets[task] = target_dataset
        source_transforms[task] = _context_value(
            context, ("source_transform", "source_preprocess")
        )
        target_transforms[task] = _context_value(
            context, ("target_transform", "target_preprocess")
        )
        contexts_by_task[task] = context
        names = _context_value(context, ("classnames", "class_names"))
        if names is None:
            names = _context_value(context, ("source_classnames", "source_class_names"))
        if names is not None:
            if isinstance(names, (str, bytes)):
                raise TypeError(f"Classnames for task {task!r} must be a sequence of names.")
            declared_class_counts[task] = len(names)

    built = build_balanced_vision_calibration_loaders(
        source_datasets,
        target_datasets,
        source_transforms=source_transforms,
        target_transforms=target_transforms,
        batch_size=batch_size,
        n_batches=n_batches,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        class_counts_by_task=declared_class_counts or None,
    )
    classnames_by_task = {
        task: _context_classnames(task, contexts_by_task[task], built.plan["class_counts"][task])
        for task in task_names
    }
    union_classnames = tuple(
        name for task in task_names for name in classnames_by_task[task]
    )
    plan = _jsonable_plan(
        {
            **built.plan,
            "api": "build_balanced_vision8_calibration_loaders",
            "classnames_by_task": {task: list(names) for task, names in classnames_by_task.items()},
            "union_classnames": list(union_classnames),
        }
    )
    return BalancedVision8CalibrationLoaders(
        source_loaders=built.source_loader,
        target_loaders=built.target_loader,
        union_classnames=union_classnames,
        label_offsets=dict(built.plan["label_offsets"]),
        plan=plan,
        fingerprint=_plan_fingerprint(plan),
    )


# Short aliases make the API convenient without creating a second implementation.
build_balanced_calibration_loaders = build_balanced_vision_calibration_loaders
make_balanced_vision_calibration_loaders = build_balanced_vision_calibration_loaders


__all__ = [
    "BalancedVision8CalibrationLoaders",
    "BalancedVisionCalibrationLoaders",
    "Vision8TaskContext",
    "build_balanced_calibration_loaders",
    "build_balanced_vision_calibration_loaders",
    "build_balanced_vision8_calibration_loaders",
    "make_balanced_vision_calibration_loaders",
]
