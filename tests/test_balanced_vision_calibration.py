from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import TensorDataset

from merge_and_rebase.data.balanced_calibration import (
    Vision8TaskContext,
    build_balanced_vision8_calibration_loaders,
    build_balanced_vision_calibration_loaders,
)


def test_hf_style_label_scan_does_not_decode_inputs() -> None:
    class Split:
        column_names = ["image", "label"]

        def __getitem__(self, key):
            assert key == "label"
            return [0, 1, 0, 1]

    class Dataset(torch.utils.data.Dataset):
        split = Split()
        label_key = "label"

        def __len__(self):
            return 4

        def __getitem__(self, index):
            raise AssertionError("image decoding must not run during label validation")

    contexts = {
        "task": Vision8TaskContext(
            source_dataset=Dataset(),
            target_dataset=Dataset(),
            classnames=["zero", "one"],
        )
    }
    built = build_balanced_vision8_calibration_loaders(contexts, n_batches=1, batch_size=2)
    assert built.plan["class_counts"] == {"task": 2}


def test_declared_class_order_survives_classes_missing_from_calibration_subset() -> None:
    labels = torch.tensor([0, 2, 0, 2])
    source = TensorDataset(torch.arange(4).reshape(-1, 1), labels)
    target = TensorDataset(torch.arange(4).reshape(-1, 1), labels.clone())
    built = build_balanced_vision8_calibration_loaders(
        {
            "task": Vision8TaskContext(
                source_dataset=source,
                target_dataset=target,
                classnames=["zero", "unobserved one", "two"],
            )
        },
        n_batches=1,
        batch_size=2,
        seed=0,
    )
    assert built.plan["class_counts"] == {"task": 3}
    assert built.label_offsets == {"task": 0}
    labels = torch.cat([batch[1] for batch in built.source_loaders]).tolist()
    assert set(labels) <= {0, 2}


def _datasets(*, target_label_shift: int = 0, size: int = 20):
    source = {}
    target = {}
    for task_index, task in enumerate(("cars", "pets", "sun")):
        values = torch.arange(size, dtype=torch.float32).reshape(-1, 1) + task_index * 100
        labels = torch.arange(size, dtype=torch.long) % 2
        source[task] = TensorDataset(values, labels)
        target[task] = TensorDataset(values + 1000, labels + target_label_shift)
    return source, target


def test_balanced_loaders_are_paired_balanced_and_remapped() -> None:
    source, target = _datasets()
    built = build_balanced_vision_calibration_loaders(
        source,
        target,
        source_transforms={task: lambda x: x + 1 for task in source},
        target_transforms={task: lambda x: x + 2 for task in target},
        batch_size=6,
        n_batches=3,
        seed=17,
    )

    assert len(built.source_loader) == len(built.target_loader) == 3
    assert built.plan["samples_per_task"] == 2
    assert json.loads(json.dumps(built.plan)) == built.plan
    assert len(built.fingerprint) == 64

    for source_batch, target_batch in zip(built.source_loader, built.target_loader, strict=True):
        source_values, source_labels = source_batch
        target_values, target_labels = target_batch
        assert source_values.shape == target_values.shape == (6, 1)
        assert torch.equal(source_labels, target_labels)
        # Records are laid out task-by-task inside every batch, with disjoint
        # two-class ranges for the three tasks.
        assert set(source_labels[:2].tolist()).issubset({0, 1})
        assert set(source_labels[2:4].tolist()).issubset({2, 3})
        assert set(source_labels[4:].tolist()).issubset({4, 5})
        assert torch.allclose(target_values - source_values, torch.full((6, 1), 1001.0))


def test_seed_controls_plan_and_fingerprint() -> None:
    source, target = _datasets()
    kwargs = dict(
        source_datasets=source,
        target_datasets=target,
        batch_size=6,
        n_batches=2,
    )
    first = build_balanced_vision_calibration_loaders(seed=9, **kwargs)
    same = build_balanced_vision_calibration_loaders(seed=9, **kwargs)
    other = build_balanced_vision_calibration_loaders(seed=10, **kwargs)

    assert first.plan == same.plan
    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != other.fingerprint


@pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
def test_batch_size_must_be_divisible_by_task_count(batch_size: int) -> None:
    source, target = _datasets()
    with pytest.raises(ValueError, match="divisible"):
        build_balanced_vision_calibration_loaders(
            source, target, batch_size=batch_size, n_batches=1
        )


def test_exhaustion_and_mismatched_labels_raise() -> None:
    source, target = _datasets(size=3)
    with pytest.raises(ValueError, match="exhausted"):
        build_balanced_vision_calibration_loaders(
            source, target, batch_size=6, n_batches=2, seed=0
        )

    source, target = _datasets(target_label_shift=1)
    with pytest.raises(ValueError, match="not aligned"):
        build_balanced_vision_calibration_loaders(
            source, target, batch_size=6, n_batches=1
        )


def test_task_offsets_remain_disjoint_for_different_class_counts() -> None:
    source = {
        "first": TensorDataset(torch.zeros(8, 1), torch.tensor([0, 1] * 4)),
        "second": TensorDataset(torch.zeros(8, 1), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])),
    }
    target = {
        "first": TensorDataset(torch.ones(8, 1), torch.tensor([0, 1] * 4)),
        "second": TensorDataset(torch.ones(8, 1), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])),
    }
    built = build_balanced_vision_calibration_loaders(
        source, target, batch_size=4, n_batches=1, seed=3
    )

    assert built.plan["label_offsets"] == {"first": 0, "second": 2}
    _, labels = next(iter(built.source_loader))
    assert set(labels.tolist()).issubset({0, 1, 2, 3, 4})


def test_vision8_api_returns_loaders_union_names_offsets_and_plan() -> None:
    source, target = _datasets(size=12)
    contexts = {
        task: Vision8TaskContext(
            source_dataset=source[task],
            target_dataset=target[task],
            source_transform=lambda x: x + 1,
            target_transform=lambda x: x + 2,
            classnames=(f"{task}-zero", f"{task}-one"),
        )
        for task in source
    }
    built = build_balanced_vision8_calibration_loaders(contexts, 2, 6, 21)

    assert built.source_loaders is built.source_loader
    assert built.target_loaders is built.target_loader
    assert built.union_classnames == (
        "cars-zero",
        "cars-one",
        "pets-zero",
        "pets-one",
        "sun-zero",
        "sun-one",
    )
    assert built.label_offsets == {"cars": 0, "pets": 2, "sun": 4}
    assert built.plan["api"] == "build_balanced_vision8_calibration_loaders"
    assert built.plan["union_classnames"] == list(built.union_classnames)
    assert len(json.dumps(built.plan)) > 0


def test_vision8_api_rejects_endpoint_classname_mismatch() -> None:
    source, target = _datasets(size=8)
    contexts = {
        "cars": {
            "source_dataset": source["cars"],
            "target_dataset": target["cars"],
            "source_classnames": ["car-a", "car-b"],
            "target_classnames": ["car-b", "car-a"],
        },
    }
    with pytest.raises(ValueError, match="classnames are not aligned"):
        build_balanced_vision8_calibration_loaders(contexts, 1, 1, 0)
