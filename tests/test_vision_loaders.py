from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage
from PIL import Image

from merge_and_rebase.data.vision_loaders import KMNIST_CLASSNAMES, build_vision_loaders, emnist_fix_transform, load_hf_splits


def _identity(x):
    return x


def test_load_hf_splits_loads_only_requested_splits(monkeypatch) -> None:
    calls: list[tuple[str | None, str | None]] = []
    datasets_by_split = {
        "train": Dataset.from_dict({"label": [0, 1]}),
        "test": Dataset.from_dict({"label": [1, 0]}),
    }

    def _fake_load_dataset(path, *args, split=None, **kwargs):
        config = args[0] if args else None
        calls.append((config, split))
        if split is None:
            raise AssertionError("Fallback loading should not run when requested splits load directly.")
        return datasets_by_split[split]

    monkeypatch.setattr("merge_and_rebase.data.vision_loaders.hf_load_dataset", _fake_load_dataset)

    ds = load_hf_splits("tanganke/sun397", requested_splits=("train", "test"))

    assert list(ds.keys()) == ["train", "test"]
    assert calls == [(None, "train"), (None, "test")]


def test_load_hf_splits_rejects_unsupported_requested_split() -> None:
    with pytest.raises(ValueError, match="Unsupported requested splits"):
        load_hf_splits("tanganke/sun397", requested_splits=("train", "foo"))


def test_emnist_fix_transform_is_picklable() -> None:
    img = Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8), mode="L")
    transform = emnist_fix_transform(_identity)

    pickle.dumps(transform)

    out = torch.from_numpy(np.array(transform(img)))
    expected = torch.tensor([[0, 2], [1, 3]], dtype=torch.uint8)

    assert torch.equal(out, expected)


def test_kmnist_loader_uses_kana_classnames() -> None:
    features = Features({"image": HFImage(), "label": ClassLabel(names=[str(i) for i in range(10)])})
    img = Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L")
    ds = Dataset.from_dict({"image": [img] * 20, "label": list(range(10)) * 2}, features=features)
    loaders = build_vision_loaders(
        hf_ds=DatasetDict({"train": ds, "test": ds}),
        hf_path="tanganke/kmnist",
        preprocess=_identity,
        ft_epochs=1,
        batch_size=2,
        num_workers=0,
    )

    assert list(loaders.classnames) == KMNIST_CLASSNAMES
