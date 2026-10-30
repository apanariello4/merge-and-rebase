"""Opt-in val_size: validation from TRAIN (removed from it), whole test kept; default carve unchanged."""

from __future__ import annotations

import datasets
import numpy as np
import torch
from PIL import Image

from merge_and_rebase.data.vision_loaders import build_vision_loaders


def _hf(n_train, n_test):
    feats = datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=["a", "b"])})

    def split(n, off):
        imgs = [Image.fromarray(np.full((4, 4, 3), (i + off) % 255, dtype=np.uint8)) for i in range(n)]
        return datasets.Dataset.from_dict({"image": imgs, "label": [i % 2 for i in range(n)]}, features=feats)

    return datasets.DatasetDict({"train": split(n_train, 0), "test": split(n_test, 100)})


def _pre(img):
    return torch.as_tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1)


def _pixels(ds):
    return sorted(int(ds[i][0][0, 0, 0]) for i in range(len(ds)))


def test_default_carves_val_from_test():
    loaders = build_vision_loaders(_hf(40, 20), hf_path="toy", preprocess=_pre, ft_epochs=1, val_fraction=0.25, seed=3)
    assert (len(loaders.train.dataset), len(loaders.val.dataset), len(loaders.test.dataset)) == (40, 5, 15)


def test_val_size_draws_val_from_train_and_keeps_full_test():
    loaders = build_vision_loaders(_hf(40, 20), hf_path="toy", preprocess=_pre, ft_epochs=1, val_size=12, seed=3)
    assert (len(loaders.train.dataset), len(loaders.val.dataset), len(loaders.test.dataset)) == (28, 12, 20)
    train, val = set(_pixels(loaders.train.dataset)), set(_pixels(loaders.val.dataset))
    assert not train & val and train | val == set(range(40))
    again = build_vision_loaders(_hf(40, 20), hf_path="toy", preprocess=_pre, ft_epochs=1, val_size=12, seed=3)
    assert _pixels(again.val.dataset) == _pixels(loaders.val.dataset)
