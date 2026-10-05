from __future__ import annotations

import pytest

from merge_and_rebase.eval.utils import resolve_eval_split_loader


class _Loaders:
    def __init__(self, with_val: bool = True) -> None:
        self.test = object()
        if with_val:
            self.val = object()


def test_non_strict_returns_val_when_present() -> None:
    loaders = _Loaders()
    assert resolve_eval_split_loader(loaders, "val") is loaders.val


def test_non_strict_falls_back_to_test_on_missing_val() -> None:
    loaders = _Loaders(with_val=False)
    assert resolve_eval_split_loader(loaders, "val") is loaders.test


def test_non_strict_falls_back_to_test_on_unknown_split() -> None:
    loaders = _Loaders()
    assert resolve_eval_split_loader(loaders, "validation") is loaders.test


def test_explicit_test_split() -> None:
    loaders = _Loaders()
    assert resolve_eval_split_loader(loaders, "test") is loaders.test
    assert resolve_eval_split_loader(loaders, "TEST", strict=True) is loaders.test


def test_strict_raises_on_unknown_split() -> None:
    with pytest.raises(ValueError, match="Unknown eval split"):
        resolve_eval_split_loader(_Loaders(), "train", strict=True)


def test_strict_keeps_prior_behavior_for_missing_val() -> None:
    loaders = _Loaders(with_val=False)
    with pytest.raises(AttributeError):
        resolve_eval_split_loader(loaders, "val", strict=True)
