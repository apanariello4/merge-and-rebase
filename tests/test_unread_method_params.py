"""THESEUS / BiCo ``prepare`` warn about ``method_params`` keys that nothing reads (they were dropped silently)."""

from __future__ import annotations

import warnings

import pytest

from merge_and_rebase.rebase.methods._shared import warn_unread_method_params


def test_unread_keys_warn():
    with pytest.warns(
        RuntimeWarning, match=r"\[theseus\] method_params not read by this method and ignored: \['bogus'\]"
    ):
        warn_unread_method_params("theseus", {"bogus": 1})


def test_keys_read_by_apply_and_empty_leftovers_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_unread_method_params("bico", {"zero_attention_delta": True}, read_by_apply=("zero_attention_delta",))
        warn_unread_method_params("theseus", {})
