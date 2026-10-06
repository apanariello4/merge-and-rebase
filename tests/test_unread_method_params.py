"""THESEUS / BiCo ``prepare`` reject ``method_params`` keys that nothing reads (they were dropped silently)."""

from __future__ import annotations

import pytest

from merge_and_rebase.rebase.methods._shared import reject_unread_method_params


def test_unread_keys_raise():
    with pytest.raises(ValueError, match=r"\[theseus\] unknown method_params \(not read by this method\): \['bogus'\]"):
        reject_unread_method_params("theseus", {"bogus": 1})


def test_keys_read_by_apply_and_empty_leftovers_pass():
    reject_unread_method_params("bico", {"zero_attention_delta": True}, read_by_apply=("zero_attention_delta",))
    reject_unread_method_params("theseus", {})
