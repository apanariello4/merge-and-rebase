"""P6.10: the decoder warns about vision-only BRACE fields it ignores."""

from __future__ import annotations

import warnings

import pytest

from merge_and_rebase.rebase.block_extension.config import warn_decoder_ignored_fields


def test_non_default_vision_only_fields_warn_and_are_returned():
    params = {"collapse_schedule": "last_first", "ridge_weight": 1e-3, "correction_scope": "all", "n_batches_act": 4}
    with pytest.warns(RuntimeWarning, match="ignores the vision-only"):
        ignored = warn_decoder_ignored_fields(params)
    assert ignored == ["collapse_schedule", "correction_scope", "ridge_weight"]


def test_defaults_and_honoured_fields_are_silent():
    params = {"ridge_weight": 1e-6, "skip_final_ln": True, "skip_correction": True, "extension_strategy": "x"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_decoder_ignored_fields(params) == []
        assert warn_decoder_ignored_fields(None) == []
