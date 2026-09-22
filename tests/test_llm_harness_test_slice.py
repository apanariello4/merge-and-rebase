"""The held-out test slice must be disjoint from the alpha-search slice.

`llm_rebase` selects alpha by scoring every candidate on `harness_samples` and
then reporting the winner. Selecting and reporting on the same documents makes
the reported number a maximum over the alpha grid on that slice -- biased
upward, and on a slice of ~141 documents biased by more than the differences
between the arms being compared.

`harness_test_samples` adds a disjoint slice scored once at the winning alpha.
These tests pin the validation around it, because the failure mode is silent:
an overlapping slice still produces a plausible number, just one selected on
the documents it is scored on.
"""

from __future__ import annotations

import pytest


def _validate(test_samples_cfg, search_samples):
    """Mirror of llm_rebase's harness_test_samples validation."""
    if not isinstance(test_samples_cfg, dict):
        raise ValueError("config['harness_test_samples'] must map task names to index lists.")
    test_samples = {}
    for task_name, indices in test_samples_cfg.items():
        if not isinstance(indices, list) or not all(isinstance(i, int) and i >= 0 for i in indices):
            raise ValueError("config['harness_test_samples'] values must be lists of non-negative indices.")
        test_samples[str(task_name)] = list(indices)
    overlap = {
        t: sorted(set(test_samples.get(t, ())) & set((search_samples or {}).get(t, ())))
        for t in test_samples
    }
    leaking = {t: v for t, v in overlap.items() if v}
    if leaking:
        raise ValueError(
            "harness_test_samples overlaps the alpha-search slice for "
            f"{ {t: len(v) for t, v in leaking.items()} }; the reported number would be "
            "selected on documents it is scored on."
        )
    return test_samples


def test_disjoint_slices_are_accepted():
    out = _validate({"ifeval": [5, 6, 7]}, {"ifeval": [1, 2, 3]})
    assert out == {"ifeval": [5, 6, 7]}


def test_overlapping_slices_are_refused():
    with pytest.raises(ValueError, match="overlaps the alpha-search slice"):
        _validate({"ifeval": [3, 4, 5]}, {"ifeval": [1, 2, 3]})


def test_overlap_is_detected_per_task_not_globally():
    """A clean task must not mask a leaking one."""
    with pytest.raises(ValueError, match="overlaps"):
        _validate({"ifeval": [9], "hellaswag": [1]}, {"ifeval": [50], "hellaswag": [1]})


def test_malformed_indices_are_refused():
    with pytest.raises(ValueError, match="non-negative indices"):
        _validate({"ifeval": [-1]}, {"ifeval": [0]})
    with pytest.raises(ValueError, match="must map task names"):
        _validate([0, 1, 2], {"ifeval": [9]})


def test_absent_search_slice_means_nothing_to_overlap():
    """External calibration leaves harness_samples None; the whole task is free."""
    assert _validate({"ifeval": [0, 1]}, None) == {"ifeval": [0, 1]}
