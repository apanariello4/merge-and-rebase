"""Every calibrating LLM method evaluates on the calibration hold-out only (release validation finding, 2026-10-03).

Ariadne was missing from the methods that resolve the evaluation slice up front, so its harness calls received
``samples=None`` and scored the full task, including the documents its calibration text was drawn from.
"""

from __future__ import annotations

import pytest
from golden.test_llm_main_golden import World, _base_cfg, _launch

from merge_and_rebase.rebase.capabilities import uses_calibration

HOLDOUT = {"arc_easy": [0, 1, 2], "piqa": [3, 4]}  # what the faked calibration resolver returns as eval_samples
_METHODS = {
    "theseus": {"method": "theseus"},
    "theseus_gqa": {"method": "theseus_gqa"},
    "bico": {"method": "bico"},
    "ariadne": {"method": "ariadne", "ariadne_params": {"preset": "ariadne", "num_batches": 2, "seed": 0}},
    "direct_residual": {"method": "direct_residual", "ariadne_params": {"preset": "ariadne", "num_batches": 2}},
}


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_harness_scores_only_the_holdout(method, tmp_path, monkeypatch):
    assert uses_calibration(method)
    cfg = _base_cfg(tmp_path, **_METHODS[method], eval_before_rebase=True, depth_defaults="method")
    cfg.pop("calibration_prompts")  # harness-derived calibration: the source of the hold-out
    calls = _launch(cfg, tmp_path, monkeypatch, World(), fake_calibration=True)
    assert len(calls.harness) >= 2  # before-rebase reference and the rebased model
    assert all(call["samples"] == HOLDOUT for call in calls.harness), [c["samples"] for c in calls.harness]


@pytest.mark.parametrize("method", ["identity", "transfusion", "not_a_method"])
def test_non_calibrating_methods(method):
    assert not uses_calibration(method)
