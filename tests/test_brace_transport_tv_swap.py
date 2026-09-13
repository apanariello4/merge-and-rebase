from __future__ import annotations

import torch

from merge_and_rebase.eval.vision_brace_transport_swap import _select_alpha, _validate_pair


def test_alpha_selection_uses_legacy_patience_and_smaller_alpha_tie() -> None:
    alpha, score, visited = _select_alpha([(0.0, 0.3), (0.1, 0.4), (0.2, 0.4), (0.3, 0.2)], patience=1)
    assert alpha == 0.1
    assert score == 0.4
    assert visited == [(0.0, 0.3), (0.1, 0.4), (0.2, 0.4), (0.3, 0.2)]


def test_pair_validation_rejects_incompatible_saved_vectors() -> None:
    shared = {
        "Cars": {
            "base": {"visual.a": torch.zeros(2)},
            "tv": {"visual.a": torch.zeros(2)},
            "metadata": {"source_depth": 12, "target_depth": 24, "calibration": {"seed": 89}},
        }
    }
    skip = {
        "Cars": {
            "base": {"visual.a": torch.zeros(2)},
            "tv": {"visual.a": torch.zeros(3)},
            "metadata": {"source_depth": 12, "target_depth": 24, "calibration": {"seed": 89}},
        }
    }
    try:
        _validate_pair(shared, skip)
    except ValueError as exc:
        assert "TV shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected incompatible saved vectors to fail")
