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


def test_select_alpha_does_not_early_stop_on_a_plateau() -> None:
    """The swap runner's alpha rule must match the main runner's.

    `_select_alpha` once advanced its bad-step counter on any non-improving
    step, including an exact tie, while `PerTaskAlphaTracker` resets on a
    plateau and counts only a genuine decline. The real EuroSAT shared/shared
    validation curve plateaus at 0.7741 for seven consecutive alphas, so the
    old rule exhausted patience=5 there and returned alpha=2.3 after visiting
    30 of 101 grid points -- while the in-memory runner continued past the
    plateau to alpha=6.1. That one difference made the swap table read 79.92%
    against the in-memory runner's 87.49% on a matched cell, and was initially
    misdiagnosed as a defect in artifact reconstruction.
    """
    from merge_and_rebase.eval.vision_brace_transport_swap import _select_alpha

    curve = [
        (0.0, 0.50), (0.1, 0.55), (1.8, 0.7519), (1.9, 0.7593), (2.0, 0.7593),
        (2.1, 0.7667), (2.2, 0.7667),
        # seven-step plateau: must not exhaust patience
        (2.3, 0.7741), (2.4, 0.7741), (2.5, 0.7741), (2.6, 0.7741),
        (2.7, 0.7741), (2.8, 0.7741), (2.9, 0.7741),
        # the curve then resumes climbing to its real optimum
        (3.0, 0.78), (4.0, 0.82), (5.0, 0.85), (6.1, 0.8593),
        (7.0, 0.80), (8.0, 0.70), (9.0, 0.60), (10.0, 0.50),
    ]

    alpha, score, visited = _select_alpha(curve, 5)

    assert alpha == 6.1, f"plateau terminated the search early at alpha={alpha}"
    assert score == 0.8593
    assert len(visited) == len(curve)


def test_select_alpha_still_stops_on_a_genuine_decline() -> None:
    """Plateau tolerance must not disable early stopping altogether."""
    from merge_and_rebase.eval.vision_brace_transport_swap import _select_alpha

    curve = [(float(i), acc) for i, acc in enumerate([0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])]
    alpha, score, visited = _select_alpha(curve, 2)

    assert alpha == 1.0
    assert score == 0.9
    assert len(visited) < len(curve), "early stopping never triggered on a monotone decline"


def test_select_alpha_matches_per_task_alpha_tracker() -> None:
    """Pin the two selection paths together so they cannot silently diverge.

    `_select_alpha` delegates to `PerTaskAlphaTracker`; this asserts agreement
    on curves shaped like the ones that exposed the original discrepancy.
    """
    from merge_and_rebase.eval.vision_brace_transport_swap import _select_alpha
    from merge_and_rebase.utils.alpha_search import PerTaskAlphaTracker

    curves = [
        [(0.0, 0.50), (1.0, 0.60), (2.0, 0.60), (3.0, 0.60), (4.0, 0.75), (5.0, 0.70)],
        [(0.0, 0.80), (1.0, 0.70), (2.0, 0.60), (3.0, 0.50)],
        [(0.0, 0.10), (1.0, 0.20), (2.0, 0.30), (3.0, 0.40), (4.0, 0.50)],
        [(0.0, 0.42)] * 8,
    ]
    for curve in curves:
        alpha, score, _ = _select_alpha(curve, 2)
        tracker = PerTaskAlphaTracker(task_names=["cell"], initial_alpha=curve[0][0], patience=2)
        for a, s in curve:
            if not tracker.primary_active[0]:
                break
            tracker.update(alpha=a, indices=[0], primary_accs=[s], secondary_accs=[s])
        assert alpha == tracker.best_primary_alpha[0]
        assert score == tracker.best_primary_acc[0]
