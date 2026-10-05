from __future__ import annotations

from merge_and_rebase.utils.alpha_search import PerTaskAlphaTracker, average_scores


def test_average_scores_returns_mean() -> None:
    assert average_scores([1.0, 2.0, 5.0]) == 8.0 / 3.0
    assert average_scores([]) == 0.0


def test_per_task_alpha_tracker_updates_and_early_stops() -> None:
    tracker = PerTaskAlphaTracker(task_names=["Cars", "DTD"], initial_alpha=0.0)

    stopped_reb, stopped_base = tracker.update(
        alpha=0.0, indices=[0, 1], primary_accs=[0.15, 0.25], secondary_accs=[0.10, 0.20]
    )
    assert stopped_reb == stopped_base == []
    assert tracker.active_indices() == [0, 1]
    assert tracker.best_alpha == [0.0, 0.0]
    assert tracker.best_rebase_acc == [0.15, 0.25]
    assert tracker.best_baseline_acc == [0.10, 0.20]
    assert tracker.best_secondary_alpha == [0.0, 0.0]

    stopped_reb, stopped_base = tracker.update(
        alpha=0.1, indices=[0, 1], primary_accs=[0.18, 0.25], secondary_accs=[0.10, 0.20]
    )
    assert stopped_reb == stopped_base == []
    assert tracker.active_indices() == [0, 1]
    assert tracker.best_alpha == [0.1, 0.0]
    assert tracker.best_rebase_acc == [0.18, 0.25]
    # Baseline stayed flat -> its best alpha holds at initial 0.0
    assert tracker.best_secondary_alpha == [0.0, 0.0]
    assert tracker.best_baseline_acc == [0.10, 0.20]

    stopped_reb, stopped_base = tracker.update(
        alpha=0.2, indices=[0, 1], primary_accs=[0.17, 0.24], secondary_accs=[0.08, 0.18]
    )
    assert stopped_reb == [0, 1]
    assert stopped_base == [0, 1]
    assert tracker.active_indices() == []
    # Best are frozen at the peak from the previous step.
    assert tracker.best_alpha == [0.1, 0.0]
    assert tracker.best_rebase_acc == [0.18, 0.25]
    assert tracker.best_baseline_acc == [0.10, 0.20]
    assert tracker.best_secondary_alpha == [0.0, 0.0]


def test_per_task_alpha_tracker_respects_patience() -> None:
    tracker = PerTaskAlphaTracker(task_names=["Cars"], initial_alpha=0.0, patience=1)

    stopped_reb, stopped_base = tracker.update(
        alpha=0.0, indices=[0], primary_accs=[0.20], secondary_accs=[0.10]
    )
    assert stopped_reb == stopped_base == []
    assert tracker.active_indices() == [0]

    # Both streams regress this step (primary 0.19 < 0.20, secondary 0.09 < 0.10).
    stopped_reb, stopped_base = tracker.update(
        alpha=0.1, indices=[0], primary_accs=[0.19], secondary_accs=[0.09]
    )
    assert stopped_reb == stopped_base == []
    assert tracker.active_indices() == [0]

    # Both streams regress again -> patience=1 trips, both early-stop.
    stopped_reb, stopped_base = tracker.update(
        alpha=0.2, indices=[0], primary_accs=[0.18], secondary_accs=[0.08]
    )
    assert stopped_reb == [0]
    assert stopped_base == [0]
    assert tracker.active_indices() == []


def test_per_task_alpha_tracker_resets_bad_steps_on_tie_or_improvement() -> None:
    tracker = PerTaskAlphaTracker(task_names=["Cars"], initial_alpha=0.0, patience=1)

    tracker.update(alpha=0.0, indices=[0], primary_accs=[0.20], secondary_accs=[0.10])
    stopped_reb, stopped_base = tracker.update(
        alpha=0.1, indices=[0], primary_accs=[0.19], secondary_accs=[0.10]
    )
    assert stopped_reb == stopped_base == []
    # Primary regressed -> bad_steps++.
    assert tracker.bad_steps == [1]
    # Secondary was flat (0.10 == 0.10) -> a tie, resets to 0.
    assert tracker.secondary_bad_steps == [0]

    stopped_reb, stopped_base = tracker.update(
        alpha=0.2, indices=[0], primary_accs=[0.20], secondary_accs=[0.12]
    )
    assert stopped_reb == stopped_base == []
    # Both streams improved/tied at this step -> both counters reset.
    assert tracker.bad_steps == [0]
    assert tracker.secondary_bad_steps == [0]
    assert tracker.active_indices() == [0]
    # Best alpha for secondary now advances.
    assert tracker.best_secondary_alpha == [0.2]
    assert tracker.best_baseline_acc == [0.12]


def test_per_task_alpha_tracker_decoupled_best_alphas() -> None:
    """Primary and secondary can peak at different alphas; each tracks its own best."""
    tracker = PerTaskAlphaTracker(task_names=["Cars"], initial_alpha=0.0, patience=10)

    # alpha=0.0: primary=0.20, secondary=0.50
    tracker.update(alpha=0.0, indices=[0], primary_accs=[0.20], secondary_accs=[0.50])
    assert tracker.best_alpha == [0.0]
    assert tracker.best_rebase_acc == [0.20]
    assert tracker.best_secondary_alpha == [0.0]
    assert tracker.best_baseline_acc == [0.50]

    # alpha=0.1: primary improves to 0.30, secondary drops to 0.45
    tracker.update(alpha=0.1, indices=[0], primary_accs=[0.30], secondary_accs=[0.45])
    assert tracker.best_alpha == [0.1]
    assert tracker.best_rebase_acc == [0.30]
    # Secondary's best stays at alpha=0.0 (its peak), not at primary's peak.
    assert tracker.best_secondary_alpha == [0.0]
    assert tracker.best_baseline_acc == [0.50]

    # alpha=0.2: primary drops to 0.25, secondary improves to 0.60
    tracker.update(alpha=0.2, indices=[0], primary_accs=[0.25], secondary_accs=[0.60])
    assert tracker.best_alpha == [0.1]  # primary peak still at 0.1
    assert tracker.best_rebase_acc == [0.30]
    # Secondary's best advances to 0.2 — independent of primary.
    assert tracker.best_secondary_alpha == [0.2]
    assert tracker.best_baseline_acc == [0.60]

    assert tracker.best_primary_alpha != tracker.best_secondary_alpha


def test_per_task_alpha_tracker_primary_stops_but_secondary_continues() -> None:
    """When the primary stream early-stops a task, the secondary stream keeps going."""
    tracker = PerTaskAlphaTracker(task_names=["Cars"], initial_alpha=0.0, patience=0)

    # alpha=0.0 sets peaks for both (patience=0 means any drop stops a task next step).
    stopped_reb, stopped_base = tracker.update(
        alpha=0.0, indices=[0], primary_accs=[0.30], secondary_accs=[0.40]
    )
    assert stopped_reb == stopped_base == []

    # alpha=0.1: primary drops -> stops; secondary improves -> stays active.
    stopped_reb, stopped_base = tracker.update(
        alpha=0.1, indices=[0], primary_accs=[0.29], secondary_accs=[0.45]
    )
    assert stopped_reb == [0]
    assert stopped_base == []
    assert tracker.primary_active == [False]
    assert tracker.secondary_active == [True]
    # Primary best frozen at 0.0.
    assert tracker.best_primary_alpha == [0.0]
    assert tracker.best_primary_acc == [0.30]
    # Secondary advanced.
    assert tracker.best_secondary_alpha == [0.1]
    assert tracker.best_secondary_acc == [0.45]

    # alpha=0.2: primary no longer evaluated (placeholder -inf); secondary improves again.
    stopped_reb, stopped_base = tracker.update(
        alpha=0.2, indices=[0], primary_accs=[float("-inf")], secondary_accs=[0.50]
    )
    assert stopped_reb == []
    assert stopped_base == []
    assert tracker.best_primary_alpha == [0.0]  # still frozen
    assert tracker.best_primary_acc == [0.30]
    # Secondary keeps advancing while primary is stopped.
    assert tracker.best_secondary_alpha == [0.2]
    assert tracker.best_secondary_acc == [0.50]

    # eval_active_indices() reflects the union (only secondary active now).
    assert tracker.primary_active_indices() == []
    assert tracker.secondary_active_indices() == [0]
    assert tracker.eval_active_indices() == [0]


def test_per_task_alpha_tracker_legacy_aliases() -> None:
    """The legacy single-stream aliases still read consistently with the primary stream."""
    tracker = PerTaskAlphaTracker(task_names=["Cars", "DTD"], initial_alpha=0.0)

    tracker.update(alpha=0.5, indices=[0, 1], primary_accs=[0.40, 0.55], secondary_accs=[0.30, 0.45])
    # Aliases are read-only and mirror the new fields.
    assert tracker.best_alpha is tracker.best_primary_alpha
    assert tracker.best_rebase_acc is tracker.best_primary_acc
    assert tracker.active is tracker.primary_active
    assert tracker.bad_steps is tracker.primary_bad_steps
    assert tracker.best_baseline_acc is tracker.best_secondary_acc
    assert tracker.active_indices() == tracker.primary_active_indices()