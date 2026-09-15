from __future__ import annotations

import pytest
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


def test_resolve_banks_defaults_to_the_historical_shared_skip_crossing() -> None:
    """An existing 20260913 config must keep producing the same 2x2 table."""
    from merge_and_rebase.eval.vision_brace_transport_swap import DEFAULT_BANKS, _resolve_banks

    assert _resolve_banks({}) == list(DEFAULT_BANKS) == ["shared", "skip"]


def test_resolve_banks_accepts_the_independent_bank_for_the_full_crossing() -> None:
    from merge_and_rebase.eval.vision_brace_transport_swap import _resolve_banks

    assert _resolve_banks({"banks": ["shared", "skip", "independent"]}) == ["shared", "skip", "independent"]
    assert _resolve_banks({"banks": "shared,independent"}) == ["shared", "independent"]


def test_resolve_banks_requires_the_shared_alpha_anchor() -> None:
    """`run` anchors its reported alpha on shared/shared, so shared must exist."""
    from merge_and_rebase.eval.vision_brace_transport_swap import _resolve_banks

    with pytest.raises(ValueError, match="alpha anchor"):
        _resolve_banks({"banks": ["skip", "independent"]})


def test_resolve_banks_rejects_unknown_and_repeated_conditions() -> None:
    from merge_and_rebase.eval.vision_brace_transport_swap import _resolve_banks

    with pytest.raises(ValueError, match="Unknown correction banks"):
        _resolve_banks({"banks": ["shared", "steer"]})
    with pytest.raises(ValueError, match="distinct"):
        _resolve_banks({"banks": ["shared", "shared"]})


def test_persisting_transported_deltas_records_a_verifiable_hash(tmp_path) -> None:
    """The merge stage identifies its input by this hash, so it must be exact."""
    from merge_and_rebase.eval.vision_brace_transport_swap import _persist_transported_deltas
    from merge_and_rebase.eval.vision_brace_tv_swap import state_dict_sha256

    transported = {
        (act, tv): {"visual.w": torch.full((2,), float(index))}
        for index, (act, tv) in enumerate(
            [(a, b) for a in ("shared", "skip") for b in ("shared", "skip")]
        )
    }
    records = _persist_transported_deltas(
        {"save_transported_deltas_root": str(tmp_path), "campaign": "unit"},
        transported=transported, task="Cars", method_name="bico",
        activation_banks=["shared", "skip"], vector_banks=["shared", "skip"],
    )

    assert sorted(records) == ["shared__shared", "shared__skip", "skip__shared", "skip__skip"]
    for cell, record in records.items():
        reloaded = torch.load(tmp_path / "bico" / "Cars" / f"{cell}.pt", weights_only=True)
        assert state_dict_sha256(reloaded) == record["sha256"]
    assert (tmp_path / "bico" / "Cars" / "COMPLETE").is_file()


def test_persisting_transported_deltas_is_off_by_default() -> None:
    """Persistence is opt-in so the 20260913 campaign replays byte-identically."""
    from merge_and_rebase.eval.vision_brace_transport_swap import _persist_transported_deltas

    assert _persist_transported_deltas(
        {}, transported={}, task="Cars", method_name="bico", activation_banks=[], vector_banks=[],
    ) is None


def test_persisting_transported_deltas_refuses_to_overwrite_a_run(tmp_path) -> None:
    """Historical results are never overwritten, artifacts included."""
    from merge_and_rebase.eval.vision_brace_transport_swap import _persist_transported_deltas

    (tmp_path / "bico" / "Cars").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _persist_transported_deltas(
            {"save_transported_deltas_root": str(tmp_path)},
            transported={("shared", "shared"): {"visual.w": torch.zeros(2)}},
            task="Cars", method_name="bico", activation_banks=["shared"], vector_banks=["shared"],
        )


def test_bank_axes_default_to_the_single_banks_field() -> None:
    """`banks` still sets both axes, so existing configs keep their meaning."""
    from merge_and_rebase.eval.vision_brace_transport_swap import _resolve_bank_axes

    assert _resolve_bank_axes({}) == (["shared", "skip"], ["shared", "skip"])
    assert _resolve_bank_axes({"banks": ["shared", "skip", "independent"]}) == (
        ["shared", "skip", "independent"], ["shared", "skip", "independent"],
    )


def test_bank_axes_can_be_set_independently() -> None:
    """The asymmetric grid this campaign needs: two fits, three vectors.

    BRACE builds the corrected base identically under shared and independent
    correction, so naming independent on the activation axis would recompute the
    shared column. It is only on the task-vector axis that it is a different
    vector.
    """
    from merge_and_rebase.eval.vision_brace_transport_swap import _resolve_bank_axes

    activation, vector = _resolve_bank_axes({
        "activation_banks": ["shared", "skip"],
        "vector_banks": ["shared", "skip", "independent"],
    })
    assert activation == ["shared", "skip"]
    assert vector == ["shared", "skip", "independent"]


def test_bank_axes_validate_each_axis_by_name() -> None:
    from merge_and_rebase.eval.vision_brace_transport_swap import _resolve_bank_axes

    with pytest.raises(ValueError, match="activation_banks"):
        _resolve_bank_axes({"activation_banks": ["skip"], "vector_banks": ["shared"]})
    with pytest.raises(ValueError, match="vector_banks"):
        _resolve_bank_axes({"activation_banks": ["shared"], "vector_banks": ["shared", "bogus"]})


def test_determinism_switch_defaults_off_and_is_recorded(monkeypatch) -> None:
    """Seeding is opt-in, and every summary states which way it ran.

    The swap runner borrowed vision_rebase's prepare/context helpers but not its
    `_set_deterministic_seed` call, so BiCo -- which runs a backward pass to
    populate its hooks -- fitted under unseeded, non-deterministic kernels. All
    32 BiCo cells of the 20260914 campaign disagreed with their 20260913
    counterparts at hash level while all 32 Theseus cells matched. The switch
    stays off by default so the historical campaigns replay as they ran, and the
    resolved value is returned so a summary can never be ambiguous about it.
    """
    import merge_and_rebase.eval.vision_brace_transport_swap as swap

    seeded: list[int] = []
    monkeypatch.setattr(swap, "_set_deterministic_seed", lambda seed: seeded.append(int(seed)))

    assert swap._apply_determinism({}) is False
    assert swap._apply_determinism({"deterministic": False, "seed": 89}) is False
    assert seeded == []

    assert swap._apply_determinism({"deterministic": True, "seed": 89}) is True
    assert seeded == [89]


def test_determinism_switch_seeds_from_the_run_seed(monkeypatch) -> None:
    """The campaign seed is the one pinned, not a hardcoded default."""
    import merge_and_rebase.eval.vision_brace_transport_swap as swap

    seeded: list[int] = []
    monkeypatch.setattr(swap, "_set_deterministic_seed", lambda seed: seeded.append(int(seed)))

    swap._apply_determinism({"deterministic": True, "seed": 7})
    assert seeded == [7]


def test_strict_determinism_demands_deterministic_kernels(monkeypatch) -> None:
    """`warn_only=True` is not enough for BiCo, so strict mode is separate.

    `_set_deterministic_seed` asks with `warn_only=True`, under which PyTorch
    warns about a non-deterministic kernel and runs it anyway. BiCo's backward
    pass reaches the memory-efficient attention backward, which did exactly that
    in the 20260915 probe: four runs, four different transported deltas, with the
    warning naming the kernel. Strict mode re-asserts the request with
    `warn_only=False` so such an op raises rather than silently varying.
    """
    import torch

    import merge_and_rebase.eval.vision_brace_transport_swap as swap

    seeded: list[int] = []
    asked: list[bool] = []
    monkeypatch.setattr(swap, "_set_deterministic_seed", lambda seed: seeded.append(int(seed)))
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda flag, warn_only=False: asked.append(bool(warn_only))
    )

    assert swap._apply_determinism({"deterministic": True, "seed": 89}) is True
    assert asked == [], "non-strict mode must not re-assert the flag"

    # Strict implies deterministic: seeding is a precondition, not an alternative.
    assert swap._apply_determinism({"deterministic_strict": True, "seed": 89}) is True
    assert seeded == [89, 89]
    assert asked == [False]
