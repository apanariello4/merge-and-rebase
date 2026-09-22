"""Tests for direct_residual_params.merge_mode="merge_in_source_then_fit".

This is a Direct-Residual-specific config field (`DirectResidualConfig.merge_mode`),
distinct from vision_rebase.py's own top-level `merge_mode` cfg key. It fits
Direct Residual's correction exactly once, against the native source deltas
merged first -- so a per-task alpha search over an identical delta would just
search the same objective under a different name per task, which
vision_rebase.py rejects up front (mirroring the existing shared-alpha
requirement for `merge_mode="merge_then_brace_then_transport"`).

The guard fires very early in `main()` -- right after `direct_residual_params`
is parsed, using `cfg.get("alpha_selection", "shared")` directly rather than
the (not-yet-resolved-at-that-point) `alpha_selection` local -- well before
any network/model-build step, so both the "raises" and the "proceeds past
this guard" cases are exercised through a real `main()` call, following
`tests/test_vision_rebase_direct_residual_dispatch.py`'s technique. The
"proceeds" case is confirmed by checking execution reaches the *next* config
validation failure (missing tuned checkpoints) rather than this guard's error.
"""

from __future__ import annotations

import json

import pytest

from merge_and_rebase.eval import vision_rebase


def _run_main_with_cfg(monkeypatch, tmp_path, cfg: dict) -> Exception:
    cfg = dict(cfg)
    cfg.setdefault("logging", {"local_log_dir": str(tmp_path / "logs")})
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr("sys.argv", ["vision_rebase", "--config", str(config_path)])
    with pytest.raises(Exception) as excinfo:
        vision_rebase.main()
    return excinfo.value


def test_merge_in_source_then_fit_raises_under_per_task_alpha_selection(monkeypatch, tmp_path):
    exc = _run_main_with_cfg(
        monkeypatch,
        tmp_path,
        {
            "method": "direct_residual",
            "direct_residual_params": {"merge_mode": "merge_in_source_then_fit"},
            "alpha_selection": "per_task",
        },
    )
    assert isinstance(exc, ValueError)
    assert "merge_in_source_then_fit" in str(exc)
    assert "alpha_selection='shared'" in str(exc)


def test_merge_in_source_then_fit_proceeds_under_shared_alpha_selection(monkeypatch, tmp_path):
    exc = _run_main_with_cfg(
        monkeypatch,
        tmp_path,
        {
            "method": "direct_residual",
            "direct_residual_params": {"merge_mode": "merge_in_source_then_fit"},
            "alpha_selection": "shared",
        },
    )
    # The merge_in_source_then_fit guard did not fire; execution reached a
    # later, unrelated validation error (no tuned checkpoints were provided).
    assert "merge_in_source_then_fit" not in str(exc)
    assert "tuned checkpoints" in str(exc)


def test_merge_in_source_then_fit_default_alpha_selection_is_shared_and_proceeds(monkeypatch, tmp_path):
    """alpha_selection defaults to 'shared' when omitted entirely, so the guard
    is inert by default -- an ablation switch matching this file's CLAUDE.md
    'current behavior as default' convention.
    """
    exc = _run_main_with_cfg(
        monkeypatch,
        tmp_path,
        {
            "method": "direct_residual",
            "direct_residual_params": {"merge_mode": "merge_in_source_then_fit"},
        },
    )
    assert "merge_in_source_then_fit" not in str(exc)
    assert "tuned checkpoints" in str(exc)


def test_direct_residual_rejects_single_transport_merge_modes(monkeypatch, tmp_path):
    """method='direct_residual' has no transport() call for the top-level
    merge-mode dispatch (merge_then_rebase / brace_merge_then_transport /
    merge_then_brace_then_transport) to invoke; this must be rejected early
    with a clear message rather than crashing later on a missing attribute.
    """
    exc = _run_main_with_cfg(
        monkeypatch,
        tmp_path,
        {
            "method": "direct_residual",
            "merge_mode": "merge_then_rebase",
        },
    )
    assert isinstance(exc, ValueError)
    assert "does not support merge_mode='merge_then_rebase'" in str(exc)


def test_final_summary_carries_alignment_and_correction_fit_timing_keys():
    """Smoke test: the timing-bracket dicts are threaded into final_summary
    as sibling keys next to transport_timings, always present (default {}),
    per the plan's uniform-downstream-parsing requirement. Verified via
    source inspection since final_summary is only ever assembled at the very
    end of a (network-requiring) main() run.
    """
    import inspect

    source = inspect.getsource(vision_rebase.main)
    assert '"alignment_calibration_timings": alignment_calibration_timings' in source
    assert '"correction_fit_timings": correction_fit_timings' in source
    assert '"transport_timings": transport_timings' in source
