from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/aggregate_brace_merge_order_lambda100_20260906.py"
TASKS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]


def _load():
    spec = importlib.util.spec_from_file_location("merge_order_aggregate", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(*, status="success", baseline_label="target_zeroshot", bad_ratio=False, diagnostics=None, hashes=True):
    baseline = {task: 0.5 for task in TASKS}
    absolute = {task: 0.6 for task in TASKS}
    ratios = {task: (1.0 if bad_ratio and task == "Cars" else 1.2) for task in TASKS}
    payload = {
        "baseline_label": baseline_label,
        "tasks": TASKS,
        "selected_baseline_alpha_by_task": {task: 0.0 for task in TASKS},
        "run_logging": {"status": status},
        "test_results": {
            "per_task_baseline_accuracy": baseline,
            "per_task_absolute_accuracy": absolute,
            "per_task_normalized_accuracy_ratio": ratios,
        },
    }
    if diagnostics is not None:
        payload["strict_diagnostics"] = diagnostics
    if hashes:
        payload["target_state_hash"] = {"before": "same", "after": "same"}
    return payload


def _row(path: Path, *, index=0, calibration="tiny_5", **axes):
    defaults = {
        "run_id": f"run_{index}",
        "pipeline": "brace_transport_then_merge",
        "direction": "extend",
        "correction": "shared",
        "calibration": calibration,
        "transport": "theseus",
        "merger": "task_arithmetic",
        "alpha_policy": "shared",
        "summary_path": str(path),
    }
    defaults.update(axes)
    return defaults


def test_validate_summary_enforces_the_requested_contract():
    module = _load()
    metrics = module.validate_summary(_summary())
    assert metrics["test_mean"] == pytest.approx(0.6)
    assert metrics["ratio_mean"] == pytest.approx(1.2)
    assert metrics["diagnostic_missing"] == 0

    with pytest.raises(ValueError, match="successful"):
        module.validate_summary(_summary(status="failed"))
    with pytest.raises(ValueError, match="target_zeroshot"):
        module.validate_summary(_summary(baseline_label="untransported"))
    with pytest.raises(ValueError, match="ratio mismatch"):
        module.validate_summary(_summary(bad_ratio=True))
    with pytest.raises(ValueError, match="diagnostics"):
        module.validate_summary(_summary(diagnostics={"missing": 1}))


def test_target_hash_is_checked_when_present():
    module = _load()
    summary = _summary()
    summary["target_state_hash"]["after"] = "changed"
    with pytest.raises(ValueError, match="target hash changed"):
        module.validate_summary(summary)


def test_collect_requires_all_summaries_unless_incomplete(tmp_path):
    module = _load()
    existing = tmp_path / "existing.json"
    existing.write_text(json.dumps(_summary()), encoding="utf-8")
    manifest = {"campaign": "synthetic", "logical_rows": [_row(existing), _row(tmp_path / "missing.json", index=1, transport="bico")]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing 1"):
        module.collect(manifest_path)
    _manifest, loaded, missing = module.collect(manifest_path, allow_incomplete=True)
    assert len(loaded) == 1
    assert len(missing) == 1


def test_aggregate_emits_config_task_and_all_paired_contrasts(tmp_path):
    module = _load()
    rows = []
    variants = [
        ("brace_transport_then_merge", "shared", "tiny_5", "theseus", "task_arithmetic", "shared"),
        ("brace_merge_then_transport", "independent", "vision8_mix_40", "bico", "tsv", "shared"),
    ]
    for index, (pipeline, correction, calibration, transport, merger, alpha_policy) in enumerate(variants):
        summary_path = tmp_path / f"summary_{index}.json"
        summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
        rows.append(
            _row(
                summary_path,
                index=index,
                pipeline=pipeline,
                correction=correction,
                calibration=calibration,
                transport=transport,
                merger=merger,
                alpha_policy=alpha_policy,
            )
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"campaign": "synthetic", "logical_rows": rows}), encoding="utf-8")
    output = tmp_path / "aggregate"
    metadata = module.aggregate(manifest_path, output)

    assert metadata["completed"] == 2
    assert len((output / "config_summary.csv").read_text(encoding="utf-8").splitlines()) == 3
    assert len((output / "task_results.csv").read_text(encoding="utf-8").splitlines()) == 17
    for factor in module.CONTRAST_AXES:
        assert (output / f"paired_{factor}_contrasts.csv").is_file()
    assert (output / "aggregate_metadata.json").is_file()


def test_calibration_source_and_budget_are_derived_and_pairable(tmp_path):
    module = _load()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_summary()), encoding="utf-8")
    second.write_text(json.dumps(_summary()), encoding="utf-8")
    rows = [
        _row(first, index=0, calibration="tiny_5"),
        _row(second, index=1, calibration="tiny_40"),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"logical_rows": rows}), encoding="utf-8")
    _manifest, loaded, _missing = module.collect(manifest)
    assert {row["calibration_source"] for row, _summary, _metrics in loaded} == {"tiny"}
    assert {row["calibration_budget"] for row, _summary, _metrics in loaded} == {"5", "40"}

    output = tmp_path / "out"
    module.aggregate(manifest, output)
    contrast = (output / "paired_calibration_budget_contrasts.csv").read_text(encoding="utf-8")
    assert "calibration_budget,5,40" in contrast
