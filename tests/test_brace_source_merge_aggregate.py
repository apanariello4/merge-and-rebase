from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/aggregate_brace_source_merge.py"
TASKS = ["Cars", "EuroSAT"]
ARMS = ["shared", "independent", "independent_shared_norm", "shared_independent_norm", "midpoint_raw", "midpoint_shared_norm"]
RECIPIENTS = ["consensus", "EuroSAT"]


def _module():
    spec = importlib.util.spec_from_file_location("source_merge_aggregate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(arm: str, recipient: str, *, positive: bool = True, base_hash: str = "base") -> dict:
    merged_value = 0.7 if (positive and arm == "shared") or (not positive and arm != "shared") else 0.6
    test = {}
    for task in TASKS:
        test[task] = {}
    for task in TASKS:
        test[task] = {
            "selected": {"merged": merged_value, "isolated": 0.55, "merged_minus_isolated": merged_value - 0.55},
            "fixed_alpha1": {"merged": merged_value - 0.02, "isolated": 0.54, "merged_minus_isolated": merged_value - 0.56},
        }
    return {
        "status": "success", "arm": arm, "recipient_base": recipient,
        "tasks": TASKS, "transport_method": "none", "merge": "task_arithmetic",
        "alpha_values": [0.0, 1.0], "recipient_base_sha256": base_hash, "test": {arm: test},
    }


def _manifest(tmp_path: Path, *, positive: bool = True, base_hashes: dict[str, str] | None = None):
    base_hashes = base_hashes or {r: "base" for r in RECIPIENTS}
    runs = []
    for arm in ARMS:
        for recipient in RECIPIENTS:
            out = tmp_path / f"{arm}_{recipient}"
            out.mkdir()
            (out / "summary.json").write_text(json.dumps(_summary(arm, recipient, positive=positive, base_hash=base_hashes[recipient])))
            config = out / "config.json"
            config.write_text(json.dumps({"arm": arm, "recipient_base": recipient}))
            runs.append({"arm": arm, "recipient_base": recipient, "config_path": str(config), "output_dir": str(out)})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tasks": TASKS, "merge_runs": runs}))
    return path


def test_aggregate_emits_metrics_and_positive_shared_contrast(tmp_path):
    module = _module()
    output = tmp_path / "aggregate"
    metadata = module.aggregate(_manifest(tmp_path), output)
    assert metadata["completed"] == 12
    assert (output / "source_merge_per_task.csv").is_file()
    assert (output / "source_merge_means.csv").is_file()
    assert (output / "source_merge_contrasts.csv").is_file()
    report = (output / "report.md").read_text()
    assert "supports Shared" in report
    assert "merged-minus-isolated" in report


def test_negative_shared_contrast_is_reported_without_presumption(tmp_path):
    module = _module()
    output = tmp_path / "aggregate"
    module.aggregate(_manifest(tmp_path, positive=False), output)
    assert "refutes Shared" in (output / "report.md").read_text()


def test_rejects_mismatched_recipient_base_hash(tmp_path):
    module = _module()
    manifest = _manifest(tmp_path)
    summary_path = tmp_path / "shared_consensus" / "summary.json"
    payload = json.loads(summary_path.read_text())
    payload["recipient_base_sha256"] = "different"
    summary_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="common"):
        module.collect(manifest)


def test_rejects_transport_or_merge_method(tmp_path):
    module = _module()
    manifest = _manifest(tmp_path)
    summary_path = tmp_path / "shared_consensus" / "summary.json"
    payload = json.loads(summary_path.read_text())
    payload["transport_method"] = "theseus"
    summary_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Transport method"):
        module.collect(manifest)

    payload["transport_method"] = "none"
    payload["merge"] = "tsv"
    summary_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task_arithmetic"):
        module.collect(manifest)
