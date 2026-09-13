from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_brace_tv_swap.py"


def _module():
    spec = importlib.util.spec_from_file_location("aggregate_brace_tv_swap", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(tasks: list[str]):
    rows = []
    for c_idx, condition in enumerate(("independent", "shared", "skip")):
        for d_idx, donor in enumerate(tasks):
            for r_idx, recipient in enumerate([*tasks, "consensus"]):
                kind = "consensus" if recipient == "consensus" else ("self" if recipient == donor else "cross_task")
                rows.append({
                    "condition": condition,
                    "donor_task": donor,
                    "recipient_base": recipient,
                    "recipient_kind": kind,
                    "test_selected_accuracy": 0.1 * c_idx + 0.01 * d_idx + 0.001 * r_idx,
                    "selected_alpha": float(c_idx),
                    "test_drop_from_donor_self": float(r_idx),
                })
    return rows


def test_aggregate_requires_exact_full_matrix_and_excludes_diagonal_from_cross_mean() -> None:
    module = _module()
    tasks = ["EuroSAT", "GTSRB"]
    rows = _rows(tasks)
    module.validate_rows(rows, tasks)
    summary = module.summarize(rows, tasks)
    assert abs(summary["condition_means"]["shared"]["cross_task"] - 0.1055) < 1e-12
    assert abs(summary["condition_means"]["shared"]["consensus"] - 0.107) < 1e-12
    assert len(module.contrasts(rows)) == 3 * len(tasks) * (len(tasks) + 1)
