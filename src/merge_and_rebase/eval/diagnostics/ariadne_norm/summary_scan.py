"""N4 (and the summary-level part of N1) from existing Ariadne vision run summaries; CPU only, no model loads.

``python -m merge_and_rebase.eval.diagnostics.ariadne_norm.summary_scan --out <json> <summary.json> ...``

Reads, per task and target position, the solver rows the run already recorded (``direct_residual.diagnostics_by_task``)
and the selected alpha; nothing is recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from merge_and_rebase.run_logging import _code_fingerprint

ROW_FIELDS = (
    "position",
    "source_position",
    "n_rows",
    "ridge",
    "effective_ridge_relative",
    "correction_norm",
    "bias_norm",
    "desired_norm",
    "relative_residual_after",
    "reachable_residual_norm",
    "unreachable_residual_norm",
    "trace_centered_feature_gram",
)


def _rows(summary: dict, task: str) -> list[dict[str, Any]]:
    block = summary.get("direct_residual") or {}
    rows = (block.get("diagnostics_by_task") or {}).get(task) or []
    return [{k: r.get(k) for k in ROW_FIELDS} for r in rows]


def scan_summary(path: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    summary = json.loads(raw)
    meta = (summary.get("run_logging") or {}).get("metadata") or {}
    resolved = meta.get("resolved_config") or {}
    params = resolved.get("direct_residual_params") or resolved.get("ariadne_params") or {}
    alphas = summary.get("selected_alpha_by_task") or {}
    tasks = {}
    for task in sorted((summary.get("direct_residual") or {}).get("diagnostics_by_task") or {}):
        rows = _rows(summary, task)
        w_sq = sum(r["correction_norm"] ** 2 for r in rows)
        b_sq = sum(r["bias_norm"] ** 2 for r in rows)
        for r in rows:
            r["unreachable_fraction"] = (
                r["unreachable_residual_norm"] / r["desired_norm"] if r["desired_norm"] else None
            )
            r["desired_rms_per_row"] = r["desired_norm"] / r["n_rows"] ** 0.5 if r["n_rows"] else None
        tasks[task] = {
            "alpha_star": alphas.get(task),
            "tau_weight_norm": w_sq**0.5,
            "tau_bias_norm": b_sq**0.5,
            "rows": rows,
        }
    per_task = list(tasks.values())

    def med(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "code_fingerprint": meta.get("code_fingerprint"),
        "source": resolved.get("source_clip_model"),
        "target": resolved.get("target_clip_model"),
        "num_batches": params.get("num_batches"),
        "seed": params.get("seed"),
        "ridge_estimator": params.get("ridge_estimator"),
        "tasks": tasks,
        "aggregate": {
            "median_tau_weight_norm": med(t["tau_weight_norm"] for t in per_task),
            "median_tau_bias_norm": med(t["tau_bias_norm"] for t in per_task),
            "median_alpha_star": med(t["alpha_star"] for t in per_task),
            "median_relative_residual": med(r["relative_residual_after"] for t in per_task for r in t["rows"]),
            "median_unreachable_fraction": med(r["unreachable_fraction"] for t in per_task for r in t["rows"]),
            "effective_ridge_relative": med(r["effective_ridge_relative"] for t in per_task for r in t["rows"]),
            "n_rows": med(r["n_rows"] for t in per_task for r in t["rows"]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("summaries", nargs="+")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; results are write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"code_fingerprint": _code_fingerprint(), "runs": [scan_summary(p) for p in args.summaries]}
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
