"""Per-cell aggregation of ``ariadne_norm`` task records (CPU only).

``python -m merge_and_rebase.eval.diagnostics.ariadne_norm.report --out <json> <run_dir> ...``

For every statistic: per task, the median over target blocks (``all``) or over the deeper half of the blocks
(``deep``, ``j >= depth // 2``); per cell, the median and the [min, max] range of the per-task values. Nothing is
recomputed from models; every value is read from ``<run_dir>/<task>/record.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from merge_and_rebase.run_logging import _code_fingerprint


def _median(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _nearest(grid: dict[str, Any], alpha: float) -> str:
    return min(grid, key=lambda k: abs(float(k) - alpha))


def task_statistics(record: dict[str, Any]) -> dict[str, float | None]:
    joint = record["n2"]["joint_by_alpha"]
    blocks = sorted(joint["1.0"], key=int)
    deep = blocks[len(blocks) // 2 :]
    single = record["n2"]["single_block"]
    alpha_star = record["n2"]["alpha_star"]
    at_star = joint[_nearest(joint, alpha_star)] if alpha_star is not None else None
    n3, n7 = record["n3"], record["n7"]

    def over(js, fn: Callable[[str], float | None]) -> float | None:
        return _median(fn(j) for j in js)

    def rho(j: str, value: float) -> dict[str, Any]:
        return next(p for p in n3[j]["l_curve"] if p["rho"] == value)

    def n7_rho(j: str, value: float) -> dict[str, Any]:
        return next(p for p in n7[j]["regression"]["by_rho"] if p["rho"] == value)

    n5 = record.get("n5", {})
    superposition = record["n2"].get("superposition", {})
    n1 = record["n1"]
    gen = record["n6"].get("1.0") or record["n6"].get(1.0)
    return {
        "n0_passed": float(bool((record["n0"]["task_vector"]["reference"] or {}).get("passed"))),
        "alpha_star": alpha_star,
        "alpha_hat": record["n2"]["label_free_alpha"]["alpha_hat"],
        "tau_target_norm": n1["tau_target_norm"],
        "tau_ratio_to_source_full": n1["tau_target_norm"] / max(n1["tau_source_full_norm"], 1e-300),
        "tau_ratio_to_source_component": n1["tau_target_norm"] / max(n1["tau_source_component_norm"], 1e-300),
        "bias_share_of_tau_sq": (n1["tau_target_bias_norm"] / n1["tau_target_norm"]) ** 2,
        "n2_own_r_deep": over(deep, lambda j: single[j]["own_r"]),
        "n2_own_cos_deep": over(deep, lambda j: single[j]["own_cos"]),
        "n2_joint_r_alpha1_deep": over(deep, lambda j: joint["1.0"][j]["r"]),
        "n2_joint_cos_alpha1_deep": over(deep, lambda j: joint["1.0"][j]["cos"]),
        "n2_joint_projection_alpha1_deep": over(deep, lambda j: joint["1.0"][j]["projection_coefficient"]),
        "n2_error_at_alpha_star_deep": over(deep, lambda j: at_star[j]["relative_error"]) if at_star else None,
        "n2_superposition_ratio_deep": over(
            deep,
            lambda j: (
                superposition[j]["sum_single_projection"] / superposition[j]["joint_projection"]
                if superposition.get(j) and superposition[j]["joint_projection"]
                else None
            ),
        ),
        "n3_lam_over_mu_median": over(blocks, lambda j: n3[j]["lam_over_mu_median"]),
        "n3_dof_fraction": over(blocks, lambda j: n3[j]["at_official"]["effective_dof_fraction"]),
        "n3_tail1pct_norm_share": over(blocks, lambda j: n3[j]["tail_shares"]["0.01"]["correction_norm_sq_share"]),
        "n3_tail1pct_fit_share": over(blocks, lambda j: n3[j]["tail_shares"]["0.01"]["fitted_output_sq_share"]),
        "n3_norm_ratio_rho10": over(
            blocks, lambda j: rho(j, 10.0)["correction_norm"] / max(n3[j]["at_official"]["correction_norm"], 1e-300)
        ),
        "n3_relres_increase_rho10": over(
            blocks, lambda j: rho(j, 10.0)["relative_residual_after"] - n3[j]["at_official"]["relative_residual_after"]
        ),
        "n3_unreachable_fraction": over(
            blocks, lambda j: next(p for p in n1["per_position"] if str(p["position"]) == j)["unreachable_fraction"]
        ),
        "n5_first_token_target_share": _median(v["first_token"]["target_energy_share"] for v in n5.values()),
        "n5_top1pct_energy_norm_ratio": _median(
            next(f for f in v["refits"] if f["criterion"] == "target_energy" and f["quantile"] == 0.01)[
                "correction_norm_ratio"
            ]
            for v in n5.values()
        ),
        "n6_generic_over_calibration_alpha1": over(
            blocks,
            lambda j: (
                gen["generic"][j]["relative_perturbation"] / max(gen["calibration"][j]["relative_perturbation"], 1e-300)
            ),
        )
        if gen
        else None,
        "n7_param_rel_diff": over(blocks, lambda j: n7[j]["regression"]["official"]["param_rel_diff"]),
        "n7_param_cosine": over(blocks, lambda j: n7[j]["regression"]["official"]["param_cosine"]),
        "n7_functional_rel_diff_deep": over(deep, lambda j: n7[j]["regression"]["official"]["functional_rel_diff"]),
        "n7_param_rel_diff_rho10": over(blocks, lambda j: n7_rho(j, 10.0)["param_rel_diff"]),
        "n7_q_rel_diff": over(blocks, lambda j: n7[j]["procrustes"]["q_rel_diff"]),
        "n7_desired_rel_diff": over(blocks, lambda j: n7[j]["procrustes"]["desired_rel_diff"]),
    }


def cell_report(run_dir: str) -> dict[str, Any]:
    root = Path(run_dir)
    tasks = {}
    hashes = {}
    for path in sorted(root.glob("*/record.json")):
        raw = path.read_bytes()
        hashes[path.parent.name] = hashlib.sha256(raw).hexdigest()
        tasks[path.parent.name] = task_statistics(json.loads(raw))
    keys = sorted({k for stats in tasks.values() for k in stats})
    summary = {}
    for key in keys:
        vals = [s[key] for s in tasks.values() if s.get(key) is not None]
        summary[key] = {
            "median": statistics.median(vals) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n_tasks": len(vals),
        }
    return {"run_dir": str(root.resolve()), "record_sha256": hashes, "tasks": tasks, "summary": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; results are write-once")
    payload = {"code_fingerprint": _code_fingerprint(), "cells": {Path(d).name: cell_report(d) for d in args.run_dirs}}
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
