"""Per-block table of a finished anchors run (PIANO_METODO_K, section 2: "per blocco").

Post hoc and CPU only. For every task and target block ``j`` (index into
``direct_residual.diagnostics_by_task[task]`` of the run JSON, written by ``anchors/method.py``):

- ``e_norm``, ``d_norm`` (``anchors_e_norm``, ``anchors_d_norm``), ``c_source``, ``c_target``
  (``anchors_c_source``, ``anchors_c_target``);
- ``ratio_raw = ||D|| / ||E||``;
- ``ratio_c = (||D|| / c_k) / (||E|| / c_m) = ratio_raw * c_m / c_k``: both norms expressed in the scale (``c``) of
  their own model that the decoder used (``anchors_c_source_used`` / ``anchors_c_target_used``: the filtered ``c``
  under ``anchor_filter``; runs written before these keys existed fall back to the registration ``c``, and the
  columns ``c_source_used`` / ``c_target_used`` give the ``c`` used), so the ratio is comparable across model pairs. ``ratio_raw`` and the ``c`` are
  kept in the table, so the other convention is one division away;
- ``drift_norm_ratio``, ``drift_cosine`` (``||realized|| / ||D_j||`` and cosine, optional): read from a
  ``depth_drift`` output JSON (that diagnostic needs the models and a GPU, so it is run separately; use
  :func:`drift_config` to build its config at the run's selected alphas) at the alpha closest to the run's
  selected alpha of the task;
- ``update_cos_ref`` (optional): cosine between the run's transported update of ``mlp.c_proj`` of block ``j`` and the
  reference run's (Ariadne), reusing ``anchors_precision._cos`` / ``_tvs``. Task vectors are compared unscaled.

Run::

    python -m merge_and_rebase.eval.diagnostics.metodoK_blocks --run-json RUN.json --out TABLE.json \\
        [--run-dir RUN_DIR] [--reference ARIADNE_RUN_DIR] [--drift DRIFT.json]

``--run-dir`` defaults to the directory of ``--run-json``. Writes ``TABLE.json`` and ``TABLE.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .anchors_precision import _cos, _tvs

_COLUMNS = (
    "task", "block", "source_block", "e_norm", "d_norm", "c_source", "c_target", "c_source_used", "c_target_used", "ratio_raw", "ratio_c",
    "drift_norm_ratio", "drift_cosine", "update_cos_ref",
)


def _nearest(per_alpha: Mapping[str, Any], alpha: float) -> tuple[str, Any]:
    key = min(per_alpha, key=lambda a: abs(float(a) - alpha))
    return key, per_alpha[key]


def _tasks(run: Mapping[str, Any]) -> list[str]:
    return run["tasks"].split(",") if isinstance(run["tasks"], str) else list(run["tasks"])


def _selected(run: Mapping[str, Any]) -> dict[str, float]:
    """``selected_alpha_by_task``: a {task: alpha} dict in run JSONs (a list in task order is also accepted)."""
    sel = run.get("selected_alpha_by_task") or {}
    return {str(t): float(a) for t, a in (sel.items() if isinstance(sel, Mapping) else zip(_tasks(run), sel, strict=True))}


def block_rows(
    run: Mapping[str, Any],
    *,
    run_tvs: Mapping[str, Mapping[str, Any]] | None = None,
    ref_tvs: Mapping[str, Mapping[str, Any]] | None = None,
    drift: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per (task, target block). Missing optional inputs give ``None`` columns."""
    diag = run["direct_residual"]["diagnostics_by_task"]
    selected = _selected(run)
    rows: list[dict[str, Any]] = []
    for task, blocks in diag.items():
        drift_task = (drift or {}).get("results", {}).get(task)
        drift_alpha = None
        if drift_task is not None and task in selected:
            drift_alpha, drift_blocks = _nearest(drift_task["per_alpha"], float(selected[task]))
        for j, d in enumerate(blocks):
            e, dn = float(d["anchors_e_norm"]), float(d["anchors_d_norm"])
            c_m, c_k = float(d["anchors_c_source"]), float(d["anchors_c_target"])
            used = "anchors_c_source_used" in d and "anchors_c_target_used" in d
            cu_m, cu_k = (float(d["anchors_c_source_used"]), float(d["anchors_c_target_used"])) if used else (c_m, c_k)
            row: dict[str, Any] = {
                "task": task, "block": j, "source_block": d.get("anchors_source_block"),
                "e_norm": e, "d_norm": dn, "c_source": c_m, "c_target": c_k,
                "ratio_raw": dn / e if e > 0 else float("nan"),
                "ratio_c": (dn / cu_k) / (e / cu_m) if e > 0 and cu_k > 0 and cu_m > 0 else float("nan"),
                "c_source_used": cu_m, "c_target_used": cu_k,
                "c_used_fallback": not used,
                "drift_norm_ratio": None, "drift_cosine": None, "update_cos_ref": None,
            }
            if drift_alpha is not None:
                row["drift_alpha"] = float(drift_alpha)
                row["drift_norm_ratio"] = drift_blocks[str(j)]["norm_ratio"]
                row["drift_cosine"] = drift_blocks[str(j)]["cosine"]
            key = f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"
            if run_tvs and ref_tvs and task in run_tvs and task in ref_tvs and key in run_tvs[task]:
                row["update_cos_ref"] = _cos(run_tvs[task][key], ref_tvs[task][key])
            rows.append(row)
    return rows


def drift_config(run_json: Mapping[str, Any], run_config: str, tv_dir: str, out: str, *, num_batches: int = 4,
                 seed: int = 0) -> dict[str, Any]:
    """Config for ``depth_drift`` at the selected alpha of each task of the run (run it on a GPU)."""
    sel = _selected(run_json)
    tasks = _tasks(run_json)
    return {"run_config": run_config, "tv_dir": tv_dir, "tasks": tasks, "num_batches": num_batches, "seed": seed,
            "alphas": {t: [sel[t]] for t in tasks}, "out": out}


def write_table(rows: list[dict[str, Any]], out: Path, meta: Mapping[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": dict(meta), "rows": rows}, indent=1))
    with out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_COLUMNS), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--reference")
    ap.add_argument("--drift")
    a = ap.parse_args(argv)
    run = json.loads(Path(a.run_json).read_text())
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.run_json).parent
    tasks = _tasks(run)
    run_tvs = ref_tvs = None
    if a.reference:
        run_tvs, ref_tvs = _tvs(run_dir, tasks), _tvs(Path(a.reference), tasks)
    drift = json.loads(Path(a.drift).read_text()) if a.drift else None
    rows = block_rows(run, run_tvs=run_tvs, ref_tvs=ref_tvs, drift=drift)
    write_table(rows, Path(a.out), {"run_json": a.run_json, "reference": a.reference, "drift": a.drift})
    print(f"[metodoK_blocks] {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
