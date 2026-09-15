"""Score a transport run against what the task vector does in its own basis.

A transport run is usually read off `best_alpha` / `avg_acc`, which averages the
per-task accuracies into one number. For the Qwen2-instruct vector that average
is the wrong instrument: the vector *helps* ARC-Easy and *hurts* GSM8K and
TruthfulQA by comparable amounts, so the mean cancels to roughly zero no matter
how good the transport is.

What a transport should reproduce is the per-task *signature* -- the signed
pattern of gains and losses -- so this script reports:

  oracle[t]      delta the task vector produces on the source model itself
                 (Qwen2-1.5B -> Qwen2-1.5B-Instruct), per task
  observed[t]    delta the transported vector produces on the target model
                 (Qwen2.5-3B base -> rebased), per task, at each alpha
  cosine         alignment between the observed and oracle delta vectors;
                 1.0 = same signature, 0 = unrelated, -1 = inverted

Usage:
  python scripts/task_vector_oracle.py                       # oracle table only
  python scripts/task_vector_oracle.py results/.../run.json  # + transport runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path("results/eval")
SOURCE_BASE = "qwen2_1.5b_base"
SOURCE_TUNED = "qwen2_1.5b_instruct"


def _task_metrics(model: str, task: str) -> dict[str, float]:
    path = EVAL_DIR / f"{model}_{task}.json"
    if not path.exists():
        return {}
    harness = json.loads(path.read_text()).get("harness_results") or {}
    return {k: v for k, v in harness.items() if isinstance(v, (int, float))}


def _task_score(metrics: dict[str, float], task: str) -> float | None:
    vals = [v for k, v in metrics.items() if k.startswith(f"{task}_")]
    return sum(vals) / len(vals) if vals else None


def oracle_deltas(tasks: list[str]) -> dict[str, float]:
    """Per-task delta of the task vector applied natively to its own source."""
    out: dict[str, float] = {}
    for task in tasks:
        base = _task_score(_task_metrics(SOURCE_BASE, task), task)
        tuned = _task_score(_task_metrics(SOURCE_TUNED, task), task)
        if base is not None and tuned is not None:
            out[task] = tuned - base
    return out


def _cosine(a: list[float], b: list[float]) -> float | None:
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def report_run(path: Path, oracle: dict[str, float]) -> None:
    summary = json.loads(path.read_text())
    logging_block = summary.get("run_logging", {})
    if logging_block.get("status") != "success":
        err = (logging_block.get("error") or {}).get("message", "unknown")
        print(f"\n### {path.name}: run FAILED ({err})")
        return

    before = summary.get("harness_results_before_rebase") or {}
    tasks = list(oracle)
    print(f"\n### {path.name}   method={summary.get('method')}")

    stats = summary.get("merged_delta")
    if stats:
        print(
            f"  merged delta: nonzero_keys={int(stats['nonzero_key_count'])}"
            f"/{int(stats['key_count'])} rel_norm={stats['merged_delta_rel_norm']:.6f}"
        )

    base_scores = {t: _task_score(before, t) for t in tasks}
    header = "  " + "alpha".ljust(8) + "".join(t[:14].ljust(16) for t in tasks) + "cosine"
    print(header)
    print("  " + "oracle".ljust(8) + "".join(f"{oracle[t] * 100:+.2f}".ljust(16) for t in tasks) + "1.000")

    by_alpha = summary.get("harness_results_by_alpha") or {}
    for entry in summary.get("search_results", []):
        results = by_alpha.get(f"{entry['alpha']:g}")
        if results is None:
            # Pre-fix summaries only keep an unnamed per_task_acc list.
            continue
        observed, cells = [], ""
        for task in tasks:
            got, ref = _task_score(results, task), base_scores.get(task)
            if got is None or ref is None:
                cells += "n/a".ljust(16)
                observed.append(0.0)
                continue
            observed.append(got - ref)
            cells += f"{(got - ref) * 100:+.2f}".ljust(16)
        cos = _cosine(observed, [oracle[t] for t in tasks])
        cos_txt = f"{cos:+.3f}" if cos is not None else "n/a"
        print("  " + f"{entry['alpha']:g}".ljust(8) + cells + cos_txt)


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]]
    tasks = ["arc_easy", "gsm8k", "truthfulqa_mc2"]
    oracle = oracle_deltas(tasks)

    print("=== Source-side oracle (Qwen2-1.5B -> Qwen2-1.5B-Instruct) ===")
    for task, delta in oracle.items():
        print(f"  {task:16s} {delta * 100:+6.2f}")
    mean = sum(oracle.values()) / len(oracle) * 100
    print(f"  {'mean':16s} {mean:+6.2f}   <- why the averaged score is uninformative")

    for run in runs:
        report_run(run, oracle)


if __name__ == "__main__":
    main()
