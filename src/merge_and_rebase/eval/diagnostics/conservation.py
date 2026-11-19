"""Conservation metric (docs/.local/AGENT_NEXT_PHASE.md section 2): the damage a transported update does to the tasks
it was NOT made for.

For every task t of a finished ``eval.vision_rebase`` run: the target base plus ``alpha_t * tau_t`` (the run's saved
unit-scale vector and its selected alpha) is evaluated on the TEST split of every Vision8 task, with the same zero-shot
heads, loaders, ``seed``, ``val_fraction`` and ``val_size`` as the run (rebuilt from the run's resolved config). The
transferred task's own accuracy is recomputed too and checked against the run's summary (a reproduction check).
Reported per t: accuracy minus the target base on each other task, their mean and the largest drop.

    python -m merge_and_rebase.eval.diagnostics.conservation RUN_SUMMARY.json [--tv-dir DIR] [--out OUT.json]
        [--alpha-key selected_alpha_by_task|per_task_premerge_alphas] [--alpha 1.0] [--device cuda]

``--alpha A`` replaces every task's selected alpha by the fixed ``A`` (e.g. 1.0: the update as compiled, what a user
without a validation set gets); the default output is then ``conservation_alpha<A>.json`` (the selected-alpha file is
not overwritten). The transferred task's accuracy at that alpha is checked against the run's ``alpha_curve_test``
(``test_curve`` runs) at that grid point (NaN in the log when the grid has no such point).

``--tv-dir`` defaults to ``<run dir>/transported_tvs`` (any ``{task}_*_transported_native.pt``). The target base
accuracies are cached next to the output (``conservation_base_<hash>.json``), keyed by the target model and split
settings, so several runs share them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.utils import eval_task_top1
from merge_and_rebase.eval.vision_rebase.context import _build_task_context
from merge_and_rebase.eval.vision_rebase.pipeline import _owned_cpu_fp32
from merge_and_rebase.io.ckpt import load_into_model
from merge_and_rebase.merge.methods._common import axpy_state_dict
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

SPLIT_KEYS = ("seed", "val_fraction", "val_size", "batch_size", "num_workers", "no_humanize")


def find_vector(tv_dir: Path, task: str) -> Path:
    found = sorted(tv_dir.glob(f"{task}_*_transported_native.pt"))
    if len(found) != 1:
        raise FileNotFoundError(f"{tv_dir}: expected one {task}_*_transported_native.pt, found {len(found)}")
    return found[0]


def accuracy_delta(acc: dict[str, float], base: dict[str, float], task: str) -> dict[str, Any]:
    """Points (x100) of change on every task except ``task``: per task, mean, and the largest drop (min delta)."""
    others = {t: 100.0 * (acc[t] - base[t]) for t in acc if t != task}
    return {"delta_by_task": others, "mean_delta": sum(others.values()) / len(others), "max_drop": min(others.values())}


def curve_accuracy(summary: dict[str, Any], task: str, alpha: float, tol: float = 1e-6) -> float | None:
    """Own-task test accuracy of the run's ``alpha_curve_test`` at ``alpha`` (``per_task_rebased``), None if the run has
    no such curve or grid point."""
    for point in summary.get("alpha_curve_test") or []:
        if abs(float(point["alpha"]) - alpha) <= tol and task in point.get("per_task_rebased", {}):
            return float(point["per_task_rebased"][task])
    return None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary")
    ap.add_argument("--tv-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha-key", default="selected_alpha_by_task")
    ap.add_argument("--alpha", type=float, default=None, help="fixed alpha for every task instead of the selected one")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text())
    cfg = dict(summary["run_logging"]["metadata"]["resolved_config"])
    tv_dir = Path(args.tv_dir) if args.tv_dir else summary_path.parent / "transported_tvs"
    name = "conservation.json" if args.alpha is None else f"conservation_alpha{args.alpha:g}.json"
    out_path = Path(args.out) if args.out else summary_path.parent / name
    alphas = {t: float(a) for t, a in summary[args.alpha_key].items()}
    if args.alpha is not None:
        alphas = {t: float(args.alpha) for t in alphas}
    suite = SUITES[str(cfg.get("suite", "vision8"))]
    tasks_all = list(suite.tasks)

    build = OpenClipBuildConfig(
        model_name=cfg["target_clip_model"], pretrained=cfg["target_clip_pretrained"], device=args.device,
        dtype=cfg.get("dtype"),
    )
    clf = OpenClipClassifier.build(build)
    base_sd = _owned_cpu_fp32({k: v for k, v in clf.model.state_dict().items()})
    contexts = {
        t: _build_task_context(
            t, suite=suite, cfg=cfg, clf_target=clf, clf_source=clf, source_cfg=build, target_cfg=build,
            use_humanized_classnames=not bool(cfg.get("no_humanize", True)), need_source_loaders=False,
        )
        for t in tasks_all
    }

    def evaluate() -> dict[str, float]:
        return {
            t: eval_task_top1(clf=clf, loaders=c.loaders, classnames=c.classnames, build_cfg_task=c.build_cfg_task,
                              device=args.device, split="test")
            for t, c in contexts.items()
        }

    key = {"model": [build.model_name, build.pretrained], **{k: cfg.get(k) for k in SPLIT_KEYS}, "tasks": tasks_all}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    base_cache = out_path.parent / f"conservation_base_{digest}.json"
    if base_cache.exists():
        base = json.loads(base_cache.read_text())["test_accuracy"]
    else:
        load_into_model(clf.model, base_sd, strict=True)
        base = evaluate()
        base_cache.write_text(json.dumps({"key": key, "test_accuracy": base}, indent=1))

    result: dict[str, Any] = {"summary": str(summary_path), "tv_dir": str(tv_dir), "alpha_key": args.alpha_key,
                              "alpha_fixed": args.alpha, "base_test_accuracy": base, "by_task": {}}
    run_test = summary.get("test_results", {}).get("per_task_absolute_accuracy", {})
    for task in summary["tasks"]:
        path = find_vector(tv_dir, task)
        tv = torch.load(path, map_location="cpu", weights_only=True)
        sd = axpy_state_dict(base_sd, tv, alphas[task])
        load_into_model(clf.model, sd, strict=True)
        acc = evaluate()
        reference = run_test.get(task) if args.alpha is None else curve_accuracy(summary, task, alphas[task])
        entry = {"vector": str(path), "alpha": alphas[task], "test_accuracy": acc,
                 "own_task_accuracy": acc[task], "own_task_accuracy_run": reference,
                 **accuracy_delta(acc, base, task)}
        result["by_task"][task] = entry
        print(f"[conservation] {task}: alpha {alphas[task]:.3f} own {100 * acc[task]:.2f} "
              f"(run {100 * (float('nan') if reference is None else reference):.2f}) | others mean {entry['mean_delta']:+.2f} "
              f"max drop {entry['max_drop']:+.2f}", flush=True)
    load_into_model(clf.model, base_sd, strict=True)
    out_path.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
