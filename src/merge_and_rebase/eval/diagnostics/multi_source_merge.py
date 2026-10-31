"""Fusion of transported experts in the target's weights (docs/.local/AGENT_NEXT_PHASE.md O4, run 2: F0 / F1).

Each member is one task of a finished ``eval.vision_rebase`` run: its saved unit-scale vector times the run's selected
alpha (``alpha_i tau_i``, the expert as transferred). The members are fused by a merge method of the public registry:

    F0  task_arithmetic   sum_i w_i alpha_i tau_i
    F1  tsv_merge         TSV on the same members (``vector_1d_merge: average``, so the biases are kept)

and the fused direction is added to the target base as ``base + lam * direction``. The per-member weights (w_1 = 1,
the others on ``--weights-grid``) and the global ``lam`` (``--lam-grid``) are chosen on the VALIDATION split by the
mean accuracy over the distinct member tasks; the chosen model is then evaluated on the TEST split of every task of
the suite: accuracy per member task, and conservation (change against the target base) on the tasks no member was
made for. Same zero-shot heads, loaders, seed and split settings as the conservation metric (and its cached base
accuracies).

    python -m merge_and_rebase.eval.diagnostics.multi_source_merge --member EuroSAT=RUN_A.json \
        --member DTD=RUN_C.json --merge task_arithmetic --out OUT.json

A member may also be ``TASK=RUN.json@ALPHA`` (that alpha instead of the run's selection) or ``TASK=VECTOR.pt@ALPHA``
(a vector produced elsewhere, e.g. a decode of fused operators, F2 / F3).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.diagnostics.conservation import SPLIT_KEYS, find_vector
from merge_and_rebase.eval.utils import eval_task_top1
from merge_and_rebase.eval.vision_rebase.context import _build_task_context
from merge_and_rebase.eval.vision_rebase.merge import _merge_direction
from merge_and_rebase.eval.vision_rebase.pipeline import _owned_cpu_fp32
from merge_and_rebase.io.ckpt import load_into_model
from merge_and_rebase.merge.methods._common import axpy_state_dict
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

MERGE_PARAMS = {"task_arithmetic": {}, "tsv_merge": {"method_params": {"vector_1d_merge": "average"}}}


def parse_member(spec: str) -> dict[str, Any]:
    """``TASK=PATH[@ALPHA]`` -> {task, path, alpha (None: the run's selected alpha)}."""
    task, rest = spec.split("=", 1)
    path, _, alpha = rest.partition("@")
    return {"task": task, "path": path, "alpha": float(alpha) if alpha else None}


def load_member(member: dict[str, Any]) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    """``(unit-scale vector, alpha, run config or {})`` of one member."""
    path = Path(member["path"])
    if path.suffix == ".pt":
        if member["alpha"] is None:
            raise ValueError(f"{path}: a bare vector needs an explicit @ALPHA")
        return torch.load(path, map_location="cpu", weights_only=True), member["alpha"], {}
    summary = json.loads(path.read_text())
    vector = torch.load(find_vector(path.parent / "transported_tvs", member["task"]), map_location="cpu",
                        weights_only=True)
    alpha = member["alpha"] if member["alpha"] is not None else float(summary["selected_alpha_by_task"][member["task"]])
    return vector, alpha, dict(summary["run_logging"]["metadata"]["resolved_config"])


def frange(spec: str) -> list[float]:
    """``a:b:step`` (inclusive) or a comma list."""
    if ":" in spec:
        a, b, s = (float(x) for x in spec.split(":"))
        n = int(round((b - a) / s))
        return [round(a + i * s, 10) for i in range(n + 1)]
    return [float(x) for x in spec.split(",")]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--merge", choices=sorted(MERGE_PARAMS), default="task_arithmetic")
    ap.add_argument("--weights-grid", default="0.5,0.75,1.0,1.5,2.0")
    ap.add_argument("--lam-grid", default="0.2:2.0:0.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    members = [parse_member(s) for s in args.member]
    loaded = [load_member(m) for m in members]
    cfg = next(c for _, _, c in loaded if c)
    suite = SUITES[str(cfg.get("suite", "vision8"))]
    build = OpenClipBuildConfig(model_name=cfg["target_clip_model"], pretrained=cfg["target_clip_pretrained"],
                                device=args.device, dtype=cfg.get("dtype"))
    clf = OpenClipClassifier.build(build)
    base_sd = _owned_cpu_fp32({k: v for k, v in clf.model.state_dict().items()})
    contexts = {
        t: _build_task_context(t, suite=suite, cfg=cfg, clf_target=clf, clf_source=clf, source_cfg=build,
                               target_cfg=build, use_humanized_classnames=not bool(cfg.get("no_humanize", True)),
                               need_source_loaders=False)
        for t in suite.tasks
    }
    member_tasks = list(dict.fromkeys(m["task"] for m in members))

    def evaluate(sd: dict[str, torch.Tensor], tasks, split: str) -> dict[str, float]:
        load_into_model(clf.model, sd, strict=True)
        return {t: eval_task_top1(clf=clf, loaders=contexts[t].loaders, classnames=contexts[t].classnames,
                                  build_cfg_task=contexts[t].build_cfg_task, device=args.device, split=split)
                for t in tasks}

    deltas = [{k: alpha * v.float() for k, v in vec.items()} for vec, alpha, _ in loaded]
    keys = sorted(set.intersection(*(set(d) for d in deltas)))
    merge_base = {k: base_sd[k] for k in keys}

    def direction(weights) -> dict[str, torch.Tensor]:
        return _merge_direction(base_sd=merge_base, deltas=[{k: d[k] for k in keys} for d in deltas],
                                merge_method_name=args.merge, weights=list(weights),
                                merge_params=MERGE_PARAMS[args.merge])

    grid_w = [(1.0, *rest) for rest in itertools.product(frange(args.weights_grid), repeat=len(deltas) - 1)]
    search = []
    best = None
    for w in grid_w:
        dirn = direction(w)
        for lam in frange(args.lam_grid):
            acc = evaluate(axpy_state_dict(base_sd, dirn, lam), member_tasks, "val")
            score = sum(acc.values()) / len(acc)
            search.append({"weights": list(w), "lam": lam, "val": acc, "score": score})
            if best is None or score > best["score"]:
                best = search[-1]
        print(f"[fusion] weights {w}: best so far {best['weights']} lam {best['lam']} val {100 * best['score']:.2f}",
              flush=True)

    key = {"model": [build.model_name, build.pretrained], **{k: cfg.get(k) for k in SPLIT_KEYS},
           "tasks": list(suite.tasks)}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_cache = out_path.parent / f"conservation_base_{digest}.json"
    if base_cache.exists():
        base = json.loads(base_cache.read_text())["test_accuracy"]
    else:
        base = evaluate(base_sd, suite.tasks, "test")
        base_cache.write_text(json.dumps({"key": key, "test_accuracy": base}, indent=1))
    test = evaluate(axpy_state_dict(base_sd, direction(best["weights"]), best["lam"]), suite.tasks, "test")
    load_into_model(clf.model, base_sd, strict=True)
    others = {t: 100.0 * (test[t] - base[t]) for t in suite.tasks if t not in member_tasks}
    result = {
        "merge": args.merge, "members": [{**m, "alpha_used": a} for m, (_, a, _) in zip(members, loaded, strict=True)],
        "selected": {"weights": best["weights"], "lam": best["lam"], "val_score": best["score"]},
        "test_accuracy": test, "base_test_accuracy": base,
        "member_task_accuracy": {t: test[t] for t in member_tasks},
        "conservation": {"delta_by_task": others, "mean_delta": sum(others.values()) / max(len(others), 1),
                         "max_drop": min(others.values()) if others else None},
        "search": search,
    }
    out_path.write_text(json.dumps(result, indent=1))
    print(f"[fusion] {args.merge}: " + " ".join(f"{t} {100 * test[t]:.2f}" for t in member_tasks)
          + f" | others mean {result['conservation']['mean_delta']:+.2f} max drop {result['conservation']['max_drop']:+.2f}",
          flush=True)
    return result


if __name__ == "__main__":
    main()
