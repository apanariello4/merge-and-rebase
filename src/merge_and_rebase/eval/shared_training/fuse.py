"""LIN-3: fusion of per-task shared adaptations in the common space (AGENT_SHARED_TRAINING.md section 3).

    python -m merge_and_rebase.eval.shared_training.fuse --out OUT.json --method sum|tsv THETA_TASK1.pt THETA_TASK2.pt

``sum``: ``W = alpha sum_t W_t``, ``b = alpha sum_t b_t`` (task arithmetic). ``tsv``: per grid point the TSV merge of the
``W_t`` (``merge.methods.functional.tsv_merge_impl``), biases summed; times ``alpha``. ``alpha`` in ``--alphas`` chosen
by the mean VALIDATION accuracy over the member tasks on the members' students (adapter, true model); then TEST
accuracy on every member task for each eval model, adapter and compiled (calibration: ``--calib-batches`` per task,
pooled). Compare with F0 (weight-space fusion of transported vectors, anchors_ops_20261030/fusion) and M0.
"""
import argparse
import json
from pathlib import Path

import torch

from merge_and_rebase.merge.methods.functional.tsv_merge import tsv_merge_impl
from merge_and_rebase.rebase.methods.anchors.config import AnchorsConfig
from merge_and_rebase.rebase.methods.anchors.shared_space import MAIN_FRAME, SharedFrame
from merge_and_rebase.rebase.methods.anchors.state import AnchorState

from . import evaluate as ev
from .data import anchor_images, build_classifier, model_id, paired_images, split_loader, task_data
from .lin import payload_to_theta, theta_to_payload


def fuse(thetas, method, alpha):
    grid = sorted(thetas[0])
    out = {}
    for j in grid:
        ws = [t[j][0] for t in thetas]
        w = sum(ws) if method == "sum" else tsv_merge_impl([x.float() for x in ws], torch.ones(len(ws)), {}).double()
        out[j] = (alpha * w, alpha * sum(t[j][1] for t in thetas))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("thetas", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="sum", choices=["sum", "tsv"])
    ap.add_argument("--alphas", default="0.5,0.75,1.0")
    ap.add_argument("--eval-models", default="A,B,C")
    ap.add_argument("--calib-batches", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device
    pays = [torch.load(p, map_location="cpu") for p in a.thetas]
    tasks = [p["config"]["task"] for p in pays]
    students = [model_id(m) for m in pays[0]["config"]["students"]]
    evals = [model_id(m) for m in a.eval_models.split(",")]
    c0 = pays[0]["config"]
    fit = [model_id(m) for m in c0["fit_models"]]
    frame = SharedFrame.from_state(c0["state_dir"], fit_models=fit, add_models=[m for m in evals if m not in fit],
                                   n_grid=c0["n_grid"], depths=c0["depths"], rank=c0["rank"], filter_k=c0["filter_k"],
                                   frame_map=c0.get("frame_map", "gpa"), kernel_eps=c0.get("kernel_eps", 1e-2),
                                   anchor_rows=c0.get("anchor_rows", 512),
                                   anchor_subset_seed=c0.get("anchor_subset_seed", 89))
    thetas = [payload_to_theta(p, "cpu") for p in pays]
    clfs = {m: build_classifier(m, dev) for m in dict.fromkeys([*evals, *students])}
    tds = {t: task_data(t, clfs) for t in tasks}

    def acc(m, theta, split):
        return {t: ev.adapter_accuracy(clfs[m][0], frame[m], theta, tds[t].contexts[m], tds[t].text_features[m],
                                       split, dev) for t in tasks}

    search = []
    for alpha in [float(x) for x in a.alphas.split(",")]:
        th = fuse(thetas, a.method, alpha)
        val = {m: acc(m, th, "val") for m in students}
        score = sum(sum(v.values()) / len(v) for v in val.values()) / len(val)
        search.append({"alpha": alpha, "val": val, "score": score})
        print("alpha", alpha, score, flush=True)
    best = max(search, key=lambda e: e["score"])
    th = fuse(thetas, a.method, best["alpha"])
    state = AnchorState(AnchorsConfig(**MAIN_FRAME), c0["state_dir"])
    anchors = anchor_images({m: clfs[m] for m in evals})
    res = {}
    for m in evals:
        clf = clfs[m][0]
        r = {"base": {t: ev.top1(clf, tds[t].contexts[m], tds[t].text_features[m], "test", dev) for t in tasks},
             "adapter": acc(m, th, "test")}
        calib = []
        for t in tasks:
            calib += paired_images({m: split_loader(tds[t].contexts[m], "train")}, n=a.calib_batches * 32, seed=89,
                                   batch_size=32)[m]
        tv, cd = ev.compile_theta(clf, frame[m], th, calib, anchors[m],
                                  norm_filter=state.load_norm_filter(m, k=float(c0["filter_k"])), device=dev)
        with ev.applied(clf.model, tv):
            r["compiled"] = {t: ev.top1(clf, tds[t].contexts[m], tds[t].text_features[m], "test", dev) for t in tasks}
        r["compile_diag"] = cd
        res[m] = r
        print(m, r["base"], r["adapter"], r["compiled"], flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(theta_to_payload(th), out.with_suffix(".theta.pt"))
    out.write_text(json.dumps({"members": a.thetas, "tasks": tasks, "method": a.method, "search": search,
                               "alpha": best["alpha"], "results": res}, indent=1, default=str))


if __name__ == "__main__":
    main()
