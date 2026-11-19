"""Runner of the linear shared-space experiments LIN-1 / LIN-2 (AGENT_SHARED_TRAINING.md section 3).

    python -m merge_and_rebase.eval.shared_training CONFIG.json [--device cuda]

Config keys (strict; defaults in ``DEFAULTS``):

* ``name``, ``out_dir``; ``experiment``: ``lin1`` (distillation of the teachers' logit change, closed form by CG) or
  ``lin2`` (CE on the linearised model, labels, L-BFGS);
* ``task``; ``students`` (M_train), ``teachers`` (lin1: experts whose mean logits are imitated), ``eval_models``;
* ``lin_point``: ``task`` | ``anchors`` (``jbar``: path of the H3 output ``jbar_<task>.pt``);
* frame: ``state_dir``, ``fit_models``, ``n_grid``, ``depths``, ``rank``, ``filter_k``, ``frame_map``
  (gpa | anchor_zca | anchor_tikhonov; the anchor maps use ``kernel_eps``, ``anchor_rows``, ``anchor_subset_seed``, no common frame);
* data: ``n_train`` paired TRAIN images (``seed``), ``batch``;
* regularisation grid: ``lam_rel`` x ``lam_p_rel`` (relative to the data and anchor Hessian scales, lin.py), chosen
  by the students' mean VALIDATION accuracy of the true model with the adapter (the eval models outside the students
  never enter the choice);
* ``rank_project``: optional final truncation of each W; ``subspace``: ``{"path": functional_directions_<task>.pt,
  "k": int}`` restricts W to ``W_J W~ W_J^T`` (LIN-2d, lin2 only);
* evaluation: ``compiled`` (S0 synthesis into c_proj, ``rho_g``, ``calib_batches`` x 32 train images, seed 89 as the
  transport runs), ``linearized_batches`` (test batches for the linear-true gap), ``conservation`` (list of modes
  among ``adapter``, ``compiled``: the other 7 Vision8 tasks, test split).

Output ``out_dir/<name>.json`` (summary, per eval model: base / adapter / compiled val and test accuracy, linearised
accuracy, compile diagnostics, conservation), ``theta.pt`` (W, b per grid point), ``transported_tvs/`` (compiled
vectors ``<task>_<model>_shared_transported_native.pt``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.rebase.methods.anchors.config import AnchorsConfig
from merge_and_rebase.rebase.methods.anchors.jacobian import VisionTail
from merge_and_rebase.rebase.methods.anchors.shared_space import MAIN_FRAME, SharedFrame
from merge_and_rebase.rebase.methods.anchors.state import AnchorState

from . import evaluate as ev
from .data import (
    RUN_CFG,
    anchor_images,
    build_classifier,
    expert_model,
    expert_path,
    model_id,
    paired_images,
    split_loader,
    task_data,
)
from .lin import (
    LinearizedProblem,
    anchor_hessian_scale,
    collect_student,
    data_scale,
    lin1_fit_quality,
    lin2_train_accuracy,
    project_rank,
    solve_lin1,
    solve_lin2,
    theta_to_payload,
)

DEFAULTS: dict[str, Any] = {
    "name": None,
    "out_dir": None,
    "experiment": "lin1",
    "task": None,
    "students": ["A"],
    "teachers": ["A"],
    "eval_models": ["A", "B", "C"],
    "lin_point": "task",
    "jbar": None,
    "state_dir": "results/anchors_state/tiny_imagenet_A16_T2048",
    "fit_models": ["A", "B"],
    "n_grid": 12,
    "depths": "last4",
    "rank": 512,
    "filter_k": 3.0,
    "frame_map": "gpa",
    "kernel_eps": 1e-2,
    "anchor_rows": 512,
    "anchor_subset_seed": 89,
    "n_train": 2048,
    "seed": 89,
    "batch": 32,
    "lam_rel": [1e-3, 1e-2, 1e-1],
    "lam_p_rel": [0.0],
    "cg_tol": 1e-4,
    "cg_max_iter": 200,
    "lbfgs_max_iter": 100,
    "rank_project": None,
    "subspace": None,
    "compiled": True,
    "rho_g": 0.1,
    "calib_batches": 10,
    "linearized_batches": 10,
    "conservation": [],
    "val_on": "students",
    "init": None,
    "lr": [1e-4, 3e-4, 1e-3],
    "epochs": 5,
    "nl_lam": 0.0,
    "nl_lam_p": 0.0,
}


def load_config(path: str) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text())
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config keys {unknown}")
    out = {**DEFAULTS, **cfg}
    for k in ("name", "out_dir", "task"):
        if not out[k]:
            raise ValueError(f"config needs {k!r}")
    if out["experiment"] not in ("lin1", "lin2", "nl2_local", "nl1", "nl2_global"):
        raise ValueError(f"experiment must be lin1, lin2, nl2_local, nl1 or nl2_global, got {out['experiment']!r}")
    if out["lin_point"] == "anchors" and not out["jbar"] and out["experiment"] in ("lin1", "lin2"):
        raise ValueError("lin_point='anchors' needs 'jbar' (the H3 output jbar_<task>.pt)")
    if out["subspace"] is not None and out["experiment"] != "lin2":
        raise ValueError("subspace (functional_subspace, LIN-2d) is a lin2 option")
    if out["frame_map"] not in ("gpa", "anchor_zca", "anchor_tikhonov"):
        raise ValueError(f"frame_map must be gpa, anchor_zca or anchor_tikhonov, got {out['frame_map']!r}")
    bad = set(out["conservation"]) - {"adapter", "compiled"}
    if bad:
        raise ValueError(f"conservation modes must be among adapter, compiled; got {sorted(bad)}")
    return out


def _git() -> dict[str, Any]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
        return {"commit": head, "dirty_tracked": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run(cfg: dict[str, Any], device: str) -> dict[str, Any]:
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    task = cfg["task"]
    students = [model_id(m) for m in cfg["students"]]
    teachers = [model_id(m) for m in cfg["teachers"]]
    evals = [model_id(m) for m in cfg["eval_models"]]
    fit = [model_id(m) for m in cfg["fit_models"]]
    everyone = list(dict.fromkeys([*students, *evals, *teachers]))
    frame = SharedFrame.from_state(cfg["state_dir"], fit_models=fit, add_models=[m for m in everyone if m not in fit],
                                   n_grid=cfg["n_grid"], depths=cfg["depths"], rank=cfg["rank"],
                                   filter_k=cfg["filter_k"], frame_map=cfg["frame_map"],
                                   kernel_eps=cfg["kernel_eps"], anchor_rows=cfg["anchor_rows"],
                                   anchor_subset_seed=cfg["anchor_subset_seed"])
    grid, r = frame.grid, frame.rank
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[shared] {cfg['name']}: {cfg['experiment']} {task} students {students} lin_point {cfg['lin_point']} "
        f"grid {grid} blocks {frame.meta['blocks']}")
    classifiers = {m: build_classifier(m, device) for m in everyone}
    td = task_data(task, classifiers, {"seed": RUN_CFG["seed"]})
    loaders = {m: split_loader(td.contexts[m], "train") for m in everyone}
    train = paired_images(loaders, n=cfg["n_train"], seed=cfg["seed"], batch_size=cfg["batch"])

    teacher_logits = None
    teacher_info = {}
    if cfg["experiment"] in ("lin1", "nl2_global"):
        acc = []
        for t in teachers:
            emodel = expert_model(classifiers[t][0], expert_path(t, task)).to(device)
            tail = VisionTail(emodel, td.text_features[t].to(device))
            with torch.no_grad():
                z = torch.cat([tail.full_logits(x).double().cpu() for x, _ in train[t]])
            y = torch.cat([y for _, y in train[t]])
            teacher_info[t] = {"expert": str(expert_path(t, task)), "train_acc": float((z.argmax(-1) == y).float().mean())}
            acc.append(z - z.mean(-1, keepdim=True))
            del emodel, tail
            torch.cuda.empty_cache()
        teacher_logits = sum(acc) / len(acc)

    jbar = None
    if cfg["lin_point"] == "anchors" and cfg["experiment"] in ("lin1", "lin2"):
        payload = torch.load(cfg["jbar"], map_location="cpu")
        jbar = {j: payload["jbar"][j] for j in grid}

    gmat = {j: frame.g[j] for j in grid}
    s_data = s_p = None
    if cfg["experiment"] in ("lin1", "lin2"):
        search, theta, chosen, s_data, s_p = _fit_linear(cfg, students, classifiers, td, frame, train, teacher_logits,
                                                         jbar, gmat, device, log)
    elif cfg["experiment"] == "nl2_local":
        search, theta, chosen = _fit_nl2_local(cfg, students, classifiers, td, frame, train, gmat, device, log)
    else:
        search, theta, chosen = _fit_through_models(cfg, students, classifiers, td, frame, train, teacher_logits,
                                                    gmat, device, log)
    torch.save({**theta_to_payload(theta), "frame": frame.meta, "config": cfg, "chosen": chosen}, out_dir / "theta.pt")

    results: dict[str, Any] = {}
    anchors = None
    state = None
    if cfg["compiled"]:
        anchors = anchor_images({m: classifiers[m] for m in evals})
        state = AnchorState(AnchorsConfig(**MAIN_FRAME), cfg["state_dir"])
    tv_dir = out_dir / "transported_tvs"
    for m in evals:
        clf = classifiers[m][0]
        ctx, head = td.contexts[m], td.text_features[m]
        res: dict[str, Any] = {
            "seen": m in students,
            "base": {sp: ev.top1(clf, ctx, head, sp, device) for sp in ("val", "test")},
            "adapter": {sp: ev.adapter_accuracy(clf, frame[m], theta, ctx, head, sp, device) for sp in ("val", "test")},
        }
        if cfg["linearized_batches"]:
            res["linearized_vs_true"] = ev.linearized_accuracy(
                clf, frame[m], theta, split_loader(ctx, "test"), head, lin_point=cfg["lin_point"], device=device,
                jbar=jbar, max_batches=cfg["linearized_batches"])
        tv = None
        if cfg["compiled"]:
            calib = paired_images({m: split_loader(ctx, "train")}, n=cfg["calib_batches"] * 32, seed=89,
                                  batch_size=32)[m]
            norm = state.load_norm_filter(m, k=float(cfg["filter_k"])) if cfg["filter_k"] is not None else None
            tv, cdiag = ev.compile_theta(clf, frame[m], theta, calib, anchors[m], norm_filter=norm,
                                         rho_g=cfg["rho_g"], device=device)
            tv_dir.mkdir(exist_ok=True)
            torch.save(tv, tv_dir / f"{task}_{m}_shared_transported_native.pt")
            with ev.applied(clf.model, tv):
                res["compiled"] = {sp: ev.top1(clf, ctx, head, sp, device) for sp in ("val", "test")}
            res["compile_diag"] = cdiag
        if cfg["conservation"]:
            res["conservation"] = _conservation(cfg, m, classifiers[m], frame[m], theta, tv, task, device)
        results[m] = res
        log(f"[shared] eval {m}: base {res['base']['test']:.4f} adapter {res['adapter']['test']:.4f} "
            f"compiled {res.get('compiled', {}).get('test', float('nan')):.4f}")

    summary = {
        "name": cfg["name"],
        "config": cfg,
        "git": _git(),
        "frame": frame.meta,
        "teachers": teacher_info,
        "scales": {"data": s_data, "anchors": s_p},
        "search": search,
        "chosen": chosen,
        "results": results,
        "w_parameters": len(grid) * (r * r + r),
        "seconds": time.time() - started,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None,
        "log": log_lines,
    }
    (out_dir / f"{cfg['name']}.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def _val_score(cfg, students, classifiers, td, frame, theta, device):
    val = {m: ev.adapter_accuracy(classifiers[m][0], frame[m], theta, td.contexts[m], td.text_features[m], "val",
                                  device) for m in students}
    return val, sum(val.values()) / len(val)


def _fit_linear(cfg, students, classifiers, td, frame, train, teacher_logits, jbar, gmat, device, log):
    grid, r = frame.grid, frame.rank
    sts = []
    for m in students:
        tail = VisionTail(classifiers[m][0].model, td.text_features[m].to(device))
        sts.append(collect_student(m, tail, frame[m], grid, train[m], lin_point=cfg["lin_point"],
                                   teacher_logits=teacher_logits))
    problem = LinearizedProblem(sts, grid, r, lin_point=cfg["lin_point"], device=device, jbar=jbar)
    s_data = data_scale(problem)
    s_p = anchor_hessian_scale(gmat)
    log(f"[shared] scales: data {s_data:.4e} anchors {s_p:.4e}")
    subspace = None
    if cfg["subspace"] is not None:
        fd = torch.load(cfg["subspace"]["path"], map_location="cpu")["fit"]
        k = int(cfg["subspace"]["k"])
        subspace = {j: fd[j]["eigvecs"][:, :k].to(torch.float64) for j in grid}
    search, best, x0 = [], None, None
    for lam_rel, lam_p_rel in sorted(itertools.product(cfg["lam_rel"], cfg["lam_p_rel"]), reverse=True):
        lam, lam_p = lam_rel * s_data, (lam_p_rel * s_data / s_p if lam_p_rel else 0.0)
        if cfg["experiment"] == "lin1":
            theta, info = solve_lin1(problem, lam=lam, lam_p=lam_p, g=gmat, x0=x0, tol=cfg["cg_tol"],
                                     max_iter=cfg["cg_max_iter"], log=log)
            info["train_rel_residual"] = lin1_fit_quality(problem, theta)
        else:
            if subspace is not None:
                theta, info = _solve_lin2_subspace(problem, subspace, lam, lam_p, gmat, cfg, log)
            else:
                theta, info = solve_lin2(problem, lam=lam, lam_p=lam_p, g=gmat, x0=x0,
                                         max_iter=cfg["lbfgs_max_iter"], log=log)
            info["train_acc_linearized"] = lin2_train_accuracy(problem, theta)
        x0 = theta
        if cfg["rank_project"]:
            theta = project_rank(theta, int(cfg["rank_project"]))
        val, score = _val_score(cfg, students, classifiers, td, frame, theta, device)
        entry = {"lam_rel": lam_rel, "lam_p_rel": lam_p_rel, "lam": lam, "lam_p": lam_p, "val_acc": val,
                 "score": score, **{k: v for k, v in info.items() if k != "cg_history"}}
        search.append(entry)
        log(f"[shared] lam_rel {lam_rel:g} lam_p_rel {lam_p_rel:g}: val {val} ({info.get('seconds', 0):.0f}s)")
        if best is None or score > best[0]:
            best = (score, theta, entry)
    del problem, sts
    torch.cuda.empty_cache()
    return search, best[1], best[2], s_data, s_p


def _fit_nl2_local(cfg, students, classifiers, td, frame, train, gmat, device, log):
    from .nonlinear import fit_nl2_local

    sts = []
    for m in students:
        expert = expert_model(classifiers[m][0], expert_path(m, cfg["task"])).to(device)
        sts.append({"model": classifiers[m][0].model, "expert": expert, "frame": frame[m],
                    "batches": [x for x, _ in train[m]]})
    search, best = [], None
    for lam_rel, lam_p_rel in sorted(itertools.product(cfg["lam_rel"], cfg["lam_p_rel"]), reverse=True):
        theta, info = fit_nl2_local(sts, frame.grid, frame.rank, gmat, lam_rel=lam_rel, lam_p_rel=lam_p_rel,
                                    device=device, log=log)
        val, score = _val_score(cfg, students, classifiers, td, frame, theta, device)
        entry = {"lam_rel": lam_rel, "lam_p_rel": lam_p_rel, "val_acc": val, "score": score, **info}
        search.append(entry)
        log(f"[shared] NL-2 local lam_rel {lam_rel:g} lam_p_rel {lam_p_rel:g}: val {val}")
        if best is None or score > best[0]:
            best = (score, theta, entry)
    for st in sts:
        del st["expert"]
    torch.cuda.empty_cache()
    return search, best[1], best[2]


def _fit_through_models(cfg, students, classifiers, td, frame, train, teacher_logits, gmat, device, log):
    from .lin import payload_to_theta
    from .nonlinear import train_through_models

    grid, r = frame.grid, frame.rank
    init = payload_to_theta(torch.load(cfg["init"], map_location="cpu"), "cpu") if cfg["init"] else None
    sts = []
    for m in students:
        tail = VisionTail(classifiers[m][0].model, td.text_features[m].to(device))
        fdev = ev.frame_on(frame[m], device)
        start = min(fdev.blocks[fdev.index(j)] for j in grid)
        with torch.no_grad():
            h0 = torch.cat([tail.prefix(x, start).cpu() for x, _ in train[m]])
        labels = torch.cat([torch.as_tensor(y) for _, y in train[m]])
        sts.append({"tail": tail, "frame": fdev, "start": start, "h0": h0, "labels": labels,
                    "teacher": teacher_logits})
    loss = "ce" if cfg["experiment"] == "nl1" else "kl"
    search, best = [], None
    for lr in cfg["lr"]:
        theta, info = train_through_models(
            sts, grid, r, gmat, init=init, loss=loss, lr=float(lr), epochs=int(cfg["epochs"]), batch=cfg["batch"],
            lam=float(cfg["nl_lam"]), lam_p=float(cfg["nl_lam_p"]), device=device,
            val_fn=lambda th: _val_score(cfg, students, classifiers, td, frame, th, device)[1], log=log)
        entry = {"lr": lr, "score": info["best_val"], **info}
        search.append(entry)
        log(f"[shared] {cfg['experiment']} lr {lr:g}: best val {info['best_val']:.4f} at epoch {info['best_epoch']}")
        if best is None or info["best_val"] > best[0]:
            best = (info["best_val"], theta, entry)
    del sts
    torch.cuda.empty_cache()
    return search, best[1], best[2]


def _conservation(cfg, m, classifier, frame, theta, tv, task, device) -> dict[str, Any]:
    from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES

    tasks = list(SUITES["vision8"].tasks)
    td_all = {t: task_data(t, {m: classifier}) for t in tasks}
    contexts = {t: td_all[t].contexts[m] for t in tasks}
    heads = {t: td_all[t].text_features[m] for t in tasks}
    clf = classifier[0]
    base = {t: ev.top1(clf, contexts[t], heads[t], "test", device) for t in tasks}
    out: dict[str, Any] = {"base_test_accuracy": base}
    if "adapter" in cfg["conservation"]:
        out["adapter"] = ev.conservation(clf, contexts, heads, task, base, device=device, frame=frame, theta=theta)
    if "compiled" in cfg["conservation"] and tv is not None:
        out["compiled"] = ev.conservation(clf, contexts, heads, task, base, device=device, tv=tv)
    return out


def _solve_lin2_subspace(problem, subspace, lam, lam_p, gmat, cfg, log):
    """LIN-2d: W = W_J W~ W_J^T. Solved as LIN-2 on the full W with the gradient projected onto the subspace
    (``W_J W_J^T G W_J W_J^T``): with x0 = 0 every L-BFGS iterate stays in the subspace."""
    from . import lin

    proj = {j: q.to(problem.device) @ q.to(problem.device).T for j, q in subspace.items()}
    orig = problem.sweep

    def sweep(theta, cot):
        acc, loss = orig(theta, cot)
        return {j: (proj[j] @ w @ proj[j], b) for j, (w, b) in acc.items()}, loss

    problem.sweep = sweep
    orig_reg = lin._reg

    def reg(theta, lam_, lam_p_, g):
        out = orig_reg(theta, lam_, lam_p_, g)
        return {j: (proj[j] @ w @ proj[j], b) for j, (w, b) in out.items()}

    lin._reg = reg
    try:
        return solve_lin2(problem, lam=lam, lam_p=lam_p, g=gmat, max_iter=cfg["lbfgs_max_iter"], log=log)
    finally:
        problem.sweep = orig
        lin._reg = orig_reg


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    return run(load_config(args.config), args.device)
