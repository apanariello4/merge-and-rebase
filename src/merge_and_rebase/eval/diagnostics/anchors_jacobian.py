"""Linear diagnostics H1-H3 of the shared space (docs/.local/AGENT_SHARED_TRAINING.md section 2). No training.

Common space: the GPA universe of ``--fit-models`` (default A, B) on the main anchors state, the other models added by
Procrustes to the fixed centre; ``Phi_m = A_m[:r]``, ``r = --rank``; depth grid of ``--n-grid`` points, block
``pi_m(j)`` per model (``shared_space.SharedFrame``). Images: ``--n-images`` items of the task's TRAIN split (seeded
permutation, ``--seed``), identical across models (labels checked). Every logit change is centred over the classes
(softmax is shift invariant) before any metric; cosines are per image and averaged, relative errors are aggregate
``||a - b|| / ||a||`` over all images.

H1 (validity). For model m with its expert, at grid point j: ``E = h_exp(pi_m(j)) - h_base(pi_m(j))``. The true change
``z_tail(h + a E) - z(h)`` (base tail) vs the linear ``a J E``, for ``a`` in ``--alphas``; also for ``E`` projected on
the frame (``E_P = c Phi^T Phi E / c``, the part of E the common space can carry) and its energy ``||E_P|| / ||E||``.

H2 (functional agreement). Directions ``v`` in canonical common coordinates (197 tokens x r), mapped to each model's
tokens: (i) random Gaussian, (ii) each model's expert displacement ``to_canonical(E Phi^T / c)``. For every pair
(m, k): cosine and norm ratio of ``J_m v`` and ``J_k v``. On ``--n-spec`` images and the last ``--spec-grid`` grid
points: the explicit rows ``J_m`` (C x tokens x r, canonical) give the alignment ``<J_m, J_k>_F / (|J_m| |J_k|)`` and
the normalised singular values of ``J_m J_k^T`` (C x C). ``M_m(j) = sum J_m^T J_m`` (r x r) over the task images gives
the functional directions ``W_J(j)``: eigenvectors of ``sum_{m in fit} M_m / tr M_m``, each with the agreement
``min_m q_m / max_m q_m``, ``q_m = v^T M_m v / tr M_m`` (saved to ``functional_directions.pt``).

H3 (universal Jacobian). ``Jbar(j)`` = mean of the canonical rows ``J_m`` over the anchor images (the frame's
tiny-imagenet anchor draw, ``--n-anchor-images``) and the fit models. ``Jbar v`` vs ``J_m(x) v`` on the task images,
``v`` = m's expert displacement and random directions.

    python -m merge_and_rebase.eval.diagnostics.anchors_jacobian --out DIR [--tasks GTSRB,EuroSAT] [--models A,B,C]
        [--fit-models A,B] [--n-grid 12] [--rank 512] [--n-images 128] [--n-dir 64] [--n-spec 8] [--device cuda]

Output: ``DIR/jacobian_diag.json`` (+ ``functional_directions_<task>.pt``, ``jbar_<task>.pt``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.eval.shared_training.data import (
    anchor_images,
    build_classifier,
    expert_model,
    expert_path,
    model_id,
    paired_images,
    split_loader,
    task_data,
)
from merge_and_rebase.rebase.methods.anchors.jacobian import (
    VisionTail,
    canonical_adjoint,
    jacobian_rows,
    jvp_logits,
    to_tokens,
)
from merge_and_rebase.rebase.methods.anchors.shared_space import SharedFrame
from merge_and_rebase.rebase.methods.anchors.state import to_canonical

Tensor = torch.Tensor
CANON = 197
STATE = "results/anchors_state/tiny_imagenet_A16_T2048"


# ------------------------------------------------------------------ pure helpers (tested)


def centred(z: Tensor) -> Tensor:
    return z - z.mean(dim=-1, keepdim=True)


class Pair:
    """Accumulates the comparison of a reference change ``a`` with a prediction ``b`` (``[B, C]``, centred)."""

    def __init__(self) -> None:
        self.err2 = 0.0
        self.ref2 = 0.0
        self.pred2 = 0.0
        self.cos: list[float] = []

    def add(self, a: Tensor, b: Tensor) -> None:
        a, b = centred(a.double()), centred(b.double())
        self.err2 += float(((a - b) ** 2).sum())
        self.ref2 += float((a**2).sum())
        self.pred2 += float((b**2).sum())
        den = a.norm(dim=-1) * b.norm(dim=-1)
        ok = den > 0
        self.cos.extend((((a * b).sum(-1)[ok]) / den[ok]).tolist())

    def result(self) -> dict[str, float]:
        cos = torch.tensor(self.cos, dtype=torch.float64) if self.cos else torch.tensor([float("nan")])
        return {
            "rel_err": math.sqrt(self.err2 / self.ref2) if self.ref2 > 0 else float("nan"),
            "cos_mean": float(cos.mean()),
            "cos_median": float(cos.median()),
            "norm_ratio": math.sqrt(self.pred2 / self.ref2) if self.ref2 > 0 else float("nan"),
            "n": len(self.cos),
        }


def alignment(jm: Tensor, jk: Tensor) -> dict[str, Any]:
    """Per image ``[C, ...]`` rows of two models: Frobenius alignment and normalised singular values of ``J_m J_k^T``,
    averaged over images (rows centred over classes, the shift-invariant part)."""
    al, svals = [], []
    for a, b in zip(jm, jk, strict=True):
        a = a.reshape(a.shape[0], -1).double()
        b = b.reshape(b.shape[0], -1).double()
        a, b = a - a.mean(0, keepdim=True), b - b.mean(0, keepdim=True)
        al.append(float((a * b).sum() / (a.norm() * b.norm())))
        s = torch.linalg.svdvals(a @ b.T) / (a.norm() * b.norm())
        svals.append(s)
    s = torch.stack(svals).mean(0)
    return {"alignment_mean": sum(al) / len(al), "cross_svals_top": s[:10].tolist(), "cross_svals_sum": float(s.sum())}


def functional_directions(ms: dict[str, Tensor]) -> dict[str, Tensor]:
    """``W_J`` from ``M_m`` (r x r, PSD): eigenvectors of ``sum_m M_m / tr M_m`` (descending) with per-model shares
    ``q_m = v^T M_m v / tr M_m`` and the agreement ``min_m q_m / max_m q_m``."""
    normed = {m: M / torch.trace(M) for m, M in ms.items()}
    total = sum(normed.values())
    evals, evecs = torch.linalg.eigh(total)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    q = torch.stack([torch.einsum("ri,rs,si->i", evecs, M, evecs) for M in normed.values()])  # [models, r]
    agreement = q.min(0).values / q.max(0).values.clamp_min(1e-300)
    return {"eigvals": evals, "eigvecs": evecs, "shares": q, "agreement": agreement, "models": list(normed)}


# ------------------------------------------------------------------ the run


def _batches(pairs, n_take):
    """First ``n_take`` items of a list of (images, labels) batches."""
    out, seen = [], 0
    for x, y in pairs:
        if seen >= n_take:
            break
        k = min(len(x), n_take - seen)
        out.append((x[:k], y[:k]))
        seen += k
    return out


def _rebatch(pairs, bs):
    x = torch.cat([p[0] for p in pairs])
    y = torch.cat([p[1] for p in pairs])
    return [(x[i : i + bs], y[i : i + bs]) for i in range(0, len(x), bs)]


def run(args) -> dict[str, Any]:
    started = time.time()
    device = args.device
    models = [model_id(m) for m in args.models.split(",")]
    fit = [model_id(m) for m in args.fit_models.split(",")]
    frame = SharedFrame.from_state(
        args.state_dir, fit_models=fit, add_models=[m for m in models if m not in fit], n_grid=args.n_grid,
        depths="all", rank=args.rank, filter_k=args.filter_k,
    )
    grid = frame.grid
    spec_grid = grid[-args.spec_grid :]
    alphas = [float(a) for a in args.alphas.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"args": vars(args), "frame": frame.meta, "tasks": {}}
    classifiers = {m: build_classifier(m, device) for m in models}
    anchors = anchor_images(classifiers, anchor_batches=max(1, args.n_anchor_images // 32)) if args.n_anchor_images else None
    r = frame.rank
    live_fit = [m for m in fit if m in models]  # Jbar and W_J over the fit models evaluated in this run
    for task in args.tasks.split(","):
        t0 = time.time()
        td = task_data(task, classifiers)
        n_cls = len(td.classnames)
        loaders = {m: split_loader(td.contexts[m], "train") for m in models}
        pairs = paired_images(loaders, n=args.n_images, seed=args.seed, batch_size=args.batch)
        res: dict[str, Any] = {"n_classes": n_cls, "sanity": {}, "H1": {}, "H2": {}, "H3": {}, "spectrum": {}}
        gen = torch.Generator().manual_seed(args.seed)
        rand_dirs = {j: torch.randn(args.n_dir, CANON, r, generator=gen) for j in grid}
        exp_dirs: dict[str, dict[int, Tensor]] = {}
        tails = {m: VisionTail(classifiers[m][0].model, td.text_features[m].to(device)) for m in models}

        # ---- pass 1: H1 and the expert directions
        for m in models:
            fm = frame[m]
            ckpt = expert_path(m, task)
            emodel = expert_model(classifiers[m][0], ckpt).to(device)
            etail = VisionTail(emodel, td.text_features[m].to(device))
            h1 = {j: {a: Pair() for a in alphas} for j in grid}
            h1p = {j: {a: Pair() for a in alphas} for j in grid}
            energy = {j: [0.0, 0.0] for j in grid}
            correct = {"base": 0, "expert": 0, "n": 0}
            dirs = {j: [] for j in grid}
            seen = 0
            for x, y in pairs[m]:
                hb = tails[m].streams(x, fm.blocks)
                he = etail.streams(x, fm.blocks)
                z0 = tails[m].logits(hb[fm.blocks[0]], fm.blocks[0])
                ze = etail.full_logits(x)
                correct["base"] += int((z0.argmax(-1).cpu() == y).sum())
                correct["expert"] += int((ze.argmax(-1).cpu() == y).sum())
                correct["n"] += len(y)
                for j, b in zip(grid, fm.blocks, strict=True):
                    e = he[b] - hb[b]
                    du = fm.encode_delta(e.double(), j)
                    ep = fm.decode(du, j).to(e.dtype)
                    energy[j][0] += float((ep.double() ** 2).sum())
                    energy[j][1] += float((e.double() ** 2).sum())
                    if seen < args.n_dir:
                        k = min(len(x), args.n_dir - seen)
                        dirs[j].append(to_canonical(du[:k].float(), CANON).cpu())
                    for disp, acc in ((e, h1[j]), (ep, h1p[j])):
                        _, lin = jvp_logits(tails[m], hb[b], b, {b: disp})
                        for a in alphas:
                            true = tails[m].logits(hb[b], b, {b: a * disp}) - z0
                            acc[a].add(true, a * lin)
                seen += len(x)
            exp_dirs[m] = {j: torch.cat(v)[: args.n_dir] for j, v in dirs.items()}
            res["sanity"][m] = {
                "expert": str(ckpt),
                "base_acc": correct["base"] / correct["n"],
                "expert_acc": correct["expert"] / correct["n"],
                "n": correct["n"],
                "blocks": fm.blocks,
                "c": fm.c,
            }
            res["H1"][m] = {
                str(j): {
                    "block": fm.blocks[i],
                    "proj_energy": math.sqrt(energy[j][0] / max(energy[j][1], 1e-300)),
                    "full": {str(a): h1[j][a].result() for a in alphas},
                    "projected": {str(a): h1p[j][a].result() for a in alphas},
                }
                for i, j in enumerate(grid)
            }
            print(f"[H1] {task} {m}: base {res['sanity'][m]['base_acc']:.3f} expert {res['sanity'][m]['expert_acc']:.3f}; "
                  f"last-block rel_err(a=1) {res['H1'][m][str(grid[-1])]['full']['1.0']['rel_err']:.3f}", flush=True)
            del emodel, etail
            torch.cuda.empty_cache() if device.startswith("cuda") else None

        # ---- universal Jacobian Jbar (anchors, fit models) and M_m on the task images
        jbar = {j: torch.zeros(n_cls, CANON, r, dtype=torch.float64) for j in grid}
        n_jbar = 0
        if anchors is not None:
            for m in live_fit:
                fm = frame[m]
                start = fm.blocks[0]
                for x in anchors[m][: max(1, args.n_anchor_images // 32)]:
                    for xs in x.split(args.batch_rows):
                        h = tails[m].prefix(xs, start)
                        rows = jacobian_rows(tails[m], fm, h, start, grid, num_classes=n_cls)
                        for j in grid:
                            jbar[j] += canonical_adjoint(rows[j].double(), CANON).sum(0).cpu()
                        n_jbar += len(xs)
            jbar = {j: v / n_jbar for j, v in jbar.items()}
            torch.save({"jbar": jbar, "n": n_jbar, "models": live_fit, "grid": grid}, out_dir / f"jbar_{task}.pt")

        mmat = {m: {j: torch.zeros(r, r, dtype=torch.float64) for j in grid} for m in models}
        spec_rows: dict[str, dict[int, Tensor]] = {}
        for m in models:
            fm = frame[m]
            start = fm.blocks[0]
            keep = {j: [] for j in spec_grid}
            for x, _ in _rebatch(_batches(pairs[m], args.n_spec), args.batch_rows):
                h = tails[m].prefix(x, start)
                rows = jacobian_rows(tails[m], fm, h, start, grid, num_classes=n_cls)
                for j in grid:
                    rj = rows[j].double()
                    rj = rj - rj.mean(1, keepdim=True)  # centred over classes
                    mmat[m][j] += torch.einsum("bcnr,bcns->rs", rj, rj).cpu()
                    if j in keep:
                        keep[j].append(canonical_adjoint(rows[j].double(), CANON).cpu())
            spec_rows[m] = {j: torch.cat(v) for j, v in keep.items()}
        res["spectrum"] = {
            f"{m}|{k}": {str(j): alignment(spec_rows[m][j], spec_rows[k][j]) for j in spec_grid}
            for i, m in enumerate(models) for k in models[i + 1 :]
        }
        fdirs = {j: functional_directions({m: mmat[m][j] for m in live_fit}) for j in grid}
        fdirs_all = {j: functional_directions({m: mmat[m][j] for m in models}) for j in grid}
        torch.save({"fit": fdirs, "all": fdirs_all, "grid": grid, "fit_models": live_fit, "models": models},
                   out_dir / f"functional_directions_{task}.pt")
        res["functional_directions"] = {
            str(j): {
                "agreement_top": fdirs[j]["agreement"][:16].tolist(),
                "eigval_share_top": (fdirs[j]["eigvals"][:16] / fdirs[j]["eigvals"].sum()).tolist(),
                "n_agree_0.5_in_top64": int((fdirs[j]["agreement"][:64] > 0.5).sum()),
                "all_models_agreement_top": fdirs_all[j]["agreement"][:16].tolist(),
            }
            for j in grid
        }
        del spec_rows

        # ---- pass 2: H2 (J_m v vs J_k v) and H3 (Jbar v vs J_m v)
        jv: dict[str, dict[int, dict[str, list[Tensor]]]] = {}
        for m in models:
            fm = frame[m]
            jv[m] = {j: {} for j in grid}
            offset = 0
            for x, _ in _rebatch(_batches(pairs[m], args.n_dir), args.batch):
                hb = tails[m].streams(x, fm.blocks)
                n_tok = hb[fm.blocks[0]].shape[1]
                sl = slice(offset, offset + len(x))
                for j, b in zip(grid, fm.blocks, strict=True):
                    sets = {"random": rand_dirs[j][sl], **{f"expert_{k}": exp_dirs[k][j][sl] for k in models}}
                    for name, v in sets.items():
                        dh = fm.decode(to_tokens(v.to(device), n_tok).double(), j).to(hb[b].dtype)
                        _, dz = jvp_logits(tails[m], hb[b], b, {b: dh})
                        jv[m][j].setdefault(name, []).append(dz.cpu())
                offset += len(x)
            jv[m] = {j: {k: torch.cat(v) for k, v in d.items()} for j, d in jv[m].items()}
        for i, m in enumerate(models):
            for k in models[i + 1 :]:
                res["H2"][f"{m}|{k}"] = {}
                for j in grid:
                    entry = {}
                    for name in jv[m][j]:
                        p = Pair()
                        p.add(jv[m][j][name], jv[k][j][name])
                        entry[name] = p.result()
                    res["H2"][f"{m}|{k}"][str(j)] = entry
        if anchors is not None and n_jbar:
            for m in models:
                res["H3"][m] = {}
                for j in grid:
                    entry = {}
                    for name, v in (("expert_own", exp_dirs[m][j]), ("random", rand_dirs[j])):
                        pred = torch.einsum("cnr,bnr->bc", jbar[j], v.double())
                        p = Pair()
                        p.add(jv[m][j][name if name == "random" else f"expert_{m}"], pred)
                        entry[name] = p.result()
                    res["H3"][m][str(j)] = entry
        res["seconds"] = time.time() - t0
        result["tasks"][task] = res
        (out_dir / "jacobian_diag.json").write_text(json.dumps(result, indent=1))
        print(f"[jacobian] {task} done in {res['seconds']:.0f}s", flush=True)
    result["seconds"] = time.time() - started
    if device.startswith("cuda"):
        result["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    (out_dir / "jacobian_diag.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="GTSRB,EuroSAT")
    ap.add_argument("--models", default="A,B,C")
    ap.add_argument("--fit-models", default="A,B")
    ap.add_argument("--state-dir", default=STATE)
    ap.add_argument("--n-grid", type=int, default=12)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--filter-k", type=float, default=3.0)
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--n-dir", type=int, default=64)
    ap.add_argument("--n-spec", type=int, default=8)
    ap.add_argument("--spec-grid", type=int, default=4)
    ap.add_argument("--n-anchor-images", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch-rows", type=int, default=4, help="images per batch of the explicit Jacobian rows")
    ap.add_argument("--seed", type=int, default=89)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return run(args)


if __name__ == "__main__":
    main()
