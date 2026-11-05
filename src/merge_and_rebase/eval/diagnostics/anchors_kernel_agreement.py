"""Kernel agreement of the registered models on the anchors (docs/.local/AGENT_ADDENDUM_CONSENSUS.md section 4).

CPU only, no forward pass: reads the stored anchor frames of the main state (read-only; the norm filter of every
model must already be stored). At each relative depth ``s_j`` of a grid of ``--n-grid`` points, model m contributes
block ``pi_m(j)`` (``universe._block``); the anchors flagged high-norm in ANY of the models are dropped (one common
anchor set per depth), ``P_m`` as ``method.filtered_frame``.

* linear CKA: ``<H K_m H, H K_k H>_F / (||H K_m H|| ||H K_k H||)``, ``K = P P^T``;
* mutual k-NN (Huh et al. 2024, Platonic Representation Hypothesis): for every anchor, its ``k`` nearest neighbours
  among the other anchors by cosine similarity of the centred rows, in each model; score = mean over anchors of
  ``|N_m(i) & N_k(i)| / k``;
* per model: mean CKA / mutual k-NN to the others, and the vision tower's parameter count (``open_clip`` architecture,
  no weights loaded).

    python -m merge_and_rebase.eval.diagnostics.anchors_kernel_agreement --out OUT.json [--models ...] [--n-grid 12]
        [--knn 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.rebase.methods.anchors.config import AnchorsConfig
from merge_and_rebase.rebase.methods.anchors.method import anchor_keep, filtered_frame
from merge_and_rebase.rebase.methods.anchors.shared_space import MAIN_FRAME
from merge_and_rebase.rebase.methods.anchors.state import AnchorState
from merge_and_rebase.rebase.methods.anchors.universe import _block

Tensor = torch.Tensor
STATE = "results/anchors_state/tiny_imagenet_A16_T2048"


def centred_kernel(p: Tensor) -> Tensor:
    p = p.to(torch.float64)
    p = p - p.mean(0, keepdim=True)
    return p @ p.T


def cka(km: Tensor, kk: Tensor) -> float:
    """Linear CKA of two centred kernels."""
    return float((km * kk).sum() / (km.norm() * kk.norm()))


def knn_sets(p: Tensor, k: int) -> Tensor:
    """``[T, k]`` indices of the k nearest anchors (cosine of the centred rows, self excluded)."""
    x = p.to(torch.float64)
    x = x - x.mean(0, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-300)
    sim = x @ x.T
    sim.fill_diagonal_(-float("inf"))
    return sim.topk(k, dim=1).indices


def mutual_knn(a: Tensor, b: Tensor) -> float:
    """Mean overlap fraction of two ``[T, k]`` neighbour sets."""
    k = a.shape[1]
    hits = (a[:, :, None] == b[:, None, :]).any(-1).sum(1).to(torch.float64)
    return float((hits / k).mean())


def vision_params(model_id: str) -> int:
    import open_clip

    arch = model_id.split("__", 1)[0]
    model = open_clip.create_model(arch, pretrained=None)
    return int(sum(p.numel() for p in model.visual.parameters()))


def run(args) -> dict[str, Any]:
    state = AnchorState(AnchorsConfig(**MAIN_FRAME), args.state_dir)
    models = args.models.split(",") if args.models else sorted(
        p.name for p in (Path(args.state_dir) / "models").iterdir() if (p / "meta.json").exists())
    k_filter = float(args.filter_k)
    for m in models:
        state.load_norm_filter(m, k=k_filter)
    out: dict[str, Any] = {"models": models, "n_grid": args.n_grid, "knn": args.knn, "filter_k": k_filter,
                           "depths": {}}
    for j in range(args.n_grid):
        regs = [(state.model(m), _block(state, m, args.n_grid, j)) for m in models]
        keep = anchor_keep(regs, k_filter)
        frames = {m: filtered_frame(reg, b, keep, k_filter)[0] for m, (reg, b) in zip(models, regs, strict=True)}
        kers = {m: centred_kernel(p) for m, p in frames.items()}
        nn = {m: knn_sets(p, args.knn) for m, p in frames.items()}
        cka_m = torch.eye(len(models), dtype=torch.float64)
        knn_m = torch.eye(len(models), dtype=torch.float64)
        for a in range(len(models)):
            for b in range(a + 1, len(models)):
                cka_m[a, b] = cka_m[b, a] = cka(kers[models[a]], kers[models[b]])
                knn_m[a, b] = knn_m[b, a] = mutual_knn(nn[models[a]], nn[models[b]])
        off = 1 - torch.eye(len(models), dtype=torch.float64)
        out["depths"][str(j)] = {
            "blocks": {m: b for m, (_, b) in zip(models, regs, strict=True)},
            "n_anchors": int(keep.sum()),
            "cka": cka_m.tolist(),
            "mutual_knn": knn_m.tolist(),
            "mean_cka_to_others": dict(zip(models, ((cka_m * off).sum(1) / off.sum(1)).tolist(), strict=True)),
            "mean_knn_to_others": dict(zip(models, ((knn_m * off).sum(1) / off.sum(1)).tolist(), strict=True)),
        }
        print(f"[kernel] depth {j}: anchors {int(keep.sum())}, mean CKA "
              f"{float((cka_m * off).sum() / off.sum()):.3f}, mean mkNN {float((knn_m * off).sum() / off.sum()):.3f}",
              flush=True)
    out["vision_params"] = {m: vision_params(m) for m in models}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="comma-separated model ids (default: every registered model)")
    ap.add_argument("--state-dir", default=STATE)
    ap.add_argument("--n-grid", type=int, default=12)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--filter-k", type=float, default=3.0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
