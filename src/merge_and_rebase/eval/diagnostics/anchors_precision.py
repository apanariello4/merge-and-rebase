"""S1 (AGENT_SHARED_SPACE): how precise is the shared space in the deep blocks, where the task update lives?

CPU only, no forward pass: everything is read from the anchors state of a run (registered frames, stored universe)
and from its saved transported task vectors. For each target block ``j`` in ``--blocks`` (default 20-23) and its
paired source block (``DiscreteLayerPairing``, as ``anchors`` pairs them), on the frames the run actually used
(norm filter, R2 dedup, R3 row normalisation):

- ``gpa_residual``: per-model GPA residual stored in the universe (``frame_map: gpa`` only);
- ``cka`` and ``rel_err`` of the source and target anchor kernels (``core.kernel_diagnostics``);
- ``agree_energy``: with ``K~_m = K_m / tr K_m`` and the eigenbasis ``u_i`` of ``(K~_s + K~_t) / 2``, the share of
  the mean kernel energy carried by the directions where ``u_i^T K~_s u_i`` and ``u_i^T K~_t u_i`` agree within a
  factor ``--agree-factor`` (1.5); ``agree_energy_top64`` restricts both sums to the 64 leading directions;
- ``update_cos``: per task, cosine between the run's transported update of block ``j`` (``mlp.c_proj`` weight) and
  the reference run's (Ariadne, ``--reference``). Task vectors are compared unscaled (alpha does not change it).

Run::

    python -m merge_and_rebase.eval.diagnostics.anchors_precision --config RUN_CONFIG.json \\
        --reference results/anchors_kernel_20261030/runs/K1_1_ariadne [--blocks 20,21,22,23] [--out OUT.json]

``--config`` is the vision_rebase config of an ``anchors`` run (its ``logging`` names the run directory with
``transported_tvs/``). Output: JSON next to the run (``precision.json``) unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ...rebase.methods.anchors import AnchorState
from ...rebase.methods.anchors.config import parse_anchors_config
from ...rebase.methods.anchors.core import kernel, kernel_diagnostics
from ...rebase.methods.anchors.method import anchor_keep
from ...rebase.methods.anchors.universe import _block, _processed_frames, fit_universe

Tensor = torch.Tensor
REPO = Path(__file__).resolve().parents[4]


def model_id(build: Mapping[str, Any]) -> str:
    """``<model>__<pretrained>``, filesystem-safe (as ``eval/vision_rebase/method_stages._anchors_model_id``)."""
    raw = f"{build['model']}__{build['pretrained']}"
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)


def agreement(k_s: Tensor, k_t: Tensor, *, factor: float, top: int = 64) -> dict[str, float]:
    """Energy share of the mean normalised kernel on directions where the two kernels agree within ``factor``."""
    a = k_s.to(torch.float64) / torch.trace(k_s.to(torch.float64))
    b = k_t.to(torch.float64) / torch.trace(k_t.to(torch.float64))
    evals, u = torch.linalg.eigh((a + b) / 2)
    evals, u = evals.flip(0), u.flip(1)
    keep = evals > evals[0] * 1e-12  # numerically null directions carry no energy and an undefined ratio
    evals, u = evals[keep], u[:, keep]
    e_a = ((a @ u) * u).sum(0)
    e_b = ((b @ u) * u).sum(0)
    ratio = e_a.clamp_min(1e-300) / e_b.clamp_min(1e-300)
    ok = (ratio <= factor) & (ratio >= 1.0 / factor)
    mean = (e_a + e_b) / 2
    n = min(top, mean.numel())
    return {
        "agree_energy": float(mean[ok].sum() / mean.sum()),
        f"agree_energy_top{top}": float(mean[:n][ok[:n]].sum() / mean[:n].sum()),
        "n_directions": int(mean.numel()),
    }


def _cos(x: Tensor, y: Tensor) -> float:
    x, y = x.double().flatten(), y.double().flatten()
    return float(x @ y / (x.norm() * y.norm()))


def _tvs(run_dir: Path, tasks: Sequence[str]) -> dict[str, dict[str, Tensor]]:
    out = {}
    for t in tasks:
        paths = sorted((run_dir / "transported_tvs").glob(f"{t}_*_transported_native.pt"))  # anchors_ or ariadne_
        if paths:
            out[t] = torch.load(paths[0], map_location="cpu", weights_only=False)
    return out


def run(cfg: Mapping[str, Any], reference: Path | None, blocks: Sequence[int], factor: float) -> dict[str, Any]:
    params = dict(cfg["method"]["params"])
    acfg = parse_anchors_config(params)
    state_dir = Path(params["state_dir"])
    state = AnchorState(acfg, REPO / state_dir if not state_dir.is_absolute() else state_dir)
    src, tgt = model_id(cfg["models"]["source"]), model_id(cfg["models"]["target"])
    n_grid = state.model(tgt).depth
    filter_k = None if acfg.anchor_filter is None else float(acfg.anchor_filter_k)
    dedup = None if acfg.anchor_dedup_cos is None else (
        src, float(acfg.anchor_dedup_cos), acfg.anchor_dedup_max, int(acfg.anchor_seed))
    universe = None
    if acfg.frame_map == "gpa":
        fit_models = [m for m in (src, tgt, *acfg.gpa_models) if m not in set(acfg.gpa_add_models)]
        if filter_k is not None:
            for m in fit_models:
                state.load_norm_filter(m, k=filter_k)
        universe = fit_universe(state, fit_models, n_grid=n_grid, d_u=acfg.gpa_dim, add_models=acfg.gpa_add_models,
                                filter_k=filter_k, dedup=dedup, row_norm=bool(acfg.anchor_row_norm))
        keep_models = sorted(set(fit_models))
    else:
        if filter_k is not None:
            for m in (src, tgt):
                state.load_norm_filter(m, k=filter_k)
        keep_models = [src, tgt]
    run_dir = Path(cfg["logging"]["local_log_dir"])
    tasks = [t.strip() for t in str(cfg["tasks"]).split(",")]
    tv_run, tv_ref = _tvs(run_dir, tasks), (_tvs(reference, tasks) if reference else {})
    rows = []
    for j in blocks:
        keep = None if filter_k is None else anchor_keep([(state.model(m), _block(state, m, n_grid, j))
                                                          for m in keep_models], filter_k)
        frames = _processed_frames(state, [src, tgt], n_grid, j, keep, filter_k, dedup, bool(acfg.anchor_row_norm))
        k_s, k_t = kernel(frames[src]), kernel(frames[tgt])
        row: dict[str, Any] = {"target_block": j, "source_block": _block(state, src, n_grid, j),
                               "n_anchors": int(k_s.shape[0]), **kernel_diagnostics(k_s, k_t),
                               **agreement(k_s, k_t, factor=factor)}
        if universe is not None:
            row["gpa_residual"] = {m: universe.info[j]["gpa_residual_per_model"].get(m) for m in (src, tgt)}
        key = f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"
        row["update_cos"] = {t: _cos(tv_run[t][key], tv_ref[t][key]) for t in tasks if t in tv_run and t in tv_ref}
        rows.append(row)
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("cka", "rel_err", "agree_energy", "agree_energy_top64")}
    mean["update_cos"] = {t: sum(r["update_cos"][t] for r in rows) / len(rows)
                          for t in tasks if all(t in r["update_cos"] for r in rows)}
    return {"run": cfg["logging"].get("run_name"), "source": src, "target": tgt, "reference": str(reference),
            "frame_map": acfg.frame_map, "universe_version": None if universe is None else universe.version,
            "agree_factor": factor, "blocks": rows, "mean": mean}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference", default=None, help="run directory of the reference (Ariadne) transport")
    parser.add_argument("--blocks", default="20,21,22,23")
    parser.add_argument("--agree-factor", type=float, default=1.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    result = run(cfg, Path(args.reference) if args.reference else None,
                 [int(b) for b in args.blocks.split(",")], args.agree_factor)
    out = Path(args.out) if args.out else Path(cfg["logging"]["local_log_dir"]) / "precision.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
