"""Step 2 of AGENT_NEXT.md: are the deep-block kernel collapses (M2) caused by a few high-norm token rows?

Forward passes only, on the anchor pool of the frame (Tiny-ImageNet ``valid``, the seeded ``paired_calibration`` draw
of ``anchor_batches + registration_batches`` batches, as ``AnchorState.register_model`` replays it).

Per model and block:

- registration statistics ``mu``, ``c`` (as the frame) from the registration images;
- the norm of every registration token row after centring, ``n = ||x - mu||``; a row is *high-norm* when
  ``n > k * median_block(n)`` (``k`` in ``ks``; the median is over the registration rows of that block);
- ``mu_k``, ``c_k`` recomputed without the high-norm registration rows;
- the anchor rows: the first ``num_anchors`` of the frame's seeded draw over (anchor image, canonical position)
  (nested with the frames of the T sweep), plus the CLS row of every anchor image.

Reported per block: flagged fraction, its distribution over the canonical patch grid, and the overlap of the flagged
(image, position) sets of the paired source / target blocks (Jaccard). Kernel profile per target block ``j`` and
variant (CKA of the centred kernels with a permutation null and two disjoint halves, relative error, effective
rank; all through ``d x d`` products):

- ``F0``: every anchor, frame ``mu``, ``c`` (reference);
- ``F1_k``: anchors high-norm in the source OR the target block removed, ``mu_k``, ``c_k`` without high-norm rows;
- ``F3``: every anchor, rows normalised to unit norm (cosine kernel);
- ``F4``: CLS rows of the anchor images only (control).

``F2`` (high-norm rows also removed from E and H in the ridge) changes the transport, not this profile.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ...rebase.discrete_layer_match import DiscreteLayerPairing
from ...rebase.methods.anchors.core import (
    MomentAccumulator,
    kernel_diagnostics_from_frames,
    kernel_spectrum_from_frame,
)

Tensor = torch.Tensor

DEFAULTS: dict[str, Any] = {
    "anchor_batches": 16,
    "registration_batches": 10,
    "pool_seed": 0,
    "anchor_seed": 0,
    "num_anchors": 1024,
    "canonical_tokens": 197,
    "ks": [3.0, 5.0],
    "n_perm": 30,
    "analysis_seed": 0,
    "batch_size": 32,
    "num_workers": 0,
    "val_fraction": 0.1,
    "seed": 89,
    "suite": "vision8",
    "device": "cuda",
}


# ---- pure analysis (unit-tested without models) -------------------------------------------------------------------


def frame_anchor_rows(n_images: int, canonical_tokens: int, num_anchors: int, anchor_seed: int) -> list[list[int]]:
    """The frame's anchor draw (``AnchorState._establish_frame`` with ``anchor_positions='all'``): sorted
    ``[image, position]`` rows. Prefixes of the same permutation, so smaller T are subsets of larger ones."""
    n_candidates = n_images * canonical_tokens
    perm = torch.randperm(n_candidates, generator=torch.Generator().manual_seed(int(anchor_seed)))
    chosen = sorted(int(i) for i in perm[:num_anchors])
    return [[i // canonical_tokens, i % canonical_tokens] for i in chosen]


def profile_stats(p_a: Tensor, p_b: Tensor, *, n_perm: int, seed: int) -> dict[str, float]:
    """CKA / relative error of the kernels of two frames on the same anchors, permutation null, halves, ranks."""
    p_a = p_a.to(torch.float64)
    p_b = p_b.to(torch.float64)
    t = int(p_a.shape[0])
    gen = torch.Generator().manual_seed(int(seed))
    point = kernel_diagnostics_from_frames(p_a, p_b)
    null = torch.tensor(
        [kernel_diagnostics_from_frames(p_a, p_b[torch.randperm(t, generator=gen)])["cka"] for _ in range(n_perm)],
        dtype=torch.float64,
    )
    halves = torch.randperm(t, generator=gen)
    h1, h2 = halves[: t // 2], halves[t // 2 :]
    return {
        "T": t,
        "cka": point["cka"],
        "rel_err": point["rel_err"],
        "cka_null_q95": float(torch.quantile(null, 0.95)) if n_perm else float("nan"),
        "cka_null_mean": float(null.mean()) if n_perm else float("nan"),
        "cka_half_1": kernel_diagnostics_from_frames(p_a[h1], p_b[h1])["cka"],
        "cka_half_2": kernel_diagnostics_from_frames(p_a[h2], p_b[h2])["cka"],
        "source_effective_rank": kernel_spectrum_from_frame(p_a)["effective_rank"],
        "target_effective_rank": kernel_spectrum_from_frame(p_b)["effective_rank"],
    }


def unit_rows(p: Tensor) -> Tensor:
    """Rows scaled to unit norm (the cosine kernel's frame); zero rows stay zero."""
    n = p.norm(dim=1, keepdim=True)
    return p / torch.where(n > 0, n, torch.ones_like(n))


def jaccard(a: Tensor, b: Tensor) -> float:
    union = int((a | b).sum())
    return float(int((a & b).sum()) / union) if union else float("nan")


# ---- capture -------------------------------------------------------------------------------------------------------


@torch.no_grad()
def model_statistics(model, batches, cfg: Mapping[str, Any], anchor_rows: list[list[int]], device) -> dict[str, Any]:
    """Two sweeps over the pool draw: (1) frame mu/c from registration rows, anchor + CLS rows of anchor images;
    (2) registration row norms, median, flags per k, mu_k/c_k without flagged rows."""
    from ...rebase.methods._ariadne.capture import iter_capture_tokens
    from ...rebase.methods.anchors.state import to_canonical, vision_block_count

    canonical = int(cfg["canonical_tokens"])
    n_anchor_batches = int(cfg["anchor_batches"])
    depth = vision_block_count(model)
    requests = {str(j): (j, "boundary") for j in range(depth)}
    sizes = [int(b[0].shape[0]) for b in batches]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    moments = [MomentAccumulator() for _ in range(depth)]
    anchors: list[list[Tensor]] = [[] for _ in range(depth)]
    cls: list[list[Tensor]] = [[] for _ in range(depth)]
    for b, values in enumerate(iter_capture_tokens(model, batches, requests, device)):
        if b < n_anchor_batches:
            lo, hi = offsets[b], offsets[b + 1]
            sel = [(img - lo, pos) for img, pos in anchor_rows if lo <= img < hi]
            img_idx = torch.tensor([s[0] for s in sel], dtype=torch.long)
            pos_idx = torch.tensor([s[1] for s in sel], dtype=torch.long)
        for j in range(depth):
            canon = to_canonical(values[str(j)], canonical)
            if b < n_anchor_batches:
                anchors[j].append(canon[img_idx, pos_idx].double())
                cls[j].append(canon[:, 0].double())
            else:
                moments[j].update(canon)
    frame = [moments[j].finalize() for j in range(depth)]
    reg_batches = batches[n_anchor_batches:]
    norms: list[list[Tensor]] = [[] for _ in range(depth)]
    for values in iter_capture_tokens(model, reg_batches, requests, device):
        for j in range(depth):
            canon = to_canonical(values[str(j)], canonical).double()
            norms[j].append((canon - frame[j][0].to(canon.device)).norm(dim=-1).cpu())  # [B, canonical]
    norm_grid = [torch.cat(n, 0) for n in norms]  # [n_reg_images, canonical]
    medians = [float(g.median()) for g in norm_grid]
    ks = [float(k) for k in cfg["ks"]]
    filtered = {k: [MomentAccumulator() for _ in range(depth)] for k in ks}
    for values in iter_capture_tokens(model, reg_batches, requests, device):
        for j in range(depth):
            canon = to_canonical(values[str(j)], canonical).double()
            n = (canon - frame[j][0].to(canon.device)).norm(dim=-1)
            for k in ks:
                keep = n <= k * medians[j]
                filtered[k][j].update(canon[keep])
    return {
        "depth": depth,
        "mu_c": frame,
        "mu_c_filtered": {k: [acc.finalize() for acc in filtered[k]] for k in ks},
        "medians": medians,
        "norm_grid": norm_grid,
        "anchors": [torch.cat(a, 0).cpu() for a in anchors],
        "cls": [torch.cat(c, 0).cpu() for c in cls],
    }


def analyse(src: Mapping[str, Any], tgt: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    pairing = DiscreteLayerPairing.compute(src["depth"], tgt["depth"])
    ks = [float(k) for k in cfg["ks"]]
    canonical = int(cfg["canonical_tokens"])
    grid = int(round((canonical - 1) ** 0.5))
    seed = int(cfg["analysis_seed"])
    n_perm = int(cfg["n_perm"])
    per_model: dict[str, Any] = {}
    for name, st in (("source", src), ("target", tgt)):
        rows = []
        for j in range(st["depth"]):
            g, med = st["norm_grid"][j], st["medians"][j]
            entry: dict[str, Any] = {"median_norm": med}
            for k in ks:
                flag = g > k * med
                pos_frac = flag.float().mean(0)  # per canonical position
                entry[f"k{k:g}_fraction"] = float(flag.float().mean())
                entry[f"k{k:g}_cls_fraction"] = float(pos_frac[0])
                entry[f"k{k:g}_patch_grid"] = pos_frac[1:].reshape(grid, grid).tolist()
            rows.append(entry)
        per_model[name] = rows
    profile = []
    for j in range(tgt["depth"]):
        i = int(pairing.pairing[j])
        mu_s, c_s = src["mu_c"][i]
        mu_t, c_t = tgt["mu_c"][j]
        a_s, a_t = src["anchors"][i], tgt["anchors"][j]
        p_s = (a_s - mu_s) / c_s
        p_t = (a_t - mu_t) / c_t
        entry: dict[str, Any] = {"target_block": j, "source_block": i, "s": j / max(tgt["depth"] - 1, 1)}
        entry["F0"] = profile_stats(p_s, p_t, n_perm=n_perm, seed=seed)
        for k in ks:
            flag_s = (a_s - mu_s).norm(dim=1) > k * src["medians"][i]
            flag_t = (a_t - mu_t).norm(dim=1) > k * tgt["medians"][j]
            keep = ~(flag_s | flag_t)
            mu_sk, c_sk = src["mu_c_filtered"][k][i]
            mu_tk, c_tk = tgt["mu_c_filtered"][k][j]
            stats = profile_stats((a_s[keep] - mu_sk) / c_sk, (a_t[keep] - mu_tk) / c_tk, n_perm=n_perm, seed=seed)
            reg_s = src["norm_grid"][i] > k * src["medians"][i]
            reg_t = tgt["norm_grid"][j] > k * tgt["medians"][j]
            entry[f"F1_k{k:g}"] = {
                **stats,
                "anchors_removed": int((~keep).sum()),
                "anchors_flagged_source": int(flag_s.sum()),
                "anchors_flagged_target": int(flag_t.sum()),
                "registration_jaccard_source_target": jaccard(reg_s, reg_t),
            }
        entry["F3"] = profile_stats(unit_rows(p_s), unit_rows(p_t), n_perm=n_perm, seed=seed)
        cls_s = (src["cls"][i] - mu_s) / c_s
        cls_t = (tgt["cls"][j] - mu_t) / c_t
        entry["F4"] = profile_stats(cls_s, cls_t, n_perm=n_perm, seed=seed)
        profile.append(entry)
    return {"per_model": per_model, "profile": profile}


def run(cfg_in: Mapping[str, Any]) -> dict[str, Any]:
    from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
    from ...rebase.methods._ariadne.capture import paired_calibration
    from ...run_logging import _code_fingerprint
    from ..datasets.vision8_14_20 import SUITES
    from ..vision_rebase.context import _build_direct_residual_calibration

    cfg = {**DEFAULTS, **dict(cfg_in)}
    out_dir = Path(cfg["out_dir"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"out_dir {out_dir} is not empty (write-once)")
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = SUITES[str(cfg["suite"])]
    loader_cfg = {k: cfg[k] for k in ("batch_size", "num_workers", "val_fraction", "seed")}
    builds = {
        role: OpenClipBuildConfig(model_name=m["model"], pretrained=m["pretrained"], device=str(cfg["device"]))
        for role, m in (("source", cfg["source"]), ("target", cfg["target"]))
    }
    clf = {role: OpenClipClassifier.build(b) for role, b in builds.items()}
    pool_ctx, pool_meta = _build_direct_residual_calibration(
        "tiny_imagenet",
        per_task=[],
        suite=suite,
        cfg=loader_cfg,
        clf_source=clf["source"],
        clf_target=clf["target"],
        source_cfg=builds["source"],
        target_cfg=builds["target"],
        num_batches=int(cfg["anchor_batches"]) + int(cfg["registration_batches"]),
        calibration_seed=int(cfg["pool_seed"]),
    )
    n_batches = int(cfg["anchor_batches"]) + int(cfg["registration_batches"])
    src_batches, tgt_batches, draw = paired_calibration(
        pool_ctx.source_loaders.train, pool_ctx.loaders.train, num_batches=n_batches, seed=int(cfg["pool_seed"])
    )
    n_anchor_images = sum(int(b[0].shape[0]) for b in src_batches[: int(cfg["anchor_batches"])])
    rows = frame_anchor_rows(
        n_anchor_images, int(cfg["canonical_tokens"]), int(cfg["num_anchors"]), int(cfg["anchor_seed"])
    )
    stats = {}
    for role, batches in (("source", src_batches), ("target", tgt_batches)):
        started = time.perf_counter()
        stats[role] = model_statistics(clf[role].model, batches, cfg, rows, str(cfg["device"]))
        print(f"[anchors_artifacts] {role}: {time.perf_counter() - started:.1f}s", flush=True)
        clf[role].model.cpu()
    started = time.perf_counter()
    result = analyse(stats["source"], stats["target"], cfg)
    print(f"[anchors_artifacts] analysis: {time.perf_counter() - started:.1f}s", flush=True)
    payload = {
        "config": cfg,
        "pool": {**pool_meta, "indices_head": list(draw["indices"][:8]), "n_anchor_images": n_anchor_images},
        "code": _code_fingerprint(),
        **result,
    }
    (out_dir / "anchors_artifacts.json").write_text(json.dumps(payload, indent=1, default=str))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
