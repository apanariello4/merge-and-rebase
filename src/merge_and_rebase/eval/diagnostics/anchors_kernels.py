"""M2 of the shared-anchor method: do two models "speak the same language"? Forward passes only.

For every pool (``tiny_imagenet`` / ``vision8_mix``) the models are registered ONCE with ``T_max`` anchors
(``AnchorState.register_model``, the same code ``anchors`` runs) and every smaller ``T`` is a seeded subset of
those anchor rows (same frame, same mu and c). For each pair (A, B), target block ``j`` and its relatively
paired source block ``i = round(j (L_A - 1) / (L_B - 1))``:

- ``cka`` (linear CKA of the centred kernels) and ``rel_err = ||K_A - K_B||_F / ||K_A||_F``;
- **null**: CKA against ``Pi K_B Pi^T`` for random anchor permutations ``Pi`` (the value two unrelated geometries of
  the same spectra reach), mean and 95th percentile;
- **sampling noise**: bootstrap over anchors (95% interval of cka and rel_err) and the CKA on two disjoint halves;
- **conditioning** of K_A and K_B (condition number, participation-ratio effective rank, numerical rank);
- at ``T_max``, the full ``L_A x L_B`` CKA matrix (is the maximum on the relative-depth diagonal?).

Run::

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m merge_and_rebase.eval.diagnostics.anchors_kernels \\
        --config configs/anchors/m2_kernels_b16_l14.json

Config keys: ``models`` (list of ``{id, model, pretrained}``), ``pairs`` (list of ``[source_id, target_id]``),
``pools``, ``num_anchors`` (list of T; the largest defines the frame), ``anchor_batches``, ``registration_batches``,
``pool_seed``, ``anchor_seed``, ``canonical_tokens``, ``anchor_positions``, ``batch_size``, ``num_workers``,
``val_fraction``, ``seed``, ``suite``, ``device``, ``dtype``, ``n_perm``, ``n_boot``, ``analysis_seed``,
``state_root`` (one frame per pool under ``<state_root>/<pool>_T<T_max>``) and ``out_dir``.
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
from ...rebase.methods.anchors import AnchorsConfig, AnchorState, RegisteredModel
from ...rebase.methods.anchors.core import kernel, kernel_diagnostics, kernel_spectrum

Tensor = torch.Tensor

DEFAULTS: dict[str, Any] = {
    "pools": ["tiny_imagenet", "vision8_mix"],
    "num_anchors": [128, 512],
    "anchor_batches": 4,
    "registration_batches": 10,
    "pool_seed": 0,
    "anchor_seed": 0,
    "canonical_tokens": 197,
    "anchor_positions": "all",
    "batch_size": 32,
    "num_workers": 0,
    "val_fraction": 0.1,
    "seed": 89,
    "suite": "vision8",
    "device": "cuda",
    "dtype": None,
    "n_perm": 100,
    "n_boot": 200,
    "analysis_seed": 0,
}


# ---- pure analysis (unit-tested without models) -------------------------------------------------------------------


def anchor_subset(t_max: int, t: int, seed: int) -> Tensor:
    """Sorted indices of a seeded size-``t`` subset of the ``t_max`` frame anchors (all of them when ``t == t_max``)."""
    if not 1 <= t <= t_max:
        raise ValueError(f"T={t} must be in [1, {t_max}]")
    if t == t_max:
        return torch.arange(t_max)
    perm = torch.randperm(t_max, generator=torch.Generator().manual_seed(int(seed)))
    return torch.sort(perm[:t]).values


def _quantiles(values: Tensor) -> tuple[float, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return float(torch.quantile(finite, 0.025)), float(torch.quantile(finite, 0.975))


def kernel_pair_stats(k_a: Tensor, k_b: Tensor, *, n_perm: int, n_boot: int, seed: int) -> dict[str, Any]:
    """Point CKA / rel_err of two kernels on the same anchors, with permutation null, bootstrap and split halves."""
    k_a = k_a.to(torch.float64)
    k_b = k_b.to(torch.float64)
    t = k_a.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    point = kernel_diagnostics(k_a, k_b)
    null = torch.empty(n_perm, dtype=torch.float64)
    for p in range(n_perm):
        pi = torch.randperm(t, generator=gen)
        null[p] = kernel_diagnostics(k_a, k_b[pi][:, pi])["cka"]
    boot_cka = torch.empty(n_boot, dtype=torch.float64)
    boot_err = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, t, (t,), generator=gen)
        d = kernel_diagnostics(k_a[idx][:, idx], k_b[idx][:, idx])
        boot_cka[b] = d["cka"]
        boot_err[b] = d["rel_err"]
    halves = torch.randperm(t, generator=gen)
    h1, h2 = halves[: t // 2], halves[t // 2 :]
    cka_lo, cka_hi = _quantiles(boot_cka)
    err_lo, err_hi = _quantiles(boot_err)
    return {
        "T": int(t),
        "cka": point["cka"],
        "rel_err": point["rel_err"],
        "cka_boot_lo": cka_lo,
        "cka_boot_hi": cka_hi,
        "rel_err_boot_lo": err_lo,
        "rel_err_boot_hi": err_hi,
        "cka_null_mean": float(null.mean()) if n_perm else float("nan"),
        "cka_null_q95": float(torch.quantile(null, 0.95)) if n_perm else float("nan"),
        "cka_half_1": kernel_diagnostics(k_a[h1][:, h1], k_b[h1][:, h1])["cka"],
        "cka_half_2": kernel_diagnostics(k_a[h2][:, h2], k_b[h2][:, h2])["cka"],
        **{f"source_{k}": v for k, v in kernel_spectrum(k_a).items()},
        **{f"target_{k}": v for k, v in kernel_spectrum(k_b).items()},
    }


def _kernels(reg: RegisteredModel, idx: Tensor) -> list[Tensor]:
    return [kernel(reg.p[j].to(torch.float64)[idx]) for j in range(reg.depth)]


def pair_profile(
    source: RegisteredModel,
    target: RegisteredModel,
    *,
    t_values: Sequence[int],
    n_perm: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Per-T, per-target-block statistics for one pair, plus the T_max cross-depth CKA matrix."""
    t_max = int(source.p[0].shape[0])
    if int(target.p[0].shape[0]) != t_max:
        raise ValueError("source and target were registered with different anchor counts")
    pairing = DiscreteLayerPairing.compute(source.depth, target.depth)
    by_t: dict[str, list[dict[str, Any]]] = {}
    for t in sorted(set(int(v) for v in t_values)):
        idx = anchor_subset(t_max, t, seed)
        k_src, k_tgt = _kernels(source, idx), _kernels(target, idx)
        rows = []
        for j in range(target.depth):
            i = int(pairing.pairing[j])
            stats = kernel_pair_stats(k_src[i], k_tgt[j], n_perm=n_perm, n_boot=n_boot, seed=seed + 1000 * j + t)
            rows.append(
                {
                    "target_block": j,
                    "source_block": i,
                    "s": j / max(target.depth - 1, 1),
                    "c_source": source.c[i],
                    "c_target": target.c[j],
                    **stats,
                }
            )
        by_t[str(t)] = rows
    full = torch.arange(t_max)
    k_src, k_tgt = _kernels(source, full), _kernels(target, full)
    matrix = [[kernel_diagnostics(k_src[i], k_tgt[j])["cka"] for j in range(target.depth)] for i in range(source.depth)]
    argmax_source = [max(range(source.depth), key=lambda i, j=j: matrix[i][j]) for j in range(target.depth)]
    return {
        "pairing": list(pairing.pairing),
        "by_T": by_t,
        "cross_depth_cka": matrix,
        "argmax_source_block_per_target_block": argmax_source,
        "relative_pairing_is_argmax_fraction": sum(
            int(argmax_source[j] == pairing.pairing[j]) for j in range(target.depth)
        )
        / target.depth,
    }


# ---- driver -------------------------------------------------------------------------------------------------------


def _frame_config(cfg: Mapping[str, Any], pool: str, state_dir: Path) -> AnchorsConfig:
    return AnchorsConfig(
        anchor_pool=pool,
        num_anchors=max(int(t) for t in cfg["num_anchors"]),
        anchor_batches=int(cfg["anchor_batches"]),
        registration_batches=int(cfg["registration_batches"]),
        pool_seed=int(cfg["pool_seed"]),
        anchor_seed=int(cfg["anchor_seed"]),
        canonical_tokens=int(cfg["canonical_tokens"]),
        anchor_positions=str(cfg["anchor_positions"]),
        state_dir=str(state_dir),
    )


def _register_pool(cfg: Mapping[str, Any], pool: str, state: AnchorState, models: Mapping[str, Any]) -> dict:
    """Register every model of every pair under ``pool``, building the pool loaders exactly as the pipeline does."""
    from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
    from ..datasets.vision8_14_20 import SUITES
    from ..vision_rebase.context import _build_direct_residual_calibration, _build_task_context

    suite = SUITES[str(cfg["suite"])]
    loader_cfg = {k: cfg[k] for k in ("batch_size", "num_workers", "val_fraction", "seed")}
    builds = {
        mid: OpenClipBuildConfig(
            model_name=m["model"], pretrained=m["pretrained"], device=str(cfg["device"]), dtype=cfg["dtype"]
        )
        for mid, m in models.items()
    }
    meta: dict[str, Any] = {}
    for source_id, target_id in cfg["pairs"]:
        if state.has_model(source_id) and state.has_model(target_id):
            continue
        clf_s = OpenClipClassifier.build(builds[source_id])
        clf_t = OpenClipClassifier.build(builds[target_id])
        per_task: list[dict[str, Any]] = []
        if pool == "vision8_mix":
            for task in suite.tasks:
                ctx = _build_task_context(
                    task,
                    suite=suite,
                    cfg=loader_cfg,
                    clf_target=clf_t,
                    clf_source=clf_s,
                    source_cfg=builds[source_id],
                    target_cfg=builds[target_id],
                    use_humanized_classnames=False,
                    need_source_loaders=True,
                )
                per_task.append(
                    {
                        "task": task,
                        "loaders": ctx.loaders,
                        "classnames": ctx.classnames,
                        "build_cfg_task": ctx.build_cfg_task,
                        "source_loaders": ctx.source_loaders,
                        "source_build_cfg_task": ctx.source_build_cfg_task,
                    }
                )
        pool_ctx, pool_meta = _build_direct_residual_calibration(
            pool,
            per_task=per_task,
            suite=suite,
            cfg=loader_cfg,
            clf_source=clf_s,
            clf_target=clf_t,
            source_cfg=builds[source_id],
            target_cfg=builds[target_id],
            num_batches=int(cfg["anchor_batches"]) + int(cfg["registration_batches"]),
            calibration_seed=int(cfg["pool_seed"]),
        )
        meta[f"{source_id}->{target_id}"] = pool_meta
        for mid, clf, loader in (
            (source_id, clf_s, pool_ctx.source_loaders.train),
            (target_id, clf_t, pool_ctx.loaders.train),
        ):
            started = time.perf_counter()
            state.register_model(
                clf.model,
                mid,
                loader,
                device=str(cfg["device"]),
                model_meta={"model_name": models[mid]["model"], "pretrained": models[mid]["pretrained"]},
            )
            print(f"[anchors M2] {pool}: registered {mid} in {time.perf_counter() - started:.1f}s", flush=True)
        del clf_s, clf_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return meta


def _state_hashes(state_dir: Path) -> dict[str, str]:
    out = {}
    for meta in sorted(state_dir.glob("models/*/meta.json")):
        out[meta.parent.name] = json.loads(meta.read_text())["blocks_sha256"]
    return out


def plot_profiles(results: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pools = list(results)
    pairs = sorted({pair for pool in pools for pair in results[pool]["pairs"]})
    fig, axes = plt.subplots(2, max(len(pairs), 1), figsize=(6 * max(len(pairs), 1), 7), squeeze=False)
    for col, pair in enumerate(pairs):
        for pool in pools:
            profile = results[pool]["pairs"].get(pair)
            if profile is None:
                continue
            for t, rows in profile["by_T"].items():
                s = [r["s"] for r in rows]
                label = f"{pool} T={t}"
                (line,) = axes[0][col].plot(s, [r["cka"] for r in rows], marker="o", ms=3, label=label)
                axes[0][col].fill_between(
                    s,
                    [r["cka_boot_lo"] for r in rows],
                    [r["cka_boot_hi"] for r in rows],
                    alpha=0.2,
                    color=line.get_color(),
                )
                axes[0][col].plot(s, [r["cka_null_q95"] for r in rows], ls=":", color=line.get_color(), lw=1)
                axes[1][col].plot(
                    s, [r["rel_err"] for r in rows], marker="o", ms=3, color=line.get_color(), label=label
                )
                axes[1][col].fill_between(
                    s,
                    [r["rel_err_boot_lo"] for r in rows],
                    [r["rel_err_boot_hi"] for r in rows],
                    alpha=0.2,
                    color=line.get_color(),
                )
        axes[0][col].set_title(f"{pair}: kernel CKA (band: bootstrap 95%, dotted: permutation null q95)", fontsize=9)
        axes[0][col].set_ylabel("linear CKA of centred kernels")
        axes[1][col].set_ylabel("||K_A - K_B||_F / ||K_A||_F")
        axes[1][col].set_xlabel("relative depth s (target block / (L_B - 1))")
        axes[0][col].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg_in: Mapping[str, Any]) -> dict[str, Any]:
    from ...run_logging import _code_fingerprint

    cfg = {**DEFAULTS, **dict(cfg_in)}
    models = {m["id"]: m for m in cfg["models"]}
    out_dir = Path(cfg["out_dir"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"out_dir {out_dir} is not empty (write-once)")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_root = Path(cfg["state_root"])
    t_max = max(int(t) for t in cfg["num_anchors"])
    results: dict[str, Any] = {}
    for pool in cfg["pools"]:
        state_dir = state_root / f"{pool}_T{t_max}"
        frame_cfg = _frame_config(cfg, pool, state_dir)
        state = AnchorState(frame_cfg, state_dir)
        pool_meta = _register_pool(cfg, pool, state, models)
        pairs: dict[str, Any] = {}
        for source_id, target_id in cfg["pairs"]:
            started = time.perf_counter()
            pairs[f"{source_id}->{target_id}"] = pair_profile(
                state.model(source_id),
                state.model(target_id),
                t_values=cfg["num_anchors"],
                n_perm=int(cfg["n_perm"]),
                n_boot=int(cfg["n_boot"]),
                seed=int(cfg["analysis_seed"]),
            )
            print(f"[anchors M2] {pool} {source_id}->{target_id}: {time.perf_counter() - started:.1f}s", flush=True)
        results[pool] = {
            "state_dir": str(state_dir),
            "frame": state.frame["config"],
            "pool_meta": pool_meta,
            "blocks_sha256": _state_hashes(state_dir),
            "pairs": pairs,
        }
    payload = {"config": cfg, "code": _code_fingerprint(), "results": results}
    (out_dir / "anchors_kernels.json").write_text(json.dumps(payload, indent=1, default=str))
    plot_profiles(results, out_dir / "anchors_kernels_profile.png")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
