"""Block-end drift of a saved transported update: realized change vs desired ``D_j`` at every target block.

For a run that compiled per-block desired changes ``D_j`` into ``mlp.c_proj`` updates ``tau`` (Ariadne or
``anchors``), this measures on held-out paired images (the task's ``val`` split, disjoint from the fit's train
draw) what the target with ``alpha * tau`` mounted actually produces at each block boundary:

- ``realized_j = out_j(B + alpha tau) - out_j(B)``, the cumulative change at the end of block ``j``;
- ``desired_j = D_j``, recomputed for the same images exactly as the method defines it (Ariadne: ``(f - b) Q_j``
  with the fit's Procrustes maps; ``anchors``: the anchor decode with the run's frame and decode options).

Per block it reports the norm ratio ``||realized|| / ||desired||``, the cosine and the relative error
``||realized - desired|| / ||desired||`` (sums over all images, tokens and dimensions). Size and direction are
reported separately because the relative error alone rewards small updates when the cosine is low. With an
independent fit each block is regressed on the cumulative ``D_j`` while upstream blocks also move boundary
``j``, so the ratio is expected above 1; a sequential fit targets ``D_j - U_j`` and should stay near 1.

Usage::

    python -m merge_and_rebase.eval.diagnostics.depth_drift --config <drift.json>

with ``{"run_config": <pipeline config of the run>, "tv_dir": <dir of {task}_{method}_transported_native.pt>,
"tasks": [...], "alphas": {"<task>": [1.0, 0.32], ...}, "num_batches": 4, "seed": 0, "out": <json path>}``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

Tensor = torch.Tensor


class DriftAccumulator:
    """Streaming ``||r||^2``, ``||d||^2``, ``<r, d>`` of realized ``r`` and desired ``d`` (float64)."""

    def __init__(self) -> None:
        self.rr = self.dd = self.rd = 0.0

    def update(self, realized: Tensor, desired: Tensor) -> None:
        r = realized.double().reshape(-1)
        d = desired.double().reshape(-1)
        self.rr += float(r @ r)
        self.dd += float(d @ d)
        self.rd += float(r @ d)

    def summary(self) -> dict[str, float]:
        nr, nd = self.rr**0.5, self.dd**0.5
        err = max(self.rr + self.dd - 2.0 * self.rd, 0.0) ** 0.5
        return {
            "realized_norm": nr,
            "desired_norm": nd,
            "norm_ratio": nr / nd if nd > 0 else float("nan"),
            "cosine": self.rd / (nr * nd) if nr * nd > 0 else float("nan"),
            "relative_error": err / nd if nd > 0 else float("nan"),
        }


def _desired_fn(method: str, params: Mapping[str, Any], prepared_q, source_reg, target_reg):
    """``(pos, i, b, f, out) -> D_pos`` for one batch, as the method defines ``D``."""
    from ...rebase.methods._ariadne.layouts import _aligned

    if method == "ariadne":

        def ariadne(pos, i, b, f, out):
            return (_aligned([f], [out])[0] - _aligned([b], [out])[0]) @ prepared_q[pos]

        return ariadne
    from dataclasses import replace

    from ...rebase.methods.anchors import parse_anchors_config
    from ...rebase.methods.anchors.method import SequentialDecoder

    config = replace(parse_anchors_config(params), coverage_diagnostics=False)
    return SequentialDecoder(source_reg, target_reg, config)


def run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from ...io.ckpt import load_ckpt, load_into_model
    from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
    from ...rebase.discrete_layer_match import DiscreteLayerPairing
    from ...rebase.methods._ariadne.capture import iter_capture_tokens, paired_calibration
    from ...rebase.methods._ariadne.streaming import prepare_direct_residual_streaming
    from ...run_logging import _code_fingerprint
    from ..datasets.vision8_14_20 import SUITES
    from ..vision_rebase.context import _build_task_context

    out_path = Path(cfg["out"])
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists (write-once)")
    run_cfg = json.loads(Path(cfg["run_config"]).read_text())
    method = str(run_cfg["method"]["name"])
    params = dict(run_cfg["method"].get("params", {}))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    suite = SUITES[str(run_cfg["suite"])]
    loader_cfg = {
        "batch_size": run_cfg.get("batch_size", 32),
        "num_workers": run_cfg.get("num_workers", 0),
        "val_fraction": run_cfg.get("val_fraction", 0.1),
        "seed": run_cfg.get("seed", 89),
    }
    models = run_cfg["models"]
    builds = {
        role: OpenClipBuildConfig(
            model_name=models[role]["model"], pretrained=models[role]["pretrained"], device=device
        )
        for role in ("source", "target")
    }
    clf_s = OpenClipClassifier.build(builds["source"])
    clf_t = OpenClipClassifier.build(builds["target"])
    target_base_sd = {k: v.detach().cpu().clone() for k, v in clf_t.model.state_dict().items()}
    depth_s = len(clf_s.model.visual.transformer.resblocks)
    depth_t = len(clf_t.model.visual.transformer.resblocks)
    pairing = DiscreteLayerPairing.compute(depth_s, depth_t)

    source_reg = target_reg = None
    if method == "anchors":
        from ...rebase.methods.anchors import AnchorState, parse_anchors_config

        anchors_cfg = parse_anchors_config(params)
        state = AnchorState(anchors_cfg, anchors_cfg.state_dir)
        source_reg = state.model(f"{models['source']['model']}__{models['source']['pretrained']}")
        target_reg = state.model(f"{models['target']['model']}__{models['target']['pretrained']}")

    results: dict[str, Any] = {}
    for task in cfg["tasks"]:
        started = time.perf_counter()
        ctx = _build_task_context(
            task,
            suite=suite,
            cfg=loader_cfg,
            clf_target=clf_t,
            clf_source=clf_s,
            source_cfg=builds["source"],
            target_cfg=builds["target"],
            use_humanized_classnames=False,
            need_source_loaders=True,
        )
        source_base = deepcopy(clf_s.model)
        source_ft = deepcopy(clf_s.model)
        load_into_model(source_ft, load_ckpt(str(run_cfg["models"]["tuned_checkpoints"][task])), strict=False)
        prepared_q = None
        if method == "ariadne":
            # The fit's own Procrustes maps: same train draw (num_batches, seed) as the run.
            prepared = prepare_direct_residual_streaming(
                source_base,
                clf_t.model,
                ctx.source_loaders.train,
                ctx.loaders.train,
                pairing,
                num_batches=int(params.get("num_batches", 10)),
                seed=int(params.get("seed", 89)),
                device=device,
                source_ft_model=source_ft,
            )
            prepared_q = prepared["q_by_position"]
            del prepared
        desired_fn = _desired_fn(method, params, prepared_q, source_reg, target_reg)
        source_batches, target_batches, _ = paired_calibration(
            ctx.source_loaders.val, ctx.loaders.val, num_batches=int(cfg["num_batches"]), seed=int(cfg["seed"])
        )
        tv_path = Path(cfg["tv_dir"]) / f"{task}_{method}_transported_native.pt"
        tv = load_ckpt(str(tv_path))
        src_requests = {str(i): (i, "boundary") for i in sorted(set(pairing.pairing))}
        tgt_requests = {str(j): (j, "boundary") for j in range(depth_t)}
        per_alpha: dict[str, Any] = {}
        for alpha in cfg["alphas"][task]:
            edited = deepcopy(clf_t.model)
            sd = {k: v.clone() for k, v in target_base_sd.items()}
            for key, value in tv.items():
                sd[key] = sd[key] + float(alpha) * value.to(sd[key].dtype)
            edited.load_state_dict(sd, strict=True)
            accs = {j: DriftAccumulator() for j in range(depth_t)}
            gens = (
                iter_capture_tokens(source_base, source_batches, src_requests, device),
                iter_capture_tokens(source_ft, source_batches, src_requests, device),
                iter_capture_tokens(clf_t.model, target_batches, tgt_requests, device),
                iter_capture_tokens(edited, target_batches, tgt_requests, device),
            )
            for sb, sf, tb, te in zip(*gens, strict=True):
                for j in range(depth_t):
                    i = int(pairing.pairing[j])
                    out = tb[str(j)]
                    desired = desired_fn(j, i, sb[str(i)], sf[str(i)], out)
                    accs[j].update(te[str(j)] - out, desired)
            per_alpha[f"{float(alpha):g}"] = {str(j): acc.summary() for j, acc in accs.items()}
            del edited
        results[task] = {"tv": str(tv_path), "per_alpha": per_alpha, "seconds": time.perf_counter() - started}
        print(f"[depth_drift] {task}: {time.perf_counter() - started:.1f}s", flush=True)
        del source_base, source_ft
    payload = {"config": dict(cfg), "method": method, "params": params, "code": _code_fingerprint(), "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, default=str))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
