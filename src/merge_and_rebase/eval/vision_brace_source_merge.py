"""Transport-free, source-space BRACE task-arithmetic evaluation.

The two artifact banks are the only experimental inputs.  A single expanded
base is mounted with sums of their saved source-space task vectors; consequently
transport is deliberately absent from this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from merge_and_rebase.utils.helpers import load_json, parse_csv
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import resolve_eval_split_loader
from ..io.ckpt import load_into_model
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from .datasets.vision8_14_20 import SUITES
from .block_extension import resolve_block_extension_config, run_block_extension
from .vision_brace_tv_swap import (
    consensus_base, reconstruct_endpoint, state_dict_sha256,
    validate_artifact_bank, _task_loader_context,
)

ARMS = ("shared", "independent", "independent_shared_norm", "shared_independent_norm",
        "midpoint_raw", "midpoint_shared_norm")


def _validate_vector(v: Mapping[str, torch.Tensor], *, name: str = "vector") -> None:
    if not isinstance(v, Mapping) or not v:
        raise ValueError(f"{name} must be a non-empty mapping.")
    keys = list(v)
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError(f"{name} keys must be unique and sorted.")
    for key, value in v.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            raise ValueError(f"{name} tensor '{key}' must be floating point.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} tensor '{key}' contains NaN or Inf.")


def tensor_stats(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], *, allow_zero: bool = True) -> dict[str, float]:
    """Per-tensor, float64 norm/dot diagnostics; inputs are never modified."""
    _validate_vector(a, name="a"); _validate_vector(b, name="b")
    if set(a) != set(b):
        raise ValueError("Vector keyspaces differ.")
    aa = bb = dot = 0.0
    for key in sorted(a):
        x, y = a[key].detach().to(device="cpu", dtype=torch.float64), b[key].detach().to(device="cpu", dtype=torch.float64)
        if x.shape != y.shape:
            raise ValueError(f"Vector shape mismatch for '{key}'.")
        aa += float(torch.sum(x * x).item()); bb += float(torch.sum(y * y).item())
        dot += float(torch.sum(x * y).item())
    na, nb = math.sqrt(aa), math.sqrt(bb)
    if not allow_zero and (na == 0 or nb == 0):
        raise ValueError("Zero vector norm is not allowed for norm control.")
    return {"norm_a": na, "norm_b": nb, "dot": dot, "cosine": dot / (na * nb) if na and nb else 0.0}


def _scale(v: Mapping[str, torch.Tensor], factor: float) -> dict[str, torch.Tensor]:
    return {k: x.detach().clone().float() * float(factor) for k, x in v.items()}


def _add(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if set(a) != set(b): raise ValueError("Vector keyspaces differ.")
    return {k: a[k].detach().clone().float() + b[k].detach().float() for k in a}


def _unit(v: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    n = tensor_stats(v, v)["norm_a"]
    return _scale(v, 1.0 / n)


def arm_vector(independent: Mapping[str, torch.Tensor], shared: Mapping[str, torch.Tensor], arm: str) -> dict[str, torch.Tensor]:
    """Construct one arm while preserving source-space task-vector semantics."""
    if arm not in ARMS: raise ValueError(f"Unknown arm '{arm}'.")
    tensor_stats(independent, shared)
    if arm == "independent": return {k: v.detach().clone() for k, v in independent.items()}
    if arm == "shared": return {k: v.detach().clone() for k, v in shared.items()}
    ni, ns = tensor_stats(independent, independent)["norm_a"], tensor_stats(shared, shared)["norm_a"]
    if arm == "independent_shared_norm":
        if ni == 0: raise ValueError("Zero vector norm is not allowed for norm control.")
        return _scale(independent, ns / ni)
    if arm == "shared_independent_norm":
        if ns == 0: raise ValueError("Zero vector norm is not allowed for norm control.")
        return _scale(shared, ni / ns)
    if arm == "midpoint_raw": return _scale(_add(independent, shared), 0.5)
    # Raw midpoint direction, with the shared vector's original magnitude.
    raw_mid = _scale(_add(independent, shared), 0.5)
    nm = tensor_stats(raw_mid, raw_mid)["norm_a"]
    if nm == 0: raise ValueError("Zero midpoint norm is not allowed for norm control.")
    return _scale(raw_mid, ns / nm)


def _validate_banks(cfg: Mapping[str, Any], tasks: list[str]) -> tuple[dict, dict, dict[str, Any]]:
    roots = cfg.get("artifact_capture_root")
    if isinstance(roots, (str, os.PathLike)):
        parent = Path(roots)
        roots = {name: str(parent / name) for name in ("independent", "shared")}
    if not isinstance(roots, Mapping) or not roots.get("independent") or not roots.get("shared"):
        raise ValueError("artifact_capture_root must contain independent and shared paths.")
    for name in ("independent", "shared"):
        root = Path(roots[name]); manifest = root / "artifact_manifest.json"
        if not (root / "COMPLETE").is_file() or not manifest.is_file():
            raise FileNotFoundError(f"Incomplete artifact bank: {root}")
    banks = {name: validate_artifact_bank(Path(roots[name]), tasks) for name in ("independent", "shared")}
    # Manifest and calibration/structure compatibility is part of the causal gate.
    manifests = [json.loads((Path(roots[n]) / "artifact_manifest.json").read_text()) for n in ("independent", "shared")]
    for key in ("source_model", "source_pretrained", "source_depth", "target_depth", "tasks"):
        if manifests[0].get(key) != manifests[1].get(key): raise ValueError(f"Bank mismatch in manifest field '{key}'.")
    for task in tasks:
        ma, mb = banks["independent"][task]["metadata"], banks["shared"][task]["metadata"]
        for key in ("source_depth", "target_depth", "calibration"):
            if ma.get(key) != mb.get(key): raise ValueError(f"Bank mismatch for {task} in '{key}'.")
        params = []
        for metadata in (ma, mb):
            value = dict(metadata.get("block_extension_params", {}))
            value.pop("lmc_mode", None)
            value.pop("skip_correction", None)
            params.append(value)
        if params[0] != params[1]:
            raise ValueError(f"Bank mismatch for {task} in 'block_extension_params'.")
        if ma.get("base_sha256") != mb.get("base_sha256") or state_dict_sha256(banks["independent"][task]["base"]) != state_dict_sha256(banks["shared"][task]["base"]):
            raise ValueError(f"Independent/Shared base hashes differ for {task}.")
    return banks["independent"], banks["shared"], {"independent": manifests[0], "shared": manifests[1]}


def _alphas(cfg: Mapping[str, Any]) -> list[float]:
    vals = cfg.get("alpha_values")
    if vals is None:
        vals = [float(cfg.get("alpha_min", 0.0)) + i * float(cfg.get("alpha_step", .1)) for i in range(int(round((float(cfg.get("alpha_max", 10.0))-float(cfg.get("alpha_min", 0.0)))/float(cfg.get("alpha_step", .1))))+1)]
    vals = [float(x) for x in vals]
    if any(not math.isfinite(x) or x < 0 for x in vals):
        raise ValueError("alpha_values must be finite and nonnegative")
    if vals != sorted(set(vals)):
        raise ValueError("alpha_values must be sorted and unique")
    if 0.0 not in vals or 1.0 not in vals: raise ValueError("alpha_values must include exact 0 and 1")
    return vals


def _template(clf: OpenClipClassifier, depth: int, device: str):
    base = deepcopy(clf.model); ft = deepcopy(clf.model)
    _, sc = resolve_block_extension_config({"block_extension_enabled": True, "block_extension_params": {"target_layers_total": depth, "extension_strategy": "duplicate_per_weight", "skip_correction": True, "lmc_mode": "independent", "verbose": False, "show_progress": False}})
    run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=(),
        target_layers_total=depth,
        config=sc,
        device=device,
    )
    return base


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    suite = SUITES[str(cfg.get("suite", "vision8"))]; raw = cfg.get("tasks", "all")
    tasks = list(suite.tasks) if raw == "all" else parse_csv(str(raw))
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - set(suite.tasks): raise ValueError("Invalid or duplicate tasks.")
    ind, shared, manifests = _validate_banks(cfg, tasks)
    alphas = _alphas(cfg); arms = [cfg["arm"]] if cfg.get("arm") else list(ARMS)
    if any(a not in ARMS for a in arms): raise ValueError("Unknown arm.")
    # All arms use exactly the same SH base, including named recipient bases.
    bases = {t: shared[t]["base"] for t in tasks}
    recipient = str(cfg.get("recipient_base", "consensus"));
    base = consensus_base(bases)[0] if recipient == "consensus" else bases.get(recipient)
    if base is None: raise ValueError(f"Unknown recipient base task '{recipient}'.")
    build = OpenClipBuildConfig(model_name=str(cfg.get("source_clip_model", "ViT-B-16")), pretrained=str(cfg.get("source_clip_pretrained", "datacomp_xl_s13b_b90k")), device=str(cfg.get("device", "cuda")), dtype=cfg.get("dtype"))
    clf = OpenClipClassifier.build(build); depth = int(manifests["shared"]["target_depth"]); model = _template(clf, depth, build.device)
    template_keys = set(k for k, v in model.state_dict().items() if k.startswith("visual.") and torch.is_tensor(v))
    if template_keys != set(base): raise ValueError("Expanded template visual keyspace differs from artifact bank.")
    for key, value in base.items():
        if tuple(value.shape) != tuple(model.state_dict()[key].shape): raise ValueError(f"Expanded template shape mismatch for '{key}'.")
    eval_clf = OpenClipClassifier(model=model, tokenizer=clf.tokenizer, preprocess=clf.preprocess, normalize=clf.normalize, logit_scale=clf.logit_scale)
    curves, loaders, text = {}, {}, {}
    first_n = cfg.get("eval_first_n_batches")
    for task in tasks:
        loaders[task], names, task_cfg = _task_loader_context(task=task, suite=suite, clf=clf, build_cfg=build, cfg=cfg)
        eval_clf.build_zeroshot_text_features(names, task_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False); text[task] = eval_clf._zs_text_features.detach().clone()
    def ld(task, split):
        x = resolve_eval_split_loader(loaders[task], split)
        return itertools.islice(iter(x), max(1, int(first_n))) if first_n is not None else x
    def score(state, task, split): load_into_model(model, dict(state), strict=False); eval_clf._zs_text_features = text[task]; return float(eval_clf.top1(ld(task, split), build.device))
    # Validation curves are complete before any test evaluation.
    vectors_by_arm = {
        arm: {t: arm_vector(ind[t]["tv"], shared[t]["tv"], arm) for t in tasks}
        for arm in arms
    }
    merged_by_arm = {}
    for arm in arms:
        vectors = vectors_by_arm[arm]
        merged = {}
        for t in tasks:
            merged = _add(merged, vectors[t]) if merged else {k: v.detach().clone() for k, v in vectors[t].items()}
        merged_by_arm[arm] = merged
        curves[arm] = {t: {str(a): score(reconstruct_endpoint(base, merged, a), t, "val") for a in alphas} for t in tasks}
    macro = {arm: {str(a): sum(curves[arm][t][str(a)] for t in tasks)/len(tasks) for a in alphas} for arm in arms}
    selected = {arm: max(alphas, key=lambda a: (macro[arm][str(a)], -a)) for arm in arms}
    results = {"status": "success", "arm": arms[0] if len(arms) == 1 else arms, "val_curves": curves,
               "macro_val_curves": macro, "selected_alpha": selected[arms[0]] if len(arms) == 1 else selected,
               "alpha_values": alphas, "validation_curve": [{"alpha": a, "mean": macro[arms[0]][str(a)],
               "per_task": {t: curves[arms[0]][t][str(a)] for t in tasks}} for a in alphas],
               "test": {}, "test_selected": {}, "test_fixed": {}, "arm_stats": {}}
    for arm in arms:
        results["test"][arm] = {}
        for task in tasks:
            v = vectors_by_arm[arm][task]; own = v
            merged_vector = merged_by_arm[arm]
            results["test"][arm][task] = {}
            for label, alpha in (("base0", 0.0), ("fixed_alpha1", 1.0), ("selected", selected[arm])):
                merged = score(reconstruct_endpoint(base, merged_vector, alpha), task, "test")
                isolated = score(reconstruct_endpoint(base, own, alpha), task, "test")
                results["test"][arm][task][label] = {"merged": merged, "isolated": isolated, "merged_minus_isolated": merged-isolated}
            results["arm_stats"][arm] = results["arm_stats"].get(arm, {})
            i_tv, s_tv = ind[task]["tv"], shared[task]["tv"]
            eta = _add(i_tv, _scale(s_tv, -1.0))
            pair = tensor_stats(i_tv, s_tv)
            eta_norm = tensor_stats(eta, eta)["norm_a"]
            shared_norm = pair["norm_b"]
            projection = pair["dot"] / shared_norm if shared_norm else 0.0
            parallel = projection - shared_norm
            orthogonal_sq = max(0.0, eta_norm * eta_norm - parallel * parallel)
            results["arm_stats"][arm][task] = {
                "ind_shared": pair,
                "vector_norm": tensor_stats(v, v)["norm_a"],
                "eta_norm": eta_norm,
                "eta_relative_to_shared": eta_norm / shared_norm if shared_norm else None,
                "eta_parallel_to_shared": parallel,
                "eta_orthogonal_to_shared": math.sqrt(orthogonal_sq),
            }
        # Compact schema consumed by campaign aggregators.
        selected_label = "selected"
        fixed_label = "fixed_alpha1"
        for out_name, label in (("test_selected", selected_label), ("test_fixed", fixed_label)):
            rows = results["test"][arm]
            results[out_name] = {"per_task_merged": {t: rows[t][label]["merged"] for t in tasks},
                                 "per_task_isolated": {t: rows[t][label]["isolated"] for t in tasks},
                                 "per_task_baseline": {t: rows[t]["base0"]["merged"] for t in tasks},
                                 "per_task_merged_minus_isolated": {t: rows[t][label]["merged_minus_isolated"] for t in tasks}}
            results[out_name]["mean_merged"] = sum(results[out_name]["per_task_merged"].values()) / len(tasks)
            results[out_name]["mean_isolated"] = sum(results[out_name]["per_task_isolated"].values()) / len(tasks)
            results[out_name]["mean_baseline"] = sum(results[out_name]["per_task_baseline"].values()) / len(tasks)
    results.update({"transport_method": "none", "merge": "task_arithmetic", "tasks": tasks, "recipient_base": recipient, "recipient_base_sha256": state_dict_sha256(base), "input_hashes": {"independent": {t: {"base": state_dict_sha256(ind[t]["base"]), "ft": state_dict_sha256(ind[t]["ft"]), "tv": state_dict_sha256(ind[t]["tv"])} for t in tasks}, "shared": {t: {"base": state_dict_sha256(shared[t]["base"]), "ft": state_dict_sha256(shared[t]["ft"]), "tv": state_dict_sha256(shared[t]["tv"])} for t in tasks}}, "resolved_config": cfg})
    return results


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--device", default=None)
    args = p.parse_args(); cfg = load_json(args.config); cfg["device"] = args.device or cfg.get("device", "cuda")
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to use non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True); target = out / "summary.json"
    try:
        result = run(cfg); target.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        (out / "failure.json").write_text(json.dumps({"error": repr(exc), "resolved_config": cfg}, indent=2) + "\n")
        raise

if __name__ == "__main__": main()
