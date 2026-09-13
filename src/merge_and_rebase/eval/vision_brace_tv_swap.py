"""Source-space BRACE task-vector portability evaluation.

This entrypoint intentionally has no width transport or merge-method dependency.
It first captures task-local BRACE endpoints, then mounts each saved task vector
on task-specific or consensus extended bases.  Keeping capture and evaluation
separate makes the expensive BRACE fits reusable and prevents alpha selection
from ever changing a saved endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..cli_args import add_config_arg, add_device_dtype_args, add_suite_arg, add_tasks_arg, merge_non_none
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import humanize, resolve_eval_split_loader
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model, resolve_ckpt_path
from ..merge.methods._common import axpy_state_dict
from ..merge.task_vectors import TaskVector
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from .block_extension import resolve_block_extension_config, run_block_extension, select_loader
from .datasets.vision8_14_20 import SUITES


CONDITIONS = ("independent", "shared", "skip")
CONSENSUS_BASE = "consensus"


def _visual_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        key: value.detach().to(device="cpu").clone()
        for key, value in model.state_dict().items()
        if key.startswith("visual.") and torch.is_tensor(value)
    }
    if not state:
        raise ValueError("Expected a visual state dict, found no visual tensors.")
    return state


def _cpu_state_fp32(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy an endpoint without changing non-floating visual buffers."""

    return {
        key: value.detach().to(device="cpu", dtype=torch.float32).clone()
        if torch.is_floating_point(value)
        else value.detach().to(device="cpu").clone()
        for key, value in state.items()
    }


def _visual_float_filter(key: str, value: torch.Tensor) -> bool:
    return key.startswith("visual.") and torch.is_floating_point(value) and ".aligner." not in key


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable digest for CPU tensors; mapping insertion order is ignored."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def visual_key_fingerprint(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "key_count": len(state),
        "keys": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(state.items())
        },
    }


def task_vector_from_endpoints(
    base: Mapping[str, torch.Tensor], ft: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(base) != set(ft):
        raise ValueError("BRACE base and FT endpoint keyspaces differ.")
    delta: dict[str, torch.Tensor] = {}
    for key in sorted(base):
        if torch.is_floating_point(base[key]):
            if base[key].shape != ft[key].shape:
                raise ValueError(f"Endpoint shape mismatch for '{key}'.")
            delta[key] = ft[key].detach().float() - base[key].detach().float()
        elif not torch.equal(base[key], ft[key]):
            raise ValueError(f"Non-floating endpoint buffer differs for '{key}'.")
    if not delta:
        raise ValueError("No floating visual tensors available for task vector.")
    return delta


def reconstruct_endpoint(
    base: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor], alpha: float = 1.0
) -> dict[str, torch.Tensor]:
    return axpy_state_dict(dict(base), dict(delta), alpha=float(alpha))


def assert_endpoint_reconstruction(
    base: Mapping[str, torch.Tensor], ft: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    reconstructed = reconstruct_endpoint(base, delta, 1.0)
    if set(reconstructed) != set(ft):
        raise ValueError("Reconstructed endpoint keyspace differs from FT endpoint.")
    max_abs_error = 0.0
    max_relative_error = 0.0
    for key in ft:
        if torch.is_floating_point(ft[key]):
            actual = reconstructed[key].float()
            expected = ft[key].float()
            absolute = (actual - expected).abs()
            key_max_abs = float(absolute.max().item()) if absolute.numel() else 0.0
            denominator = expected.abs().clamp_min(1e-12)
            key_max_relative = float((absolute / denominator).max().item()) if absolute.numel() else 0.0
            max_abs_error = max(max_abs_error, key_max_abs)
            max_relative_error = max(max_relative_error, key_max_relative)
            # ``delta`` is stored as float32 after subtracting two float32
            # endpoints.  Adding it back can differ by a few float32 ulps;
            # accept that representation roundoff while retaining a strict
            # endpoint-integrity check.
            if not torch.allclose(actual, expected, rtol=1e-6, atol=2e-7):
                raise ValueError(
                    f"Task vector does not numerically reconstruct FT endpoint at '{key}' "
                    f"(max_abs_error={key_max_abs:.3e}, max_relative_error={key_max_relative:.3e})."
                )
        elif not torch.equal(reconstructed[key], ft[key]):
            raise ValueError(f"Reconstructed non-floating endpoint differs at '{key}'.")
    return {"max_abs_error": max_abs_error, "max_relative_error": max_relative_error}


def consensus_base(
    bases_by_task: Mapping[str, Mapping[str, torch.Tensor]]
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Average all floating visual tensors and reject incompatible buffers."""

    if not bases_by_task:
        raise ValueError("Cannot construct a consensus base from an empty endpoint bank.")
    tasks = sorted(bases_by_task)
    first = bases_by_task[tasks[0]]
    keyspace = set(first)
    if any(set(bases_by_task[task]) != keyspace for task in tasks[1:]):
        raise ValueError("Task-specific extended bases have different visual keyspaces.")

    consensus: dict[str, torch.Tensor] = {}
    for key in sorted(first):
        values = [bases_by_task[task][key] for task in tasks]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"Task-specific base shape mismatch for '{key}'.")
        if torch.is_floating_point(values[0]):
            consensus[key] = torch.stack([value.float() for value in values], dim=0).mean(dim=0)
        else:
            if any(not torch.equal(value, values[0]) for value in values[1:]):
                raise ValueError(f"Non-floating visual buffer differs across bases: '{key}'.")
            consensus[key] = values[0].detach().clone()

    denominator = sum(
        float(value.float().square().sum().item())
        for value in consensus.values()
        if torch.is_floating_point(value)
    )
    distances: dict[str, float] = {}
    for task in tasks:
        numerator = sum(
            float((bases_by_task[task][key].float() - consensus[key].float()).square().sum().item())
            for key in consensus
            if torch.is_floating_point(consensus[key])
        )
        distances[task] = (numerator / denominator) ** 0.5 if denominator else 0.0
    return consensus, distances


def validate_artifact_bank(root: Path, tasks: list[str]) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_dir = root / "tasks" / task
        metadata_path = task_dir / "metadata.json"
        paths = {name: task_dir / f"{name}.pt" for name in ("base", "ft", "tv")}
        if not metadata_path.is_file() or any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Incomplete endpoint artifact for task '{task}' under {root}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        base = torch.load(paths["base"], map_location="cpu", weights_only=True)
        ft = torch.load(paths["ft"], map_location="cpu", weights_only=True)
        tv = torch.load(paths["tv"], map_location="cpu", weights_only=True)
        reconstruction = assert_endpoint_reconstruction(base, ft, tv)
        if metadata.get("base_sha256") != state_dict_sha256(base):
            raise ValueError(f"Base hash mismatch for task '{task}'.")
        if metadata.get("ft_sha256") != state_dict_sha256(ft):
            raise ValueError(f"FT hash mismatch for task '{task}'.")
        if metadata.get("tv_sha256") != state_dict_sha256(tv):
            raise ValueError(f"TV hash mismatch for task '{task}'.")
        bank[task] = {"base": base, "ft": ft, "tv": tv, "metadata": metadata}
    return bank


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _resolve_tasks(cfg: Mapping[str, Any]) -> tuple[Any, list[str]]:
    suite_name = str(cfg.get("suite", "vision8"))
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'.")
    suite = SUITES[suite_name]
    raw = cfg.get("tasks", "all")
    tasks = list(suite.tasks) if raw == "all" else parse_csv(str(raw))
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must name one or more distinct suite tasks.")
    unknown = sorted(set(tasks) - set(suite.tasks))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return suite, tasks


def _task_loader_context(
    *, task: str, suite: Any, clf: OpenClipClassifier, build_cfg: OpenClipBuildConfig, cfg: Mapping[str, Any]
) -> tuple[Any, list[str], OpenClipBuildConfig]:
    hf_path, hf_config, split_map = suite.resolver(task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        ft_epochs=1,
        split_map=split_map,
        preprocess=clf.preprocess,
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", 89)),
    )
    classnames = list(loaders.classnames)
    if not bool(cfg.get("no_humanize", True)):
        classnames = [humanize(name) for name in classnames]
    templates = get_templates(task)
    if not templates:
        raise ValueError(f"get_templates('{task}') returned no prompts.")
    return loaders, classnames, OpenClipBuildConfig(
        model_name=build_cfg.model_name,
        pretrained=build_cfg.pretrained,
        device=build_cfg.device,
        dtype=build_cfg.dtype,
        prompt_templates=templates,
    )


def _condition_config(params: Mapping[str, Any], condition: str):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}.")
    raw = dict(params)
    raw["skip_correction"] = condition == "skip"
    raw["lmc_mode"] = "independent" if condition == "skip" else condition
    enabled, config = resolve_block_extension_config({"block_extension_enabled": True, "block_extension_params": raw})
    if not enabled:
        raise AssertionError("BRACE capture requires block extension.")
    return config


def capture(cfg: dict[str, Any], *, condition: str, artifact_root: Path, tasks: list[str]) -> dict[str, Any]:
    suite, resolved_tasks = _resolve_tasks({**cfg, "tasks": ",".join(tasks)})
    if resolved_tasks != tasks:
        raise AssertionError("Task resolution changed requested task order.")
    params = cfg.get("block_extension_params", {})
    if not isinstance(params, Mapping):
        raise ValueError("block_extension_params must be an object.")
    extension_cfg = _condition_config(params, condition)
    target_depth = int(cfg.get("target_layers_total", 24))
    build_cfg = OpenClipBuildConfig(
        model_name=str(cfg.get("source_clip_model", "ViT-B-16")),
        pretrained=str(cfg.get("source_clip_pretrained", "datacomp_xl_s13b_b90k")),
        device=str(cfg.get("device", "cuda")),
        dtype=cfg.get("dtype", None),
    )
    clf = OpenClipClassifier.build(build_cfg)
    source_depth = len(clf.model.visual.transformer.resblocks)
    if source_depth >= target_depth:
        raise ValueError(f"Expected extension, got source depth {source_depth} and target depth {target_depth}.")
    tuned = cfg.get("tuned_ckpts")
    if not isinstance(tuned, Mapping):
        raise ValueError("capture requires tuned_ckpts keyed by task.")
    missing = [task for task in tasks if task not in tuned]
    if missing:
        raise ValueError(f"Missing tuned checkpoints for: {missing}")
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(f"Artifact namespace is not empty: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=False)

    task_metadata: dict[str, Any] = {}
    for task in tasks:
        loaders, _, _ = _task_loader_context(task=task, suite=suite, clf=clf, build_cfg=build_cfg, cfg=cfg)
        base_model = deepcopy(clf.model)
        ft_model = deepcopy(clf.model)
        checkpoint_path = resolve_ckpt_path(str(tuned[task]))
        ft_state = align_to_base_keys(load_ckpt(checkpoint_path), _visual_state(base_model))
        if not ft_state:
            raise ValueError(f"No visual tensors from checkpoint for '{task}': {checkpoint_path}")
        load_into_model(ft_model, ft_state, strict=False)
        calibration_loader = select_loader(
            extension_cfg.calibration_split,
            train_loader=loaders.train,
            test_loader=loaders.test,
            val_loader=loaders.val,
        )
        final_depth = run_block_extension(
            source_base_model=base_model,
            source_ft_model=ft_model,
            calibration_loader=calibration_loader,
            target_layers_total=target_depth,
            config=extension_cfg,
            device=build_cfg.device,
        )
        if final_depth != target_depth:
            raise RuntimeError(f"{task}: BRACE produced depth {final_depth}, expected {target_depth}.")
        base = _cpu_state_fp32(_visual_state(base_model))
        ft = _cpu_state_fp32(_visual_state(ft_model))
        tv = task_vector_from_endpoints(base, ft)
        reconstruction = assert_endpoint_reconstruction(base, ft, tv)
        task_dir = artifact_root / "tasks" / task
        task_dir.mkdir(parents=True, exist_ok=False)
        for name, value in (("base", base), ("ft", ft), ("tv", tv)):
            torch.save(value, task_dir / f"{name}.pt")
        task_meta = {
            "task": task,
            "condition": condition,
            "checkpoint": str(checkpoint_path),
            "source_depth": source_depth,
            "target_depth": target_depth,
            "base_sha256": state_dict_sha256(base),
            "ft_sha256": state_dict_sha256(ft),
            "tv_sha256": state_dict_sha256(tv),
            "fp32_endpoint_reconstruction": reconstruction,
            "visual_key_fingerprint": visual_key_fingerprint(base),
            "block_extension_params": dict(params),
            "resolved_condition_params": {
                **dict(params), "skip_correction": extension_cfg.skip_correction, "lmc_mode": extension_cfg.lmc_mode,
            },
            "calibration": {"split": extension_cfg.calibration_split, "n_batches": extension_cfg.n_batches_act, "seed": cfg.get("seed", 89)},
        }
        _write_json_new(task_dir / "metadata.json", task_meta)
        task_metadata[task] = task_meta

    bank = validate_artifact_bank(artifact_root, tasks)
    bases = {task: bank[task]["base"] for task in tasks}
    consensus, distances = consensus_base(bases)
    torch.save(consensus, artifact_root / "consensus_base.pt")
    manifest = {
        "condition": condition,
        "tasks": tasks,
        "source_model": build_cfg.model_name,
        "source_pretrained": build_cfg.pretrained,
        "source_depth": source_depth,
        "target_depth": target_depth,
        "consensus_sha256": state_dict_sha256(consensus),
        "task_base_sha256": {task: task_metadata[task]["base_sha256"] for task in tasks},
        "distance_to_consensus": distances,
        "task_metadata": {task: str(artifact_root / "tasks" / task / "metadata.json") for task in tasks},
        "resolved_config": cfg,
    }
    _write_json_new(artifact_root / "artifact_manifest.json", manifest)
    (artifact_root / "COMPLETE").touch(exist_ok=False)
    return manifest


def _expanded_template(clf: OpenClipClassifier, target_depth: int, device: str) -> torch.nn.Module:
    base = deepcopy(clf.model)
    ft = deepcopy(clf.model)
    _, structural_cfg = resolve_block_extension_config({
        "block_extension_enabled": True,
        "block_extension_params": {
            "target_layers_total": target_depth,
            "extension_strategy": "duplicate_per_weight",
            "skip_correction": True,
            "lmc_mode": "independent",
            "verbose": False,
            "show_progress": False,
        },
    })
    run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=(),
        target_layers_total=target_depth,
        config=structural_cfg,
        device=device,
    )
    return base


def _alpha_grid(cfg: Mapping[str, Any]) -> list[float]:
    lo, hi, step = float(cfg.get("alpha_min", 0.0)), float(cfg.get("alpha_max", 10.0)), float(cfg.get("alpha_step", 0.1))
    if step <= 0 or hi < lo:
        raise ValueError("Invalid alpha range.")
    values = [round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)]
    if not any(abs(alpha) < 1e-8 for alpha in values) or not any(abs(alpha - 1.0) < 1e-8 for alpha in values):
        raise ValueError("TV-swap alpha grid must include both 0 and 1.")
    return values


@torch.no_grad()
def evaluate_donor(cfg: dict[str, Any], *, condition: str, artifact_root: Path, output_dir: Path, donor: str, tasks: list[str]) -> dict[str, Any]:
    if not (artifact_root / "COMPLETE").is_file():
        raise FileNotFoundError(f"Capture is incomplete: {artifact_root}")
    suite, resolved_tasks = _resolve_tasks({**cfg, "tasks": ",".join(tasks)})
    if donor not in resolved_tasks:
        raise ValueError(f"Donor '{donor}' is not in selected task bank.")
    bank = validate_artifact_bank(artifact_root, resolved_tasks)
    consensus_path = artifact_root / "consensus_base.pt"
    manifest_path = artifact_root / "artifact_manifest.json"
    if not consensus_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Capture lacks consensus_base.pt or artifact_manifest.json.")
    consensus = torch.load(consensus_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("condition") != condition:
        raise ValueError(
            f"Artifact condition mismatch: requested '{condition}', manifest has '{manifest.get('condition')}'."
        )
    if manifest.get("consensus_sha256") != state_dict_sha256(consensus):
        raise ValueError("Consensus base hash mismatch.")
    recipients: dict[str, dict[str, torch.Tensor]] = {task: bank[task]["base"] for task in resolved_tasks}
    recipients[CONSENSUS_BASE] = consensus
    donor_tv = bank[donor]["tv"]
    alpha_values = _alpha_grid(cfg)

    build_cfg = OpenClipBuildConfig(
        model_name=str(cfg.get("source_clip_model", "ViT-B-16")),
        pretrained=str(cfg.get("source_clip_pretrained", "datacomp_xl_s13b_b90k")),
        device=str(cfg.get("device", "cuda")), dtype=cfg.get("dtype", None),
    )
    clf_source = OpenClipClassifier.build(build_cfg)
    target_depth = int(manifest["target_depth"])
    model = _expanded_template(clf_source, target_depth, build_cfg.device)
    eval_clf = OpenClipClassifier(model=model, tokenizer=clf_source.tokenizer, preprocess=clf_source.preprocess,
                                  normalize=clf_source.normalize, logit_scale=clf_source.logit_scale)
    loaders, classnames, task_cfg = _task_loader_context(task=donor, suite=suite, clf=clf_source, build_cfg=build_cfg, cfg=cfg)
    eval_clf.build_zeroshot_text_features(classnames, task_cfg, cache_dir="src/.cache/zs_cache", force_rebuild=False)
    first_n = cfg.get("eval_first_n_batches", None)

    def loader_for(split: str) -> Any:
        loader = resolve_eval_split_loader(loaders, split)
        if first_n is not None:
            return itertools.islice(iter(loader), max(1, int(first_n)))
        return loader

    def score(state: Mapping[str, torch.Tensor], loader: Any) -> float:
        load_into_model(model, dict(state), strict=False)
        return float(eval_clf.top1(loader, device=build_cfg.device))

    rows: list[dict[str, Any]] = []
    for recipient, base in recipients.items():
        if set(base) != set(donor_tv) and not set(donor_tv).issubset(base):
            raise ValueError(f"TV/base keyspace mismatch for donor={donor}, base={recipient}.")
        val_scores: list[tuple[float, float]] = []
        for alpha in alpha_values:
            mounted = reconstruct_endpoint(base, donor_tv, alpha)
            val_scores.append((alpha, score(mounted, loader_for("val"))))
        best_alpha, best_val = max(val_scores, key=lambda pair: (pair[1], -pair[0]))
        base_val = val_scores[0][1]
        alpha_one_val = next(value for alpha, value in val_scores if abs(alpha - 1.0) < 1e-8)
        base_test = score(base, loader_for("test"))
        alpha_one_test = score(reconstruct_endpoint(base, donor_tv, 1.0), loader_for("test"))
        selected_test = score(reconstruct_endpoint(base, donor_tv, best_alpha), loader_for("test"))
        rows.append({
            "condition": condition,
            "donor_task": donor,
            "recipient_base": recipient,
            "recipient_kind": "consensus" if recipient == CONSENSUS_BASE else ("self" if recipient == donor else "cross_task"),
            "selected_alpha": best_alpha,
            "validation_selected_accuracy": best_val,
            "validation_base_accuracy": base_val,
            "validation_alpha_one_accuracy": alpha_one_val,
            "test_base_accuracy": base_test,
            "test_alpha_one_accuracy": alpha_one_test,
            "test_selected_accuracy": selected_test,
            "test_gain_over_base": selected_test - base_test,
            "artifact_root": str(artifact_root),
        })
    self_score = next(row["test_selected_accuracy"] for row in rows if row["recipient_base"] == donor)
    for row in rows:
        row["test_drop_from_donor_self"] = row["test_selected_accuracy"] - self_score
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{condition}_{donor}.json"
    _write_json_new(output_path, {"condition": condition, "donor_task": donor, "rows": rows, "alpha_protocol": {"split": "val", "values": alpha_values}, "test_split": "test"})
    return {"output": str(output_path), "rows": rows}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    add_suite_arg(parser, choices=sorted(SUITES))
    add_tasks_arg(parser, help_text="CSV Vision8 tasks or 'all'.")
    add_device_dtype_args(parser, device_default=None, dtype_default=None)
    parser.add_argument("--phase", choices=("capture", "evaluate"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--donor-task", default=None)
    parser.add_argument("--eval-first-n-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_json(args.config) if args.config else {}
    cfg = merge_non_none(cfg, {"suite": args.suite, "tasks": args.tasks, "device": args.device, "dtype": args.dtype,
                               "eval_first_n_batches": args.eval_first_n_batches})
    _, tasks = _resolve_tasks(cfg)
    logging_cfg = merge_logging_config(cfg.get("logging", {}), {})
    if not logging_cfg.get("run_name"):
        suffix = args.donor_task if args.phase == "evaluate" else "all_tasks"
        logging_cfg["run_name"] = f"brace_tv_swap_{args.phase}_{args.condition}_{suffix}"
    # Capture treats artifact_root as an immutable, initially empty namespace;
    # place runtime logs beside it so logger setup cannot pre-create that root.
    default_parent = (
        Path(args.artifact_root).parent / "run_logs"
        if args.phase == "capture"
        else (args.output_dir or args.artifact_root)
    )
    summary_path = default_summary_path(entrypoint="eval.vision_brace_tv_swap", logging_cfg=logging_cfg,
                                        default_parent=default_parent)
    logger = start_run(entrypoint="eval.vision_brace_tv_swap", logging_cfg=logging_cfg, summary_path=summary_path,
                       metadata={"resolved_config": cfg, "phase": args.phase, "condition": args.condition})
    try:
        root = Path(args.artifact_root)
        if args.phase == "capture":
            result = capture(cfg, condition=args.condition, artifact_root=root, tasks=tasks)
        else:
            if args.output_dir is None or args.donor_task is None:
                raise ValueError("evaluate phase requires --output-dir and --donor-task.")
            result = evaluate_donor(cfg, condition=args.condition, artifact_root=root, output_dir=Path(args.output_dir), donor=args.donor_task, tasks=tasks)
        logger.log_summary({"phase": args.phase, "condition": args.condition, "result": result})
        logger.finish("success")
    except Exception as exc:
        finish_with_error(logger, exc)
        raise


if __name__ == "__main__":
    main()
