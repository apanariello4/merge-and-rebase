"""Build task-data-free calibration images for ``calibration_data: "precomputed"``.

    python -m merge_and_rebase.eval.vision_rebase.build_calibration_images \\
        --config <ariadne run config.json> --source retrieval|inversion --task DTD --out-dir <dir>

Writes ``<out-dir>/<Task>.pt`` (``{"images": float32 [N,3,224,224] in the models' preprocessed space, ...}``, stored
in batch order: batch ``k`` = rows ``[k*B, (k+1)*B)``) and ``<out-dir>/<Task>.manifest.json``.

* ``retrieval``: class names only. Pool images are scored with the TARGET base model against the target's zero-shot
  text features of the task (task templates); see ``data/calibration_retrieval.py``.
* ``inversion``: DeepInversion from the SOURCE fine-tuned checkpoint of the task with its zero-shot head;
  see ``data/calibration_inversion.py``.

Models, tuned checkpoints, seed and batch geometry come from the run config (``method.params.num_batches`` and
``batch_size`` give the number of images; ``--calib-seed`` defaults to ``method.params.seed``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from ...data.calibration_inversion import class_targets, invert, teacher_report
from ...data.calibration_retrieval import (
    POOL_SPECS,
    assert_pool_disjoint_from,
    embed_pool,
    load_or_build_pool_embeddings,
    load_pool_datasets,
    locate_pool_row,
    parse_extra_pool,
    select_class_balanced,
)
from ...data.templates import get_templates
from ...data.vision_loaders import extract_classnames, load_hf_splits
from ...io.ckpt import load_ckpt, load_into_model, resolve_ckpt_path
from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ...rebase.config_schema import load_run_config
from ...rebase.run_config import resolve_run_config
from ...utils.helpers import load_json
from ..datasets.vision8_14_20 import SUITES
from ..utils import humanize
from .precomputed_calibration import assert_same_preprocessing, tensor_sha256

# pool-embedding cache; a different location only means the embeddings are recomputed
DEFAULT_CACHE_DIR = os.environ.get("CALIBSRC_CACHE_DIR", "calibsrc_cache")


def task_classnames(task: str, suite: Any, *, humanized: bool) -> list[str]:
    """Class names exactly as the run pipeline derives them (ClassLabel names of the task's test split)."""
    path, config, split_map = suite.resolver(task)
    ds = load_hf_splits(path, config=config, requested_splits=(split_map["test"],))
    names = list(extract_classnames(ds, strict=True))
    return [humanize(c) for c in names] if humanized else names


def _task_cfg(model: str, pretrained: str, device: str, dtype: Any, task: str) -> OpenClipBuildConfig:
    templates = get_templates(task)
    return OpenClipBuildConfig(
        model_name=model, pretrained=pretrained, device=device, dtype=dtype, prompt_templates=templates
    )


def _batches_geometry(cfg: dict[str, Any], run_params: dict[str, Any]) -> tuple[int, int]:
    batch_size = int(cfg.get("batch_size", 128))
    num_batches = int(run_params.get("num_batches", 10))
    return num_batches, batch_size


def build_retrieval(
    *,
    task: str,
    classnames: list[str],
    clf_target: OpenClipClassifier,
    target_build_cfg: OpenClipBuildConfig,
    pool: list[dict[str, str]],
    cache_dir: Path,
    n_images: int,
    batch_size: int,
    seed: int,
    device: str,
    pool_workers: int = 8,
    extra_pool: list[dict[str, str]] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``extra_pool`` (opt-in, default None = unchanged): datasets appended after the generic pool. The generic
    embeddings keep their own cache file; each extra dataset gets its own cache file, so the generic cache is never
    rewritten and pool rows = generic rows followed by the extra datasets in the given order."""
    t0 = time.time()
    extra_pool = list(extra_pool or [])
    base_pool = list(pool)
    pool = base_pool + extra_pool
    datasets_all = load_pool_datasets(pool)
    datasets_ = datasets_all[: len(base_pool)]
    sizes = [len(d) for d in datasets_all]
    model_tag = f"{target_build_cfg.model_name}_{target_build_cfg.pretrained}".replace("/", "-")

    def _embed(ds_list: list[Any]) -> torch.Tensor:
        return embed_pool(
            ds_list, clf_target.model.encode_image, clf_target.preprocess, device=device, num_workers=pool_workers
        )

    cache_path = cache_dir / f"pool_embeddings_{model_tag}.pt"
    emb, from_cache = load_or_build_pool_embeddings(
        cache_path,
        base_pool,
        datasets_,
        model=target_build_cfg.model_name,
        pretrained=target_build_cfg.pretrained,
        build=lambda: _embed(datasets_),
    )
    extra_caches: list[str] = []
    for spec, ds in zip(extra_pool, datasets_all[len(base_pool) :], strict=True):
        ecache = cache_dir / f"pool_embeddings_extra_{spec['name']}_{model_tag}.pt"
        e_emb, e_cached = load_or_build_pool_embeddings(
            ecache,
            [spec],
            [ds],
            model=target_build_cfg.model_name,
            pretrained=target_build_cfg.pretrained,
            build=lambda ds=ds: _embed([ds]),
        )
        emb = torch.cat([emb, e_emb], dim=0)
        from_cache = from_cache and e_cached
        extra_caches.append(str(ecache))
    t_embed = time.time() - t0
    with torch.no_grad():
        text = clf_target._compute_zeroshot_text_features(classnames, target_build_cfg).float().cpu()
    scores = emb @ torch.nn.functional.normalize(text, dim=-1).T  # [N_pool, C] cosine
    sel = select_class_balanced(scores, n_images=n_images, batch_size=batch_size, seed=seed)

    images, picks = [], []
    for pos, (row, cls, score, rnd) in enumerate(zip(sel.pool_rows, sel.class_ids, sel.scores, sel.round_index, strict=True)):
        k, idx = locate_pool_row(row, sizes)
        img = datasets_all[k][idx]["image"].convert("RGB")
        images.append(clf_target.preprocess(img))
        picks.append(
            {
                "position": pos,
                "batch": pos // batch_size,
                "class_id": cls,
                "class": classnames[cls],
                "score": score,
                "round": rnd,
                "pool_dataset": pool[k]["name"],
                "dataset_index": idx,
                "pool_row": row,
            }
        )
    tensor = torch.stack(images).float()
    by_dataset: dict[str, int] = {}
    for p in picks:
        by_dataset[p["pool_dataset"]] = by_dataset.get(p["pool_dataset"], 0) + 1
    per_class: dict[str, int] = {}
    for p in picks:
        per_class[p["class"]] = per_class.get(p["class"], 0) + 1
    manifest = {
        "source": "retrieval",
        "task": task,
        "scorer": {
            "model": target_build_cfg.model_name,
            "pretrained": target_build_cfg.pretrained,
            "head": "target zero-shot text features, task templates",
        },
        "pool": [{**spec, "size": size} for spec, size in zip(pool, sizes, strict=True)],
        "pool_embedding_cache": str(cache_path),
        "extra_pool": [spec["name"] for spec in extra_pool],
        "extra_pool_embedding_caches": extra_caches,
        "pool_embeddings_from_cache": from_cache,
        "pool_embedding_seconds": t_embed,
        "seed": seed,
        "n_images": n_images,
        "batch_size": batch_size,
        "num_classes": len(classnames),
        "classes_used": [classnames[c] for c in sel.classes_used],
        "picked_by_pool_dataset": by_dataset,
        "picked_per_class": per_class,
        "score_mean": float(sum(sel.scores) / len(sel.scores)),
        "score_min": float(min(sel.scores)),
        "picks": picks,
    }
    return tensor, manifest


def build_inversion(
    *,
    task: str,
    classnames: list[str],
    clf_source: OpenClipClassifier,
    source_build_cfg: OpenClipBuildConfig,
    ckpt_path: str,
    n_images: int,
    seed: int,
    iters: int,
    device: str,
    amp_dtype: torch.dtype | None,
    micro_batch: int,
    log_every: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    load_into_model(clf_source.model, load_ckpt(ckpt_path), strict=False)
    clf_source.model.eval()
    for p in clf_source.model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        text = clf_source._compute_zeroshot_text_features(classnames, source_build_cfg).float().to(device)
    scale = clf_source.logit_scale

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        feats = torch.nn.functional.normalize(clf_source.model.encode_image(x).float(), dim=-1)
        return scale * feats @ text.T

    visual = clf_source.model.visual
    mean, std = list(visual.image_mean), list(visual.image_std)
    size = int(tuple(visual.image_size)[0])
    targets, classes = class_targets(len(classnames), n_images, seed)
    t0 = time.time()
    images, trace = invert(
        logits_fn,
        targets,
        iters=iters,
        seed=seed,
        mean=mean,
        std=std,
        image_size=size,
        device=device,
        amp_dtype=amp_dtype,
        micro_batch=micro_batch,
        log_every=log_every,
    )
    seconds = time.time() - t0
    report = teacher_report(logits_fn, images, targets, device=device)
    manifest = {
        "source": "inversion",
        "task": task,
        "teacher": {
            "model": source_build_cfg.model_name,
            "pretrained": source_build_cfg.pretrained,
            "checkpoint": ckpt_path,
            "head": "zero-shot text features of the loaded model, task templates, logit_scale "
            f"{scale}",
        },
        "recipe": {
            "iters": iters,
            "lr": 0.25,
            "betas": [0.5, 0.9],
            "warmup": 100,
            "jitter": 30,
            "flip": True,
            "l2": 1e-5,
            "tv_l2": 1e-4,
            "amp_dtype": str(amp_dtype),
            "micro_batch": micro_batch,
            "space": "CLIP-normalised, clamped to [0,1] after de-normalisation",
        },
        "seed": seed,
        "n_images": n_images,
        "num_classes": len(classnames),
        "classes_used": [classnames[c] for c in classes],
        "synthesis_seconds": seconds,
        "ce_trace_first_last": [trace[0], trace[-1]] if trace else None,
        "teacher_summary": {k: report[k] for k in ("mean_ce", "mean_target_prob", "top1_rate")},
        "per_image": [
            {
                "position": i,
                "class_id": int(targets[i]),
                "class": classnames[targets[i]],
                "final_ce": report["ce"][i],
                "teacher_confidence": report["target_prob"][i],
                "top1_is_target": bool(report["top1_is_target"][i]),
            }
            for i in range(n_images)
        ],
    }
    return images, manifest


def save_outputs(out_dir: Path, task: str, images: torch.Tensor, manifest: dict[str, Any]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = tensor_sha256(images)
    manifest["tensor_sha256"] = digest
    manifest["tensor_shape"] = list(images.shape)
    torch.save({"images": images, "sha256": digest}, out_dir / f"{task}.pt")
    (out_dir / f"{task}.manifest.json").write_text(json.dumps(manifest, indent=1))
    return digest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True, choices=["retrieval", "inversion"])
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calib-seed", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--extra-pool",
        action="append",
        default=[],
        help="retrieval only, opt-in: append a dataset to the generic pool, name=hf_path:split[@cache_dir] "
        "(own embedding cache; default none = unchanged pool)",
    )
    ap.add_argument("--pool-workers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2000, help="inversion iterations")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--amp", choices=["none", "bf16"], default="bf16", help="inversion autocast (synthesis only)")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)

    cfg, _ = load_run_config(load_json(args.config))
    resolved = resolve_run_config(cfg | {"block_extension_enabled": cfg.get("block_extension_enabled", True)}, suites=SUITES)
    params = dict(cfg.get("ariadne_params") or cfg.get("direct_residual_params") or {})
    num_batches, batch_size = _batches_geometry(cfg, params)
    n_images = num_batches * batch_size
    seed = int(args.calib_seed if args.calib_seed is not None else params.get("seed", 54))
    humanized = not bool(cfg.get("no_humanize", True))
    dtype = cfg.get("dtype")
    device = args.device
    src_m, src_p = cfg["source_clip_model"], cfg["source_clip_pretrained"]
    tgt_m, tgt_p = cfg["target_clip_model"], cfg["target_clip_pretrained"]

    clf_target = clf_source = None
    for task in args.task:
        classnames = task_classnames(task, resolved.suite, humanized=humanized)
        out_dir = Path(args.out_dir)
        if args.source == "retrieval":
            if clf_target is None:
                clf_target = OpenClipClassifier.build(OpenClipBuildConfig(model_name=tgt_m, pretrained=tgt_p, device=device, dtype=dtype))
            forbidden = [resolved.suite.resolver(t)[0] for t in resolved.suite.tasks]
            pool = [dict(s) for s in POOL_SPECS if s["path"].lower() not in {f.lower() for f in forbidden}]
            assert_pool_disjoint_from(pool, forbidden)
            extra_pool = parse_extra_pool(args.extra_pool)
            assert_pool_disjoint_from(extra_pool, forbidden)
            images, manifest = build_retrieval(
                task=task,
                classnames=classnames,
                clf_target=clf_target,
                target_build_cfg=_task_cfg(tgt_m, tgt_p, device, dtype, task),
                pool=pool,
                cache_dir=Path(args.cache_dir),
                n_images=n_images,
                batch_size=batch_size,
                seed=seed,
                device=device,
                pool_workers=args.pool_workers,
                extra_pool=extra_pool,
            )
        else:
            clf_source = OpenClipClassifier.build(OpenClipBuildConfig(model_name=src_m, pretrained=src_p, device=device, dtype=dtype))
            if clf_target is None:
                # preprocessing guard only (the tensors go to both models unchanged)
                clf_target = OpenClipClassifier.build(OpenClipBuildConfig(model_name=tgt_m, pretrained=tgt_p, device="cpu", dtype=dtype))
            assert_same_preprocessing(clf_source, clf_target)
            images, manifest = build_inversion(
                task=task,
                classnames=classnames,
                clf_source=clf_source,
                source_build_cfg=_task_cfg(src_m, src_p, device, dtype, task),
                ckpt_path=resolve_ckpt_path(str(cfg["tuned_ckpts"][task])),
                n_images=n_images,
                seed=seed,
                iters=args.iters,
                device=device,
                amp_dtype=torch.bfloat16 if args.amp == "bf16" else None,
                micro_batch=args.micro_batch,
                log_every=args.log_every,
            )
            clf_source = None
        manifest["config"] = args.config
        manifest["models"] = {"source": [src_m, src_p], "target": [tgt_m, tgt_p]}
        digest = save_outputs(out_dir, task, images, manifest)
        print(f"{task}: wrote {out_dir / (task + '.pt')} sha256={digest}")


if __name__ == "__main__":
    main()
