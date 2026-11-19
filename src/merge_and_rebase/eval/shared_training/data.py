"""Models, experts, zero-shot heads and paired images for the shared-space experiments (AGENT_SHARED_TRAINING.md).

Splits, loaders and heads are those of ``eval.vision_rebase`` (``_build_task_context`` with the run keys ``seed``,
``val_fraction``, ``batch_size``), so accuracies are comparable with the transport runs and their conservation
(``eval.diagnostics.conservation``: same keys, no humanised class names).

Experts are visual-only: the tuned checkpoint's ``visual.*`` tensors replace the base ones, the text tower (and so
the zero-shot head) stays the base one, as in the transport runs (``_visual_only_filter``).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.vision_rebase.context import (
    DIRECT_RESIDUAL_TINY_IMAGENET_SPEC,
    _build_task_context,
)
from merge_and_rebase.io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from merge_and_rebase.models.openclip_classifier import (
    ZERO_SHOT_CACHE_DIR,
    OpenClipBuildConfig,
    OpenClipClassifier,
)

THESEUS = Path("/data/merge_and_rebase_ckpts/theseus/models/checkpoints")
TSV = Path("/data/merge_and_rebase_ckpts/tsv/models/checkpoints")

#: The three models of AGENT_SHARED_TRAINING.md section 7 (anchors model ids: "<model>__<pretrained>").
ALIASES = {
    "A": "ViT-B-16__datacomp_xl_s13b_b90k",
    "B": "ViT-L-14__datacomp_xl_s13b_b90k",
    "C": "ViT-B-16__openai",
}

#: Split keys of a vision_rebase run (conservation.SPLIT_KEYS) at the values of the anchors runs.
RUN_CFG = {"seed": 89, "val_fraction": 0.1, "batch_size": 32, "num_workers": 0, "no_humanize": True}


def model_id(name: str) -> str:
    return ALIASES.get(name, name)


def split_id(mid: str) -> tuple[str, str]:
    arch, pretrained = model_id(mid).split("__", 1)
    return arch, pretrained


def expert_path(mid: str, task: str) -> Path:
    """Fully fine-tuned expert of ``mid`` on ``task``: THESEUS release, or the TSV release for OpenAI ViT-B-16."""
    arch, pretrained = split_id(mid)
    theseus = THESEUS / arch / pretrained / task.lower() / "best.pt"
    if theseus.exists():
        return theseus
    if pretrained == "openai":
        tsv = TSV / arch / f"{task}Val" / "nonlinear_finetuned.pt"
        if tsv.exists():
            return tsv
    raise FileNotFoundError(f"no expert for {mid} on {task} (looked in {theseus} and the TSV release)")


def build_classifier(mid: str, device: str) -> tuple[OpenClipClassifier, OpenClipBuildConfig]:
    arch, pretrained = split_id(mid)
    build = OpenClipBuildConfig(model_name=arch, pretrained=pretrained, device=device)
    clf = OpenClipClassifier.build(build)
    clf.model.eval()
    for p in clf.model.parameters():
        p.requires_grad_(False)
    return clf, build


def expert_state(clf: OpenClipClassifier, path: str | Path) -> dict[str, torch.Tensor]:
    """Base state dict with the tuned checkpoint's visual tensors (CPU fp32)."""
    base = {k: v.detach().cpu().float() for k, v in clf.model.state_dict().items()}
    tuned = align_to_base_keys(load_ckpt(str(path)), base)
    visual = {k: v.float() for k, v in tuned.items() if k.startswith("visual.") and v.is_floating_point()}
    if not visual:
        raise ValueError(f"{path}: no visual tensors aligned to the base keys")
    changed = sum(int(not torch.equal(visual[k], base[k])) for k in visual)
    if changed == 0:
        raise ValueError(f"{path}: the visual tensors equal the base model (not a fine-tune of this base)")
    return {**base, **visual}


def expert_model(clf: OpenClipClassifier, path: str | Path):
    """A frozen copy of ``clf.model`` with the expert's visual weights."""
    model = deepcopy(clf.model)
    load_into_model(model, expert_state(clf, path), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@dataclass
class TaskData:
    task: str
    contexts: dict[str, Any]  # model id -> _TaskContext (loaders under that model's preprocessing)
    classnames: list[str]
    text_features: dict[str, torch.Tensor]  # model id -> [C, D] (normalised, the zero-shot head)


def task_data(task: str, classifiers: dict[str, tuple[OpenClipClassifier, OpenClipBuildConfig]],
              cfg: dict[str, Any] | None = None) -> TaskData:
    cfg = {**RUN_CFG, **(cfg or {})}
    suite = SUITES[str(cfg.get("suite", "vision8"))]
    contexts, feats = {}, {}
    classnames = None
    for mid, (clf, build) in classifiers.items():
        ctx = _build_task_context(
            task, suite=suite, cfg=cfg, clf_target=clf, clf_source=clf, source_cfg=build, target_cfg=build,
            use_humanized_classnames=not bool(cfg.get("no_humanize", True)), need_source_loaders=False,
        )
        clf.build_zeroshot_text_features(list(ctx.classnames), ctx.build_cfg_task, cache_dir=ZERO_SHOT_CACHE_DIR)
        feats[mid] = clf._zs_text_features.detach().clone()
        contexts[mid] = ctx
        if classnames is None:
            classnames = list(ctx.classnames)
        elif list(ctx.classnames) != classnames:
            raise ValueError(f"{task}: class names differ between models")
    return TaskData(task, contexts, classnames, feats)


def split_loader(ctx, split: str) -> DataLoader:
    return {"train": ctx.loaders.train, "val": ctx.loaders.val, "test": ctx.loaders.test}[split]


def paired_images(loaders: dict[str, DataLoader], *, n: int, seed: int | None, batch_size: int = 32
                  ) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    """The same ``n`` dataset items under every model's preprocessing (seeded permutation, or the first ``n`` when
    ``seed`` is None), as lists of ``(images, labels)`` batches; labels are checked equal across models."""
    ref = next(iter(loaders.values()))
    size = len(ref.dataset)
    if any(len(dl.dataset) != size for dl in loaders.values()):
        raise ValueError("paired loaders have different lengths")
    order = list(range(size)) if seed is None else torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    order = order[: min(int(n), size)]
    out: dict[str, list] = {}
    for mid, dl in loaders.items():
        sub = DataLoader(Subset(dl.dataset, order), batch_size=batch_size, shuffle=False, num_workers=0,
                         collate_fn=dl.collate_fn)
        out[mid] = [(b[0], torch.as_tensor(b[1])) for b in sub]
    labels = [torch.cat([b[1] for b in v]) for v in out.values()]
    if any(not torch.equal(labels[0], lab) for lab in labels[1:]):
        raise ValueError("paired images disagree on labels: dataset orders differ between models")
    return out


def anchor_images(classifiers: dict[str, tuple[OpenClipClassifier, OpenClipBuildConfig]], *, anchor_batches: int = 16,
                  pool_seed: int = 0, batch_size: int = 32) -> dict[str, list[torch.Tensor]]:
    """The anchor images of the main frame (tiny-imagenet valid, seeded draw ``pool_seed``, the first
    ``anchor_batches`` batches of ``batch_size``: AnchorState._draw), under each model's preprocessing."""
    from merge_and_rebase.data.vision_loaders import build_vision_calibration_loader

    suite = SUITES["vision8"]
    loaders = {}
    for mid, (clf, _) in classifiers.items():
        loaders[mid] = build_vision_calibration_loader(
            DIRECT_RESIDUAL_TINY_IMAGENET_SPEC, resolver=suite.resolver, preprocess=clf.preprocess,
            calibration_split=str(DIRECT_RESIDUAL_TINY_IMAGENET_SPEC["split"]), batch_size=batch_size,
            num_workers=0, pin_memory=False, val_fraction=float(RUN_CFG["val_fraction"]), seed=int(RUN_CFG["seed"]),
        )
    pairs = paired_images(loaders, n=anchor_batches * batch_size, seed=pool_seed, batch_size=batch_size)
    return {mid: [b[0] for b in batches] for mid, batches in pairs.items()}
