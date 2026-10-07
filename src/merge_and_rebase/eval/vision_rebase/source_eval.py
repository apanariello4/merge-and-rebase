"""Top-1 of a source model on a target task's dataset (the ``eval_before_rebase`` prestep evaluation)."""

from __future__ import annotations

import itertools
import os
from typing import Any

import torch

from ...eval.utils import resolve_eval_split_loader
from ...models.openclip_classifier import ZERO_SHOT_CACHE_DIR, OpenClipBuildConfig, OpenClipClassifier

_ZERO_SHOT_CACHE_DIR = os.environ.get("BRACE_ZS_CACHE_DIR", ZERO_SHOT_CACHE_DIR)


def _evaluate_source_model_top1(
    *,
    model: torch.nn.Module,
    clf_source: OpenClipClassifier,
    loaders_obj: Any,
    classnames_task: list[str],
    source_build_cfg_task: OpenClipBuildConfig,
    split: str,
    first_n_batches: int | None,
    device: str,
) -> float:
    eval_clf = OpenClipClassifier(
        model=model,
        tokenizer=clf_source.tokenizer,
        preprocess=clf_source.preprocess,
        normalize=clf_source.normalize,
        logit_scale=clf_source.logit_scale,
    )
    eval_loader = resolve_eval_split_loader(loaders_obj, split)
    if first_n_batches is not None:
        eval_loader = itertools.islice(iter(eval_loader), max(1, int(first_n_batches)))

    eval_clf.build_zeroshot_text_features(
        classnames_task,
        source_build_cfg_task,
        cache_dir=_ZERO_SHOT_CACHE_DIR,
        force_rebuild=False,
    )
    return float(eval_clf.top1(eval_loader, device=device))
