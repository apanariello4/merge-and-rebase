"""Readout diagnostic (vision): is the accuracy gain of an Ariadne task vector carried by the part of its effect that is
ALIGNED with the transported target, or by the ORTHOGONAL remainder?

``python -m merge_and_rebase.eval.diagnostics.ariadne_norm.readout --config <json> --device cuda``

For every image the classifier of a CLIP ViT reads only the final CLS token (``pool_type == "tok"``: after the last
resblock -> ``ln_post`` -> ``proj`` -> L2 normalisation -> zero-shot logits; verified at run time, any other pooling is
refused). With ``tau`` the reference run's saved unit task vector and ``N`` the target depth, per image:

- ``t0``: CLS state after the last target block of the target base; ``t1(alpha)``: the same for ``target_base + alpha *
  tau``; ``delta = t1 - t0``.
- ``d = (f - b) @ Q_{N-1}``: the transported target at the final block. ``f``/``b`` are the CLS states of the source
  fine-tuned / base model after source block ``pairing[N-1]`` on the SOURCE view of the same image, ``Q_{N-1}`` the
  Procrustes map the reference fit used (re-derived with the fit's own ``prepare_direct_residual_streaming`` on the
  fit's calibration loaders).
- ``c = <delta, d> / ||d||^2``; ``aligned = c d``; ``orthogonal = delta - c d`` (so ``aligned . orthogonal = 0``).

Variants (the final CLS state is replaced, then the model's own head is applied): ``base`` = ``t0``; ``full`` =
``t0 + delta`` (= ``t1``); ``aligned`` = ``t0 + c d``; ``orthogonal`` = ``t0 + delta - c d``; ``target`` =
``t0 + alpha d`` (the transported target itself, no update); ``aligned_global`` = ``t0 + c_g d`` with ONE scalar
``c_g(alpha) = sum_val <delta, d> / sum_val ||d||^2`` per task and alpha, fitted on the validation split (control for
the per-image fit). All variants are scored at every alpha; alpha is chosen per variant on the split the reference run
searched alpha on (``alpha_search_split``, normally ``val``; first maximum, like ``PerTaskAlphaTracker``) and the
accuracy on the test split at that alpha is reported, together with the test accuracy of every variant at the
reference run's own alpha.

Config keys:

- ``family``: ``"vision"`` (only).
- ``reference_summary``: summary JSON of the reference Ariadne run (``merge_mode == "none"``, unit saved vectors).
- ``tasks``: ``"all"`` or a list. ``out_dir``: result directory; must not exist (results are write-once).
- ``alpha_grid``: default ``0, 0.05, ..., 1.0``. ``0``, ``1`` and the reference alpha are always added (snapped to
  the reference run's float spelling, e.g. ``0.20000000298023224``) so that the correctness gates can be evaluated.
- ``gate_batches`` (default 2), ``gate_logit_tol`` (1e-4), ``gate_acc_tol`` (1e-3), ``gate_cls_tol`` (1e-5, relative):
  correctness gates, see ``check_gates``. ``zs_cache_dir`` (default ``src/.cache/zs_cache``, as the evaluation).
- ``max_batches_per_split``: testing only; accuracy gates against the reference are then skipped (subset).

Optional analyses (all off by default; without them the output is unchanged):

- ``native_target_ckpt_template`` (``"...{task}..."``): the NATIVE fine-tuned target checkpoint. Adds the ceiling (native
  forward, and native CLS through the base head), cos(d, n) / ||d||/||n|| at the final CLS, the mounted delta
  decomposed against n (``aligned_native`` / ``orthogonal_native``, alpha chosen on the search split, plus the norms of
  the span{d, n} / rest split), and the all-block boundary comparison (streamed Frobenius sums, all tokens and CLS
  only, on the first ``boundary_max_batches`` (10; ``null`` = all) search-split batches, ``boundary_block_group`` blocks
  per forward group) at alpha in {reference alpha, 1}. Result: ``extras.native`` / ``extras.boundary_comparison``.
- ``oracle_patch_blocks`` (list of target blocks or ``"auto"``) with ``oracle_beta_grid`` (default 0.25, 0.5, 1, 1.5):
  single-block oracle patching, the base target with block j's output replaced by ``T_j + beta * X_j`` (all tokens,
  X = D, the transported target, and, with the native checkpoint, N, the native displacement), read out at the final
  CLS; beta chosen on the search split. Result: ``extras.oracle``.
- ``eval_subset_size`` / ``eval_subset_seed`` (0) / ``eval_subset_applies_to_all`` (false): fixed seeded random subset
  (sorted indices, sha256 recorded per split) used by the optional analyses; the main variants keep the full split
  unless ``eval_subset_applies_to_all`` (then the accuracy gates against the reference are skipped). Base / full /
  target accuracies on the same rows are in ``extras.subset.reference_curves_same_rows``. Exclusive with
  ``max_batches_per_split``.

Output: ``<out_dir>/<task>/readout.json`` (curves, selection, statistics, gates, provenance),
``<out_dir>/<task>/arrays.pt`` (per-image predictions / scalars) and ``<out_dir>/run.json``. A violated gate raises
after the task's files are written.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import itertools
import json
import math
import os
import resource
import socket
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset

from merge_and_rebase.eval.vision_rebase.merge import _infer_ckpt_base
from merge_and_rebase.io.ckpt import load_ckpt, load_into_model
from merge_and_rebase.merge.methods._common import axpy_state_dict
from merge_and_rebase.models.openclip_classifier import (
    ZERO_SHOT_CACHE_DIR,
    normalize_features,
    zero_shot_logits_from_features,
)
from merge_and_rebase.rebase.methods._ariadne.capture import (
    _dataset_identity,
    _register_capture_hooks,
    iter_capture_tokens,
)
from merge_and_rebase.rebase.methods._ariadne.layouts import _aligned, _layout_for
from merge_and_rebase.rebase.methods._ariadne.streaming import prepare_direct_residual_streaming
from merge_and_rebase.rebase.methods._shared import _to_tokens
from merge_and_rebase.run_logging import _code_fingerprint

Tensor = torch.Tensor

VARIANTS = ("base", "full", "aligned", "orthogonal", "target", "aligned_global")
#: Variants that are scored at every alpha (``base`` does not depend on it).
ALPHA_VARIANTS = VARIANTS[1:]
DEFAULT_ALPHA_GRID = [round(0.05 * k, 10) for k in range(0, 21)]
_EPS = 1e-12  # PerTaskAlphaTracker's improvement threshold


class GateError(RuntimeError):
    """A correctness gate of the readout diagnostic failed."""


# ---- pure helpers -------------------------------------------------------------------------------------------------


def decompose(delta: Tensor, d: Tensor) -> dict[str, Tensor]:
    """Per-row projection of ``delta`` on ``d`` in float64: ``c``, ``aligned``, ``orthogonal`` and the dot/norm terms.

    Rows with ``d == 0`` get ``c = 0`` (aligned part 0, orthogonal part = delta); ``d_sq`` lets callers exclude them.
    """
    delta64, d64 = delta.double(), d.double()
    d_sq = (d64 * d64).sum(-1)
    dot = (delta64 * d64).sum(-1)
    c = torch.where(d_sq > 0, dot / d_sq.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(dot))
    aligned = c[:, None] * d64
    return {"c": c, "aligned": aligned, "orthogonal": delta64 - aligned, "dot": dot, "d_sq": d_sq}


def global_coefficient(delta: Tensor, d: Tensor) -> float:
    """``sum <delta, d> / sum ||d||^2`` over all rows (the task-level scalar of ``aligned_global``)."""
    parts = decompose(delta, d)
    den = float(parts["d_sq"].sum())
    return float(parts["dot"].sum()) / den if den > 0 else 0.0


def select_alpha(alphas: Sequence[float], accs: Sequence[float]) -> int:
    """Index of the first maximum (an alpha replaces the incumbent only if better by ``1e-12``, like the tracker)."""
    best_i, best = 0, float("-inf")
    for i, acc in enumerate(accs):
        if float(acc) > best + _EPS:
            best_i, best = i, float(acc)
    return best_i


def resolve_alpha_grid(
    alpha_grid: Sequence[float] | None, reference_alpha: float | None, *, snap: float = 1e-6
) -> tuple[list[float], dict[str, Any]]:
    """Sorted unique grid containing 0, 1 and ``reference_alpha``; near-duplicates snap to the required value."""
    grid = [float(a) for a in (DEFAULT_ALPHA_GRID if alpha_grid is None else alpha_grid)]
    required = [0.0, 1.0] + ([float(reference_alpha)] if reference_alpha is not None else [])
    added: list[float] = []
    for value in required:
        near = [i for i, a in enumerate(grid) if abs(a - value) < snap]
        if near:
            for i in near:
                grid[i] = value
        else:
            grid.append(value)
            added.append(value)
    out = sorted(set(grid))
    return out, {
        "requested": [float(a) for a in (DEFAULT_ALPHA_GRID if alpha_grid is None else alpha_grid)],
        "added": added,
    }


def _finite_stat(values: Tensor, fn: str) -> float:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan")
    return float(values.mean() if fn == "mean" else values.median())


def alpha_statistics(parts: Mapping[str, Tensor], delta: Tensor, c_global: float | None) -> dict[str, float]:
    """Per-alpha summary of one split: c, ||orth||/||d||, ||delta||/||d||, cos(delta, d)."""
    d_norm = parts["d_sq"].sqrt()
    nonzero = d_norm > 0
    safe = d_norm.clamp_min(torch.finfo(torch.float64).tiny)
    delta_norm = delta.double().norm(dim=-1)
    orth_ratio = torch.where(nonzero, parts["orthogonal"].norm(dim=-1) / safe, torch.full_like(d_norm, float("nan")))
    delta_ratio = torch.where(nonzero, delta_norm / safe, torch.full_like(d_norm, float("nan")))
    cos = torch.where(
        nonzero & (delta_norm > 0),
        parts["dot"] / (safe * delta_norm.clamp_min(torch.finfo(torch.float64).tiny)),
        torch.full_like(d_norm, float("nan")),
    )
    return {
        "c_mean": _finite_stat(parts["c"], "mean"),
        "c_median": _finite_stat(parts["c"], "median"),
        "orth_over_d_mean": _finite_stat(orth_ratio, "mean"),
        "orth_over_d_median": _finite_stat(orth_ratio, "median"),
        "delta_over_d_mean": _finite_stat(delta_ratio, "mean"),
        "delta_over_d_median": _finite_stat(delta_ratio, "median"),
        "cos_mean": _finite_stat(cos, "mean"),
        "n_zero_d": int((~nonzero).sum()),
        "c_global": c_global,
    }


# ---- the model head -----------------------------------------------------------------------------------------------


class VisionHead:
    """The classifier's read-out of a final CLS state: ``ln_post -> proj -> (L2 norm) -> zero-shot logits``.

    ``from_classifier`` verifies the pooling (open_clip ``VisionTransformer``: ``attn_pool is None`` and
    ``pool_type == "tok"``): ``_pool`` then reads token 0 after ``ln_post`` (a per-token LayerNorm, so applying it to
    the CLS state alone is identical) and ``forward`` applies ``@ proj``. ``OpenClipClassifier.forward`` finishes with
    ``normalize_features`` (if ``normalize``) and ``zero_shot_logits_from_features`` -- reused here, not re-implemented.
    """

    def __init__(self, classifier: Any, visual: torch.nn.Module, text_features: Tensor):
        pool_type = getattr(visual, "pool_type", None)
        attn_pool = getattr(visual, "attn_pool", None)
        if attn_pool is not None or pool_type != "tok":
            raise NotImplementedError(
                f"readout supports CLS pooling only; got pool_type={pool_type!r}, attn_pool={type(attn_pool).__name__}"
            )
        self.classifier = classifier
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.pooling = {
            "pool_type": pool_type,
            "attn_pool": None,
            "final_ln_after_pool": bool(getattr(visual, "final_ln_after_pool", False)),
            "global_average_pool": getattr(visual, "global_average_pool", None),
        }
        self._dtype = self.ln_post.weight.dtype
        self.text_features = text_features

    @classmethod
    def from_classifier(cls, classifier: Any, device: Any) -> VisionHead:
        text = classifier._zs_text_features
        if not isinstance(text, torch.Tensor) or text.numel() == 0:
            raise RuntimeError("zero-shot text features are not built")
        classifier._zs_text_features = text.to(device)
        return cls(classifier, classifier.model.visual, classifier._zs_text_features)

    def finish(self, visual_features: Tensor) -> Tensor:
        """Logits of already-projected image features (``OpenClipClassifier.forward`` after ``encode_image``)."""
        feats = normalize_features(visual_features) if self.classifier.normalize else visual_features
        return zero_shot_logits_from_features(self.classifier, feats, normalize_image_features=False)

    @torch.no_grad()
    def logits(self, cls_state: Tensor) -> Tensor:
        pooled = self.ln_post(cls_state.to(self._dtype))
        if self.proj is not None:
            pooled = pooled @ self.proj
        return self.finish(pooled)


# ---- sweeps over the loaders ---------------------------------------------------------------------------------------


def _take(loader: Any, max_batches: int | None) -> Iterator[Any]:
    it = iter(loader)
    return itertools.islice(it, int(max_batches)) if max_batches else it


def _tap(it: Iterator[Any], labels: list[Tensor], kept: list[Any] | None = None, keep: int = 0) -> Iterator[Any]:
    """Pass batches through, recording their labels (and the first ``keep`` batches) as they are consumed."""
    for batch in it:
        labels.append(torch.as_tensor(batch[1]).detach().cpu())
        if kept is not None and len(kept) < keep:
            kept.append(batch)
        yield batch


def check_loader_pairing(source_loader: Any, target_loader: Any, split: str) -> dict[str, Any]:
    """The two views of ``split`` must be the same rows in the same order: sequential, same batch size and identity."""
    for name, loader in (("source", source_loader), ("target", target_loader)):
        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError(f"{split}: the {name} loader is not sequential ({type(loader.sampler).__name__})")
    if source_loader.batch_size != target_loader.batch_size:
        raise ValueError(f"{split}: batch sizes differ ({source_loader.batch_size} vs {target_loader.batch_size})")
    if len(source_loader.dataset) != len(target_loader.dataset):
        raise ValueError(f"{split}: dataset sizes differ")
    if _dataset_identity(source_loader.dataset) != _dataset_identity(target_loader.dataset):
        raise ValueError(f"{split}: source/target dataset identities (split fingerprint, length) differ")
    return {
        "n": len(target_loader.dataset),
        "batch_size": int(target_loader.batch_size),
        "sequential": True,
        "dataset_identity_equal": True,
    }


def _assert_cls_layout(tokens: Tensor, what: str) -> None:
    side = int(round((tokens.shape[1] - 1) ** 0.5))
    if tokens.ndim != 3 or tokens.shape[1] < 2 or side * side != tokens.shape[1] - 1:
        raise ValueError(f"{what}: expected [B, 1 + g^2, D] tokens (CLS first), got {tuple(tokens.shape)}")


@torch.no_grad()
def collect_reference_states(
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_loader: Any,
    target_loader: Any,
    source_block: int,
    target_block: int,
    q: Tensor,
    device: Any,
    max_batches: int | None = None,
) -> dict[str, Tensor]:
    """One lockstep sweep of a split: ``t0`` (target base CLS), ``d = (f - b) @ q`` and the labels, all on CPU.

    The source loader is iterated once and fed to the base and fine-tuned source models (``itertools.tee``); labels of
    the two views are compared batch by batch. The CLS token is index 0 of both captures; ``_aligned``'s token
    interpolation (``_interp_2d_tokens``) leaves it untouched, so no interpolation is needed for CLS.
    """
    src_labels: list[Tensor] = []
    tgt_labels: list[Tensor] = []
    src_a, src_b = itertools.tee(_tap(_take(source_loader, max_batches), src_labels))
    src_req = {"s": (source_block, "boundary")}
    gens = [
        iter_capture_tokens(source_base_model, src_a, src_req, device, store_device=device),
        iter_capture_tokens(source_ft_model, src_b, src_req, device, store_device=device),
        iter_capture_tokens(
            target_model,
            _tap(_take(target_loader, max_batches), tgt_labels),
            {"t": (target_block, "boundary")},
            device,
            store_device=device,
        ),
    ]
    q = q.to(device=device, dtype=torch.float32)
    t0_all: list[Tensor] = []
    d_all: list[Tensor] = []
    with contextlib.ExitStack() as stack:
        for gen in gens:
            stack.enter_context(contextlib.closing(gen))
        for k, (sb, sf, tg) in enumerate(zip(*gens, strict=True)):
            if k == 0:
                _assert_cls_layout(sb["s"], "source")
                _assert_cls_layout(tg["t"], "target")
            if not torch.equal(src_labels[k], tgt_labels[k]):
                raise ValueError(f"batch {k}: source and target views carry different labels")
            f, b = sf["s"][:, 0], sb["s"][:, 0]
            if f.shape[-1] != q.shape[0] or tg["t"].shape[-1] != q.shape[1]:
                raise ValueError(f"Q {tuple(q.shape)} does not map source width {f.shape[-1]} to {tg['t'].shape[-1]}")
            d_all.append(((f - b) @ q).cpu())
            t0_all.append(tg["t"][:, 0].cpu())
    return {"t0": torch.cat(t0_all), "d": torch.cat(d_all), "labels": torch.cat(tgt_labels)}


@torch.no_grad()
def collect_mounted_cls(
    model: torch.nn.Module,
    loader: Any,
    *,
    target_block: int,
    device: Any,
    expected_labels: Tensor,
    max_batches: int | None = None,
    keep_batches: int = 0,
) -> tuple[Tensor, list[Any]]:
    """Final-block CLS states ``t1`` of ``model`` over a split (CPU float32) and the first ``keep_batches`` batches."""
    labels: list[Tensor] = []
    kept: list[Any] = []
    gen = iter_capture_tokens(
        model,
        _tap(_take(loader, max_batches), labels, kept, keep_batches),
        {"t": (target_block, "boundary")},
        device,
        store_device=device,
    )
    out: list[Tensor] = []
    with contextlib.closing(gen):
        for tg in gen:
            out.append(tg["t"][:, 0].cpu())
    if not torch.equal(torch.cat(labels), expected_labels):
        raise ValueError("the split's label sequence changed between sweeps (loader order is not stable)")
    return torch.cat(out), kept


@torch.no_grad()
def predictions_from_states(head: VisionHead, states: Tensor, device: Any, chunk: int = 2048) -> Tensor:
    preds = []
    for start in range(0, states.shape[0], chunk):
        logits = head.logits(states[start : start + chunk].to(device=device, dtype=torch.float32))
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds)


@torch.no_grad()
def evaluate_variants(
    head: VisionHead,
    t0: Tensor,
    d: Tensor,
    t1: Tensor,
    labels: Tensor,
    alpha: float,
    c_global: float | None,
    device: Any,
) -> dict[str, Any]:
    """Predictions and accuracy of every alpha-dependent variant, plus the per-image decomposition scalars."""
    delta = (t1.double() - t0.double()).to(device)
    d_dev, t0_dev = d.to(device), t0.to(device)
    parts = decompose(delta, d_dev)
    states: dict[str, Tensor] = {
        "full": t0_dev.double() + delta,
        "aligned": t0_dev.double() + parts["aligned"],
        "orthogonal": t0_dev.double() + parts["orthogonal"],
        "target": t0_dev.double() + float(alpha) * d_dev.double(),
    }
    if c_global is not None:
        states["aligned_global"] = t0_dev.double() + float(c_global) * d_dev.double()
    preds = {name: predictions_from_states(head, s.float(), device) for name, s in states.items()}
    accs = {name: float((p == labels).double().mean()) for name, p in preds.items()}
    d_norm = parts["d_sq"].sqrt()
    safe = d_norm.clamp_min(torch.finfo(torch.float64).tiny)
    per_image = {
        "c": parts["c"].float().cpu(),
        "orth_over_d": (parts["orthogonal"].norm(dim=-1) / safe).float().cpu(),
        "delta_over_d": (delta.norm(dim=-1) / safe).float().cpu(),
    }
    return {
        "preds": preds,
        "acc": accs,
        "stats": alpha_statistics({k: v.cpu() for k, v in parts.items()}, delta.cpu(), c_global),
        "per_image": per_image,
        "delta_dot_sum": float(parts["dot"].sum()),
        "d_sq_sum": float(parts["d_sq"].sum()),
        "delta": delta.cpu(),
    }


# ---- optional analyses: subsets, native target, all-block comparison, oracle patching -------------------------------


def subset_indices(n: int, size: int | None, seed: int) -> tuple[Tensor | None, str | None]:
    """Fixed seeded random subset (sorted, so loader order is preserved) and the sha256 of its index list."""
    if size is None or int(size) >= n:
        return None, None
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))[: int(size)].sort().values
    return idx, hashlib.sha256(repr(idx.tolist()).encode()).hexdigest()


def subset_loader(loader: Any, idx: Tensor) -> DataLoader:
    """Sequential loader over ``idx`` (in that order) of ``loader.dataset`` with the loader's batch size."""
    return DataLoader(
        Subset(loader.dataset, [int(i) for i in idx]),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
    )


def resolve_oracle_blocks(spec: Any, depth: int) -> list[int]:
    if spec is None:
        return []
    if spec == "auto":
        blocks = {int(round(f * (depth - 1))) for f in (0.2, 0.4, 0.6, 0.8)}
    else:
        blocks = {int(b) for b in spec}
    if not blocks or min(blocks) < 0 or max(blocks) > depth - 1:
        raise ValueError(f"oracle_patch_blocks {sorted(blocks)} outside [0, {depth - 1}]")
    return sorted(blocks)


def span_decomposition(delta: Tensor, d: Tensor, n: Tensor) -> dict[str, Tensor]:
    """Per-row least-squares split of ``delta`` into ``span{d, n}`` plus the rest (float64, pseudo-inverse Gram)."""
    delta, d, n = delta.double(), d.double(), n.double()
    gram = torch.stack(
        [
            torch.stack([(d * d).sum(-1), (d * n).sum(-1)], dim=-1),
            torch.stack([(d * n).sum(-1), (n * n).sum(-1)], dim=-1),
        ],
        dim=-2,
    )
    rhs = torch.stack([(delta * d).sum(-1), (delta * n).sum(-1)], dim=-1)
    coef = (torch.linalg.pinv(gram, hermitian=True) @ rhs[..., None])[..., 0]
    proj = coef[:, :1] * d + coef[:, 1:] * n
    return {"coef_d": coef[:, 0], "coef_n": coef[:, 1], "span": proj, "rest": delta - proj}


def _ratio_stats(values: Tensor) -> dict[str, float]:
    return {"mean": _finite_stat(values, "mean"), "median": _finite_stat(values, "median")}


def native_alpha_statistics(
    parts_n: Mapping[str, Tensor], span: Mapping[str, Tensor], delta: Tensor, d: Tensor, n: Tensor
) -> dict[str, float]:
    """Per-alpha summary of the mounted delta against the native displacement (norms only for the 3-way split)."""
    delta64 = delta.double()
    dn = delta64.norm(dim=-1)
    safe = dn.clamp_min(torch.finfo(torch.float64).tiny)
    nan = torch.full_like(dn, float("nan"))
    nonzero = dn > 0
    n_norm = n.double().norm(dim=-1)
    d_norm = d.double().norm(dim=-1)
    total_sq = float((dn**2).sum())
    out = {
        "c_n_mean": _finite_stat(parts_n["c"], "mean"),
        "c_n_median": _finite_stat(parts_n["c"], "median"),
        "cos_delta_n_mean": _finite_stat(
            torch.where(nonzero & (n_norm > 0), parts_n["dot"] / (safe * n_norm.clamp_min(1e-300)), nan), "mean"
        ),
        "orth_native_over_delta_mean": _finite_stat(
            torch.where(nonzero, parts_n["orthogonal"].norm(dim=-1) / safe, nan), "mean"
        ),
        "span_norm_over_delta_mean": _finite_stat(torch.where(nonzero, span["span"].norm(dim=-1) / safe, nan), "mean"),
        "rest_norm_over_delta_mean": _finite_stat(torch.where(nonzero, span["rest"].norm(dim=-1) / safe, nan), "mean"),
        "span_sq_fraction": float((span["span"].norm(dim=-1) ** 2).sum()) / total_sq if total_sq > 0 else float("nan"),
        "rest_sq_fraction": float((span["rest"].norm(dim=-1) ** 2).sum()) / total_sq if total_sq > 0 else float("nan"),
        "coef_d_mean": _finite_stat(span["coef_d"], "mean"),
        "coef_n_mean": _finite_stat(span["coef_n"], "mean"),
        "d_component_norm_over_delta_mean": _finite_stat(
            torch.where(nonzero, (span["coef_d"].abs() * d_norm) / safe, nan), "mean"
        ),
        "n_component_norm_over_delta_mean": _finite_stat(
            torch.where(nonzero, (span["coef_n"].abs() * n_norm) / safe, nan), "mean"
        ),
    }
    return out


@torch.no_grad()
def evaluate_native_variants(
    head: VisionHead, t0: Tensor, d: Tensor, n: Tensor, t1: Tensor, labels: Tensor, device: Any
) -> dict[str, Any]:
    """``aligned_native`` / ``orthogonal_native`` (delta against the native displacement ``n``) at one alpha."""
    delta = (t1.double() - t0.double()).to(device)
    n_dev, d_dev, t0_dev = n.to(device).double(), d.to(device), t0.to(device).double()
    parts = decompose(delta, n_dev)
    states = {"aligned_native": t0_dev + parts["aligned"], "orthogonal_native": t0_dev + parts["orthogonal"]}
    preds = {k: predictions_from_states(head, v.float(), device) for k, v in states.items()}
    span = span_decomposition(delta, d_dev, n_dev)
    return {
        "preds": preds,
        "acc": {k: float((p == labels).double().mean()) for k, p in preds.items()},
        "stats": native_alpha_statistics(parts, span, delta, d_dev, n_dev),
        "c_n": parts["c"].float().cpu(),
    }


def build_native_model(
    target_model: torch.nn.Module,
    target_base_state: Mapping[str, Tensor],
    source_base_state: Mapping[str, Tensor],
    ckpt_path: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """The fine-tuned NATIVE target, built like vision.py builds the source ft model (deepcopy, base strict, ckpt loose)."""
    raw = load_ckpt(str(ckpt_path))
    inferred = _infer_ckpt_base(raw, source_base_sd=source_base_state, target_base_sd=target_base_state)
    if inferred != "target":
        raise ValueError(f"{ckpt_path}: checkpoint matches the {inferred!r} backbone, not the target")
    native = copy.deepcopy(target_model)
    load_into_model(native, target_base_state, strict=True)
    load_into_model(native, raw, strict=False)
    base = target_base_state
    native_sd = native.state_dict()
    non_visual_changed = sorted(
        k
        for k in raw
        if k in base
        and not k.startswith("visual.")
        and not torch.equal(native_sd[k].cpu(), base[k].to(native_sd[k].dtype).cpu())
    )
    head_keys = ("visual.ln_post.weight", "visual.ln_post.bias", "visual.proj")
    head_diff = {
        k: bool(not torch.equal(native_sd[k].cpu(), base[k].to(native_sd[k].dtype).cpu()))
        for k in head_keys
        if k in base
    }
    info = {
        "ckpt": str(ckpt_path),
        "inferred_base": inferred,
        "n_ckpt_keys": len(raw),
        "keys_in_ckpt_not_in_target": sorted(k for k in raw if k not in base),
        "keys_in_target_not_in_ckpt": sum(1 for k in base if k not in raw),
        "non_visual_keys_changed": non_visual_changed,
        "head_keys_differ_from_base": head_diff,
        "ln_post_or_proj_differs": any(head_diff.values()),
    }
    return native.eval(), info


@torch.no_grad()
def capture_batch(model: torch.nn.Module, images: Tensor, requests: Mapping[Any, tuple[int, str]], device: Any) -> dict:
    """One forward of ``images`` with the capture hooks of ``iter_capture_tokens``: ``{key: [B, T, D] float32}``."""
    layout = _layout_for(None)
    blocks = layout.blocks(model)
    batch_size = len(images)
    values: dict[Any, Tensor] = {}

    def store(name, tensor):
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        values[name] = _to_tokens(tensor.detach(), batch_size=batch_size).float().clone()

    handles = _register_capture_hooks(layout, blocks, requests, store)
    try:
        layout.forward(model, (images,), device)
    finally:
        for handle in handles:
            handle.remove()
    if set(values) != set(requests):
        raise RuntimeError("a requested activation hook did not fire")
    return values


def aligned_displacement(f: Tensor, b: Tensor, target_tokens: Tensor, q: Tensor) -> Tensor:
    """``(aligned f - aligned b) @ q`` exactly as the fit / runner cache build it (all tokens, ``_aligned`` first)."""
    fa = _aligned([f], [target_tokens])[0]
    ba = _aligned([b], [target_tokens])[0]
    return (fa - ba) @ q


@torch.no_grad()
def forward_with_block_patch(
    model: torch.nn.Module,
    images: Tensor,
    block: int,
    patch: Tensor | None,
    device: Any,
    capture_blocks: Sequence[int] = (),
) -> tuple[Tensor, dict[int, Tensor]]:
    """Forward ``model`` with block ``block``'s output replaced by ``output + patch`` (all tokens, that block only).

    ``patch`` is ``[B, T, D]`` (``None``: plain forward). The patch hook is registered before the capture hooks, so a
    capture on the same block sees the patched output. Returns the final-block CLS state ``[B, D]`` (float32) and
    ``{i: [B, T, D]}`` for ``capture_blocks`` (the outputs the downstream blocks produce).
    """
    layout = _layout_for(None)
    blocks = layout.blocks(model)
    last = len(blocks) - 1
    batch_first = bool(getattr(model.visual.transformer, "batch_first", True))
    batch_size = len(images)
    seen: dict[str, Tensor] = {}
    captured: dict[int, Tensor] = {}

    def patch_hook(_module, _inputs, out):
        tup = isinstance(out, tuple)
        x = out[0] if tup else out
        p = patch if batch_first else patch.transpose(0, 1)
        if x.shape != p.shape:
            raise ValueError(f"patch {tuple(p.shape)} does not match the block output {tuple(x.shape)}")
        x = x + p.to(x.dtype)
        return (x, *out[1:]) if tup else x

    def tokens(out):
        return _to_tokens((out[0] if isinstance(out, tuple) else out).detach(), batch_size=batch_size).float().clone()

    handles = []
    try:
        if patch is not None:
            handles.append(layout.block_module(blocks[block]).register_forward_hook(patch_hook))
        for i in sorted({last, *capture_blocks}):

            def hook(_m, _i, out, *, i=i):
                t = tokens(out)
                if i == last:
                    seen["cls"] = t[:, 0]
                if i in capture_blocks:
                    captured[i] = t

            handles.append(layout.block_module(blocks[i]).register_forward_hook(hook))
        layout.forward(model, (images,), device)
    finally:
        for handle in handles:
            handle.remove()
    return seen["cls"], captured


class BlockSums:
    """Streamed Frobenius sums of the all-block comparison (no per-image storage), per block and scope."""

    SCOPES = ("all", "cls")

    def __init__(self, blocks: Sequence[int], alphas: Sequence[float]):
        self.alphas = [float(a) for a in alphas]
        self.sums: dict[int, dict[str, dict[str, float]]] = {j: {s: {} for s in self.SCOPES} for j in blocks}

    @staticmethod
    def _rows(x: Tensor, scope: str) -> Tensor:
        return (x[:, 0] if scope == "cls" else x.reshape(-1, x.shape[-1])).double()

    def update(self, j: int, D: Tensor, N: Tensor, dT: Mapping[float, Tensor]) -> None:
        for scope in self.SCOPES:
            d, nn_ = self._rows(D, scope), self._rows(N, scope)
            acc = self.sums[j][scope]
            terms = {"dn": (d * nn_).sum(), "dd": (d * d).sum(), "nn": (nn_ * nn_).sum()}
            for a in self.alphas:
                t = self._rows(dT[a], scope)
                terms[f"tn@{a}"] = (t * nn_).sum()
                terms[f"td@{a}"] = (t * d).sum()
                terms[f"tt@{a}"] = (t * t).sum()
                terms[f"err@{a}"] = ((t - nn_) ** 2).sum()
            for key, value in terms.items():
                acc[key] = acc.get(key, 0.0) + float(value)

    def summary(self) -> dict[int, dict[str, dict[str, Any]]]:
        def div(a: float, b: float) -> float:
            return a / b if b > 0 else float("nan")

        out: dict[int, dict[str, dict[str, Any]]] = {}
        for j, scopes in self.sums.items():
            out[j] = {}
            for scope, acc in scopes.items():
                row: dict[str, Any] = {
                    "sums": dict(acc),
                    "cos_D_N": div(acc["dn"], (acc["dd"] * acc["nn"]) ** 0.5),
                    "norm_D_over_norm_N": div(acc["dd"], acc["nn"]) ** 0.5 if acc["nn"] > 0 else float("nan"),
                    "by_alpha": {},
                }
                for a in self.alphas:
                    row["by_alpha"][a] = {
                        "cos_dT_N": div(acc[f"tn@{a}"], (acc[f"tt@{a}"] * acc["nn"]) ** 0.5),
                        "cos_dT_D": div(acc[f"td@{a}"], (acc[f"tt@{a}"] * acc["dd"]) ** 0.5),
                        "projection_coefficient_dT_on_N": div(acc[f"tn@{a}"], acc["nn"]),
                        "relative_error_to_N": div(acc[f"err@{a}"], acc["nn"]) ** 0.5
                        if acc["nn"] > 0
                        else float("nan"),
                        "norm_dT_over_norm_N": div(acc[f"tt@{a}"], acc["nn"]) ** 0.5 if acc["nn"] > 0 else float("nan"),
                    }
                out[j][scope] = row
        return out


@torch.no_grad()
def boundary_comparison(
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    native_model: torch.nn.Module,
    mounted_models: Mapping[float, torch.nn.Module],
    source_loader: Any,
    target_loader: Any,
    pairing: Any,
    q_all: Mapping[int, Tensor],
    device: Any,
    max_batches: int | None,
    block_group: int = 4,
) -> dict[str, Any]:
    """D_j (transported target), N_j (native displacement) and dT_j (mounted delta) at every target block.

    ``D_j = (aligned f - aligned b) @ Q_j`` at source block ``pairing[j]`` (`aligned_displacement`), ``N_j = T_j(native) -
    T_j(base)``, ``dT_j = T_j(mounted_alpha) - T_j(base)``; sums over all tokens and over CLS only, over the first
    ``max_batches`` batches of the given (search-split) loaders, streamed. Block groups bound activation memory: each
    group re-runs the forwards of its blocks.
    """
    depth = pairing.target_depth
    alphas = sorted(mounted_models)
    sums = BlockSums(range(depth), alphas)
    groups = [
        list(range(g, min(g + max(1, int(block_group)), depth))) for g in range(0, depth, max(1, int(block_group)))
    ]
    src_labels: list[Tensor] = []
    tgt_labels: list[Tensor] = []
    n_images = 0
    n_batches = 0
    for k, (sb_in, tg_in) in enumerate(
        zip(
            _tap(_take(source_loader, max_batches), src_labels),
            _tap(_take(target_loader, max_batches), tgt_labels),
            strict=True,
        )
    ):
        if not torch.equal(src_labels[k], tgt_labels[k]):
            raise ValueError(f"boundary batch {k}: source and target views carry different labels")
        xs, xt = sb_in[0].to(device), tg_in[0].to(device)
        n_images += len(xt)
        n_batches += 1
        for group in groups:
            tgt_req = {j: (j, "boundary") for j in group}
            src_req = {i: (i, "boundary") for i in sorted({int(pairing.pairing[j]) for j in group})}
            sb = capture_batch(source_base_model, xs, src_req, device)
            sf = capture_batch(source_ft_model, xs, src_req, device)
            tb = capture_batch(target_model, xt, tgt_req, device)
            nat = capture_batch(native_model, xt, tgt_req, device)
            mounted = {a: capture_batch(m, xt, tgt_req, device) for a, m in mounted_models.items()}
            for j in group:
                i = int(pairing.pairing[j])
                D = aligned_displacement(sf[i], sb[i], tb[j], q_all[j].to(device=device, dtype=torch.float32))
                sums.update(j, D, nat[j] - tb[j], {a: mounted[a][j] - tb[j] for a in alphas})
            del sb, sf, tb, nat, mounted
    return {
        "alphas": alphas,
        "batches_used": list(range(n_batches)),
        "n_images": n_images,
        "block_group": int(block_group),
        "blocks": sums.summary(),
    }


@torch.no_grad()
def oracle_patching(
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    native_model: torch.nn.Module | None,
    head: VisionHead,
    loaders: Mapping[str, tuple[Any, Any]],
    rows: Mapping[str, Tensor | None],
    states: Mapping[str, Mapping[str, Tensor]],
    n_final: Mapping[str, Tensor] | None,
    pairing: Any,
    q_all: Mapping[int, Tensor],
    blocks: Sequence[int],
    betas: Sequence[float],
    device: Any,
    select_split: str,
) -> dict[str, Any]:
    """Single-block oracle: base target with block ``j``'s output ``T_j + beta * X_j`` (X = D or native N), read out.

    ``rows[split]`` indexes the rows of ``states[split]`` (``t0``, ``d``, ``labels`` of the whole split) that the
    loaders yield, in order (``None``: all rows). ``n_final[split]`` is the native final-CLS displacement aligned to
    those rows (kind ``N`` is skipped without it). Final-CLS change statistics are against ``d`` and ``n``.
    """
    kinds = ["D"] + (["N"] if native_model is not None and n_final is not None else [])
    last = pairing.target_depth - 1
    acc: dict = {}
    for split, (src_loader, tgt_loader) in loaders.items():
        idx = rows[split]
        st = states[split]
        src_labels: list[Tensor] = []
        tgt_labels: list[Tensor] = []
        offset = 0
        for k, (sb_in, tg_in) in enumerate(
            zip(_tap(_take(src_loader, None), src_labels), _tap(_take(tgt_loader, None), tgt_labels), strict=True)
        ):
            if not torch.equal(src_labels[k], tgt_labels[k]):
                raise ValueError(f"oracle batch {k}: source and target views carry different labels")
            xs, xt = sb_in[0].to(device), tg_in[0].to(device)
            b = len(xt)
            sel = torch.arange(offset, offset + b) if idx is None else idx[offset : offset + b]
            offset += b
            labels = st["labels"][sel].to(device)
            t0 = st["t0"][sel].to(device).double()
            d_final = st["d"][sel].to(device).double()
            n_fin = (
                n_final[split][torch.arange(offset - b, offset)].to(device).double() if n_final is not None else None
            )
            src_req = {i: (i, "boundary") for i in sorted({int(pairing.pairing[j]) for j in blocks})}
            sbc = capture_batch(source_base_model, xs, src_req, device)
            sfc = capture_batch(source_ft_model, xs, src_req, device)
            tbc = capture_batch(target_model, xt, {j: (j, "boundary") for j in blocks}, device)
            nac = capture_batch(native_model, xt, {j: (j, "boundary") for j in blocks}, device) if "N" in kinds else {}
            for j in blocks:
                i = int(pairing.pairing[j])
                X = {"D": aligned_displacement(sfc[i], sbc[i], tbc[j], q_all[j].to(device=device, dtype=torch.float32))}
                if "N" in kinds:
                    X["N"] = nac[j] - tbc[j]
                for kind in kinds:
                    for beta in betas:
                        cls, _ = forward_with_block_patch(target_model, xt, j, float(beta) * X[kind], device)
                        pred = head.logits(cls).argmax(-1)
                        change = cls.double() - t0
                        cn = change.norm(dim=-1)
                        dn = d_final.norm(dim=-1)
                        safe = lambda v: v.clamp_min(1e-300)  # noqa: E731
                        rec = acc.setdefault(
                            (kind, j, float(beta), split),
                            {"correct": 0, "n": 0, "over_d": [], "cos_d": [], "cos_n": []},
                        )
                        rec["correct"] += int((pred == labels).sum())
                        rec["n"] += b
                        rec["over_d"].append((cn / safe(dn)).cpu())
                        rec["cos_d"].append(((change * d_final).sum(-1) / safe(cn * dn)).cpu())
                        if n_fin is not None:
                            rec["cos_n"].append(((change * n_fin).sum(-1) / safe(cn * n_fin.norm(dim=-1))).cpu())
        if idx is None and offset != int(st["labels"].shape[0]):
            raise ValueError("oracle sweep did not cover the split")
    out: dict[str, Any] = {"blocks": list(blocks), "betas": [float(b) for b in betas], "kinds": kinds, "results": {}}
    for kind in kinds:
        for j in blocks:
            curves = {
                split: [
                    acc[(kind, j, float(b), split)]["correct"] / acc[(kind, j, float(b), split)]["n"] for b in betas
                ]
                for split in loaders
            }
            bi = select_alpha(betas, curves[select_split])
            final_change = {}
            for b in betas:
                final_change[str(float(b))] = {}
                for split in loaders:
                    rec = acc[(kind, j, float(b), split)]
                    entry = {
                        "norm_over_d_final": _ratio_stats(torch.cat(rec["over_d"])),
                        "cos_with_d_final": _ratio_stats(torch.cat(rec["cos_d"])),
                    }
                    if rec["cos_n"]:
                        entry["cos_with_n_final"] = _ratio_stats(torch.cat(rec["cos_n"]))
                    final_change[str(float(b))][split] = entry
            out["results"][f"{kind}_{j}"] = {
                "kind": kind,
                "block": j,
                "is_last_block": j == last,
                "curves": curves,
                "selected_beta": float(betas[bi]),
                "selection_split": select_split,
                "val_acc": curves["val"][bi] if "val" in curves else None,
                "test_acc": curves["test"][bi] if "test" in curves else None,
                "final_change": final_change,
            }
    return out


def _same_rows_curves(
    arrays: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Tensor]],
    rows: Mapping[str, Tensor | None],
    alphas: Sequence[float],
) -> dict[str, Any]:
    """Accuracy of the main variants (and base) restricted to the subset rows, from the stored full-split predictions."""
    out: dict[str, Any] = {}
    for split, idx in rows.items():
        labels = states[split]["labels"]
        lab = labels if idx is None else labels[idx]
        preds = arrays["preds"][split]
        out[split] = {"base": float(((preds["base"] if idx is None else preds["base"][idx]) == lab).double().mean())}
        for v in ALPHA_VARIANTS:
            p = preds[v] if idx is None else preds[v][:, idx]
            out[split][v] = [float((p[i] == lab).double().mean()) for i in range(len(alphas))]
    return out


def _same_rows_base_full(
    arrays: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Tensor]],
    rows: Mapping[str, Tensor | None],
    alphas: Sequence[float],
    ref_i: int,
) -> dict[str, Any]:
    curves = _same_rows_curves(arrays, states, rows, alphas)
    return {
        split: {
            "base": c["base"],
            "full_at_reference_alpha": c["full"][ref_i],
            "target_at_reference_alpha": c["target"][ref_i],
        }
        for split, c in curves.items()
    }


# ---- gates ---------------------------------------------------------------------------------------------------------


def check_gates(
    *,
    plain_vs_full_logit_diff: Mapping[str, float],
    alpha0_cls_diff: float,
    t0_scale: float,
    full_test_acc_at_ref: float | None,
    full_val_acc_at_ref: float | None,
    base_test_acc: float,
    reference: Mapping[str, Any],
    n_test: int,
    subset: bool,
    logit_tol: float,
    acc_tol: float,
    cls_tol: float,
) -> dict[str, Any]:
    """Gate verdicts (``passed`` is ``None`` when a gate cannot be evaluated; a ``False`` makes the run fail)."""
    gates: dict[str, Any] = {}
    worst = max(plain_vs_full_logit_diff.values()) if plain_vs_full_logit_diff else float("nan")
    gates["g1_full_equals_plain_forward"] = {
        "max_abs_logit_diff_by_alpha": dict(plain_vs_full_logit_diff),
        "tolerance": logit_tol,
        "passed": bool(worst < logit_tol) if plain_vs_full_logit_diff else None,
    }
    rel0 = alpha0_cls_diff / max(t0_scale, 1e-30)
    gates["g0_alpha0_reproduces_base"] = {
        "max_abs_cls_diff": alpha0_cls_diff,
        "max_abs_t0": t0_scale,
        "relative": rel0,
        "tolerance_relative": cls_tol,
        "passed": bool(rel0 <= cls_tol),
    }

    def acc_gate(name: str, ours: float | None, ref: float | None, n: int, applicable: bool, why: str) -> None:
        if subset or not applicable or ours is None or ref is None:
            gates[name] = {"passed": None, "skipped": "subset run" if subset else why, "ours": ours, "reference": ref}
            return
        diff = ours - ref
        gates[name] = {
            "ours": ours,
            "reference": ref,
            "difference": diff,
            "difference_in_images": diff * n,
            "tolerance": acc_tol,
            "exact": bool(diff == 0.0),
            "passed": bool(abs(diff) <= acc_tol),
        }

    acc_gate(
        "g2_full_test_acc_equals_reference",
        full_test_acc_at_ref,
        reference.get("test_rebased"),
        n_test,
        True,
        "reference summary has no per-task test accuracy",
    )
    acc_gate(
        "g2b_full_val_acc_equals_reference",
        full_val_acc_at_ref,
        reference.get("val_rebased"),
        int(reference.get("n_val") or 1),
        reference.get("alpha_search_split") == "val",
        "reference validation accuracy not recorded (alpha_search_split != val or alpha_selection != per_task)",
    )
    baseline_is_zeroshot = float(reference.get("baseline_alpha") or 0.0) == 0.0
    acc_gate(
        "g3_base_test_acc_equals_reference_baseline",
        base_test_acc,
        reference.get("test_baseline"),
        n_test,
        baseline_is_zeroshot,
        f"the reference baseline is untransported at alpha={reference.get('baseline_alpha')}, not the zero-shot base",
    )
    return gates


# ---- one task ------------------------------------------------------------------------------------------------------


def _reference_block(summary: Mapping[str, Any], resolved: Any, task: str, alpha_star: float | None) -> dict[str, Any]:
    test = summary.get("test_results", {}) or {}
    val = summary.get("validation_results") or {}
    return {
        "alpha_star": alpha_star,
        "alpha_search_split": resolved.alpha.search_split,
        "alpha_selection": resolved.alpha.selection,
        "alpha_patience": resolved.alpha.patience,
        "merge_mode": resolved.merge.mode,
        "baseline_label": summary.get("baseline_label"),
        "baseline_alpha": (summary.get("selected_baseline_alpha_by_task") or {}).get(task),
        "test_rebased": (test.get("per_task_rebased") or {}).get(task),
        "test_baseline": (test.get("per_task_baseline_accuracy") or {}).get(task),
        "val_rebased": (val.get("per_task_rebased") or {}).get(task) if val.get("split") == "val" else None,
    }


@torch.no_grad()
def readout_task(
    inp: Any,
    sink: Mapping[str, Any],
    *,
    alpha_grid: Sequence[float] | None,
    gate_batches: int = 2,
    gate_logit_tol: float = 1e-4,
    gate_acc_tol: float = 1e-3,
    gate_cls_tol: float = 1e-5,
    zs_cache_dir: str | None = ZERO_SHOT_CACHE_DIR,
    max_batches: int | None = None,
    native_ckpt_path: str | None = None,
    boundary_max_batches: int | None = 10,
    boundary_block_group: int = 4,
    oracle_patch_blocks: Sequence[int] | str | None = None,
    oracle_beta_grid: Sequence[float] = (0.25, 0.5, 1.0, 1.5),
    eval_subset_size: int | None = None,
    eval_subset_applies_to_all: bool = False,
    eval_subset_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the readout for one task. ``inp`` is the task's `DiagnosticInputs`, ``sink`` the adapter's context sink."""
    device = inp.device
    resolved, summary = sink["resolved"], sink["summary"]
    ctx, clf = sink["eval_context"], sink["target_clf"]
    task = inp.task
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + now - clock
        clock = now

    reference = _reference_block(summary, resolved, task, inp.alpha_star)
    if reference["merge_mode"] != "none":
        raise NotImplementedError(
            f"reference merge_mode={reference['merge_mode']!r}: only single-task runs are read out"
        )
    if inp.alpha_star is None:
        raise ValueError(f"{task}: the reference summary has no selected alpha")
    tau = inp.reference_task_vector
    if tau is None:
        raise ValueError(f"{task}: no saved unit task vector under the reference run's transported_tvs")
    base_keys = set(inp.target_base_state)
    if not set(tau) <= base_keys or not all(k.startswith("visual.") for k in tau):
        raise ValueError(f"{task}: task vector keys are not all visual keys of the target base")
    alphas, grid_info = resolve_alpha_grid(alpha_grid, inp.alpha_star)
    index = {a: i for i, a in enumerate(alphas)}
    ref_i = index[float(inp.alpha_star)]
    gate_alphas = sorted({alphas[ref_i], 1.0})

    target_block = inp.pairing.target_depth - 1
    source_block = int(inp.pairing.pairing[target_block])
    loaders = {
        "val": (ctx.source_loaders.val, ctx.loaders.val),
        "test": (ctx.source_loaders.test, ctx.loaders.test),
    }
    pairing_checks = {s: check_loader_pairing(src, tgt, s) for s, (src, tgt) in loaders.items()}

    # Optional fixed seeded subset (same indices for every variant). ``extra_loaders`` / ``extra_rows`` are what the
    # optional analyses iterate; with eval_subset_applies_to_all the main sweeps use them too.
    if eval_subset_size is not None and max_batches:
        raise ValueError("eval_subset_size and max_batches_per_split are mutually exclusive")
    subset_info: dict[str, Any] | None = None
    extra_loaders = dict(loaders)
    extra_rows: dict[str, Tensor | None] = {s: None for s in loaders}
    if eval_subset_size is not None:
        subset_info = {
            "size": int(eval_subset_size),
            "seed": int(eval_subset_seed),
            "applies_to_all": bool(eval_subset_applies_to_all),
            "splits": {},
        }
        for s, (src, tgt) in loaders.items():
            idx, digest = subset_indices(len(tgt.dataset), eval_subset_size, eval_subset_seed)
            subset_info["splits"][s] = {
                "n_full": len(tgt.dataset),
                "n": len(tgt.dataset) if idx is None else len(idx),
                "indices_sha256": digest,
            }
            if idx is not None:
                extra_loaders[s] = (subset_loader(src, idx), subset_loader(tgt, idx))
                extra_rows[s] = idx
        if eval_subset_applies_to_all:
            loaders = dict(extra_loaders)
            extra_rows = {s: None for s in loaders}

    # Text features exactly as the evaluation builds them (eval_task_top1), on the pristine target base.
    clf.build_zeroshot_text_features(
        list(ctx.classnames), ctx.build_cfg_task, cache_dir=zs_cache_dir, force_rebuild=False
    )
    head = VisionHead.from_classifier(clf, device)
    lap("text_features")

    # Q_{N-1}: the reference fit's own Pass A on its calibration loaders (a re-derivation, not a stored artifact).
    cfg = inp.config
    prepared = prepare_direct_residual_streaming(
        inp.source_base_model,
        inp.target_model,
        inp.source_loader,
        inp.target_loader,
        inp.pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device=device,
        family_adapter=inp.family_adapter,
        source_ft_model=inp.source_ft_model,
        procrustes_source=cfg.procrustes_source,
        alignment_map=cfg.alignment_map,
        alignment_row_weighting=cfg.alignment_row_weighting,
        alignment_seed=cfg.alignment_seed,
        fidelity_alignment_diagnostics=bool(cfg.fidelity_holdout),
    )
    q = prepared["q_by_position"][target_block].float().cpu()
    need_all_q = native_ckpt_path is not None or oracle_patch_blocks is not None
    q_all = {int(j): v.float().cpu() for j, v in prepared["q_by_position"].items()} if need_all_q else {}
    calibration = {k: v for k, v in prepared["calibration"].items() if k != "indices"}
    q_sha = hashlib.sha256(q.contiguous().numpy().tobytes()).hexdigest()
    del prepared
    lap("fit_pass_a_q")

    ref_states: dict[str, dict[str, Tensor]] = {}
    for split, (src_loader, tgt_loader) in loaders.items():
        ref_states[split] = collect_reference_states(
            source_base_model=inp.source_base_model,
            source_ft_model=inp.source_ft_model,
            target_model=inp.target_model,
            source_loader=src_loader,
            target_loader=tgt_loader,
            source_block=source_block,
            target_block=target_block,
            q=q,
            device=device,
            max_batches=max_batches,
        )
        lap(f"reference_sweep_{split}")
    n = {s: int(v["labels"].shape[0]) for s, v in ref_states.items()}
    base_pred = {s: predictions_from_states(head, v["t0"], device) for s, v in ref_states.items()}
    base_acc = {s: float((base_pred[s] == ref_states[s]["labels"]).double().mean()) for s in ref_states}
    d_norm_mean = {s: float(v["d"].double().norm(dim=-1).mean()) for s, v in ref_states.items()}
    t0_scale = max(float(v["t0"].abs().max()) for v in ref_states.values())

    # (A.1) native fine-tuned target: ceiling and per-image native displacement n = native final CLS - t0.
    native = None
    native_rec: dict[str, Any] | None = None
    native_state: dict[str, dict[str, Tensor]] = {}
    if native_ckpt_path is not None:
        native, native_rec = build_native_model(
            inp.target_model, inp.target_base_state, inp.source_base_model.state_dict(), native_ckpt_path
        )
        native = native.to(device)
        native_head = VisionHead(clf, native.visual, head.text_features)
        native_rec["ceiling"] = {}
        native_rec["d_vs_n"] = {}
        native_rec["rows"] = {}
        native_gate: dict[str, float] = {}
        for split, (_, tgt_loader) in extra_loaders.items():
            ridx = extra_rows[split]
            st = ref_states[split]
            lab = st["labels"] if ridx is None else st["labels"][ridx]
            t0_r = st["t0"] if ridx is None else st["t0"][ridx]
            d_r = st["d"] if ridx is None else st["d"][ridx]
            ncls, kept_n = collect_mounted_cls(
                native,
                tgt_loader,
                target_block=target_block,
                device=device,
                expected_labels=lab,
                max_batches=max_batches,
                keep_batches=gate_batches if split == "test" else 0,
            )
            if kept_n:
                plain_n = torch.cat([native_head.finish(native.encode_image(b[0].to(device))) for b in kept_n]).float()
                head_n = native_head.logits(ncls[: plain_n.shape[0]].to(device)).float()
                native_gate[split] = float((plain_n - head_n).abs().max())
            n_r = ncls.double() - t0_r.double()
            native_state[split] = {"n": n_r.float(), "t0": t0_r, "d": d_r, "labels": lab}
            acc_native = float((predictions_from_states(native_head, ncls, device) == lab).double().mean())
            acc_base_head = float(
                (predictions_from_states(head, (t0_r.double() + n_r).float(), device) == lab).double().mean()
            )
            acc_base = float((predictions_from_states(head, t0_r, device) == lab).double().mean())
            native_rec["ceiling"][split] = {
                "native_full_head": acc_native,
                "native_cls_base_head": acc_base_head,
                "base": acc_base,
                "n": int(lab.shape[0]),
            }
            dn_ = d_r.double().norm(dim=-1)
            nn_ = n_r.norm(dim=-1)
            cos_dn = (d_r.double() * n_r).sum(-1) / (dn_ * nn_).clamp_min(1e-300)
            native_rec["d_vs_n"][split] = {
                "cos_d_n": _ratio_stats(cos_dn),
                "norm_d_over_norm_n": _ratio_stats(dn_ / nn_.clamp_min(1e-300)),
            }
            native_state[split].update(cos_dn=cos_dn.float(), d_over_n=(dn_ / nn_.clamp_min(1e-300)).float())
        native_rec["native_head_vs_plain_forward_max_logit_diff"] = native_gate
        lap("native_pass")

    mounted = copy.deepcopy(inp.target_model)
    native_curves = {s: {v: [] for v in ("aligned_native", "orthogonal_native")} for s in native_state}
    native_stats: dict[str, dict[str, list]] = {s: {} for s in native_state}
    native_preds = {s: {v: [] for v in ("aligned_native", "orthogonal_native")} for s in native_state}
    curves = {s: {v: [] for v in ALPHA_VARIANTS} for s in ref_states}
    stats: dict[str, dict[str, list]] = {s: {} for s in ref_states}
    arrays: dict[str, Any] = {
        "alphas": torch.tensor(alphas, dtype=torch.float64),
        "variants": list(VARIANTS),
        "d_norm": {s: v["d"].norm(dim=-1) for s, v in ref_states.items()},
        "labels": {s: v["labels"] for s, v in ref_states.items()},
        "preds": {s: {v: [] for v in VARIANTS} for s in ref_states},
        "c": {s: [] for s in ref_states},
        "orth_over_d": {s: [] for s in ref_states},
        "delta_over_d": {s: [] for s in ref_states},
        "c_global": [],
    }
    for s in ref_states:
        arrays["preds"][s]["base"] = base_pred[s].to(torch.int32)
    gate_logit_diff: dict[str, float] = {}
    alpha0_cls_diff = 0.0
    for alpha in alphas:
        mounted.load_state_dict(axpy_state_dict(inp.target_base_state, tau, alpha), strict=True)
        t1: dict[str, Tensor] = {}
        kept: list[Any] = []
        for split, (_, tgt_loader) in loaders.items():
            keep = gate_batches if (split == "test" and alpha in gate_alphas) else 0
            t1[split], batches = collect_mounted_cls(
                mounted,
                tgt_loader,
                target_block=target_block,
                device=device,
                expected_labels=ref_states[split]["labels"],
                max_batches=max_batches,
                keep_batches=keep,
            )
            kept = batches if split == "test" else kept
        lap("alpha_sweeps")
        if alpha == 0.0:
            alpha0_cls_diff = max(float((t1[s] - ref_states[s]["t0"]).abs().max()) for s in t1)
        # c_g(alpha) on validation only, then applied to both splits.
        c_g = global_coefficient(t1["val"] - ref_states["val"]["t0"], ref_states["val"]["d"])
        arrays["c_global"].append(c_g)
        results = {
            s: evaluate_variants(
                head, ref_states[s]["t0"], ref_states[s]["d"], t1[s], ref_states[s]["labels"], alpha, c_g, device
            )
            for s in ref_states
        }
        for s, res in results.items():
            for v in ALPHA_VARIANTS:
                curves[s][v].append(res["acc"][v])
                arrays["preds"][s][v].append(res["preds"][v].to(torch.int32))
            for key, value in res["stats"].items():
                stats[s].setdefault(key, []).append(value)
            for key in ("c", "orth_over_d", "delta_over_d"):
                arrays[key][s].append(res["per_image"][key])
        for s, nst in native_state.items():
            ridx = extra_rows[s]
            t1_r = t1[s] if ridx is None else t1[s][ridx]
            nres = evaluate_native_variants(head, nst["t0"], nst["d"], nst["n"], t1_r, nst["labels"], device)
            for v, a_ in nres["acc"].items():
                native_curves[s][v].append(a_)
                native_preds[s][v].append(nres["preds"][v].to(torch.int32))
            for key, value in nres["stats"].items():
                native_stats[s].setdefault(key, []).append(value)
        if kept:
            # Gate 1: the plain forward of the mounted model on the same images vs the "full" variant.
            plain = torch.cat([head.finish(mounted.encode_image(b[0].to(device))) for b in kept]).float()
            rows = plain.shape[0]
            full = head.logits(
                (ref_states["test"]["t0"][:rows].double() + results["test"]["delta"][:rows]).float().to(device)
            ).float()
            gate_logit_diff[f"{alpha:.17g}"] = float((plain - full).abs().max())
        lap("variants")
    del mounted

    for s in ref_states:
        for v in ("c", "orth_over_d", "delta_over_d"):
            arrays[v][s] = torch.stack(arrays[v][s])  # [A, n]
        arrays["preds"][s] = {v: (torch.stack(p) if isinstance(p, list) else p) for v, p in arrays["preds"][s].items()}
    arrays["c_global"] = torch.tensor(arrays["c_global"], dtype=torch.float64)
    for s in curves:
        curves[s]["base"] = [base_acc[s]] * len(alphas)

    # Alpha is selected on the split the reference run searched on; test accuracy is read at that alpha.
    sel_split = resolved.alpha.search_split
    selection: dict[str, Any] = {}
    for v in ALPHA_VARIANTS:
        i = select_alpha(alphas, curves[sel_split][v])
        selection[v] = {
            "selected_alpha": alphas[i],
            "selected_index": i,
            "selection_split": sel_split,
            "val_acc": curves["val"][v][i],
            "test_acc": curves["test"][v][i],
            "test_acc_at_reference_alpha": curves["test"][v][ref_i],
            "val_acc_at_reference_alpha": curves["val"][v][ref_i],
        }
    selection["base"] = {"val_acc": base_acc["val"], "test_acc": base_acc["test"], "selected_alpha": None}

    extras: dict[str, Any] = {}
    if subset_info is not None:
        subset_info["reference_curves_same_rows"] = _same_rows_curves(arrays, ref_states, extra_rows, alphas)
        extras["subset"] = subset_info
    if native is not None:
        for s_ in native_state:
            arrays.setdefault("native", {})[s_] = {
                "cos_d_n": native_state[s_]["cos_dn"],
                "d_over_n": native_state[s_]["d_over_n"],
                "n_norm": native_state[s_]["n"].norm(dim=-1),
                "preds": {v: torch.stack(p_) for v, p_ in native_preds[s_].items()},
            }
        native_rec["curves"] = native_curves
        native_rec["stats"] = native_stats
        native_rec["selection"] = {}
        for v in ("aligned_native", "orthogonal_native"):
            i = select_alpha(alphas, native_curves[sel_split][v])
            native_rec["selection"][v] = {
                "selected_alpha": alphas[i],
                "selected_index": i,
                "selection_split": sel_split,
                "val_acc": native_curves["val"][v][i],
                "test_acc": native_curves["test"][v][i],
                "test_acc_at_reference_alpha": native_curves["test"][v][ref_i],
            }
        extras["native"] = native_rec
        # (A.3) all-block boundary comparison on the search-split images (first boundary_max_batches batches).
        ref_alphas = sorted({alphas[ref_i], 1.0})
        mounted_by_alpha = {}
        for a_ in ref_alphas:
            m_ = copy.deepcopy(inp.target_model)
            m_.load_state_dict(axpy_state_dict(inp.target_base_state, tau, a_), strict=True)
            mounted_by_alpha[a_] = m_.to(device).eval()
        src_l, tgt_l = loaders[sel_split]
        extras["boundary_comparison"] = {
            "split": sel_split,
            **boundary_comparison(
                source_base_model=inp.source_base_model,
                source_ft_model=inp.source_ft_model,
                target_model=inp.target_model,
                native_model=native,
                mounted_models=mounted_by_alpha,
                source_loader=src_l,
                target_loader=tgt_l,
                pairing=inp.pairing,
                q_all=q_all,
                device=device,
                max_batches=boundary_max_batches,
                block_group=boundary_block_group,
            ),
        }
        del mounted_by_alpha
        lap("boundary_comparison")
    if oracle_patch_blocks is not None:
        blocks_o = resolve_oracle_blocks(oracle_patch_blocks, inp.pairing.target_depth)
        o_states = {s_: ref_states[s_] for s_ in extra_loaders}
        extras["oracle"] = oracle_patching(
            source_base_model=inp.source_base_model,
            source_ft_model=inp.source_ft_model,
            target_model=inp.target_model,
            native_model=native,
            head=head,
            loaders=extra_loaders,
            rows=extra_rows,
            states=o_states,
            n_final=({s_: native_state[s_]["n"] for s_ in native_state} if native is not None else None),
            pairing=inp.pairing,
            q_all=q_all,
            blocks=blocks_o,
            betas=[float(b) for b in oracle_beta_grid],
            device=device,
            select_split=sel_split,
        )
        lap("oracle_patching")
        same_rows = _same_rows_base_full(arrays, ref_states, extra_rows, alphas, ref_i)
        extras["oracle"]["same_rows_references"] = same_rows
    del native

    reference["n_val"] = n["val"]
    gates = check_gates(
        plain_vs_full_logit_diff=gate_logit_diff,
        alpha0_cls_diff=alpha0_cls_diff,
        t0_scale=t0_scale,
        full_test_acc_at_ref=curves["test"]["full"][ref_i],
        full_val_acc_at_ref=curves["val"]["full"][ref_i],
        base_test_acc=base_acc["test"],
        reference=reference,
        n_test=n["test"],
        subset=bool(max_batches) or bool(eval_subset_applies_to_all and eval_subset_size is not None),
        logit_tol=gate_logit_tol,
        acc_tol=gate_acc_tol,
        cls_tol=gate_cls_tol,
    )
    if native_rec is not None:
        diffs = native_rec["native_head_vs_plain_forward_max_logit_diff"]
        gates["g1n_native_head_equals_plain_forward"] = {
            "max_abs_logit_diff": diffs,
            "tolerance": gate_logit_tol,
            "passed": bool(max(diffs.values()) < gate_logit_tol) if diffs else None,
        }
    failed = [name for name, g in gates.items() if g.get("passed") is False]
    record = {
        "task": task,
        "family": "vision",
        "alphas": alphas,
        "alpha_grid": grid_info,
        "reference": reference,
        "reference_alpha_index": ref_i,
        "n": n,
        "subset_max_batches": max_batches,
        "curves": curves,
        "selection": selection,
        "stats": stats,
        "gates": gates,
        "failed_gates": failed,
        "pooling": head.pooling,
        "blocks": {
            "target_block": target_block,
            "source_block": source_block,
            "target_depth": inp.pairing.target_depth,
        },
        "q": {"shape": list(q.shape), "sha256": q_sha, "calibration": calibration},
        "pairing_checks": pairing_checks,
        "mean_d_norm": d_norm_mean,
        "provenance": inp.provenance,
        "timings_seconds": timings,
        "extras": extras,
    }
    return record, arrays


# ---- entry point ---------------------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_new(path: Path, text: str) -> None:
    with open(path, "x") as handle:  # "x": never overwrite a result
        handle.write(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    if cfg.get("family", "vision") != "vision":
        raise NotImplementedError("readout is vision-only")
    out_dir = Path(cfg["out_dir"])
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; readout results are write-once")
    out_dir.mkdir(parents=True)

    from .vision import build_vision_inputs

    run_meta: dict[str, Any] = {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": cfg,
        "code_fingerprint": _code_fingerprint(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "torch": torch.__version__,
        "tasks": {},
    }
    vision_cfg = {**cfg, "generic_num_batches": 0}
    sink: dict[str, Any] = {}
    started = time.perf_counter()
    for inp in build_vision_inputs(vision_cfg, args.device, context_sink=sink):
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        record, arrays = readout_task(
            inp,
            sink,
            alpha_grid=cfg.get("alpha_grid"),
            gate_batches=int(cfg.get("gate_batches", 2)),
            gate_logit_tol=float(cfg.get("gate_logit_tol", 1e-4)),
            gate_acc_tol=float(cfg.get("gate_acc_tol", 1e-3)),
            gate_cls_tol=float(cfg.get("gate_cls_tol", 1e-5)),
            zs_cache_dir=cfg.get("zs_cache_dir", ZERO_SHOT_CACHE_DIR),
            max_batches=cfg.get("max_batches_per_split"),
            native_ckpt_path=(
                cfg["native_target_ckpt_template"].format(task=inp.task)
                if cfg.get("native_target_ckpt_template")
                else None
            ),
            boundary_max_batches=cfg.get("boundary_max_batches", 10),
            boundary_block_group=int(cfg.get("boundary_block_group", 4)),
            oracle_patch_blocks=cfg.get("oracle_patch_blocks"),
            oracle_beta_grid=cfg.get("oracle_beta_grid", [0.25, 0.5, 1.0, 1.5]),
            eval_subset_size=cfg.get("eval_subset_size"),
            eval_subset_applies_to_all=bool(cfg.get("eval_subset_applies_to_all", False)),
            eval_subset_seed=int(cfg.get("eval_subset_seed", 0)),
        )
        task_dir = out_dir / inp.task
        task_dir.mkdir()
        _write_new(task_dir / "readout.json", json.dumps(_jsonable(record), indent=1, sort_keys=True))
        torch.save(arrays, task_dir / "arrays.pt")
        run_meta["tasks"][inp.task] = {
            "seconds": time.perf_counter() - t0,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gates": {k: g.get("passed") for k, g in record["gates"].items()},
            "failed_gates": record["failed_gates"],
        }
        run_meta["total_seconds"] = time.perf_counter() - started
        (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))
        print(f"[readout] {inp.task}: done in {run_meta['tasks'][inp.task]['seconds']:.0f}s", flush=True)
        if record["failed_gates"]:
            raise GateError(f"correctness gate(s) failed for {inp.task}: {record['failed_gates']}; see {task_dir}")
    run_meta["total_seconds"] = time.perf_counter() - started
    (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
