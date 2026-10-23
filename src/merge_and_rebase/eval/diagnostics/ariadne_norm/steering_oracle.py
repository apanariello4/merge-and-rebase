"""Per-block steering oracle (vision): the target base with the transported source change added at EVERY block.

Opt-in mode of ``readout`` (config field ``steer_mode``; ``"last_block"`` = the existing readout, unchanged):

``python -m merge_and_rebase.eval.diagnostics.ariadne_norm.readout --config <json> --device cuda``

For each image ``x`` (source models run on ``x`` at evaluation time, same sequential source/target views and the same
seeded ``eval_subset`` rows as the readout), a forward hook on every target block ``j`` adds ``alpha * v_j(x)`` to the
block output (all tokens, residual stream); the classifier reads the final CLS of the steered model. With
``i = pairing[j]``, ``Q_j`` the fit's own block-boundary Procrustes map (re-derived by ``prepare_direct_residual_streaming``
on the fit's calibration loaders, exactly as ``readout_task`` re-derives ``Q_{N-1}``), ``f_i`` / ``b_i`` the source
fine-tuned / base block-``i`` outputs on ``x`` (``_aligned`` to the target token grid):

- ``block_total``:     ``v_j = (f_i - b_i) Q_j``  (the total target ``D_j`` at every block; double counts by design);
- ``block_increment``: ``v_j = [(f_i - b_i) - (f_p - b_p)] Q_j / g(j)`` with ``p = pairing[p(j)]`` (``p(j)`` = last
  target position before j's run of blocks sharing one source block; first run: ``(f_i - b_i) Q_j / g(j)``) and ``g(j)``
  the run length -- ``increment_references`` / ``target_groups`` of ``_ariadne/config.py``, as ``block_target='increment'``.

One alpha per task (shared by all blocks), chosen on the reference's ``alpha_search_split`` with ``select_alpha`` over
the same grid logic as the readout (first maximum). Test accuracy is read at that alpha; alpha = 0 is the untouched
target (gate: the hooked alpha = 0 forward equals the hook-free forward).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from merge_and_rebase.rebase.methods._ariadne.config import increment_references, target_groups
from merge_and_rebase.rebase.methods._ariadne.layouts import _aligned, _layout_for
from merge_and_rebase.rebase.methods._ariadne.streaming import prepare_direct_residual_streaming
from merge_and_rebase.rebase.methods._shared import _to_tokens

from . import readout as R

Tensor = torch.Tensor

STEER_MODES = ("last_block", "block_increment", "block_total")


def validate_steer_mode(mode: Any) -> str:
    if mode not in STEER_MODES:
        raise ValueError(f"steer_mode must be one of {STEER_MODES}, got {mode!r}")
    return str(mode)


def steering_references(pairing: Sequence[int], mode: str) -> dict[int, tuple[int | None, int]]:
    """``{j: (previous source block or None, g(j))}``: ``block_total`` -> ``(None, 1)``; ``block_increment`` follows
    ``increment_references`` / ``target_groups`` (same pairing validation: non-decreasing)."""
    pairing = [int(i) for i in pairing]
    if mode == "block_total":
        return {j: (None, 1) for j in range(len(pairing))}
    if mode != "block_increment":
        raise ValueError(f"no per-block steering vectors for steer_mode={mode!r}")
    prev = increment_references(pairing)
    groups = target_groups(pairing)
    return {j: (None if prev[j] is None else pairing[prev[j]], groups[j][0]) for j in range(len(pairing))}


def source_blocks_needed(pairing: Sequence[int], refs: Mapping[int, tuple[int | None, int]]) -> list[int]:
    return sorted({int(i) for i in pairing} | {p for p, _ in refs.values() if p is not None})


def steering_vectors(
    sf: Mapping[int, Tensor],
    sb: Mapping[int, Tensor],
    target_tokens: Tensor,
    q_all: Mapping[int, Tensor],
    pairing: Sequence[int],
    refs: Mapping[int, tuple[int | None, int]],
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """``({j: v_j}, {j: D_j})`` with ``D_j = (f_i - b_i) Q_j`` and ``v_j`` the (increment) steering vector, as the fit
    builds ``desired``: ``_aligned`` first, one Q_j, ``/ g(j)`` on the increment."""
    v: dict[int, Tensor] = {}
    total: dict[int, Tensor] = {}
    for j, (prev_src, g) in refs.items():
        i = int(pairing[j])
        q = q_all[j].to(device=target_tokens.device, dtype=torch.float32)
        b = _aligned([sb[i]], [target_tokens])[0]
        f = _aligned([sf[i]], [target_tokens])[0]
        total[j] = (f - b) @ q
        if prev_src is None:
            v[j] = total[j] / g if g != 1 else total[j]
        else:
            b_p = _aligned([sb[prev_src]], [target_tokens])[0]
            f_p = _aligned([sf[prev_src]], [target_tokens])[0]
            v[j] = ((f - b) - (f_p - b_p)) @ q / g
    return v, total


@torch.no_grad()
def forward_with_block_patches(
    model: torch.nn.Module, images: Tensor, patches: Mapping[int, Tensor], device: Any
) -> Tensor:
    """Final-block CLS ``[B, D]`` (float32) of ``model`` with block ``j``'s output replaced by ``output + patches[j]``
    for every ``j`` in ``patches`` (``[B, T, D]``, all tokens). The final-CLS hook is registered after the patch hooks,
    so it sees the patched last block (same convention as ``readout.forward_with_block_patch``)."""
    layout = _layout_for(None)
    blocks = layout.blocks(model)
    last = len(blocks) - 1
    batch_first = bool(getattr(model.visual.transformer, "batch_first", True))
    batch_size = len(images)
    seen: dict[str, Tensor] = {}

    def make_hook(patch: Tensor):
        def hook(_module, _inputs, out):
            tup = isinstance(out, tuple)
            x = out[0] if tup else out
            p = patch if batch_first else patch.transpose(0, 1)
            if x.shape != p.shape:
                raise ValueError(f"patch {tuple(p.shape)} does not match the block output {tuple(x.shape)}")
            x = x + p.to(x.dtype)
            return (x, *out[1:]) if tup else x

        return hook

    def cls_hook(_m, _i, out):
        t = _to_tokens((out[0] if isinstance(out, tuple) else out).detach(), batch_size=batch_size)
        seen["cls"] = t.float().clone()[:, 0]

    handles = []
    try:
        for j in sorted(patches):
            handles.append(layout.block_module(blocks[j]).register_forward_hook(make_hook(patches[j])))
        handles.append(layout.block_module(blocks[last]).register_forward_hook(cls_hook))
        layout.forward(model, (images,), device)
    finally:
        for handle in handles:
            handle.remove()
    return seen["cls"]


@torch.no_grad()
def steering_oracle_task(
    inp: Any,
    sink: Mapping[str, Any],
    *,
    steer_mode: str,
    alpha_grid: Sequence[float] | None,
    zs_cache_dir: str | None = R.ZERO_SHOT_CACHE_DIR,
    max_batches: int | None = None,
    eval_subset_size: int | None = None,
    eval_subset_seed: int = 0,
    micro_batch: int = 32,
    alpha_tol: float = 1e-5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Steering oracle for one task: ``(record, arrays)`` shaped like ``readout_task``'s so ``readout.main`` writes them."""
    mode = validate_steer_mode(steer_mode)
    if mode == "last_block":
        raise ValueError("steering_oracle_task is for the per-block modes; last_block is readout_task")
    if eval_subset_size is not None and max_batches:
        raise ValueError("eval_subset_size and max_batches_per_split are mutually exclusive")
    device = inp.device
    resolved, ctx, clf = sink["resolved"], sink["eval_context"], sink["target_clf"]
    if resolved.merge.mode != "none":
        raise NotImplementedError(f"reference merge_mode={resolved.merge.mode!r}: only single-task runs are supported")
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + now - clock
        clock = now

    alphas, grid_info = R.resolve_alpha_grid(alpha_grid, None)
    pairing = [int(i) for i in inp.pairing.pairing]
    depth = int(inp.pairing.target_depth)
    refs = steering_references(pairing, mode)
    src_blocks = source_blocks_needed(pairing, refs)
    loaders = {
        "val": (ctx.source_loaders.val, ctx.loaders.val),
        "test": (ctx.source_loaders.test, ctx.loaders.test),
    }
    pairing_checks = {s: R.check_loader_pairing(a, b, s) for s, (a, b) in loaders.items()}
    subset_info: dict[str, Any] | None = None
    if eval_subset_size is not None:  # same rows as the readout with eval_subset_applies_to_all
        subset_info = {"size": int(eval_subset_size), "seed": int(eval_subset_seed), "splits": {}}
        for s, (src, tgt) in list(loaders.items()):
            idx, digest = R.subset_indices(len(tgt.dataset), eval_subset_size, eval_subset_seed)
            subset_info["splits"][s] = {
                "n_full": len(tgt.dataset),
                "n": len(tgt.dataset) if idx is None else len(idx),
                "indices_sha256": digest,
            }
            if idx is not None:
                loaders[s] = (R.subset_loader(src, idx), R.subset_loader(tgt, idx))

    clf.build_zeroshot_text_features(
        list(ctx.classnames), ctx.build_cfg_task, cache_dir=zs_cache_dir, force_rebuild=False
    )
    head = R.VisionHead.from_classifier(clf, device)
    lap("text_features")

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
    q_all = {int(j): v.float().cpu() for j, v in prepared["q_by_position"].items()}
    calibration = {k: v for k, v in prepared["calibration"].items() if k != "indices"}
    del prepared
    if any(j not in q_all for j in range(depth)):
        raise ValueError(f"no Q_j for target blocks {[j for j in range(depth) if j not in q_all]}")
    lap("fit_pass_a_q")

    sb_model, sf_model, tgt_model = inp.source_base_model, inp.source_ft_model, inp.target_model
    src_req = {i: (i, "boundary") for i in src_blocks}
    a_n = len(alphas)
    correct = {s: torch.zeros(a_n, dtype=torch.long) for s in loaders}
    base_correct = dict.fromkeys(loaders, 0)
    count = dict.fromkeys(loaders, 0)
    preds = {s: [] for s in loaders}
    base_preds: dict[str, list[Tensor]] = {s: [] for s in loaders}
    labels_all: dict[str, list[Tensor]] = {s: [] for s in loaders}
    v_norm = {s: torch.zeros(depth, dtype=torch.float64) for s in loaders}
    d_norm = {s: torch.zeros(depth, dtype=torch.float64) for s in loaders}
    tokens_seen = dict.fromkeys(loaders, 0)
    alpha0_diff = 0.0
    t0_scale = 0.0
    n_tgt_tokens: int | None = None
    zero_i = alphas.index(0.0)
    for split, (src_loader, tgt_loader) in loaders.items():
        src_labels: list[Tensor] = []
        tgt_labels: list[Tensor] = []
        for k, (s_in, t_in) in enumerate(
            zip(
                R._tap(R._take(src_loader, max_batches), src_labels),
                R._tap(R._take(tgt_loader, max_batches), tgt_labels),
                strict=True,
            )
        ):
            if not torch.equal(src_labels[k], tgt_labels[k]):
                raise ValueError(f"{split} batch {k}: source and target views carry different labels")
            xs_all, xt_all = s_in[0].to(device), t_in[0].to(device)
            lab = tgt_labels[k].to(device)
            for lo in range(0, len(xt_all), int(micro_batch)):
                xs, xt, y = xs_all[lo : lo + micro_batch], xt_all[lo : lo + micro_batch], lab[lo : lo + micro_batch]
                if n_tgt_tokens is None:
                    n_tgt_tokens = int(R.capture_batch(tgt_model, xt[:1], {0: (0, "boundary")}, device)[0].shape[1])
                sbc = R.capture_batch(sb_model, xs, src_req, device)
                sfc = R.capture_batch(sf_model, xs, src_req, device)
                dummy = torch.empty(len(xt), n_tgt_tokens, 1, device=device)
                v, total = steering_vectors(sfc, sbc, dummy, q_all, pairing, refs)
                for j in range(depth):
                    v_norm[split][j] += float(v[j].double().norm(dim=-1).sum())
                    d_norm[split][j] += float(total[j].double().norm(dim=-1).sum())
                tokens_seen[split] += len(xt) * n_tgt_tokens
                plain = forward_with_block_patches(tgt_model, xt, {}, device)
                t0_scale = max(t0_scale, float(plain.abs().max()))
                bp = head.logits(plain).argmax(-1)
                base_correct[split] += int((bp == y).sum())
                base_preds[split].append(bp.cpu().to(torch.int32))
                batch_preds = []
                for ai, a in enumerate(alphas):
                    cls = forward_with_block_patches(tgt_model, xt, {j: float(a) * v[j] for j in range(depth)}, device)
                    if ai == zero_i:
                        alpha0_diff = max(alpha0_diff, float((cls - plain).abs().max()))
                    p = head.logits(cls).argmax(-1)
                    correct[split][ai] += int((p == y).sum())
                    batch_preds.append(p.cpu().to(torch.int32))
                preds[split].append(torch.stack(batch_preds))
                labels_all[split].append(y.cpu())
                count[split] += len(y)
        lap(f"sweep_{split}")

    curves = {s: [int(c) / count[s] for c in correct[s]] for s in loaders}
    base_acc = {s: base_correct[s] / count[s] for s in loaders}
    sel_split = resolved.alpha.search_split
    i = R.select_alpha(alphas, curves[sel_split])
    selection = {
        "selected_alpha": alphas[i],
        "selected_index": i,
        "selection_split": sel_split,
        "val_acc": curves["val"][i],
        "test_acc": curves["test"][i],
        "untouched_target_val_acc": base_acc["val"],
        "untouched_target_test_acc": base_acc["test"],
    }
    block_norms = {
        s: {
            "mean_v_norm": (v_norm[s] / tokens_seen[s]).tolist(),
            "mean_D_norm": (d_norm[s] / tokens_seen[s]).tolist(),
            "v_over_D": (v_norm[s] / d_norm[s].clamp_min(1e-300)).tolist(),
        }
        for s in loaders
    }
    tol = max(alpha_tol * max(t0_scale, 1.0), 1e-6)
    gates = {
        "g_alpha0_equals_untouched_target": {
            "max_abs_cls_diff": alpha0_diff,
            "tolerance": tol,
            "passed": bool(alpha0_diff <= tol),
        }
    }
    record = {
        "task": inp.task,
        "family": "vision",
        "steer_mode": mode,
        "alphas": alphas,
        "alpha_grid": grid_info,
        "n": dict(count),
        "curves": curves,
        "selection": selection,
        "block_norms": block_norms,
        "references": {
            str(j): {"source_block": pairing[j], "previous_source_block": p, "g": g} for j, (p, g) in refs.items()
        },
        "gates": gates,
        "failed_gates": [name for name, g in gates.items() if g.get("passed") is False],
        "subset": subset_info,
        "pairing_checks": pairing_checks,
        "q": {"calibration": calibration},
        "provenance": inp.provenance,
        "timings_seconds": timings,
    }
    arrays = {
        "alphas": torch.tensor(alphas, dtype=torch.float64),
        "labels": {s: torch.cat(labels_all[s]) for s in loaders},
        "preds": {s: torch.cat(preds[s], dim=1) for s in loaders},  # [A, n]
        "untouched_preds": {s: torch.cat(base_preds[s]) for s in loaders},
    }
    return record, arrays
