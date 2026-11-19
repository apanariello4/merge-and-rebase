"""Joint function-space refinement of the ``mlp.c_proj`` updates of ``anchors`` (``refine='joint'``), label-free.

Every depth fit of ``synthesis.py`` (independent / tracking / spread, ridge or ridge_preserve) solves one block at a
time on frozen inputs. Here the whole set ``{(dW_j, beta_j)}`` is trained jointly by gradient descent, starting from
the fit's solution, against the quantity that matters at alpha = 1: what the updated network does to the residual
stream. Nothing but the ``c_proj`` updates is trainable and no label is used.

Notation. ``S_j(x; theta)`` is the residual stream at the output boundary of target block ``j`` (``[B, T, d]``, row =
(image b, token t)) of the target with ``c_proj_j`` replaced by ``(W_j + dW_j) x + (b_j + beta_j)``; ``S_base,j`` the same
at ``dW = 0``. ``D_j[k]`` (``desired[j][k]``, ``[B, T, d]``) is the wanted CUMULATIVE change of the stream at that
boundary on calibration batch ``k``, row-aligned with ``captured['target_batches'][k]`` (image b of the batch, token t).

Loss (all terms are dimensionless relative errors, so depths weigh equally and rho / prox have no unit):

    L_task = sum_{j in M_task} w_j * E_rows ||S_j(x) - S_base,j(x) - D_j(x)||^2 / c_j,      c_j = E_rows ||D_j||^2
    L_keep = rho * sum_{j in M}   E_rows ||S_j(g) - S_base,j(g)||^2 / v_j,                   v_j = E_rows ||S_base,j(g) - mu_j||^2
    L_prox = prox * sum_j ||dW_j - dW_j,init||_F^2 / ||W_j||_F^2
    L      = L_task + L_keep + L_prox

With ``refine_target='final'`` rho is scaled by ``|M_task| / |M|`` (``rho_effective``), so the task / keep balance of
'all_depths' is kept. The generic held-out batch is, when the caller provides it, one O2's G did not use.

``E_rows`` is the mean over the (unmasked) token rows of the micro-batch; ``c_j`` is taken once over the training
calibration rows and ``v_j`` once over the training generic rows (the variance of the base stream around its mean,
``mu_j`` = row mean), both constants, so the stochastic loss is an unbiased estimate of a fixed objective. ``x`` are
calibration rows (not the held-out batches), ``g`` generic rows (anchor-pool images, the source of O2's G). The keep term
is stream-level: it also counts drift caused by upstream blocks, unlike O2's per-block penalty. ``M`` = measurement
blocks (``refine_depths``); ``M_task`` = ``M`` (``refine_target='all_depths'``) or its last block (``'final'``).

Optimiser: Adam, per-element learning rate ``lr * rms(dW_j,init)`` (weights; the bias with rms(beta_init)), linear warmup then cosine decay
to 0, micro-batches of ``batch`` images, one calibration and one generic micro-batch per step, seeded. The model is
frozen; only ``dW, beta`` (float32) receive gradients, ``c_proj`` is evaluated as ``F.linear(x, W + dW, b + beta)`` through
an instance-level ``forward`` (so ``torch.utils.checkpoint`` recompute sees the same parameters). The forward stops at
the deepest block that a loss needs. Selection (label-free): the held-out objective ``L_task`` (calibration batches held out)
``+ rho * sum_j drift_j`` (the held-out generic batch; with rho = 0 just ``L_task``) is evaluated at step 0 (the fit's own solution), every ``eval_every`` steps and at the end; the best checkpoint (step 0 included, so
the result is never worse than the start on the held-out rows) is returned.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .._ariadne.layouts import _layout_for
from .._shared import _to_tokens

Tensor = torch.Tensor


class _Stop(Exception):
    """Raised by the capture hook of the deepest needed block: the rest of the forward is not needed."""


def refine_blocks(depths, available: Sequence[int], target_depth: int) -> list[int]:
    """Measurement blocks ``M``: ``'all'`` = every block of ``available``; else ``round(s (L - 1))`` of each ``s``."""
    avail = sorted(int(j) for j in available)
    if isinstance(depths, str):
        if depths != "all":
            raise ValueError("refine_depths must be 'all' or a list of relative depths")
        return avail
    blocks = sorted({int(round(float(s) * (target_depth - 1))) for s in depths})
    missing = [j for j in blocks if j not in avail]
    if not blocks or missing:
        raise ValueError(f"refine_depths {tuple(depths)} map to blocks {blocks}, not all fitted ({avail})")
    return blocks


def _masked_mean(e: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of the per-row squared errors ``e`` ``[b, T]`` over the unmasked rows (``None``: all rows)."""
    if mask is None:
        return e.mean()
    m = mask.to(device=e.device, dtype=e.dtype)
    return (e * m).sum() / m.sum().clamp_min(1.0)


def _rows(mask_by_block: Mapping[int, Sequence[Tensor | None]] | None, j: int, k: int, idx=None):
    if mask_by_block is None or mask_by_block.get(j) is None:
        return None
    m = mask_by_block[j][k]
    if m is None:
        return None
    return m if idx is None else m[idx]


@contextmanager
def _installed(model, layout, params: Mapping[int, tuple[Tensor, Tensor]], upto: int, use_checkpoint: bool):
    """``c_proj_j(x) = F.linear(x, W_j + dW_j, b_j + beta_j)`` for ``j in params``; optional checkpointing of the blocks
    ``0..upto``. Instance-level ``forward`` attributes, removed on exit (the state dict never changes); re-entrant:
    modules already patched by an enclosing context are left to it."""
    blocks = layout.blocks(model)
    patched: list[Any] = []
    try:
        for j, (dw, db) in params.items():
            mod = layout.proj_module(blocks[j])
            if "forward" in mod.__dict__:
                continue  # re-entrant: already installed by an enclosing context, which removes it
            if mod.bias is None:
                raise ValueError(f"refine: block {j} c_proj has no bias (missing_bias='materialize' is not supported)")

            def fwd(x, _m=mod, _dw=dw, _db=db):
                return F.linear(x, _m.weight + _dw.to(_m.weight.dtype), _m.bias + _db.to(_m.bias.dtype))

            mod.forward = fwd
            patched.append(mod)
        if use_checkpoint:
            for j in range(upto + 1):
                inner = layout.block_module(blocks[j])
                if "forward" in inner.__dict__:
                    continue
                orig = inner.forward

                def ckpt(*a, _orig=orig, **k):
                    if torch.is_grad_enabled():
                        return checkpoint(_orig, *a, use_reentrant=False, **k)
                    return _orig(*a, **k)

                inner.forward = ckpt
                patched.append(inner)
        yield
    finally:
        for mod in patched:
            if "forward" in mod.__dict__:
                del mod.forward


def _forward(model, layout, images: Tensor, need: Sequence[int], device, amp: bool) -> dict[int, Tensor]:
    """Boundaries ``{j: [b, T, d] float32}`` of the blocks ``need`` (graph kept when grad is enabled)."""
    blocks = layout.blocks(model)
    last = max(need)
    outs: dict[int, Tensor] = {}
    bs = int(images.shape[0])

    def make(j):
        def hook(_m, _i, value):
            value = value[0] if isinstance(value, tuple) else value
            outs[j] = _to_tokens(value, batch_size=bs).float()
            if j == last:
                raise _Stop

        return hook

    handles = [layout.block_module(blocks[j]).register_forward_hook(make(j)) for j in need]
    try:
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=bool(amp)):
            layout.forward(model, (images,), device)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    return outs


def refine_joint(
    target_model,
    target_base_sd: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    corrections: Mapping[str, Tensor],
    generic_batches: Sequence[Any] | None,
    *,
    generic_holdout: Sequence[Any] | None = None,
    warmup: float = 0.1,
    blocks: Sequence[int],
    target: str = "all_depths",
    rho: float = 0.1,
    prox: float = 0.0,
    steps: int = 300,
    lr: float = 1e-3,
    batch: int = 8,
    holdout: int = 2,
    eval_every: int = 25,
    amp: bool = False,
    use_checkpoint: bool = True,
    seed: int = 0,
    device="cpu",
    norm_filter: Mapping[str, Any] | None = None,
    row_masks: Mapping[int, Sequence[Tensor | None]] | None = None,
    key_fmt: str = "visual.transformer.resblocks.{}.mlp.c_proj",
) -> tuple[dict[str, Tensor], dict[int, dict[str, Any]], dict[str, Any]]:
    """Refine ``corrections`` (``{key.weight: dW [d_out, d_in], key.bias: beta}``, the output of a depth fit) jointly.

    ``blocks``: measurement blocks ``M`` (see ``refine_blocks``). ``generic_batches``: batches ``(images, ...)`` of the
    anchor pool (``None`` or ``rho == 0``: no keep term); with two or more the last one is held out of the training
    (drift diagnostics), with one it is used for both. ``row_masks``: optional ``{j: [bool [B, T] per calibration batch]}``,
    rows outside the mask are ignored by every calibration loss and statistic (``None``: all rows, as in the fits).
    ``norm_filter`` (``{'mu': {j: [d]}, 'k', 'median': {j}}``): generic rows whose base boundary norm ``||x - mu_j||``
    exceeds ``k * median_j`` are masked, as in O2's G.

    Returns ``(new_corrections, per_block, info)``; ``per_block[j]`` and ``info`` are JSON-ready diagnostics. The model
    is restored (weights, device, train mode, ``requires_grad``).
    """
    if target not in {"all_depths", "final"}:
        raise ValueError("refine_target must be 'all_depths' or 'final'")
    layout = _layout_for(None)
    cal = captured["target_batches"]
    base_out = captured["target_base_outputs_by_position"]
    n = len(cal)
    if not 1 <= holdout < n:
        raise ValueError(f"refine_holdout_batches={holdout} needs 1 <= holdout < the {n} calibration batches")
    train_idx, hold_idx = list(range(n - holdout)), list(range(n - holdout, n))
    m_blocks = sorted(int(j) for j in blocks)
    task_blocks = m_blocks if target == "all_depths" else [m_blocks[-1]]
    fitted = sorted(desired)
    use_keep = generic_batches is not None and len(generic_batches) > 0 and float(rho) > 0.0
    rho_e = float(rho) * len(task_blocks) / len(m_blocks)  # keeps the task / keep balance when only some blocks are in L_task
    dev = torch.device(device)
    started = time.perf_counter()

    # ---- constants c_j (calibration, training rows) -------------------------------------------------------------
    c_const: dict[int, float] = {}
    for j in m_blocks:
        tot, cnt = 0.0, 0.0
        for k in train_idx:
            d = desired[j][k].to(torch.float64)
            e = d.pow(2).sum(-1)
            m = _rows(row_masks, j, k)
            tot += float(e.sum() if m is None else (e * m.to(e.dtype)).sum())
            cnt += float(e.numel() if m is None else m.sum())
        c_const[j] = tot / max(cnt, 1.0)

    original = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    was_training = target_model.training
    original_device = next(target_model.parameters()).device
    req_grad = {id(p): p.requires_grad for p in target_model.parameters()}
    dw0 = {j: corrections[f"{key_fmt.format(j)}.weight"].detach().to(torch.float32).cpu() for j in fitted}
    db0 = {j: corrections[f"{key_fmt.format(j)}.bias"].detach().to(torch.float32).cpu() for j in fitted}
    try:
        target_model.load_state_dict({k: v.detach().cpu() for k, v in target_base_sd.items()}, strict=True)
        target_model.to(dev).eval()
        for p in target_model.parameters():
            p.requires_grad_(False)
        w_norm = {j: float(layout.proj_module(layout.blocks(target_model)[j]).weight.detach().double().norm())
                  for j in fitted}
        w_rms = {j: w_norm[j] / math.sqrt(layout.proj_module(layout.blocks(target_model)[j]).weight.numel())
                 for j in fitted}

        # ---- generic rows: base boundaries at M, row masks, constants v_j --------------------------------------
        g_imgs: list[Tensor] = []
        g_base: list[dict[int, Tensor]] = []
        g_mask: list[dict[int, Tensor | None]] = []
        g_train: list[int] = []
        g_hold: list[int] = []
        v_const: dict[int, float] = {}
        all_g = list(generic_batches or []) + list(generic_holdout or [])
        if use_keep:
            with torch.no_grad():
                for b in all_g:
                    imgs = b[0]
                    outs = _forward(target_model, layout, imgs.to(dev), m_blocks, dev, False)
                    g_imgs.append(imgs.cpu())
                    g_base.append({j: outs[j].cpu() for j in m_blocks})
                    masks: dict[int, Tensor | None] = {}
                    for j in m_blocks:
                        if norm_filter is None:
                            masks[j] = None
                        else:
                            mu = norm_filter["mu"][j].to("cpu", torch.float64)
                            nrm = (outs[j].cpu().to(torch.float64) - mu).norm(dim=-1)
                            masks[j] = nrm <= float(norm_filter["k"]) * float(norm_filter["median"][j])
                    g_mask.append(masks)
            ng = len(g_imgs)
            if generic_holdout:
                g_train, g_hold = list(range(len(generic_batches))), list(range(len(generic_batches), ng))
            else:
                g_train = list(range(ng - 1)) if ng >= 2 else [0]
                g_hold = [ng - 1] if ng >= 2 else [0]
            for j in m_blocks:
                s1 = torch.zeros(g_base[0][j].shape[-1], dtype=torch.float64)
                cnt = 0.0
                for i in g_train:
                    x = g_base[i][j].to(torch.float64)
                    m = g_mask[i][j]
                    s1 += (x.sum((0, 1)) if m is None else x[m].sum(0))
                    cnt += float(x.shape[0] * x.shape[1] if m is None else m.sum())
                mu_j = s1 / max(cnt, 1.0)
                s2 = 0.0
                for i in g_train:
                    e = (g_base[i][j].to(torch.float64) - mu_j).pow(2).sum(-1)
                    m = g_mask[i][j]
                    s2 += float(e.sum() if m is None else (e * m.to(e.dtype)).sum())
                v_const[j] = s2 / max(cnt, 1.0)

        # ---- trainable parameters ---------------------------------------------------------------------------
        dw = {j: torch.nn.Parameter(dw0[j].clone().to(dev)) for j in fitted}
        db = {j: torch.nn.Parameter(db0[j].clone().to(dev)) for j in fitted}
        params = {j: (dw[j], db[j]) for j in fitted}
        # per-element step scale: the rms of the fit's own update (weights and bias separately), floored at 1e-3 of rms(W)
        # so a zero start can still move. Adam steps are sign-like (every element moves ~lr), so a step scaled to rms(W)
        # (~10x rms(dW)) rewrote the update in a handful of steps.
        s_w = {j: max(float(dw0[j].double().norm()) / math.sqrt(dw0[j].numel()), 1e-3 * w_rms[j]) for j in fitted}
        s_b = {j: max(float(db0[j].double().norm()) / math.sqrt(db0[j].numel()), 1e-3 * w_rms[j]) for j in fitted}
        opt = torch.optim.Adam(
            [{"params": [dw[j]], "lr": float(lr) * s_w[j]} for j in fitted]
            + [{"params": [db[j]], "lr": float(lr) * s_b[j]} for j in fitted], eps=1e-10
        )
        n_warm = int(math.ceil(float(warmup) * int(steps)))

        def lr_factor(t: int) -> float:  # t = number of optimiser steps taken so far
            if n_warm > 0 and t < n_warm:
                return (t + 1) / n_warm
            return 0.5 * (1.0 + math.cos(math.pi * (t - n_warm) / max(int(steps) - n_warm, 1)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        need_train = sorted(set(m_blocks))
        need_eval = fitted
        upto = max(max(need_train), 0)

        def task_loss(outs, k, idx) -> Tensor:
            total = outs[need_train[0]].new_zeros(())
            for j in task_blocks:
                if c_const[j] <= 0.0:
                    continue
                real = outs[j] - base_out[j][k][idx].to(dev)
                diff = real - desired[j][k][idx].to(dev)
                total = total + _masked_mean(diff.pow(2).sum(-1), _rows(row_masks, j, k, idx)) / c_const[j]
            return total

        def keep_loss(outs, i, idx) -> Tensor:
            total = outs[need_train[0]].new_zeros(())
            for j in m_blocks:
                if v_const[j] <= 0.0:
                    continue
                diff = outs[j] - g_base[i][j][idx].to(dev)
                total = total + _masked_mean(diff.pow(2).sum(-1), None if g_mask[i][j] is None else g_mask[i][j][idx]) \
                    / v_const[j]
            return rho_e * total

        chunk = max(4 * int(batch), 1)

        @torch.no_grad()
        def evaluate(ids=None, generic=True) -> dict[str, Any]:
            """Calibration rows of the batches ``ids`` (default: held out; all fitted blocks) and, if ``generic``, the held-out
            generic rows (blocks M)."""
            acc = {j: [0.0, 0.0, 0.0, 0.0, 0.0] for j in fitted}  # e2, r2, d2, rd, rows
            with _installed(target_model, layout, params, upto=-1, use_checkpoint=False):
                for k in (hold_idx if ids is None else ids):
                    imgs = cal[k][0]
                    for a in range(0, int(imgs.shape[0]), chunk):
                        idx = torch.arange(a, min(a + chunk, int(imgs.shape[0])))
                        outs = _forward(target_model, layout, imgs[idx].to(dev), need_eval, dev, amp)
                        for j in fitted:
                            real = (outs[j] - base_out[j][k][idx].to(dev)).double()
                            want = desired[j][k][idx].to(dev).double()
                            m = _rows(row_masks, j, k, idx)
                            m = None if m is None else m.to(dev)
                            ew = (real - want).pow(2).sum(-1)
                            rw, dw_ = real.pow(2).sum(-1), want.pow(2).sum(-1)
                            rd = (real * want).sum(-1)
                            if m is not None:
                                ew, rw, dw_, rd = (t * m.to(t.dtype) for t in (ew, rw, dw_, rd))
                            a_ = acc[j]
                            a_[0] += float(ew.sum()); a_[1] += float(rw.sum()); a_[2] += float(dw_.sum())  # noqa: E702
                            a_[3] += float(rd.sum()); a_[4] += float(ew.numel() if m is None else m.sum())  # noqa: E702
                drift: dict[int, float] = {}
                if use_keep and generic:
                    sums = {j: [0.0, 0.0] for j in m_blocks}
                    for i in g_hold:
                        for a in range(0, int(g_imgs[i].shape[0]), chunk):
                            idx = torch.arange(a, min(a + chunk, int(g_imgs[i].shape[0])))
                            outs = _forward(target_model, layout, g_imgs[i][idx].to(dev), m_blocks, dev, amp)
                            for j in m_blocks:
                                e = (outs[j] - g_base[i][j][idx].to(dev)).double().pow(2).sum(-1)
                                m = None if g_mask[i][j] is None else g_mask[i][j][idx].to(dev)
                                sums[j][0] += float(e.sum() if m is None else (e * m.to(e.dtype)).sum())
                                sums[j][1] += float(e.numel() if m is None else m.sum())
                    drift = {j: sums[j][0] / max(sums[j][1], 1.0) / v_const[j] for j in m_blocks if v_const[j] > 0}
            per = {}
            for j in fitted:
                e2, r2, d2, rd, cnt = acc[j]
                per[j] = {
                    "ratio": r2**0.5 / d2**0.5 if d2 > 0 else float("nan"),
                    "cosine": rd / (r2 * d2) ** 0.5 if r2 * d2 > 0 else float("nan"),
                    "rel_err": e2**0.5 / d2**0.5 if d2 > 0 else float("nan"),
                    "mse_over_c": e2 / max(cnt, 1.0) / c_const[j] if j in c_const and c_const[j] > 0 else float("nan"),
                }
            task = sum(per[j]["mse_over_c"] for j in task_blocks)
            return {"per_block": per, "task": task, "keep": drift, "obj": task + rho_e * sum(drift.values())}

        def snapshot():
            return {j: (dw[j].detach().cpu().clone(), db[j].detach().cpu().clone()) for j in fitted}

        # ---- training ---------------------------------------------------------------------------------------
        # forward consistency (same batch): the c_proj patch with and without checkpointing against the model with the
        # update mounted in its weights (a plain capture); a mismatch would make the optimiser fit a different network
        fc_idx = torch.arange(min(int(batch), int(cal[train_idx[0]][0].shape[0])))
        fc_img = cal[train_idx[0]][0][fc_idx].to(dev)
        keys_wb = [(f"{key_fmt.format(j)}.weight", f"{key_fmt.format(j)}.bias", j) for j in fitted]
        with torch.no_grad():
            target_model.load_state_dict(
                {k: target_base_sd[k].detach().cpu() + (dw0[j] if k.endswith("weight") else db0[j]).to(target_base_sd[k].dtype)
                 for wk, bk, j in keys_wb for k in (wk, bk)}, strict=False)
            ref = _forward(target_model, layout, fc_img, need_train, dev, False)
            target_model.load_state_dict({k: target_base_sd[k].detach().cpu() for wk, bk, _ in keys_wb for k in (wk, bk)},
                                         strict=False)
            ref = {j: v.detach().clone() for j, v in ref.items()}
            base_fc = _forward(target_model, layout, fc_img, need_train, dev, False)
            base_fc = {j: v.detach().clone() for j, v in base_fc.items()}
            with _installed(target_model, layout, params, upto=upto, use_checkpoint=False):
                plain = _forward(target_model, layout, fc_img, need_train, dev, amp)
            with torch.enable_grad(), _installed(target_model, layout, params, upto=upto, use_checkpoint=True):
                ckpt_out = _forward(target_model, layout, fc_img, need_train, dev, amp)

        def rel(a, b):
            return float((a - b).detach().double().norm() / b.detach().double().norm().clamp_min(1e-30))

        # errors are measured against the whole stream S and against the change S - S_base the optimiser fits: a small
        # error relative to S can still be large relative to the change
        fcheck = {"patched_vs_mounted": max(rel(plain[j], ref[j]) for j in need_train),
                  "checkpoint_vs_plain": max(rel(ckpt_out[j], plain[j]) for j in need_train),
                  "patched_vs_mounted_over_change": max(rel(plain[j] - base_fc[j], ref[j] - base_fc[j])
                                                        for j in need_train),
                  "checkpoint_vs_plain_over_change": max(rel(ckpt_out[j] - base_fc[j], plain[j] - base_fc[j])
                                                         for j in need_train)}
        del ref, plain, ckpt_out, base_fc
        if not amp and (max(fcheck["patched_vs_mounted"], fcheck["checkpoint_vs_plain"]) > 1e-3
                        or max(fcheck["patched_vs_mounted_over_change"],
                               fcheck["checkpoint_vs_plain_over_change"]) > 1e-2):
            raise RuntimeError(f"refine: the patched forward does not reproduce the mounted model: {fcheck}")
        with _installed(target_model, layout, params, upto=upto, use_checkpoint=False):
            before = evaluate()
            tr_ids = train_idx[: len(hold_idx)]
            before_tr = evaluate(tr_ids, generic=False)

        def depth_vals(ev):
            return {str(j): ev["per_block"][j]["mse_over_c"] for j in m_blocks}

        best = {"obj": before["obj"], "task": before["task"], "step": 0, "state": snapshot(), "eval": before}
        curve = [{"step": 0, "heldout_task": before["task"], "heldout_objective": before["obj"],
                  "heldout_keep": sum(before["keep"].values()) * rho_e if before["keep"] else None,
                  "train_task": None, "train_keep": None, "train_subset_task": before_tr["task"],
                  "heldout_by_depth": depth_vals(before), "train_subset_by_depth": depth_vals(before_tr)}]
        step1 = {}
        gen = torch.Generator().manual_seed(int(seed))
        run_task = run_keep = 0.0
        n_run = 0
        with _installed(target_model, layout, params, upto=upto, use_checkpoint=use_checkpoint):
            for step in range(1, int(steps) + 1):
                opt.zero_grad(set_to_none=True)
                k = train_idx[int(torch.randint(len(train_idx), (1,), generator=gen))]
                imgs = cal[k][0]
                idx = torch.randperm(int(imgs.shape[0]), generator=gen)[: int(batch)]
                outs = _forward(target_model, layout, imgs[idx].to(dev), need_train, dev, amp)
                lt = task_loss(outs, k, idx)
                lt.backward()
                run_task += float(lt.detach())
                del outs
                if use_keep:
                    i = g_train[int(torch.randint(len(g_train), (1,), generator=gen))]
                    idx_g = torch.randperm(int(g_imgs[i].shape[0]), generator=gen)[: int(batch)]
                    outs = _forward(target_model, layout, g_imgs[i][idx_g].to(dev), need_train, dev, amp)
                    lk = keep_loss(outs, i, idx_g)
                    lk.backward()
                    run_keep += float(lk.detach())
                    del outs
                if float(prox) > 0.0:
                    lp = sum(((dw[j] - dw0[j].to(dev)).pow(2).sum() / w_norm[j] ** 2) for j in fitted) * float(prox)
                    lp.backward()
                if step == 1:
                    step1["loss_before"] = float(lt.detach()) + (float(lk.detach()) if use_keep else 0.0)
                opt.step()
                sched.step()
                if step == 1:
                    with torch.no_grad():  # the same micro-batches after the first update
                        o2 = _forward(target_model, layout, imgs[idx].to(dev), need_train, dev, amp)
                        after1 = float(task_loss(o2, k, idx))
                        if use_keep:
                            o3 = _forward(target_model, layout, g_imgs[i][idx_g].to(dev), need_train, dev, amp)
                            after1 += float(keep_loss(o3, i, idx_g))
                    step1["loss_after"] = after1
                n_run += 1
                if step % int(eval_every) == 0 or step == int(steps):
                    ev = evaluate()
                    ev_tr = evaluate(tr_ids, generic=False)
                    curve.append({"step": step, "heldout_task": ev["task"], "heldout_objective": ev["obj"],
                                  "train_subset_task": ev_tr["task"], "heldout_by_depth": depth_vals(ev),
                                  "train_subset_by_depth": depth_vals(ev_tr),
                                  "heldout_keep": sum(ev["keep"].values()) * rho_e if ev["keep"] else None,
                                  "train_task": run_task / n_run, "train_keep": run_keep / n_run if use_keep else None})
                    run_task = run_keep = 0.0
                    n_run = 0
                    if ev["obj"] < best["obj"]:
                        best = {"obj": ev["obj"], "task": ev["task"], "step": step, "state": snapshot(), "eval": ev}
            final_eval = curve[-1]["heldout_task"]
        # ---- best checkpoint, diagnostics ---------------------------------------------------------------------
        with torch.no_grad():
            for j in fitted:
                dw[j].copy_(best["state"][j][0].to(dev))
                db[j].copy_(best["state"][j][1].to(dev))
        after = best["eval"]
        new: dict[str, Tensor] = {}
        per_block: dict[int, dict[str, Any]] = {}
        for j in fitted:
            wk, bk = f"{key_fmt.format(j)}.weight", f"{key_fmt.format(j)}.bias"
            w_new, b_new = best["state"][j]
            new[wk] = w_new.to(corrections[wk].dtype)
            new[bk] = b_new.to(corrections[bk].dtype)
            row: dict[str, Any] = {
                "refine_measured": j in m_blocks, "refine_in_task_loss": j in task_blocks,
                "refine_ratio_before": before["per_block"][j]["ratio"], "refine_ratio_after": after["per_block"][j]["ratio"],
                "refine_cosine_before": before["per_block"][j]["cosine"],
                "refine_cosine_after": after["per_block"][j]["cosine"],
                "refine_rel_err_before": before["per_block"][j]["rel_err"],
                "refine_rel_err_after": after["per_block"][j]["rel_err"],
                "refine_dw_norm_before": float(dw0[j].double().norm()),
                "refine_dw_norm_after": float(w_new.double().norm()),
                "refine_dw_change": float((w_new - dw0[j]).double().norm()),
                "refine_bias_norm_before": float(db0[j].double().norm()),
                "refine_bias_norm_after": float(b_new.double().norm()),
            }
            if j in m_blocks:
                row["refine_desired_sq_mean"] = c_const[j]
                if use_keep:
                    row["refine_stream_var"] = v_const[j]
                    row["refine_desired_over_stream_var"] = c_const[j] / v_const[j] if v_const[j] > 0 else float("nan")
                    row["refine_generic_drift_before"] = before["keep"].get(j, float("nan")) ** 0.5
                    row["refine_generic_drift_after"] = after["keep"].get(j, float("nan")) ** 0.5
            per_block[j] = row
        seconds = time.perf_counter() - started
        info = {
            "target": target, "rho_effective": rho_e, "warmup_steps": n_warm, "step1": step1,
            "forward_check": fcheck, "lr_scale": "rms of the fitted update (weights, bias separately)", "blocks": m_blocks, "task_blocks": task_blocks, "rho": float(rho), "prox": float(prox),
            "steps": int(steps), "lr_relative": float(lr), "batch": int(batch), "holdout_batches": hold_idx,
            "train_batches": train_idx, "generic_train_batches": g_train, "generic_holdout_batches": g_hold,
            "amp": bool(amp), "checkpoint": bool(use_checkpoint), "seed": int(seed),
            "heldout_task_init": before["task"], "heldout_task_best": best["task"], "best_step": best["step"],
            "heldout_objective_init": before["obj"], "heldout_objective_best": best["obj"],
            "heldout_task_final": final_eval, "curve": curve, "seconds": seconds,
            "seconds_per_step": seconds / max(int(steps), 1),
            "generic_drift_note": "relative RMS drift sqrt(mean ||S - S_base||^2 / v_j) on the held-out generic batch",
        }
        return new, per_block, info
    finally:
        target_model.load_state_dict(original, strict=True)
        for p in target_model.parameters():
            p.requires_grad_(req_grad[id(p)])
        target_model.to(original_device).train(was_training)
