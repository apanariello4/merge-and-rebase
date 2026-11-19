"""Evaluation of a shared adaptation ``(W, b)`` on a model (AGENT_SHARED_TRAINING.md sections 1.6, 6).

Always the TRUE model:

* ``adapter``: the forward hooks of ``shared_space.adapter_hooks`` (u from the current stream);
* ``compiled``: per block ``pi_m(s)``, the desired displacement ``D = c (h_bar Phi^T W + 1 b^T) Phi`` on the target's
  calibration images (base stream, independent per block) is synthesised into ``mlp.c_proj`` with the S0 ridge
  (Ariadne's empirical-Bayes lambda on ``ResidualSufficientStatistics``, then ``solve_preserve`` with ``rho_G`` on the
  target base's ``c_proj`` inputs over the anchor images, norm filter as the anchors runs). The result is a task
  vector in the ``transported_tvs`` format, so ``eval.diagnostics.conservation`` reads it unchanged;
* ``linearized``: ``z + J . vec(u W + 1 b^T)`` (only to measure the linear-true gap).

Conservation: accuracy minus the target base on the other Vision8 tasks (``conservation.accuracy_delta``), adapter
active / compiled vector applied, TEST split, same loaders and heads as the run.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from typing import Any

import torch

from merge_and_rebase.eval.diagnostics.conservation import accuracy_delta
from merge_and_rebase.eval.utils import eval_task_top1
from merge_and_rebase.io.ckpt import load_into_model
from merge_and_rebase.rebase.methods._ariadne.capture import iter_capture_tokens
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods._ariadne.layouts import COMPONENT_INPUT_KIND
from merge_and_rebase.rebase.methods.anchors.jacobian import VisionTail
from merge_and_rebase.rebase.methods.anchors.shared_space import ModelFrame, SharedAdapter, adapter_hooks
from merge_and_rebase.rebase.methods.anchors.synthesis import capture_generic_inputs, generic_change, solve_preserve

from .lin import LinearizedProblem, Student, Theta

Tensor = torch.Tensor


def frame_on(frame: ModelFrame, device, dtype=torch.float32) -> ModelFrame:
    """A copy of the frame with its tensors on ``device`` (hooks then do no host transfers)."""
    return ModelFrame(frame.model_id, list(frame.grid), list(frame.blocks),
                      [m.to(device, dtype) for m in frame.mu], list(frame.c), [p.to(device, dtype) for p in frame.phi],
                      None if frame.psi is None else [p.to(device, dtype) for p in frame.psi])


def adapter_from_theta(theta: Theta, device, dtype=torch.float32) -> SharedAdapter:
    grid = sorted(theta)
    r = int(theta[grid[0]][0].shape[0])
    ad = SharedAdapter(grid, r, dtype=dtype)
    ad.set_({j: theta[j][0] for j in grid}, {j: theta[j][1] for j in grid})
    return ad.to(device)


def top1(clf, ctx, text_features: Tensor, split: str, device: str) -> float:
    return eval_task_top1(clf=clf, loaders=ctx.loaders, classnames=list(ctx.classnames),
                          build_cfg_task=ctx.build_cfg_task, device=device, split=split,
                          text_features=text_features)


def adapter_accuracy(clf, frame: ModelFrame, theta: Theta | None, ctx, text_features: Tensor, split: str,
                     device: str, scale: float = 1.0) -> float:
    """Top-1 of the true model with the adapter hooks (``theta=None``: the base model)."""
    if theta is None:
        return top1(clf, ctx, text_features, split, device)
    ad = adapter_from_theta(theta, device)
    with adapter_hooks(clf.model, frame_on(frame, device), ad, scale=scale):
        return top1(clf, ctx, text_features, split, device)


@torch.no_grad()
def linearized_accuracy(clf, frame: ModelFrame, theta: Theta, loader, text_features: Tensor, *, lin_point: str,
                        device: str, jbar: dict[int, Tensor] | None = None, max_batches: int | None = None
                        ) -> dict[str, float]:
    """Top-1 of ``z + J . vec(u W + 1 b^T)`` and of the true adapter on the same batches (the linear-true gap)."""
    from .lin import collect_student

    tail = VisionTail(clf.model, text_features.to(device))
    grid = sorted(theta)
    th = {j: (w.to(device), b.to(device)) for j, (w, b) in theta.items()}
    ad = adapter_from_theta(theta, device)
    fdev = frame_on(frame, device)
    n = c_lin = c_true = c_base = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        st = collect_student(frame.model_id, tail, frame, grid, [(x, y)], lin_point=lin_point)
        prob = LinearizedProblem([st], grid, int(th[grid[0]][0].shape[0]), lin_point=lin_point, device=device,
                                 jbar=jbar)
        dz = prob.predict(th)[frame.model_id]
        z0 = st.batches[0].z0
        with adapter_hooks(clf.model, fdev, ad):
            zt = tail.full_logits(x).double().cpu()
        y = torch.as_tensor(y)
        c_lin += int(((z0 + dz).argmax(-1) == y).sum())
        c_true += int((zt.argmax(-1) == y).sum())
        c_base += int((z0.argmax(-1) == y).sum())
        n += len(y)
    return {"linearized": c_lin / n, "adapter": c_true / n, "base": c_base / n, "n": n}


def compile_theta(clf, frame: ModelFrame, theta: Theta, calib_batches: Sequence, anchor_batches: Sequence, *,
                  norm_filter: dict[str, Any] | None, rho_g: float = 0.1, preserve_tokens: int = 32768,
                  seed: int = 89, device: str = "cuda") -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Synthesise ``(W, b)`` into ``mlp.c_proj`` of the blocks ``pi_m(s)``: a native task vector (CPU fp32) and the
    per-block diagnostics. ``calib_batches``: (images, labels) of the target task under the model's preprocessing;
    ``anchor_batches``: the frame's anchor images (generic inputs of the S0 term)."""
    started = time.time()
    model = clf.model
    grid = sorted(theta)
    fdev = frame_on(frame, device, torch.float64)
    blocks = {j: frame.blocks[frame.index(j)] for j in grid}
    kind = COMPONENT_INPUT_KIND["mlp.c_proj"]
    requests = {}
    for b in blocks.values():
        requests[f"{b}.h"] = (b, kind)
        requests[f"{b}.out"] = (b, "boundary")
    stats = {b: ResidualSufficientStatistics(device=device) for b in blocks.values()}
    eye = None
    for values in iter_capture_tokens(model, list(calib_batches), requests, device):
        for j, b in blocks.items():
            h_in = values[f"{b}.h"].reshape(-1, values[f"{b}.h"].shape[-1])
            out = values[f"{b}.out"].to(device, torch.float64)
            w, bias = theta[j]
            du = fdev.encode(out, j) @ w.to(device, torch.float64) + bias.to(device, torch.float64)
            d = fdev.decode(du, j).reshape(-1, out.shape[-1])
            if eye is None or eye.shape[0] != d.shape[1]:
                eye = torch.eye(d.shape[1], dtype=torch.float32, device=device)
            stats[b].update(h_in.to(device), d.float(), None, eye)
    anchor_batches = [b if isinstance(b, (tuple, list)) else (b, torch.zeros(len(b), dtype=torch.long))
                      for b in anchor_batches]  # the capture reads (images, labels) batches
    generic = capture_generic_inputs(model, anchor_batches, max(blocks.values()) + 1, component="mlp.c_proj",
                                     tokens_per_block=preserve_tokens, seed=seed, device=device,
                                     norm_filter=norm_filter, blocks=sorted(set(blocks.values())))
    tv: dict[str, Tensor] = {}
    diag: dict[str, Any] = {}
    for j, b in blocks.items():
        _, info = stats[b].solve(ridge_relative=0.01, ridge_estimator="empirical_bayes")
        dw, beta, pinfo = solve_preserve(stats[b], generic[b], float(rho_g), float(info["ridge"]))
        key = f"visual.transformer.resblocks.{b}.mlp.c_proj"
        weight = model.state_dict()[f"{key}.weight"]
        tv[f"{key}.weight"] = dw.T.to(torch.float32).cpu()
        tv[f"{key}.bias"] = beta.to(torch.float32).cpu()
        diag[str(j)] = {
            "block": b,
            "n_rows": int(stats[b].n_rows),
            "ridge": float(info["ridge"]),
            "desired_norm": float(stats[b].sum_e2) ** 0.5,
            "residual_norm_after": pinfo["residual_norm_after"],
            "relative_residual": pinfo["residual_norm_after"] / max(float(stats[b].sum_e2) ** 0.5, 1e-300),
            "generic_change": generic_change(dw, beta, weight, generic[b]),
        }
    diag["seconds"] = time.time() - started
    return tv, diag


@contextlib.contextmanager
def applied(model, tv: dict[str, Tensor], scale: float = 1.0):
    """The model with ``base + scale * tv`` (restored on exit)."""
    sd = model.state_dict()
    keep = {k: sd[k].detach().clone() for k in tv}
    with torch.no_grad():
        for k, v in tv.items():
            sd[k].add_(scale * v.to(sd[k].device, sd[k].dtype))
    try:
        yield
    finally:
        load_into_model(model, {**model.state_dict(), **keep}, strict=True)


def conservation(clf, contexts: dict[str, Any], heads: dict[str, Tensor], task: str, base_acc: dict[str, float], *,
                 device: str, frame: ModelFrame | None = None, theta: Theta | None = None,
                 tv: dict[str, Tensor] | None = None) -> dict[str, Any]:
    """Test accuracy on every task with the adapter (``frame``, ``theta``) or the compiled vector (``tv``)."""
    if tv is not None:
        with applied(clf.model, tv):
            acc = {t: top1(clf, contexts[t], heads[t], "test", device) for t in contexts}
    else:
        ad = adapter_from_theta(theta, device)
        with adapter_hooks(clf.model, frame_on(frame, device), ad):
            acc = {t: top1(clf, contexts[t], heads[t], "test", device) for t in contexts}
    return {"test_accuracy": acc, "own_task_accuracy": acc[task], **accuracy_delta(acc, base_acc, task)}


__all__ = [
    "Student",
    "adapter_accuracy",
    "adapter_from_theta",
    "applied",
    "compile_theta",
    "conservation",
    "frame_on",
    "linearized_accuracy",
    "top1",
]
