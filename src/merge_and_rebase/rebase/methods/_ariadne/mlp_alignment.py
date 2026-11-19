"""Opt-in non-linear alignment ceiling for Ariadne (``alignment='mlp'``, vision only).

Per target block ``j`` (paired source block ``pi(j)``) a map ``g_j: R^{d_s} -> R^{d_t}`` replaces the Procrustes map
``Q_j``. ``g_j`` is a 2-layer MLP ``Linear(d_s, d_t) -> GELU -> Linear(d_t, d_t)`` fit on the BASE activations of the
calibration draw Pass A already walks: ``x`` = source-base residual rows after block ``pi(j)`` token-interpolated to the
target grid (exactly ``_aligned``), ``y`` = target-base rows after block ``j``.

Centering (same convention as the Procrustes fit): ``g_j`` is fit on the CENTERED rows ``x - mu_s``, ``y - mu_t``
(``mu_s``/``mu_t`` = the Pass-A means over all calibration rows, ``prepared['source_mean_by_position']`` /
``['target_mean_by_position']``) and the desired change is ``D_j = g_j(f - mu_s) - g_j(b - mu_s)``: ``b`` and ``f`` are
centered with the SAME mean, so the difference is the change of ``g_j`` between two points of the same (centered) input
space. For ``init='residual'`` and an untrained MLP this equals ``(f - b) Q_j`` (the linear part cancels the mean).

Initialisations: ``residual``: ``g(x) = x Q_j + MLP(x)`` with the last Linear zero-initialised (training starts exactly at
the Procrustes map; ``Q_j`` is the Pass-A polar factor, fit on all calibration images, so the Procrustes baseline is, if
anything, mildly in-sample on the held-out images). ``plain``: ``g(x) = MLP(x)``, default torch init, seeded.

Training: AdamW (lr 1e-3, wd 1e-4), MSE, row minibatches, images split train/held-out by image (deterministic from the
seed), early stopping on the held-out MSE (patience epochs without improvement, at most ``max_epochs``), best state kept.
Epoch 0 (the initial map) is a candidate, so ``best_epoch == 0`` means no epoch improved on the initialisation.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn

from .layouts import _aligned

Tensor = torch.Tensor

HELDOUT_IMAGES = 40
BATCH_ROWS = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 5
_EVAL_ROWS = 16384


def _seed_for(seed: int | None, position: int, tag: str) -> int:
    digest = hashlib.sha256(f"ariadne_mlp_alignment:{tag}:{int(seed or 0)}:{int(position)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def heldout_image_split(n_images: int, seed: int | None, n_heldout: int = HELDOUT_IMAGES) -> tuple[Tensor, Tensor]:
    """Deterministic ``(train_idx, heldout_idx)`` over image indices; disjoint, sorted, depends only on ``seed``."""
    if n_images < 2:
        raise ValueError("alignment='mlp' needs at least 2 calibration images")
    n_heldout = min(int(n_heldout), n_images - 1)
    gen = torch.Generator().manual_seed(_seed_for(seed, 0, "split"))
    perm = torch.randperm(n_images, generator=gen)
    return perm[n_heldout:].sort().values, perm[:n_heldout].sort().values


class AlignmentMLP(nn.Module):
    """``g(x) = [x Q +] Linear(d_t, d_t)(GELU(Linear(d_s, d_t)(x)))``; see the module docstring."""

    def __init__(self, d_s: int, d_t: int, *, init: str, q: Tensor | None, seed: int) -> None:
        super().__init__()
        if init not in {"residual", "plain"}:
            raise ValueError("init must be 'residual' or 'plain'")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.l1 = nn.Linear(d_s, d_t)
            self.l2 = nn.Linear(d_t, d_t)
        self.init = init
        if init == "residual":
            if q is None or tuple(q.shape) != (d_s, d_t):
                raise ValueError("init='residual' needs the Procrustes map Q of shape (d_s, d_t)")
            nn.init.zeros_(self.l2.weight)
            nn.init.zeros_(self.l2.bias)
            self.register_buffer("q", q.detach().clone().float())
        else:
            self.q = None

    def forward(self, x: Tensor) -> Tensor:
        h = self.l2(nn.functional.gelu(self.l1(x)))
        return x @ self.q + h if self.q is not None else h


@torch.no_grad()
def _sq_err(model: nn.Module, x: Tensor, y: Tensor) -> float:
    total = 0.0
    for s in range(0, x.shape[0], _EVAL_ROWS):
        total += float(((model(x[s : s + _EVAL_ROWS]) - y[s : s + _EVAL_ROWS]).double() ** 2).sum().item())
    return total


def fit_block_mlp(
    x_images: Tensor,
    y_images: Tensor,
    q: Tensor,
    mu_s: Tensor,
    mu_t: Tensor,
    *,
    position: int,
    init: str = "residual",
    seed: int | None = 0,
    device="cpu",
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    n_heldout: int = HELDOUT_IMAGES,
) -> tuple[AlignmentMLP, dict[str, Any]]:
    """Fit ``g_j`` on ``x_images`` [n, N, d_s], ``y_images`` [n, N, d_t] (raw, uncentered). Returns ``(g, row)``.

    ``row`` holds scalar diagnostics: held-out/train relative errors of the MLP and of Procrustes ``Q`` on the same
    images, epochs run, best epoch, split sizes and hyperparameters.
    """
    n_img = x_images.shape[0]
    train_idx, held_idx = heldout_image_split(n_img, seed, n_heldout)
    xc = x_images.float() - mu_s.float()
    yc = y_images.float() - mu_t.float()
    d_s, d_t = xc.shape[-1], yc.shape[-1]
    x_tr = xc[train_idx].reshape(-1, d_s).to(device)
    y_tr = yc[train_idx].reshape(-1, d_t).to(device)
    x_ho = xc[held_idx].reshape(-1, d_s).to(device)
    y_ho = yc[held_idx].reshape(-1, d_t).to(device)
    del xc, yc
    q_dev = q.float().to(device)
    ss_tr = float((y_tr.double() ** 2).sum().item())
    ss_ho = float((y_ho.double() ** 2).sum().item())

    model = AlignmentMLP(
        d_s, d_t, init=init, q=q if init == "residual" else None, seed=_seed_for(seed, position, "init")
    )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(_seed_for(seed, position, "shuffle"))

    best_err = _sq_err(model, x_ho, y_ho)
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_run = 0
    since_best = 0
    n_rows = x_tr.shape[0]
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_rows, generator=gen).to(device)
        for s in range(0, n_rows, BATCH_ROWS):
            idx = perm[s : s + BATCH_ROWS]
            loss = nn.functional.mse_loss(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        epochs_run = epoch
        err = _sq_err(model, x_ho, y_ho)
        if err < best_err:
            best_err, best_epoch, since_best = err, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    class _Linear(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x @ q_dev

    lin = _Linear()
    row = {
        "alignment": "mlp",
        "mlp_init": init,
        "mlp_heldout_rel_err": (_sq_err(model, x_ho, y_ho) / max(ss_ho, 1e-300)) ** 0.5,
        "procrustes_heldout_rel_err": (_sq_err(lin, x_ho, y_ho) / max(ss_ho, 1e-300)) ** 0.5,
        "mlp_train_rel_err": (_sq_err(model, x_tr, y_tr) / max(ss_tr, 1e-300)) ** 0.5,
        "procrustes_train_rel_err": (_sq_err(lin, x_tr, y_tr) / max(ss_tr, 1e-300)) ** 0.5,
        "mlp_epochs": int(epochs_run),
        "mlp_best_epoch": int(best_epoch),
        "mlp_train_images": int(train_idx.numel()),
        "mlp_heldout_images": int(held_idx.numel()),
        "mlp_max_epochs": int(max_epochs),
        "mlp_patience": int(patience),
        "mlp_lr": LR,
        "mlp_weight_decay": WEIGHT_DECAY,
        "mlp_batch_rows": BATCH_ROWS,
    }
    return model, row


def fit_alignment_mlps(
    rows_cache: dict[int, tuple[list[Tensor], list[Tensor]]],
    prepared_q: dict[int, Tensor],
    mu_s: dict[int, Tensor],
    mu_t: dict[int, Tensor],
    *,
    init: str,
    seed: int | None,
    device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    n_heldout: int = HELDOUT_IMAGES,
) -> tuple[dict[int, AlignmentMLP], dict[int, dict[str, Any]]]:
    """Fit every block's ``g_j`` from the Pass-A row caches (per-batch lists of ``[B, N, d]`` host fp32 tensors).

    One block at a time: its cache is concatenated, used, and released (``rows_cache`` entries are popped).
    """
    models: dict[int, AlignmentMLP] = {}
    rows: dict[int, dict[str, Any]] = {}
    for j in sorted(rows_cache):
        xs, ys = rows_cache.pop(j)
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        del xs, ys
        model, row = fit_block_mlp(
            x,
            y,
            prepared_q[j],
            mu_s[j],
            mu_t[j],
            position=j,
            init=init,
            seed=seed,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            n_heldout=n_heldout,
        )
        models[j] = model
        rows[j] = row
        del x, y
    return models, rows


def make_mlp_desired_fn(prepared):
    """``desired_fn(pos, i, b, f, out) -> D_j`` of ``fit_direct_residual_streaming`` for ``alignment='mlp'``.

    ``b``/``f``: source base/FT boundary of block ``i`` on the SOURCE grid (host); ``out``: target boundary (fixes the
    target grid). Returns ``g(f_al - mu_s) - g(b_al - mu_s)`` on the host, float32, shaped like ``out``.
    """
    state = prepared["mlp_alignment"]
    models, mu_s_by_pos = state["models"], state["mu_s"]

    @torch.no_grad()
    def desired_fn(pos, i, b, f, out):
        del i
        b_al = _aligned([b], [out])[0]
        f_al = _aligned([f], [out])[0]
        g = models[pos]
        dev = next(g.parameters()).device
        mu = mu_s_by_pos[pos].float()
        shape = b_al.shape[:-1] + (g.l2.out_features,)
        rows_f = (f_al.float() - mu).reshape(-1, f_al.shape[-1]).to(dev)
        rows_b = (b_al.float() - mu).reshape(-1, b_al.shape[-1]).to(dev)
        return (g(rows_f) - g(rows_b)).cpu().reshape(shape)

    return desired_fn
