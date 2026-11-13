"""Rank-truncated Procrustes screen (vision): does keeping only the top-k singular pairs of the cross-covariance give a
more stable, better-aligned transported target? Analysis only: the Ariadne method itself is untouched.

``python -m merge_and_rebase.eval.diagnostics.ariadne_norm.procrustes_screen --config <json> --device cuda``

Ariadne maps source-base boundary states to target-base boundary states with ``Q_j = U V^T`` (polar factor of the
centered cross-covariance ``C_j = U S V^T``, ``_procrustes_from_cross``). The transported target of target block ``j`` is
``D_j = dS_{pi(j)} Q_j`` with ``dS = aligned(source ft) - aligned(source base)`` at source block ``pi(j)``. Here the same
quantities are computed with ``Q_k = U[:, :k] V[:, :k]^T`` (``k`` = number of kept singular pairs; ``"full"`` =
``min(d_source, d_target)`` and equals ``Q``).

Per task:

1. Pass A of the method (``prepare_direct_residual_streaming``, called as ``runner.run_task`` calls it) -> the replayed
   calibration batches and the method's ``Q`` per block.
2. Sweep A over those batches: three centered cross-covariances per target block (all batches / odd-index batches /
   even-index batches, rows built as ``runner._cache_sweep`` builds them). GATE ``q_full_equals_method_q``: the polar
   factor of the all-batch cross equals ``prepared["activation_q64_by_position"]`` (max abs diff <= 1e-9). GATE
   ``odd_even_cross_merge``: Chan's merge of the odd and even crosses equals the all-batch cross.
3. SVD of the three crosses -> ``Q_k^all``, ``Q_k^odd``, ``Q_k^even`` per block and rank (float64, applied as float32 like
   the fit); singular values and the share of singular-value mass (sum s) and energy (sum s^2) kept are recorded.
4. Sweep B, separately on the calibration batches and on held-out batches (first ``heldout_max_batches`` batches of
   the task's val split, paired source/target views): source base, source ft, target base and, optionally, the native
   fine-tuned target. Per batch and block, float64 scalar sums (all tokens, and the CLS token only) of ``||D_k||^2``,
   ``<D_k, N>``, ``||N||^2``, ``||dS (Q_k^odd - Q_k^even)||^2``, ``||D_full||^2`` and ``<D_full, N>``, with ``N = T_native
   - T_base`` at the target block. At the last target block the per-image CLS cosine ``cos(D_k^cls, N^cls)`` is kept.
5. Derived per (block, rank, split, scope): ``energy_kept = ||D_k||^2 / ||D_full||^2``; ``cos_D_N``;
   ``size_D_over_N = ||D_k|| / ||N||``; ``split_half_target_diff = ||dS (Q_k^odd - Q_k^even)|| / ||D_full||`` and, per rank,
   the map stability ``||Q_k^odd - Q_k^even||_F / ||Q_k^all||_F``. Without a native checkpoint everything except the ``N``
   terms is computed.

Config keys: ``family`` ("vision"), ``reference_summary``, ``tasks`` ("all" | list), ``out_dir`` (write-once), ``ranks``
(default 32, 64, 128, 256, 512, "full"; ranks above ``min(d_source, d_target)`` are dropped and reported),
``native_target_ckpt_template`` (optional, as in ``readout``), ``heldout_max_batches`` (default 10);
``generic_num_batches`` is forced to 0.

Output: ``<out_dir>/<task>/screen.json``, ``<out_dir>/<task>/arrays.pt`` (singular values, per-image last-block CLS
cosines) and ``<out_dir>/run.json``. The models are left unchanged (``_state_sha`` before/after is checked).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import resource
import socket
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.rebase.methods._ariadne.capture import iter_capture_tokens
from merge_and_rebase.rebase.methods._ariadne.layouts import _aligned
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    _StreamingCrossCovariance,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.run_logging import _code_fingerprint

from .readout import GateError, _jsonable, _write_new, build_native_model, check_loader_pairing
from .runner import _state_sha

Tensor = torch.Tensor

DEFAULT_RANKS: list[int | str] = [32, 64, 128, 256, 512, "full"]
SCOPES = ("all", "cls")
Q_GATE_TOL = 1e-9
MERGE_GATE_TOL = 1e-9  # relative to max |C_all|


# ---- pure helpers -------------------------------------------------------------------------------------------------


def resolve_ranks(
    ranks: Sequence[int | str] | None, d_source: int, d_target: int
) -> tuple[list[tuple[str, int]], list]:
    """``[(label, k)]`` in the given order, ``"full"`` -> ``min(d_source, d_target)``; numeric ranks above it are
    dropped (returned second) and duplicates removed."""
    full = min(int(d_source), int(d_target))
    out: list[tuple[str, int]] = []
    dropped: list = []
    seen: set[int] = set()
    for r in DEFAULT_RANKS if ranks is None else ranks:
        if r == "full":
            label, k = "full", full
        else:
            label, k = str(int(r)), int(r)
            if k < 1:
                raise ValueError(f"rank must be >= 1, got {r!r}")
            if k > full:
                dropped.append(r)
                continue
        if k not in seen:
            seen.add(k)
            out.append((label, k))
    if not out:
        raise ValueError(f"no usable rank (full = {full}, dropped {dropped})")
    return out, dropped


def truncated_map(u: Tensor, vh: Tensor, k: int) -> Tensor:
    """``Q_k = U[:, :k] V[:, :k]^T`` from ``torch.linalg.svd(..., full_matrices=False)`` factors."""
    return u[:, :k] @ vh[:k]


def merge_cross(a: _StreamingCrossCovariance, b: _StreamingCrossCovariance) -> Tensor:
    """Cross-covariance of the union of the rows of two accumulators (Chan's pairwise formula, as ``update``)."""
    n = a.n + b.n
    return a.c + b.c + (a.n * b.n / n) * torch.outer(b.mean_x - a.mean_x, b.mean_y - a.mean_y)


def displacement_cache(sb: Mapping, sf: Mapping, ref_cpu: Tensor, distinct: Sequence[int]) -> dict[int, Tensor]:
    """``{i: aligned(f_i) - aligned(b_i)}`` on CPU float32, exactly as ``runner._cache_sweep`` builds it."""
    out = {}
    for i in distinct:
        b = _aligned([sb[str(i)].cpu()], [ref_cpu])[0]
        f = _aligned([sf[str(i)].cpu()], [ref_cpu])[0]
        out[i] = (f - b).clone()
    return out


def _rows(x: Tensor, scope: str) -> Tensor:
    return (x[:, 0] if scope == "cls" else x.reshape(-1, x.shape[-1])).double()


def _div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


# ---- Sweep A: cross-covariances ------------------------------------------------------------------------------------


@torch.no_grad()
def cross_sweep(
    *,
    source_base_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_batches: Sequence[Any],
    target_batches: Sequence[Any],
    pairing: Any,
    device: Any,
    family_adapter: Any = None,
) -> dict[str, dict[int, _StreamingCrossCovariance]]:
    """Per target block the all / odd-index / even-index batch cross-covariances (rows as in ``_cache_sweep``)."""
    depth = pairing.target_depth
    src_req = {str(i): (i, "boundary") for i in sorted(set(pairing.pairing))}
    tgt_req = {str(j): (j, "boundary") for j in range(depth)}
    acc = {name: {j: _StreamingCrossCovariance(device=device) for j in range(depth)} for name in ("all", "odd", "even")}
    gens = [
        iter_capture_tokens(
            source_base_model, source_batches, src_req, device, family_adapter=family_adapter, store_device=device
        ),
        iter_capture_tokens(
            target_model, target_batches, tgt_req, device, family_adapter=family_adapter, store_device=device
        ),
    ]
    with contextlib.ExitStack() as stack:
        for g in gens:
            stack.enter_context(contextlib.closing(g))
        for k, (sb, tgt) in enumerate(zip(*gens, strict=True)):
            half = acc["even" if k % 2 == 0 else "odd"]
            for j in range(depth):
                t = tgt[str(j)]
                x = _aligned([sb[str(pairing.pairing[j])]], [t])[0]
                x_rows, y_rows = x.reshape(-1, x.shape[-1]).double(), t.reshape(-1, t.shape[-1]).double()
                acc["all"][j].update(x_rows, y_rows)
                half[j].update(x_rows, y_rows)
    return acc


class MapBank:
    """Per target block: SVDs of the three crosses and the float32 maps applied in sweep B."""

    def __init__(self, crosses: Mapping[str, Mapping[int, _StreamingCrossCovariance]], ranks: list[tuple[str, int]]):
        self.ranks = ranks
        self.positions = sorted(crosses["all"])
        self.svals: dict[str, dict[int, Tensor]] = {n: {} for n in crosses}
        self.q_full64: dict[int, Tensor] = {}
        self.q_all: dict[int, list[Tensor]] = {}  # float32, rank order
        self.q_diff: dict[int, list[Tensor]] = {}  # Q_k^odd - Q_k^even, float32
        self.q_full: dict[int, Tensor] = {}
        self.map_stats: dict[int, list[dict[str, float]]] = {}
        self.mass: dict[int, dict[str, list[float]]] = {}
        for j in self.positions:
            usv = {n: torch.linalg.svd(crosses[n][j].cross(), full_matrices=False) for n in crosses}
            for n, (_, s, _) in usv.items():
                self.svals[n][j] = s.cpu()
            u, s, vh = usv["all"]
            self.q_full64[j] = (u @ vh).cpu()
            self.q_full[j] = (u @ vh).float()
            self.q_all[j], self.q_diff[j], self.map_stats[j] = [], [], []
            for _, k in ranks:
                qa = truncated_map(usv["all"][0], usv["all"][2], k)
                qo = truncated_map(usv["odd"][0], usv["odd"][2], k)
                qe = truncated_map(usv["even"][0], usv["even"][2], k)
                self.q_all[j].append(qa.float())
                self.q_diff[j].append(qo.float() - qe.float())
                self.map_stats[j].append(
                    {
                        "q_split_half_rel_frobenius": float(torch.linalg.norm(qo - qe) / torch.linalg.norm(qa)),
                        "q_k_frobenius": float(torch.linalg.norm(qa)),
                    }
                )
            total, total_sq = float(s.sum()), float((s * s).sum())
            self.mass[j] = {
                "mass_share": [float(s[:k].sum()) / total if total > 0 else float("nan") for _, k in ranks],
                "energy_share": [
                    float((s[:k] ** 2).sum()) / total_sq if total_sq > 0 else float("nan") for _, k in ranks
                ],
            }


# ---- Sweep B: scalar sums --------------------------------------------------------------------------------------------


class Sums:
    """float64 accumulators: ``k_terms[j, rank, scope] = (||D_k||^2, <D_k, N>, ||dS(Q_odd-Q_even)||^2)`` and
    ``p_terms[j, scope] = (||N||^2, ||D_full||^2, <D_full, N>)``; ``cos_last[rank]`` per-image CLS cosines."""

    def __init__(self, n_pos: int, n_rank: int, device: Any):
        self.k_terms = torch.zeros(n_pos, n_rank, len(SCOPES), 3, dtype=torch.float64, device=device)
        self.p_terms = torch.zeros(n_pos, len(SCOPES), 3, dtype=torch.float64, device=device)
        self.cos_last: list[list[Tensor]] = [[] for _ in range(n_rank)]
        self.n_images = 0
        self.n_batches = 0


@torch.no_grad()
def sweep_sums(
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    native_model: torch.nn.Module | None,
    source_batches: Sequence[Any],
    target_batches: Sequence[Any],
    pairing: Any,
    bank: MapBank,
    device: Any,
    family_adapter: Any = None,
) -> Sums:
    depth = pairing.target_depth
    distinct = sorted(set(pairing.pairing))
    src_req = {str(i): (i, "boundary") for i in distinct}
    tgt_req = {str(j): (j, "boundary") for j in range(depth)}
    sums = Sums(depth, len(bank.ranks), device)
    gens = [
        iter_capture_tokens(
            source_base_model, source_batches, src_req, device, family_adapter=family_adapter, store_device=device
        ),
        iter_capture_tokens(
            source_ft_model, source_batches, src_req, device, family_adapter=family_adapter, store_device=device
        ),
        iter_capture_tokens(
            target_model, target_batches, tgt_req, device, family_adapter=family_adapter, store_device=device
        ),
    ]
    if native_model is not None:
        gens.append(
            iter_capture_tokens(
                native_model, target_batches, tgt_req, device, family_adapter=family_adapter, store_device=device
            )
        )
    with contextlib.ExitStack() as stack:
        for g in gens:
            stack.enter_context(contextlib.closing(g))
        for captured in zip(*gens, strict=True):
            sb, sf, tb = captured[:3]
            nat = captured[3] if native_model is not None else None
            cache = displacement_cache(sb, sf, tb["0"].cpu(), distinct)
            sums.n_batches += 1
            sums.n_images += int(tb["0"].shape[0])
            for j in range(depth):
                ds = cache[pairing.pairing[j]].to(device)
                n_all = None if nat is None else (nat[str(j)] - tb[str(j)])
                n_rows = {} if n_all is None else {s: _rows(n_all, s) for s in SCOPES}
                d_full = ds @ bank.q_full[j]
                for si, scope in enumerate(SCOPES):
                    df = _rows(d_full, scope)
                    sums.p_terms[j, si, 1] += (df * df).sum()
                    if n_rows:
                        sums.p_terms[j, si, 0] += (n_rows[scope] ** 2).sum()
                        sums.p_terms[j, si, 2] += (df * n_rows[scope]).sum()
                for ri in range(len(bank.ranks)):
                    dk = ds @ bank.q_all[j][ri]
                    dd = ds @ bank.q_diff[j][ri]
                    for si, scope in enumerate(SCOPES):
                        r = _rows(dk, scope)
                        sums.k_terms[j, ri, si, 0] += (r * r).sum()
                        sums.k_terms[j, ri, si, 2] += (_rows(dd, scope) ** 2).sum()
                        if n_rows:
                            sums.k_terms[j, ri, si, 1] += (r * n_rows[scope]).sum()
                    if n_rows and j == depth - 1:
                        c, n = _rows(dk, "cls"), n_rows["cls"]
                        den = c.norm(dim=-1) * n.norm(dim=-1)
                        cos = torch.where(
                            den > 0, (c * n).sum(-1) / den.clamp_min(1e-300), torch.full_like(den, float("nan"))
                        )
                        sums.cos_last[ri].append(cos.cpu())
    return sums


def derive(sums: Sums, bank: MapBank, *, has_native: bool) -> list[dict[str, Any]]:
    """Per block, rank, scope the quantities of the module docstring from the accumulated sums."""
    k_terms, p_terms = sums.k_terms.cpu(), sums.p_terms.cpu()
    out = []
    for j in bank.positions:
        ranks = []
        for ri, (label, k) in enumerate(bank.ranks):
            scopes = {}
            for si, scope in enumerate(SCOPES):
                dd, dn, diff2 = (float(v) for v in k_terms[j, ri, si])
                nn, full2, full_dn = (float(v) for v in p_terms[j, si])
                row = {
                    "energy_kept": _div(dd, full2),
                    "D_norm": dd**0.5,
                    "D_full_norm": full2**0.5,
                    "split_half_target_diff": _div(diff2, full2) ** 0.5 if full2 > 0 else float("nan"),
                }
                if has_native:
                    row.update(
                        {
                            "cos_D_N": dn / (dd * nn) ** 0.5 if dd > 0 and nn > 0 else float("nan"),
                            "size_D_over_N": (dd / nn) ** 0.5 if nn > 0 else float("nan"),
                            "N_norm": nn**0.5,
                        }
                    )
                scopes[scope] = row
            ranks.append(
                {
                    "label": label,
                    "k": k,
                    **bank.map_stats[j][ri],
                    "mass_share": bank.mass[j]["mass_share"][ri],
                    "energy_share": bank.mass[j]["energy_share"][ri],
                    "scopes": scopes,
                }
            )
        full_ref = {}
        for si, scope in enumerate(SCOPES):
            nn, full2, full_dn = (float(v) for v in p_terms[j, si])
            full_ref[scope] = {"D_full_norm": full2**0.5}
            if has_native:
                full_ref[scope].update(
                    {
                        "cos_D_N": full_dn / (full2 * nn) ** 0.5 if full2 > 0 and nn > 0 else float("nan"),
                        "size_D_over_N": (full2 / nn) ** 0.5 if nn > 0 else float("nan"),
                        "N_norm": nn**0.5,
                    }
                )
        out.append({"position": j, "full_reference": full_ref, "ranks": ranks})
    return out


def _cos_summary(sums: Sums, bank: MapBank) -> dict[str, Any] | None:
    if not sums.cos_last[0]:
        return None
    out = {}
    for ri, (label, _) in enumerate(bank.ranks):
        v = torch.cat(sums.cos_last[ri])
        v = v[torch.isfinite(v)]
        out[label] = {
            "median": float(v.median()) if v.numel() else float("nan"),
            "mean": float(v.mean()) if v.numel() else float("nan"),
            "n": int(v.numel()),
        }
    return out


# ---- one task ---------------------------------------------------------------------------------------------------


def heldout_batches(loader_src: Any, loader_tgt: Any, max_batches: int) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """First ``max_batches`` batches of the paired sequential val views (labels compared batch by batch)."""
    check = check_loader_pairing(loader_src, loader_tgt, "val")
    src = list(itertools.islice(iter(loader_src), int(max_batches)))
    tgt = list(itertools.islice(iter(loader_tgt), int(max_batches)))
    if len(src) != len(tgt):
        raise ValueError("held-out source/target views yield different batch counts")
    for k, (a, b) in enumerate(zip(src, tgt, strict=True)):
        if not torch.equal(torch.as_tensor(a[1]), torch.as_tensor(b[1])):
            raise ValueError(f"held-out batch {k}: source and target views carry different labels")
    n_used = sum(len(b[1]) for b in tgt)
    indices = list(range(n_used))  # sequential loaders: dataset rows 0..n_used-1
    return (
        src,
        tgt,
        {
            **check,
            "split": "val",
            "n_batches": len(tgt),
            "n_images": n_used,
            "indices_range": [0, n_used],
            "indices_sha256": hashlib.sha256(repr(indices).encode()).hexdigest(),
        },
    )


def _model_shas(inp: Any, native: torch.nn.Module | None) -> dict[str, str]:
    shas = {
        "source_base": _state_sha(inp.source_base_model),
        "source_ft": _state_sha(inp.source_ft_model),
        "target": _state_sha(inp.target_model),
    }
    if native is not None:
        shas["native"] = _state_sha(native)
    return shas


@torch.no_grad()
def screen_task(
    inp: Any,
    sink: Mapping[str, Any],
    *,
    ranks: Sequence[int | str] | None = None,
    heldout_max_batches: int = 10,
    native_ckpt_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the screen for one task. ``inp`` is the task's `DiagnosticInputs`, ``sink`` the adapter's context sink."""
    device, cfg, pairing = inp.device, inp.config, inp.pairing
    if (cfg.alignment_map, cfg.alignment_row_weighting, cfg.procrustes_source) != ("polar", "uniform", "activation"):
        raise NotImplementedError(
            "procrustes_screen covers the polar / uniform / activation alignment only; got "
            f"{(cfg.alignment_map, cfg.alignment_row_weighting, cfg.procrustes_source)}"
        )
    ctx = sink["eval_context"]
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + now - clock
        clock = now

    native = None
    native_info = None
    if native_ckpt_path is not None:
        native, native_info = build_native_model(
            inp.target_model, inp.target_base_state, inp.source_base_model.state_dict(), native_ckpt_path
        )
    entry = _model_shas(inp, native)

    prepared = prepare_direct_residual_streaming(
        inp.source_base_model,
        inp.target_model,
        inp.source_loader,
        inp.target_loader,
        pairing,
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
    lap("pass_a")
    cal_src, cal_tgt = prepared["source_batches"], prepared["target_batches"]
    if len(cal_tgt) < 2:
        raise ValueError("the odd/even split needs at least 2 calibration batches")

    crosses = cross_sweep(
        source_base_model=inp.source_base_model,
        target_model=inp.target_model,
        source_batches=cal_src,
        target_batches=cal_tgt,
        pairing=pairing,
        device=device,
        family_adapter=inp.family_adapter,
    )
    lap("sweep_a")
    d_source, d_target = (int(v) for v in crosses["all"][0].cross().shape)
    rank_list, dropped = resolve_ranks(ranks, d_source, d_target)
    bank = MapBank(crosses, rank_list)

    q_diff = {
        j: float((bank.q_full64[j] - prepared["activation_q64_by_position"][j]).abs().max()) for j in bank.positions
    }
    merge_err = {}
    for j in bank.positions:
        c_all = crosses["all"][j].cross()
        merge_err[j] = float(
            (merge_cross(crosses["even"][j], crosses["odd"][j]) - c_all).abs().max() / c_all.abs().max()
        )
    gates = {
        "q_full_equals_method_q": {
            "max_abs_diff": max(q_diff.values()),
            "tol": Q_GATE_TOL,
            "passed": max(q_diff.values()) <= Q_GATE_TOL,
            "by_position": q_diff,
        },
        "odd_even_cross_merge": {
            "max_rel_diff": max(merge_err.values()),
            "tol": MERGE_GATE_TOL,
            "passed": max(merge_err.values()) <= MERGE_GATE_TOL,
            "by_position": merge_err,
        },
    }
    failed = [k for k, g in gates.items() if not g["passed"]]
    if failed:
        raise GateError(f"{inp.task}: gate(s) {failed} violated: { {k: gates[k] for k in failed} }")
    lap("maps")

    splits: dict[str, Any] = {}
    cos_summary: dict[str, Any] = {}
    arrays: dict[str, Any] = {
        "singular_values": {n: torch.stack([bank.svals[n][j] for j in bank.positions]) for n in bank.svals}
    }
    heldout_info: dict[str, Any] = {}
    jobs = {"calibration": (cal_src, cal_tgt)}
    h_src, h_tgt, heldout_info = heldout_batches(ctx.source_loaders.val, ctx.loaders.val, heldout_max_batches)
    jobs["heldout"] = (h_src, h_tgt)
    arrays["last_block_cls_cos"] = {}
    for name, (s_batches, t_batches) in jobs.items():
        sums = sweep_sums(
            source_base_model=inp.source_base_model,
            source_ft_model=inp.source_ft_model,
            target_model=inp.target_model,
            native_model=native,
            source_batches=s_batches,
            target_batches=t_batches,
            pairing=pairing,
            bank=bank,
            device=device,
            family_adapter=inp.family_adapter,
        )
        splits[name] = {
            "n_batches": sums.n_batches,
            "n_images": sums.n_images,
            "positions": derive(sums, bank, has_native=native is not None),
        }
        cos = _cos_summary(sums, bank)
        if cos is not None:
            cos_summary[name] = cos
            arrays["last_block_cls_cos"][name] = torch.stack(
                [torch.cat(sums.cos_last[ri]) for ri in range(len(rank_list))], dim=1
            )  # [n_images, n_ranks]
        lap(f"sweep_b_{name}")

    exit_shas = _model_shas(inp, native)
    if exit_shas != entry:
        raise RuntimeError(f"a model state changed during the screen: {entry} -> {exit_shas}")
    record = {
        "task": inp.task,
        "family": "vision",
        "ranks": [{"label": label, "k": k} for label, k in rank_list],
        "ranks_dropped_above_full": dropped,
        "d_source": d_source,
        "d_target": d_target,
        "has_native": native is not None,
        "native": native_info,
        "pairing": {
            "source_depth": pairing.source_depth,
            "target_depth": pairing.target_depth,
            "pairing": list(pairing.pairing),
        },
        "gates": gates,
        "failed_gates": failed,
        "splits": {
            n: {"batches": s["n_batches"], "n_images": s["n_images"], "blocks": s["positions"]}
            for n, s in splits.items()
        },
        "last_block_cls_cos": cos_summary,
        "calibration": {k: v for k, v in prepared["calibration"].items() if k != "indices"},
        "calibration_indices_sha256": hashlib.sha256(
            str(list(prepared["calibration"].get("indices", []))).encode()
        ).hexdigest(),
        "heldout": heldout_info,
        "model_state_sha256": entry,
        "model_state_unchanged": True,
        "provenance": inp.provenance,
        "timings_seconds": timings,
    }
    return record, arrays


# ---- entry point ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    if cfg.get("family", "vision") != "vision":
        raise NotImplementedError("procrustes_screen is vision-only")
    out_dir = Path(cfg["out_dir"])
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; procrustes_screen results are write-once")
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
        record, arrays = screen_task(
            inp,
            sink,
            ranks=cfg.get("ranks"),
            heldout_max_batches=int(cfg.get("heldout_max_batches", 10)),
            native_ckpt_path=(
                cfg["native_target_ckpt_template"].format(task=inp.task)
                if cfg.get("native_target_ckpt_template")
                else None
            ),
        )
        task_dir = out_dir / inp.task
        task_dir.mkdir()
        _write_new(task_dir / "screen.json", json.dumps(_jsonable(record), indent=1, sort_keys=True))
        torch.save(arrays, task_dir / "arrays.pt")
        run_meta["tasks"][inp.task] = {
            "seconds": time.perf_counter() - t0,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gates": {k: g["passed"] for k, g in record["gates"].items()},
        }
        run_meta["total_seconds"] = time.perf_counter() - started
        (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))
        print(f"[procrustes_screen] {inp.task}: done in {run_meta['tasks'][inp.task]['seconds']:.0f}s", flush=True)
    run_meta["total_seconds"] = time.perf_counter() - started
    (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
