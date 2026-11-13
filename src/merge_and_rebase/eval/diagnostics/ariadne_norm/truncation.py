"""Depth truncation (vision, evaluation only): how much of a transported task vector's gain is carried by its early
target blocks?

``python -m merge_and_rebase.eval.diagnostics.ariadne_norm.truncation --config <json> --device cuda``

For a finished reference Ariadne run (single-task, unit saved vector ``tau``) and each cut ``k`` the vector
``tau_k`` keeps the updates of the target blocks ``j < k`` (every key ``visual.transformer.resblocks.<j>.*``) and is zero
elsewhere. Keys of ``tau`` that belong to no block (e.g. ``visual.proj``) are zeroed too unless ``keep_non_block`` is
true; the keys that exist outside blocks are listed in the output. The target model ``base + alpha * tau_k`` is
evaluated on val and test over the alpha grid with the run's own evaluation (``eval_task_top1`` on the task's own
loaders, exactly what ``TargetEvaluator.eval_rebased_indices`` calls) and alpha is selected on the split the run
searched on with the run's own rule (``PerTaskAlphaTracker``, first maximum, patience), also reported over the
full grid. Nothing is fitted and no calibration data is used.

Config keys (the reference run is read through ``ariadne_norm.vision.build_vision_inputs``, as ``readout`` does):

- ``family`` ("vision"), ``reference_summary``, ``tasks`` ("all" or a list), ``out_dir`` (``output_dir`` accepted; must
  not exist, results are write-once).
- ``ks``: list of cuts (0 <= k <= target depth; ``k = depth`` keeps every block).
- ``keep_non_block`` (false): keep the non-block keys of ``tau`` in every ``tau_k``.
- ``alpha_grid``: default = the reference run's own grid (``resolved.alpha.alphas``); a given grid always gains 0 and
  the run's selected alpha (snapped to the run's float spelling).
- ``gate_acc_tol`` (0.0): tolerance of the accuracy gates (the same evaluation code path is expected to reproduce the
  run's numbers exactly; ``difference_in_images`` is recorded).

Gates (a failure is recorded in ``truncation.json`` and raised after all tasks are written):

- ``g0_alpha0_reproduces_base``: the mounted state at alpha = 0 equals the target base bitwise, and the test accuracy
  equals the reference summary's baseline accuracy (only when the reference baseline is the zero-shot target, i.e.
  ``selected_baseline_alpha == 0``; otherwise that part is skipped).
- ``g1_full_reproduces_reference``: the UNTRUNCATED vector (identical to ``k = depth`` when ``tau`` has no non-block
  keys or ``keep_non_block`` is true; checked bitwise) evaluated at the run's selected alpha on test equals
  ``test_results.per_task_rebased``; with ``alpha_grid`` unset, the patience-emulated selection on the full vector
  must also pick that alpha and the val accuracy there is recorded.

Output: ``<out_dir>/<task>/truncation.json`` and ``<out_dir>/run.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.eval.utils import eval_task_top1
from merge_and_rebase.io.ckpt import load_into_model
from merge_and_rebase.merge.methods._common import axpy_state_dict
from merge_and_rebase.models.openclip_classifier import ZERO_SHOT_CACHE_DIR
from merge_and_rebase.rebase.methods._ariadne.fit import _task_vector_sha256
from merge_and_rebase.run_logging import _code_fingerprint
from merge_and_rebase.utils.alpha_search import PerTaskAlphaTracker

from .readout import GateError, _jsonable, _reference_block, _write_new, resolve_alpha_grid

Tensor = torch.Tensor
SPLITS = ("val", "test")
_BLOCK_RE = re.compile(r"(?:^|\.)resblocks\.(\d+)\.")


# ---- pure helpers -------------------------------------------------------------------------------------------------


def block_index(key: str) -> int | None:
    """Index ``j`` of the ``resblocks.<j>.`` a parameter key belongs to, ``None`` for non-block keys."""
    match = _BLOCK_RE.search(key)
    return int(match.group(1)) if match else None


def non_block_keys(tau: Mapping[str, Tensor]) -> list[str]:
    return sorted(k for k in tau if block_index(k) is None)


def truncate_task_vector(tau: Mapping[str, Tensor], k: int, *, keep_non_block: bool = False) -> dict[str, Tensor]:
    """``tau`` with every key of a block ``j >= k`` (and, unless ``keep_non_block``, every non-block key) zeroed.

    The key set and dtypes are those of ``tau`` (so ``axpy_state_dict`` touches the same keys); kept entries are the
    SAME tensors (bitwise identical), zeroed ones are ``zeros_like``.
    """
    out: dict[str, Tensor] = {}
    for key, value in tau.items():
        j = block_index(key)
        keep = keep_non_block if j is None else j < int(k)
        out[key] = value if keep else torch.zeros_like(value)
    return out


def select_like_run(alphas: Sequence[float], accs: Sequence[float], patience: int) -> int:
    """Index the run's per-task alpha search selects: ``PerTaskAlphaTracker`` (primary stream) fed in grid order,
    stopping once it early-stops, so alphas after the stop never compete."""
    tracker = PerTaskAlphaTracker(task_names=["t"], initial_alpha=float(alphas[0]), patience=int(patience))
    for alpha, acc in zip(alphas, accs, strict=True):
        tracker.update(alpha=float(alpha), indices=[0], primary_accs=[float(acc)], secondary_accs=[float("-inf")])
        if not tracker.primary_active[0]:
            break
    best = float(tracker.best_primary_alpha[0])
    return next(i for i, a in enumerate(alphas) if float(a) == best)


def select_first_max(accs: Sequence[float]) -> int:
    """First maximum over the whole grid (same ``1e-12`` improvement rule as the tracker, no early stop)."""
    best_i, best = 0, float("-inf")
    for i, acc in enumerate(accs):
        if float(acc) > best + 1e-12:
            best_i, best = i, float(acc)
    return best_i


def _acc_gate(ours: float | None, ref: float | None, n: int | None, tol: float, why: str) -> dict[str, Any]:
    if ours is None or ref is None:
        return {"passed": None, "skipped": why, "ours": ours, "reference": ref}
    diff = float(ours) - float(ref)
    return {
        "ours": float(ours),
        "reference": float(ref),
        "difference": diff,
        "difference_in_images": diff * n if n else None,
        "tolerance": tol,
        "exact": bool(diff == 0.0),
        "passed": bool(abs(diff) <= tol),
    }


def truncation_task(
    *,
    tau: Mapping[str, Tensor],
    base_state: Mapping[str, Tensor],
    depth: int,
    ks: Sequence[int],
    alphas: Sequence[float],
    evaluate: Callable[[Mapping[str, Tensor], str], float],
    alpha_star: float,
    search_split: str,
    patience: int,
    reference: Mapping[str, Any],
    keep_non_block: bool = False,
    gate_acc_tol: float = 0.0,
    n_test: int | None = None,
    grid_is_runs_own: bool = True,
) -> dict[str, Any]:
    """All curves, selections and gates of one task. ``evaluate(state_dict, split) -> accuracy`` mounts the state on
    the target model and scores it (injected so the logic is testable with a toy model)."""
    ks = [int(k) for k in ks]
    if any(k < 0 or k > depth for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"ks must be unique integers in [0, {depth}]; got {ks}")
    alphas = [float(a) for a in alphas]
    if 0.0 not in alphas or float(alpha_star) not in alphas:
        raise ValueError("the alpha grid must contain 0 and the reference run's selected alpha")
    i0, i_star = alphas.index(0.0), alphas.index(float(alpha_star))
    outside = non_block_keys(tau)
    blocks_present = sorted({j for key in tau if (j := block_index(key)) is not None})
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + now - clock
        clock = now

    # alpha = 0: independent of k (every tau_k contributes 0), evaluated once on the untouched base.
    zero_sd = axpy_state_dict(dict(base_state), dict(tau), 0.0)
    bitwise_base = all(torch.equal(zero_sd[key], base_state[key]) for key in base_state)
    zero_acc = {s: float(evaluate(zero_sd, s)) for s in SPLITS}
    del zero_sd
    lap("alpha0")

    curves: dict[str, dict[str, list[float]]] = {}
    tv_hash: dict[str, str] = {}
    full_equal: dict[str, bool] = {}
    for k in ks:
        tau_k = truncate_task_vector(tau, k, keep_non_block=keep_non_block)
        tv_hash[str(k)] = _task_vector_sha256(tau_k)
        full_equal[str(k)] = all(torch.equal(tau_k[key], tau[key]) for key in tau)
        curve = {s: [] for s in SPLITS}
        for i, alpha in enumerate(alphas):
            for s in SPLITS:
                if i == i0:
                    curve[s].append(zero_acc[s])
                else:
                    sd = axpy_state_dict(dict(base_state), dict(tau_k), alpha)
                    curve[s].append(float(evaluate(sd, s)))
                    del sd
        curves[str(k)] = curve
        lap(f"k{k}")

    selection: dict[str, Any] = {}
    for k in ks:
        c = curves[str(k)]
        i_run = select_like_run(alphas, c[search_split], patience)
        i_full = select_first_max(c[search_split])
        selection[str(k)] = {
            "selection_split": search_split,
            "run_rule": {  # PerTaskAlphaTracker with the run's patience over the grid in order
                "patience": int(patience),
                "selected_alpha": alphas[i_run],
                "val_acc": c["val"][i_run],
                "test_acc": c["test"][i_run],
            },
            "full_grid": {"selected_alpha": alphas[i_full], "val_acc": c["val"][i_full], "test_acc": c["test"][i_full]},
            "test_acc_at_reference_alpha": c["test"][i_star],
            "val_acc_at_reference_alpha": c["val"][i_star],
        }

    # G1: the untruncated vector at the run's alpha. Reuse the k = depth curve only when it is bitwise tau.
    reuse = full_equal.get(str(depth), False)
    if reuse:
        full_test, full_val, source = curves[str(depth)]["test"][i_star], curves[str(depth)]["val"][i_star], "k=depth"
    else:
        sd = axpy_state_dict(dict(base_state), dict(tau), float(alpha_star))
        full_test, full_val, source = (
            float(evaluate(sd, "test")),
            float(evaluate(sd, "val")),
            "untruncated (extra eval)",
        )
        del sd
        lap("g1_extra_eval")
    g1 = _acc_gate(
        full_test,
        reference.get("test_rebased"),
        n_test,
        gate_acc_tol,
        "reference summary has no per-task test accuracy",
    )
    g1.update(full_vector_source=source, val_acc_at_reference_alpha=full_val, k_depth_equals_tau_bitwise=reuse)
    if grid_is_runs_own:
        # the run's search, replayed on the full vector, must land on the run's alpha (same grid, same rule)
        full_curve = curves[str(depth)][search_split] if reuse else None
        if full_curve is not None:
            replay = alphas[select_like_run(alphas, full_curve, patience)]
            g1["patience_replay_selects_reference_alpha"] = bool(replay == float(alpha_star))
            g1["patience_replay_alpha"] = replay
    baseline_zeroshot = float(reference.get("baseline_alpha") or 0.0) == 0.0
    g0 = _acc_gate(
        zero_acc["test"],
        reference.get("test_baseline") if baseline_zeroshot else None,
        n_test,
        gate_acc_tol,
        "the reference baseline is not the zero-shot target",
    )
    g0.update(mounted_state_equals_target_base_bitwise=bitwise_base)
    if not bitwise_base:
        g0["passed"] = False
    gates = {"g0_alpha0_reproduces_base": g0, "g1_full_reproduces_reference": g1}
    return {
        "alphas": alphas,
        "reference_alpha": float(alpha_star),
        "ks": ks,
        "depth": int(depth),
        "keep_non_block": bool(keep_non_block),
        "blocks_in_task_vector": blocks_present,
        "non_block_keys": outside,
        "curves": curves,
        "zero_alpha_acc": zero_acc,
        "selection": selection,
        "task_vector_sha256_by_k": tv_hash,
        "k_equals_untruncated_vector": full_equal,
        "gates": gates,
        "failed_gates": [n for n, g in gates.items() if g.get("passed") is False],
        "timings_seconds": timings,
    }


# ---- model-bound evaluator and entry point ------------------------------------------------------------------------


def _make_evaluator(inp: Any, sink: Mapping[str, Any], strict: bool) -> Callable[[Mapping[str, Tensor], str], float]:
    ctx, clf = sink["eval_context"], sink["target_clf"]

    def evaluate(sd: Mapping[str, Tensor], split: str) -> float:
        load_into_model(clf.model, dict(sd), strict=strict)
        return float(
            eval_task_top1(
                clf=clf,
                loaders=ctx.loaders,
                classnames=list(ctx.classnames),
                build_cfg_task=ctx.build_cfg_task,
                device=inp.device,
                split=split,
            )
        )

    return evaluate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    if cfg.get("family", "vision") != "vision":
        raise NotImplementedError("truncation is vision-only")
    out_dir = Path(cfg.get("out_dir") or cfg["output_dir"])
    ks = [int(k) for k in cfg["ks"]]
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; truncation results are write-once")
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
    sink: dict[str, Any] = {}
    started = time.perf_counter()
    failures: dict[str, list[str]] = {}
    for inp in build_vision_inputs({**cfg, "generic_num_batches": 0}, args.device, context_sink=sink):
        t_task = time.perf_counter()
        resolved, summary = sink["resolved"], sink["summary"]
        reference = _reference_block(summary, resolved, inp.task, inp.alpha_star)
        if reference["merge_mode"] != "none":
            raise NotImplementedError(f"reference merge_mode={reference['merge_mode']!r}: single-task runs only")
        if inp.alpha_star is None or inp.reference_task_vector is None:
            raise ValueError(f"{inp.task}: needs the reference run's selected alpha and its saved unit task vector")
        tau = inp.reference_task_vector
        if not set(tau) <= set(inp.target_base_state):
            raise ValueError(f"{inp.task}: task vector keys are not all keys of the target base")
        own_grid = cfg.get("alpha_grid") is None
        if own_grid:  # the run's own float spelling, untouched (the selected alpha must be one of its entries)
            alphas, grid_info = [float(a) for a in resolved.alpha.alphas], {"source": "reference run"}
            if float(inp.alpha_star) not in alphas:
                raise ValueError(f"{inp.task}: the run's own grid lacks its selected alpha {inp.alpha_star}")
        else:
            alphas, grid_info = resolve_alpha_grid(cfg["alpha_grid"], inp.alpha_star)
        ctx = sink["eval_context"]
        record = truncation_task(
            tau=tau,
            base_state=inp.target_base_state,
            depth=inp.pairing.target_depth,
            ks=ks,
            alphas=alphas,
            evaluate=_make_evaluator(inp, sink, bool(resolved.strict_load)),
            alpha_star=float(inp.alpha_star),
            search_split=resolved.alpha.search_split,
            patience=int(resolved.alpha.patience),
            reference=reference,
            keep_non_block=bool(cfg.get("keep_non_block", False)),
            gate_acc_tol=float(cfg.get("gate_acc_tol", 0.0)),
            n_test=len(ctx.loaders.test.dataset),
            grid_is_runs_own=own_grid,
        )
        load_into_model(sink["target_clf"].model, dict(inp.target_base_state), strict=bool(resolved.strict_load))
        record.update(
            task=inp.task,
            reference=reference,
            alpha_grid=grid_info,
            alpha_grid_is_runs_own=own_grid,
            provenance=inp.provenance,
            zs_cache_dir=ZERO_SHOT_CACHE_DIR,
        )
        task_dir = out_dir / inp.task
        task_dir.mkdir()
        _write_new(task_dir / "truncation.json", json.dumps(_jsonable(record), indent=1, sort_keys=True))
        run_meta["tasks"][inp.task] = {
            "seconds": time.perf_counter() - t_task,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gates": {k: g.get("passed") for k, g in record["gates"].items()},
            "failed_gates": record["failed_gates"],
        }
        if record["failed_gates"]:
            failures[inp.task] = record["failed_gates"]
        run_meta["total_seconds"] = time.perf_counter() - started
        (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))
        print(f"[truncation] {inp.task}: done in {run_meta['tasks'][inp.task]['seconds']:.0f}s", flush=True)
    run_meta["total_seconds"] = time.perf_counter() - started
    (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))
    if failures:
        raise GateError(f"correctness gate(s) failed: {failures}; see {out_dir}")


if __name__ == "__main__":
    main()
