"""Family-agnostic N0-N7 measurement of one task of an Ariadne reference run.

Sweeps (all over the reference run's own calibration batches unless stated):

1. Pass A + instrumented Pass B: the method's own ``prepare_direct_residual_streaming`` /
   ``fit_direct_residual_streaming`` with a `FitRecorder` stats factory (N0, N3, N5, N7-regression).
2. Cache sweep: per batch, the aligned source displacement ``f - b`` per distinct source block (CPU, float32, same
   arithmetic as Pass B) and odd/even Procrustes cross-covariances (N7-Procrustes).
3. Mount sweeps: pristine target and a mounted copy in lockstep; joint ``alpha * tau`` over an alpha grid (N2),
   every single block at alpha = 1 (N2 accumulation matrix), and the joint vector on off-task batches (N6).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from merge_and_rebase.rebase.methods._ariadne.alignment import _procrustes_from_cross
from merge_and_rebase.rebase.methods._ariadne.capture import iter_capture_tokens
from merge_and_rebase.rebase.methods._ariadne.fit import _task_vector_sha256
from merge_and_rebase.rebase.methods._ariadne.layouts import _aligned
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    _StreamingCrossCovariance,
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)

from .analysis import RealizationAccumulator, label_free_alpha, procrustes_split_half
from .inputs import DiagnosticInputs
from .instrument import FitRecorder

Tensor = torch.Tensor


def check_supported_config(cfg) -> None:
    """The analysis covers the paper configuration only; anything else would be measured incorrectly."""
    expected = {
        "activation_storage": "streaming",
        "component_target": "block_boundary",
        "block_split": "none",
        "residual_target": "transported_delta",
        "procrustes_source": "activation",
        "alignment_map": "polar",
        "alignment_row_weighting": "uniform",
        "tv_scaling": "none",
        "endpoint_construction": "native_delta",
        "exact_form": True,
    }
    bad = {k: getattr(cfg, k) for k, v in expected.items() if getattr(cfg, k) != v}
    if len(cfg.components) != 1:
        bad["components"] = cfg.components
    if cfg.ridge_estimator not in {"empirical_bayes", "fixed_relative"}:
        bad["ridge_estimator"] = cfg.ridge_estimator
    if bad:
        raise ValueError(f"ariadne_norm diagnostics support only the paper configuration; got {bad}")


def _bias_key(weight_key: str) -> str:
    return f"{weight_key[: -len('.weight')]}.bias"


def compare_task_vectors(ours: Mapping[str, Tensor], reference: Mapping[str, Tensor] | None, mode: str) -> dict:
    """N0: ``exact`` requires identical SHA-256; ``tolerance`` reports max abs and relative Frobenius differences."""
    out: dict[str, Any] = {"sha256": _task_vector_sha256(ours), "reference": None}
    if reference is None:
        return out
    keys_ours, keys_ref = set(ours), set(reference)
    diff_sq = ref_sq = 0.0
    max_abs = 0.0
    for key in sorted(keys_ours & keys_ref):
        a = ours[key].double().cpu()
        b = reference[key].double().cpu()
        if a.shape != b.shape:
            raise ValueError(f"N0: shape mismatch at {key}: {tuple(a.shape)} vs {tuple(b.shape)}")
        diff_sq += float(((a - b) ** 2).sum())
        ref_sq += float((b**2).sum())
        max_abs = max(max_abs, float((a - b).abs().max()))
    ref_sha = _task_vector_sha256(reference)
    out["reference"] = {
        "mode": mode,
        "sha256": ref_sha,
        "sha256_equal": ref_sha == out["sha256"],
        "missing_in_ours": sorted(keys_ref - keys_ours),
        "missing_in_reference": sorted(keys_ours - keys_ref),
        "max_abs_diff": max_abs,
        "relative_frobenius_diff": (diff_sq / ref_sq) ** 0.5 if ref_sq > 0 else 0.0,
    }
    ok = not out["reference"]["missing_in_ours"] and not out["reference"]["missing_in_reference"]
    if mode == "exact":
        ok = ok and out["reference"]["sha256_equal"]
    else:
        ok = ok and out["reference"]["relative_frobenius_diff"] < 1e-5
    out["reference"]["passed"] = bool(ok)
    return out


def compare_rows(ours: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]] | None) -> dict | None:
    """N0 for runs without a saved vector: per-position solver norms against the summary rows."""
    if not reference:
        return None
    ref_by_pos = {int(r["position"]): r for r in reference}
    worst = 0.0
    per_pos = {}
    for row in ours:
        j = int(row["position"])
        ref = ref_by_pos.get(j)
        if ref is None:
            continue
        rel = {
            k: abs(float(row[k]) - float(ref[k])) / max(abs(float(ref[k])), 1e-300)
            for k in ("correction_norm", "bias_norm", "desired_norm", "residual_norm_after", "ridge")
            if k in ref
        }
        per_pos[j] = rel
        worst = max([worst, *rel.values()])
    return {"max_relative_diff": worst, "by_position": per_pos, "passed": worst < 1e-5}


def norm_profile(inp: DiagnosticInputs, corrections: Mapping[str, Tensor], rows: Sequence[Mapping[str, Any]]) -> dict:
    """N1: where the norm sits, against the source fine-tuning delta of the same projection."""
    src_base = inp.source_base_model.state_dict()
    src_ft = inp.source_ft_model.state_dict()
    full_sq = 0.0
    for key, base in src_base.items():
        if key in src_ft and base.is_floating_point():
            full_sq += float(((src_ft[key].double().cpu() - base.double().cpu()) ** 2).sum())
    src_component_sq = 0.0
    for i in range(inp.pairing.source_depth):
        wk = inp.source_component_key(i)
        for key in (wk, _bias_key(wk)):
            if key in src_base:
                src_component_sq += float(((src_ft[key].double().cpu() - src_base[key].double().cpu()) ** 2).sum())
    per_position = []
    for row in rows:
        j = int(row["position"])
        i = int(inp.pairing.pairing[j])
        wk = inp.component_key(j)
        w = corrections[wk].double()
        b = corrections.get(_bias_key(wk))
        swk = inp.source_component_key(i)
        sw = src_ft[swk].double().cpu() - src_base[swk].double().cpu()
        sb_key = _bias_key(swk)
        sb = (src_ft[sb_key].double().cpu() - src_base[sb_key].double().cpu()) if sb_key in src_base else None
        n_rows = int(row["n_rows"])
        per_position.append(
            {
                "position": j,
                "source_position": i,
                "correction_norm": float(w.norm()),
                "correction_rms": float(w.norm()) / math.sqrt(w.numel()),
                "bias_norm": float(b.double().norm()) if b is not None else 0.0,
                "desired_norm": float(row["desired_norm"]),
                "desired_rms_per_row": float(row["desired_norm"]) / math.sqrt(n_rows),
                "source_weight_delta_norm": float(sw.norm()),
                "source_weight_delta_rms": float(sw.norm()) / math.sqrt(sw.numel()),
                "source_bias_delta_norm": float(sb.norm()) if sb is not None else None,
                "norm_ratio_to_source": float(w.norm()) / max(float(sw.norm()), 1e-300),
                "rms_ratio_to_source": (float(w.norm()) / math.sqrt(w.numel()))
                / max(float(sw.norm()) / math.sqrt(sw.numel()), 1e-300),
                "n_rows": n_rows,
                "unreachable_fraction": float(row["unreachable_residual_norm"])
                / max(float(row["desired_norm"]), 1e-300),
                "relative_residual_after": float(row["relative_residual_after"]),
            }
        )
    tau_sq = sum(float((v.double() ** 2).sum()) for v in corrections.values())
    tau_w_sq = sum(float((v.double() ** 2).sum()) for k, v in corrections.items() if k.endswith(".weight"))
    return {
        "tau_target_norm": tau_sq**0.5,
        "tau_target_weight_norm": tau_w_sq**0.5,
        "tau_target_bias_norm": max(0.0, tau_sq - tau_w_sq) ** 0.5,
        "tau_source_full_norm": full_sq**0.5,
        "tau_source_component_norm": src_component_sq**0.5,
        "per_position": per_position,
    }


def _cache_sweep(
    inp: DiagnosticInputs, prepared: Mapping[str, Any], *, token_split: bool = False
) -> tuple[list[dict[int, Tensor]], dict]:
    """Aligned source displacements per batch and odd/even Procrustes maps (N7-Procrustes).

    ``token_split`` adds a second pair of maps from token-index-parity halves of the SAME batches (every image in
    both halves), keyed ``_qa_token`` / ``_qb_token``.
    """
    pairing = inp.pairing
    distinct = sorted(set(pairing.pairing))
    src_req = {str(i): (i, "boundary") for i in distinct}
    tgt_req = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    halves = [{j: _StreamingCrossCovariance(device=inp.device) for j in range(pairing.target_depth)} for _ in range(2)]
    token_halves = (
        [{j: _StreamingCrossCovariance(device=inp.device) for j in range(pairing.target_depth)} for _ in range(2)]
        if token_split
        else []
    )
    gens = [
        iter_capture_tokens(
            inp.source_base_model,
            prepared["source_batches"],
            src_req,
            inp.device,
            family_adapter=inp.family_adapter,
            store_device=inp.device,
        ),
        iter_capture_tokens(
            inp.source_ft_model,
            prepared["source_batches"],
            src_req,
            inp.device,
            family_adapter=inp.family_adapter,
            store_device=inp.device,
        ),
        iter_capture_tokens(
            inp.target_model,
            prepared["target_batches"],
            tgt_req,
            inp.device,
            family_adapter=inp.family_adapter,
            store_device=inp.device,
        ),
    ]
    cache: list[dict[int, Tensor]] = []
    with contextlib.ExitStack() as stack:
        for g in gens:
            stack.enter_context(contextlib.closing(g))
        for k, (sb, sf, tgt) in enumerate(zip(*gens, strict=True)):
            ref_cpu = tgt["0"].cpu()
            batch_cache = {}
            for i in distinct:
                b = _aligned([sb[str(i)].cpu()], [ref_cpu])[0]
                f = _aligned([sf[str(i)].cpu()], [ref_cpu])[0]
                batch_cache[i] = (f - b).clone()
            cache.append(batch_cache)
            for j in range(pairing.target_depth):
                i = pairing.pairing[j]
                t = tgt[str(j)]
                x = _aligned([sb[str(i)]], [t])[0]
                x_rows, y_rows = x.reshape(-1, x.shape[-1]).double(), t.reshape(-1, t.shape[-1]).double()
                halves[k % 2][j].update(x_rows, y_rows)
                if token_halves:
                    odd = (inp.token_index(prepared["target_batches"][k], int(y_rows.shape[0])) % 2).to(torch.bool)
                    token_halves[0][j].update(x_rows, y_rows, row_mask=~odd.to(x_rows.device))
                    token_halves[1][j].update(x_rows, y_rows, row_mask=odd.to(x_rows.device))
    q_full = prepared["activation_q64_by_position"]
    stability = {}
    for j in range(pairing.target_depth):
        qa = _procrustes_from_cross(halves[0][j].cross()).cpu()
        qb = _procrustes_from_cross(halves[1][j].cross()).cpu()
        stability[j] = {"q": procrustes_split_half(qa, qb, q_full[j]), "_qa": qa.float(), "_qb": qb.float()}
        if token_halves:
            qa_t = _procrustes_from_cross(token_halves[0][j].cross()).cpu()
            qb_t = _procrustes_from_cross(token_halves[1][j].cross()).cpu()
            stability[j]["q_token_split"] = procrustes_split_half(qa_t, qb_t, q_full[j])
            stability[j]["_qa_token"], stability[j]["_qb_token"] = qa_t.float(), qb_t.float()
    return cache, stability


def _desired(cache_k: Mapping[int, Tensor], pairing, q_by_position, j: int) -> Tensor:
    return cache_k[pairing.pairing[j]] @ q_by_position[j]


def _mounted_state(base: Mapping[str, Tensor], delta: Mapping[str, Tensor], alpha: float) -> dict[str, Tensor]:
    state = dict(base)
    for key, value in delta.items():
        state[key] = base[key] + alpha * value.to(base[key].dtype)
    return state


def _measure(
    inp: DiagnosticInputs,
    mounted_model: torch.nn.Module,
    delta: Mapping[str, Tensor],
    alpha: float,
    batches: Sequence[Any],
    positions: Sequence[int],
    desired_fn=None,
    first_token: bool = False,
) -> dict[int, dict[str, float]]:
    """Realization sums for one mounted delta. ``first_token`` adds, per position, the same sums restricted to the
    rows with token index 0 (vision: CLS; decoders: first content token) under ``rows[j]["first_token"]``."""
    mounted_model.load_state_dict(_mounted_state(inp.target_base_state, delta, alpha), strict=True)
    req = {str(j): (j, "boundary") for j in positions}
    acc = RealizationAccumulator(positions)
    acc_first = RealizationAccumulator(positions) if first_token else None
    g0 = iter_capture_tokens(
        inp.target_model, batches, req, inp.device, family_adapter=inp.family_adapter, store_device=inp.device
    )
    g1 = iter_capture_tokens(
        mounted_model, batches, req, inp.device, family_adapter=inp.family_adapter, store_device=inp.device
    )
    with contextlib.closing(g0), contextlib.closing(g1):
        for k, (t0, t1) in enumerate(zip(g0, g1, strict=True)):
            first = None
            for j in positions:
                d = desired_fn(k, j) if desired_fn is not None else None
                dt = t1[str(j)] - t0[str(j)]
                acc.add(j, dt, d, t0[str(j)])
                if acc_first is not None:
                    width = dt.shape[-1]
                    if first is None:
                        first = inp.token_index(batches[k], dt.numel() // width).to(dt.device) == 0
                    acc_first.add(
                        j,
                        dt.reshape(-1, width)[first],
                        None if d is None else d.reshape(-1, width).to(dt.device)[first],
                        t0[str(j)].reshape(-1, width)[first],
                    )
    rows = acc.rows()
    if acc_first is not None:
        for j, row in acc_first.rows().items():
            rows[j]["first_token"] = row
    return rows


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def run_task(
    inp: DiagnosticInputs,
    *,
    alpha_grid: Sequence[float],
    row_positions: Sequence[int],
    generic_alphas: Sequence[float] | None = None,
    measure_realization: bool = True,
    token_split: bool = False,
    first_token_realization: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
    """Returns ``(record, arrays, task_vector)`` for one task.

    ``measure_realization=False`` skips the mount sweeps (N2, N6); ``token_split`` adds token-parity split halves to
    N7 (regression and Procrustes) next to the odd/even-batch split.
    """
    cfg = inp.config
    check_supported_config(cfg)
    entry_sha = _state_sha(inp.target_model)
    prepared = prepare_direct_residual_streaming(
        inp.source_base_model,
        inp.target_model,
        inp.source_loader,
        inp.target_loader,
        inp.pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device=inp.device,
        family_adapter=inp.family_adapter,
        source_ft_model=inp.source_ft_model,
        procrustes_source=cfg.procrustes_source,
        alignment_map=cfg.alignment_map,
        alignment_row_weighting=cfg.alignment_row_weighting,
        alignment_seed=cfg.alignment_seed,
        fidelity_alignment_diagnostics=bool(cfg.fidelity_holdout),
    )
    target_batches = prepared["target_batches"]
    recorder = FitRecorder(
        ridge_estimator=cfg.ridge_estimator,
        ridge_relative=cfg.ridge_relative,
        exact_form=cfg.exact_form,
        row_positions=row_positions,
        token_index_fn=lambda k, n: inp.token_index(target_batches[k], n),
        token_split=token_split,
    )
    corrections, rows = fit_direct_residual_streaming(
        inp.target_model,
        inp.target_base_state,
        inp.source_base_model,
        inp.source_ft_model,
        prepared,
        inp.pairing,
        config=cfg,
        device=inp.device,
        family_adapter=inp.family_adapter,
        stats_factory=recorder.factory,
    )
    corrections = {k: v.detach().float().cpu() for k, v in corrections.items()}
    rows_clean = [{k: v for k, v in r.items() if not isinstance(v, torch.Tensor)} for r in rows]

    record: dict[str, Any] = {
        "task": inp.task,
        "family": inp.family,
        "provenance": inp.provenance,
        "calibration": {k: v for k, v in prepared["calibration"].items() if k != "indices"},
        "calibration_indices_sha256": hashlib.sha256(
            str(list(prepared["calibration"].get("indices", []))).encode()
        ).hexdigest(),
        "fit_rows": rows_clean,
    }
    record["n0"] = {
        "task_vector": compare_task_vectors(
            corrections, inp.reference_task_vector, inp.reference_task_vector_tolerance
        ),
        "rows": compare_rows(rows_clean, inp.reference_rows),
        "analysis_matches_solver": {
            j: {
                "correction_norm_rel": abs(
                    r["n3"]["at_official"]["correction_norm"] - r["n3"]["solver_correction_norm"]
                )
                / max(r["n3"]["solver_correction_norm"], 1e-300),
                "bias_norm_rel": abs(r["n3"]["at_official"]["bias_norm"] - r["n3"]["solver_bias_norm"])
                / max(r["n3"]["solver_bias_norm"], 1e-300),
                "residual_rel": abs(
                    r["n3"]["at_official"]["residual_norm_after"] - r["n3"]["solver_residual_norm_after"]
                )
                / max(r["n3"]["solver_residual_norm_after"], 1e-300),
                "lam_rel": abs(r["n3"]["lam"] - r["n3"]["solver_lam"]) / max(r["n3"]["solver_lam"], 1e-300),
            }
            for j, r in recorder.records.items()
        },
    }
    record["n1"] = norm_profile(inp, corrections, rows_clean)
    record["n3"] = {j: r["n3"] for j, r in sorted(recorder.records.items())}
    record["n5"] = {j: r["n5"] for j, r in sorted(recorder.records.items()) if "n5" in r}
    record["n7"] = {j: {"regression": r["n7_regression"]} for j, r in sorted(recorder.records.items())}

    cache, q_stability = _cache_sweep(inp, prepared, token_split=token_split)
    q_by_position = prepared["q_by_position"]
    for j, value in q_stability.items():
        # Desired-effect stability under the half maps, measured on all calibration batches.
        num = den = 0.0
        for cache_k in cache:
            ds = cache_k[inp.pairing.pairing[j]]
            da, db, df = ds @ value["_qa"], ds @ value["_qb"], ds @ q_by_position[j]
            num += float(((da - db).double() ** 2).sum())
            den += float((df.double() ** 2).sum())
        record["n7"][j]["procrustes"] = {**value["q"], "desired_rel_diff": (num / den) ** 0.5 if den > 0 else 0.0}
        if "_qa_token" in value:
            num_t = 0.0
            for cache_k in cache:
                ds = cache_k[inp.pairing.pairing[j]]
                num_t += float(((ds @ value["_qa_token"] - ds @ value["_qb_token"]).double() ** 2).sum())
            record["n7"][j]["procrustes_token_split"] = {
                **value["q_token_split"],
                "desired_rel_diff": (num_t / den) ** 0.5 if den > 0 else 0.0,
            }
        if "n7_regression_token_split" in recorder.records.get(j, {}):
            record["n7"][j]["regression_token_split"] = recorder.records[j]["n7_regression_token_split"]

    if measure_realization:
        positions = list(range(inp.pairing.target_depth))

        def desired_fn(k, j):
            return _desired(cache[k], inp.pairing, q_by_position, j)

        mounted = copy.deepcopy(inp.target_model)
        try:
            joint = {}
            for alpha in alpha_grid:
                joint[float(alpha)] = _measure(
                    inp,
                    mounted,
                    corrections,
                    float(alpha),
                    target_batches,
                    positions,
                    desired_fn,
                    first_token=first_token_realization,
                )
            own = {}
            for j in positions:
                wk = inp.component_key(j)
                delta_j = {k: v for k, v in corrections.items() if k in (wk, _bias_key(wk))}
                own[j] = _measure(
                    inp,
                    mounted,
                    delta_j,
                    1.0,
                    target_batches,
                    positions[j:],
                    desired_fn,
                    first_token=first_token_realization,
                )
            generic = {}
            alphas_generic = sorted(
                {1.0, *([inp.alpha_star] if inp.alpha_star is not None else []), *(generic_alphas or [])}
            )
            for alpha in alphas_generic:
                generic[float(alpha)] = {
                    "generic": _measure(
                        inp,
                        mounted,
                        corrections,
                        alpha,
                        inp.generic_target_batches,
                        positions,
                        None,
                        first_token=first_token_realization,
                    ),
                    "calibration": joint.get(float(alpha))
                    or _measure(
                        inp,
                        mounted,
                        corrections,
                        alpha,
                        target_batches,
                        positions,
                        desired_fn,
                        first_token=first_token_realization,
                    ),
                }
        finally:
            del mounted
        record["n2"] = {
            "alpha_star": inp.alpha_star,
            "joint_by_alpha": {a: rows_a for a, rows_a in joint.items()},
            "label_free_alpha": label_free_alpha(joint),
            "single_block": {
                j: {
                    "own_r": own[j][j]["r"],
                    "own_cos": own[j][j]["cos"],
                    "own_relative_error": own[j][j]["relative_error"],
                    "projection_by_position": {jj: own[j][jj]["projection_coefficient"] for jj in own[j]},
                    "r_by_position": {jj: own[j][jj]["r"] for jj in own[j]},
                }
                for j in positions
            },
        }
        # Superposition check: sum over k<=j of block k's projection onto D_j against the joint projection at alpha=1.
        if 1.0 in joint:
            record["n2"]["superposition"] = {
                j: {
                    "sum_single_projection": sum(
                        own[k][j]["projection_coefficient"] or 0.0 for k in positions if k <= j
                    ),
                    "joint_projection": joint[1.0][j]["projection_coefficient"],
                }
                for j in positions
            }
        record["n6"] = generic
    exit_sha = _state_sha(inp.target_model)
    if exit_sha != entry_sha:
        raise RuntimeError("target model state changed during the diagnostics")
    record["target_state_sha256"] = entry_sha
    return record, recorder.arrays, corrections
