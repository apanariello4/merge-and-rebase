"""Realization drift of a SAVED Ariadne transported task vector, block by block (vision; no fitting, no accuracy).

For each task and strength ``alpha`` the whole saved vector is mounted as ``base + alpha * delta`` on the target and,
on the run's own calibration batches, every target block-end output is compared with the pristine one. With
``dT_j`` = (mounted block-j output) - (pristine block-j output) and ``D_j`` the TOTAL block-end target of the run's fit
(``_streaming_desired``: the paired source difference carried through the run's Procrustes map ``Q_j``), the rows give

- ``desired_norm``: ``||D_j||_F`` over all calibration rows;
- ``delta_norm``: ``||dT_j||_F``;
- ``ratio``: ``||dT_j|| / ||D_j||``;
- ``rel_error``: ``||dT_j - D_j|| / ||D_j||``;
- ``cosine``: ``<dT_j, D_j> / (||dT_j|| ||D_j||)`` with both flattened over the calibration rows.

Setup is rebuilt by ``ariadne_norm.vision.build_vision_inputs`` (same models, loaders, seed, pairing as the run), and
``Q_j`` by the fit's own ``prepare_direct_residual_streaming`` call (same ``num_batches``/``seed``/alignment options).
``D_j`` does not depend on ``block_target`` (that only changes the regression target of the fit), so runs fitted with
``total`` and ``increment`` are compared against the same reference.

Config JSON: ``{"run_summary", "tvs_dir" (default <run_dir>/transported_tvs), "tasks": "all"|[...],
"alphas": ["selected", 1.0, ...], "out_dir"}``. Run::

    python -m merge_and_rebase.eval.diagnostics.ariadne_realization --config cfg.json --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.rebase.methods._ariadne.capture import iter_capture_tokens
from merge_and_rebase.rebase.methods._ariadne.config import order_components
from merge_and_rebase.rebase.methods._ariadne.fit import _task_vector_sha256
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    _streaming_desired,
    _streaming_source_iters,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.run_logging import _code_fingerprint

ROW_KEYS = ("position", "desired_norm", "delta_norm", "ratio", "rel_error", "cosine")


@torch.no_grad()
def measure_block_realization(
    target_model,
    target_base_state: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    alphas: list[float],
    positions: list[int],
    target_batches: list,
    desired_fn,
    source_iters_fn,
    *,
    device,
    family_adapter=None,
) -> dict[float, dict[int, dict[str, Any]]]:
    """One lockstep sweep over the calibration batches: ``{alpha: {j: row}}`` (see module docstring).

    Same scheme as `measure_direct_residual_realization_streaming` (pristine target + one mounted deep copy per alpha
    + the source generators; ``desired_fn(k, source_values, t0)`` gives ``{j: D_j}``), with the inner product added.
    ``target_model`` is left untouched (its entry state is restored).
    """
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    base_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
    copies = {}
    for alpha in alphas:
        state = dict(base_state)
        for key, value in delta.items():
            state[key] = state[key] + float(alpha) * value.to(state[key])
        model = copy.deepcopy(target_model)
        model.load_state_dict(state, strict=True)
        copies[alpha] = model
    requests = {str(pos): (pos, "boundary") for pos in positions}
    acc = {a: {p: dict(d=0.0, t=0.0, err=0.0, dot=0.0) for p in positions} for a in alphas}
    try:
        target_model.load_state_dict(base_state, strict=True)
        gens = {
            "__t0__": iter_capture_tokens(target_model, target_batches, requests, device, family_adapter=family_adapter)
        }
        for alpha, model in copies.items():
            gens[alpha] = iter_capture_tokens(model, target_batches, requests, device, family_adapter=family_adapter)
        source_gens = source_iters_fn()
        names = list(gens)
        with contextlib.ExitStack() as stack:
            for gen in list(gens.values()) + list(source_gens.values()):
                stack.enter_context(contextlib.closing(gen))
            for k, values in enumerate(zip(*(gens[n] for n in names), *source_gens.values(), strict=True)):
                by_name = dict(zip(names, values[: len(names)], strict=True))
                source_values = dict(zip(source_gens, values[len(names) :], strict=True))
                t0 = by_name["__t0__"]
                desired = desired_fn(k, source_values, t0)
                for pos in positions:
                    d = desired[pos]
                    for alpha in alphas:
                        dt = by_name[alpha][str(pos)] - t0[str(pos)]
                        a = acc[alpha][pos]
                        a["d"] += float((d.double() ** 2).sum().item())
                        a["t"] += float((dt.double() ** 2).sum().item())
                        a["err"] += float(((dt - d).double() ** 2).sum().item())
                        a["dot"] += float((dt.double() * d.double()).sum().item())
    finally:
        del copies
        target_model.load_state_dict(entry_state, strict=True)
    out: dict[float, dict[int, dict[str, Any]]] = {}
    for alpha in alphas:
        out[alpha] = {}
        for pos in positions:
            a = acc[alpha][pos]
            d_norm, t_norm = a["d"] ** 0.5, a["t"] ** 0.5
            out[alpha][pos] = {
                "position": pos,
                "desired_norm": d_norm,
                "delta_norm": t_norm,
                "ratio": t_norm / (d_norm + 1e-12),
                "rel_error": (a["err"] ** 0.5) / (d_norm + 1e-12),
                "cosine": a["dot"] / (t_norm * d_norm + 1e-12),
            }
    return out


def measure_for_prepared(
    target_model,
    target_base_state,
    source_base_model,
    source_ft_model,
    prepared,
    pairing,
    delta,
    alphas,
    *,
    residual_target: str,
    device,
    family_adapter=None,
):
    """Bind `measure_block_realization` to a prepared streaming fit (its calibration batches, ``Q_j``, pairing)."""
    positions = list(range(pairing.target_depth))
    return measure_block_realization(
        target_model,
        target_base_state,
        delta,
        list(alphas),
        positions,
        prepared["target_batches"],
        lambda _k, source_values, t0: _streaming_desired(
            prepared, pairing, residual_target, source_values, t0, positions
        ),
        lambda: _streaming_source_iters(
            source_base_model, source_ft_model, prepared, pairing, device, family_adapter=family_adapter
        ),
        device=device,
        family_adapter=family_adapter,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_alphas(spec: list, task: str, summary: Mapping[str, Any]) -> list[tuple[str, float]]:
    """``[(label, alpha)]`` for the requested spec; ``"selected"`` is the run's ``selected_alpha_by_task[task]``."""
    out: list[tuple[str, float]] = []
    for item in spec:
        if item == "selected":
            selected = (summary.get("selected_alpha_by_task") or {}).get(task)
            if selected is None:
                raise ValueError(f"run has no selected alpha for {task!r}")
            out.append(("selected", float(selected)))
        else:
            out.append((str(float(item)), float(item)))
    return out


def run(cfg: dict, device: str) -> Path:
    from .ariadne_norm.vision import build_vision_inputs

    out_dir = Path(cfg["out_dir"])
    if out_dir.exists():
        raise FileExistsError(f"out_dir {out_dir} exists; refusing to overwrite")
    summary_path = Path(cfg["run_summary"]).resolve()
    summary = json.loads(summary_path.read_bytes())
    tvs_dir = Path(cfg.get("tvs_dir") or summary_path.parent / "transported_tvs")
    alpha_spec = list(cfg["alphas"])
    inputs = build_vision_inputs(
        {"reference_summary": str(summary_path), "tasks": cfg.get("tasks", "all"), "generic_num_batches": 0}, device
    )
    results: dict[str, Any] = {}
    shas: dict[str, str] = {}
    for inp in inputs:
        config = inp.config
        if config.procrustes_source != "activation" or config.component_target == "output_total":
            raise NotImplementedError("only procrustes_source='activation' block-boundary fits are supported")
        files = [tvs_dir / f"{inp.task}_{n}_transported_native.pt" for n in ("direct_residual", "ariadne")]
        files = [f for f in files if f.exists()]
        if len(files) != 1:
            raise FileNotFoundError(f"expected exactly one saved vector for {inp.task!r} in {tvs_dir}, found {files}")
        delta = torch.load(files[0], map_location="cpu")
        shas[inp.task] = _sha256(files[0])
        alphas = resolve_alphas(alpha_spec, inp.task, summary)
        prepared = prepare_direct_residual_streaming(
            inp.source_base_model,
            inp.target_model,
            inp.source_loader,
            inp.target_loader,
            inp.pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device=device,
            family_adapter=inp.family_adapter,
            source_ft_model=inp.source_ft_model,
            procrustes_source=config.procrustes_source,
            alignment_map=config.alignment_map,
            alignment_row_weighting=config.alignment_row_weighting,
            alignment_seed=config.alignment_seed,
            fidelity_alignment_diagnostics=bool(config.fidelity_holdout),
            procrustes_rank=config.procrustes_rank,
            procrustes_energy=config.procrustes_energy,
            exclude_first_token=config.exclude_first_token,
            mid_alignment=config.sequential_points == "mid_end",
        )
        measured = measure_for_prepared(
            inp.target_model,
            inp.target_base_state,
            inp.source_base_model,
            inp.source_ft_model,
            prepared,
            inp.pairing,
            delta,
            sorted({a for _, a in alphas}),
            residual_target=config.residual_target,
            device=device,
            family_adapter=inp.family_adapter,
        )
        results[inp.task] = {
            "tv_sha256_of_tensors": _task_vector_sha256(delta),
            "alphas": {
                label: {"alpha": a, "rows": [measured[a][j] for j in sorted(measured[a])]} for label, a in alphas
            },
        }
        del prepared, delta
    out_dir.mkdir(parents=True)
    payload = {
        "meta": {
            "code_fingerprint": _code_fingerprint(),
            "run_summary": str(summary_path),
            "run_summary_sha256": _sha256(summary_path),
            "tvs_dir": str(tvs_dir),
            "tv_file_sha256": shas,
            "alphas_requested": alpha_spec,
            "components": list(order_components(inp.config.components)) if shas else None,
            "row_keys": list(ROW_KEYS),
            "d_target": "total block-end target (_streaming_desired); independent of block_target",
        },
        "tasks": results,
    }
    (out_dir / "realization.json").write_text(json.dumps(payload, indent=2))
    return out_dir / "realization.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(run(json.loads(Path(args.config).read_text()), args.device))


if __name__ == "__main__":
    main()
