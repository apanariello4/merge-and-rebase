"""GPA universe of the anchor frames (AGENT_NEXT.md section 4): one centre ``G(s)`` and one ``A_m(s)`` per model.

The universe lives on a depth grid ``s_j = j / (n_grid - 1)`` (the run's target depth); at grid point ``j`` every
model contributes its frame at block ``pi_m(s_j) = round(s_j (L_m - 1))`` (the relative pairing). It is fitted on
``fit_models`` and stored under ``<state_dir>/universes/<version>/universe.pt``; ``version`` hashes the fitted
models (with their stored frame hashes), the frame config, the grid and ``d_U``, so a different set of models
gives a different universe and an encoded effect can name the frame it lives in. Models in ``add_models`` are
added afterwards by a Procrustes towards the FIXED ``G`` (``procrustes_to_universe``): ``G`` and the fitted
models' maps are unchanged, and so is the version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ...discrete_layer_match import DiscreteLayerPairing
from .core import dedup_mask, gpa, procrustes_to_universe, row_normalize
from .state import AnchorState

Tensor = torch.Tensor


@dataclass
class Universe:
    version: str
    fit_models: list[str]
    n_grid: int
    d_u: int
    g: list[Tensor]  # per grid point, [T, d_U]
    maps: dict[str, list[Tensor]]  # model id -> per grid point [d_U, d_m]
    info: list[dict[str, Any]]  # per grid point (fit diagnostics)
    added: list[str] = field(default_factory=list)

    def block_maps(self, j: int, source_id: str, target_id: str) -> tuple[Tensor, Tensor]:
        return self.maps[source_id][j], self.maps[target_id][j]


def _block(state: AnchorState, model_id: str, n_grid: int, j: int) -> int:
    reg = state.model(model_id)
    return int(DiscreteLayerPairing.compute(reg.depth, n_grid).pairing[j])


def _keep_masks(state: AnchorState, fit_models: Sequence[str], n_grid: int, filter_k: float | None):
    """Per grid point, the anchors kept by the high-norm filter (flagged in none of the fitted models), or None."""
    if filter_k is None:
        return [None] * n_grid
    from .method import anchor_keep

    return [
        anchor_keep([(state.model(m), _block(state, m, n_grid, j)) for m in fit_models], filter_k)
        for j in range(n_grid)
    ]


def _frame_at(state: AnchorState, model_id: str, n_grid: int, j: int, keep, filter_k: float | None) -> Tensor:
    reg = state.model(model_id)
    block = _block(state, model_id, n_grid, j)
    if keep is None:
        return reg.p[block].to(torch.float64)
    from .method import filtered_frame

    return filtered_frame(reg, block, keep, filter_k)[0]


def _processed_frames(state, models, n_grid, j, keep, filter_k, dedup, row_norm) -> dict[str, Tensor]:
    """Frames of ``models`` at grid point ``j``: norm filter (``keep``), then the R2 dedup measured in the reference
    model ``dedup[0]`` (``(ref_id, cos_max, max_count, seed)``), then the R3 row normalisation."""
    frames = {m: _frame_at(state, m, n_grid, j, keep, filter_k) for m in models}
    if dedup is not None:
        ref, cos_max, max_count, seed = dedup
        ref_frame = frames[ref] if ref in frames else _frame_at(state, ref, n_grid, j, keep, filter_k)
        sel = dedup_mask(ref_frame, cos_max, max_count=max_count, seed=seed)
        frames = {m: f[sel] for m, f in frames.items()}
    if row_norm:
        frames = {m: row_normalize(f) for m, f in frames.items()}
    return frames


def universe_version(
    state: AnchorState,
    fit_models: Sequence[str],
    n_grid: int,
    d_u: int | None,
    filter_k: float | None = None,
    dedup: tuple | None = None,
    row_norm: bool = False,
) -> str:
    payload = {
        "frame": state.config.frame(),
        "models": {m: state.model_meta_hash(m) for m in sorted(fit_models)},
        "n_grid": int(n_grid),
        "d_u": d_u,
    }
    if filter_k is not None:
        payload["norm_filter_k"] = float(filter_k)
    if dedup is not None:  # the defaults leave the version of existing universes unchanged
        payload["dedup"] = [dedup[0], float(dedup[1]), dedup[2], int(dedup[3])]
    if row_norm:
        payload["row_norm"] = True
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def fit_universe(
    state: AnchorState,
    fit_models: Sequence[str],
    *,
    n_grid: int,
    d_u: int | None = None,
    add_models: Sequence[str] = (),
    filter_k: float | None = None,
    dedup: tuple | None = None,
    row_norm: bool = False,
) -> Universe:
    """Load the universe of ``fit_models`` from the state (or fit and store it), then add ``add_models``.

    ``filter_k``: the high-norm anchor filter (``AnchorState.norm_filter`` of every fitted model must be computed);
    at each grid point the anchors flagged in any fitted model are dropped, and an added model uses the same rows.
    """
    fit_models = sorted(set(fit_models))
    version = universe_version(state, fit_models, n_grid, d_u, filter_k, dedup, row_norm)
    keeps = _keep_masks(state, fit_models, n_grid, filter_k)
    path = None if state.state_dir is None else state.state_dir / "universes" / version / "universe.pt"
    if path is not None and path.exists():
        stored = torch.load(path, map_location="cpu")
        uni = Universe(
            version,
            stored["fit_models"],
            stored["n_grid"],
            stored["d_u"],
            [g.to(torch.float64) for g in stored["g"]],
            {m: list(a) for m, a in stored["maps"].items()},
            stored["info"],
        )
    else:
        per_j = [_processed_frames(state, fit_models, n_grid, j, keeps[j], filter_k, dedup, row_norm) for j in range(n_grid)]
        frames = {m: [per_j[j][m] for j in range(n_grid)] for m in fit_models}
        gs, info = [], []
        maps: dict[str, list[Tensor]] = {m: [] for m in fit_models}
        resolved_d_u = None
        for j in range(n_grid):
            g, a, inf = gpa([frames[m][j] for m in fit_models], d_u=d_u)
            gs.append(g)
            for m, a_m in zip(fit_models, a, strict=True):
                maps[m].append(a_m)
            per_model = dict(zip(fit_models, inf.pop("gpa_residual_per_model"), strict=True))
            info.append({**inf, "gpa_residual_per_model": per_model})
            resolved_d_u = inf["gpa_d_u"]
        uni = Universe(version, fit_models, n_grid, int(resolved_d_u), gs, maps, info)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "fit_models": fit_models,
                    "n_grid": n_grid,
                    "d_u": uni.d_u,
                    "g": gs,
                    "maps": maps,
                    "info": info,
                },
                path,
            )
    for m in add_models:
        if m in uni.maps:
            continue
        uni.maps[m] = [
            procrustes_to_universe(
                _processed_frames(state, [m], n_grid, j, keeps[j], filter_k, dedup, row_norm)[m], uni.g[j]
            )
            for j in range(n_grid)
        ]
        uni.added.append(m)
    return uni
