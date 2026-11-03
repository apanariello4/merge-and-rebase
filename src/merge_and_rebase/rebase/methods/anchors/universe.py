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
from .core import consensus_frame, consensus_kernel, consensus_register, kernel_stats_to_consensus
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
    # decode maps ([d_U, d_m]) where they differ from the encode maps (consensus 'correct' / 'zca_rot'); None = maps
    dec: dict[str, list[Tensor]] | None = None

    def block_maps(self, j: int, source_id: str, target_id: str) -> tuple[Tensor, Tensor]:
        """``(encode map of the source, decode map of the target)``: a row h goes to ``h A_m^T``, ``u`` back to
        ``u B_k``; ``B_k = A_k`` for GPA and for the isometric consensus registrations."""
        dec = self.maps if self.dec is None else self.dec
        return self.maps[source_id][j], dec[target_id][j]


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


def consensus_version(state: AnchorState, fit_models: Sequence[str], n_grid: int, options: dict[str, Any],
                      filter_k: float | None = None, dedup: tuple | None = None, row_norm: bool = False) -> str:
    payload = {
        "frame": state.config.frame(),
        "models": {m: state.model_meta_hash(m) for m in sorted(fit_models)},
        "n_grid": int(n_grid),
        "consensus": dict(sorted(options.items())),
    }
    if filter_k is not None:
        payload["norm_filter_k"] = float(filter_k)
    if dedup is not None:
        payload["dedup"] = [dedup[0], float(dedup[1]), dedup[2], int(dedup[3])]
    if row_norm:
        payload["row_norm"] = True
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def fit_consensus(
    state: AnchorState,
    fit_models: Sequence[str],
    *,
    n_grid: int,
    k: int,
    construction: str = "mean",
    register: str = "rot",
    eps: float = 1e-2,
    add_models: Sequence[str] = (),
    filter_k: float | None = None,
    dedup: tuple | None = None,
    row_norm: bool = False,
) -> Universe:
    """Consensus-kernel universe (AGENT_ADDENDUM_CONSENSUS section 2), in the shape of the GPA ``Universe``.

    Per grid point: ``K*`` of the fitted models (``core.consensus_kernel``), ``g[j] = Z* = V_k Lambda_k^(1/2)``,
    encode maps ``maps[m][j]`` and decode maps ``dec[m][j]`` (``[k, d_m]``, ``core.consensus_register``). Stored and
    FROZEN under ``<state_dir>/universes/<version>/``: ``universe.pt``, ``k_star.pt`` (K* per grid point, float32)
    and ``consensus_stats.json`` (per model and grid point: 1 - CKA to K*, mutual 10-NN with K*, the map residual and
    condition number; K* spectrum). ``add_models`` are registered against the stored ``Z*``: it does not change.
    """
    fit_models = sorted(set(fit_models))
    options = {"construction": construction, "k": int(k), "register": register, "eps": float(eps)}
    version = consensus_version(state, fit_models, n_grid, options, filter_k, dedup, row_norm)
    keeps = _keep_masks(state, fit_models, n_grid, filter_k)
    root = None if state.state_dir is None else state.state_dir / "universes" / version
    path = None if root is None else root / "universe.pt"
    isometric = register != "correct" and register != "zca_rot"

    def frames_of(models, j):
        return _processed_frames(state, list(models), n_grid, j, keeps[j], filter_k, dedup, row_norm)

    if path is not None and path.exists():
        stored = torch.load(path, map_location="cpu")
        uni = Universe(version, stored["fit_models"], stored["n_grid"], stored["d_u"],
                       [g.to(torch.float64) for g in stored["g"]], {m: list(a) for m, a in stored["maps"].items()},
                       stored["info"], dec=None if stored.get("dec") is None
                       else {m: list(b) for m, b in stored["dec"].items()})
        stats = json.loads((root / "consensus_stats.json").read_text())
        spectra = None
    else:
        gs, info, k_stars, spectra = [], [], [], []
        maps: dict[str, list[Tensor]] = {m: [] for m in fit_models}
        dec: dict[str, list[Tensor]] = {m: [] for m in fit_models}
        stats = {"options": options, "fit_models": fit_models, "per_grid": []}
        d_u = None
        for j in range(n_grid):
            frames = frames_of(fit_models, j)
            k_star, k_info = consensus_kernel([frames[m] for m in fit_models], construction=construction)
            z, lam = consensus_frame(k_star, k)
            d_u = int(z.shape[1])
            gs.append(z)
            k_stars.append(k_star.to(torch.float32))
            per_model = {}
            for m in fit_models:
                a, b, r_info = consensus_register(frames[m], z, lam, register=register, eps=eps)
                maps[m].append(a)
                dec[m].append(b)
                per_model[m] = {**r_info, **kernel_stats_to_consensus(frames[m], k_star)}
            info.append({"consensus": {**options, **k_info, "d_u": d_u},
                         "gpa_residual_per_model": {m: v["map_residual"] for m, v in per_model.items()}})
            stats["per_grid"].append({"j": j, "spectrum_top64": [float(x) for x in lam[:64]],
                                      "k_star_effective_rank": k_info["effective_rank"], "models": per_model})
        uni = Universe(version, fit_models, n_grid, int(d_u), gs, maps, info, dec=None if isometric else dec)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            torch.save({"fit_models": fit_models, "n_grid": n_grid, "d_u": uni.d_u, "g": gs, "maps": maps,
                        "dec": uni.dec, "info": info, "consensus": options}, path)
            torch.save(k_stars, root / "k_star.pt")
            (root / "consensus_stats.json").write_text(json.dumps(stats, indent=1))
    k_star_cache = None
    for m in add_models:
        if m in uni.maps:
            continue
        if k_star_cache is None:
            k_star_cache = torch.load(root / "k_star.pt") if root is not None and (root / "k_star.pt").exists() else None
        enc_m, dec_m = [], []
        for j in range(n_grid):
            frame = frames_of([m], j)[m]
            z = uni.g[j]
            lam = (z * z).sum(0)  # Z*^T Z* = Lambda_k (orthogonal columns)
            a, b, r_info = consensus_register(frame, z, lam, register=register, eps=eps)
            enc_m.append(a)
            dec_m.append(b)
            if k_star_cache is not None:
                r_info.update(kernel_stats_to_consensus(frame, k_star_cache[j].to(torch.float64)))
            stats["per_grid"][j]["models"][m] = {**r_info, "added": True}
        uni.maps[m] = enc_m
        if uni.dec is not None:
            uni.dec[m] = dec_m
        uni.added.append(m)
    if add_models and root is not None:
        (root / "consensus_stats.json").write_text(json.dumps(stats, indent=1))
    return uni
