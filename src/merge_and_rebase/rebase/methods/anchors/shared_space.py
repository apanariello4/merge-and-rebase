"""Shared adaptation in the common GPA space (docs/.local/AGENT_SHARED_TRAINING.md sections 1.2-1.7).

One operator ``W(s)`` (``[r, r]``) and one bias ``b(s)`` (``[r]``) per relative depth ``s``, shared by every model.
In model ``m`` they act after block ``pi_m(s)`` on every token row ``h``::

    u   = ((h - mu_m) / c_m) Phi_m^T          # encode, [.., r]
    du  = u W + b                             # move in the common space
    h  <- h + c_m du Phi_m                    # decode and add

``Phi_m = A_m[:r]`` (``[r, d_m]``): the top-``r`` rows of the GPA map of the universe (``universe.py``), so the frame
is the one of the transport runs (norm filter ``k``: ``mu_m``, ``c_m`` of the filtered registration statistics, as
``method.filtered_frame``). ``W = 0, b = 0`` is the base model.

The depth grid is the universe's: grid point ``j`` of ``n_grid`` is block ``pi_m(j) = round(j (L_m - 1) / (n_grid -
1))`` of model ``m`` (``universe._block``). ``depths`` selects the grid points the adapter acts at ('last4', 'all', or
explicit indices).

The map is per token row and only touches the last dimension, so it is the same in batch-first and sequence-first
layouts and for any number of tokens (ViT-B/16 197, ViT-L/14 257): no token resampling is needed to apply ``W``.
Canonical tokens (``to_canonical``) are only needed when per-position objects (the universal Jacobian) are compared
across models.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import AnchorsConfig
from .state import AnchorState
from .universe import Universe, _block, fit_universe

Tensor = torch.Tensor

#: Frame of the main anchors state (results/anchors_state/tiny_imagenet_A16_T2048, AGENT_SHARED_SPACE S0).
MAIN_FRAME = dict(
    anchor_pool="tiny_imagenet",
    num_anchors=2048,
    anchor_batches=16,
    registration_batches=10,
    pool_seed=0,
    anchor_seed=0,
    canonical_tokens=197,
)


def resolve_depths(depths: str | Sequence[int], n_grid: int) -> list[int]:
    """Grid indices of ``S``: 'last4' (the last four grid points), 'all', or explicit indices."""
    if isinstance(depths, str):
        if depths == "all":
            return list(range(n_grid))
        if depths.startswith("last"):
            k = int(depths[4:])
            if not 1 <= k <= n_grid:
                raise ValueError(f"depths={depths!r} with n_grid={n_grid}")
            return list(range(n_grid - k, n_grid))
        raise ValueError(f"unknown depths {depths!r} (expected 'all', 'lastK' or a list of grid indices)")
    out = sorted({int(j) for j in depths})
    if not out or out[0] < 0 or out[-1] >= n_grid:
        raise ValueError(f"grid indices {depths} outside [0, {n_grid})")
    return out


@dataclass
class ModelFrame:
    """The frame of one model at the grid points of ``S`` (float64, CPU)."""

    model_id: str
    grid: list[int]  # grid indices j (the s in S)
    blocks: list[int]  # pi_m(s_j)
    mu: list[Tensor]  # [d_m]
    c: list[float]
    phi: list[Tensor]  # [r, d_m]

    @property
    def width(self) -> int:
        return int(self.phi[0].shape[1])

    def index(self, j: int) -> int:
        return self.grid.index(int(j))

    def encode(self, h: Tensor, j: int) -> Tensor:
        """``u = ((h - mu) / c) Phi^T`` on the last dimension (dtype of ``h``)."""
        i = self.index(j)
        phi = self.phi[i].to(device=h.device, dtype=h.dtype)
        mu = self.mu[i].to(device=h.device, dtype=h.dtype)
        return ((h - mu) / self.c[i]) @ phi.T

    def encode_delta(self, e: Tensor, j: int) -> Tensor:
        """A stream displacement ``e`` in common coordinates: ``(e / c) Phi^T`` (no mean)."""
        i = self.index(j)
        return (e / self.c[i]) @ self.phi[i].to(device=e.device, dtype=e.dtype).T

    def decode(self, du: Tensor, j: int) -> Tensor:
        """``dh = c du Phi`` (a displacement: no mean)."""
        i = self.index(j)
        return self.c[i] * (du @ self.phi[i].to(device=du.device, dtype=du.dtype))

    def encode_grad(self, g_h: Tensor, j: int) -> Tensor:
        """Adjoint of ``decode``: a gradient w.r.t. ``h`` as a gradient w.r.t. the common coordinates ``du``."""
        i = self.index(j)
        return self.c[i] * (g_h @ self.phi[i].to(device=g_h.device, dtype=g_h.dtype).T)


class SharedFrame:
    """Frames of several models at the grid points ``S`` of one GPA universe, plus the centre ``G(s)``."""

    def __init__(self, universe: Universe, frames: dict[str, ModelFrame], g: dict[int, Tensor], rank: int, meta: dict):
        self.universe = universe
        self.frames = frames
        self.g = g  # grid index -> [T', r] (kept anchors, top-r axes)
        self.rank = int(rank)
        self.meta = meta

    @property
    def grid(self) -> list[int]:
        return next(iter(self.frames.values())).grid

    def __getitem__(self, model_id: str) -> ModelFrame:
        return self.frames[model_id]

    @classmethod
    def from_state(
        cls,
        state_dir: str | Path,
        *,
        fit_models: Sequence[str],
        add_models: Sequence[str] = (),
        n_grid: int,
        depths: str | Sequence[int] = "last4",
        rank: int = 512,
        filter_k: float | None = 3.0,
        frame: dict[str, Any] | None = None,
    ) -> SharedFrame:
        """Load (or fit and store) the universe of ``fit_models`` and add ``add_models`` by Procrustes to the fixed G.

        The state is only read, except that a universe absent from ``universes/`` is fitted and stored there (the
        same cache the transport runs use)."""
        state = AnchorState(AnchorsConfig(**(frame or MAIN_FRAME)), state_dir)
        models = list(dict.fromkeys([*fit_models, *add_models]))
        if filter_k is not None:
            for m in models:
                state.load_norm_filter(m, k=float(filter_k))
        uni = fit_universe(
            state, list(fit_models), n_grid=n_grid, add_models=[m for m in add_models if m not in fit_models],
            filter_k=filter_k,
        )
        grid = resolve_depths(depths, n_grid)
        r = min(int(rank), int(uni.d_u))
        frames: dict[str, ModelFrame] = {}
        for m in models:
            reg = state.model(m)
            blocks = [_block(state, m, n_grid, j) for j in grid]
            if filter_k is None:
                mus = [reg.mu[b].to(torch.float64) for b in blocks]
                cs = [float(reg.c[b]) for b in blocks]
            else:
                flt = reg.extras[f"norm_k{float(filter_k):g}"]
                mus = [flt["mu"][b].to(torch.float64) for b in blocks]
                cs = [float(flt["c"][b]) for b in blocks]
            phis = [uni.maps[m][j][:r].to(torch.float64) for j in grid]
            frames[m] = ModelFrame(m, grid, blocks, mus, cs, phis)
        g = {j: uni.g[j][:, :r].to(torch.float64) for j in grid}
        meta = {
            "state_dir": str(state_dir),
            "universe_version": uni.version,
            "fit_models": list(uni.fit_models),
            "added": list(uni.added),
            "n_grid": int(n_grid),
            "grid": grid,
            "rank": r,
            "d_u": int(uni.d_u),
            "filter_k": filter_k,
            "blocks": {m: f.blocks for m, f in frames.items()},
            "gpa_residual": {
                j: uni.info[j].get("gpa_residual_per_model") for j in grid
            },
        }
        return cls(uni, frames, g, r, meta)


class SharedAdapter(nn.Module):
    """``W(s)`` and ``b(s)`` at the grid points ``S``. Full ``W`` by default; ``rank`` gives ``W = L R^T``;
    ``subspace`` (``[r, k]``, orthonormal columns ``W_J``) gives ``W = W_J W~ W_J^T`` with ``W~`` of size ``k x k``
    (the ``functional_subspace`` option)."""

    def __init__(
        self,
        grid: Sequence[int],
        dim: int,
        *,
        rank: int | None = None,
        subspace: dict[int, Tensor] | None = None,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.grid = [int(j) for j in grid]
        self.dim = int(dim)
        self.rank = None if rank is None else int(rank)
        self.has_bias = bool(bias)
        if self.rank is not None and subspace is not None:
            raise ValueError("rank and subspace are exclusive")
        self.w = nn.ParameterDict()
        self.left = nn.ParameterDict()
        self.right = nn.ParameterDict()
        self.b = nn.ParameterDict()
        self._subspace: dict[str, Tensor] = {}
        for j in self.grid:
            key = str(j)
            if subspace is not None:
                basis = subspace[j].to(dtype)
                self._subspace[key] = basis
                k = int(basis.shape[1])
                self.w[key] = nn.Parameter(torch.zeros(k, k, dtype=dtype))
            elif self.rank is None:
                self.w[key] = nn.Parameter(torch.zeros(self.dim, self.dim, dtype=dtype))
            else:
                # L zero, R small random: W = 0 at init with a non-zero gradient for L.
                self.left[key] = nn.Parameter(torch.zeros(self.dim, self.rank, dtype=dtype))
                self.right[key] = nn.Parameter(torch.randn(self.dim, self.rank, dtype=dtype) / self.dim**0.5)
            if self.has_bias:
                self.b[key] = nn.Parameter(torch.zeros(self.dim, dtype=dtype))

    def weight(self, j: int) -> Tensor:
        key = str(j)
        if key in self._subspace:
            basis = self._subspace[key].to(self.w[key].device)
            return basis @ self.w[key] @ basis.T
        if self.rank is None:
            return self.w[key]
        return self.left[key] @ self.right[key].T

    def bias(self, j: int) -> Tensor | None:
        return self.b[str(j)] if self.has_bias else None

    def displacement(self, u: Tensor, j: int) -> Tensor:
        """``du = u W + 1 b^T``."""
        w = self.weight(j).to(device=u.device, dtype=u.dtype)
        du = u @ w
        b = self.bias(j)
        if b is not None:
            du = du + b.to(device=u.device, dtype=u.dtype)
        return du

    def num_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    @torch.no_grad()
    def set_(self, weights: dict[int, Tensor], biases: dict[int, Tensor] | None = None) -> SharedAdapter:
        """Load full ``W`` (and ``b``) per grid point (only for the full parametrisation)."""
        if self.rank is not None or self._subspace:
            raise ValueError("set_ needs the full parametrisation")
        for j in self.grid:
            self.w[str(j)].copy_(weights[j].to(self.w[str(j)].dtype))
            if biases is not None and self.has_bias:
                self.b[str(j)].copy_(biases[j].to(self.b[str(j)].dtype))
        return self

    def state(self) -> dict[str, Any]:
        """Effective ``W`` and ``b`` per grid point (float64, CPU): what is saved and fused."""
        return {
            "grid": list(self.grid),
            "w": {j: self.weight(j).detach().double().cpu() for j in self.grid},
            "b": {j: (self.bias(j).detach().double().cpu() if self.has_bias else torch.zeros(self.dim, dtype=torch.float64))
                  for j in self.grid},
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any], dtype: torch.dtype = torch.float32) -> SharedAdapter:
        grid = [int(j) for j in payload["grid"]]
        dim = int(next(iter(payload["w"].values())).shape[0])
        ad = cls(grid, dim, dtype=dtype)
        ad.set_({int(j): w for j, w in payload["w"].items()}, {int(j): b for j, b in payload["b"].items()})
        return ad


def resblocks(model) -> nn.ModuleList:
    visual = model.visual if hasattr(model, "visual") else model
    return visual.transformer.resblocks


def _block_output(out):
    return out[0] if isinstance(out, tuple) else out


def _with_block_output(out, new):
    return (new, *out[1:]) if isinstance(out, tuple) else new


@contextlib.contextmanager
def adapter_hooks(model, frame: ModelFrame, adapter: SharedAdapter, *, scale: float = 1.0) -> Iterator[None]:
    """Forward hooks after the blocks ``pi_m(s)``: ``h <- h + scale c (u W + 1 b^T) Phi`` with ``u`` encoded from the
    CURRENT stream (earlier adapter steps included). Removed on exit."""
    blocks = resblocks(model)
    handles = []

    def make(j: int):
        def hook(_module, _args, out):
            h = _block_output(out)
            u = frame.encode(h.float() if h.dtype in (torch.float16, torch.bfloat16) else h, j)
            dh = frame.decode(adapter.displacement(u, j), j)
            return _with_block_output(out, h + (scale * dh).to(h.dtype))

        return hook

    try:
        for j, blk in zip(frame.grid, frame.blocks, strict=True):
            if j in adapter.grid:
                handles.append(blocks[blk].register_forward_hook(make(j)))
        yield
    finally:
        for hd in handles:
            hd.remove()


def anchor_conservation(adapter: SharedAdapter, g: dict[int, Tensor]) -> Tensor:
    """``R_P = sum_s (1/T) || G(s) W(s) + 1 b(s)^T ||^2`` (frame gpa: the same for every model)."""
    total = None
    for j in adapter.grid:
        gj = g[j].to(device=adapter.weight(j).device, dtype=adapter.weight(j).dtype)
        term = adapter.displacement(gj, j).pow(2).sum() / gj.shape[0]
        total = term if total is None else total + term
    return total
