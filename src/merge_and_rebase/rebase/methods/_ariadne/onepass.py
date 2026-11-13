"""Opt-in one-pass Ariadne fit (``DirectResidualConfig.onepass``), numerically equivalent to the streaming fit.

The streaming fit (`prepare_direct_residual_streaming` + `fit_direct_residual_streaming`) sweeps the calibration data
twice (Pass A: Procrustes cross-covariance ``C_j``; Pass B: regression statistics) plus an untimed diagnostics sweep.
For the plain configuration it supports here (``depth_fit='independent'``, ``block_split='none'``,
``components=('mlp.c_proj',)``, polar activation Procrustes, ``residual_target='transported_delta'``,
``component_target='block_boundary'``) the second sweep is not needed:

* every block is fitted against the UNMODIFIED target base, so the pre-fit effect is exactly zero and the regression
  target of block ``j`` is ``E_j = D_j = (f_i - b_i) Q_j`` (``i = pairing[j]``; ``b``/``f`` the source base / source FT
  boundary resized to the target token grid);
* with ``dS = f_i - b_i`` and the c_proj input rows ``a`` of block ``j``, the sufficient statistics of the solve are
  ``S = sum a^T a``, ``B = (sum a^T dS) Q_j L`` (``L = effective_out^T``), ``sum_a``, ``sum_e = (sum dS) Q_j`` and ``N``;
  ``Q_j`` is linear in the statistics, so it is applied once at the end, after the polar factor of the centred
  cross-covariance ``C_j`` (accumulated by the same Chan accumulator as Pass A) is known.

So ONE sweep (source base, source FT, target base; three forwards per batch) accumulates ``S, P = sum a^T dS, sum_a,
sum_dS, N`` and ``C_j`` per target block, and the existing empirical-Bayes eigendecomposition solve finishes each block.
No target activation is stored. The analysis-only quantities (residual norms, the ridge-free reference solve,
``matrix_rank``, the alignment diagnostics sweep, the fingerprints of the target boundary) are not computed; the solve
runs with ``diagnostics=False`` and the corresponding row fields are ``None``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from typing import Any

import torch

from ....utils.cost_accounting import cost_phase
from ...discrete_layer_match import DiscreteLayerPairing
from .._shared import _to_tokens
from .alignment import _procrustes_from_cross, procrustes_rank_diagnostics
from .capture import _register_capture_hooks, iter_capture_tokens, paired_calibration
from .config import DirectResidualConfig, annotate_target_split
from .fit import ResidualSufficientStatistics, _finalize_independent_component
from .layouts import COMPONENT_INPUT_KIND, _aligned, _component_effective_out, _layout_for
from .streaming import _StreamingCrossCovariance

Tensor = torch.Tensor
logger = logging.getLogger(__name__)

COMPONENT = "mlp.c_proj"
#: Number of column blocks of the upper-triangular Gram update (private; 1 = plain ``a.T @ a``). Cost (k+1)/(2k) of
#: the full product. The result does not depend on it beyond rounding.
_GRAM_BLOCKS = 4

#: Accumulate on rows shifted by the first batch's column means (see `_BlockStats`). Private; False only for the ablation
#: that shows the float32 cancellation the shift removes.
_SHIFT_BY_FIRST_BATCH = True

_DEFAULT_FINGERPRINT_TOL = 1e-9  # DirectResidualConfig.streaming_fingerprint_tol default

_STATS_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def _block_bounds(d: int, k: int) -> list[tuple[int, int]]:
    """Split ``range(d)`` into ``min(k, d)`` contiguous, near-equal column blocks (uneven when ``k`` does not divide ``d``)."""
    k = max(1, min(int(k), int(d)))
    edges = [(d * i) // k for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k)]


def gram_upper_update_(s: Tensor, a: Tensor, k: int = _GRAM_BLOCKS) -> None:
    """``s[i, j] += a[:, i].T @ a[:, j]`` for the column blocks ``i <= j`` only (in place; ``s`` is ``[d, d]``).

    The strictly lower blocks are left untouched: call `gram_mirror_upper_` once after the last update.
    """
    bounds = _block_bounds(a.shape[1], k)
    for bi, (i0, i1) in enumerate(bounds):
        left = a[:, i0:i1].T
        for j0, j1 in bounds[bi:]:
            s[i0:i1, j0:j1].addmm_(left, a[:, j0:j1])


def gram_mirror_upper_(s: Tensor, k: int = _GRAM_BLOCKS) -> Tensor:
    """Copy the upper block triangle written by `gram_upper_update_` onto the lower one (in place) and return ``s``."""
    bounds = _block_bounds(s.shape[0], k)
    for bi, (i0, i1) in enumerate(bounds):
        for j0, j1 in bounds[bi + 1 :]:
            s[j0:j1, i0:i1] = s[i0:i1, j0:j1].T
    return s


def gram_blocked(a: Tensor, k: int = _GRAM_BLOCKS) -> Tensor:
    """``a.T @ a`` through the blocked upper-triangular update (reference helper for tests)."""
    s = torch.zeros(a.shape[1], a.shape[1], dtype=a.dtype, device=a.device)
    gram_upper_update_(s, a, k)
    return gram_mirror_upper_(s, k)


@contextlib.contextmanager
def _strict_fp32_matmul():
    """TF32 off for the enclosed matmuls (restored on exit); a no-op on CPU-only builds."""
    prev_flag = torch.backends.cuda.matmul.allow_tf32
    prev_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_flag
        torch.set_float32_matmul_precision(prev_precision)


class _BlockStats:
    """Per-target-block accumulators of the one-pass fit (``S`` upper triangle only until `finalize`).

    The moments are accumulated on SHIFTED rows ``a - c_a`` and ``dS - c_e``, with ``c_a`` / ``c_e`` the column means of
    the first batch (rounded to the stats dtype and kept exactly). The solve centres its statistics, so the shift only
    improves conditioning: without it the float32 Gram of a channel with mean >> std loses the variance to cancellation
    (``S - Σa Σa^T / N``). `raw` undoes the shift in float64 and returns exactly the raw moments ``solve`` expects.
    """

    def __init__(self, m: int, d_s: int, dtype: torch.dtype, device) -> None:
        self.s = torch.zeros(m, m, dtype=dtype, device=device)
        self.p = torch.zeros(m, d_s, dtype=dtype, device=device)
        self.sum_a = torch.zeros(m, dtype=torch.float64, device=device)  # of the shifted rows
        self.sum_ds = torch.zeros(d_s, dtype=torch.float64, device=device)
        self.c_a: Tensor | None = None
        self.c_e: Tensor | None = None
        self.n = 0

    def update(self, a: Tensor, ds: Tensor, *, shift: bool = True) -> None:
        dtype = self.s.dtype
        a = a.to(dtype)
        ds = ds.to(dtype)
        if self.c_a is None:
            self.c_a = a.mean(dim=0) if shift else torch.zeros_like(a[0])
            self.c_e = ds.mean(dim=0) if shift else torch.zeros_like(ds[0])
        a = a - self.c_a
        ds = ds - self.c_e
        gram_upper_update_(self.s, a)
        self.p.addmm_(a.T, ds)
        self.sum_a += a.sum(dim=0, dtype=torch.float64)
        self.sum_ds += ds.sum(dim=0, dtype=torch.float64)
        self.n += int(a.shape[0])

    def raw(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(S, P, sum_a, sum_dS)`` of the UNSHIFTED rows, float64 (``S`` mirrored to the full matrix)."""
        if not (torch.isfinite(self.sum_a).all() and torch.isfinite(self.sum_ds).all()):
            raise ValueError("solver inputs must be finite")
        n = float(self.n)
        c, ce = self.c_a.double(), self.c_e.double()
        sa, se = self.sum_a, self.sum_ds
        s = gram_mirror_upper_(self.s).double()
        p = self.p.double()
        s = s + torch.outer(c, sa) + torch.outer(sa, c) + n * torch.outer(c, c)
        p = p + torch.outer(c, se) + torch.outer(sa, ce) + n * torch.outer(c, ce)
        return s, p, sa + n * c, se + n * ce


@torch.no_grad()
def fit_onepass(
    target_model,
    target_base_state: Mapping[str, Tensor],
    source_base_model,
    source_ft_model,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """One-sweep equivalent of Pass A + Pass B of the streaming fit (see the module docstring).

    Returns ``(target_corrections, diagnostics_rows, calibration_metadata, info)``. ``target_corrections`` and the rows have
    the format of `fit_direct_residual_streaming` (rows additionally carry the Procrustes rank diagnostics Pass A would
    have merged in; analysis-only residual fields are ``None``). ``info`` records ``onepass_stats_dtype`` and the TF32
    flags. ``target_model`` must hold the native target base weights (not verified here: there is no Pass-A fingerprint).
    """
    if not config.onepass:
        raise ValueError("fit_onepass requires config.onepass=True")
    if pairing.target_depth < 1 or pairing.source_depth < 1:
        raise ValueError("pairing depths must be positive")
    stats_dtype = _STATS_DTYPES[config.onepass_stats_dtype]
    ignored: dict[str, Any] = {}
    if config.streaming_position_chunk is not None:
        ignored["streaming_position_chunk"] = config.streaming_position_chunk
    if config.streaming_fingerprint_tol != _DEFAULT_FINGERPRINT_TOL:
        ignored["streaming_fingerprint_tol"] = config.streaming_fingerprint_tol
    if ignored:
        logger.warning("onepass ignores %s (one sweep over all blocks, no target fingerprint check)", ignored)
    shim = _layout_for(None)
    input_kind = COMPONENT_INPUT_KIND[COMPONENT]
    depth = pairing.target_depth

    source_batches, target_batches, metadata = paired_calibration(
        source_loader, target_loader, num_batches=config.num_batches, seed=config.seed, family_adapter=None
    )
    distinct_source = sorted(set(pairing.pairing))
    src_requests = {str(i): (i, "boundary") for i in distinct_source}
    tgt_requests: dict[str, tuple[int, str]] = {}
    for j in range(depth):
        tgt_requests[f"{j}.out"] = (j, "boundary")
        tgt_requests[f"{j}.h"] = (j, input_kind)

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}
    ambient_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)

    chan = {j: _StreamingCrossCovariance(device=device) for j in range(depth)}
    stats: dict[int, _BlockStats] = {}
    state: dict[str, Any] = {"cache": {}, "fired": {}}

    def pair_for(i: int, tokens: Tensor, sb: Mapping[str, Tensor], sf: Mapping[str, Tensor]):
        """Source base ``b`` and ``dS = f - b`` of source block ``i`` resized to ``tokens``' grid (cached per batch)."""
        cache = state["cache"]
        if i not in cache:
            b = _aligned([sb[str(i)]], [tokens])[0]
            f = _aligned([sf[str(i)]], [tokens])[0]
            cache[i] = (b, f - b)
        return cache[i]

    corrections: dict[str, Tensor] = {}
    rows: list[dict[str, Any]] = []
    try:
        for m in unique_models.values():
            m.to(device).eval()
        blocks = shim.blocks(target_model)

        sb_gen = iter_capture_tokens(
            source_base_model, source_batches, src_requests, device, family_adapter=None, store_device=device
        )
        sf_gen = iter_capture_tokens(
            source_ft_model, source_batches, src_requests, device, family_adapter=None, store_device=device
        )
        with contextlib.closing(sb_gen), contextlib.closing(sf_gen):
            for k, (sb, sf) in enumerate(zip(sb_gen, sf_gen, strict=True)):
                batch = target_batches[k]
                batch_size = shim.batch_size(batch)
                state["cache"] = {}
                state["fired"] = {}

                def store(name, tensor, *, _batch_size=batch_size, _sb=sb, _sf=sf):
                    if isinstance(tensor, tuple):
                        tensor = tensor[0]
                    tokens = _to_tokens(tensor.detach(), batch_size=_batch_size).float().to(device)
                    state["fired"][name] = state["fired"].get(name, 0) + 1
                    j_str, kind = name.split(".")
                    j = int(j_str)
                    b, ds = pair_for(pairing.pairing[j], tokens, _sb, _sf)
                    if kind == "out":
                        chan[j].update(
                            b.reshape(-1, b.shape[-1]).double(), tokens.reshape(-1, tokens.shape[-1]).double()
                        )
                        return
                    a = tokens.reshape(-1, tokens.shape[-1])
                    d = ds.reshape(-1, ds.shape[-1])
                    if j not in stats:
                        stats[j] = _BlockStats(a.shape[1], d.shape[1], stats_dtype, device)
                    if stats_dtype == torch.float32:
                        with _strict_fp32_matmul():
                            stats[j].update(a, d, shift=_SHIFT_BY_FIRST_BATCH)
                    else:
                        stats[j].update(a, d, shift=_SHIFT_BY_FIRST_BATCH)

                handles = _register_capture_hooks(shim, blocks, tgt_requests, store, family_adapter=None)
                try:
                    with cost_phase("activation_collection"):
                        shim.forward(target_model, batch, device)
                finally:
                    for handle in handles:
                        handle.remove()
                if len(state["fired"]) != len(tgt_requests) or any(v != 1 for v in state["fired"].values()):
                    raise RuntimeError("A requested activation hook did not fire exactly once per batch")
        state["cache"] = {}

        with cost_phase("transformation"):
            for j in range(depth):
                st = stats.pop(j)
                key = shim.component_key(j, COMPONENT, prefixed=True)
                width = int(target_base_state[key].shape[0])
                t_out = _component_effective_out(shim, target_model, j, COMPONENT, width).to(device)
                cross = chan[j].cross()
                q, rank = _procrustes_from_cross(cross, return_rank=True)
                lmat = t_out.to(torch.float64).T
                # Same statistics the streaming Pass B accumulates, with Q_j applied once to the raw moments.
                sufficient = ResidualSufficientStatistics()
                raw_s, raw_p, raw_sum_a, raw_sum_ds = st.raw()
                sufficient.s = raw_s
                sufficient.g = lmat.T @ lmat
                sufficient.b = (raw_p @ q) @ lmat
                sufficient.sum_a = raw_sum_a
                sufficient.sum_e = raw_sum_ds @ q
                sufficient.sum_e2 = float("nan")  # analysis-only (residual norms); never read with diagnostics=False
                sufficient.n_rows = st.n
                sufficient.m_source, sufficient.d_source = st.s.shape[0], lmat.shape[1]
                sufficient._t_in = None
                sufficient._t_out = t_out.detach().clone()
                del st
                position_corrections: dict[str, Tensor] = {}
                block_rows: list[dict[str, Any]] = []
                _finalize_independent_component(
                    sufficient,
                    0.0,
                    0.0,
                    j,
                    COMPONENT,
                    key,
                    _state_slice(target_base_state, key),
                    t_out,
                    config,
                    {j: float(pairing.pairing[j])},
                    position_corrections,
                    block_rows,
                    h_batches=None,
                    diagnostics=False,
                )
                corrections.update(position_corrections)
                for row in block_rows:
                    row["source_position"] = row.pop("source_coordinate")
                    row["procrustes_source"] = "activation"
                    row.update(procrustes_rank_diagnostics(rank, q.shape[0], q.shape[1], polar_derived=True))
                    rows.append(row)
    finally:
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    annotate_target_split(rows, config, pairing)
    info = {
        "onepass_stats_dtype": config.onepass_stats_dtype,
        "onepass_tf32_stats_matmul": False if stats_dtype == torch.float32 else None,
        "onepass_ambient_allow_tf32": ambient_tf32,
        "onepass_gram_blocks": _GRAM_BLOCKS,
        "onepass_shift_by_first_batch": _SHIFT_BY_FIRST_BATCH,
        "onepass_ignored_options": ignored,
    }
    return corrections, rows, metadata, info


def _state_slice(state: Mapping[str, Tensor], key: str) -> dict[str, Tensor]:
    """The weight (and bias, when present) entries of one projection, as `_finalize_independent_component` mutates them."""
    bias_key = f"{key[: -len('.weight')]}.bias"
    return {k: state[k] for k in (key, bias_key) if k in state}
