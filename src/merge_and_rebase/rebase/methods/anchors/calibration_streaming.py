"""Streaming calibration for ``anchors`` on decoders (``calibration_streaming``): host RAM independent of the token count.

The resident path stores, for the whole calibration draw, the boundary activations of the three models (float32), the
decoded effects ``D`` of every block and the ridge inputs ``h`` and outputs of every block, and only then solves. All the
ridge needs from them are the sufficient statistics of ``ResidualSufficientStatistics`` (``X^T X``, ``X^T Y``, the row
sums, ``sum Y^2``, the row count; float64), so here the draw is swept batch by batch:

* per batch: source base / ft boundaries -> ``E = S_ft - S_base`` -> ``D = E Phi`` (the same map ``Phi`` as the resident
  decode) -> ``stats.update(h, D)`` for every (block, component) of the current group of blocks;
* the group is ``calibration_streaming_chunk`` blocks (the statistics of an MLP ``down_proj`` are ``[d_mlp, d_mlp]``
  float64: the group size bounds that constant part of the memory; the cost is one more sweep per group);
* ``map_rows='task'``: the frame (``mu``, ``c``, ``P^T P``, the cross-Gram of every paired block) is itself accumulated in
  one extra sweep from raw moments (``P = (S - mu) / c``  =>  ``P^T P = (S^T S - n mu mu^T) / c^2``), so ``Phi`` exists
  before ``D`` is formed. The row subsample ``map_num_rows`` is reproduced exactly (the same seeded permutation, applied
  by global row offset).

Equivalence with the resident path: identical rows, identical ``Phi``, identical ``stats.solve`` / ``solve_hook``;
differences are float summation order (and, for ``map_rows='task'``, the resident frame is rounded to float32). The
resident pristine-effect check (a second forward of the unmodified target, effect = 0) is not repeated: the target is
mounted at its base weights and never modified during the sweeps, so ``effect_before_norm`` is 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ...discrete_layer_match import DiscreteLayerPairing
from .._ariadne.capture import ROW_MASK_KEY, iter_capture_tokens, paired_calibration
from .._ariadne.config import annotate_target_split, order_components, resolve_fit_blocks
from .._ariadne.fit import ResidualSufficientStatistics, _finalize_independent_component
from .._ariadne.layouts import COMPONENT_INPUT_KIND, _component_effective_out, _layout_for
from .core import CoverageAccumulator

Tensor = torch.Tensor


# ------------------------------------------------------------------------------------------- task-row frame statistics


@dataclass
class StreamedTaskFrame:
    """One side of the ``map_rows='task'`` frame as statistics (no rows): ``mu``, ``c`` and ``P^T P`` per block.

    ``mu`` / ``c`` / ``gram`` are keyed by block index; ``c`` indexes like ``RegisteredModel.c`` (``frame.c[i]``).
    ``n_rows`` is the number of frame rows (after ``map_num_rows``).
    """

    model_id: str
    n_rows: int
    mu: dict[int, Tensor]
    c: dict[int, float]
    gram: dict[int, Tensor]

    @property
    def depth(self) -> int:
        return len(self.c)


class _FrameAccumulator:
    """Raw moments of the task rows of every needed block, in float64 (host)."""

    def __init__(self, source_blocks: list[int], target_blocks: list[int], pairs: list[tuple[int, int]]) -> None:
        self.n = 0
        self.sum: dict[tuple[str, int], Tensor] = {}
        self.sq: dict[tuple[str, int], Tensor] = {}
        self.cross: dict[tuple[int, int], Tensor] = {}
        self.source_blocks, self.target_blocks, self.pairs = source_blocks, target_blocks, pairs

    def update(self, src: Mapping[str, Tensor], tgt: Mapping[str, Tensor], keep: Tensor | None, device) -> None:
        rows_s: dict[int, Tensor] = {}
        rows_t: dict[int, Tensor] = {}
        for side, blocks, banks, out in (("s", self.source_blocks, src, rows_s), ("t", self.target_blocks, tgt, rows_t)):
            for b in blocks:
                x = banks[str(b)]
                x = x.reshape(-1, x.shape[-1])
                if keep is not None:
                    x = x[keep]
                x = x.to(device=device, dtype=torch.float64)
                out[b] = x
                key = (side, b)
                s, g = x.sum(0).cpu(), (x.T @ x).cpu()
                if key not in self.sum:
                    self.sum[key], self.sq[key] = s, g
                else:
                    self.sum[key] += s
                    self.sq[key] += g
        for i, j in self.pairs:
            c = (rows_s[i].T @ rows_t[j]).cpu()
            if (i, j) not in self.cross:
                self.cross[(i, j)] = c
            else:
                self.cross[(i, j)] += c
        self.n += int(next(iter(rows_s.values())).shape[0])

    def finalize(self) -> tuple[StreamedTaskFrame, StreamedTaskFrame, dict[tuple[int, int], Tensor]]:
        """``(source frame, target frame, cross-Gram P_m^T P_k per (i, j))`` of ``P = (S - mu) / c``."""
        if self.n == 0:
            raise ValueError("task frame: no rows")
        n = float(self.n)
        frames: dict[str, StreamedTaskFrame] = {}
        for side, name in (("s", "task_source"), ("t", "task_target")):
            mu, c, gram = {}, {}, {}
            for (sd, b), total in self.sum.items():
                if sd != side:
                    continue
                m = total / n
                g_raw = self.sq[(sd, b)]
                scale = float((torch.trace(g_raw) - n * (m @ m)).clamp_min(0.0).item() / n) ** 0.5
                centred = g_raw - n * torch.outer(m, m)
                g = centred / scale**2
                mu[b], c[b], gram[b] = m, scale, 0.5 * (g + g.T)
            frames[side] = StreamedTaskFrame(name, self.n, mu, c, gram)
        cross = {}
        for (i, j), raw in self.cross.items():
            mi, mj = frames["s"].mu[i], frames["t"].mu[j]
            cross[(i, j)] = (raw - n * torch.outer(mi, mj)) / (frames["s"].c[i] * frames["t"].c[j])
        return frames["s"], frames["t"], cross


def _count_rows(batches: list[Any]) -> int:
    return sum(int(b[ROW_MASK_KEY].sum()) for b in batches)


def _map_keep_mask(n_rows: int, config) -> Tensor | None:
    """The ``map_num_rows`` subsample of ``task_map_frames`` (decoders draw nothing before it): ``None`` keeps all."""
    if config.map_num_rows is None or config.map_num_rows >= n_rows:
        return None
    generator = torch.Generator().manual_seed(int(config.map_seed))
    idx = torch.randperm(n_rows, generator=generator)[: int(config.map_num_rows)]
    keep = torch.zeros(n_rows, dtype=torch.bool)
    keep[idx] = True
    return keep


@torch.no_grad()
def accumulate_task_frame(
    source_base_model, target_model, source_batches, target_batches, pairing: DiscreteLayerPairing, *,
    config, device, family_adapter,
) -> tuple[StreamedTaskFrame, StreamedTaskFrame, dict[tuple[int, int], Tensor]]:
    """One sweep over the map draw (base boundaries of both models, content rows) -> the frame statistics.

    The target must be mounted at its base weights by the caller (as ``task_map_frames`` requires)."""
    n_rows = _count_rows(source_batches)
    keep_all = _map_keep_mask(n_rows, config)
    src_blocks = sorted(set(int(i) for i in pairing.pairing))
    tgt_blocks = list(range(pairing.target_depth))
    pairs = [(int(pairing.pairing[j]), j) for j in range(pairing.target_depth)]
    acc = _FrameAccumulator(src_blocks, tgt_blocks, pairs)
    src_req = {str(i): (i, "boundary") for i in src_blocks}
    tgt_req = {str(j): (j, "boundary") for j in tgt_blocks}
    gens = (
        iter_capture_tokens(source_base_model, source_batches, src_req, device, family_adapter=family_adapter),
        iter_capture_tokens(target_model, target_batches, tgt_req, device, family_adapter=family_adapter),
    )
    offset = 0
    try:
        for src, tgt in zip(*gens, strict=True):
            n = int(src[str(src_blocks[0])].shape[1])
            keep = None if keep_all is None else keep_all[offset: offset + n]
            offset += n
            acc.update(src, tgt, keep, device)
    finally:
        for g in gens:
            g.close()
    return acc.finalize()


# ------------------------------------------------------------------------------------------- Gram-form decode + diagnostics


def gram_spectral_map(
    cross: Tensor, gram_k: Tensor, c_m: float, *, scale: float, mode: str, rank: int | None
) -> tuple[Tensor, dict[str, float]]:
    """``anchors.core.spectral_transport_map`` from ``C = P_m^T P_k`` and ``G_k = P_k^T P_k`` (same operations, same
    tolerances; ``Phi`` is ``[d_m, d_k]``, ``D = E Phi``)."""
    cross = cross.to(torch.float64)
    tol = torch.finfo(torch.float64).eps
    factor = float(scale) / float(c_m)
    if mode == "truncate_target":
        lam, v = torch.linalg.eigh(gram_k)
        lam, v = lam.flip(0), v.flip(1)
        valid = int((lam > lam[0] * lam.numel() * tol).sum().item())
        r = valid if rank is None else min(int(rank), valid)
        vr = v[:, :r]
        phi = factor * ((cross @ vr) / lam[:r]) @ vr.T
        kept = float(lam[:r].sum().item()) / float(lam.clamp_min(0.0).sum().item())
        return phi, {"spectral_rank_used": float(r), "spectral_kept_fraction": kept}
    u, sv, wt = torch.linalg.svd(cross, full_matrices=False)
    valid = int((sv > sv[0] * max(cross.shape) * tol).sum().item())
    r = valid if rank is None else min(int(rank), valid)
    kept = float((sv[:r] ** 2).sum().item()) / float((sv**2).sum().item())
    if mode == "truncate_cross":
        gbar = float(torch.trace(gram_k).item()) / int(gram_k.shape[1])
        phi = factor * (u[:, :r] * sv[:r]) @ wt[:r] / gbar
    else:
        phi = factor * u[:, :r] @ wt[:r]
    return phi, {"spectral_rank_used": float(r), "spectral_kept_fraction": kept}


def gram_pair_diagnostics(g_m: Tensor, g_k: Tensor, cross: Tensor, n_rows: int) -> dict[str, float]:
    """``method._pair_diagnostics(P_m, P_k)`` from the Grams (``kernel_diagnostics_from_frames`` and
    ``kernel_spectrum_from_frame``; the same quantities in the kernel form when ``T <= d``)."""
    aa, bb = float(torch.linalg.norm(g_m).item()), float(torch.linalg.norm(g_k).item())
    mm, kk, mk = aa**2, bb**2, float(torch.linalg.norm(cross).item()) ** 2
    cka = mk / (aa * bb) if aa * bb > 0.0 else float("nan")
    rel_err = max(mm + kk - 2.0 * mk, 0.0) ** 0.5 / mm**0.5
    ev = torch.linalg.eigvalsh(g_k).clamp_min(0.0)
    d = int(ev.numel())
    if n_rows > d:
        ev = torch.cat([torch.zeros(n_rows - d, dtype=ev.dtype, device=ev.device), ev])
    elif n_rows < d:
        ev = ev[d - n_rows:]  # eigvalsh ascending: the T x T kernel keeps the T largest
    top, low = float(ev.max().item()), float(ev.min().item())
    s1, s2 = float(ev.sum().item()), float((ev**2).sum().item())
    return {
        "anchors_kernel_cka": cka,
        "anchors_kernel_rel_err": rel_err,
        "anchors_target_kernel_condition_number": top / low if low > 0.0 else float("inf"),
        "anchors_target_kernel_effective_rank": (s1 * s1 / s2) if s2 > 0.0 else 0.0,
        "anchors_target_kernel_trace": s1,
        "anchors_target_kernel_numerical_rank": int(
            (ev > ev.max() * ev.numel() * torch.finfo(torch.float64).eps).sum().item()
        ),
    }


# ------------------------------------------------------------------------------------------- the fused sweep + solve


class _Decoders:
    """Per target block ``j``: ``(fn, eps, spectral info)``, the pair diagnostics and the source coverage, memoised.

    ``frames`` (the registered ``source`` / ``target``) given: the resident definitions (``method.transport_map`` etc.).
    ``task`` given (``(src_frame, tgt_frame, cross)``): the Gram forms above."""

    def __init__(self, source, target, config, pairing, gpa_maps, task=None) -> None:
        self.source, self.target, self.config, self.pairing, self.gpa_maps, self.task = (
            source, target, config, pairing, gpa_maps, task)
        self._fn: dict[int, tuple] = {}
        self._pair: dict[int, dict[str, float]] = {}
        self.coverage: dict[int, dict[str, float]] = {}

    def fn(self, j: int):
        if j not in self._fn:
            from . import method as m

            i = int(self.pairing.pairing[j])
            if self.task is None:
                self._fn[j] = m.transport_map(self.source, i, self.target, j, config=self.config, gpa_maps=self.gpa_maps)
            else:
                _, tgt_frame, cross = self.task
                cfg = self.config
                scale = self.target.c[j] if cfg.rescale_target else self.source.c[i]
                phi, info = gram_spectral_map(
                    cross[(i, j)], tgt_frame.gram[j], self.source.c[i], scale=scale, mode=cfg.spectral_filter,
                    rank=cfg.spectral_rank,
                )
                self._fn[j] = ((lambda e, _phi=phi: e.to(torch.float64) @ _phi.to(e.device)), None, info)
        return self._fn[j]

    def pair_diag(self, j: int) -> dict[str, float]:
        if j not in self._pair:
            from . import method as m

            i = int(self.pairing.pairing[j])
            if self.task is None:
                self._pair[j] = m._pair_diagnostics(self.source.p[i].to(torch.float64), self.target.p[j].to(torch.float64))
            else:
                src_frame, tgt_frame, cross = self.task
                self._pair[j] = gram_pair_diagnostics(src_frame.gram[i], tgt_frame.gram[j], cross[(i, j)], src_frame.n_rows)
        return self._pair[j]

    def source_coverage(self, i: int, acc: CoverageAccumulator) -> dict[str, float]:
        from . import method as m

        if self.task is None:
            return m._source_coverage(acc, self.source.p[i].to(torch.float64), self.config.eps_rel)
        src_frame = self.task[0]
        if src_frame.n_rows <= int(src_frame.gram[i].shape[0]):
            raise ValueError("calibration_streaming with map_rows='task' and coverage_diagnostics needs more frame rows "
                             "than the source width (the kernel form of the coverage needs the rows themselves); "
                             "set coverage_diagnostics: false")
        g = src_frame.gram[i]
        eps = float(self.config.eps_rel) * float(torch.trace(g).item()) / int(src_frame.n_rows)
        a = g + eps * torch.eye(g.shape[0], dtype=torch.float64)
        return acc.coverage_from_operator(torch.linalg.solve(a, g))


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[k: k + size] for k in range(0, len(items), size)]


@torch.no_grad()
def fit_streaming_independent(
    *,
    target_model,
    target_base_sd: Mapping[str, Tensor],
    source_base_model,
    source_ft_model,
    source_batches: list[Any],
    target_batches: list[Any],
    pairing: DiscreteLayerPairing,
    source,
    target,
    config,
    device,
    family_adapter,
    gpa_maps=None,
    solve_hook=None,
    task_stats=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Streaming equivalent of ``decode_desired_effects`` + ``fit_direct_residual`` (independent depth fit, decoders).

    Returns ``(corrections, rows, position_diag)`` with the same keys and fields as the resident path (``rows`` already
    carry ``source_position`` / ``procrustes_source`` / the target split annotation; the caller adds ``position_diag``).
    ``source`` / ``target`` are the registered frames (``task_stats=None``) or ``StreamedTaskFrame`` (``task_stats`` =
    ``(source frame, target frame, cross)``). The target is mounted at its base weights for the sweeps and restored.
    """
    from . import method as m

    solver_config = config.ridge_config()
    components = order_components(solver_config.components)
    shim = _layout_for(family_adapter)
    decoders = _Decoders(source, target, config, pairing, gpa_maps, task_stats)
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    corrections: dict[str, Tensor] = {}
    rows: list[dict[str, Any]] = []
    position_diag: dict[int, dict[str, Any]] = {}
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
        target_model.load_state_dict(current_state, strict=True)
        positions = resolve_fit_blocks(solver_config, pairing.target_depth)
        source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
        fitted: dict[int, tuple[dict, list]] = {}
        for group in _chunks(list(positions), int(config.calibration_streaming_chunk)):
            src_idx = sorted({int(pairing.pairing[j]) for j in group})
            src_req = {str(i): (i, "boundary") for i in src_idx}
            tgt_req = {
                f"{pos}.{comp}.h": (pos, COMPONENT_INPUT_KIND[comp]) for pos in group for comp in components
            }
            stats: dict[tuple[int, str], ResidualSufficientStatistics] = {}
            eff: dict[tuple[int, str], Tensor] = {}
            keys: dict[tuple[int, str], str] = {}
            for pos in group:
                for comp in components:
                    keys[(pos, comp)] = shim.component_key(pos, comp, prefixed=True)
                    width = int(current_state[keys[(pos, comp)]].shape[0])
                    eff[(pos, comp)] = _component_effective_out(shim, target_model, pos, comp, width)
                    stats[(pos, comp)] = ResidualSufficientStatistics(device=device)
            d_sq = {pos: 0.0 for pos in group}
            e_sq = {pos: 0.0 for pos in group}
            cov_acc = {
                i: CoverageAccumulator()
                for i in src_idx
                if config.coverage_diagnostics and i not in decoders.coverage
            }
            gens = (
                iter_capture_tokens(source_base_model, source_batches, src_req, device, family_adapter=family_adapter),
                iter_capture_tokens(source_ft_model, source_batches, src_req, device, family_adapter=family_adapter),
                iter_capture_tokens(target_model, target_batches, tgt_req, device, family_adapter=family_adapter),
            )
            try:
                for b0, b1, tgt in zip(*gens, strict=True):
                    e_rows: dict[int, Tensor] = {}
                    for i in src_idx:
                        e = b1[str(i)] - b0[str(i)]  # [1, N, d_m] content rows (never resampled on decoders)
                        e_rows[i] = e.reshape(-1, e.shape[-1])
                        if i in cov_acc:
                            cov_acc[i].update(e_rows[i])
                    for pos in group:
                        i = int(pairing.pairing[pos])
                        fn = decoders.fn(pos)[0]
                        d = fn(e_rows[i]).float()  # [N, d_t]
                        e_sq[pos] += float((e_rows[i].double() ** 2).sum().item())
                        d_sq[pos] += float((d.double() ** 2).sum().item())
                        for comp in components:
                            h = tgt[f"{pos}.{comp}.h"]
                            stats[(pos, comp)].update(h.reshape(-1, h.shape[-1]), d, None, eff[(pos, comp)])
            finally:
                for g in gens:
                    g.close()
            for i, acc in cov_acc.items():
                decoders.coverage[i] = decoders.source_coverage(i, acc)
            del cov_acc
            for pos in group:
                i = int(pairing.pairing[pos])
                _, eps, spectral = decoders.fn(pos)
                position_diag[pos] = {
                    "anchors_source_block": i,
                    "anchors_c_source": source.c[i],
                    "anchors_c_target": target.c[pos],
                    **dict(zip(("anchors_c_source_used", "anchors_c_target_used"),
                               m.decoder_scales(source, i, target, pos, config), strict=True)),
                    "anchors_eps": eps,
                    **{f"anchors_{k}": v for k, v in spectral.items()},
                    "anchors_e_norm": e_sq[pos] ** 0.5,
                    "anchors_d_norm": d_sq[pos] ** 0.5,
                    **decoders.pair_diag(pos),
                    **({f"anchors_{k}": v for k, v in decoders.coverage[i].items()} if i in decoders.coverage else {}),
                }
                position_corrections: dict[str, Tensor] = {}
                block_rows: list[dict[str, Any]] = []
                for comp in components:
                    _finalize_independent_component(
                        stats.pop((pos, comp)), d_sq[pos], 0.0, pos, comp, keys[(pos, comp)], current_state,
                        eff[(pos, comp)], solver_config, source_coordinates, position_corrections, block_rows,
                        h_batches=None, solve_hook=solve_hook,
                    )
                fitted[pos] = (position_corrections, block_rows)
        for pos in positions:
            position_corrections, block_rows = fitted[pos]
            corrections.update(position_corrections)
            for row in block_rows:
                row["source_position"] = row.pop("source_coordinate")
                row["procrustes_source"] = solver_config.procrustes_source
                rows.append(row)
    finally:
        target_model.load_state_dict(original_state, strict=True)
    annotate_target_split(rows, solver_config, pairing)
    return corrections, rows, position_diag


def draw_streaming_calibration(
    source_loader, target_loader, config, family_adapter
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """The calibration draw of the resident path (``paired_calibration``, first tokens excluded): batches hold token ids
    and masks only, so they are token-count independent in the sense that matters (no activations)."""
    if family_adapter is None:
        raise ValueError("calibration_streaming is implemented for decoders (family_adapter) only")
    return paired_calibration(
        source_loader, target_loader, num_batches=config.num_batches, seed=config.seed,
        family_adapter=family_adapter, exclude_first_token=True,
    )


def draw_map_batches(source_loader, target_loader, config, family_adapter) -> tuple[list[Any], list[Any]]:
    """The ``map_rows='task'`` draw of ``task_map_frames`` (``map_num_batches`` or ``num_batches``; first tokens excluded)."""
    s, t, _ = paired_calibration(
        source_loader, target_loader, num_batches=config.map_num_batches or config.num_batches, seed=config.seed,
        family_adapter=family_adapter, exclude_first_token=True,
    )
    return s, t
