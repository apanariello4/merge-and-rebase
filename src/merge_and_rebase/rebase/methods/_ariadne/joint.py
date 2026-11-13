"""Joint-across-depth Ariadne fit, case J ~ I (``depth_fit="joint"``; ariadne_linear.tex Sec. 6.3, "Joint solution").

Every target block k edits ``mlp.c_proj`` (decoder: ``down_proj``) by ``(dW_k, b_k)``; with the c_proj input ``H_k`` of
the TARGET BASE (unmounted, as in the independent fit) block k writes ``Z_k = (H_k dW_k + 1 b_k^T) L_k`` into the
residual stream (``dW_k`` stored as ``m x d_w`` like the independent solver's ``x``, ``L_k`` = its ``effective_out``:
identity, or the LayerScale diagonal). With ``J_{k->j} ~ I`` the change at boundary j is the cumulative sum
``S_j = sum_{k<=j} Z_k`` and the objective is

    sum_j w_j ||S_j - D_j||_F^2 + sum_k lambda_k ||dW_k||_F^2,        w_j = 1 (``joint_weights='uniform'``),

``D_j`` the usual TOTAL block-boundary target (same Q_j, alignment, row handling as `fit_direct_residual_streaming`),
``lambda_k`` the independent fit's ridge of block k (``rows[k]['ridge']``, EB or fixed_relative), ``b_k`` unpenalised
(only with ``exact_form``; without it the intercept is absent, as in the independent fit).

Normal equations with ``w = (P_k)``, ``P_k = [dW_k; b_k^T]``, ``Ht_k = [H_k, 1]``, ``c_k = sum_{j>=k} w_j``:

    N(w)_k = Ht_k^T G_k L_k^T + Lambda_k P_k,   G_k = sum_{j>=k} w_j S_j,   rhs_k = Ht_k^T (sum_{j>=k} w_j D_j) L_k^T.

Solved by preconditioned conjugate gradients on ``N``; ``A w`` and ``A^T`` are never formed. One matvec = ONE sweep of the
target base over the calibration batches with the c_proj inputs of ALL blocks captured at once (per batch: ``Z_k``,
prefix sums ``S_j``, reverse sums ``G_k``, accumulation of ``Ht_k^T G_k L_k^T``); no source model enters ``N(w)``. The
right-hand side and the exact final errors need ``D_j`` (source models) and take one sweep each. The preconditioner is
the block diagonal ``c_k gbar_k Ht_k^T Ht_k + diag(lambda_k, ..., lambda_k, 0)`` (Cholesky, float64), ``gbar_k =
tr(L_k L_k^T) / d_w``: exact for the diagonal blocks when ``L_k L_k^T = gbar_k I`` (identity; decoders), an approximation
for a LayerScale diagonal (the CG solution does not depend on it). The start is the independent (Jacobi) solution.
Float64 throughout; per-block factors, CG vectors and Cholesky factors live on the host (``m x d_w`` and ``(m+1)^2``
doubles per block each). The independent pass honours ``streaming_position_chunk``; the CG / rhs / evaluation sweeps
do not (all blocks at once).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from ...discrete_layer_match import DiscreteLayerPairing
from .capture import iter_capture_tokens
from .config import DirectResidualConfig, order_components
from .fit import ResidualSufficientStatistics
from .layouts import COMPONENT_INPUT_KIND, _aligned, _component_effective_out, _layout_for
from .streaming import _check_boundary_fingerprint, fit_direct_residual_streaming

Tensor = torch.Tensor


def _dot(a: list[Tensor], b: list[Tensor]) -> float:
    return float(sum((x * y).sum().item() for x, y in zip(a, b, strict=True)))


class _JointProblem:
    """The stacked unknowns, the normal operator ``N`` (one target sweep per call) and its block preconditioner."""

    def __init__(self, target_model, prepared, pairing, config, device, family_adapter, blocks, with_bias) -> None:
        self.model, self.prepared, self.pairing, self.config = target_model, prepared, pairing, config
        self.device, self.family_adapter = device, family_adapter
        self.blocks = blocks  # per position: {"m", "d_w", "lam", "out" (L, float64), "chol"}
        self.with_bias = with_bias
        self.n_blocks = len(blocks)
        self.shim = _layout_for(family_adapter)
        self.sweeps = 0
        self._h_requests = {f"{k}.h": (k, COMPONENT_INPUT_KIND["mlp.c_proj"]) for k in range(self.n_blocks)}

    # -- shapes and algebra ---------------------------------------------------------------------------------------

    def zeros(self) -> list[Tensor]:
        return [torch.zeros(b["m"] + int(self.with_bias), b["d_w"], dtype=torch.float64) for b in self.blocks]

    def penalty(self, w: list[Tensor]) -> float:
        return float(sum(b["lam"] * (p[: b["m"]] ** 2).sum().item() for b, p in zip(self.blocks, w, strict=True)))

    def _write(self, k: int, h: Tensor, p: Tensor) -> Tensor:
        """``Z_k = (H_k dW_k + 1 b_k^T) L_k`` for one batch (rows ``[n, d_res]``, float64, on the compute device)."""
        b = self.blocks[k]
        y = h.reshape(-1, b["m"]).to(self.device, torch.float64) @ p[: b["m"]].to(self.device)
        if self.with_bias:
            y = y + p[b["m"]].to(self.device)
        return y @ b["out"]

    def _adjoint(self, k: int, h: Tensor, g: Tensor) -> Tensor:
        """``[H_k, 1]^T (g L_k^T)`` for one batch, on the host."""
        b = self.blocks[k]
        e = g @ b["out"].T
        rows = h.reshape(-1, b["m"]).to(self.device, torch.float64).T @ e
        if self.with_bias:
            rows = torch.cat([rows, e.sum(0, keepdim=True)], dim=0)
        return rows.cpu()

    # -- sweeps -----------------------------------------------------------------------------------------------------

    def _target_sweep(self, requests):
        self.sweeps += 1
        return iter_capture_tokens(
            self.model,
            self.prepared["target_batches"],
            requests,
            self.device,
            family_adapter=self.family_adapter,
            store_device="cpu",
        )

    def matvec(self, w: list[Tensor]) -> tuple[list[Tensor], float]:
        """``(N w, sum_j ||S_j||^2)``: one sweep of the target base, all blocks at once."""
        out = self.zeros()
        data_sq = 0.0
        with contextlib.closing(self._target_sweep(self._h_requests)) as gen:
            for tgt in gen:
                hs = [tgt[f"{k}.h"] for k in range(self.n_blocks)]
                s = [self._write(k, hs[k], w[k]) for k in range(self.n_blocks)]
                for k in range(1, self.n_blocks):
                    s[k] = s[k] + s[k - 1]  # prefix sums S_k = sum_{l<=k} Z_l
                data_sq += sum(float((x * x).sum().item()) for x in s)
                g = torch.zeros_like(s[0])
                for k in reversed(range(self.n_blocks)):
                    g = g + s[k]  # G_k = sum_{j>=k} S_j (uniform weights)
                    out[k] += self._adjoint(k, hs[k], g)
        for k, b in enumerate(self.blocks):
            out[k][: b["m"]] += b["lam"] * w[k][: b["m"]]
        return out, data_sq

    def _source_sweep(self, source_base_model, source_ft_model):
        """Lockstep (target with all c_proj inputs and boundaries, source base, source FT) -> ``(hs, [D_j])`` per batch."""
        pairing, prepared, config = self.pairing, self.prepared, self.config
        q_by_position, fingerprints = prepared["q_by_position"], prepared["fingerprints"]
        distinct = sorted({pairing.pairing[j] for j in range(self.n_blocks)})
        src_requests = {str(i): (i, "boundary") for i in distinct}
        tgt_requests = {**self._h_requests, **{f"{k}.out": (k, "boundary") for k in range(self.n_blocks)}}
        sb = iter_capture_tokens(
            source_base_model,
            prepared["source_batches"],
            src_requests,
            self.device,
            family_adapter=self.family_adapter,
            store_device=self.device,
        )
        sf = iter_capture_tokens(
            source_ft_model,
            prepared["source_batches"],
            src_requests,
            self.device,
            family_adapter=self.family_adapter,
            store_device=self.device,
        )
        tg = self._target_sweep(tgt_requests)
        with contextlib.closing(sb), contextlib.closing(sf), contextlib.closing(tg):
            for k_batch, (b_cap, f_cap, tgt) in enumerate(zip(sb, sf, tg, strict=True)):
                hs, ds = [], []
                for j in range(self.n_blocks):
                    out = tgt[f"{j}.out"]
                    _check_boundary_fingerprint(out, fingerprints[(k_batch, j)], config, j, k_batch)
                    out_cpu = out.cpu()
                    i = pairing.pairing[j]
                    b = _aligned([b_cap[str(i)].cpu()], [out_cpu])[0]
                    f = _aligned([f_cap[str(i)].cpu()], [out_cpu])[0]
                    desired = (f - b) @ q_by_position[j]
                    hs.append(tgt[f"{j}.h"])
                    ds.append(desired.reshape(-1, desired.shape[-1]).to(self.device, torch.float64))
                yield hs, ds

    def rhs(self, source_base_model, source_ft_model) -> tuple[list[Tensor], list[float]]:
        """``(A^T Omega d, ||D_j||^2 per block)`` in one sweep."""
        b = self.zeros()
        d_sq = [0.0] * self.n_blocks
        for hs, ds in self._source_sweep(source_base_model, source_ft_model):
            g = torch.zeros_like(ds[0])
            for k in reversed(range(self.n_blocks)):
                d_sq[k] += float((ds[k] * ds[k]).sum().item())
                g = g + ds[k]  # sum_{j>=k} D_j
                b[k] += self._adjoint(k, hs[k], g)
        return b, d_sq

    def errors(self, ws: list[list[Tensor]], source_base_model, source_ft_model) -> list[list[float]]:
        """Exact ``sum over rows of ||S_j - D_j||^2`` per block j for each weight set, in one sweep."""
        err = [[0.0] * self.n_blocks for _ in ws]
        for hs, ds in self._source_sweep(source_base_model, source_ft_model):
            for n, w in enumerate(ws):
                s = None
                for k in range(self.n_blocks):
                    z = self._write(k, hs[k], w[k])
                    s = z if s is None else s + z
                    err[n][k] += float(((s - ds[k]) ** 2).sum().item())
        return err

    # -- preconditioner -----------------------------------------------------------------------------------------------

    def precondition(self, r: list[Tensor]) -> list[Tensor]:
        return [
            torch.cholesky_solve(x.to(self.device), b["chol"].to(self.device)).cpu()
            for b, x in zip(self.blocks, r, strict=True)
        ]


def _gram_preconditioner(stats: ResidualSufficientStatistics, c_k: int, gbar: float, lam: float, with_bias: bool):
    """Cholesky factor of ``c_k gbar [H,1]^T [H,1] + diag(lam, ..., lam, 0)`` from the independent fit's raw statistics."""
    s = (stats.s + stats.s.T) * 0.5
    m = s.shape[0]
    if with_bias:
        a = stats.sum_a.reshape(m, 1)
        gram = torch.cat(
            [
                torch.cat([s, a], dim=1),
                torch.cat([a.T, torch.full((1, 1), float(stats.n_rows), dtype=s.dtype, device=s.device)], dim=1),
            ],
            dim=0,
        )
    else:
        gram = s
    mat = (c_k * gbar) * gram
    idx = torch.arange(m, device=mat.device)
    mat[idx, idx] += lam
    return torch.linalg.cholesky(mat).cpu()


@torch.no_grad()
def fit_direct_residual_joint_streaming(
    target_model,
    target_base_state: Mapping[str, Tensor],
    source_base_model,
    source_ft_model,
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """``depth_fit="joint"``: the independent fit, then PCG on the cumulative objective (see the module docstring).

    Returns ``(corrections, rows)`` with the independent schema; per row the mounted ``correction_norm`` /
    ``bias_norm`` / ``bias_correction`` / ``correction_rank`` are the JOINT solution's (the independent ones are kept
    as ``independent_*``), the other independent fields (``residual_norm_after`` etc.) stay the independent fit's own
    per-block numbers. New per-row fields: ``depth_fit``, ``joint_cumulative_rel_error`` (``||S_j - D_j|| / ||D_j||``
    under the linear model) and ``independent_cumulative_rel_error``; ``rows[0]['joint_run']`` carries the run record
    (iterations, residual / objective histories, exact objectives of both solutions, sweep counts).
    """
    if config.depth_fit != "joint":
        raise ValueError("fit_direct_residual_joint_streaming requires depth_fit='joint'")
    (component,) = order_components(config.components)
    n_blocks = pairing.target_depth
    shim = _layout_for(family_adapter)
    with_bias = bool(config.exact_form)

    recorded: dict[int, ResidualSufficientStatistics] = {}

    def keep(position, _component, dev):
        recorded[position] = ResidualSufficientStatistics(device=dev)
        return recorded[position]

    indep_corrections, rows = fit_direct_residual_streaming(
        target_model,
        target_base_state,
        source_base_model,
        source_ft_model,
        prepared,
        pairing,
        config=replace(config, depth_fit="independent"),
        device=device,
        family_adapter=family_adapter,
        stats_factory=keep,
    )

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        for m in unique_models.values():
            m.to(device).eval()

        row_of = {int(r["position"]): r for r in rows}
        blocks, w0, keys = [], [], []
        for k in range(n_blocks):
            key = shim.component_key(k, component, prefixed=True)
            bias_key = f"{key[: -len('.weight')]}.bias"
            d_w, m_in = (int(x) for x in target_base_state[key].shape)
            out = _component_effective_out(shim, target_model, k, component, d_w).to(torch.float64).to(device)
            lam = float(row_of[k]["ridge"])
            stats = recorded.pop(k)
            gbar = float(torch.trace(out @ out.T).item()) / d_w
            chol = _gram_preconditioner(stats, n_blocks - k, gbar, lam, with_bias)
            del stats
            blocks.append({"m": m_in, "d_w": d_w, "lam": lam, "out": out, "chol": chol})
            p0 = indep_corrections[key].double().T
            if with_bias:
                if bias_key not in indep_corrections:
                    raise RuntimeError(f"depth_fit='joint' with exact_form needs the bias {bias_key} to be fitted")
                p0 = torch.cat([p0, indep_corrections[bias_key].double()[None]], dim=0)
            w0.append(p0.contiguous())
            keys.append((key, bias_key))

        problem = _JointProblem(target_model, prepared, pairing, config, device, family_adapter, blocks, with_bias)
        b_vec, d_sq = problem.rhs(source_base_model, source_ft_model)
        dd = float(sum(d_sq))
        b_norm = math.sqrt(_dot(b_vec, b_vec))

        def objective(w, r):
            total = dd - _dot(w, b_vec) - _dot(w, r)  # w^T N w - 2 w^T b + dd with N w = b - r
            pen = problem.penalty(w)
            return total, total - pen, pen

        w = w0
        nw, _ = problem.matvec(w)
        r = [x - y for x, y in zip(b_vec, nw, strict=True)]
        rel = math.sqrt(_dot(r, r)) / b_norm if b_norm > 0 else 0.0
        res_hist, obj_hist, data_hist, pen_hist = [rel], [], [], []
        o = objective(w, r)
        obj_hist.append(o[0]), data_hist.append(o[1]), pen_hist.append(o[2])
        iterations = 0
        if rel > config.joint_tol:
            z = problem.precondition(r)
            p = [x.clone() for x in z]
            rz = _dot(r, z)
            for iteration in range(1, config.joint_max_iters + 1):
                iterations = iteration
                q, _ = problem.matvec(p)
                alpha = rz / _dot(p, q)
                w = [x + alpha * y for x, y in zip(w, p, strict=True)]
                r = [x - alpha * y for x, y in zip(r, q, strict=True)]
                rel = math.sqrt(_dot(r, r)) / b_norm
                res_hist.append(rel)
                o = objective(w, r)
                obj_hist.append(o[0]), data_hist.append(o[1]), pen_hist.append(o[2])
                if rel <= config.joint_tol:
                    break
                z = problem.precondition(r)
                rz_new = _dot(r, z)
                p = [x + (rz_new / rz) * y for x, y in zip(z, p, strict=True)]
                rz = rz_new
        converged = rel <= config.joint_tol

        err_ind, err_joint = problem.errors([w0, w], source_base_model, source_ft_model)
        pen_ind, pen_joint = problem.penalty(w0), problem.penalty(w)
        corrections: dict[str, Tensor] = {}
        for k, (key, bias_key) in enumerate(keys):
            m_in = blocks[k]["m"]
            corrections[key] = w[k][:m_in].T.contiguous().to(torch.float32)
            if with_bias:
                corrections[bias_key] = w[k][m_in].to(torch.float32)
            row = row_of[k]
            d_norm = math.sqrt(d_sq[k])
            row["depth_fit"] = "joint"
            row["independent_correction_norm"] = row["correction_norm"]
            row["independent_bias_norm"] = row["bias_norm"]
            row["correction_norm"] = float(torch.linalg.norm(w[k][:m_in]).item())
            row["bias_norm"] = float(torch.linalg.norm(w[k][m_in]).item()) if with_bias else 0.0
            row["bias_correction"] = (
                w[k][m_in].to(torch.float32) if with_bias else torch.zeros(blocks[k]["d_w"], dtype=torch.float32)
            )
            row["correction_rank"] = int(torch.linalg.matrix_rank(w[k][:m_in]).item())
            row["joint_cumulative_rel_error"] = math.sqrt(err_joint[k]) / d_norm if d_norm else 0.0
            row["independent_cumulative_rel_error"] = math.sqrt(err_ind[k]) / d_norm if d_norm else 0.0
        rows[0]["joint_run"] = {
            "iterations": iterations,
            "converged": converged,
            "joint_tol": float(config.joint_tol),
            "joint_max_iters": int(config.joint_max_iters),
            "rhs_norm": b_norm,
            "residual_history": res_hist,
            "objective_history": obj_hist,
            "data_term_history": data_hist,
            "penalty_history": pen_hist,
            "objective_independent": sum(err_ind) + pen_ind,
            "objective_joint": sum(err_joint) + pen_joint,
            "data_term_independent": sum(err_ind),
            "data_term_joint": sum(err_joint),
            "penalty_independent": pen_ind,
            "penalty_joint": pen_joint,
            "target_sq_norm": dd,
            "weights": config.joint_weights,
            "bias": with_bias,
            "target_sweeps": problem.sweeps,
        }
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    return corrections, rows
