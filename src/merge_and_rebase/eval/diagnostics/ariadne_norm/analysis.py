"""Closed-form analysis of one Ariadne ridge regression (N3 spectral budget, N5 row concentration, N7 split-half).

Everything here reads a filled `ResidualSufficientStatistics` (or row banks for N5) and never feeds a fit.
Notation follows ``docs/methods/ariadne.md``: ``Sc = H~^T H~`` (centered input Gram), ``Bc = H~^T D~ L^T``,
``G = L L^T``. The solver's update is ``X = U xhat V^T`` with ``xhat_kl = (U^T Bc V)_kl / (mu_k e_l + lam)``, where
``(mu, U)`` and ``(e, V)`` are the clamped eigenpairs of ``Sc`` and ``G`` (``fit.py:ResidualSufficientStatistics``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics

Tensor = torch.Tensor

# Fixed-relative ridge grid for the L-curve (paper Table 12 range) and the bottom-variance fractions reported.
DEFAULT_RHO_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_TAIL_FRACTIONS = (1e-4, 1e-3, 1e-2, 1e-1)
TAIL_FRACTION = 1e-2


@dataclass(frozen=True)
class CenteredSystem:
    """Centered normal equations of one fit, in float64 on the statistics' device."""

    sc: Tensor
    bc: Tensor
    g: Tensor
    t_out: Tensor
    mu_a: Tensor
    mu_e: Tensor
    n: int
    sum_e2: float

    @property
    def m(self) -> int:
        return int(self.sc.shape[0])

    @property
    def d(self) -> int:
        return int(self.g.shape[0])

    @property
    def centered_target_sq(self) -> float:
        """``||D~||_F^2 = sum ||e_t||^2 - n ||mu_e||^2``."""
        return max(0.0, self.sum_e2 - self.n * float((self.mu_e @ self.mu_e).item()))


def centered_system(stats: ResidualSufficientStatistics) -> CenteredSystem:
    """Same centering as ``ResidualSufficientStatistics.solve(exact_form=True)``."""
    if stats.s is None:
        raise ValueError("statistics are empty")
    if stats._t_in is not None:
        raise ValueError("analysis assumes the direct-target convention t_in=None (A = H)")
    n = int(stats.n_rows)
    s = (stats.s + stats.s.T) * 0.5
    g = (stats.g + stats.g.T) * 0.5
    t_out64 = stats._t_out.to(device=s.device, dtype=torch.float64)
    lmat = t_out64.T
    mu_a = stats.sum_a / n
    mu_e = stats.sum_e / n
    sc = s - torch.outer(stats.sum_a, stats.sum_a) / n
    sc = (sc + sc.T) * 0.5
    bc = stats.b - torch.outer(stats.sum_a, mu_e @ lmat)
    return CenteredSystem(sc=sc, bc=bc, g=g, t_out=t_out64, mu_a=mu_a, mu_e=mu_e, n=n, sum_e2=float(stats.sum_e2))


def official_lambda(system: CenteredSystem, *, ridge_estimator: str, ridge_relative: float) -> float:
    """The ``lam`` of ``solve(exact_form=True, ridge_mode='trace_normalized')`` (``fit.py`` ridge block)."""
    trace_sc = float(torch.trace(system.sc).item())
    trace_g = float(torch.trace(system.g).item())
    if ridge_estimator == "empirical_bayes":
        base = trace_sc / float(system.n - 1)
    elif ridge_estimator == "fixed_relative":
        base = float(ridge_relative) * trace_sc / float(system.m)
    else:
        raise ValueError(f"unsupported ridge_estimator {ridge_estimator!r} for the analysis")
    return base * (trace_g / float(system.d))


def lambda_for_rho(system: CenteredSystem, rho: float) -> float:
    return official_lambda(system, ridge_estimator="fixed_relative", ridge_relative=rho)


class EigenSystem:
    """Eigendecomposition of one centered system, reused for every ``lam``."""

    def __init__(self, system: CenteredSystem) -> None:
        self.system = system
        mu, u = torch.linalg.eigh(system.sc)
        e, v = torch.linalg.eigh(system.g)
        self.mu = mu.clamp_min(0.0)
        self.u = u
        self.e = e.clamp_min(0.0)
        self.v = v
        self.rhs = u.T @ system.bc @ v
        self.weight = self.mu[:, None] * self.e[None, :]

    def xhat(self, lam: float) -> Tensor:
        denom = self.weight + lam
        return torch.where(denom > 0, self.rhs / denom, torch.zeros_like(self.rhs))

    def solution(self, lam: float) -> tuple[Tensor, Tensor]:
        """``(X [m,d], beta [d])`` exactly as the solver forms them (``beta = G^+ (L mu_e - G X^T mu_a)``)."""
        x = self.u @ self.xhat(lam) @ self.v.T
        g_inv = self.v @ torch.diag(torch.where(self.e > 1e-12, 1.0 / self.e, torch.zeros_like(self.e))) @ self.v.T
        beta = g_inv @ (self.system.t_out @ self.system.mu_e - self.system.g @ (x.T @ self.system.mu_a))
        return x, beta

    def point(self, lam: float) -> dict[str, float]:
        """Norms, residual, effective dof and GCV at ``lam``; residual of the centered system (the intercept is
        fitted exactly when ``G`` is invertible, which the caller checks)."""
        xh = self.xhat(lam)
        norm_sq = float((xh**2).sum().item())
        fit_sq = float((self.weight * xh**2).sum().item())
        cross = float((self.rhs * xh).sum().item())
        rss = max(0.0, self.system.centered_target_sq - 2.0 * cross + fit_sq)
        denom = self.weight + lam
        dof = float(torch.where(denom > 0, self.weight / denom, torch.zeros_like(denom)).sum().item())
        n_obs = float(self.system.n - 1) * float(self.system.d)
        gcv = (rss / n_obs) / (1.0 - dof / n_obs) ** 2 if dof < n_obs else math.inf
        _, beta = self.solution(lam)
        return {
            "lam": lam,
            "correction_norm": norm_sq**0.5,
            "bias_norm": float(torch.linalg.norm(beta).item()),
            "fitted_output_norm": fit_sq**0.5,
            "residual_norm_after": rss**0.5,
            "relative_residual_after": (rss / self.system.sum_e2) ** 0.5 if self.system.sum_e2 > 0 else 0.0,
            "effective_dof": dof,
            "effective_dof_fraction": dof / float(self.system.m * self.system.d),
            "gcv": gcv,
        }

    def tail_shares(self, lam: float, fractions: Sequence[float]) -> dict[str, dict[str, float]]:
        """Share of ``||X||^2`` and of ``||H~ X L||^2`` carried by the input directions holding the lowest
        ``fraction`` of the input variance ``tr(Sc)``."""
        xh = self.xhat(lam)
        norm_k = (xh**2).sum(dim=1)
        fit_k = (self.weight * xh**2).sum(dim=1)
        # eigh is ascending: cumulative variance from the bottom.
        cum = torch.cumsum(self.mu, dim=0) / self.mu.sum().clamp_min(1e-300)
        total_norm = float(norm_k.sum().item())
        total_fit = float(fit_k.sum().item())
        out = {}
        for f in fractions:
            tail = cum <= f
            out[f"{f:g}"] = {
                "n_directions": int(tail.sum().item()),
                "correction_norm_sq_share": float(norm_k[tail].sum().item()) / total_norm if total_norm else 0.0,
                "fitted_output_sq_share": float(fit_k[tail].sum().item()) / total_fit if total_fit else 0.0,
            }
        return out

    def spectrum_summary(self) -> dict[str, Any]:
        mu = self.mu
        top = float(mu.max().item())
        tol = top * float(mu.numel()) * torch.finfo(torch.float64).eps
        positive = mu[mu > tol]
        return {
            "m": int(mu.numel()),
            "n_rows": self.system.n,
            "rows_over_inputs": self.system.n / float(mu.numel()),
            "numerical_rank": int(positive.numel()),
            "condition_number": (top / float(positive.min().item())) if positive.numel() else math.inf,
            "trace": float(mu.sum().item()),
            "output_gram_is_identity": bool(
                torch.allclose(
                    self.system.g, torch.eye(self.system.d, dtype=self.system.g.dtype, device=self.system.g.device)
                )
            ),
            "output_gram_min_eig": float(self.e.min().item()),
        }


def spectral_budget(
    stats: ResidualSufficientStatistics,
    *,
    ridge_estimator: str,
    ridge_relative: float,
    rho_grid: Sequence[float] = DEFAULT_RHO_GRID,
    tail_fractions: Sequence[float] = DEFAULT_TAIL_FRACTIONS,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """N3 for one fitted component. Returns (scalar record, arrays for the npz sidecar)."""
    system = centered_system(stats)
    eig = EigenSystem(system)
    lam = official_lambda(system, ridge_estimator=ridge_estimator, ridge_relative=ridge_relative)
    spectrum = eig.spectrum_summary()
    if spectrum["output_gram_min_eig"] <= 1e-12:
        raise ValueError("G is singular: the centered-residual identity used by the analysis does not hold")
    at_official = eig.point(lam)
    curve = [{"rho": "official", **at_official}]
    for rho in rho_grid:
        curve.append({"rho": float(rho), **eig.point(lambda_for_rho(system, rho))})
    finite = [p for p in curve if math.isfinite(p["gcv"])]
    gcv_best = min(finite, key=lambda p: p["gcv"]) if finite else None
    xh = eig.xhat(lam)
    record = {
        "lam": lam,
        "rho_equivalent": lam
        / (float(torch.trace(system.sc).item()) / system.m * float(torch.trace(system.g).item()) / system.d),
        "spectrum": spectrum,
        "lam_over_mu_max": lam / max(float(eig.mu.max().item()), 1e-300),
        "lam_over_mu_median": lam / max(float(eig.mu.median().item()), 1e-300),
        "at_official": at_official,
        "tail_shares": eig.tail_shares(lam, tail_fractions),
        "l_curve": curve,
        "gcv_best_rho": None if gcv_best is None else gcv_best["rho"],
    }
    arrays = {
        "mu_ascending": eig.mu.float().cpu(),
        "correction_norm_sq_by_direction": (xh**2).sum(dim=1).float().cpu(),
        "fitted_output_sq_by_direction": (eig.weight * xh**2).sum(dim=1).float().cpu(),
        "target_sq_by_direction": (eig.rhs**2).sum(dim=1).float().cpu(),
    }
    return record, arrays


def _frob(x: Tensor) -> float:
    return float(torch.linalg.norm(x).item())


def _cos(a: Tensor, b: Tensor) -> float:
    na, nb = _frob(a), _frob(b)
    return float((a * b).sum().item()) / (na * nb) if na > 0 and nb > 0 else 0.0


def split_half_stability(
    full: ResidualSufficientStatistics,
    halves: Sequence[ResidualSufficientStatistics],
    *,
    ridge_estimator: str,
    ridge_relative: float,
    rho_grid: Sequence[float] = DEFAULT_RHO_GRID,
) -> dict[str, Any]:
    """N7: odd- vs even-batch fits with the same Procrustes maps (regression stability only).

    Parameter-space difference ``||X_a - X_b|| / mean norm`` is compared with the functional difference measured on
    the full calibration inputs, ``||H~ (X_a - X_b) L|| / ||H~ X_full L||``: a large parameter difference with a small
    functional one means the norm of the update is not identified by the data.
    """
    if len(halves) != 2:
        raise ValueError("split_half_stability needs exactly two halves")
    sys_full = centered_system(full)
    eig_full = EigenSystem(sys_full)
    systems = [centered_system(h) for h in halves]
    eigs = [EigenSystem(s) for s in systems]

    def functional_sq(dx: Tensor) -> float:
        return float(torch.trace(dx.T @ sys_full.sc @ dx @ sys_full.g).item())

    def compare(lams_half: Sequence[float], lam_full: float) -> dict[str, float]:
        (xa, ba), (xb, bb) = (e.solution(lam) for e, lam in zip(eigs, lams_half, strict=True))
        xf, bf = eig_full.solution(lam_full)
        mean_norm = 0.5 * (_frob(xa) + _frob(xb))
        fit_full = functional_sq(xf) ** 0.5
        return {
            "lam_half_a": lams_half[0],
            "lam_half_b": lams_half[1],
            "correction_norm_full": _frob(xf),
            "correction_norm_half_a": _frob(xa),
            "correction_norm_half_b": _frob(xb),
            "param_rel_diff": _frob(xa - xb) / mean_norm if mean_norm > 0 else 0.0,
            "param_cosine": _cos(xa, xb),
            "functional_rel_diff": functional_sq(xa - xb) ** 0.5 / fit_full if fit_full > 0 else 0.0,
            "bias_rel_diff": _frob(ba - bb) / (0.5 * (_frob(ba) + _frob(bb)) or 1.0),
            "bias_cosine": _cos(ba, bb),
        }

    out: dict[str, Any] = {
        "n_rows_halves": [s.n for s in systems],
        "official": compare(
            [official_lambda(s, ridge_estimator=ridge_estimator, ridge_relative=ridge_relative) for s in systems],
            official_lambda(sys_full, ridge_estimator=ridge_estimator, ridge_relative=ridge_relative),
        ),
        "by_rho": [],
    }
    for rho in rho_grid:
        row = compare([lambda_for_rho(s, rho) for s in systems], lambda_for_rho(sys_full, rho))
        out["by_rho"].append({"rho": float(rho), **row})
    return out


def procrustes_split_half(q_a: Tensor, q_b: Tensor, q_full: Tensor) -> dict[str, float]:
    """N7 for the Procrustes map: distance and subspace overlap of the odd/even polar factors."""
    q_a, q_b, q_full = (q.double() for q in (q_a, q_b, q_full))
    d_min = min(q_full.shape)
    return {
        "q_rel_diff": _frob(q_a - q_b) / _frob(q_full),
        "q_overlap": float(((q_a.T @ q_b).norm() ** 2).item()) / d_min,
        "q_half_a_vs_full_rel_diff": _frob(q_a - q_full) / _frob(q_full),
    }


def row_concentration(
    h_rows: Tensor,
    e_rows: Tensor,
    token_index: Tensor,
    t_out: Tensor,
    *,
    ridge_estimator: str,
    ridge_relative: float,
    exact_form: bool,
    quantiles: Sequence[float] = (1e-3, 1e-2),
    device=None,
) -> dict[str, Any]:
    """N5 on the row banks of one position: concentration of target energy and leverage, token-type split, and exact
    refits without the most extreme rows (same solver as the method)."""
    if h_rows.shape[0] != e_rows.shape[0] or token_index.shape[0] != h_rows.shape[0]:
        raise ValueError("row banks and token index must have the same number of rows")
    full = ResidualSufficientStatistics(device=device)
    full.update(h_rows, e_rows, None, t_out)
    system = centered_system(full)
    eig = EigenSystem(system)
    if not torch.allclose(system.g, torch.eye(system.d, dtype=system.g.dtype, device=system.g.device)):
        raise ValueError("row_concentration assumes an identity output map (no LayerScale)")
    lam = official_lambda(system, ridge_estimator=ridge_estimator, ridge_relative=ridge_relative)
    x_full, _ = full.solve(ridge_relative=ridge_relative, ridge_estimator=ridge_estimator, exact_form=exact_form)
    base_norm = _frob(x_full)
    h = h_rows.to(system.sc.device, torch.float64) - system.mu_a
    e = e_rows.to(system.sc.device, torch.float64) - system.mu_e
    proj = h @ eig.u
    leverage = (proj**2 / (eig.mu + lam)).sum(dim=1)
    energy = (e**2).sum(dim=1)
    h_energy = (h**2).sum(dim=1)
    n = h.shape[0]

    def top_share(values: Tensor, q: float) -> float:
        k = max(1, int(math.ceil(q * n)))
        return float(values.topk(k).values.sum().item()) / max(float(values.sum().item()), 1e-300)

    first = token_index.to(h.device) == 0
    record: dict[str, Any] = {
        "n_rows": int(n),
        "lam": lam,
        "correction_norm": base_norm,
        "shares": {
            f"{q:g}": {
                "target_energy": top_share(energy, q),
                "input_energy": top_share(h_energy, q),
                "leverage": top_share(leverage, q),
            }
            for q in quantiles
        },
        "first_token": {
            "row_share": float(first.double().mean().item()),
            "target_energy_share": float(energy[first].sum().item()) / max(float(energy.sum().item()), 1e-300),
            "input_energy_share": float(h_energy[first].sum().item()) / max(float(h_energy.sum().item()), 1e-300),
            "mean_leverage_first": float(leverage[first].mean().item()) if bool(first.any()) else 0.0,
            "mean_leverage_rest": float(leverage[~first].mean().item()) if bool((~first).any()) else 0.0,
        },
        "refits": [],
    }

    def refit(keep: Tensor, label: str, q: float | None) -> dict[str, Any]:
        stats = ResidualSufficientStatistics(device=device)
        stats.update(h_rows, e_rows, None, t_out, row_mask=keep.cpu())
        x, diag = stats.solve(ridge_relative=ridge_relative, ridge_estimator=ridge_estimator, exact_form=exact_form)
        x64 = x.double().T.to(system.sc.device)
        beta = diag["bias_correction"].double().to(system.sc.device)
        # Residual of the refit on ALL rows (exact, from the full statistics).
        resid_all = full._residual_sq(x64, beta, system.t_out, system.mu_a, system.mu_e) ** 0.5
        return {
            "criterion": label,
            "quantile": q,
            "rows_removed": int((~keep).sum().item()),
            "correction_norm": float(diag["correction_norm"]),
            "correction_norm_ratio": float(diag["correction_norm"]) / base_norm if base_norm else 0.0,
            "relative_residual_kept_rows": float(diag["residual_norm_after"])
            / max(diag["residual_norm_before"], 1e-300),
            "relative_residual_all_rows": resid_all / max(system.sum_e2**0.5, 1e-300),
        }

    record["baseline_relative_residual"] = refit(torch.ones(n, dtype=torch.bool), "none", None)[
        "relative_residual_all_rows"
    ]
    for label, values in (("target_energy", energy), ("leverage", leverage), ("input_energy", h_energy)):
        for q in quantiles:
            k = max(1, int(math.ceil(q * n)))
            keep = torch.ones(n, dtype=torch.bool, device=values.device)
            keep[values.topk(k).indices] = False
            record["refits"].append(refit(keep, label, q))
    if bool(first.any()) and bool((~first).any()):
        record["refits"].append(refit(~first, "drop_first_token", None))
    return record


class RealizationAccumulator:
    """N2/N6 sums for one mounted delta: per position ``||dT||^2``, ``<dT, D>``, ``||D||^2``, ``||dT - D||^2`` and the
    base-state norm ``||T~_0||^2`` (centered per batch-free running mean is not needed: the ratio uses raw norms)."""

    def __init__(self, positions: Sequence[int]) -> None:
        self.positions = list(positions)
        self.sums = {j: {"dt": 0.0, "dt_d": 0.0, "d": 0.0, "err": 0.0, "t0": 0.0} for j in self.positions}

    def add(self, j: int, dt: Tensor, d: Tensor | None, t0: Tensor) -> None:
        dt64 = dt.double()
        s = self.sums[j]
        s["dt"] += float((dt64**2).sum().item())
        s["t0"] += float((t0.double() ** 2).sum().item())
        if d is not None:
            d64 = d.double().to(dt64.device)
            s["dt_d"] += float((dt64 * d64).sum().item())
            s["d"] += float((d64**2).sum().item())
            s["err"] += float(((dt64 - d64) ** 2).sum().item())

    def rows(self) -> dict[int, dict[str, float]]:
        out = {}
        for j, s in self.sums.items():
            d_norm = s["d"] ** 0.5
            dt_norm = s["dt"] ** 0.5
            out[j] = {
                "delta_norm": dt_norm,
                "desired_norm": d_norm,
                "base_state_norm": s["t0"] ** 0.5,
                "relative_perturbation": dt_norm / s["t0"] ** 0.5 if s["t0"] > 0 else 0.0,
                "r": dt_norm / d_norm if d_norm > 0 else None,
                "cos": s["dt_d"] / (dt_norm * d_norm) if dt_norm > 0 and d_norm > 0 else None,
                "projection_coefficient": s["dt_d"] / s["d"] if s["d"] > 0 else None,
                "relative_error": s["err"] ** 0.5 / d_norm if d_norm > 0 else None,
            }
        return out


def label_free_alpha(curve: Mapping[float, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    """``alpha_hat = argmin_alpha sum_j ||dT_j(alpha) - D_j||^2 / ||D_j||^2`` over the measured grid."""
    objective = {}
    for alpha, rows in curve.items():
        vals = [r["relative_error"] ** 2 for r in rows.values() if r["relative_error"] is not None]
        objective[float(alpha)] = sum(vals) / len(vals) if vals else math.inf
    best = min(objective, key=objective.get)
    return {"alpha_hat": best, "objective": objective}
