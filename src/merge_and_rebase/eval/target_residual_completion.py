"""Closed-form target residual completion for inserted ``c_proj`` TVs.

The module deliberately contains no model hooks.  Callers collect the current
target activations and native boundary residuals, then feed them here.  Both
Theseus and BiCo use the same weight convention in this API::

    C_target = t_out.T @ C_source @ t_in

The solve is performed in source coordinates and uses only sufficient
statistics, so it never materializes a Kronecker system.

Objective (exact form, ``ResidualCompletionConfig.exact_form=True``, the
default)::

    min_{X, beta}  || (A X + 1 beta^T) t_out - E ||_F^2 + lam ||X||_F^2

where ``A`` is the source-coordinate feature bank (``h @ t_in.T``), ``E`` is
the native boundary residual, ``X`` is ``Delta_C_source.T``, and ``beta`` is
a source-coordinate intercept for ``c_proj.bias`` (mirrors ``X``: fit before
transport, transported the same way afterwards).  This is the affine ridge
of ICLR2027BlockExtension.pdf Eq. 8-9 (``g(H) = H W^T + 1 b^T``), specialized
to the case where the fitted map is further pushed through the fixed,
generally non-square, generally non-orthogonal transport matrix ``t_out``.
Unlike the plain (no-transport) case, ``beta`` is *constrained* to influence
the fit only through ``t_out^T @ beta`` -- it is not a free target-space
vector -- because it is transported exactly like the weight's output side
(see ``theseus._transport_bias``, which computes ``delta_vec @ t_out`` for a
source-space bias delta; that is algebraically ``t_out.T @ delta_vec``, the
same convention used here).

The reduced form (``exact_form=False``) is the literal previous
implementation: no centering, no intercept, and a ridge scale built only
from the feature Gram.  It is exact only under the idealized assumptions
that the residual (and, if present, the source feature bank) is
zero-mean and that ``t_out`` is exactly isometric (``trace(t_out t_out^T)
== d_source``).  It is kept reachable purely as an ablation switch; no
published or historical proposal-1 result has ever been produced with
either form, so there is no prior default to preserve and the scientifically
exact affine form is used by default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class ResidualCompletionConfig:
    enabled: bool = False
    added_blocks: str = "all"
    # ``inserted`` is the Proposal-1 protocol used by existing campaigns.
    # ``all`` is an opt-in ablation that completes every realized target block
    # position, including original descendants, using its source ancestry.
    target_scope: str = "inserted"
    component: str = "c_proj.weight"
    ridge_relative: float = 1e-3
    strength: float = 1.0
    num_batches: int = 10
    # Exact affine fit (centered sufficient statistics + closed-form intercept,
    # ridge scaled by both the feature and transport Gram spectra) vs. the
    # historical reduced fit (no centering, no intercept, ridge scaled by the
    # feature Gram alone). The reduced form is valid only when the residual
    # (and source features) are zero-mean and t_out is exactly isometric; no
    # proposal-1 result has ever been produced under either setting, so there
    # is no historical default to preserve, and the exact form is the default.
    exact_form: bool = True
    # What to do when the residual-writing projection has no bias parameter to
    # receive the affine intercept. CLIP's mlp.c_proj always has one; HF decoder
    # MLPs are bias-free (Qwen2.5 sets mlp_bias=False), so the exact form has
    # nowhere to put it.
    #   "error"       -- refuse (the default, and vision's only reachable path)
    #   "materialize" -- add a zero bias to the target projection and write the
    #                    intercept there; exact, at the cost of a checkpoint
    #                    carrying a parameter stock Qwen does not have
    #   "skip"        -- drop the intercept, allowed ONLY when it is exactly
    #                    zero (exact_form=False). Dropping a fitted, nonzero
    #                    intercept is refused: W is fitted on centered banks, so
    #                    applying it without the intercept is not the same map.
    missing_bias: str = "error"


def parse_residual_completion_config(value: Mapping[str, Any] | None) -> ResidualCompletionConfig:
    """Parse and validate the narrow proposal-1 configuration schema."""
    if value is None:
        return ResidualCompletionConfig()
    if not isinstance(value, Mapping):
        raise TypeError("target_residual_completion must be a mapping")
    allowed = {
        "enabled", "added_blocks", "target_scope", "component", "ridge_relative", "strength", "num_batches",
        "exact_form", "missing_bias",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown target_residual_completion fields: {sorted(unknown)}")
    cfg = ResidualCompletionConfig(**dict(value))
    if not isinstance(cfg.enabled, bool):
        raise TypeError("enabled must be bool")
    if cfg.added_blocks != "all":
        raise ValueError("added_blocks must be 'all' for the initial all-inserted-block protocol")
    if cfg.target_scope not in {"inserted", "all"}:
        raise ValueError("target_scope must be 'inserted' or 'all'")
    if cfg.component != "c_proj.weight":
        raise ValueError("component must be 'c_proj.weight'")
    if isinstance(cfg.num_batches, bool) or not isinstance(cfg.num_batches, int) or cfg.num_batches <= 0:
        raise ValueError("num_batches must be a positive integer")
    if isinstance(cfg.ridge_relative, bool) or not isinstance(cfg.ridge_relative, (int, float)):
        raise ValueError("ridge_relative must be a finite real number")
    if not math.isfinite(float(cfg.ridge_relative)):
        raise ValueError("ridge_relative must be finite")
    if cfg.ridge_relative <= 0:
        raise ValueError("ridge_relative must be > 0")
    if isinstance(cfg.strength, bool) or not isinstance(cfg.strength, (int, float)):
        raise ValueError("strength must be a finite real number")
    if not math.isfinite(float(cfg.strength)):
        raise ValueError("strength must be finite")
    if cfg.strength < 0:
        raise ValueError("strength must be >= 0")
    if not isinstance(cfg.exact_form, bool):
        raise TypeError("exact_form must be bool")
    if cfg.missing_bias not in {"error", "materialize", "skip"}:
        raise ValueError("missing_bias must be 'error', 'materialize' or 'skip'")
    if cfg.missing_bias == "skip" and cfg.exact_form:
        # The exact form fits a centered map and recovers a nonzero intercept;
        # applying its weight without that intercept is a different map, not an
        # approximation of it. Refused up front rather than at the write.
        raise ValueError(
            "missing_bias='skip' requires exact_form=false: the exact form fits a "
            "nonzero intercept, and dropping it would apply a centered-fit weight "
            "without the centering it assumes"
        )
    return cfg


def centered_rectangular_procrustes(
    source_rows: Tensor, target_rows: Tensor, *, eps: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Return row map ``Q`` minimizing centered orthogonal Procrustes error.

    ``source_rows`` is ``[N, d_source]`` and ``target_rows`` is
    ``[N, d_target]``; the returned map is ``[d_source, d_target]``.
    """
    _check_rows(source_rows, target_rows, "source_rows", "target_rows")
    if source_rows.shape[0] == 0:
        raise ValueError("Procrustes requires at least one row")
    if not torch.isfinite(source_rows).all() or not torch.isfinite(target_rows).all():
        raise ValueError("Procrustes inputs must be finite")
    if eps <= 0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be finite and > 0")
    source_rows = source_rows.to(torch.float64)
    target_rows = target_rows.to(torch.float64)
    src_mean = source_rows.mean(dim=0)
    tgt_mean = target_rows.mean(dim=0)
    cross = (source_rows - src_mean).T @ (target_rows - tgt_mean)
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh
    return q, src_mean, tgt_mean


def backproject_target_rows(target_rows: Tensor, q: Tensor, target_mean: Tensor, source_mean: Tensor) -> Tensor:
    """Map target rows back to source coordinates, retaining the source mean."""
    if target_rows.ndim != 2 or q.ndim != 2:
        raise ValueError("target_rows and q must be matrices")
    if target_rows.shape[1] != q.shape[1] or source_mean.shape != (q.shape[0],):
        raise ValueError("incompatible target map or source mean shapes")
    if target_mean.shape != (q.shape[1],):
        raise ValueError("target_mean has the wrong shape")
    return (target_rows - target_mean) @ q.T + source_mean


def _check_rows(h: Tensor, e: Tensor, h_name: str, e_name: str) -> None:
    if h.ndim != 2 or e.ndim != 2:
        raise ValueError(f"{h_name} and {e_name} must be rank-2")
    if h.shape[0] != e.shape[0]:
        raise ValueError("activation and residual row counts must match")


def _clamped_eigh_inverse(sym: Tensor, *, eps: float = 1e-12) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(eigvals_clamped, eigvecs, pseudo_inverse)`` of a symmetric PSD matrix."""
    vals, vecs = torch.linalg.eigh(sym)
    vals = vals.clamp_min(0.0)
    inv = vecs @ torch.diag(torch.where(vals > eps, 1.0 / vals, torch.zeros_like(vals))) @ vecs.T
    return vals, vecs, inv


class ResidualSufficientStatistics:
    """Streaming statistics for the source-space Sylvester solve."""

    def __init__(self) -> None:
        self.s: Tensor | None = None
        self.g: Tensor | None = None
        self.b: Tensor | None = None
        self.sum_a: Tensor | None = None
        self.sum_e: Tensor | None = None
        self.sum_e2 = 0.0
        self.n_rows = 0
        self.m_source: int | None = None
        self.d_source: int | None = None

    def update(self, h: Tensor, e: Tensor, t_in: Tensor, t_out: Tensor) -> None:
        _check_rows(h, e, "h", "e")
        if h.ndim != 2 or e.ndim != 2 or t_in.ndim != 2 or t_out.ndim != 2:
            raise ValueError("activations and transport maps must be matrices")
        if h.shape[1] != t_in.shape[1] or e.shape[1] != t_out.shape[1]:
            raise ValueError("transport maps do not match target activation dimensions")
        tensors = (h, e, t_in, t_out)
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError("solver inputs must be finite")
        # C_target = t_out.T C_source t_in; X = Delta_C_source.T.
        a = h.to(torch.float64) @ t_in.to(torch.float64).T
        lmat = t_out.to(torch.float64).T
        er = e.to(torch.float64)
        s = a.T @ a
        g = lmat.T @ lmat
        b = a.T @ er @ lmat
        sum_a = a.sum(dim=0)
        sum_e = er.sum(dim=0)
        if self.s is None:
            self.s, self.g, self.b = s, g, b
            self.sum_a, self.sum_e = sum_a, sum_e
            self.m_source, self.d_source = a.shape[1], lmat.shape[1]
            self._t_in = t_in.detach().clone()
            self._t_out = t_out.detach().clone()
        else:
            if (s.shape, g.shape, b.shape) != (self.s.shape, self.g.shape, self.b.shape):
                raise ValueError("inconsistent source dimensions across updates")
            if not torch.equal(t_in, self._t_in) or not torch.equal(t_out, self._t_out):
                raise ValueError("transport maps must remain fixed across streaming updates")
            self.s += s
            self.b += b
            self.sum_a += sum_a
            self.sum_e += sum_e
        self.sum_e2 += float((er * er).sum().item())
        self.n_rows += int(h.shape[0])

    def _residual_sq(self, x: Tensor, beta: Tensor, t_out64: Tensor, mu_a: Tensor, mu_e: Tensor) -> float:
        """Exact total ||(A X + 1 beta^T) t_out - E||_F^2, from raw sufficient statistics.

        No centering shortcut is used here: the reduction of the intercept
        optimum to a plain centered sum-of-squares only holds when ``t_out``
        is surjective onto its output space, which does not hold in general
        (e.g. ``d_source < d_target``). This formula is exact for any
        ``t_out`` rank.
        """
        predicted_sq = torch.trace(x.T @ self.s @ x @ self.g).item()
        cross = 2.0 * torch.sum(x * self.b).item()
        resid_no_bias = self.sum_e2 - cross + predicted_sq
        c_vec = t_out64.T @ beta
        pred_mean = mu_a @ x @ t_out64
        n = float(self.n_rows)
        bias_term = 2.0 * n * float((c_vec @ (pred_mean - mu_e)).item()) + n * float((c_vec @ c_vec).item())
        return max(0.0, resid_no_bias + bias_term)

    def solve(self, *, ridge_relative: float, exact_form: bool = True) -> tuple[Tensor, dict[str, Any]]:
        if self.s is None or self.g is None or self.b is None or self.n_rows == 0:
            raise ValueError("cannot solve empty residual statistics")
        if isinstance(ridge_relative, bool) or ridge_relative <= 0 or not math.isfinite(float(ridge_relative)):
            raise ValueError("ridge_relative must be finite and > 0")
        s = (self.s + self.s.T) * 0.5
        g = (self.g + self.g.T) * 0.5
        n = float(self.n_rows)
        mu_a = self.sum_a / n
        mu_e = self.sum_e / n
        t_out64 = self._t_out.to(torch.float64)
        lmat64 = t_out64.T

        if exact_form:
            # Center the sufficient statistics (Eq. 10 pattern) so the ridge-
            # penalized slope solve is unaffected by the residual/feature means.
            sc = s - torch.outer(self.sum_a, self.sum_a) / n
            bc = self.b - torch.outer(self.sum_a, mu_e @ lmat64)
            sc = (sc + sc.T) * 0.5
        else:
            sc, bc = s, self.b

        trace_sc = float(torch.trace(sc).item())
        trace_g = float(torch.trace(g).item())
        base = float(ridge_relative) * trace_sc / float(self.m_source)
        lam = base * (trace_g / float(self.d_source)) if exact_form else base

        es, us, sc_inv = _clamped_eigh_inverse(sc)
        eg, ug, g_inv = _clamped_eigh_inverse(g)

        if trace_sc == 0.0 or trace_g == 0.0:
            x = torch.zeros(self.m_source, self.d_source, dtype=torch.float64)
        else:
            denom = es[:, None] * eg[None, :] + lam
            rhs = us.T @ bc @ ug
            xhat = torch.where(denom > 0, rhs / denom, torch.zeros_like(rhs))
            x = us @ xhat @ ug.T

        if exact_form:
            beta = g_inv @ (t_out64 @ mu_e - g @ (x.T @ mu_a))
        else:
            beta = torch.zeros(self.d_source, dtype=torch.float64)

        residual_sq = self._residual_sq(x, beta, t_out64, mu_a, mu_e)

        # Ridge-free (best possible) reference solve, for the reachable/unreachable split.
        x0 = sc_inv @ bc @ g_inv
        if exact_form:
            beta0 = g_inv @ (t_out64 @ mu_e - g @ (x0.T @ mu_a))
        else:
            beta0 = torch.zeros(self.d_source, dtype=torch.float64)
        best_possible_sq = self._residual_sq(x0, beta0, t_out64, mu_a, mu_e)
        reachable_sq = max(0.0, self.sum_e2 - best_possible_sq)
        unreachable_sq = best_possible_sq

        diag: dict[str, Any] = {
            "n_rows": self.n_rows,
            "ridge": lam,
            "residual_norm_before": self.sum_e2**0.5,
            "residual_norm_after": residual_sq**0.5,
            "reachable_residual_norm": reachable_sq**0.5,
            "unreachable_residual_norm": unreachable_sq**0.5,
            "correction_norm": float(torch.linalg.norm(x).item()),
            "bias_norm": float(torch.linalg.norm(beta).item()),
            "exact_form": exact_form,
            # Source-coordinate c_proj.bias delta, transported exactly like the
            # weight's output side (t_out.T @ bias_correction, matching
            # theseus._transport_bias's `delta_vec @ t_out` convention).
            "bias_correction": beta.to(torch.float32),
        }
        return x.T.to(torch.float32), diag


def fit_cproj_residual(
    h: Tensor, e: Tensor, t_in: Tensor, t_out: Tensor, *, ridge_relative: float, exact_form: bool = True
) -> tuple[Tensor, dict[str, Any]]:
    stats = ResidualSufficientStatistics()
    stats.update(h, e, t_in, t_out)
    return stats.solve(ridge_relative=ridge_relative, exact_form=exact_form)
