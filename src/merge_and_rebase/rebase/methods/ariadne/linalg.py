"""Pure linear-algebra kernels (centered Procrustes / ridge alignment and the
source-space sufficient statistics) used by the Ariadne fits.

The module has no dependency on any run entrypoint; ``eval/target_residual_completion``
re-exports these names for its own importers.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ....utils.cost_accounting import cost_phase_decorator

Tensor = torch.Tensor


@cost_phase_decorator("transformation")
def centered_rectangular_procrustes(
    source_rows: Tensor, target_rows: Tensor, *, eps: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the polar factor of the centered source/target cross-covariance.

    ``source_rows`` is ``[N, d_source]`` and ``target_rows`` is
    ``[N, d_target]``; the returned map is ``[d_source, d_target]``.
    For same-width maps and source-to-target extensions this also minimizes
    ``||X Q - Y||_F`` under the corresponding orthogonality constraint. For
    shrink maps it maximizes cross-covariance alignment, but generally does
    not minimize that least-squares objective because ``||X Q||`` varies with
    the selected source subspace.
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
    q = _procrustes_from_cross(cross)
    return q, src_mean, tgt_mean


@cost_phase_decorator("transformation")
def centered_ridge_alignment(
    source_rows: Tensor, target_rows: Tensor, *, ridge: float | None = None
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """Fit a centered, ridge-regularized linear source-to-target map.

    With no explicit ridge, uses ``trace(X.T @ X) / (N - 1)``. Returns the
    map, row means and compact solver diagnostics. The zero-covariance case
    maps to zero; one-row inputs are accepted and also map to zero.
    """
    _check_rows(source_rows, target_rows, "source_rows", "target_rows")
    if source_rows.shape[0] == 0:
        raise ValueError("ridge alignment requires at least one row")
    if not torch.isfinite(source_rows).all() or not torch.isfinite(target_rows).all():
        raise ValueError("ridge alignment inputs must be finite")
    x, y = source_rows.to(torch.float64), target_rows.to(torch.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    gram, cross = xc.T @ xc, xc.T @ yc
    tr = float(torch.trace(gram).item())
    lam = tr / float(x.shape[0] - 1) if ridge is None and x.shape[0] > 1 else (0.0 if ridge is None else float(ridge))
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("ridge must be finite and >= 0")
    if tr == 0.0:
        mapping = torch.zeros((x.shape[1], y.shape[1]), dtype=x.dtype, device=x.device)
    else:
        mapping = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], dtype=x.dtype, device=x.device), cross)
    return mapping, mx, my, {"ridge": lam, "source_trace": tr}


@cost_phase_decorator("transformation")
def _procrustes_from_cross(cross: Tensor) -> Tensor:
    """Orthogonal Procrustes map from a cross-covariance matrix via SVD."""
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def _check_rows(h: Tensor, e: Tensor, h_name: str, e_name: str) -> None:
    if h.ndim != 2 or e.ndim != 2:
        raise ValueError(f"{h_name} and {e_name} must be rank-2")
    if h.shape[0] != e.shape[0]:
        raise ValueError("activation and residual row counts must match")


# Above this condition number (2-norm, float64), ridge_estimator="none"
# refuses to solve rather than return a numerically meaningless "exact" fit.
# 1e8 is a conventional float64 well-posedness threshold (loses roughly half
# of float64's ~15-16 decimal digits of precision to the solve).
_RIDGE_NONE_CONDITION_THRESHOLD = 1e8


def _clamped_eigh_inverse(sym: Tensor, *, eps: float = 1e-12) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(eigvals_clamped, eigvecs, pseudo_inverse)`` of a symmetric PSD matrix."""
    vals, vecs = torch.linalg.eigh(sym)
    vals = vals.clamp_min(0.0)
    inv = vecs @ torch.diag(torch.where(vals > eps, 1.0 / vals, torch.zeros_like(vals))) @ vecs.T
    return vals, vecs, inv


class ResidualSufficientStatistics:
    """Streaming statistics for the source-space Sylvester solve.

    ``device``, when given, moves every batch's inputs onto it before the
    float64 Gram accumulation in ``update()``: the accumulated statistics
    (and therefore ``solve()``'s ``eigh``/``linalg.solve``) then live on that
    device instead of wherever the caller's activations happened to be. Left
    ``None`` (the default), nothing moves, matching the historical CPU-only
    behaviour exactly -- so every existing caller and test is unaffected.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device
        self.s: Tensor | None = None
        self.g: Tensor | None = None
        self.b: Tensor | None = None
        self.sum_a: Tensor | None = None
        self.sum_e: Tensor | None = None
        self.sum_e2 = 0.0
        self.n_rows = 0
        self.m_source: int | None = None
        self.d_source: int | None = None

    @cost_phase_decorator("transformation")
    def update(self, h: Tensor, e: Tensor, t_in: Tensor | None, t_out: Tensor) -> None:
        """Accumulate one batch.

        ``t_in=None`` means an identity input map: the fitted weight then lives
        in the *target* input coordinates of ``h`` itself and no input-side
        reprojection happens.  This is the ``direct_target`` mode, where there
        is no parameter transport to respect on the input side; it is spelled
        as ``None`` rather than an explicit identity so that the ``d_mlp``-sized
        identity matmul is never materialized (4096x4096 per batch on a
        ViT-L/14 target).  The solve itself is unchanged -- with ``t_in = I``
        the normal equations reduce exactly to ``A = H``.
        """
        _check_rows(h, e, "h", "e")
        if h.ndim != 2 or e.ndim != 2 or t_out.ndim != 2 or (t_in is not None and t_in.ndim != 2):
            raise ValueError("activations and transport maps must be matrices")
        if t_in is not None and h.shape[1] != t_in.shape[1]:
            raise ValueError("transport maps do not match target activation dimensions")
        if e.shape[1] != t_out.shape[1]:
            raise ValueError("transport maps do not match target activation dimensions")
        tensors = (h, e, t_out) if t_in is None else (h, e, t_in, t_out)
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError("solver inputs must be finite")
        if self.device is not None:
            h = h.to(self.device)
            e = e.to(self.device)
            t_in = None if t_in is None else t_in.to(self.device)
            t_out = t_out.to(self.device)
        # C_target = t_out.T C_source t_in; X = Delta_C_source.T.
        a = h.to(torch.float64) if t_in is None else h.to(torch.float64) @ t_in.to(torch.float64).T
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
            self._t_in = None if t_in is None else t_in.detach().clone()
            self._t_out = t_out.detach().clone()
        else:
            if (s.shape, g.shape, b.shape) != (self.s.shape, self.g.shape, self.b.shape):
                raise ValueError("inconsistent source dimensions across updates")
            same_in = (t_in is None) == (self._t_in is None) and (t_in is None or torch.equal(t_in, self._t_in))
            if not same_in or not torch.equal(t_out, self._t_out):
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

    @cost_phase_decorator("transformation")
    def solve(
        self,
        *,
        ridge_relative: float,
        ridge_estimator: str = "fixed_relative",
        exact_form: bool = True,
        ridge_mode: str = "trace_normalized",
        ridge_absolute: float | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if self.s is None or self.g is None or self.b is None or self.n_rows == 0:
            raise ValueError("cannot solve empty residual statistics")
        # ridge_relative is still required to be a finite positive number even
        # under ridge_estimator="none" (which never reads it): the field is
        # shared config-schema surface with fixed_relative/empirical_bayes,
        # and loosening this check for "none" would let a config author write
        # an otherwise-invalid ridge_relative that silently becomes "correct"
        # only because it happens to be paired with "none".
        if isinstance(ridge_relative, bool) or ridge_relative <= 0 or not math.isfinite(float(ridge_relative)):
            raise ValueError("ridge_relative must be finite and > 0")
        if ridge_estimator not in {"fixed_relative", "empirical_bayes", "none"}:
            raise ValueError("ridge_estimator must be 'fixed_relative', 'empirical_bayes' or 'none'")
        if ridge_mode not in {"trace_normalized", "absolute"}:
            raise ValueError("ridge_mode must be 'trace_normalized' or 'absolute'")
        if ridge_mode == "absolute":
            if isinstance(ridge_absolute, bool) or not isinstance(ridge_absolute, (int, float)):
                raise ValueError("ridge_absolute must be a finite real number for absolute ridge")
            if not math.isfinite(float(ridge_absolute)) or float(ridge_absolute) <= 0:
                raise ValueError("ridge_absolute must be finite and > 0 for absolute ridge")
            if ridge_estimator != "fixed_relative":
                raise ValueError("ridge_mode='absolute' is only valid with ridge_estimator='fixed_relative'")
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
        condition_number: float | None = None
        if ridge_estimator == "none":
            # Exact least squares: lambda = 0, straight off the (centered)
            # normal equations -- no shrinkage at all. This is only a
            # well-posed solve when the normal-equations system is actually
            # invertible, so -- unlike the ridge-regularized estimators,
            # which are well-defined even for a singular sc/g via the
            # clamped-eigenvalue pseudo-inverse below -- a singular or
            # numerically ill-conditioned system must fail loudly here rather
            # than silently falling back to that pseudo-inverse's zeroed
            # near-null directions, which would look like a valid exact
            # solve but is not one.
            effective_ridge_relative = 0.0
            base = 0.0
            cond_sc = float(torch.linalg.cond(sc).item()) if sc.shape[0] > 0 else 1.0
            cond_g = float(torch.linalg.cond(g).item()) if g.shape[0] > 0 else 1.0
            condition_number = max(cond_sc, cond_g)
            if not math.isfinite(condition_number) or condition_number > _RIDGE_NONE_CONDITION_THRESHOLD:
                raise ValueError(
                    f"ridge_estimator='none': the normal-equations system is singular or "
                    f"ill-conditioned (condition number {condition_number:.6e} exceeds the "
                    f"{_RIDGE_NONE_CONDITION_THRESHOLD:.0e} threshold for an exact solve); use "
                    "ridge_estimator='fixed_relative' or 'empirical_bayes' instead"
                )
        elif ridge_estimator == "empirical_bayes":
            if self.n_rows <= 1:
                raise ValueError("empirical_bayes ridge requires at least two activation rows")
            # With empirical covariance Sigma_hat = S_c / (N - 1), the
            # empirical-Bayes precision denominator is
            #   S_c + trace(Sigma_hat) I.
            # Therefore the scalar ridge is trace(S_c) / (N - 1), which is
            # the historical relative parameterization evaluated at the
            # component-specific value d_in / (N - 1).
            effective_ridge_relative = float(self.m_source) / float(self.n_rows - 1)
            base = trace_sc / float(self.n_rows - 1)
        else:
            effective_ridge_relative = float(ridge_relative)
            base = effective_ridge_relative * trace_sc / float(self.m_source)
        trace_normalized_lam = base * (trace_g / float(self.d_source)) if exact_form else base
        # ridge_mode="absolute" is validated above to require
        # ridge_estimator="fixed_relative", so this override never fights
        # the "none"/"empirical_bayes" branches above.
        lam = float(ridge_absolute) if ridge_mode == "absolute" else trace_normalized_lam

        es, us, sc_inv = _clamped_eigh_inverse(sc)
        eg, ug, g_inv = _clamped_eigh_inverse(g)

        if trace_sc == 0.0 or trace_g == 0.0:
            x = torch.zeros(self.m_source, self.d_source, dtype=torch.float64, device=t_out64.device)
        else:
            denom = es[:, None] * eg[None, :] + lam
            rhs = us.T @ bc @ ug
            xhat = torch.where(denom > 0, rhs / denom, torch.zeros_like(rhs))
            x = us @ xhat @ ug.T

        if exact_form:
            beta = g_inv @ (t_out64 @ mu_e - g @ (x.T @ mu_a))
        else:
            # Bare torch.zeros defaults to CPU. t_out64/x/mu_a are on whatever
            # device the caller ran on (CUDA for device_transform="gpu"), and
            # _residual_sq below does t_out64.T @ beta -- a CPU/CUDA mismatch
            # that only reduced-form (exact_form=False) fits reach, since the
            # exact-form branch derives beta from GPU tensors already.
            beta = torch.zeros(self.d_source, dtype=torch.float64, device=t_out64.device)

        residual_sq = self._residual_sq(x, beta, t_out64, mu_a, mu_e)

        # Ridge-free (best possible) reference solve, for the reachable/unreachable split.
        x0 = sc_inv @ bc @ g_inv
        if exact_form:
            beta0 = g_inv @ (t_out64 @ mu_e - g @ (x0.T @ mu_a))
        else:
            beta0 = torch.zeros(self.d_source, dtype=torch.float64, device=t_out64.device)
        best_possible_sq = self._residual_sq(x0, beta0, t_out64, mu_a, mu_e)
        reachable_sq = max(0.0, self.sum_e2 - best_possible_sq)
        unreachable_sq = best_possible_sq

        diag: dict[str, Any] = {
            "n_rows": self.n_rows,
            "ridge": lam,
            "ridge_estimator": ridge_estimator,
            # Only populated for ridge_estimator="none" (the estimator that
            # can actually fail on this quantity); None for the two
            # regularized estimators, which are well-posed regardless.
            "condition_number": condition_number,
            "configured_ridge_relative": float(ridge_relative),
            "effective_ridge_relative": effective_ridge_relative,
            "ridge_mode": ridge_mode,
            "ridge_absolute": float(ridge_absolute) if ridge_absolute is not None else None,
            "ridge_trace_normalized": trace_normalized_lam,
            "trace_centered_feature_gram": trace_sc,
            "trace_output_transport_gram": trace_g,
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
