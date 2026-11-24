"""Shared-anchor algebra: model-free, float64, row = example (token), column = hidden dimension.

Notation follows the reference document ("Rebasin universale con ancore condivise"):

- ``P_m = (S_anc - mu_m) / c_m``: the ``T`` anchor rows of model ``m``, centred on the registration mean and
  scaled by ``c_m = ||S_reg - mu_m||_F / sqrt(n)``.
- ``K_m = P_m P_m^T``: the ``T x T`` anchor kernel.
- encode: ``R = (E / c_m) P_m^T`` with ``E`` the UNCENTRED fine-tuning displacement (its mean must reach the
  ridge bias; centring here would zero it).
- decode: ``D = scale * R (K_k + eps I)^{-1} P_k`` with ``eps = eps_rel * tr(K_k) / T`` and ``scale = c_k``
  (``rescale_target``) or ``c_m``.

When ``T > d`` the kernel is singular and ``T x T`` work is wasteful, so the ``*_primal`` / ``*_from_frames``
functions use the push-through identity ``(P P^T + eps I)^{-1} P = P (P^T P + eps I)^{-1}`` and the ``d x d`` Gram
``G = P^T P`` (``tr G = tr K``; the nonzero spectra of ``G`` and ``K`` coincide). They are exact rewrites, not
approximations.

Everything here is linear in ``E`` / ``R`` row by row, so it commutes with any row interpolation (token grids).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch

Tensor = torch.Tensor


class MomentAccumulator:
    """Streaming mean and total scatter ``sum_i ||x_i - mu||^2`` of ``[N, d]`` rows (Chan's parallel merge, float64).

    The residual stream has a large common mean, so the naive ``sum ||x||^2 - n ||mu||^2`` would cancel
    catastrophically; Chan's merge keeps the deviation sum exact up to rounding.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean: Tensor | None = None
        self.m2 = 0.0

    def update(self, rows: Tensor) -> None:
        x = rows.reshape(-1, rows.shape[-1]).to(torch.float64)
        nb = int(x.shape[0])
        if nb == 0:
            return
        mb = x.mean(0)
        m2b = float(((x - mb) ** 2).sum().item())
        if self.mean is None:
            self.n, self.mean, self.m2 = nb, mb, m2b
            return
        n = self.n + nb
        delta = mb - self.mean
        self.mean = self.mean + delta * (nb / n)
        self.m2 += m2b + float((delta @ delta).item()) * self.n * nb / n
        self.n = n

    def finalize(self) -> tuple[Tensor, float]:
        """``(mu [d], c)`` with ``c = sqrt(scatter / n)``, the typical norm of a centred row."""
        if self.mean is None or self.n == 0:
            raise ValueError("MomentAccumulator saw no rows")
        c = (self.m2 / self.n) ** 0.5
        if not c > 0.0:
            raise ValueError("registration rows have zero spread (c == 0)")
        return self.mean.clone(), float(c)


def anchor_frame(anchor_rows: Tensor, mu: Tensor, c: float) -> Tensor:
    """``P = (S_anc - mu) / c`` in float64 (``[T, d]``)."""
    return (anchor_rows.to(torch.float64) - mu.to(torch.float64)) / float(c)


def kernel(p: Tensor) -> Tensor:
    """Anchor kernel ``K = P P^T`` (``[T, T]``, float64, exactly symmetrised)."""
    p64 = p.to(torch.float64)
    k = p64 @ p64.T
    return 0.5 * (k + k.T)


def resolve_eps(k: Tensor, eps_rel: float) -> float:
    """``eps = eps_rel * tr(K) / T``: ``eps_rel`` times the mean self-similarity of an anchor."""
    return float(eps_rel) * float(torch.trace(k.to(torch.float64)).item()) / int(k.shape[0])


def encode_rows(e: Tensor, p_m: Tensor, c_m: float) -> Tensor:
    """``R = (E / c_m) P_m^T`` (``[n, T]``, float64). ``E`` is NOT centred (see module docstring)."""
    return (e.to(torch.float64) / float(c_m)) @ p_m.to(torch.float64).T


def decode_weights(r: Tensor, k_k: Tensor, eps: float) -> Tensor:
    """``W = R (K_k + eps I)^{-1}`` via a linear solve (never an explicit inverse); float64 ``[n, T]``."""
    k64 = k_k.to(torch.float64)
    a = k64 + float(eps) * torch.eye(k64.shape[0], dtype=torch.float64, device=k64.device)
    # K is symmetric, so R A^{-1} = (A^{-1} R^T)^T.
    return torch.linalg.solve(a, r.to(torch.float64).T).T


def decode_rows(r: Tensor, p_k: Tensor, k_k: Tensor, *, scale: float, eps: float) -> Tensor:
    """``D = scale * R (K_k + eps I)^{-1} P_k`` (``[n, d_k]``, float64)."""
    return float(scale) * (decode_weights(r, k_k, eps) @ p_k.to(torch.float64))


def round_trip_operator(p: Tensor, k: Tensor, eps: float) -> Tensor:
    """``M = P^T (K + eps I)^{-1} P`` (``[d, d]``): encode then decode in the same model maps ``e -> e M``.

    With ``eps = 0`` and linearly independent anchors this is the orthogonal projector onto span(P).
    """
    p64 = p.to(torch.float64)
    return decode_weights(p64.T, k, eps) @ p64


@dataclass
class CoverageAccumulator:
    """Streaming coverage of a displacement by the anchors of its own model.

    Accumulates the first and second moments of ``E`` (``sum e``, ``E^T E``), so the centred scatter
    ``C = E_c^T E_c`` is exact, and ``coverage = 1 - ||E_c (I - M)||^2 / ||E_c||^2`` with ``M`` the
    round-trip operator. Means are removed so only the token-varying part is scored (the mean reaches the
    bias through the ridge, not through the anchors).
    """

    n: int = 0
    total: Tensor | None = None
    gram: Tensor | None = None

    def update(self, e: Tensor) -> None:
        x = e.reshape(-1, e.shape[-1]).to(torch.float64)
        if self.total is None:
            self.total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
            self.gram = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
        self.n += int(x.shape[0])
        self.total += x.sum(0)
        self.gram += x.T @ x

    def centred_scatter(self) -> Tensor:
        if self.total is None or self.n == 0:
            raise ValueError("CoverageAccumulator saw no rows")
        return self.gram - torch.outer(self.total, self.total) / self.n

    def coverage(self, p: Tensor, k: Tensor, eps: float) -> dict[str, float]:
        return self.coverage_from_operator(round_trip_operator(p, k, eps))

    def coverage_from_operator(self, m: Tensor) -> dict[str, float]:
        """Coverage for a given round-trip operator ``M`` (``[d, d]``)."""
        c = self.centred_scatter()
        m = m.to(c)
        resid = torch.eye(m.shape[0], dtype=c.dtype, device=c.device) - m
        energy = float(torch.trace(c).item())
        lost = float(torch.trace(resid.T @ c @ resid).item())
        kept = float(torch.trace(m.T @ c @ m).item())
        if energy <= 0.0:
            return {"coverage": float("nan"), "reconstructed_energy_ratio": float("nan"), "energy": energy}
        return {"coverage": 1.0 - lost / energy, "reconstructed_energy_ratio": kept / energy, "energy": energy}


def kernel_diagnostics(k_m: Tensor, k_k: Tensor) -> dict[str, float]:
    """Linear CKA of the centred kernels (``H = I - 11^T/T``) and ``||K_m - K_k||_F / ||K_m||_F``."""
    a = k_m.to(torch.float64)
    b = k_k.to(torch.float64)
    if a.shape != b.shape:
        raise ValueError(f"kernels must share the anchor set, got {tuple(a.shape)} vs {tuple(b.shape)}")
    t = a.shape[0]
    h = torch.eye(t, dtype=torch.float64, device=a.device) - 1.0 / t
    ac = h @ a @ h
    bc = h @ b @ h
    denom = float(torch.linalg.norm(ac).item()) * float(torch.linalg.norm(bc).item())
    cka = float((ac * bc).sum().item()) / denom if denom > 0.0 else float("nan")
    rel_err = float(torch.linalg.norm(a - b).item()) / float(torch.linalg.norm(a).item())
    return {"cka": cka, "rel_err": rel_err}


def kernel_spectrum(k: Tensor) -> dict[str, float]:
    """Conditioning of an anchor kernel: condition number and participation-ratio effective rank."""
    ev = torch.linalg.eigvalsh(k.to(torch.float64)).clamp_min(0.0)
    top = float(ev.max().item())
    low = float(ev.min().item())
    s1 = float(ev.sum().item())
    s2 = float((ev**2).sum().item())
    return {
        "condition_number": top / low if low > 0.0 else float("inf"),
        "effective_rank": (s1 * s1 / s2) if s2 > 0.0 else 0.0,
        "trace": s1,
        "numerical_rank": int((ev > ev.max() * ev.numel() * torch.finfo(torch.float64).eps).sum().item()),
    }


# ---------------------------------------------------------------- primal (d x d) forms, for T > d


def gram(p: Tensor) -> Tensor:
    """``G = P^T P`` (``[d, d]``, float64, exactly symmetrised)."""
    p64 = p.to(torch.float64)
    g = p64.T @ p64
    return 0.5 * (g + g.T)


def resolve_eps_from_frame(p: Tensor, eps_rel: float) -> float:
    """Same ``eps`` as ``resolve_eps(kernel(p), eps_rel)``: ``tr K = ||P||_F^2``."""
    p64 = p.to(torch.float64)
    return float(eps_rel) * float((p64 * p64).sum().item()) / int(p64.shape[0])


def transport_map_primal(p_m: Tensor, c_m: float, p_k: Tensor, *, scale: float, eps: float) -> Tensor:
    """``Phi = (scale / c_m) P_m^T P_k (P_k^T P_k + eps I)^{-1}`` (``[d_m, d_k]``): ``D = E Phi``.

    Equal to encode followed by decode (``scale * (E / c_m) P_m^T (K_k + eps I)^{-1} P_k``) for ``eps > 0``.
    """
    pm = p_m.to(torch.float64)
    pk = p_k.to(torch.float64)
    a = gram(pk) + float(eps) * torch.eye(pk.shape[1], dtype=torch.float64, device=pk.device)
    cross = pm.T @ pk
    # A is symmetric, so cross A^{-1} = (A^{-1} cross^T)^T.
    return (float(scale) / float(c_m)) * torch.linalg.solve(a, cross.T).T


def round_trip_operator_primal(p: Tensor, eps: float) -> Tensor:
    """``M = G (G + eps I)^{-1}``, equal to ``round_trip_operator`` (``G`` and ``(G + eps I)^{-1}`` commute)."""
    g = gram(p)
    a = g + float(eps) * torch.eye(g.shape[0], dtype=torch.float64, device=g.device)
    return torch.linalg.solve(a, g)


def kernel_diagnostics_from_frames(p_m: Tensor, p_k: Tensor) -> dict[str, float]:
    """``kernel_diagnostics(kernel(p_m), kernel(p_k))`` through ``d x d`` products only.

    With ``A = H P_m`` and ``B = H P_k`` (anchors centred): ``<H K_m H, H K_k H> = ||A^T B||_F^2`` and
    ``||K_m - K_k||_F^2 = ||P_m^T P_m||^2 + ||P_k^T P_k||^2 - 2 ||P_m^T P_k||^2``.
    """
    pm = p_m.to(torch.float64)
    pk = p_k.to(torch.float64)
    if pm.shape[0] != pk.shape[0]:
        raise ValueError(f"frames must share the anchor set, got {pm.shape[0]} vs {pk.shape[0]} anchors")
    a = pm - pm.mean(0, keepdim=True)
    b = pk - pk.mean(0, keepdim=True)
    ab = float(torch.linalg.norm(a.T @ b).item()) ** 2
    aa = float(torch.linalg.norm(a.T @ a).item())
    bb = float(torch.linalg.norm(b.T @ b).item())
    cka = ab / (aa * bb) if aa * bb > 0.0 else float("nan")
    mm = float(torch.linalg.norm(pm.T @ pm).item()) ** 2
    kk = float(torch.linalg.norm(pk.T @ pk).item()) ** 2
    mk = float(torch.linalg.norm(pm.T @ pk).item()) ** 2
    rel_err = max(mm + kk - 2.0 * mk, 0.0) ** 0.5 / mm**0.5
    return {"cka": cka, "rel_err": rel_err}


def kernel_spectrum_from_frame(p: Tensor) -> dict[str, float]:
    """``kernel_spectrum(kernel(p))`` from the smaller Gram; ``T - d`` zero eigenvalues are added when ``T > d``."""
    p64 = p.to(torch.float64)
    t, d = p64.shape
    ev = torch.linalg.eigvalsh(gram(p64) if t > d else kernel(p64)).clamp_min(0.0)
    if t > d:
        ev = torch.cat([torch.zeros(t - d, dtype=ev.dtype, device=ev.device), ev])
    top = float(ev.max().item())
    low = float(ev.min().item())
    s1 = float(ev.sum().item())
    s2 = float((ev**2).sum().item())
    return {
        "condition_number": top / low if low > 0.0 else float("inf"),
        "effective_rank": (s1 * s1 / s2) if s2 > 0.0 else 0.0,
        "trace": s1,
        "numerical_rank": int((ev > ev.max() * ev.numel() * torch.finfo(torch.float64).eps).sum().item()),
    }


# ---------------------------------------------------------------- spectral filters (alternatives to Tikhonov eps)

SPECTRAL_FILTERS = ("tikhonov", "truncate_target", "truncate_cross", "procrustes_cross")


def spectral_transport_map(
    p_m: Tensor, c_m: float, p_k: Tensor, *, scale: float, mode: str, rank: int | None
) -> tuple[Tensor, dict[str, float]]:
    """``Phi`` (``[d_m, d_k]``, ``D = E Phi``) with a hard spectral cut instead of Tikhonov's ``1/(lambda + eps)``.

    With ``C = P_m^T P_k = U S W^T`` and ``G_k = P_k^T P_k = V Lambda V^T`` (eigenpairs in decreasing order):

    - ``truncate_target``: ``(scale / c_m) C V_r Lambda_r^{-1} V_r^T``. At full rank this is the ``eps -> 0``
      Tikhonov decode (pseudo-inverse of the target kernel); the kept directions are not shrunk.
    - ``truncate_cross``: ``(scale / c_m) U_r S_r W_r^T / gbar``, ``gbar = tr(G_k) / d_k``: the ``eps -> inf``
      Tikhonov limit (``C / eps``) restricted to the top-``r`` shared directions, at ``eps = gbar``.
    - ``procrustes_cross``: ``(scale / c_m) U_r W_r^T``: an orthogonal (partial isometry) map between the top-``r``
      shared directions, i.e. a Procrustes estimated on the anchor rows only.

    ``rank`` None keeps every direction with a nonzero eigen/singular value.
    """
    if mode not in SPECTRAL_FILTERS[1:]:
        raise ValueError(f"spectral mode must be one of {SPECTRAL_FILTERS[1:]}, got {mode!r}")
    pm = p_m.to(torch.float64)
    pk = p_k.to(torch.float64)
    cross = pm.T @ pk
    tol = torch.finfo(torch.float64).eps
    factor = float(scale) / float(c_m)
    if mode == "truncate_target":
        lam, v = torch.linalg.eigh(gram(pk))
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
        gbar = float(torch.trace(gram(pk)).item()) / int(pk.shape[1])
        phi = factor * (u[:, :r] * sv[:r]) @ wt[:r] / gbar
    else:
        phi = factor * u[:, :r] @ wt[:r]
    return phi, {"spectral_rank_used": float(r), "spectral_kept_fraction": kept}


# ---------------------------------------------------------------- anchor selection / normalisation (AGENT_KERNEL.md R2, R3)


def row_normalize(p: Tensor) -> Tensor:
    """R3: every (centred) anchor row scaled to unit norm; zero rows stay zero. ``E`` is never normalised."""
    p64 = p.to(torch.float64)
    norms = torch.linalg.vector_norm(p64, dim=1, keepdim=True)
    return torch.where(norms > 0, p64 / norms.clamp_min(torch.finfo(torch.float64).tiny), p64)


def dedup_mask(p: Tensor, cos_max: float, *, max_count: int | None = None, seed: int = 0) -> Tensor:
    """R2: greedy selection of anchors whose pairwise cosine (centred rows of ``p``) stays below ``cos_max``.

    Anchors are visited in a seeded random order; one is kept when its cosine with every anchor kept so far is
    ``< cos_max``. Stops after ``max_count`` kept anchors (None: visit all). Returns a bool mask over the rows.
    """
    x = row_normalize(p).to(torch.float32)
    t = x.shape[0]
    order = torch.randperm(t, generator=torch.Generator().manual_seed(int(seed)))
    keep = torch.zeros(t, dtype=torch.bool)
    kept = torch.empty(0, x.shape[1], dtype=x.dtype)
    limit = t if max_count is None else int(max_count)
    for i in order.tolist():
        xi = x[i]
        if kept.shape[0] and float((kept @ xi).max()) >= float(cos_max):
            continue
        keep[i] = True
        kept = torch.cat([kept, xi[None]], dim=0)
        if kept.shape[0] >= limit:
            break
    return keep


# ---------------------------------------------------------------- whitened kernel decodes (AGENT_KERNEL.md V2, V3)

#: Decoders of the unified canonical family (``canonical_transport_map``): F = Y_r Z_r^T (hard) / Y g(Gamma) Z^T (soft).
CANONICAL_DECODES = ("canonical_hard", "canonical_soft")
KERNEL_DECODES = ("tikhonov", "zca", "agreement", *CANONICAL_DECODES)
_WHITENED_DECODES = ("zca", "agreement")  # the modes of whitened_transport_map
PINV_RTOL = 1e-10  # relative eigenvalue threshold of the pseudo-inverses (eps = 0)


def _whitening_factors(p: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Thin SVD ``P = U S V^T`` and ``f = s / sqrt(s^2 + eps)``, so that ``(K + eps I)^{-1/2} P = U diag(f) V^T``.

    With ``eps = 0`` the inverse root is the pseudo-inverse one: ``f = 1`` on the eigenvalues ``s^2`` above
    ``PINV_RTOL * s_max^2``, 0 below. Returns ``(U, s, f, V^T)``.
    """
    u, s, vt = torch.linalg.svd(p.to(torch.float64), full_matrices=False)
    lam = s * s
    if eps > 0.0:
        f = s / torch.sqrt(lam + float(eps))
    else:
        f = (lam > PINV_RTOL * lam[0]).to(torch.float64)
    return u, s, f, vt


def whitened_transport_map(
    p_m: Tensor,
    c_m: float,
    p_k: Tensor,
    *,
    scale: float,
    eps_rel: float,
    mode: str,
    tau: float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """``Phi`` (``[d_m, d_k]``, ``D = E Phi``) of the whitened kernel decodes; each model whitened by its own kernel.

    ``Q_m = (K_m + eps_m I)^{-1/2} P_m`` with ``eps_m = eps_rel tr(K_m) / T`` (pseudo-inverse root at ``eps_rel = 0``).

    - ``zca`` (V2): ``Phi = (scale / c_m) Q_m^T Q_k``: encode ``R~ = (E / c_m) Q_m^T``, decode ``D = scale R~ Q_k``.
      At ``eps = 0`` ``Q_m^T`` and ``Q_k`` are partial isometries, so ``||D|| / scale <= ||E|| / c_m``; with
      ``K_m = K_k`` it equals the Tikhonov decode at the same ``eps``.
    - ``agreement`` (V3): ``M = (K_k + eps_k I)^{-1/2} K_m (K_k + eps_k I)^{-1/2} = W Gamma W^T``; only the directions
      with ``1/tau <= gamma_i <= tau`` (``W_S``) are kept: ``Phi = (scale / c_m) Q_m^T W_S W_S^T Q_k``. Built with
      ``T x T`` matrices; with ``T > d`` the directions outside the range of ``M`` (``gamma = 0``) are never kept.
    """
    if mode not in _WHITENED_DECODES:
        raise ValueError(f"whitened decode must be one of {_WHITENED_DECODES}, got {mode!r}")
    pm = p_m.to(torch.float64)
    pk = p_k.to(torch.float64)
    t = pm.shape[0]
    if pk.shape[0] != t:
        raise ValueError(f"frames must share the anchor set, got {t} vs {pk.shape[0]} anchors")
    eps_m = resolve_eps_from_frame(pm, eps_rel)
    eps_k = resolve_eps_from_frame(pk, eps_rel)
    u_m, _, f_m, vt_m = _whitening_factors(pm, eps_m)
    u_k, s_k, f_k, vt_k = _whitening_factors(pk, eps_k)
    factor = float(scale) / float(c_m)
    info = {"kernel_eps_source": eps_m, "kernel_eps_target": eps_k}
    if mode == "zca":
        core = (u_m * f_m).T @ (u_k * f_k)  # diag(f_m) U_m^T U_k diag(f_k)
        phi = factor * vt_m.T @ core @ vt_k
        info["kernel_map_spectral_norm"] = float(torch.linalg.matrix_norm(core, ord=2).item())
        return phi, info
    if tau is None or not float(tau) >= 1.0:
        raise ValueError("agreement decode needs tau >= 1")
    lam_k = s_k * s_k
    if eps_k > 0.0:
        inv_root = 1.0 / torch.sqrt(lam_k + eps_k)
    else:
        keep_k = lam_k > PINV_RTOL * lam_k[0]
        inv_root = torch.where(keep_k, 1.0 / torch.sqrt(lam_k.clamp_min(1e-300)), torch.zeros_like(lam_k))
    w_k = (u_k * inv_root) @ u_k.T
    if eps_k > 0.0:  # the null space of K_k is whitened by eps_k^{-1/2}
        w_k = w_k + (torch.eye(t, dtype=torch.float64) - u_k @ u_k.T) / eps_k**0.5
    a = w_k @ pm  # (K_k + eps_k I)^{-1/2} P_m, so M = a a^T
    gamma, w = torch.linalg.eigh(0.5 * ((a @ a.T) + (a @ a.T).T))
    sel = (gamma >= 1.0 / float(tau)) & (gamma <= float(tau))
    ws = w[:, sel]
    q_m = (u_m * f_m) @ vt_m
    q_k = (u_k * f_k) @ vt_k
    phi = factor * (q_m.T @ ws) @ (ws.T @ q_k)
    pos = gamma[gamma > PINV_RTOL * gamma.abs().max()]
    info.update(
        {
            "agreement_kept": float(sel.sum().item()),
            "agreement_nonzero": float(pos.numel()),
            "agreement_gamma_median": float(pos.median().item()) if pos.numel() else float("nan"),
            "agreement_gamma_in_band_fraction": float(sel.sum().item()) / max(int(pos.numel()), 1),
        }
    )
    return phi, info


# ---------------------------------------------------------------- unified canonical decoder (PIANO_METODO_K section 1)
#
# With P_m = U_m S_m V_m^T (thin SVD, numerical rank r_m) and M = U_m^T U_k = Y Gamma Z^T, the canonical correlations
# Gamma are the singular values of Sigma_mm^(-1/2) Sigma_mk Sigma_kk^(-1/2) (Sigma_ab = P_a^T P_b, NOT divided by T).
# Every pairwise decode is Phi = (scale / c_m) V_m F V_k^T with a different [r_m, r_k] core F:
#   V0 Tikhonov  F = S_m M S_k (S_k^2 + eps)^-1        ZCA (eps = 0)   F = M
#   cross-cov.   F = S_m M S_k / eps                   Procrustes      F = polar(S_m M S_k)
#   canonical_hard  F = Y_r Z_r^T                      canonical_soft  F = Y g(Gamma) Z^T,  g(G) = G * (G >= threshold)
# V_m, V_k and M may also be built from Sigma alone (streaming statistics, canonical_stats.py): V S^2 V^T = eigh(Sigma_mm),
# M = S_m^-1 V_m^T Sigma_mk V_k S_k^-1.
# Degenerate case: inside R^T the column spaces span(U_m) and span(U_k) intersect in at least r_m + r_k - T dimensions,
# so for T < r_m + r_k at least that many canonical correlations equal 1 from noise alone (forced_unit_pairs); at
# T <= max(d) M is orthogonal and Gamma == 1 identically (hard == soft == ZCA). The canonical decoders are refused for
# T <= r_m + r_k (numerical ranks; widths d_m + d_k where the ranks are not known yet).


@dataclass(frozen=True)
class CanonicalFactors:
    """``V_m [d_m, r_m]``, ``S_m [r_m]``, ``V_k``, ``S_k`` (decreasing, numerical rank) and ``M = U_m^T U_k`` ``[r_m, r_k]``."""

    v_m: Tensor
    s_m: Tensor
    v_k: Tensor
    s_k: Tensor
    m: Tensor
    n_rows: int


def forced_unit_pairs(n_rows: int, r_m: int, r_k: int) -> int:
    """Canonical correlations forced to 1 by the geometry alone: ``max(0, r_m + r_k - T)``."""
    return max(0, int(r_m) + int(r_k) - int(n_rows))


def check_canonical_widths(n_rows: int, r_m: int, r_k: int) -> None:
    """Refuse ``T <= r_m + r_k`` (``r`` = numerical rank of a frame; pass the widths ``d_m``, ``d_k`` where only
    those are known): the two column spaces of a ``T``-row frame pair then share at least ``r_m + r_k - T`` dimensions
    (``T = r_m + r_k`` leaves none to spare and the sample estimate is saturated), so those canonical correlations are 1
    from noise alone and the decoder is degenerate (== zca for ``T <= max(d)``)."""
    if int(n_rows) <= int(r_m) + int(r_k):
        raise ValueError(
            f"canonical decoders need more anchors than the sum of the frame ranks: T={int(n_rows)} <= "
            f"r_m + r_k = {int(r_m)} + {int(r_k)}. Then at least {forced_unit_pairs(n_rows, r_m, r_k)} canonical "
            "correlations are 1 from noise alone (the column spaces of the frames intersect in R^T) and "
            "canonical_hard / canonical_soft are degenerate; raise num_anchors above r_m + r_k"
        )


def canonical_factors_from_frames(p_m: Tensor, p_k: Tensor, *, check: bool = True) -> CanonicalFactors:
    """Factors from the anchor frames ``P_m [T, d_m]``, ``P_k [T, d_k]`` (thin SVDs; directions below
    ``PINV_RTOL * s_max^2`` are dropped, like the pseudo-inverse of the whitened decodes)."""
    pm = p_m.to(torch.float64)
    pk = p_k.to(torch.float64)
    t = int(pm.shape[0])
    if int(pk.shape[0]) != t:
        raise ValueError(f"frames must share the anchor set, got {t} vs {pk.shape[0]} anchors")
    u_m, s_m, vt_m = torch.linalg.svd(pm, full_matrices=False)
    u_k, s_k, vt_k = torch.linalg.svd(pk, full_matrices=False)
    keep_m = s_m * s_m > PINV_RTOL * s_m[0] ** 2
    keep_k = s_k * s_k > PINV_RTOL * s_k[0] ** 2
    if check:
        check_canonical_widths(t, int(keep_m.sum()), int(keep_k.sum()))
    m = u_m[:, keep_m].T @ u_k[:, keep_k]
    return CanonicalFactors(vt_m[keep_m].T, s_m[keep_m], vt_k[keep_k].T, s_k[keep_k], m, t)


def canonical_factors_from_sigma(
    sigma_mm: Tensor, sigma_kk: Tensor, sigma_mk: Tensor, n_rows: int, *, check: bool = True
) -> CanonicalFactors:
    """Same factors from ``Sigma_mm = P_m^T P_m``, ``Sigma_kk``, ``Sigma_mk = P_m^T P_k`` (``eigh``; precision of the
    small directions is that of the Gram, ``eps_mach * lambda_max / lambda_i``)."""
    smm = sigma_mm.to(torch.float64)
    skk = sigma_kk.to(torch.float64)
    smk = sigma_mk.to(torch.float64)
    lam_m, v_m = torch.linalg.eigh(0.5 * (smm + smm.T))
    lam_k, v_k = torch.linalg.eigh(0.5 * (skk + skk.T))
    lam_m, v_m, lam_k, v_k = lam_m.flip(0), v_m.flip(1), lam_k.flip(0), v_k.flip(1)
    keep_m = lam_m > PINV_RTOL * lam_m[0]
    keep_k = lam_k > PINV_RTOL * lam_k[0]
    if check:  # the rule applies to whatever rows the Sigma were accumulated from (n_rows of a half for a half)
        check_canonical_widths(n_rows, int(keep_m.sum()), int(keep_k.sum()))
    v_m, v_k = v_m[:, keep_m], v_k[:, keep_k]
    s_m, s_k = lam_m[keep_m].sqrt(), lam_k[keep_k].sqrt()
    m = (v_m.T @ smk @ v_k) / s_m[:, None] / s_k[None, :]
    return CanonicalFactors(v_m, s_m, v_k, s_k, m, int(n_rows))


def _svd_canonical(m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Thin SVD of the canonical matrix ``M``. Default: ``torch.linalg.svd`` (unchanged results whenever it converges).
    LAPACK's divide-and-conquer driver (gesdd) can fail to converge when ``M`` has many (near-)repeated singular values,
    e.g. a same-model round trip where every canonical correlation is ~1; then the QR-iteration driver (gesvd, via
    scipy, float64 on CPU) is used. A non-finite ``M`` is refused rather than decomposed."""
    try:
        return torch.linalg.svd(m, full_matrices=False)
    except torch.linalg.LinAlgError:
        if not bool(torch.isfinite(m).all()):
            raise ValueError("canonical matrix M has non-finite entries; refusing to decompose it") from None
        import scipy.linalg

        u, s, vt = scipy.linalg.svd(m.detach().cpu().to(torch.float64).numpy(), full_matrices=False,
                                    lapack_driver="gesvd")
        warnings.warn("torch.linalg.svd (gesdd) did not converge on the canonical matrix; used LAPACK gesvd",
                      RuntimeWarning, stacklevel=2)
        return tuple(torch.from_numpy(x).to(device=m.device, dtype=m.dtype) for x in (u, s, vt))


def canonical_correlations(factors: CanonicalFactors) -> Tensor:
    """Gamma (decreasing, ``min(r_m, r_k)`` values): singular values of ``M``. Per-block input of the diagnostics
    (Gamma on a held-out half, r* against a null)."""
    try:
        return torch.linalg.svdvals(factors.m)
    except torch.linalg.LinAlgError:
        return _svd_canonical(factors.m)[1]


_GAMMA_OUT_RTOL = 1e-12  # relative tolerance of the half-B variance check in canonical_correlations_out


def canonical_correlations_out(
    factors_a: CanonicalFactors, sigma_b: tuple[Tensor, Tensor, Tensor]
) -> Tensor:
    """Held-out canonical correlations ``Gamma_out`` (one per pair, in the order of ``canonical_correlations``).

    The canonical directions are estimated on half A (``factors_a``: ``M = Y Gamma Z^T``, ``a_i = V_m S_m^-1 y_i``,
    ``b_i = V_k S_k^-1 z_i`` in feature space) and their correlation is measured on half B from B's
    ``(Sigma_mm, Sigma_kk, Sigma_mk)`` (``PairSigmaAccumulator.sigma(1)[:3]``):
    ``Gamma_out,i = a_i^T Sigma_mk^B b_i / sqrt(a_i^T Sigma_mm^B a_i * b_i^T Sigma_kk^B b_i)``.
    Only half-B quantities enter (half B may have a different size; positive rescaling of its Sigma leaves the result
    unchanged). Raises ``ValueError`` if half B has no variance along a direction of half A (rank-deficient half B).
    Signed (a direction whose correlation flips sign on B is negative); with half B = half A it equals Gamma. A caller
    that wants ``canonical_soft`` weights thresholds it (``g = Gamma_out * (Gamma_out >= thr)``) and passes ``weights``.
    """
    smm, skk, smk = (x.to(torch.float64) for x in sigma_b)
    y, _, zt = _svd_canonical(factors_a.m)
    a = factors_a.v_m @ (y / factors_a.s_m[:, None])  # [d_m, q]
    b = factors_a.v_k @ (zt.T / factors_a.s_k[:, None])  # [d_k, q]
    num = (a * (smk @ b)).sum(0)
    var_a = (a * (smm @ a)).sum(0)
    var_b = (b * (skk @ b)).sum(0)
    # every denominator is a half-B quantity (the result is invariant to a positive rescaling of half B's Sigma); a
    # direction of half A with no variance on half B means half B is rank-deficient there: refuse, never nan/inf
    tol_a = _GAMMA_OUT_RTOL * float(torch.trace(smm).abs()) * (a * a).sum(0)
    tol_b = _GAMMA_OUT_RTOL * float(torch.trace(skk).abs()) * (b * b).sum(0)
    if not bool(((var_a > tol_a) & (var_b > tol_b)).all()):
        raise ValueError(
            "canonical_correlations_out: half B is rank-deficient along a canonical direction of half A "
            "(a^T Sigma_mm^B a or b^T Sigma_kk^B b <= tolerance); Gamma_out is undefined, use a larger half B"
        )
    return num / (var_a * var_b).sqrt()


def canonical_core(
    factors: CanonicalFactors,
    mode: str,
    *,
    rank: int | None = None,
    null_threshold: float | Tensor = 0.0,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """``(F [r_m, r_k], Gamma, info)`` of ``canonical_hard`` (``F = Y_r Z_r^T``, ``rank`` None = every pair) or
    ``canonical_soft`` (``F = Y g Z^T``). Default ``g(Gamma) = Gamma`` where ``Gamma >= null_threshold`` else 0 (in-sample;
    the threshold is a float or a tensor with one entry per canonical pair). With ``weights`` (``soft`` only, one entry
    per canonical pair, e.g. the held-out ``canonical_correlations_out`` thresholded by the caller) ``g = weights`` as
    given and ``null_threshold`` must be left at its default. ``weights`` must be >= 0: a negative ``g_i`` would make
    ``F = Y g Z^T`` anti-transport that direction (ValueError). A caller using the held-out ``Gamma_out`` maps it with
    ``g = Gamma_out * (Gamma_out >= thr)``, ``thr >= 0``, so a sign flip between the halves gives 0.

    ``info['canonical_soft_weights']`` (soft only) is ``'explicit'`` with ``weights`` and ``'in_sample'`` otherwise; the
    in-sample soft decode (``g`` from the same rows that fit the directions) is NOT plan decoder D3 (held-out
    ``Gamma_out``) and emits a ``UserWarning``."""
    if mode not in CANONICAL_DECODES:
        raise ValueError(f"canonical decode must be one of {CANONICAL_DECODES}, got {mode!r}")
    y, gamma, zt = _svd_canonical(factors.m)
    q = int(gamma.numel())
    if mode == "canonical_hard":
        r = q if rank is None else min(int(rank), q)
        if r < 1:
            raise ValueError("canonical rank must be >= 1")
        f = y[:, :r] @ zt[:r]
        used = float(r)
    elif weights is not None:
        if not (isinstance(null_threshold, (int, float)) and float(null_threshold) == 0.0):
            raise ValueError("weights and null_threshold are mutually exclusive")
        g = torch.as_tensor(weights, dtype=torch.float64)
        if tuple(g.shape) != (q,):
            raise ValueError(f"weights must have one entry per canonical pair ({q}), got {tuple(g.shape)}")
        if bool((g < 0).any()):
            raise ValueError(
                "weights must be >= 0 (a negative weight anti-transports that direction); map Gamma_out with "
                "g = Gamma_out * (Gamma_out >= thr), thr >= 0"
            )
        f = (y * g) @ zt
        used = float((g != 0).sum().item())
    else:
        warnings.warn(
            "canonical_soft with in-sample weights g(Gamma) (Gamma estimated on the rows that fit the directions) is "
            "NOT plan decoder D3, which needs the held-out Gamma_out (canonical_correlations_out)",
            UserWarning,
            stacklevel=2,
        )
        thr = torch.as_tensor(null_threshold, dtype=torch.float64)
        if thr.ndim > 0 and tuple(thr.shape) != (q,):
            raise ValueError(f"null_threshold array must have one entry per canonical pair ({q}), got {tuple(thr.shape)}")
        gate = gamma >= thr
        f = (y * (gamma * gate)) @ zt
        used = float(gate.sum().item())
    if weights is not None and mode != "canonical_soft":
        raise ValueError("weights are allowed with canonical_soft only")
    info = {
        "canonical_rank_source": float(factors.s_m.numel()),
        "canonical_rank_target": float(factors.s_k.numel()),
        "canonical_n_rows": float(factors.n_rows),
        "forced_unit_pairs": float(forced_unit_pairs(factors.n_rows, factors.s_m.numel(), factors.s_k.numel())),
        "canonical_pairs_total": float(q),
        "canonical_pairs_used": used,
        "canonical_gamma_max": float(gamma[0].item()),
        "canonical_gamma_min": float(gamma[-1].item()),
        "canonical_gamma_mean": float(gamma.mean().item()),
    }
    if mode == "canonical_soft":
        info["canonical_soft_weights"] = "explicit" if weights is not None else "in_sample"
    return f, gamma, info


def canonical_phi(factors: CanonicalFactors, f: Tensor, *, scale: float, c_m: float) -> Tensor:
    """``Phi = (scale / c_m) V_m F V_k^T`` (``[d_m, d_k]``, ``D = E Phi``)."""
    return (float(scale) / float(c_m)) * (factors.v_m @ f @ factors.v_k.T)


def canonical_transport_map(
    p_m: Tensor,
    c_m: float,
    p_k: Tensor,
    *,
    scale: float,
    mode: str,
    rank: int | None = None,
    null_threshold: float | Tensor = 0.0,
    weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """``Phi`` of ``canonical_hard`` / ``canonical_soft`` from the anchor frames (see ``canonical_core``)."""
    factors = canonical_factors_from_frames(p_m, p_k)
    f, _, info = canonical_core(factors, mode, rank=rank, null_threshold=null_threshold, weights=weights)
    return canonical_phi(factors, f, scale=scale, c_m=c_m), info


def canonical_transport_map_from_sigma(
    sigma_mm: Tensor,
    sigma_kk: Tensor,
    sigma_mk: Tensor,
    n_rows: int,
    c_m: float,
    *,
    scale: float,
    mode: str,
    rank: int | None = None,
    null_threshold: float | Tensor = 0.0,
    weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Same ``Phi`` built from the streaming statistics alone (``anchors.canonical_stats``)."""
    factors = canonical_factors_from_sigma(sigma_mm, sigma_kk, sigma_mk, n_rows)
    f, _, info = canonical_core(factors, mode, rank=rank, null_threshold=null_threshold, weights=weights)
    return canonical_phi(factors, f, scale=scale, c_m=c_m), info


CORE_CONTROLS = ("tikhonov", "zca", "cross", "procrustes_cross")


def control_core(factors: CanonicalFactors, kind: str, *, eps: float | None = None) -> Tensor:
    """``F`` of the existing decoders in the unified form (controls): ``tikhonov`` ``S_m M S_k (S_k^2 + eps)^-1``,
    ``zca`` ``M`` (eps = 0), ``cross`` ``S_m M S_k / eps``, ``procrustes_cross`` ``polar(S_m M S_k)`` (the singular
    directions kept as in ``spectral_transport_map``). ``eps`` is absolute (``eps_rel * ||P_k||_F^2 / T``)."""
    if kind not in CORE_CONTROLS:
        raise ValueError(f"control must be one of {CORE_CONTROLS}, got {kind!r}")
    if kind == "zca":
        return factors.m
    c = factors.s_m[:, None] * factors.m * factors.s_k[None, :]
    if kind == "tikhonov":
        if eps is None:
            raise ValueError("tikhonov control needs eps")
        return c / (factors.s_k * factors.s_k + float(eps))[None, :]
    if kind == "cross":
        if eps is None or not eps > 0.0:
            raise ValueError("cross control needs eps > 0")
        return c / float(eps)
    u, sv, wt = torch.linalg.svd(c, full_matrices=False)
    valid = int((sv > sv[0] * max(c.shape) * torch.finfo(torch.float64).eps).sum().item())
    return u[:, :valid] @ wt[:valid]


# ---------------------------------------------------------------- generalized Procrustes (GPA) on the anchors
#
# AGENT_NEXT.md section 4. Every model m gets A_m (d_U x d_m, orthonormal COLUMNS: A_m^T A_m = I_{d_m}); a row h of
# m goes to the common centre G as h A_m^T and comes back as u A_m. d_U defaults to the widest model.


def procrustes_to_universe(p: Tensor, g: Tensor) -> Tensor:
    """``A`` (``[d_U, d]``) minimising ``||P A^T - G||_F`` with orthonormal columns: ``A = (U V^T)^T``,
    ``P^T G = U S V^T``. The step that adds a new model to a FIXED centre ``G`` (no existing map changes)."""
    u, _, vt = torch.linalg.svd(p.to(torch.float64).T @ g.to(torch.float64), full_matrices=False)
    return (u @ vt).T


def gpa(
    frames: list[Tensor], *, d_u: int | None = None, tol: float = 1e-6, max_iter: int = 100
) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
    """Generalized Procrustes of the anchor frames ``P_m`` (``[T, d_m]``, same anchors): ``(G, [A_m], info)``.

    Init from the widest model's PCA (``A_{m*} = V^T[:d_U]``, ``G = P_{m*} A_{m*}^T``); alternate the per-model
    Procrustes towards ``G`` and ``G = mean_m P_m A_m^T`` until ``|obj - obj_prev| / obj_prev < tol``. The gauge
    (a common rotation of ``G`` and every ``A_m``) is then fixed by the SVD of ``G``, so the axes of ``G`` are its
    principal directions in decreasing variance and ``A_m[:r]`` is the nested top-``r`` truncation. ``info`` has
    the objective, iterations, the per-model residual ``||P_m A_m^T - G|| / ||G||`` and ``underdetermined``
    (``T < d_U``: the rotations are fixed only on the span of the anchors).
    """
    ps = [p.to(torch.float64) for p in frames]
    t = int(ps[0].shape[0])
    if any(int(p.shape[0]) != t for p in ps):
        raise ValueError("GPA frames must share the anchor set (same T)")
    widths = [int(p.shape[1]) for p in ps]
    d_u = max(widths) if d_u is None else int(d_u)
    if d_u < max(widths):
        raise ValueError(f"d_U={d_u} is smaller than the widest model ({max(widths)}); truncate with rank instead")
    star = max(range(len(ps)), key=lambda m: widths[m])
    _, _, vt = torch.linalg.svd(ps[star], full_matrices=True)
    if d_u > widths[star]:
        raise ValueError("d_U larger than every model width is not supported")
    g = ps[star] @ vt[:d_u].T
    obj_prev = float("inf")
    obj = float("nan")
    n_iter = 0
    maps: list[Tensor] = []
    while n_iter < int(max_iter):
        n_iter += 1
        maps = [procrustes_to_universe(p, g) for p in ps]
        mapped = [p @ a.T for p, a in zip(ps, maps, strict=True)]
        g = sum(mapped) / len(mapped)
        obj = float(sum(float(((m - g) ** 2).sum()) for m in mapped))
        if math.isfinite(obj_prev) and abs(obj - obj_prev) <= tol * max(obj_prev, 1e-300):
            break
        obj_prev = obj
    _, _, vgt = torch.linalg.svd(g, full_matrices=True)
    g = g @ vgt.T
    maps = [vgt @ a for a in maps]
    g_norm = float(torch.linalg.norm(g))
    residual = [float(torch.linalg.norm(p @ a.T - g)) / g_norm for p, a in zip(ps, maps, strict=True)]
    info = {
        "gpa_d_u": d_u,
        "gpa_objective": obj,
        "gpa_iters": n_iter,
        "gpa_residual_per_model": residual,
        "gpa_underdetermined": t < d_u,
    }
    return g, maps, info


def gpa_transport_map(a_m: Tensor, a_k: Tensor, c_m: float, *, scale: float, rank: int | None = None) -> Tensor:
    """``Phi = (scale / c_m) A_m[:r]^T A_k[:r]`` (``[d_m, d_k]``): source -> top-``r`` axes of G -> target."""
    r = a_m.shape[0] if rank is None else min(int(rank), int(a_m.shape[0]))
    return (float(scale) / float(c_m)) * (a_m[:r].to(torch.float64).T @ a_k[:r].to(torch.float64))


# ---------------------------------------------------------------- consensus kernel K* (AGENT_ADDENDUM_CONSENSUS s. 2)
#
# Every frame P_m ([T, d_m]) is centred on the anchors and scaled so that its kernel is Frobenius-normalised:
# X_m = H P_m / n_m, K^_m = X_m X_m^T = H K_m H / ||H K_m H||_F (n_m = ||H K_m H||_F^(1/2)). K* is built from the K^_m,
# re-centred and normalised to ||K*||_F = 1, and Z* = V_k Lambda_k^(1/2) are its top-k axes.


def normalised_frame(p: Tensor) -> tuple[Tensor, float]:
    """``(X, n)``: ``X = (P - mean_rows P) / n`` with ``n = ||Pc^T Pc||_F^(1/2)``, so ``||X X^T||_F = 1``."""
    pc = p.to(torch.float64) - p.to(torch.float64).mean(0, keepdim=True)
    n = float(torch.linalg.norm(pc.T @ pc)) ** 0.5
    return pc / n, n


def _centre(k: Tensor) -> Tensor:
    k = 0.5 * (k + k.T)
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


def consensus_kernel(frames: list[Tensor], *, construction: str = "mean", log_eps: float = 1e-3) -> tuple[Tensor, dict]:
    """``K*`` (``[T, T]``, centred, ``||K*||_F = 1``) of the normalised kernels of ``frames``.

    ``mean``: ``(1/M) sum K^_m``. ``logeuclid``: ``exp((1/M) sum log(K^_m + e_m I))``, ``e_m = log_eps tr(K^_m) / T``,
    then centred. ``robust``: weights ``w_m`` proportional to ``1 / (1 - mean_{k != m} CKA(K^_m, K^_k))``.
    """
    xs = [normalised_frame(p)[0] for p in frames]
    t = int(xs[0].shape[0])
    kh = [x @ x.T for x in xs]
    m = len(kh)
    info: dict[str, Any] = {"n_models": m}
    if construction == "mean":
        k_star = sum(kh) / m
    elif construction == "robust":
        if m < 2:
            raise ValueError("consensus 'robust' needs at least two models")
        cka = torch.tensor([[float((a * b).sum()) for b in kh] for a in kh], dtype=torch.float64)
        mean_other = (cka.sum(1) - cka.diagonal()) / (m - 1)
        w = 1.0 / (1.0 - mean_other).clamp_min(1e-12)
        w = w / w.sum()
        k_star = sum(float(wi) * k for wi, k in zip(w, kh, strict=True))
        info["weights"] = [float(x) for x in w]
    elif construction == "logeuclid":
        acc = torch.zeros(t, t, dtype=torch.float64)
        for k in kh:
            e = log_eps * float(torch.trace(k)) / t
            lam, v = torch.linalg.eigh(k + e * torch.eye(t, dtype=torch.float64))
            acc += (v * torch.log(lam.clamp_min(1e-300))) @ v.T
        lam, v = torch.linalg.eigh(0.5 * (acc + acc.T) / m)
        k_star = (v * torch.exp(lam)) @ v.T
    else:
        raise ValueError(f"unknown consensus construction {construction!r}")
    k_star = _centre(k_star)
    k_star = k_star / torch.linalg.norm(k_star)
    lam = torch.linalg.eigvalsh(k_star).clamp_min(0.0)
    info["effective_rank"] = float(lam.sum() ** 2 / (lam * lam).sum())
    return k_star, info


def consensus_frame(k_star: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """``(Z*, Lambda_k)``: ``Z* = V_k Lambda_k^(1/2)`` on the top-``k`` eigenpairs of ``K*`` with positive eigenvalue."""
    lam, v = torch.linalg.eigh(k_star.to(torch.float64))
    lam, v = lam.flip(0), v.flip(1)
    keep = int(min(int(k), int((lam > PINV_RTOL * lam[0]).sum())))
    lam = lam[:keep]
    return v[:, :keep] * lam.sqrt(), lam


def _polar(m: Tensor) -> Tensor:
    u, _, vt = torch.linalg.svd(m, full_matrices=False)
    return u @ vt


def _sym_power(a: Tensor, power: float) -> Tensor:
    lam, v = torch.linalg.eigh(0.5 * (a + a.T))
    return (v * lam.clamp_min(1e-300).pow(power)) @ v.T


def consensus_register(p: Tensor, z: Tensor, lam: Tensor, *, register: str = "rot", eps: float = 1e-2
                       ) -> tuple[Tensor, Tensor, dict[str, float]]:
    """``(A, B, info)`` with ``A, B`` of shape ``[k, d]``: a frame row ``h`` goes to ``u = h A^T``, ``u`` back to ``u B``.

    With ``X = H P / n`` (``normalised_frame``) and ``e = eps tr(X X^T) / T``:

    - ``rot``: ``A = B = polar(X^T Z*)^T`` (orthonormal rows, isometric; the GPA step towards a fixed centre);
    - ``correct``: ``A = ((X^T X + e I)^-1 X^T Z*)^T / n`` (the anchors land on ``Z*`` up to ``e``), and ``B = n
      diag(1 / (lambda + e*)) Z*^T X`` (least squares from ``Z*`` back to the frame, ``e* = eps sum(lambda) / T``);
    - ``zca_rot``: ``W = (X^T X + e I)^(-1/2)``, ``R = polar((X W)^T V_k)`` with ``V_k = Z* Lambda^(-1/2)``; ``A = (W
      R)^T / n``, ``B = n R^T W^-1`` (a rotation between the whitened frame and the whitened consensus).

    ``info``: ``map_residual`` (``||X A^T n - target|| / ||target||``, target ``Z*`` or ``V_k`` for zca_rot) and
    ``map_cond`` (condition number of ``A``).
    """
    x, n = normalised_frame(p)
    z = z.to(torch.float64)
    t, d = x.shape
    e = float(eps) * float((x * x).sum()) / t
    if register == "rot":
        a = _polar(x.T @ z).T
        b = a
        target, enc = z, x @ a.T
    elif register == "correct":
        phi_t = torch.linalg.solve(x.T @ x + e * torch.eye(d, dtype=torch.float64), x.T @ z)  # [d, k]
        a = phi_t.T / n
        e_star = float(eps) * float(lam.sum()) / t
        b = n * (z.T @ x) / (lam.to(torch.float64) + e_star)[:, None]
        target, enc = z, x @ phi_t
    elif register == "zca_rot":
        g = x.T @ x + e * torch.eye(d, dtype=torch.float64)
        w = _sym_power(g, -0.5)
        vk = z / lam.to(torch.float64).sqrt()
        r = _polar((x @ w).T @ vk)  # [d, k]
        a = (w @ r).T / n
        b = n * r.T @ _sym_power(g, 0.5)
        target, enc = vk, x @ w @ r
    else:
        raise ValueError(f"unknown consensus_register {register!r}")
    s = torch.linalg.svdvals(a)
    info = {"map_residual": float(torch.linalg.norm(enc - target) / torch.linalg.norm(target)),
            "map_cond": float(s[0] / s[-1]) if float(s[-1]) > 0 else float("inf")}
    return a, b, info


def mutual_knn(k_a: Tensor, k_b: Tensor, k: int = 10) -> float:
    """Mutual k-NN alignment (Huh et al. 2024) of two kernels on the same anchors: mean overlap of the ``k`` nearest
    anchors of every anchor (largest kernel entries, self excluded), divided by ``k``."""
    t = k_a.shape[0]
    big = torch.finfo(torch.float64).max
    eye = torch.eye(t, dtype=torch.bool)
    na = k_a.to(torch.float64).masked_fill(eye, -big).topk(k, dim=1).indices
    nb = k_b.to(torch.float64).masked_fill(eye, -big).topk(k, dim=1).indices
    hit = torch.zeros(t, t, dtype=torch.bool)
    hit.scatter_(1, na, True)
    return float(hit.gather(1, nb).float().mean())


def kernel_stats_to_consensus(p: Tensor, k_star: Tensor) -> dict[str, float]:
    """``1 - CKA(K^_m, K*)`` (both centred) and the mutual 10-NN of ``K^_m`` with ``K*``."""
    x, _ = normalised_frame(p)
    kh = x @ x.T
    ks = k_star.to(torch.float64)
    cka = float((kh * ks).sum() / (torch.linalg.norm(kh) * torch.linalg.norm(ks)))
    return {"one_minus_cka": 1.0 - cka, "mutual_knn10": mutual_knn(kh, ks, 10)}
