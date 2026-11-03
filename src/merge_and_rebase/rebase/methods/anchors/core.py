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

KERNEL_DECODES = ("tikhonov", "zca", "agreement")
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
    if mode not in KERNEL_DECODES[1:]:
        raise ValueError(f"whitened decode must be one of {KERNEL_DECODES[1:]}, got {mode!r}")
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
