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
