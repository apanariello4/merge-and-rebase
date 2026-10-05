"""Linear centered kernel alignment (CKA) between two feature matrices.

CKA measures how similar two representations are up to an orthogonal
transform and isotropic scaling of either one -- exactly the invariances a
task-vector transport can legitimately buy, so a raw weight-space comparison
cannot substitute for it. Both estimators implemented here use the *linear*
kernel (``K = X X^T``); nothing here is specific to vision or to CLIP.

Two estimators are provided:

- **Biased** (``debiased=False``): the classic HSIC-based CKA of Kornblith
  et al. 2019, computed via the efficient feature-space identity
  ``HSIC_1(K, L) = ||Yc^T Xc||_F^2`` for centered ``Xc``, ``Yc`` (valid, and
  cheap, whether ``d << n`` or ``d > n``). It is invariant to an orthogonal
  transform and to isotropic scaling of either input, but it is a biased
  estimator of the population quantity: for independent features, its
  expectation grows with the feature dimensionality relative to ``n``, so it
  does not go to zero even when the two representations are unrelated.
- **Debiased** (``debiased=True``, the default): the unbiased HSIC estimator
  of Song et al. 2012, applied to linear CKA by Nguyen et al. 2021's
  minibatch/small-sample construction. It cancels this bias term-by-term and
  is close to 0 for independent features regardless of ``d``. This is the
  estimator this module's callers should prefer whenever ``n`` is only in
  the hundreds and ``d`` is in the thousands, as is the case for ViT block
  activations. It requires ``n >= 4``.
"""

from __future__ import annotations

import torch

__all__ = ["linear_cka", "cka_bootstrap"]


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _frobenius_norm(mat: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(mat, ord="fro")


def _biased_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """``||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)``, the efficient feature-space form of HSIC-based CKA."""
    xc, yc = _center(x), _center(y)
    numerator = _frobenius_norm(yc.T @ xc) ** 2
    denominator = _frobenius_norm(xc.T @ xc) * _frobenius_norm(yc.T @ yc)
    if float(denominator.item()) <= 0.0:
        return float("nan")
    return float((numerator / denominator).item())


def _hsic_unbiased(k: torch.Tensor, l_: torch.Tensor) -> float:
    """Unbiased HSIC estimator (Song et al. 2012), Gram-matrix form.

    ``k``, ``l`` are ``[n, n]`` Gram matrices (``K = X X^T``) with their
    diagonal already zeroed by the caller (``k_tilde``/``l_tilde``).
    """
    n = k.shape[0]
    ones = torch.ones(n, dtype=k.dtype, device=k.device)
    kl = k @ l_
    term1 = torch.trace(kl)
    k_sum = ones @ k @ ones
    l_sum = ones @ l_ @ ones
    term2 = (k_sum * l_sum) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (ones @ kl @ ones)
    hsic = (term1 + term2 - term3) / (n * (n - 3))
    return float(hsic.item())


def _zero_diagonal(mat: torch.Tensor) -> torch.Tensor:
    out = mat.clone()
    out.fill_diagonal_(0.0)
    return out


def _debiased_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    n = x.shape[0]
    if n < 4:
        raise ValueError(f"Debiased CKA requires n >= 4, got n={n}")
    k = _zero_diagonal(x @ x.T)
    l_ = _zero_diagonal(y @ y.T)
    hsic_xy = _hsic_unbiased(k, l_)
    hsic_xx = _hsic_unbiased(k, k)
    hsic_yy = _hsic_unbiased(l_, l_)
    if hsic_xx <= 0.0 or hsic_yy <= 0.0:
        return float("nan")
    return hsic_xy / (hsic_xx * hsic_yy) ** 0.5


def linear_cka(x: torch.Tensor, y: torch.Tensor, *, debiased: bool = True) -> float:
    """Linear CKA between ``x`` (``[n, d_x]``) and ``y`` (``[n, d_y]``).

    Both inputs are cast to float64 before any computation, since the Gram
    matrices and their traces otherwise lose precision fast for the block
    widths (768-1024) and sample counts (hundreds) this module is used with.
    CKA is symmetric (``linear_cka(x, y) == linear_cka(y, x)``), invariant to
    any orthogonal transform ``Q`` applied to either input (``x @ Q``), and
    invariant to isotropic rescaling of either input (``c * x``, ``c != 0``).
    It equals 1 when ``x`` and ``y`` are related by such a transform (in
    particular, a representation compared to itself), and is not defined
    (returns ``nan``) if either input has degenerate (zero) variance across
    all sampled points.

    Returns ``nan`` (rather than raising) when a normalizing denominator is
    ``<= 0`` -- e.g. one input is exactly constant across all ``n`` rows -- so
    that callers can carry the missing value through a table without a
    special-cased try/except at every call site.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"linear_cka expects 2D [n, d] inputs, got shapes {tuple(x.shape)}, {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"linear_cka inputs must share n, got {x.shape[0]} vs {y.shape[0]}")
    x64 = x.detach().to(dtype=torch.float64, device="cpu")
    y64 = y.detach().to(dtype=torch.float64, device="cpu")
    if debiased:
        return _debiased_linear_cka(x64, y64)
    return _biased_linear_cka(x64, y64)


def cka_bootstrap(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_boot: int = 200,
    seed: int = 0,
    debiased: bool = True,
) -> dict[str, float]:
    """CKA with a 95% bootstrap interval over rows (images).

    Rows are resampled with replacement using a ``torch.Generator`` seeded
    with ``seed``, so the result is deterministic for a fixed ``(x, y, seed,
    n_boot)``. Note: a bootstrap resample duplicates rows, which makes some
    off-diagonal entries of the resampled Gram matrix exactly equal (a
    duplicated row against itself, off the original diagonal); the debiased
    HSIC estimator's diagonal-zeroing only removes the resampled matrix's own
    diagonal, not these duplicate-pair entries, so it remains a well-defined
    (if slightly biased-by-duplication) statistic on the resampled sample --
    this is the standard bootstrap-with-replacement caveat and is treated as
    acceptable here since ``n=512`` makes duplication mild.

    Returns ``{"value": ..., "lo": ..., "hi": ...}``, the point estimate on
    the full (unresampled) data and the 2.5th/97.5th percentiles of the
    bootstrap distribution. ``lo <= hi`` always holds; either may be ``nan``
    if a resample produced a degenerate CKA (see ``linear_cka``).
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    n = x.shape[0]
    value = linear_cka(x, y, debiased=debiased)
    generator = torch.Generator().manual_seed(int(seed))
    samples = torch.empty(n_boot, dtype=torch.float64)
    for i in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=generator)
        samples[i] = linear_cka(x[idx], y[idx], debiased=debiased)
    finite = samples[torch.isfinite(samples)]
    if finite.numel() == 0:
        lo = hi = float("nan")
    else:
        sorted_finite, _ = torch.sort(finite)
        lo = float(torch.quantile(sorted_finite, 0.025).item())
        hi = float(torch.quantile(sorted_finite, 0.975).item())
    return {"value": value, "lo": lo, "hi": hi}
