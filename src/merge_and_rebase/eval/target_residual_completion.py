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
    # Which functional-transfer path Proposal 1 takes.
    #   "transport_residual" -- the historical path: complete the residual left
    #                           by an already-transported task vector, solving
    #                           in source coordinates through (t_in, t_out).
    #   "direct_target"      -- the transport-free ablation: start from the
    #                           native target base (no tau_t at all) and solve
    #                           for the desired effect directly in target
    #                           coordinates. Algebraically this is the same
    #                           affine ridge with t_in = I and t_out carrying
    #                           only the target's own LayerScale, so it reuses
    #                           the identical, tested solve rather than a
    #                           second implementation of it.
    mode: str = "transport_residual"
    # How a realized target position picks the source effect it is asked to
    # reproduce.
    #   "step"        -- every realized position takes its recorded ancestor's
    #                    full effect D = (B_i^1 - B_i^0) Q. Where one source
    #                    block is realized as several target blocks, each of
    #                    them is asked for the whole of that block's effect.
    #   "interpolate" -- blend neighbouring source effects by the position's
    #                    fractional depth in the realized chain, so a position
    #                    realizing "half of source block i" is asked for half
    #                    the step from i-1 to i. Agrees with "step" exactly at
    #                    the integer coordinates, i.e. at the last realized
    #                    member of every ancestry group.
    target_trajectory: str = "step"
    # Which residual-writing projections the completion is allowed to fit.  A
    # transformer block adds into the residual stream twice -- once from the
    # attention output projection, once from the MLP output projection -- and
    # the historical protocol corrected only the second.  Fitted in forward
    # order, cascaded (never as one joint linear system: out_proj's change
    # moves the MLP's own input through ln_2 and GELU, so the second fit has
    # to *measure* that rather than linearize it).
    components: tuple[str, ...] = ("mlp.c_proj",)
    # The order the sequential cascade visits realized target positions.
    #   "bottom_top" -- ascending position (the historical, default order). Each
    #                   block is fitted after every block upstream of it is
    #                   final, so its measured input H_j is the one the finished
    #                   model will actually feed it.
    #   "top_bottom" -- descending position. A block is then fitted before its
    #                   upstream neighbours change, so later fits invalidate the
    #                   input distribution earlier ones were fitted against.
    #   "independent" -- no cascade at all. Every block is fitted against the
    #                    pristine target base, so E_j == D_j for all j and the
    #                    corrections cannot interfere with each other's targets.
    #                    Order is irrelevant here by construction.
    # This is an ablation of how load-bearing the cascade coupling is: measured
    # `relative_residual_before` runs *above* 1.0 for most non-first blocks
    # (median ~1.06-1.08), i.e. upstream corrections leave downstream blocks a
    # harder target than the untouched base would, so the coupling is not
    # obviously helping and its direction is worth testing.
    cascade_order: str = "bottom_top"
    # Decoder/LLM path only. That path splits a task vector into a transportable
    # "body" and a ``passthrough`` remainder (embeddings, per-layer norms,
    # lm_head) which it folds in verbatim after completion. Vision has no such
    # split, so `direct_target` there is exactly ``theta_t^0 + gamma * dtau``.
    #   False (default) -- same contract on the decoder: the fitted correction
    #                      is the entire task vector, so gamma=0 is an exact
    #                      native-target-base control.
    #   True            -- also carry the shape-compatible passthrough keys, so
    #                      the arm differs from the transport arm only in how
    #                      the body is built. The arm is then not strictly
    #                      transport-free and gamma=0 no longer reproduces the
    #                      native base; both are recorded in the run summary.
    # For a width-changing pair (Qwen 0.5B->1.5B, 896->1536 hidden) almost every
    # passthrough key is shape-incompatible and already dropped, so this is
    # close to a no-op there; it bites on a same-width, depth-only rebase.
    direct_passthrough: bool = False


#: Residual-writing projections, in the order a block executes them.
COMPONENT_FORWARD_ORDER: tuple[str, ...] = ("attn.out_proj", "mlp.c_proj")
_DEFAULT_COMPONENTS: tuple[str, ...] = ("mlp.c_proj",)


def order_components(components) -> tuple[str, ...]:
    """Return ``components`` in block-forward order.

    The config names a *set* of write surfaces; the fit order is a property of
    the architecture, not of how the config happened to list them.
    """
    selected = set(components)
    return tuple(name for name in COMPONENT_FORWARD_ORDER if name in selected)


def parse_residual_completion_config(value: Mapping[str, Any] | None) -> ResidualCompletionConfig:
    """Parse and validate the narrow proposal-1 configuration schema."""
    if value is None:
        return ResidualCompletionConfig()
    if not isinstance(value, Mapping):
        raise TypeError("target_residual_completion must be a mapping")
    allowed = {
        "enabled", "added_blocks", "target_scope", "component", "ridge_relative", "strength", "num_batches",
        "exact_form", "missing_bias", "mode", "target_trajectory", "components",
        "cascade_order",
        "direct_passthrough",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown target_residual_completion fields: {sorted(unknown)}")
    payload = dict(value)
    # JSON gives a list; the dataclass is frozen and lands in asdict() output,
    # so normalize to a tuple up front rather than leaving two shapes around.
    if "components" in payload and isinstance(payload["components"], list):
        payload["components"] = tuple(payload["components"])
    cfg = ResidualCompletionConfig(**payload)
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
    if cfg.mode not in {"transport_residual", "direct_target"}:
        raise ValueError("mode must be 'transport_residual' or 'direct_target'")
    if cfg.target_trajectory not in {"step", "interpolate"}:
        raise ValueError("target_trajectory must be 'step' or 'interpolate'")
    if cfg.cascade_order not in {"bottom_top", "top_bottom", "independent"}:
        raise ValueError("cascade_order must be 'bottom_top', 'top_bottom' or 'independent'")
    if cfg.target_trajectory == "interpolate":
        # Both new write surfaces are direct-mode features; the transport arm's
        # solve is defined against fitted (t_in, t_out) maps and is deliberately
        # left exactly as it was.
        if cfg.mode != "direct_target":
            raise ValueError("target_trajectory='interpolate' requires mode='direct_target'")
        if cfg.target_scope != "all":
            # The inserted-only path consumes the precomputed references['desired']
            # banks, which are built without ancestry-group structure, so there is
            # no fractional depth to interpolate along. Refused rather than
            # silently stepping.
            raise ValueError("target_trajectory='interpolate' requires target_scope='all'")
    components = cfg.components
    if isinstance(components, str) or not isinstance(components, (list, tuple)):
        raise ValueError("components must be a list of projection names")
    components = tuple(components)
    if not components:
        raise ValueError("components must not be empty")
    if len(set(components)) != len(components):
        raise ValueError("components must not repeat a projection")
    unsupported = set(components) - set(COMPONENT_FORWARD_ORDER)
    if unsupported:
        raise ValueError(
            f"unknown components: {sorted(unsupported)}; supported: {sorted(COMPONENT_FORWARD_ORDER)}"
        )
    if "mlp.c_proj" not in components:
        # The MLP projection is the last write into the residual stream and the
        # only one whose solver objective is exactly the post-mount residual;
        # dropping it would leave the fit unanchored.
        raise ValueError("components must contain 'mlp.c_proj'")
    if order_components(components) != _DEFAULT_COMPONENTS and cfg.mode != "direct_target":
        raise ValueError(
            "components other than ['mlp.c_proj'] require mode='direct_target': the transport "
            "arm would need fitted t_in/t_out maps for attn.out_proj, which are not produced"
        )
    if not isinstance(cfg.direct_passthrough, bool):
        raise TypeError("direct_passthrough must be bool")
    if cfg.direct_passthrough and cfg.mode != "direct_target":
        raise ValueError("direct_passthrough=true requires mode='direct_target'")
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


@dataclass(frozen=True)
class JointCorrectionConfig:
    """Configuration for the frozen-map Option 3 blockwise correction.

    ``source_weight`` weights the ordinary source ARIADNE reconstruction
    objective and ``target_weight`` weights the target-space transported-effect
    objective.  The transport maps are deliberately not configurable here:
    Option 3 fits them in a preceding, ordinary transport pass and freezes
    them for this solve.  ``enabled=False`` is the compatibility default and
    has no effect until a caller explicitly opts into the joint path.
    """

    enabled: bool = False
    source_weight: float = 1.0
    target_weight: float = 1.0
    ridge_relative: float = 1e-3


def parse_joint_correction_config(value: Mapping[str, Any] | None) -> JointCorrectionConfig:
    """Parse the narrow, explicit Option 3 configuration schema.

    This parser intentionally rejects transport/seeding controls.  Those
    belong to the transport method and changing them during the joint solve
    would violate the frozen-map protocol.
    """
    if value is None:
        return JointCorrectionConfig()
    if not isinstance(value, Mapping):
        raise TypeError("joint_blockwise_correction must be a mapping")
    allowed = {"enabled", "source_weight", "target_weight", "ridge_relative"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown joint_blockwise_correction fields: {sorted(unknown)}")
    cfg = JointCorrectionConfig(**dict(value))
    if not isinstance(cfg.enabled, bool):
        raise TypeError("enabled must be bool")
    for name in ("source_weight", "target_weight", "ridge_relative"):
        number = getattr(cfg, name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise ValueError(f"{name} must be a finite real number")
        if name == "ridge_relative" and number <= 0:
            raise ValueError("ridge_relative must be > 0")
        if name != "ridge_relative" and number < 0:
            raise ValueError(f"{name} must be >= 0")
    if cfg.source_weight == 0 and cfg.target_weight == 0:
        raise ValueError("at least one joint objective weight must be positive")
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


def fit_joint_cproj_correction(
    source_h: Tensor,
    source_target: Tensor,
    target_h: Tensor,
    target_effect_residual: Tensor,
    t_in: Tensor,
    t_out: Tensor,
    *,
    source_weight: float = 1.0,
    target_weight: float = 1.0,
    ridge_relative: float = 1e-3,
    target_intercept: bool = True,
) -> tuple[Tensor, dict[str, Any]]:
    r"""Fit the frozen-map Option 3 c_proj correction in source coordinates.

    The returned matrix is ``Delta_C_source``.  Given baseline ARIADNE source
    features ``H_s`` and target-side features ``H_t``, the exact objective is

    .. math::

       \min_X w_s\|H_s X - Y_s\|_F^2
       + w_t\| (H_t t_{in}^T)X t_{out} - E_t\|_F^2
       + \lambda\|X\|_F^2,

    where ``X = Delta_C_source.T``.  ``t_in`` and ``t_out`` are frozen maps
    fitted on the baseline ARIADNE resize; they are never differentiated or
    refit here.  The first term is the ordinary source reconstruction target,
    while the second is the target transported-effect residual.  This is the
    documented one-alternation Option 3 protocol, not Proposal 1's
    post-transport task-vector completion.

    The solve uses sufficient statistics and an exact eigendecomposed
    blockwise solve of the true two-Gram normal operator, avoiding a
    Kronecker design matrix (the source and target Gramians generally do not
    commute).  The returned matrix is the weight correction and the
    diagnostics contain ``bias_correction``, the shared source-coordinate
    intercept ``beta``.  The affine objective is

    .. math::

       w_s\|H_s X + 1\beta^T - Y_s\|_F^2
       + w_t\|(H_t T_{in}^T X + \beta^T)T_{out} - E_t\|_F^2
       + \lambda\|X\|_F^2.

    The intercept is deliberately *not* ridge-penalized, matching ARIADNE's
    affine fit.  Internally it is an extra, unregularized feature column, so
    the same output-eigendecomposition gives an exact joint solve rather than
    fitting a weight and bias in separate stages.  ``beta`` is transported as
    ``T_out.T @ beta`` by the caller, exactly like ``c_proj.bias`` in the
    Theseus/BiCo transport convention.  ``target_intercept=False`` makes the
    intercept source-only.  That variant is used when the fitted affine map is
    applied to both source endpoints: its bias cancels from their task vector
    and therefore cannot contribute to the target P1 term.
    """
    matrices = (source_h, source_target, target_h, target_effect_residual, t_in, t_out)
    if any(not isinstance(x, torch.Tensor) or x.ndim != 2 for x in matrices):
        raise ValueError("joint c_proj inputs must be rank-2 tensors")
    if any(any(int(dim) == 0 for dim in x.shape) for x in matrices):
        raise ValueError("joint c_proj inputs must have non-empty dimensions")
    if source_h.shape[0] != source_target.shape[0]:
        raise ValueError("source feature and target rows must match")
    if target_h.shape[0] != target_effect_residual.shape[0]:
        raise ValueError("target feature and residual rows must match")
    if source_h.shape[1] != t_in.shape[0] or target_h.shape[1] != t_in.shape[1]:
        raise ValueError("t_in must have shape [source_input, target_input]")
    if source_target.shape[1] != t_out.shape[0] or target_effect_residual.shape[1] != t_out.shape[1]:
        raise ValueError("t_out must have shape [source_output, target_output]")
    if any(not torch.isfinite(x).all() for x in matrices):
        raise ValueError("joint c_proj inputs must be finite")
    for name, value in (("source_weight", source_weight), ("target_weight", target_weight), ("ridge_relative", ridge_relative)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number")
        if name == "ridge_relative" and value <= 0:
            raise ValueError("ridge_relative must be > 0")
        if name != "ridge_relative" and value < 0:
            raise ValueError(f"{name} must be >= 0")
    if source_weight == 0 and target_weight == 0:
        raise ValueError("at least one joint objective weight must be positive")
    if not isinstance(target_intercept, bool):
        raise ValueError("target_intercept must be boolean")

    hs = source_h.to(torch.float64)
    ys = source_target.to(torch.float64)
    # Pull target activations back into source input coordinates.  This is the
    # same orientation used by ResidualSufficientStatistics.update.
    at = target_h.to(torch.float64) @ t_in.to(torch.float64).T
    et = target_effect_residual.to(torch.float64)
    tout64 = t_out.to(torch.float64)

    # Add one shared source-coordinate intercept feature.  For the target
    # term this is also a source-coordinate intercept after the frozen input
    # map: [A_t, 1] [X; beta] T_out.  The final row is excluded from the ridge
    # matrix below; ARIADNE's affine bias is not regularized.
    ones_s = torch.ones((hs.shape[0], 1), dtype=torch.float64)
    ones_t = torch.ones((at.shape[0], 1), dtype=torch.float64)
    hs_aug = torch.cat((hs, ones_s), dim=1)
    target_bias_feature = ones_t if target_intercept else torch.zeros_like(ones_t)
    at_aug = torch.cat((at, target_bias_feature), dim=1)
    ss = (hs_aug.T @ hs_aug)
    st = (at_aug.T @ at_aug)
    gt = (tout64 @ tout64.T)
    b = float(source_weight) * (hs_aug.T @ ys) + float(target_weight) * (at_aug.T @ et @ tout64.T)
    ss = (ss + ss.T) * 0.5
    st = (st + st.T) * 0.5
    gt = (gt + gt.T) * 0.5

    m_source = int(hs.shape[1])
    d_source = int(gt.shape[0])
    m_aug = m_source + 1
    trace_scale = float(source_weight) * float(torch.trace(hs.T @ hs).item())
    trace_scale += float(target_weight) * float(torch.trace(at.T @ at).item())
    lam = float(ridge_relative) * trace_scale / max(1, m_source)

    # The source and target terms have different output Gramians.  They cannot
    # be collapsed into ``(ws*Ss+wt*St) X (ws*I+wt*Gt)``: that creates cross
    # terms and is not the normal equation of the stated objective.  Since the
    # target Gram is symmetric PSD, diagonalize it once.  For each frozen
    # output eigendirection j, the true normal equation is the independent
    # source-space system
    #
    #   (ws Ss + wt eig_j(St) + lambda R) y_j = (B U)_j,
    #
    # followed by X = Y U^T.  This is an exact blockwise closed-form solve and
    # avoids materializing an (m*d) Kronecker matrix or relying on a
    # convergence-dependent iterative solver.
    # Always solve the augmented system, including when the activation banks
    # themselves are zero.  The all-ones intercept column still carries a
    # nonzero sufficient statistic and can exactly explain a constant target;
    # short-circuiting on ``trace_scale`` would incorrectly force beta to zero.
    eigvals, eigvecs = torch.linalg.eigh(gt)
    eigvals = eigvals.clamp_min(0.0)
    rhs = b @ eigvecs
    ridge_mask = torch.zeros((m_aug, m_aug), dtype=torch.float64)
    ridge_mask[:m_source, :m_source] = torch.eye(m_source, dtype=torch.float64)
    y = torch.empty(m_aug, d_source, dtype=torch.float64)
    source_gram = float(source_weight) * ss
    target_gram = float(target_weight) * st
    for j, eigval in enumerate(eigvals):
        lhs = source_gram + float(eigval) * target_gram + lam * ridge_mask
        lhs = (lhs + lhs.T) * 0.5
        try:
            y[:, j] = torch.linalg.solve(lhs, rhs[:, j])
        except RuntimeError:
            # A target-only solve with a rank-deficient T_out leaves some
            # intercept directions unconstrained.  The minimum-norm
            # pseudoinverse solution is the exact ridge objective minimizer
            # and keeps that valid degenerate case finite.
            y[:, j] = torch.linalg.pinv(lhs, hermitian=True) @ rhs[:, j]
    z = y @ eigvecs.T
    used_blocks = d_source
    x = z[:m_source]
    beta = z[m_source]
    if not torch.isfinite(x).all() or not torch.isfinite(beta).all():
        raise RuntimeError("joint c_proj frozen-map blockwise solve produced non-finite values")

    def _objective(x_arg: Tensor, beta_arg: Tensor) -> float:
        src_err = hs @ x_arg + beta_arg - ys
        target_bias = beta_arg if target_intercept else torch.zeros_like(beta_arg)
        tgt_err = (at @ x_arg + target_bias) @ tout64 - et
        value = float(source_weight) * float((src_err * src_err).sum().item())
        value += float(target_weight) * float((tgt_err * tgt_err).sum().item())
        value += lam * float((x_arg * x_arg).sum().item())
        return value

    zero = torch.zeros_like(x)
    zero_beta = torch.zeros_like(beta)
    diagnostics = {
        "objective_before": _objective(zero, zero_beta),
        "objective_after": _objective(x, beta),
        "source_reconstruction_before": float((ys * ys).sum().item()) ** 0.5,
        "source_reconstruction_after": float(((hs @ x + beta - ys) ** 2).sum().item()) ** 0.5,
        "target_effect_residual_before": float((et * et).sum().item()) ** 0.5,
        "target_effect_residual_after": float(
            (((at @ x + (beta if target_intercept else 0.0)) @ tout64 - et) ** 2).sum().item()
        ) ** 0.5,
        "ridge": lam,
        "source_weight": float(source_weight),
        "target_weight": float(target_weight),
        "frozen_map": True,
        "alternations": 1,
        "solver": "blockwise_eigh",
        "solver_blocks": used_blocks,
        "bias_correction": beta.to(torch.float32),
        "bias_norm": float(torch.linalg.norm(beta).item()),
        "bias_regularized": False,
        "target_intercept": target_intercept,
    }
    return x.T.to(torch.float32), diagnostics
