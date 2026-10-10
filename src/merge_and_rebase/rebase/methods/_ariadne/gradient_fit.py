"""Gradient-based component fits of Ariadne (``component_fit="gradient_exact" | "gradient_linearized"``, vision only).

Per target block ``j`` and a component group ``G`` (q/k/v slices of ``attn.in_proj``, ``attn.out_proj``, ``mlp.c_fc`` or
``mlp.c_proj``), on the block inputs ``x_j`` of the untouched target (``[B, T, d]`` sequence structure kept: attention mixes
tokens), with ``T_j`` the EXACT block forward and ``D_j`` the block-end target of the closed-form fit (``(S_ft - S_base)_{pi(j)}
Q_j``, optionally ``/ m`` for ``pair_target_split='per_block_share'``, or the native oracle ``T^ft_j - T^base_j``):

* ``gradient_exact``:      ``min_{dW, db}  sum ||T_j(W + dW, b + db; x_j) - T_j(W, b; x_j) - D_j||^2 + lam ||dW||_F^2``
  by full-batch L-BFGS (strong Wolfe) from zero; loss accumulated in float64, one backward per calibration batch.
* ``gradient_linearized``: the same with ``T_j`` replaced by its first-order expansion ``J (dW, db)`` at ``(W, b)``: a convex
  ridge problem solved by conjugate gradients on the normal equations ``(J^T J + Lambda) z = J^T D`` with forward-mode
  ``torch.func.jvp`` and reverse-mode products (``J`` is never formed).

The bias is unpenalised (``Lambda`` is ``lam`` on weight entries, 0 on bias entries). For ``mlp.c_proj`` the block output
is affine in ``(dW, db)``, so exact = linearized = the closed-form ridge fit; for ``attn.out_proj`` the objectives include
the MLP reaction downstream of the attention (``ln_2`` + MLP), which the closed-form O fit ignores.

``lam`` (``gradient_ridge``): ``eb_input`` = the empirical-Bayes value ``tr(H_c^T H_c) / (n - 1)`` of the component's own
input features (``tr(G) / d`` factor for LayerScale on the residual writers, as in the closed form); ``match_norm`` = the
``lam`` (bisection on ``log lam``, 1 % tolerance) at which ``||dW||_F`` equals that of the closed-form ``mlp.c_proj`` fit of
the same block and task, computed internally.

The block forward is applied through the module's own recorded call arguments (``args``/``kwargs`` of a forward pre-hook), so
the code does not depend on the tensor layout (``[B, T, d]`` or ``[T, B, d]``) the block consumes.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._pytree import tree_map

from ...discrete_layer_match import DiscreteLayerPairing
from .capture import iter_capture_tokens
from .config import DirectResidualConfig, annotate_target_split, order_components, resolve_fit_blocks, target_groups
from .fit import ResidualSufficientStatistics
from .layouts import (
    _PACKED_QKV_SLICE,
    COMPONENT_INPUT_KIND,
    _aligned,
    _component_effective_out,
    _family_bias_key,
    _layout_for,
)
from .streaming import _check_boundary_fingerprint, _none_stream

Tensor = torch.Tensor

#: (weight, bias) parameter names of each component, relative to the block module.
_PARAM_NAMES = {
    "mlp.c_proj": ("mlp.c_proj.weight", "mlp.c_proj.bias"),
    "mlp.c_fc": ("mlp.c_fc.weight", "mlp.c_fc.bias"),
    "attn.out_proj": ("attn.out_proj.weight", "attn.out_proj.bias"),
    "attn.q_proj": ("attn.in_proj_weight", "attn.in_proj_bias"),
    "attn.k_proj": ("attn.in_proj_weight", "attn.in_proj_bias"),
    "attn.v_proj": ("attn.in_proj_weight", "attn.in_proj_bias"),
}

_MATCH_NORM_TOL = 0.01
_MATCH_NORM_MAX_SOLVES = 40


def _record_call(store: dict, pos: int):
    """Forward pre-hook recording the block's latest call ``(args, kwargs)`` (detached) under ``store[pos]``."""

    def hook(_module, args, kwargs):
        store[pos] = tree_map(lambda t: t.detach() if isinstance(t, Tensor) else t, (args, dict(kwargs)))

    return hook


class _FeatureMoments:
    """Row count, sum and total square of the component's input features (float64), for the empirical-Bayes ridge."""

    def __init__(self) -> None:
        self.n = 0
        self.total: Tensor | None = None
        self.sumsq = 0.0

    def update(self, h: Tensor) -> None:
        rows = h.reshape(-1, h.shape[-1]).double()
        self.n += int(rows.shape[0])
        self.total = rows.sum(0) if self.total is None else self.total + rows.sum(0)
        self.sumsq += float((rows * rows).sum().item())

    def centred_trace(self) -> float:
        assert self.total is not None
        return self.sumsq - float((self.total @ self.total).item()) / self.n


class _BlockProblem:
    """One block's fit: variables ``z`` = flat ``(dW, db)`` of every component in the group, exact block forward."""

    def __init__(self, block, components: tuple[str, ...], entries: list[tuple[tuple, dict, Tensor]], device) -> None:
        self.block = block
        self.components = components
        self.entries = entries  # (args, kwargs, D in the block's own layout), all on ``device``
        self.base = {name: p.detach() for name, p in block.named_parameters()}
        self.segments: list[tuple[str, str, int, int, tuple[int, ...]]] = []
        offset = 0
        for component in components:
            w_name, b_name = _PARAM_NAMES[component]
            weight, bias = self._slice(self.base[w_name], component), None
            if b_name in self.base:
                bias = self._slice(self.base[b_name], component)
            self.segments.append((component, "w", offset, weight.numel(), tuple(weight.shape)))
            offset += weight.numel()
            if bias is not None:
                self.segments.append((component, "b", offset, bias.numel(), tuple(bias.shape)))
                offset += bias.numel()
        self.size = offset
        weight0 = self.base[_PARAM_NAMES[components[0]][0]]
        self.dtype = torch.float64 if weight0.dtype == torch.float64 else torch.float32
        self.device = weight0.device
        self.weight_mask = torch.zeros(offset, dtype=torch.bool, device=self.device)
        for _c, kind, start, numel, _shape in self.segments:
            if kind == "w":
                self.weight_mask[start : start + numel] = True
        self.data_sq = 0.0
        for _args, _kwargs, d in entries:
            self.data_sq += float((d.double() ** 2).sum().item())
        self.targets: list[Tensor] = []
        self._forward_ad = True

    @staticmethod
    def _slice(tensor: Tensor, component: str) -> Tensor:
        if component in _PACKED_QKV_SLICE:
            d = tensor.shape[0] // 3
            s = _PACKED_QKV_SLICE[component]
            return tensor[s * d : (s + 1) * d]
        return tensor

    def zeros(self) -> Tensor:
        return torch.zeros(self.size, dtype=self.dtype, device=self.device)

    def overrides(self, z: Tensor) -> dict[str, Tensor]:
        """Full parameter tensors ``W + dW`` / ``b + db`` built differentiably from the flat variable ``z``."""
        pieces: dict[str, dict[int, Tensor]] = {}
        out: dict[str, Tensor] = {}
        for component, kind, start, numel, shape in self.segments:
            name = _PARAM_NAMES[component][0 if kind == "w" else 1]
            chunk = z[start : start + numel].reshape(shape)
            if component in _PACKED_QKV_SLICE:
                pieces.setdefault(name, {})[_PACKED_QKV_SLICE[component]] = chunk
            else:
                out[name] = self.base[name] + chunk
        for name, by_slice in pieces.items():
            full = self.base[name]
            d = full.shape[0] // 3
            parts = [by_slice.get(s, torch.zeros_like(full[s * d : (s + 1) * d])) for s in range(3)]
            out[name] = full + torch.cat(parts, dim=0)
        return out

    def forward(self, z: Tensor, entry) -> Tensor:
        args, kwargs, _d = entry
        # Math SDPA backend: the fused kernels have no forward-mode / double-backward derivative, and every evaluation
        # (targets, objective, residuals) must go through the same numerics.
        with sdpa_kernel(SDPBackend.MATH):
            out = functional_call(self.block, {**self.base, **self.overrides(z)}, args, kwargs)
        return out[0] if isinstance(out, tuple) else out

    def prepare_targets(self) -> None:
        """``T_j(W; x_j) + D_j`` per batch, through the SAME code path (and grad mode) as every later evaluation."""
        z0 = self.zeros().requires_grad_(True)
        with torch.enable_grad():
            self.targets = [self.forward(z0, e).detach() + e[2] for e in self.entries]

    # -- exact objective ------------------------------------------------------------------------------------------

    def residual_sq_exact(self, z: Tensor) -> float:
        total = 0.0
        zz = z.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            for entry, target in zip(self.entries, self.targets, strict=True):
                total += float(((self.forward(zz, entry) - target).double() ** 2).sum().item())
        return total

    def solve_exact(self, lam: float, z_init: Tensor, max_iters: int, tol: float) -> tuple[Tensor, dict[str, Any]]:
        z = z_init.detach().clone().requires_grad_(True)
        ds = self.data_sq
        optimizer = torch.optim.LBFGS(
            [z],
            lr=1.0,
            max_iter=max_iters,
            max_eval=2 * max_iters,
            tolerance_grad=0.0,
            tolerance_change=tol,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            with torch.enable_grad():
                for entry, target in zip(self.entries, self.targets, strict=True):
                    loss = ((self.forward(z, entry) - target).double() ** 2).sum()
                    (loss / ds).backward()
                    total += float(loss.item())
                reg = lam * (z[self.weight_mask].double() ** 2).sum()
                (reg / ds).backward()
                total += float(reg.item())
            return torch.tensor(total / ds, dtype=torch.float64)

        first = float(closure())
        optimizer.step(closure)
        state = optimizer.state[optimizer._params[0]]
        iters = int(state.get("n_iter", 0))
        z_out = z.detach().clone()
        final = self.residual_sq_exact(z_out) + lam * float((z_out[self.weight_mask].double() ** 2).sum().item())
        return z_out, {
            "iterations": iters,
            "func_evals": int(state.get("func_evals", 0)),
            "final_loss": final,
            "converged": iters < max_iters,
            "initial_loss": first * ds,
        }

    # -- linearised objective ---------------------------------------------------------------------------------------

    def _jvp(self, entry, z0: Tensor, v: Tensor) -> Tensor:
        def fn(zz):
            return self.forward(zz, entry)

        if self._forward_ad:
            try:
                return torch.func.jvp(fn, (z0,), (v,))[1]
            except (RuntimeError, NotImplementedError):
                self._forward_ad = False
        zr = z0.detach().clone().requires_grad_(True)
        out = fn(zr)
        u = torch.zeros_like(out, requires_grad=True)
        (g,) = torch.autograd.grad(out, zr, u, create_graph=True)
        (jv,) = torch.autograd.grad(g, u, v)
        return jv.detach()

    def _vjp(self, entry, z0: Tensor, cotangent: Tensor) -> Tensor:
        zr = z0.detach().clone().requires_grad_(True)
        out = self.forward(zr, entry)
        (g,) = torch.autograd.grad(out, zr, cotangent)
        return g.detach()

    def jtj(self, v: Tensor) -> Tensor:
        z0 = self.zeros().requires_grad_(True)
        total = torch.zeros(self.size, dtype=torch.float64, device=self.device)
        vv = v.to(self.dtype)
        with torch.enable_grad():
            for entry in self.entries:
                jv = self._jvp(entry, z0, vv)
                total += self._vjp(entry, z0, jv).double()
        return total

    def jt_d(self) -> Tensor:
        z0 = self.zeros().requires_grad_(True)
        total = torch.zeros(self.size, dtype=torch.float64, device=self.device)
        with torch.enable_grad():
            for entry in self.entries:
                total += self._vjp(entry, z0, entry[2]).double()
        return total

    def residual_sq_linearized(self, z: Tensor) -> float:
        z0 = self.zeros().requires_grad_(True)
        zz = z.detach().to(self.dtype)
        total = 0.0
        with torch.enable_grad():
            for entry in self.entries:
                total += float(((self._jvp(entry, z0, zz) - entry[2]).double() ** 2).sum().item())
        return total

    def solve_linearized(
        self, lam: float, z_init: Tensor, max_iters: int, tol: float, rhs: Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        penalty = lam * self.weight_mask.double()

        def apply(v: Tensor) -> Tensor:
            return self.jtj(v) + penalty * v

        z = z_init.detach().double().clone()
        r = rhs - apply(z) if bool(z.any()) else rhs.clone()
        p = r.clone()
        rs = float((r @ r).item())
        q = -0.5 * float((z @ (rhs + r)).item())  # quadratic 1/2 z^T A z - rhs^T z at z
        converged = rs == 0.0
        iters = 0
        while not converged and iters < max_iters:
            ap = apply(p)
            curvature = float((p @ ap).item())
            if curvature <= 0.0:
                break
            alpha = rs / curvature
            z = z + alpha * p
            r = r - alpha * ap
            drop = 0.5 * alpha * rs
            q -= drop
            rs_new = float((r @ r).item())
            iters += 1
            if drop <= tol * abs(q) or rs_new == 0.0:
                converged = True
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return z.to(self.dtype), {
            "iterations": iters,
            "func_evals": None,
            "final_loss": 2.0 * q + self.data_sq,
            "converged": converged,
        }


def _solve_block(problem: _BlockProblem, method: str, lam: float, z_init: Tensor, config, rhs: Tensor | None):
    if method == "gradient_exact":
        return problem.solve_exact(lam, z_init, config.gradient_fit_max_iters, float(config.gradient_fit_tol))
    return problem.solve_linearized(lam, z_init, config.gradient_fit_max_iters, float(config.gradient_fit_tol), rhs)


def _weight_norm(problem: _BlockProblem, z: Tensor) -> float:
    return float(torch.linalg.norm(z[problem.weight_mask].double()).item())


def _match_norm_lambda(problem, method, lam0, target_norm, config, rhs):
    """Bisection on ``log lam`` (``||dW||_F`` is non-increasing in ``lam``) until within 1 % of ``target_norm``."""
    solves = 0
    z = problem.zeros()
    cache: dict[float, tuple[Tensor, dict[str, Any], float]] = {}

    def evaluate(lam: float):
        nonlocal z, solves
        solved, info = _solve_block(problem, method, lam, z, config, rhs)
        solves += 1
        norm = _weight_norm(problem, solved)
        z = solved
        cache[lam] = (solved, info, norm)
        return norm

    def within(norm: float) -> bool:
        return abs(norm - target_norm) <= _MATCH_NORM_TOL * target_norm

    lam = lam0
    norm = evaluate(lam)
    lo = hi = None  # lo: norm above target (lam too small); hi: norm below target (lam too large)
    if within(norm):
        return lam, cache[lam], solves
    if norm > target_norm:
        lo = lam
        while solves < _MATCH_NORM_MAX_SOLVES:
            lam *= 10.0
            norm = evaluate(lam)
            if within(norm):
                return lam, cache[lam], solves
            if norm < target_norm:
                hi = lam
                break
            lo = lam
    else:
        hi = lam
        while solves < _MATCH_NORM_MAX_SOLVES:
            lam /= 10.0
            norm = evaluate(lam)
            if within(norm):
                return lam, cache[lam], solves
            if norm > target_norm:
                lo = lam
                break
            hi = lam
    best = min(cache, key=lambda k: abs(cache[k][2] - target_norm))
    if lo is None or hi is None:
        return best, cache[best], solves
    while solves < _MATCH_NORM_MAX_SOLVES:
        lam = math.sqrt(lo * hi)
        norm = evaluate(lam)
        if within(norm):
            return lam, cache[lam], solves
        if norm > target_norm:
            lo = lam
        else:
            hi = lam
    best = min(cache, key=lambda k: abs(cache[k][2] - target_norm))
    return best, cache[best], solves


@torch.no_grad()
def fit_direct_residual_gradient_streaming(
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
    native_model=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Gradient (exact / linearized) counterpart of `fit_direct_residual_streaming` (independent depth fit, vision only).

    Same sweep (source base / source FT / target, optionally the native model) and the same ``D_j`` as the closed-form
    streaming fit; positions are processed in chunks of ``config.streaming_position_chunk`` (the recorded block inputs and
    targets of a chunk live on ``device``). Returns ``(corrections, rows)`` with the correction keys of the closed-form fit:
    weight and bias of ``mlp.c_proj`` / ``mlp.c_fc`` / ``attn.out_proj``, or the full ``attn.in_proj_weight`` /
    ``attn.in_proj_bias`` with zeros outside the fitted q/k/v row slices. One row per fitted component (shared fit fields).
    """
    if family_adapter is not None:
        raise NotImplementedError("component_fit=gradient is vision-only")
    method = config.component_fit
    if method not in {"gradient_exact", "gradient_linearized"}:
        raise ValueError(f"unsupported component_fit {method!r}")
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    oracle = config.residual_target == "native_oracle"
    if oracle and native_model is None:
        raise ValueError("residual_target='native_oracle' needs native_model (the native fine-tuned target)")
    components = order_components(config.components)
    shim = _layout_for(None)
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    for j in range(pairing.target_depth):
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
    positions = resolve_fit_blocks(config, pairing.target_depth)
    groups = target_groups(pairing.pairing)
    share = config.pair_target_split == "per_block_share"
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    chunk_size = config.streaming_position_chunk or len(positions)
    chunks = [positions[s : s + chunk_size] for s in range(0, len(positions), chunk_size)]
    match_norm = config.gradient_ridge == "match_norm"
    residual_writer = components[0] in {"attn.out_proj", "mlp.c_proj"}
    feature_kind = COMPONENT_INPUT_KIND[components[0]]

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        target_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        for m in unique_models.values():
            m.to(device).eval()
        blocks = shim.blocks(target_model)

        for chunk in chunks:
            distinct_i = sorted({pairing.pairing[pos] for pos in chunk})
            src_requests = {str(i): (i, "boundary") for i in distinct_i}
            tgt_requests: dict[str, tuple[int, str]] = {}
            for pos in chunk:
                tgt_requests[f"{pos}.out"] = (pos, "boundary")
                tgt_requests[f"{pos}.feat"] = (pos, feature_kind)
                if match_norm and COMPONENT_INPUT_KIND["mlp.c_proj"] != feature_kind:
                    tgt_requests[f"{pos}.cproj"] = (pos, COMPONENT_INPUT_KIND["mlp.c_proj"])
            moments = {pos: _FeatureMoments() for pos in chunk}
            cproj_stats: dict[int, ResidualSufficientStatistics] = {}
            cproj_out: dict[int, Tensor] = {}
            if match_norm:
                for pos in chunk:
                    width = int(target_base_state[shim.component_key(pos, "mlp.c_proj", prefixed=True)].shape[0])
                    cproj_out[pos] = _component_effective_out(shim, target_model, pos, "mlp.c_proj", width)
                    cproj_stats[pos] = ResidualSufficientStatistics(device=device)
            entries: dict[int, list[tuple[tuple, dict, Tensor]]] = {pos: [] for pos in chunk}
            latest: dict[int, tuple[tuple, dict]] = {}

            handles = [
                shim.block_module(blocks[pos]).register_forward_pre_hook(_record_call(latest, pos), with_kwargs=True)
                for pos in chunk
            ]
            if oracle:
                sb_gen, sf_gen = _none_stream(len(batches)), _none_stream(len(batches))
                nat_gen = iter_capture_tokens(
                    native_model,
                    batches,
                    {f"{pos}.out": (pos, "boundary") for pos in chunk},
                    device,
                    store_device=device,
                )
            else:
                sb_gen = iter_capture_tokens(
                    source_base_model, source_batches, src_requests, device, store_device=device
                )
                sf_gen = iter_capture_tokens(source_ft_model, source_batches, src_requests, device, store_device=device)
                nat_gen = _none_stream(len(batches))
            tgt_gen = iter_capture_tokens(target_model, batches, tgt_requests, device, store_device=device)
            try:
                with (
                    contextlib.closing(sb_gen),
                    contextlib.closing(sf_gen),
                    contextlib.closing(tgt_gen),
                    contextlib.closing(nat_gen),
                ):
                    for k, (sb, sf, tgt, nat) in enumerate(zip(sb_gen, sf_gen, tgt_gen, nat_gen, strict=True)):
                        for pos in chunk:
                            i = pairing.pairing[pos]
                            out = tgt[f"{pos}.out"]
                            _check_boundary_fingerprint(out, fingerprints[(k, pos)], config, pos, k)
                            out_cpu = out.cpu()
                            if oracle:
                                desired = nat[f"{pos}.out"].cpu() - out_cpu
                            else:
                                b = _aligned([sb[str(i)].cpu()], [out_cpu])[0]
                                f = _aligned([sf[str(i)].cpu()], [out_cpu])[0]
                                desired = (f - b) @ q_by_position[pos]
                                if share and groups[pos][0] > 1:
                                    desired = desired / groups[pos][0]
                            h = tgt[f"{pos}.feat"]
                            moments[pos].update(h)
                            if match_norm:
                                hc = tgt[
                                    f"{pos}.feat"
                                    if feature_kind == COMPONENT_INPUT_KIND["mlp.c_proj"]
                                    else f"{pos}.cproj"
                                ]
                                cproj_stats[pos].update(
                                    hc.cpu().reshape(-1, hc.shape[-1]),
                                    desired.reshape(-1, desired.shape[-1]),
                                    None,
                                    cproj_out[pos],
                                )
                            args, kwargs = latest[pos]
                            ref = args[0]
                            d_raw = desired.to(device)
                            if tuple(ref.shape) != tuple(d_raw.shape):
                                if ref.dim() == 3 and tuple(ref.transpose(0, 1).shape) == tuple(d_raw.shape):
                                    d_raw = d_raw.transpose(0, 1).contiguous()
                                else:
                                    raise ValueError(
                                        f"block {pos} input {tuple(ref.shape)} does not match the target {tuple(d_raw.shape)}"
                                    )
                            entries[pos].append((args, kwargs, d_raw.to(ref.dtype)))
            finally:
                for handle in handles:
                    handle.remove()

            for pos in chunk:
                _fit_position(
                    pos,
                    blocks,
                    shim,
                    components,
                    entries.pop(pos),
                    moments[pos],
                    cproj_stats.get(pos),
                    target_model,
                    target_base_state,
                    config,
                    method,
                    residual_writer,
                    source_coordinates,
                    device,
                    target_corrections,
                    diagnostics,
                )
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    for row in diagnostics:
        row["procrustes_source"] = prepared.get("procrustes_source", "activation")
        if oracle:
            row["residual_target"] = "native_oracle"
            row["procrustes_used"] = False
    annotate_target_split(diagnostics, config, pairing)
    return target_corrections, diagnostics


def _fit_position(
    pos,
    blocks,
    shim,
    components,
    entries,
    moments,
    cproj_stats,
    target_model,
    target_base_state,
    config,
    method,
    residual_writer,
    source_coordinates,
    device,
    target_corrections,
    diagnostics,
):
    block = shim.block_module(blocks[pos])
    problem = _BlockProblem(block, components, entries, device)
    desired_norm = problem.data_sq**0.5
    if moments.n <= 1:
        raise ValueError("the empirical-Bayes ridge requires at least two activation rows")
    factor = 1.0
    if residual_writer:
        width = int(entries[0][2].shape[-1])
        t_out = _component_effective_out(shim, target_model, pos, components[0], width).double()
        factor = float((t_out**2).sum().item()) / width
    lam_eb = moments.centred_trace() / float(moments.n - 1) * factor
    info: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    if problem.data_sq == 0.0:
        z = problem.zeros()
        lam = lam_eb
        info = {"iterations": 0, "func_evals": 0, "final_loss": 0.0, "converged": True}
    else:
        problem.prepare_targets()
        rhs = problem.jt_d() if method == "gradient_linearized" else None
        if config.gradient_ridge == "match_norm":
            assert cproj_stats is not None
            _, cf_diag = cproj_stats.solve(
                ridge_relative=config.ridge_relative, ridge_estimator="empirical_bayes", exact_form=True
            )
            target_norm = float(cf_diag["correction_norm"])
            extra["match_norm_target"] = target_norm
            if target_norm == 0.0:
                z, info, lam = (
                    problem.zeros(),
                    {"iterations": 0, "func_evals": 0, "final_loss": problem.data_sq},
                    math.inf,
                )
                info["converged"] = True
            else:
                lam, (z, info, norm), solves = _match_norm_lambda(problem, method, lam_eb, target_norm, config, rhs)
                extra["match_norm_achieved"] = norm
                extra["match_norm_relative_gap"] = abs(norm - target_norm) / target_norm
                extra["match_norm_solves"] = solves
        else:
            lam = lam_eb
            z, info = _solve_block(problem, method, lam, problem.zeros(), config, rhs)
    exact_after = problem.residual_sq_exact(z) ** 0.5 if problem.data_sq else 0.0
    lin_after = problem.residual_sq_linearized(z) ** 0.5 if problem.data_sq else 0.0
    # before the fit (dW = 0, db = 0) both residuals equal ||D_j|| by construction
    joint_norm = float(torch.linalg.norm(z.double()).item())
    for component, kind, start, numel, shape in problem.segments:
        w_name, b_name = _PARAM_NAMES[component]
        key = shim.component_key(pos, component, prefixed=True)
        bias_key = _family_bias_key(key)
        name_key = key if kind == "w" else bias_key
        chunk = z[start : start + numel].reshape(shape).detach().float().cpu()
        if component in _PACKED_QKV_SLICE:
            full_shape = tuple(target_base_state[name_key].shape)
            if name_key not in target_corrections:
                target_corrections[name_key] = torch.zeros(full_shape, dtype=torch.float32)
            d = full_shape[0] // 3
            s = _PACKED_QKV_SLICE[component]
            target_corrections[name_key][s * d : (s + 1) * d] = chunk
        else:
            target_corrections[name_key] = chunk
    for component in components:
        own = {seg[1]: z[seg[2] : seg[2] + seg[3]].double() for seg in problem.segments if seg[0] == component}
        diagnostics.append(
            {
                "mode": "gradient_fit",
                "component": component,
                "component_fit": method,
                "position": pos,
                "source_position": source_coordinates[pos],
                "desired_norm": desired_norm,
                "n_rows": moments.n,
                "ridge": lam,
                "ridge_eb_input": lam_eb,
                "ridge_estimator": "empirical_bayes",
                "gradient_ridge": config.gradient_ridge,
                "relative_residual_before": 1.0 if desired_norm else 0.0,
                "relative_residual_before_linearized": 1.0 if desired_norm else 0.0,
                "relative_residual_after": exact_after / desired_norm if desired_norm else 0.0,
                "relative_residual_after_linearized": lin_after / desired_norm if desired_norm else 0.0,
                "correction_norm": float(torch.linalg.norm(own["w"]).item()),
                "bias_norm": float(torch.linalg.norm(own["b"]).item()) if "b" in own else 0.0,
                "joint_correction_norm": joint_norm,
                "gradient_fit_group": list(components),
                "gradient_fit_iterations": info.get("iterations"),
                "gradient_fit_func_evals": info.get("func_evals"),
                "gradient_fit_final_loss": info.get("final_loss"),
                "gradient_fit_converged": bool(info.get("converged", False)),
                "gradient_fit_heldout": None,  # fidelity_holdout machinery is closed-form specific: not provided
                **extra,
            }
        )


__all__ = ["fit_direct_residual_gradient_streaming"]
