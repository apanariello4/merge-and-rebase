"""Closed-form pre-image correction of the attention V projection (``v_fit="preimage"``, vision only).

Per target block ``j`` (stock ``nn.MultiheadAttention``, packed ``in_proj``), with ``X`` the ``ln_1`` output and, per head ``h``,
``A_h`` the softmax attention probabilities and ``Z_h = A_h X`` (``n x d_in``): a change ``(dV_h, db_h)`` of head ``h``'s V slice
(``d_head x d_in`` and ``d_head``) moves the pre-``out_proj`` rows EXACTLY by ``Z_h dV_h^T + 1 db_h^T`` (the rows of ``A_h`` sum to
1). The residual-stream change is, to first order (the MLP is assumed not to react, as in the closed-form ``attn.out_proj`` fit),
``dR = dH L`` with the write map ``L = W_O^T diag(ls_1 gamma)`` (``d x d``; rows indexed by the concatenated head dimensions).

The block-end target ``D_j`` (the one every closed-form fit uses) is pulled back through ``L``: ``T = D_j L^+`` (float64; ``L``
square and full rank gives ``T L = D_j``). Each head then solves its own ridge regression ``Z_h -> T_h`` with an unpenalised
intercept and the empirical-Bayes ridge of the O/D path: `ResidualSufficientStatistics` with features ``Z_h``, error ``T_h`` and
identity output map (``G = I``, so ``lam_h = tr(Z~_h^T Z~_h) / (n - 1)``).

The weight/bias corrections are returned as the FULL ``attn.in_proj_weight`` / ``attn.in_proj_bias`` shapes with zeros outside the
V rows ``[2d, 3d)``.

Decoder (GQA) path: not implemented (``NotImplementedError``); the natural extension stacks the ``Z_h`` rows of the query heads
that share a key/value head.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ...discrete_layer_match import DiscreteLayerPairing
from .capture import iter_capture_tokens
from .config import DirectResidualConfig, annotate_target_split, order_components, resolve_fit_blocks
from .fit import ResidualSufficientStatistics
from .layouts import _PACKED_QKV_SLICE, _aligned, _family_bias_key, _layout_for
from .streaming import _check_boundary_fingerprint

Tensor = torch.Tensor

_V_COMPONENT = "attn.v_proj"


def vfit_write_map(shim, block, dtype=torch.float64) -> Tensor:
    """``L = W_O^T diag(ls_1 gamma)`` (``d x d``): pre-``out_proj`` rows -> residual-stream rows (``out_proj.bias`` is a constant)."""
    inner = shim.block_module(block)
    w_o = inner.attn.out_proj.weight.detach().to(dtype)
    scale = shim.component_scale_module(block, "attn.out_proj")
    if isinstance(scale, nn.Identity):
        return w_o.T.contiguous()
    gamma = getattr(scale, "gamma", None)
    if gamma is None or gamma.ndim != 1 or gamma.shape[0] != w_o.shape[0]:
        raise ValueError("Unsupported non-diagonal target LayerScale")
    return (w_o.T * gamma.detach().to(dtype)[None, :]).contiguous()


class VFitBlockStatistics:
    """Streaming statistics of one block's per-head V pre-image fit.

    ``update(z, desired)`` takes the mixed inputs ``z`` ``[n, H*d_in]`` (head-major, `_stock_mha_head_mixed_input`) and the
    block-end target rows ``desired`` ``[n, d]``. ``solve`` returns the V-slice correction; ``update_prediction`` (second sweep,
    after ``solve``) accumulates ``||P L||^2`` so the residual-space error of the correction is exact.
    """

    def __init__(self, write_map: Tensor, num_heads: int, *, device=None) -> None:
        d = int(write_map.shape[0])
        if write_map.shape != (d, d) or d % num_heads:
            raise ValueError("write_map must be square with width divisible by num_heads")
        self.device = device
        self.heads = int(num_heads)
        self.d = d
        self.dh = d // self.heads
        wm = write_map.detach().to(torch.float64)
        if device is not None:
            wm = wm.to(device)
        self.write_map = wm
        svals = torch.linalg.svdvals(wm)
        self.cond = float((svals[0] / svals[-1]).item()) if float(svals[-1]) > 0 else float("inf")
        self.pinv = torch.linalg.pinv(wm)
        self.eye = torch.eye(self.dh, dtype=torch.float32)
        self.stats = [ResidualSufficientStatistics(device=device) for _ in range(self.heads)]
        self.cross_w: list[Tensor | None] = [None] * self.heads  # Z_h^T (D L_h^T), d_in x dh
        self.cross_b: list[Tensor | None] = [None] * self.heads  # 1^T (D L_h^T), dh
        self.d_sq = 0.0
        self.t_sq = 0.0
        self.preimage_err_sq = 0.0
        self.n_rows = 0
        self.dv: Tensor | None = None
        self.db: Tensor | None = None
        self.pl_sq = 0.0

    def target(self, desired: Tensor) -> Tensor:
        """Pre-image ``T = D L^+`` (float64)."""
        return desired.to(device=self.write_map.device, dtype=torch.float64) @ self.pinv

    def update(self, z: Tensor, desired: Tensor) -> None:
        z = z.reshape(-1, z.shape[-1])
        desired = desired.reshape(-1, desired.shape[-1])
        if z.shape[0] != desired.shape[0] or z.shape[1] % self.heads:
            raise ValueError("mixed inputs and target rows do not match")
        d_in = z.shape[1] // self.heads
        dev = self.write_map.device
        d64 = desired.to(device=dev, dtype=torch.float64)
        t = d64 @ self.pinv
        self.d_sq += float((d64**2).sum().item())
        self.t_sq += float((t**2).sum().item())
        self.preimage_err_sq += float(((t @ self.write_map - d64) ** 2).sum().item())
        self.n_rows += int(z.shape[0])
        z64 = z.to(device=dev, dtype=torch.float64)
        for h in range(self.heads):
            sl = slice(h * self.dh, (h + 1) * self.dh)
            zh = z64[:, h * d_in : (h + 1) * d_in]
            self.stats[h].update(zh, t[:, sl], None, self.eye)
            dl = d64 @ self.write_map[sl, :].T  # D L_h^T, n x dh
            cw, cb = zh.T @ dl, dl.sum(dim=0)
            if self.cross_w[h] is None:
                self.cross_w[h], self.cross_b[h] = cw, cb
            else:
                self.cross_w[h] = self.cross_w[h] + cw
                self.cross_b[h] = self.cross_b[h] + cb

    def solve(self, config: DirectResidualConfig) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """``(dV [d, d_in], db [d], diagnostics)`` in float64; per-head independent ridge solves."""
        dvs, dbs, ridges = [], [], []
        before_sq = after_sq = 0.0
        cross = 0.0
        for h, st in enumerate(self.stats):
            x, diag = st.solve(
                ridge_relative=config.ridge_relative, ridge_estimator=config.ridge_estimator, exact_form=True
            )
            dv_h = x.to(torch.float64)  # [dh, d_in]
            db_h = diag["bias_correction"].to(torch.float64)
            dvs.append(dv_h)
            dbs.append(db_h)
            ridges.append(float(diag["ridge"]))
            before_sq += diag["residual_norm_before"] ** 2
            after_sq += diag["residual_norm_after"] ** 2
            cross += float((dv_h.T.to(self.cross_w[h].device) * self.cross_w[h]).sum().item())
            cross += float((db_h.to(self.cross_b[h].device) * self.cross_b[h]).sum().item())
        dv = torch.cat(dvs, dim=0)
        db = torch.cat(dbs, dim=0)
        self.dv, self.db = dv.to(self.write_map.device), db.to(self.write_map.device)
        self._cross = cross
        info = {
            "ridge_per_head": ridges,
            "relative_residual_preimage": (after_sq**0.5) / (self.t_sq**0.5) if self.t_sq else 0.0,
            "preimage_norm": self.t_sq**0.5,
            "preimage_norm_before": before_sq**0.5,
        }
        return dv, db, info

    def update_prediction(self, z: Tensor) -> None:
        """Second sweep: add ``||P L||^2`` of the fitted first-order change ``P = Z dV^T + 1 db^T`` (all heads)."""
        if self.dv is None:
            raise RuntimeError("solve() must run before update_prediction()")
        z = z.reshape(-1, z.shape[-1])
        d_in = z.shape[1] // self.heads
        z64 = z.to(device=self.write_map.device, dtype=torch.float64)
        parts = [
            z64[:, h * d_in : (h + 1) * d_in] @ self.dv[h * self.dh : (h + 1) * self.dh].T for h in range(self.heads)
        ]
        pred = (torch.cat(parts, dim=1) + self.db) @ self.write_map
        self.pl_sq += float((pred**2).sum().item())

    def residual_space_error_sq(self) -> float:
        """``||P L - D||^2 = ||D||^2 - 2 <P L, D> + ||P L||^2`` over the calibration rows (needs the second sweep)."""
        return max(0.0, self.d_sq - 2.0 * self._cross + self.pl_sq)


@torch.no_grad()
def fit_direct_residual_vfit_streaming(
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
    stats_factory=None,
    native_model=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """V pre-image counterpart of `fit_direct_residual_streaming` (independent depth fit, vision only).

    Same sweep (source base / source FT / pristine target) and the same ``D_j = (f - b) Q_j`` as the closed-form streaming fit.
    Sweep 1 accumulates the per-head statistics; sweep 2 (target only) measures the fitted correction in residual space.
    Memory: the mixed inputs ``Z`` of one batch are ``[B, T, H*d]`` per block of the chunk, so lower
    ``streaming_position_chunk`` for large ViTs. Returns the corrections (full ``in_proj_weight`` / ``in_proj_bias`` with zeros
    outside the V rows) and one diagnostics row per block (component ``attn.v_proj``).
    """
    if family_adapter is not None:
        # GQA later: stack the Z_h rows of the query heads that share a key/value head.
        raise NotImplementedError("v_fit='preimage' is vision-only (stock nn.MultiheadAttention)")
    if config.v_fit != "preimage":
        raise ValueError("fit_direct_residual_vfit_streaming requires v_fit='preimage'")
    if config.residual_target != "transported_delta":
        raise ValueError("v_fit='preimage' requires residual_target='transported_delta'")
    if order_components(config.components) != (_V_COMPONENT,):
        raise ValueError("v_fit='preimage' requires components=['attn.v_proj']")
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    shim = _layout_for(None)
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    for j in range(pairing.target_depth):
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
    positions = resolve_fit_blocks(config, pairing.target_depth)
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    chunk_size = config.streaming_position_chunk or len(positions)
    chunks = [positions[s : s + chunk_size] for s in range(0, len(positions), chunk_size)]

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    key = {pos: shim.component_key(pos, _V_COMPONENT, prefixed=True) for pos in positions}
    try:
        target_model.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()}, strict=True)
        for m in unique_models.values():
            m.to(device).eval()
        blocks = shim.blocks(target_model)

        for chunk in chunks:
            for pos in chunk:
                attn = shim.attn_module(blocks[pos])
                if not isinstance(attn, nn.MultiheadAttention) or attn.in_proj_bias is None:
                    raise NotImplementedError("v_fit='preimage' needs a stock nn.MultiheadAttention with in_proj_bias")
            acc = {
                pos: VFitBlockStatistics(
                    vfit_write_map(shim, blocks[pos]),
                    int(shim.attn_module(blocks[pos]).num_heads),
                    device=device,
                )
                for pos in chunk
            }
            distinct_i = sorted({pairing.pairing[pos] for pos in chunk})
            src_requests = {str(i): (i, "boundary") for i in distinct_i}
            tgt_requests: dict[str, tuple[int, str]] = {}
            for pos in chunk:
                tgt_requests[f"{pos}.out"] = (pos, "boundary")
                tgt_requests[f"{pos}.z"] = (pos, "attn_mixed_input")
            sb_gen = iter_capture_tokens(source_base_model, source_batches, src_requests, device, store_device=device)
            sf_gen = iter_capture_tokens(source_ft_model, source_batches, src_requests, device, store_device=device)
            tgt_gen = iter_capture_tokens(target_model, batches, tgt_requests, device, store_device=device)
            with contextlib.closing(sb_gen), contextlib.closing(sf_gen), contextlib.closing(tgt_gen):
                for k, (sb, sf, tgt) in enumerate(zip(sb_gen, sf_gen, tgt_gen, strict=True)):
                    for pos in chunk:
                        i = pairing.pairing[pos]
                        out = tgt[f"{pos}.out"]
                        _check_boundary_fingerprint(out, fingerprints[(k, pos)], config, pos, k)
                        out_cpu = out.cpu()
                        b = _aligned([sb[str(i)].cpu()], [out_cpu])[0]
                        f = _aligned([sf[str(i)].cpu()], [out_cpu])[0]
                        desired = (f - b) @ q_by_position[pos]
                        acc[pos].update(tgt[f"{pos}.z"], desired)

            solved: dict[int, dict[str, Any]] = {}
            for pos in chunk:
                dv, db, info = acc[pos].solve(config)
                solved[pos] = info
                base_w = target_base_state[key[pos]]
                bias_key = _family_bias_key(key[pos])
                if bias_key not in target_base_state:
                    raise RuntimeError(f"Target is missing {bias_key}; the V intercept needs an existing in_proj_bias")
                d = int(base_w.shape[1])
                if base_w.shape[0] != 3 * d or dv.shape != (d, d):
                    raise RuntimeError("Unexpected packed in_proj shape for the V correction")
                s = _PACKED_QKV_SLICE[_V_COMPONENT]
                w_corr = torch.zeros(tuple(base_w.shape), dtype=torch.float32)
                b_corr = torch.zeros(tuple(target_base_state[bias_key].shape), dtype=torch.float32)
                w_corr[s * d : (s + 1) * d] = dv.float().cpu()
                b_corr[s * d : (s + 1) * d] = db.float().cpu()
                if not (torch.isfinite(w_corr).all() and torch.isfinite(b_corr).all()):
                    raise RuntimeError("V pre-image fit produced a non-finite correction")
                target_corrections[key[pos]] = w_corr
                target_corrections[bias_key] = b_corr

            # Sweep 2 (pristine target only): ||P L||^2 for the exact residual-space error of the fitted correction.
            z_gen = iter_capture_tokens(
                target_model,
                batches,
                {f"{pos}.z": (pos, "attn_mixed_input") for pos in chunk},
                device,
                store_device=device,
            )
            with contextlib.closing(z_gen):
                for z_batch in z_gen:
                    for pos in chunk:
                        acc[pos].update_prediction(z_batch[f"{pos}.z"])

            for pos in chunk:
                a, info = acc[pos], solved[pos]
                d_norm = a.d_sq**0.5
                dv_cpu = a.dv.cpu()
                ridges = info["ridge_per_head"]
                row = {
                    "mode": "direct_target",
                    "component": _V_COMPONENT,
                    "v_fit": "preimage",
                    "position": pos,
                    "source_position": source_coordinates[pos],
                    "n_rows": a.n_rows,
                    "desired_norm": d_norm,
                    "effect_before_norm": 0.0,
                    "relative_residual_before": 1.0 if d_norm else 0.0,
                    "relative_residual_after": (a.residual_space_error_sq() ** 0.5 / d_norm) if d_norm else 0.0,
                    "relative_residual_preimage": info["relative_residual_preimage"],
                    "preimage_rel_error": (a.preimage_err_sq**0.5 / d_norm) if d_norm else 0.0,
                    "preimage_norm": info["preimage_norm"],
                    "out_proj_cond": a.cond,
                    "ridge": sum(ridges) / len(ridges),
                    "ridge_min": min(ridges),
                    "ridge_max": max(ridges),
                    "ridge_per_head": ridges,
                    "ridge_estimator": config.ridge_estimator,
                    "correction_norm": float(torch.linalg.norm(dv_cpu).item()),
                    "bias_norm": float(torch.linalg.norm(a.db.cpu()).item()),
                    "correction_rank": int(torch.linalg.matrix_rank(dv_cpu).item()),
                    "exact_form": True,
                    "procrustes_source": prepared.get("procrustes_source", "activation"),
                }
                diagnostics.append(row)
            del acc
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
    annotate_target_split(diagnostics, config, pairing)
    return target_corrections, diagnostics


__all__ = ["VFitBlockStatistics", "fit_direct_residual_vfit_streaming", "vfit_write_map"]
