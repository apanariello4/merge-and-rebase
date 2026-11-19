"""Update synthesis variants of ``anchors`` (docs/.local/AGENT_NEXT_PHASE.md): D2 ridge with conservation (O2).

D2 (``synthesis='ridge_preserve'``, after MEMIT's covariance term): per block, the ``c_proj`` update must give the
desired displacement on the task rows AND change the component's output on generic inputs as little as possible.
Weight and bias are estimated together (the bias also moves the output on generic inputs), intercept unpenalised:

    min_{dW, beta} (1/n)||H dW + 1 beta^T - D||^2 + (rho_G/m)||G dW + 1 beta^T||^2 + lambda' ||dW||^2

with X = [H 1], X_G = [G 1], Lambda = diag(lambda', ..., lambda', 0):

    A = (1/n) X^T X + (rho_G/m) X_G^T X_G + Lambda,   Theta = solve(A, (1/n) X^T D),   dW = Theta[:d], beta = Theta[d]

lambda' = lambda_Ariadne / n (Ariadne's rule on the unnormalised sums), so rho_G = 0 is exactly Ariadne's ridge.
H, D come from Ariadne's accumulated statistics (uncentred sums); G = inputs of the target base's ``c_proj`` on the
Tiny-ImageNet anchor images (never the mixed Vision8 pool), high-norm tokens dropped as in the anchor filter, a
seeded subsample of ``preserve_tokens`` tokens per block, all blocks captured in one sweep before the fit.

Decoders (AGENT_SHARED_SPACE.md S0, LLM track): G = inputs of the target base's ``mlp.down_proj`` on the content
tokens of the frame's anchor sequences (WikiText-103 pool; padding and the first token of every sequence excluded,
as everywhere on decoders; any ``text_units``: all content tokens of the target's own tokenization), the same seeded
``preserve_tokens`` cap per block (default 32768; the LLM pools of the runs hold ~8k anchor content tokens, so all
are used). ``G^T G`` is built one block at a time inside the solve (``LazyGenericInputs``) to bound host memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .._ariadne.capture import iter_capture_tokens
from .._ariadne.layouts import COMPONENT_INPUT_KIND

Tensor = torch.Tensor


class GenericInputStats:
    """Uncentred float64 sums of the generic inputs G of one block: G^T G, sum of rows, m."""

    def __init__(self) -> None:
        self.gg: Tensor | None = None
        self.sg: Tensor | None = None
        self.m = 0

    def update(self, g: Tensor) -> None:
        g = g.reshape(-1, g.shape[-1]).to(torch.float64)
        if g.shape[0] == 0:
            return
        if self.gg is None:
            self.gg = torch.zeros(g.shape[1], g.shape[1], dtype=torch.float64)
            self.sg = torch.zeros(g.shape[1], dtype=torch.float64)
        self.gg += (g.T @ g).cpu()
        self.sg += g.sum(0).cpu()
        self.m += int(g.shape[0])


@torch.no_grad()
def capture_generic_inputs(
    target_model,
    batches,
    depth: int,
    *,
    component: str,
    tokens_per_block: int,
    seed: int,
    device,
    norm_filter: Mapping[str, Any] | None = None,
    blocks: list[int] | None = None,
    family_adapter=None,
) -> dict[int, GenericInputStats]:
    """``{j: GenericInputStats}`` of the target base's ``component`` inputs on ``batches`` (one streamed sweep).

    Tokens are subsampled with a seeded per-batch draw of ``ceil(tokens_per_block / len(batches))`` rows (the same
    token rows at every block). ``norm_filter`` (a stored ``AnchorState.norm_filter``): tokens whose block output
    norm ``||x - mu_j||`` exceeds ``k * median_j`` are dropped first (the anchor filter's rule).

    ``blocks``: only these blocks (default: all ``depth``); the draw depends on the batches' row counts only, so
    every subset of blocks sees the same token rows. ``family_adapter``: decoders (``mlp.c_proj`` read as
    ``mlp.down_proj``); batches carrying a row mask (``paired_calibration(family_adapter=...)``) yield their
    content rows only, as one ``[1, N, d]`` bank, and the draw is over those rows.
    """
    kind = COMPONENT_INPUT_KIND[component]
    blocks = list(range(depth)) if blocks is None else [int(j) for j in blocks]
    requests = {f"{j}.h": (j, kind) for j in blocks}
    if norm_filter is not None:
        requests.update({f"{j}.out": (j, "boundary") for j in blocks})
    per_batch = -(-int(tokens_per_block) // max(len(batches), 1))
    gen = torch.Generator().manual_seed(int(seed))
    stats = {j: GenericInputStats() for j in blocks}
    for values in iter_capture_tokens(target_model, batches, requests, device, family_adapter=family_adapter):
        h0 = values[f"{blocks[0]}.h"]
        n_rows = h0.shape[0] * h0.shape[1]
        pick = torch.randperm(n_rows, generator=gen)[: min(per_batch, n_rows)]
        for j in blocks:
            rows = values[f"{j}.h"].reshape(n_rows, -1)[pick.to(values[f"{j}.h"].device)]
            if norm_filter is not None:
                out = values[f"{j}.out"].reshape(n_rows, -1)[pick.to(values[f"{j}.out"].device)].to(torch.float64)
                mu = norm_filter["mu"][j].to(out.device, torch.float64)
                keep = (out - mu).norm(dim=-1) <= float(norm_filter["k"]) * float(norm_filter["median"][j])
                rows = rows[keep.to(rows.device)]
            stats[j].update(rows)
    return stats


class LazyGenericInputs:
    """``{j: GenericInputStats}`` for decoders, built one block at a time when the solve hook first asks for it.

    Memory (AGENT_SHARED_SPACE.md S0 on LLMs): a decoder's ``down_proj`` input is wide (Qwen2.5-1.5B: 8960,
    Llama-3.2-3B: 8192, 28 blocks), so the float64 ``G^T G`` of every block at once would take ~18 GB of host RAM.
    Here only the block being solved is held (one ``d x d`` float64 matrix, 0.64 GB at d = 8960), captured by one
    extra target sweep over the anchor sequences that requests that block's input only (``capture_generic_inputs``
    with ``blocks=[j]``; same seeded rows at every block). Peak addition over the plain ridge: that matrix plus the
    ``(d+1) x (d+1)`` float64 system of ``solve_preserve`` (~3 GB at d = 8960), freed before the next block.

    ``before_capture(j)`` (optional) runs before each sweep (the caller checks that the target is at its base weights:
    the shared ridge of the independent fit keeps it there while the hook runs).
    """

    def __init__(self, target_model, batches, *, component: str, tokens_per_block: int, seed: int, device,
                 norm_filter: Mapping[str, Any] | None = None, family_adapter=None, before_capture=None) -> None:
        self._model = target_model
        self._batches = batches
        self._kw = dict(component=component, tokens_per_block=tokens_per_block, seed=seed, device=device,
                        norm_filter=norm_filter, family_adapter=family_adapter)
        self._before = before_capture
        self._block: int | None = None
        self._stats: GenericInputStats | None = None
        self.captures = 0

    def __getitem__(self, j: int) -> GenericInputStats:
        j = int(j)
        if self._block != j:
            self._block, self._stats = None, None  # release the previous block before capturing the next
            if self._before is not None:
                self._before(j)
            stats = capture_generic_inputs(self._model, self._batches, j + 1, blocks=[j], **self._kw)[j]
            self._block, self._stats = j, stats
            self.captures += 1
        return self._stats


def solve_preserve(
    stats, generic: GenericInputStats, rho_g: float, lam: float
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """``(dW [d_in, d_out], beta [d_out], info)`` of D2 from Ariadne's accumulated ``ResidualSufficientStatistics``
    (identity output map required) and the generic-input sums; ``lam`` is Ariadne's ridge (``diag['ridge']``)."""
    s = stats.s.detach().to("cpu", torch.float64)
    g_out = stats.g.detach().to("cpu", torch.float64)
    if not torch.allclose(g_out, torch.eye(g_out.shape[0], dtype=torch.float64)):
        raise ValueError("ridge_preserve needs an identity output map (no LayerScale)")
    b = stats.b.detach().to("cpu", torch.float64)
    sa = stats.sum_a.detach().to("cpu", torch.float64)
    se = stats.sum_e.detach().to("cpu", torch.float64)
    n = float(stats.n_rows)
    d = s.shape[0]
    xtx = torch.zeros(d + 1, d + 1, dtype=torch.float64)
    xtx[:d, :d], xtx[:d, d], xtx[d, :d], xtx[d, d] = s, sa, sa, n
    xtd = torch.cat([b, se[None]], dim=0)
    a = xtx / n
    del xtx  # (d+1)^2 float64 buffers are large on decoders (d = 8960): keep at most three alive
    if rho_g > 0.0:
        m = float(generic.m)
        xg = torch.zeros(d + 1, d + 1, dtype=torch.float64)
        xg[:d, :d], xg[:d, d], xg[d, :d], xg[d, d] = generic.gg, generic.sg, generic.sg, m
        a = a + (float(rho_g) / m) * xg
        del xg
    a[torch.arange(d), torch.arange(d)] += float(lam) / n
    theta = torch.linalg.solve(a, xtd / n)
    dw, beta = theta[:d], theta[d]
    # residual on the task rows from the sums: sum_e2 - 2<dW, b> - 2 beta.se + tr(dW^T s dW) + 2 beta^T dW^T sa + n|beta|^2
    res = (stats.sum_e2 - 2 * float((dw * b).sum()) - 2 * float(beta @ se) + float((dw * (s @ dw)).sum())
           + 2 * float(beta @ (dw.T @ sa)) + n * float(beta @ beta))
    return dw, beta, {"preserve_rho_g": float(rho_g), "residual_norm_after": max(res, 0.0) ** 0.5}


def generic_change(dw: Tensor, beta: Tensor, weight: Tensor, generic: GenericInputStats) -> float:
    """``||G dW + 1 beta^T||_F / ||G W||_F``: relative change of the component's output on the generic inputs
    (``weight`` the base ``[d_out, d_in]`` matrix; its bias is left out of the denominator)."""
    gg, sg, m = generic.gg.to("cpu", torch.float64), generic.sg.to("cpu", torch.float64), float(generic.m)
    dw, beta = dw.to("cpu", torch.float64), beta.to("cpu", torch.float64)
    change = float((dw * (gg @ dw)).sum()) + 2 * float(beta @ (dw.T @ sg)) + m * float(beta @ beta)
    w = weight.to("cpu", torch.float64).T
    base = float((w * (gg @ w)).sum())
    return (max(change, 0.0) / base) ** 0.5 if base > 0 else float("nan")


def reduce_rank(stats, dw: Tensor, rank: int) -> tuple[Tensor, Tensor, float]:
    """D5: ``dW_r = dW V_r V_r^T`` with V_r the top-``rank`` eigenvectors of ``dW^T H~^T H~ dW`` (the fitted task
    displacement's scatter, H centred); the unpenalised intercept is recomputed, ``beta = mu_D - dW_r^T mu_H``.
    Returns ``(dW_r, beta, kept energy fraction)``."""
    n = float(stats.n_rows)
    s = stats.s.detach().to("cpu", torch.float64)
    sa = stats.sum_a.detach().to("cpu", torch.float64)
    se = stats.sum_e.detach().to("cpu", torch.float64)
    sc = s - torch.outer(sa, sa) / n
    m = dw.T @ sc @ dw
    ev, v = torch.linalg.eigh(0.5 * (m + m.T))
    order = ev.argsort(descending=True)
    v_r = v[:, order[: int(rank)]]
    kept = float(ev[order[: int(rank)]].clamp_min(0).sum() / ev.clamp_min(0).sum())
    dw_r = dw @ v_r @ v_r.T
    return dw_r, se / n - dw_r.T @ (sa / n), kept


def make_synthesis_hook(
    *,
    generic: Mapping[int, GenericInputStats] | None = None,
    rho_g: float = 0.0,
    base_weights: Mapping[int, Tensor] | None = None,
    rank: int | None = None,
):
    """``solve_hook`` for ``fit_direct_residual``: Ariadne's solve (its diagnostics and lambda), then D2 when
    ``generic`` is given, then D5 when ``rank`` is given. Records the generic-input change of both the plain ridge
    and the final update when ``generic`` is given."""

    def hook(pos, component, stats, config):
        correction, diag = stats.solve(
            ridge_relative=config.ridge_relative, ridge_estimator=config.ridge_estimator, exact_form=config.exact_form
        )
        if component != "mlp.c_proj":
            raise ValueError("anchors synthesis variants are implemented for mlp.c_proj only")
        # Ariadne solves on the fit device; D2 / D5 and the diagnostics run on the host (float64 statistics there)
        dw, beta = correction.T.detach().to("cpu", torch.float64), diag["bias_correction"].detach().to("cpu", torch.float64)
        if generic is not None:
            diag["generic_change_ridge"] = generic_change(dw, beta, base_weights[pos], generic[pos])
            dw, beta, info = solve_preserve(stats, generic[pos], rho_g, float(diag["ridge"]))
            diag.update(info)
            diag["generic_tokens"] = generic[pos].m
        if rank is not None:
            dw, beta, kept = reduce_rank(stats, dw, rank)
            diag["update_rank"] = int(rank)
            diag["update_rank_energy_kept"] = kept
        if generic is not None:
            diag["generic_change"] = generic_change(dw, beta, base_weights[pos], generic[pos])
        diag["bias_correction"] = beta.to(torch.float32)
        diag["correction_norm"] = float(torch.linalg.norm(dw))
        diag["bias_norm"] = float(torch.linalg.norm(beta))
        return dw.T.to(torch.float32), diag

    return hook


# ---------------------------------------------------------------- D3: residual spread between measurement depths (O3)


def measure_blocks(depths, target_depth: int) -> list[int]:
    """Target blocks ``m_i = round(s_i (L_k - 1))`` of the measurement depths, sorted and distinct."""
    blocks = sorted({int(round(float(s) * (target_depth - 1))) for s in depths})
    if not blocks or blocks[0] < 0 or blocks[-1] > target_depth - 1:
        raise ValueError(f"measurement depths {depths} out of [0, 1]")
    return blocks


@torch.no_grad()
def fit_spread(
    target_model,
    target_base_sd: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    measures: list[int],
    *,
    config,
    device,
    key_fmt: str = "visual.transformer.resblocks.{}.mlp.c_proj",
    solve_hook=None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """D3 (after MEMIT's spreading): the desired displacement is only used at the measurement blocks ``measures``;
    for the interval (m_{i-1}, m_i] with blocks b_1..b_L (the first interval starts at block 0), block b_l is fitted,
    in order, to the quota of what is still missing at m_i:

        residual_l = D(m_i) - (S_cur(m_i) - S_base(m_i))     (all earlier fitted blocks mounted)
        quota_l    = residual_l / (L - l + 1),   dW_{b_l} = ridge(H_{b_l}, quota_l), then mounted

    H_{b_l} are the c_proj inputs of the CURRENT (mounted) target; the quota measured at m_i is used as the wanted
    change of the residual stream at b_l's output on the same tokens (the stream is additive, so a change written by
    b_l reaches m_i unchanged to first order). One target sweep per fitted block. Blocks after the last measurement
    are not fitted (zero update). With one block per interval (every block a measurement) this is the sequential
    fit (Ariadne's ``depth_fit='sequential'``: error = D_j - U_j on the mounted model).
    ``config`` is the Ariadne ``DirectResidualConfig`` (ridge parameters); ``solve_hook`` as in ``fit_direct_residual``.
    """
    from .._ariadne.fit import ResidualSufficientStatistics

    batches = captured["target_batches"]
    base_out = captured["target_base_outputs_by_position"]
    state = {k: v.detach().cpu().clone() for k, v in target_base_sd.items()}
    original = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    corrections: dict[str, Tensor] = {}
    rows: list[dict[str, Any]] = []
    try:
        target_model.load_state_dict(state, strict=True)
        start = 0
        for m in measures:
            interval = list(range(start, m + 1))
            for ell, b in enumerate(interval):
                left = len(interval) - ell
                requests = {"out": (m, "boundary"), "h": (b, COMPONENT_INPUT_KIND["mlp.c_proj"])}
                stats = ResidualSufficientStatistics(device=device)
                res_sq = des_sq = 0.0
                width = None
                for k, values in enumerate(iter_capture_tokens(target_model, batches, requests, device)):
                    realized = values["out"].to(torch.float64) - base_out[m][k].to(values["out"].device, torch.float64)
                    target = desired[m][k].to(values["out"].device, torch.float64)
                    residual = target - realized
                    res_sq += float((residual**2).sum())
                    des_sq += float((target**2).sum())
                    quota = residual / left
                    h = values["h"]
                    width = quota.shape[-1]
                    stats.update(h.reshape(-1, h.shape[-1]), quota.reshape(-1, width).to(h.dtype), None,
                                 torch.eye(width, dtype=torch.float32))
                if solve_hook is None:
                    correction, diag = stats.solve(ridge_relative=config.ridge_relative,
                                                   ridge_estimator=config.ridge_estimator,
                                                   exact_form=config.exact_form)
                else:
                    correction, diag = solve_hook(b, "mlp.c_proj", stats, config)
                key = key_fmt.format(b)
                w_key, b_key = f"{key}.weight", f"{key}.bias"
                correction = correction.cpu().to(state[w_key].dtype)
                bias = diag["bias_correction"].cpu().to(state[b_key].dtype)
                corrections[w_key], corrections[b_key] = correction, bias
                state[w_key] = state[w_key] + correction
                state[b_key] = state[b_key] + bias
                target_model.load_state_dict({w_key: state[w_key], b_key: state[b_key]}, strict=False)
                rows.append({
                    "mode": "anchors_spread", "component": "mlp.c_proj", "position": b, "measure_block": m,
                    "quota_divisor": left, "desired_norm": des_sq**0.5, "residual_norm_before": res_sq**0.5,
                    "relative_residual_before": res_sq**0.5 / des_sq**0.5 if des_sq else 0.0,
                    **{k_: v for k_, v in diag.items() if k_ != "bias_correction"},
                })
            # what the interval left at its measurement block (one more sweep; diagnostic only)
            fin_sq = 0.0
            for k, values in enumerate(iter_capture_tokens(target_model, batches, {"out": (m, "boundary")}, device)):
                realized = values["out"].to(torch.float64) - base_out[m][k].to(values["out"].device, torch.float64)
                fin_sq += float((desired[m][k].to(realized.device, torch.float64) - realized).pow(2).sum())
            for row in rows:
                if row["measure_block"] == m:
                    row["interval_final_residual_norm"] = fin_sq**0.5
            start = m + 1
    finally:
        target_model.load_state_dict(original, strict=True)
    return corrections, rows


# ------------------------------------------------ S2 / S3: closed-loop (error feedback) fit and label-free gain calibration


def clip_gain(g: float) -> float:
    """The S3 least-squares gain clipped into ``config.GAIN_BOUNDS``."""
    from .config import GAIN_BOUNDS

    return min(max(float(g), GAIN_BOUNDS[0]), GAIN_BOUNDS[1])


@torch.no_grad()
def fit_tracking(
    target_model,
    target_base_sd: Mapping[str, Tensor],
    captured: Mapping[str, Any],
    desired: Mapping[int, list[Tensor]],
    *,
    config,
    device,
    solve_hook=None,
    fixed: Mapping[str, Tensor] | None = None,
    gain: str = "none",
    holdout: int = 0,
    key_fmt: str = "visual.transformer.resblocks.{}.mlp.c_proj",
    sequential_impl: str = "resident",
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Closed-loop fit of the ``mlp.c_proj`` updates in depth order (S2, ``depth_fit='tracking'`` and the resident
    form of ``depth_fit='sequential'`` with ``ridge_preserve`` = S1) and the label-free gain calibration (S3).

    Definition of ``D_j`` in this codebase: ``E = S_ft,i - S_base,i`` is the change of the residual STREAM at the
    boundary of source block ``i``, so ``D_j`` (its transport to the target) is the desired CUMULATIVE stream change at
    the end of target block ``j`` (not the increment written by block ``j``). Exactly (the stream is additive),

        boundary_j(model) = boundary_j(base) + sum_{i<=j} (own write of block i)  +  (changes of the attention / MLP
                            inputs caused by earlier updates, which are inside the captured boundary).

    Block ``j`` therefore has to write what is still missing at its boundary given the updated upstream blocks:

        U_j      = boundary_j(model with blocks 0..j-1 updated) - boundary_j(base)         (upstream realized change)
        target_j = D_j - U_j                       (= (H_base,j + D_j) - H_current,j: error feedback)
        dW_j, beta_j = synthesis(H_current,j, target_j)       H_current,j = c_proj inputs of the UPDATED model

    after which block ``j`` is mounted (weight and bias) before block ``j + 1``. Block ``j``'s own write moves boundary
    ``j`` by exactly ``H dW_j + 1 beta_j^T`` (its input does not depend on its c_proj) and later blocks do not affect
    boundary ``j``, so after the whole fit the realized error at boundary ``j`` is block ``j``'s fit residual. With the
    plain ridge this is Ariadne's sequential streaming fit (``D_j - U_j`` on the mounted model), resident and
    hook-aware; summing the D_i (the literal 'cumulative target' H_base + sum_i D_i) would double count.

    ``solve_hook``: as in ``fit_direct_residual`` (D2 / D5; ``None``: Ariadne's ridge). ``fixed``: already fitted
    corrections ``{key: tensor}`` (e.g. the independent fit): nothing is fitted, only the gains are calibrated.

    S3 (``gain='lsq'``): with ``own = H dW_j + 1 beta_j^T`` on the current inputs and ``r = D_j - U_j``,
    ``g_j = <own, r> / <own, own>`` (clipped to ``GAIN_BOUNDS``), measured on the last ``holdout`` calibration batches
    (excluded from the fit; ``holdout = 0``: all rows, in-sample), minimising ``||U_j + g own - D_j||^2`` over g. For
    ``j = 0`` (``U = 0``) this is ``<realized, desired> / <realized, realized>``. ``dW_j, beta_j`` are multiplied by
    ``g_j`` and mounted, so later blocks see the scaled update (depth order).

    ``sequential_impl='incremental'`` (open_clip ViT only): the same fit with the residual stream carried from block to
    block instead of a full target forward per block (``_fit_tracking_incremental``).

    Returns ``(corrections, rows)`` (unit-strength weight / bias per block, rows per position). The model is restored.
    """
    if sequential_impl == "incremental":
        return _fit_tracking_incremental(
            target_model, target_base_sd, captured, desired, config=config, device=device, solve_hook=solve_hook,
            fixed=fixed, gain=gain, holdout=holdout, key_fmt=key_fmt,
        )
    if sequential_impl != "resident":
        raise ValueError(f"sequential_impl must be 'resident' or 'incremental': {sequential_impl!r}")
    batches = captured["target_batches"]
    base_out = captured["target_base_outputs_by_position"]
    n = len(batches)
    use_gain, n_hold, fit_rows, gain_rows = _gain_plan(n, gain, holdout, fixed)
    state, restore = _base_and_restore(target_model, target_base_sd)
    corrections: dict[str, Tensor] = {}
    rows: list[dict[str, Any]] = []
    try:
        target_model.load_state_dict(state, strict=True)
        for j in sorted(desired):
            key = key_fmt.format(j)
            w_key, b_key = f"{key}.weight", f"{key}.bias"
            requests = {"out": (j, "boundary"), "h": (j, COMPONENT_INPUT_KIND["mlp.c_proj"])}
            acc = _BlockAccumulator(device, fit_rows, gain_rows)
            for k, values in enumerate(iter_capture_tokens(target_model, batches, requests, device)):
                acc.add(k, values["h"], values["out"], base_out[j][k], desired[j][k])
            correction, bias, row = _solve_block(
                j, acc, config=config, solve_hook=solve_hook, fixed=fixed, w_key=w_key, b_key=b_key,
                use_gain=use_gain, n_hold=n_hold, state=state,
            )
            corrections[w_key], corrections[b_key] = correction, bias
            state[w_key] = state[w_key] + correction
            state[b_key] = state[b_key] + bias
            target_model.load_state_dict({w_key: state[w_key], b_key: state[b_key]}, strict=False)
            rows.append(row)
    finally:
        target_model.load_state_dict(restore, strict=True)
    return corrections, rows


def _gain_plan(n: int, gain: str, holdout: int, fixed) -> tuple[bool, int, set[int], set[int]]:
    """``(use_gain, n_hold, fit_rows, gain_rows)``: which calibration batches enter the fit and the gain."""
    use_gain = gain != "none"
    n_hold = int(holdout) if use_gain else 0
    if not 0 <= n_hold < n:
        raise ValueError(f"gain holdout of {n_hold} batches needs fewer than the {n} calibration batches")
    fit_rows = set(range(n - n_hold)) if fixed is None else set()
    gain_rows = (set(range(n - n_hold, n)) if n_hold else set(range(n))) if use_gain else set()
    return use_gain, n_hold, fit_rows, gain_rows


def _base_and_restore(target_model, target_base_sd: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """``(state, restore)``: ``state`` = the base weights on the host (no copy when they already are there; it is only
    ever extended by out-of-place sums, never written in place), ``restore`` = what the model gets back at the end.
    When the live model is bit-equal to the base (the usual case: the shared ridge leaves it there) ``restore`` IS
    ``state`` and the second full-model clone is skipped; otherwise the live weights are cloned as before."""
    state = {k: v.detach().cpu() for k, v in target_base_sd.items()}
    live = target_model.state_dict()
    live_ptrs = {v.untyped_storage().data_ptr() for v in live.values()}
    if any(v.untyped_storage().data_ptr() in live_ptrs for v in state.values()):
        # the base dict shares storage with the live parameters (e.g. to_cpu_fp32(model.state_dict()) of a CPU fp32
        # model): mounting a block writes into those tensors with copy_, so both need private copies
        state = {k: v.clone() for k, v in state.items()}
        return state, {k: v.detach().cpu().clone() for k, v in live.items()}
    same = set(live) == set(state) and all(torch.equal(v.detach().cpu(), state[k].to(v.dtype)) for k, v in live.items())
    state = dict(state)  # own dict: the caller's entries are replaced, never mutated
    if same:
        return state, dict(state)
    return state, {k: v.detach().cpu().clone() for k, v in live.items()}


class _BlockAccumulator:
    """Per-block sums of ``fit_tracking``: ridge statistics, rows kept for the gain, norms (same arithmetic for the
    resident and the incremental implementation)."""

    def __init__(self, device, fit_rows: set[int], gain_rows: set[int]) -> None:
        from .._ariadne.fit import ResidualSufficientStatistics

        self.stats = ResidualSufficientStatistics(device=device)
        self.fit_rows, self.gain_rows = fit_rows, gain_rows
        self.kept: list[tuple[Tensor, Tensor, Tensor]] = []
        self.des_sq = self.up_sq = self.res_sq = 0.0

    def add(self, k: int, h: Tensor, out: Tensor, base_out: Tensor, des: Tensor) -> None:
        out = out.to(torch.float64)
        up = out - base_out.to(out.device, torch.float64)
        des = des.to(out.device, torch.float64)
        res = des - up
        self.des_sq += float((des**2).sum())
        self.up_sq += float((up**2).sum())
        self.res_sq += float((res**2).sum())
        if k in self.fit_rows:
            width = res.shape[-1]
            self.stats.update(h.reshape(-1, h.shape[-1]), res.reshape(-1, width).to(h.dtype), None,
                              torch.eye(width, dtype=torch.float32))
        if k in self.gain_rows:
            self.kept.append((h.reshape(-1, h.shape[-1]).to(torch.float64), res.reshape(-1, res.shape[-1]),
                              up.reshape(-1, up.shape[-1])))


def _solve_block(j, acc: _BlockAccumulator, *, config, solve_hook, fixed, w_key, b_key, use_gain, n_hold, state):
    """Solve (or take the fixed correction of) block ``j``, calibrate its gain; returns the correction and bias
    (scaled, in the base dtype) and the diagnostics row."""
    stats, kept = acc.stats, acc.kept
    des_sq, up_sq, res_sq = acc.des_sq, acc.up_sq, acc.res_sq
    if fixed is None:
        if solve_hook is None:
            correction, diag = stats.solve(ridge_relative=config.ridge_relative,
                                           ridge_estimator=config.ridge_estimator,
                                           exact_form=config.exact_form)
        else:
            correction, diag = solve_hook(j, "mlp.c_proj", stats, config)
        correction = correction.detach().cpu().to(torch.float64)
        bias = diag["bias_correction"].detach().cpu().to(torch.float64)
        diag = {k_: v for k_, v in diag.items() if k_ != "bias_correction"}
    else:
        correction, bias, diag = fixed[w_key].detach().cpu().to(torch.float64), \
            fixed[b_key].detach().cpu().to(torch.float64), {}
    row: dict[str, Any] = {
        "mode": "anchors_tracking", "component": "mlp.c_proj", "position": j,
        "desired_norm": des_sq**0.5, "upstream_norm": up_sq**0.5,
        "upstream_over_desired": up_sq**0.5 / des_sq**0.5 if des_sq else 0.0,
        "residual_norm_before": res_sq**0.5,
        "relative_residual_before": res_sq**0.5 / des_sq**0.5 if des_sq else 0.0,
        "closed_loop": fixed is None, **diag,
    }
    g = 1.0
    if use_gain:
        num = den = rr = rd = dd = 0.0
        for h, res, _up in kept:
            own = h @ correction.T.to(h.device) + bias.to(h.device)
            num += float((own * res).sum())
            den += float((own * own).sum())
        g_raw = num / den if den > 0 else 1.0
        g = clip_gain(g_raw)
        for h, res, up in kept:
            own = h @ correction.T.to(h.device) + bias.to(h.device)
            real, want = up + g * own, res + up
            rr += float((real * real).sum())
            rd += float((real * want).sum())
            dd += float((want * want).sum())
        row.update({
            "gain": g, "gain_lsq_raw": g_raw, "gain_rows": "holdout" if n_hold else "in_sample",
            "gain_own_norm": den**0.5,
            "realized_over_desired_gain_rows": rr**0.5 / dd**0.5 if dd else float("nan"),
            "cosine_gain_rows": rd / (rr * dd) ** 0.5 if rr * dd > 0 else float("nan"),
        })
    del kept[:]
    return (g * correction).to(state[w_key].dtype), (g * bias).to(state[b_key].dtype), row


# ------------------------------------------------ incremental closed loop (sequential_impl='incremental')


class ViTBlockRunner:
    """Runs ONE residual block of an open_clip ``VisionTransformer`` on a residual-stream tensor, with the semantics of
    the full-model capture: the c_proj input is read by a forward hook on ``mlp.c_proj`` (kind ``mlp_c_proj_input``
    of ``iter_capture_tokens``), the block output is the ``resblock`` output (kind ``boundary``). The prefix
    (patch embedding, class token, positional embedding, ``ln_pre``) is the module's own ``_embeds``. Streams are
    ``[B, T, D]`` (batch first) whatever the transformer's own layout. Context manager: moves the model to ``device``,
    eval mode, restored on exit. Decoders and non-open_clip vision models are rejected with ``ValueError``."""

    def __init__(self, model, device) -> None:
        from .._ariadne.layouts import _layout_for

        visual = getattr(model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        if visual is None or not hasattr(visual, "_embeds") or not hasattr(transformer, "resblocks"):
            raise ValueError("sequential_impl='incremental' needs an open_clip VisionTransformer (visual._embeds, "
                             "visual.transformer.resblocks); decoders and other models are not supported")
        self.model, self.device, self.visual = model, device, visual
        self.layout = _layout_for(None)
        self.blocks = self.layout.blocks(model)
        self.batch_first = bool(getattr(transformer, "batch_first", False))

    def __enter__(self):
        self._training = self.model.training
        self._device = next(self.model.parameters()).device
        self.model.to(self.device).eval()
        return self

    def __exit__(self, *exc) -> None:
        self.model.to(self._device).train(self._training)

    @property
    def depth(self) -> int:
        return len(self.blocks)

    @torch.no_grad()
    def prefix(self, images: Tensor) -> Tensor:
        return self.visual._embeds(images.to(self.device))

    @torch.no_grad()
    def run(self, j: int, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(out [B,T,D], h [B,T,d_ff])``: block ``j``'s boundary and the input of its ``mlp.c_proj``, at the live weights."""
        block = self.blocks[j]
        held: list[Tensor] = []
        handle = self.layout.proj_module(block).register_forward_hook(lambda _m, inputs, _o: held.append(inputs[0]))
        try:
            out = block(x if self.batch_first else x.transpose(0, 1).contiguous(), attn_mask=None)
        finally:
            handle.remove()
        if len(held) != 1:
            raise RuntimeError("mlp.c_proj did not fire exactly once")
        h = held[0]
        if not self.batch_first:
            out, h = out.transpose(0, 1), h.transpose(0, 1)
        return out, h


class IncrementalGenericInputs:
    """``{j: GenericInputStats}`` of the CURRENT (updated) model for ``preserve_inputs='updated'`` in the incremental
    fit: a third running residual stream over the generic-pool images, carried from block to block like the
    calibration stream. Same seeded token rows, same norm filter and same statistics as ``capture_generic_inputs`` /
    ``LazyGenericInputs`` (``blocks=[j]``). Protocol (driven by ``fit_tracking``): ``start(runner)`` once with the model
    at its base weights, ``self[j]`` when the solve asks for block ``j`` (before ``j`` is mounted), ``advance(j)`` after
    ``j`` is mounted."""

    def __init__(self, batches, *, tokens_per_block: int, seed: int, norm_filter: Mapping[str, Any] | None = None) -> None:
        self._batches = batches
        self._tokens = int(tokens_per_block)
        self._seed = int(seed)
        self._norm = norm_filter
        self._runner: ViTBlockRunner | None = None
        self._streams: list[Tensor] = []
        self._picks: list[Tensor] = []
        self._cache: tuple[int, GenericInputStats] | None = None
        self.captures = 0

    def start(self, runner: ViTBlockRunner) -> None:
        self._runner = runner
        self._cache = None
        per_batch = -(-self._tokens // max(len(self._batches), 1))
        gen = torch.Generator().manual_seed(self._seed)
        self._streams, self._picks = [], []
        for batch in self._batches:
            x = runner.prefix(batch[0])
            n_rows = x.shape[0] * x.shape[1]
            self._streams.append(x)
            self._picks.append(torch.randperm(n_rows, generator=gen)[: min(per_batch, n_rows)])

    def __getitem__(self, j: int) -> GenericInputStats:
        j = int(j)
        if self._cache is not None and self._cache[0] == j:
            return self._cache[1]
        stats = GenericInputStats()
        for x, pick in zip(self._streams, self._picks, strict=True):
            out, h = self._runner.run(j, x)
            h, out = h.float(), out.float()
            n_rows = h.shape[0] * h.shape[1]
            rows = h.reshape(n_rows, -1)[pick.to(h.device)]
            if self._norm is not None:
                o = out.reshape(n_rows, -1)[pick.to(out.device)].to(torch.float64)
                mu = self._norm["mu"][j].to(o.device, torch.float64)
                keep = (o - mu).norm(dim=-1) <= float(self._norm["k"]) * float(self._norm["median"][j])
                rows = rows[keep.to(rows.device)]
            stats.update(rows)
        self._cache = (j, stats)
        self.captures += 1
        return stats

    def advance(self, j: int) -> None:
        """Carry the stream through block ``j`` with its mounted weights (the model must hold them)."""
        self._streams = [self._runner.run(j, x)[0] for x in self._streams]
        self._cache = None


@torch.no_grad()
def _fit_tracking_incremental(
    target_model, target_base_sd, captured, desired, *, config, device, solve_hook, fixed, gain, holdout, key_fmt
):
    """``fit_tracking`` with the residual stream of the updated model carried from block to block.

    Per calibration batch the stream ``x_j`` at the input of block ``j`` (all earlier blocks mounted) is kept
    (``[B, T, D]``, on ``device``); the prefix runs once. For block ``j``: pass 1 runs only block ``j`` on ``x_j`` ->
    ``h_j`` (c_proj input) and ``out_j`` (boundary of the updated model); ``U_j = out_j - base_out[j]`` with the stored base
    boundary of ``captured`` (no base stream is run); the statistics, the solve and the gain are those of the resident
    fit. Then ``j`` is mounted and pass 2 recomputes block ``j`` with the mounted weights, ``x_{j+1}`` = its output.
    Two block forwards per batch and block (instead of ``j + 1`` full-prefix forwards), no pass 2 after the last block.
    With ``preserve_inputs='updated'`` the hook's ``generic_provider`` (``IncrementalGenericInputs``) is a third stream.
    """
    batches = captured["target_batches"]
    base_out = captured["target_base_outputs_by_position"]
    use_gain, n_hold, fit_rows, gain_rows = _gain_plan(len(batches), gain, holdout, fixed)
    provider = getattr(solve_hook, "generic_provider", None)
    state, restore = _base_and_restore(target_model, target_base_sd)
    corrections: dict[str, Tensor] = {}
    rows: list[dict[str, Any]] = []
    runner = ViTBlockRunner(target_model, device)
    try:
        target_model.load_state_dict(state, strict=True)
        with runner:
            last = max(desired)
            streams = [runner.prefix(b[0]) for b in batches]
            if provider is not None:
                provider.start(runner)
            for j in range(last + 1):
                if j in desired:
                    key = key_fmt.format(j)
                    w_key, b_key = f"{key}.weight", f"{key}.bias"
                    acc = _BlockAccumulator(device, fit_rows, gain_rows)
                    for k, x in enumerate(streams):
                        out, h = runner.run(j, x)
                        acc.add(k, h.float(), out.float(), base_out[j][k], desired[j][k])
                    correction, bias, row = _solve_block(
                        j, acc, config=config, solve_hook=solve_hook, fixed=fixed, w_key=w_key, b_key=b_key,
                        use_gain=use_gain, n_hold=n_hold, state=state,
                    )
                    corrections[w_key], corrections[b_key] = correction, bias
                    state[w_key] = state[w_key] + correction
                    state[b_key] = state[b_key] + bias
                    target_model.load_state_dict({w_key: state[w_key], b_key: state[b_key]}, strict=False)
                    rows.append(row)
                if j < last:
                    streams = [runner.run(j, x)[0] for x in streams]
                    if provider is not None:
                        provider.advance(j)
    finally:
        target_model.load_state_dict(restore, strict=True)
        streams = []
        if provider is not None:  # free the generic-pool streams on the device (the hook outlives the fit)
            provider._streams, provider._picks = [], []
    return corrections, rows
