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
seeded subsample of ``preserve_tokens`` tokens per block.
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
) -> dict[int, GenericInputStats]:
    """``{j: GenericInputStats}`` of the target base's ``component`` inputs on ``batches`` (one streamed sweep).

    Tokens are subsampled with a seeded per-batch draw of ``ceil(tokens_per_block / len(batches))`` rows (the same
    token rows at every block). ``norm_filter`` (a stored ``AnchorState.norm_filter``): tokens whose block output
    norm ``||x - mu_j||`` exceeds ``k * median_j`` are dropped first (the anchor filter's rule).
    """
    kind = COMPONENT_INPUT_KIND[component]
    requests = {f"{j}.h": (j, kind) for j in range(depth)}
    if norm_filter is not None:
        requests.update({f"{j}.out": (j, "boundary") for j in range(depth)})
    per_batch = -(-int(tokens_per_block) // max(len(batches), 1))
    gen = torch.Generator().manual_seed(int(seed))
    stats = {j: GenericInputStats() for j in range(depth)}
    for values in iter_capture_tokens(target_model, batches, requests, device):
        h0 = values["0.h"]
        n_rows = h0.shape[0] * h0.shape[1]
        pick = torch.randperm(n_rows, generator=gen)[: min(per_batch, n_rows)]
        for j in range(depth):
            rows = values[f"{j}.h"].reshape(n_rows, -1)[pick.to(values[f"{j}.h"].device)]
            if norm_filter is not None:
                out = values[f"{j}.out"].reshape(n_rows, -1)[pick.to(values[f"{j}.out"].device)].to(torch.float64)
                mu = norm_filter["mu"][j].to(out.device, torch.float64)
                keep = (out - mu).norm(dim=-1) <= float(norm_filter["k"]) * float(norm_filter["median"][j])
                rows = rows[keep.to(rows.device)]
            stats[j].update(rows)
    return stats


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
    if rho_g > 0.0:
        m = float(generic.m)
        xg = torch.zeros(d + 1, d + 1, dtype=torch.float64)
        xg[:d, :d], xg[:d, d], xg[d, :d], xg[d, d] = generic.gg, generic.sg, generic.sg, m
        a = a + (float(rho_g) / m) * xg
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
    gg, sg, m = generic.gg, generic.sg, float(generic.m)
    change = float((dw * (gg @ dw)).sum()) + 2 * float(beta @ (dw.T @ sg)) + m * float(beta @ beta)
    w = weight.to(torch.float64).T
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
        dw, beta = correction.T.double(), diag["bias_correction"].double()
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
