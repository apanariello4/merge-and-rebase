"""DirectResidualConfig.ridge_estimator='frob_trace' (opt-in): the PatchKV layer-adaptive ridge (arXiv 2609.39329,
Eq. 7), lam = ridge_relative * ||S_c||_F^2 / tr(S_c) on the centred feature Gram S_c (times ||G||_F^2 / tr(G) for the
output map, 1 for an identity map).

Pinned here: config plumbing and rejections; lam against an independent computation from the raw rows (identity and
non-identity output map); the slope against an explicit centred ridge solve with unpenalised intercept; invariance of
the fit to duplicating the calibration rows (lam scales with N like S_c, unlike empirical Bayes); the effective-rank
diagnostic and its absence from the default rows; and the streaming independent and sequential fits end to end.

Fixture duplicated from `tests/test_ariadne_sequential_depth_fit_20261005.py` per this suite's no-cross-test-import
convention.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_sequential_streaming,
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)

NEW_KEYS = ("frobenius_sq_centered_feature_gram", "frobenius_sq_output_transport_gram", "feature_effective_rank")

# ---- config plumbing -----------------------------------------------------------


def test_config_accepts_frob_trace():
    cfg = parse_direct_residual_config({"ridge_estimator": "frob_trace", "ridge_relative": 0.3})
    assert cfg.ridge_estimator == "frob_trace"
    assert cfg.ridge_relative == 0.3


@pytest.mark.parametrize(
    "overrides",
    [
        {"ridge_estimator": "frob_trace", "ridge_relative": 0.0},
        {"ridge_estimator": "frob_trace", "component_fit": "gradient_exact", "activation_storage": "streaming"},
    ],
)
def test_config_rejects(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config(overrides)


# ---- ResidualSufficientStatistics.solve(ridge_estimator="frob_trace") ---------


def _rows(n=300, d_in=6, d_out=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Anisotropic features so ||S_c||_F^2 / tr(S_c) differs clearly from tr(S_c) / m.
    h = torch.randn(n, d_in, generator=g, dtype=torch.float64) * torch.linspace(0.2, 3.0, d_in, dtype=torch.float64)
    h = h + 1.5  # non-zero mean: the centring matters
    e = torch.randn(n, d_out, generator=g, dtype=torch.float64)
    return h, e


def _solve(h, e, t_out, lam0, estimator="frob_trace"):
    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, t_out)
    return stats.solve(ridge_relative=lam0, ridge_estimator=estimator, exact_form=True)


def test_lambda_matches_independent_formula_identity_output():
    h, e = _rows()
    lam0 = 0.37
    _w, diag = _solve(h, e, torch.eye(e.shape[1], dtype=torch.float64), lam0)
    hc = h - h.mean(dim=0, keepdim=True)
    sc = hc.T @ hc
    expected = lam0 * float((sc * sc).sum()) / float(torch.trace(sc))
    assert diag["ridge"] == pytest.approx(expected, rel=1e-10)
    assert diag["ridge_estimator"] == "frob_trace"
    r_eff = float(torch.trace(sc)) ** 2 / float((sc * sc).sum())
    assert diag["feature_effective_rank"] == pytest.approx(r_eff, rel=1e-10)
    assert diag["effective_ridge_relative"] == pytest.approx(lam0 * h.shape[1] / r_eff, rel=1e-10)
    # lam / (tr(S_c)/m) is the relative parameterization shared with fixed_relative and empirical_bayes.
    assert diag["ridge"] / (float(torch.trace(sc)) / h.shape[1]) == pytest.approx(diag["effective_ridge_relative"])


def test_lambda_output_factor_non_identity():
    h, e = _rows(d_out=3)
    t_out = torch.diag(torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64))
    lam0 = 0.2
    _w, diag = _solve(h, e, t_out, lam0)
    hc = h - h.mean(dim=0, keepdim=True)
    sc = hc.T @ hc
    g = t_out @ t_out.T
    expected = lam0 * float((sc * sc).sum()) / float(torch.trace(sc)) * float((g * g).sum()) / float(torch.trace(g))
    assert diag["ridge"] == pytest.approx(expected, rel=1e-10)


def test_slope_and_intercept_match_explicit_centred_ridge():
    h, e = _rows()
    lam0 = 0.05
    w, diag = _solve(h, e, torch.eye(e.shape[1], dtype=torch.float64), lam0)
    lam = diag["ridge"]
    mu_h, mu_e = h.mean(dim=0), e.mean(dim=0)
    hc, ec = h - mu_h, e - mu_e
    ref_w = torch.linalg.solve(hc.T @ hc + lam * torch.eye(h.shape[1], dtype=torch.float64), hc.T @ ec).T
    ref_b = mu_e - ref_w @ mu_h  # unpenalised intercept
    torch.testing.assert_close(w.double(), ref_w, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(diag["bias_correction"].double(), ref_b, rtol=1e-5, atol=1e-6)


def test_fit_invariant_to_duplicated_rows_unlike_empirical_bayes():
    h, e = _rows()
    t_out = torch.eye(e.shape[1], dtype=torch.float64)
    w1, d1 = _solve(h, e, t_out, 0.5)
    w2, d2 = _solve(torch.cat([h, h]), torch.cat([e, e]), t_out, 0.5)
    assert d2["ridge"] == pytest.approx(2 * d1["ridge"], rel=1e-10)
    torch.testing.assert_close(w1, w2, rtol=1e-5, atol=1e-6)
    eb1, _ = _solve(h, e, t_out, 0.5, "empirical_bayes")
    eb2, _ = _solve(torch.cat([h, h]), torch.cat([e, e]), t_out, 0.5, "empirical_bayes")
    assert not torch.allclose(eb1, eb2, rtol=1e-5, atol=1e-6)


def test_larger_lambda0_shrinks_the_update():
    h, e = _rows()
    t_out = torch.eye(e.shape[1], dtype=torch.float64)
    norms = [float(torch.linalg.norm(_solve(h, e, t_out, lam0)[0])) for lam0 in (0.01, 0.1, 1.0, 10.0)]
    assert norms == sorted(norms, reverse=True)
    assert norms[-1] < norms[0]


@pytest.mark.parametrize("estimator", ["fixed_relative", "empirical_bayes", "none"])
def test_default_estimators_do_not_record_the_new_keys(estimator):
    h, e = _rows(n=200, d_in=4, d_out=3)
    _w, diag = _solve(h, e, torch.eye(3, dtype=torch.float64), 0.1, estimator)
    assert not set(NEW_KEYS) & set(diag)


# ---- end-to-end streaming fits ---------------------------------------------------


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


BASE = dict(activation_storage="streaming", num_batches=3, components=("mlp.c_proj",))
REGIMES = [
    pytest.param(2, 4, id="extend"),
    pytest.param(4, 2, id="shrink"),
]


@pytest.mark.parametrize("depth_fit", ["independent", "sequential"])
@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_streaming_fits_use_frob_trace(source_depth, target_depth, depth_fit):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    extra = {"depth_fit": "sequential"} if depth_fit == "sequential" else {}
    cfg_eb = parse_direct_residual_config({**BASE, **extra, "ridge_estimator": "empirical_bayes"})
    cfg_ft = parse_direct_residual_config({**BASE, **extra, "ridge_estimator": "frob_trace", "ridge_relative": 0.5})
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg_eb.num_batches, seed=cfg_eb.seed, device="cpu"
    )
    fit = fit_direct_residual_sequential_streaming if depth_fit == "sequential" else fit_direct_residual_streaming
    args = (target_base, sd, source_base, source_ft, prepared, pairing)
    tau_eb, rows_eb = fit(*args, config=cfg_eb, device="cpu")
    tau_ft, rows_ft = fit(*args, config=cfg_ft, device="cpu")
    assert set(tau_eb) == set(tau_ft)
    assert any(not torch.allclose(tau_eb[k], tau_ft[k]) for k in tau_eb)
    for row in rows_ft:
        assert row["ridge_estimator"] == "frob_trace"
        r_eff = row["trace_centered_feature_gram"] ** 2 / row["frobenius_sq_centered_feature_gram"]
        assert row["feature_effective_rank"] == pytest.approx(r_eff, rel=1e-12)
        expected = 0.5 * row["frobenius_sq_centered_feature_gram"] / row["trace_centered_feature_gram"]
        # c_proj has an identity output map here, so the output factor is 1.
        assert row["ridge"] == pytest.approx(expected, rel=1e-10)
    for row in rows_eb:
        assert not set(NEW_KEYS) & set(row)
    # The target model is restored after both fits.
    for k, v in target_base.state_dict().items():
        assert torch.equal(v, sd[k])
