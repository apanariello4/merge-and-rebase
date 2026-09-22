from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.target_residual_completion import (
    ResidualSufficientStatistics,
    backproject_target_rows,
    centered_rectangular_procrustes,
    fit_cproj_residual,
    fit_joint_cproj_correction,
    parse_joint_correction_config,
    parse_residual_completion_config,
    validate_residual_completion_depth_direction,
)
from merge_and_rebase.rebase.methods.theseus import _transport_weight


def test_config_defaults_and_validation() -> None:
    cfg = parse_residual_completion_config({"enabled": True})
    assert cfg.enabled and cfg.num_batches == 10
    assert cfg.target_scope == "inserted"
    assert cfg.ridge_estimator == "fixed_relative"
    assert parse_residual_completion_config({"ridge_estimator": "empirical_bayes"}).ridge_estimator == "empirical_bayes"
    assert parse_residual_completion_config({"added_blocks": "all", "target_scope": "all"}).target_scope == "all"
    assert cfg.exact_form is True  # no proposal-1 result predates this fix; exact is the only sane default.
    assert parse_residual_completion_config({"num_batches": 1}).num_batches == 1
    assert parse_residual_completion_config({"exact_form": False}).exact_form is False
    with pytest.raises(ValueError):
        parse_residual_completion_config({"ridge_relative": 0})
    with pytest.raises(ValueError):
        parse_residual_completion_config({"num_batches": 0})
    with pytest.raises(TypeError):
        parse_residual_completion_config({"exact_form": 1})
    with pytest.raises(ValueError):
        parse_residual_completion_config({"target_scope": "added"})
    with pytest.raises(ValueError):
        parse_residual_completion_config({"ridge_estimator": "automatic"})


def test_empirical_bayes_ridge_matches_component_specific_relative_ridge() -> None:
    torch.manual_seed(29)
    h = torch.randn(17, 7)
    e = torch.randn(17, 5)
    tin = torch.randn(3, 7)
    tout = torch.randn(2, 5)
    stats = ResidualSufficientStatistics()
    stats.update(h, e, tin, tout)

    automatic, automatic_diag = stats.solve(
        ridge_relative=123.0,
        ridge_estimator="empirical_bayes",
        exact_form=True,
    )
    expected_relative = 3.0 / 16.0
    fixed, fixed_diag = stats.solve(ridge_relative=expected_relative, exact_form=True)

    assert torch.allclose(automatic, fixed, atol=1e-6, rtol=1e-5)
    assert automatic_diag["ridge"] == pytest.approx(fixed_diag["ridge"])
    assert automatic_diag["ridge_estimator"] == "empirical_bayes"
    assert automatic_diag["configured_ridge_relative"] == 123.0
    assert automatic_diag["effective_ridge_relative"] == pytest.approx(expected_relative)


def test_direct_target_shrink_depth_preflight_rejects_extension_semantics() -> None:
    inserted = parse_residual_completion_config(
        {"enabled": True, "mode": "direct_target", "target_scope": "inserted"}
    )
    with pytest.raises(ValueError, match="requires target_scope='all'"):
        validate_residual_completion_depth_direction(
            inserted, source_depth=24, target_depth=12
        )

    interpolated = parse_residual_completion_config(
        {
            "enabled": True,
            "mode": "direct_target",
            "target_scope": "all",
            "target_trajectory": "interpolate",
        }
    )
    with pytest.raises(ValueError, match="requires target_trajectory='step'"):
        validate_residual_completion_depth_direction(
            interpolated, source_depth=24, target_depth=12
        )


def test_direct_target_depth_preflight_accepts_step_shrink_and_extensions() -> None:
    shrink = parse_residual_completion_config(
        {
            "enabled": True,
            "mode": "direct_target",
            "target_scope": "all",
            "target_trajectory": "step",
        }
    )
    validate_residual_completion_depth_direction(shrink, source_depth=24, target_depth=12)

    extension = parse_residual_completion_config(
        {
            "enabled": True,
            "mode": "direct_target",
            "target_scope": "all",
            "target_trajectory": "interpolate",
        }
    )
    validate_residual_completion_depth_direction(extension, source_depth=12, target_depth=24)


def test_joint_correction_config_is_explicit_and_validated() -> None:
    cfg = parse_joint_correction_config({"enabled": True, "source_weight": 2.0, "target_weight": 0.5})
    assert cfg.enabled and cfg.source_weight == 2.0 and cfg.target_weight == 0.5
    assert parse_joint_correction_config(None).enabled is False
    with pytest.raises(ValueError, match="at least one"):
        parse_joint_correction_config({"source_weight": 0.0, "target_weight": 0.0})
    with pytest.raises(ValueError, match="unknown"):
        parse_joint_correction_config({"transport_seed": 0})


def test_centered_rectangular_procrustes_and_backprojection() -> None:
    torch.manual_seed(2)
    source = torch.randn(20, 3)
    q_true = torch.linalg.qr(torch.randn(4, 3), mode="reduced").Q.T
    target = (source - source.mean(0)) @ q_true + torch.randn(4) + 2.0
    q, mu_s, mu_t = centered_rectangular_procrustes(source, target)
    assert q.shape == (3, 4)
    assert torch.allclose((source - mu_s) @ q, target - mu_t, atol=1e-5)
    assert torch.allclose(backproject_target_rows(target, q, mu_t, mu_s), source.double(), atol=1e-5)


def test_streaming_statistics_equal_single_batch_and_orientation() -> None:
    torch.manual_seed(3)
    n, ms, ds, mt, dt = 13, 3, 2, 4, 5
    h = torch.randn(n, mt)
    e = torch.randn(n, dt)
    tin = torch.randn(ms, mt)
    tout = torch.randn(ds, dt)
    one, _ = fit_cproj_residual(h, e, tin, tout, ridge_relative=0.02)
    split = ResidualSufficientStatistics()
    split.update(h[:6], e[:6], tin, tout)
    split.update(h[6:], e[6:], tin, tout)
    two, _ = split.solve(ridge_relative=0.02)
    assert torch.allclose(one, two, atol=1e-6, rtol=1e-5)
    # X is Delta_C_source.T and the predicted target response is A X L.T.
    a = h.double() @ tin.double().T
    lmat = tout.double().T
    assert a.shape == (n, ms)
    assert (a @ one.double().T @ lmat.T).shape == (n, dt)


def test_rank_deficient_features_and_zero_residual() -> None:
    h = torch.zeros(8, 4)
    e = torch.zeros(8, 3)
    tin = torch.eye(2, 4)
    tout = torch.eye(2, 3)
    correction, diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=1.0)
    assert torch.isfinite(correction).all()
    assert torch.equal(correction, torch.zeros_like(correction))
    assert diag["residual_norm_after"] == 0.0


def test_solver_matches_explicit_vectorized_ridge_reference() -> None:
    """Independent brute-force check of the exact affine objective (Eq. 8-9):

        min_{X,beta} || (A X + 1 beta^T) t_out - E ||_F^2 + lam ||X||_F^2

    solved here by literally building the augmented (X, beta) design matrix
    and its normal equations (ridge only on the X block), with no centering
    trick -- an independent path to the same optimum the closed-form
    centered solver in `ResidualSufficientStatistics.solve` computes.
    """
    torch.manual_seed(11)
    h = torch.randn(7, 4)
    e = torch.randn(7, 3)
    tin = torch.randn(2, 4)
    tout = torch.randn(2, 3)
    rho = 0.07
    got, diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=rho)
    got_beta = diag["bias_correction"].double()
    a = h.double() @ tin.double().T
    lmat = tout.double().T
    e64 = e.double()
    n, m_s, d_s = a.shape[0], a.shape[1], lmat.shape[1]
    columns = []
    for i in range(m_s):
        for j in range(d_s):
            basis = torch.zeros(m_s, d_s, dtype=torch.float64)
            basis[i, j] = 1.0
            columns.append((a @ basis @ lmat.T).reshape(-1))
    for k in range(d_s):
        # Contribution of beta_k=1: every row gets the k-th row of t_out.
        columns.append((torch.ones(n, 1, dtype=torch.float64) @ tout.double()[k : k + 1, :]).reshape(-1))
    design = torch.stack(columns, dim=1)
    a_c = a - a.mean(dim=0)
    g = tout.double() @ tout.double().T
    lam = rho * torch.trace(a_c.T @ a_c) / m_s * torch.trace(g) / d_s
    ridge_diag = torch.cat([lam * torch.ones(m_s * d_s, dtype=torch.float64), torch.zeros(d_s, dtype=torch.float64)])
    rhs = design.T @ e64.reshape(-1)
    ref_vec = torch.linalg.solve(design.T @ design + torch.diag(ridge_diag), rhs)
    ref_x = ref_vec[: m_s * d_s].reshape(m_s, d_s)
    ref_beta = ref_vec[m_s * d_s :]
    assert torch.allclose(got.double(), ref_x.T, atol=2e-5, rtol=2e-5)
    assert torch.allclose(got_beta, ref_beta, atol=2e-5, rtol=2e-5)


def test_solver_matches_explicit_reference_reduced_form() -> None:
    """Same brute-force check for `exact_form=False`: no centering, no intercept."""
    torch.manual_seed(13)
    h = torch.randn(6, 4)
    e = torch.randn(6, 3)
    tin = torch.randn(2, 4)
    tout = torch.randn(2, 3)
    rho = 0.05
    got, diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=rho, exact_form=False)
    assert torch.equal(diag["bias_correction"], torch.zeros_like(diag["bias_correction"]))
    a = h.double() @ tin.double().T
    lmat = tout.double().T
    m_s, d_s = a.shape[1], lmat.shape[1]
    columns = []
    for i in range(m_s):
        for j in range(d_s):
            basis = torch.zeros(m_s, d_s, dtype=torch.float64)
            basis[i, j] = 1.0
            columns.append((a @ basis @ lmat.T).reshape(-1))
    design = torch.stack(columns, dim=1)
    lam = rho * torch.trace(a.T @ a) / m_s
    rhs = design.T @ e.double().reshape(-1)
    ref_vec = torch.linalg.solve(design.T @ design + lam * torch.eye(m_s * d_s), rhs)
    ref = ref_vec.reshape(m_s, d_s).T.float()
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)


def test_exact_and_reduced_forms_differ_on_nonorthogonal_transport_and_biased_residual() -> None:
    """Falsifiability check: the reduced form is a defect, not a style choice.

    With a non-orthogonal `t_out` and a nonzero-mean residual, the exact
    (centered + intercept) and reduced (uncentered, no intercept) fits must
    disagree, both in the returned weight correction and in the intercept.
    """
    torch.manual_seed(21)
    h = torch.randn(40, 5) + 3.0  # nonzero-mean features
    tin = torch.randn(4, 5)
    a = h.double() @ tin.double().T
    tout = torch.randn(4, 6) * torch.tensor([2.0, 0.3, 1.0, 5.0])[:, None]  # far from isometric
    e = (a.float() @ torch.randn(4, 6)) * 0.1 + torch.randn(1, 6) * 4.0  # nonzero-mean residual
    rho = 0.02

    exact_x, exact_diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=rho, exact_form=True)
    reduced_x, reduced_diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=rho, exact_form=False)

    assert not torch.allclose(exact_x, reduced_x, atol=1e-3, rtol=1e-3)
    assert torch.linalg.norm(exact_diag["bias_correction"]).item() > 1e-3
    assert torch.equal(reduced_diag["bias_correction"], torch.zeros_like(reduced_diag["bias_correction"]))
    assert exact_diag["ridge"] != pytest.approx(reduced_diag["ridge"])


def test_exact_and_reduced_forms_coincide_under_idealized_assumptions() -> None:
    """Pin the exact validity condition of the reduced form: zero-mean source
    features AND zero-mean residual (so centering is a no-op) AND an
    orthonormal `t_out` (so `trace(g)/d_source == 1`, matching the reduced
    form's implicit assumption). Under all three, exact and reduced must
    produce the identical correction and intercept.

    Note this is a strictly stronger, and more precise, condition than "the
    residual is zero-mean and t_out is orthonormal" alone: the source
    feature bank must independently be zero-mean too, since the exact
    solver centers both `a` and `e` (Eq. 10), and centering `a` changes the
    fitted `X` unless `mean(a) == 0` already.
    """
    torch.manual_seed(31)
    tin = torch.randn(3, 4)
    # t_out with orthonormal ROWS (d_source=3 <= d_target=6) makes g = t_out @ t_out.T = I_3 exactly,
    # so trace(g)/d_source == 1 and the reduced ridge formula matches the exact one.
    tout = torch.linalg.qr(torch.randn(6, 3), mode="reduced").Q.T  # [3, 6]

    # Paired +/- rows give an exact (not merely sampled) zero column-mean for both `a` and `e`.
    half_h = torch.randn(15, 4)
    h = torch.cat([half_h, -half_h], dim=0)
    half_e = torch.randn(15, 6)
    e = torch.cat([half_e, -half_e], dim=0)
    rho = 0.03

    exact_x, exact_diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=rho, exact_form=True)
    reduced_x, reduced_diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=rho, exact_form=False)

    assert torch.allclose(exact_x, reduced_x, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        exact_diag["bias_correction"], torch.zeros_like(exact_diag["bias_correction"]), atol=1e-5
    )
    assert exact_diag["ridge"] == pytest.approx(reduced_diag["ridge"], rel=1e-6)


def test_zero_features_keep_nonzero_residual_unreachable() -> None:
    """With zero features, the WEIGHT correction can never explain anything
    (A=0 forces AX=0 for every X); that part of the old (reduced-form)
    assertion still holds exactly. But under the exact affine model, a
    constant (nonzero-mean) residual can still be partly absorbed by the
    intercept alone -- here `t_out=eye(2,3)` is rank-2 into a 3-dim residual,
    so exactly one residual dimension (index 2) is structurally unreachable
    by any bias, and the other two are fully explained.
    """
    h = torch.zeros(8, 4)
    e = torch.ones(8, 3)
    tin = torch.eye(2, 4)
    tout = torch.eye(2, 3)

    correction, diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=1.0)
    assert torch.equal(correction, torch.zeros_like(correction))
    # e's mean [1, 1, 1] is only partly in range(t_out^T); the unreachable
    # 3rd component (variance 0, mean 1) contributes 8 * 1**2 = 8 to the
    # squared residual regardless of ridge, so this is exact, not approximate.
    assert diag["residual_norm_after"] == pytest.approx(8.0**0.5, abs=1e-6)
    assert diag["residual_norm_before"] == pytest.approx(24.0**0.5, abs=1e-6)
    assert torch.linalg.norm(diag["bias_correction"]).item() > 0.0

    reduced_correction, reduced_diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=1.0, exact_form=False)
    assert torch.equal(reduced_correction, torch.zeros_like(reduced_correction))
    assert reduced_diag["residual_norm_after"] == reduced_diag["residual_norm_before"]
    assert reduced_diag["unreachable_residual_norm"] == reduced_diag["residual_norm_before"]
    assert torch.equal(reduced_diag["bias_correction"], torch.zeros_like(reduced_diag["bias_correction"]))


def test_nonzero_rank_deficient_features_are_finite() -> None:
    h = torch.ones(9, 4)
    e = torch.randn(9, 3)
    tin = torch.eye(2, 4)
    tout = torch.eye(2, 3)
    correction, diag = fit_cproj_residual(h, e, tin, tout, ridge_relative=0.1)
    assert torch.isfinite(correction).all()
    assert torch.isfinite(diag["bias_correction"]).all()
    scalar_diag = {k: v for k, v in diag.items() if isinstance(v, (int, float))}
    assert torch.isfinite(torch.tensor(list(scalar_diag.values()), dtype=torch.float64)).all()


def test_transport_orientation_matches_theseus_weight_helper() -> None:
    torch.manual_seed(12)
    c = torch.randn(3, 5)
    t_in = torch.randn(5, 4)
    t_out = torch.randn(3, 2)
    expected = _transport_weight(c, t_in, t_out, key="weight")
    assert torch.allclose(expected, t_out.T @ c @ t_in)


def test_joint_frozen_map_solver_matches_explicit_objective() -> None:
    """The one-alternation solver must minimize the stated source+target objective."""
    torch.manual_seed(41)
    ns, nt, si, ti, so, to = 7, 8, 3, 4, 2, 5
    source_h = torch.randn(ns, si)
    source_y = torch.randn(ns, so)
    target_h = torch.randn(nt, ti)
    target_e = torch.randn(nt, to)
    t_in = torch.randn(si, ti)
    t_out = torch.randn(so, to)
    ws, wt, rho = 1.7, 0.6, 0.04
    got, diag = fit_joint_cproj_correction(
        source_h, source_y, target_h, target_e, t_in, t_out,
        source_weight=ws, target_weight=wt, ridge_relative=rho,
    )

    # Independent explicit vectorized normal equations for the augmented
    # Z=[X; beta], with no ridge on the final intercept row.
    hs, ys = source_h.double(), source_y.double()
    at = target_h.double() @ t_in.double().T
    et, tout = target_e.double(), t_out.double()
    hs_aug = torch.cat((hs, torch.ones(ns, 1, dtype=torch.float64)), dim=1)
    at_aug = torch.cat((at, torch.ones(nt, 1, dtype=torch.float64)), dim=1)
    columns = []
    for i in range(si + 1):
        for j in range(so):
            basis = torch.zeros(si + 1, so, dtype=torch.float64)
            basis[i, j] = 1.0
            columns.append(torch.cat([(hs_aug @ basis).reshape(-1), (at_aug @ basis @ tout).reshape(-1)]))
    design = torch.stack(columns, dim=1)
    targets = torch.cat([ys.reshape(-1), et.reshape(-1)])
    lam = rho * (ws * torch.trace(hs.T @ hs) + wt * torch.trace(at.T @ at)) / si
    weights = torch.cat([
        torch.full((ns * so,), ws, dtype=torch.float64),
        torch.full((nt * to,), wt, dtype=torch.float64),
    ])
    ridge_mask = torch.zeros((si + 1) * so, dtype=torch.float64)
    for i in range(si):
        ridge_mask[i * so : (i + 1) * so] = 1.0
    normal = design.T @ (weights[:, None] * design) + lam * torch.diag(ridge_mask)
    rhs = design.T @ (weights * targets)
    ref = torch.linalg.solve(normal, rhs).reshape(si + 1, so)
    assert torch.allclose(got.double(), ref[:si].T, atol=3e-5, rtol=3e-5)
    assert torch.allclose(diag["bias_correction"].double(), ref[si], atol=3e-5, rtol=3e-5)
    assert diag["frozen_map"] is True
    assert diag["alternations"] == 1
    assert diag["objective_after"] < diag["objective_before"]


def test_joint_solver_rejects_nonmatching_transport_shapes() -> None:
    tensors = [torch.randn(4, 3), torch.randn(4, 2), torch.randn(5, 4), torch.randn(5, 3)]
    with pytest.raises(ValueError, match="t_in"):
        fit_joint_cproj_correction(
            *tensors, torch.randn(2, 5), torch.randn(2, 3),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reproduces only across devices")
def test_reduced_form_solve_does_not_crash_on_a_non_cpu_device():
    """exact_form=False must not hard-code a CPU zero tensor.

    ResidualSufficientStatistics.solve()'s reduced-form branch built `beta`
    (and `beta0`, and the trace_sc==0 branch's `x`) with a bare
    ``torch.zeros(...)``, which defaults to CPU regardless of the device the
    caller actually ran on. On device_transform="gpu" (the LLM path's own
    setting) that produced ``t_out64.T @ beta`` mixing a CUDA tensor with a
    CPU one inside `_residual_sq`, crashing every reduced-form (exact_form=
    False, missing_bias="skip") fit -- caught only once an LLM campaign
    actually exercised that combination on a real GPU node; the CPU-only test
    suite could not have reproduced it, which is why this test is itself
    GPU-gated rather than device-agnostic.
    """
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(0)
    h = torch.randn(16, 5, generator=generator).to(device)
    e = torch.randn(16, 3, generator=generator).to(device)
    t_out = torch.eye(3, device=device)
    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, t_out)
    weight, diag = stats.solve(ridge_relative=0.1, exact_form=False)
    assert weight.device.type == "cuda"
    assert diag["bias_correction"].isfinite().all()
