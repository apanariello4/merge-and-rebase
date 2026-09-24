"""DirectResidualConfig.alignment_map / alignment_row_weighting: parser contract and map edge cases.

Resident/streaming parity of the three non-default variants is pinned in
test_direct_residual_streaming_parity.py; the default (polar, uniform) path is
pinned by the golden-hash suites and the capture fingerprint.
"""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    _fit_activation_map,
    parse_direct_residual_config,
)
from merge_and_rebase.eval.target_residual_completion import centered_ridge_alignment


def test_defaults_are_historical_polar_uniform():
    cfg = parse_direct_residual_config({})
    assert (cfg.alignment_map, cfg.alignment_row_weighting) == ("polar", "uniform")
    assert DirectResidualConfig().alignment_map == "polar"


@pytest.mark.parametrize(
    "fields",
    [
        {"alignment_map": "polar", "alignment_row_weighting": "cls_balanced"},
        {"alignment_map": "polar", "alignment_row_weighting": "delta_magnitude"},
        {"alignment_map": "ridge", "alignment_row_weighting": "uniform"},
    ],
)
def test_campaign_variants_parse(fields):
    cfg = parse_direct_residual_config(fields)
    assert (cfg.alignment_map, cfg.alignment_row_weighting) == (fields["alignment_map"], fields["alignment_row_weighting"])


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"alignment_map": "affine"}, "alignment_map must be"),
        ({"alignment_row_weighting": "attention"}, "alignment_row_weighting must be"),
        ({"alignment_map": "ridge", "alignment_row_weighting": "cls_balanced"}, "requires alignment_row_weighting='uniform'"),
        ({"alignment_map": "ridge", "procrustes_source": "gradient"}, "non-default alignment options require"),
        ({"alignment_row_weighting": "cls_balanced", "residual_target": "transported_endpoint"}, "transported_delta"),
        ({"alignment_row_weighting": "cls_balanced", "component_target": "output_local"}, "block_boundary"),
    ],
)
def test_parser_rejects_undefined_combinations(fields, match):
    with pytest.raises(ValueError, match=match):
        parse_direct_residual_config(fields)


def test_ridge_zero_source_covariance_maps_to_zero():
    x = torch.ones(6, 3, dtype=torch.float64)
    y = torch.randn(6, 2, dtype=torch.float64)
    mapping, _mx, _my, diag = centered_ridge_alignment(x, y)
    assert torch.equal(mapping, torch.zeros(3, 2, dtype=torch.float64))
    assert diag["source_trace"] == 0.0


def test_ridge_default_lambda_is_trace_over_n_minus_one():
    torch.manual_seed(3)
    x = torch.randn(20, 4, dtype=torch.float64)
    y = torch.randn(20, 3, dtype=torch.float64)
    mapping, _mx, _my, diag = centered_ridge_alignment(x, y)
    xc, yc = x - x.mean(0), y - y.mean(0)
    lam = float(torch.trace(xc.T @ xc)) / 19
    expected = torch.linalg.solve(xc.T @ xc + lam * torch.eye(4, dtype=torch.float64), xc.T @ yc)
    assert diag["ridge"] == pytest.approx(lam)
    assert torch.allclose(mapping, expected, atol=1e-12)


def test_cls_balanced_gives_cls_half_the_image_mass():
    """With a single CLS-only-informative image layout, cls_balanced must weight the
    CLS row as much as all patch rows together: scaling every patch row's centered
    contribution leaves the map unchanged only if patches share one half of the mass."""
    torch.manual_seed(5)
    x = [torch.randn(4, 3, 3, dtype=torch.float64)]
    y = [torch.randn(4, 3, 3, dtype=torch.float64)]
    q_uniform, *_ = _fit_activation_map(x, y, None, row_weighting="uniform")
    q_cls, *_ = _fit_activation_map(x, y, None, row_weighting="cls_balanced")
    assert q_cls.shape == q_uniform.shape == (3, 3)
    # Orthogonal (polar) factor either way, but the weighting must change the fit.
    assert torch.allclose(q_cls.T @ q_cls, torch.eye(3, dtype=torch.float64), atol=1e-10)
    assert not torch.allclose(q_cls, q_uniform, atol=1e-6)


def test_cls_balanced_requires_patch_tokens():
    x = [torch.randn(4, 1, 3, dtype=torch.float64)]
    with pytest.raises(ValueError, match="CLS token and at least one patch token"):
        _fit_activation_map(x, x, None, row_weighting="cls_balanced")
