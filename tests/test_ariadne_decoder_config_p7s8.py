from __future__ import annotations

import pytest


def test_ariadne_decoder_config_defaults_rejections_and_ablation_flag():
    from dataclasses import asdict

    from merge_and_rebase.rebase.methods._ariadne.config import (
        DirectResidualConfig,
        direct_residual_config_dict,
        resolve_ariadne_decoder_config,
    )

    cfg = resolve_ariadne_decoder_config({}, object())
    assert (cfg.components, cfg.activation_storage, cfg.ridge_estimator, cfg.missing_bias, cfg.exact_form) == (
        ("mlp.c_proj",), "streaming", "empirical_bayes", "materialize", True)
    assert cfg.copy_shape_matching_source_deltas is False
    assert resolve_ariadne_decoder_config({"missing_bias": "skip", "exact_form": False}, object()).missing_bias == "skip"
    assert DirectResidualConfig().activation_storage == "resident"  # dataclass defaults untouched
    assert "copy_shape_matching_source_deltas" not in direct_residual_config_dict(DirectResidualConfig())
    assert direct_residual_config_dict(DirectResidualConfig(copy_shape_matching_source_deltas=True))[
        "copy_shape_matching_source_deltas"] is True
    assert "copy_shape_matching_source_deltas" in asdict(cfg)
    for bad in (
        {"procrustes_source": "gradient"}, {"tv_scaling": "per_task"}, {"fidelity_holdout": True},
        {"block_split": "joint"}, {"alignment_row_weighting": "norm"}, {"calibration_data": "tiny_imagenet"},
        {"endpoint_construction": "sequential_source_endpoints"},
    ):
        with pytest.raises(ValueError):
            resolve_ariadne_decoder_config(bad, object())
    with pytest.raises(ValueError, match="family adapter"):
        resolve_ariadne_decoder_config({}, None)
