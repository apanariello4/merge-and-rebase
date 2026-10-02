import pytest
import torch
from test_direct_residual_ablation_v2_golden_hashes_20260925 import _setup

from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig, parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)


def _run(tol, mutate):
    sb, sf, tb, data, pairing, sd = _setup(2, 4)
    kw = {} if tol is None else {"streaming_fingerprint_tol": tol}
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, activation_storage="streaming", **kw)
    prepared = prepare_direct_residual_streaming(
        sb, tb, data, data, pairing, num_batches=3, seed=cfg.seed, device="cpu", source_ft_model=sf
    )
    if mutate:
        with torch.no_grad():
            for p in tb.parameters():
                p.mul_(1.001)
            # the fit reloads the target from this state dict, so mutate it too
            for k in sd:
                if sd[k].is_floating_point():
                    sd[k] = sd[k] * 1.001
    return fit_direct_residual_streaming(tb, sd, sb, sf, prepared, pairing, config=cfg, device="cpu")


def test_default_tol_and_parse():
    assert DirectResidualConfig().streaming_fingerprint_tol == 1e-9
    assert parse_direct_residual_config({"streaming_fingerprint_tol": 1e-6}).streaming_fingerprint_tol == 1e-6
    with pytest.raises(ValueError):
        parse_direct_residual_config({"streaming_fingerprint_tol": 0})


def test_unmutated_passes():
    _run(None, False)


def test_mutated_target_detected_with_drift_in_message():
    with pytest.raises(RuntimeError, match=r"relative drift sumsq=.*streaming_fingerprint_tol=1\.000e-09"):
        _run(None, True)


def test_default_tol_not_serialized_nondefault_is():
    from merge_and_rebase.rebase.methods._ariadne.config import direct_residual_config_dict

    assert "streaming_fingerprint_tol" not in direct_residual_config_dict(DirectResidualConfig())
    assert (
        direct_residual_config_dict(DirectResidualConfig(streaming_fingerprint_tol=1e-6))["streaming_fingerprint_tol"]
        == 1e-6
    )
