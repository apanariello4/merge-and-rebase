"""Top-level ``transport_calibration_data``: task-independent THESEUS/BiCo calibration.

The context itself is the same `_build_direct_paired_calibration_context` Tiny-ImageNet
split Direct Residual uses (identity and determinism pinned in
test_direct_residual_calibration_data.py); here only the resolver contract.
"""

from __future__ import annotations

import pytest

from merge_and_rebase.eval.vision_rebase import _resolve_transport_calibration_data


def test_default_is_task_local_for_every_method():
    for flags in ((True, False), (False, True), (False, False)):
        assert _resolve_transport_calibration_data({}, theseus_like_method=flags[0], bico_mode=flags[1]) == "task_local"


@pytest.mark.parametrize("flags", [(True, False), (False, True)])
def test_tiny_imagenet_accepted_for_theseus_and_bico(flags):
    cfg = {"transport_calibration_data": "Tiny_ImageNet"}
    assert _resolve_transport_calibration_data(cfg, theseus_like_method=flags[0], bico_mode=flags[1]) == "tiny_imagenet"


def test_rejects_unknown_value():
    with pytest.raises(ValueError, match="transport_calibration_data must be one of"):
        _resolve_transport_calibration_data(
            {"transport_calibration_data": "imagenet"}, theseus_like_method=True, bico_mode=False
        )


def test_rejects_non_default_for_other_methods():
    with pytest.raises(ValueError, match="THESEUS/BiCo only"):
        _resolve_transport_calibration_data(
            {"transport_calibration_data": "tiny_imagenet"}, theseus_like_method=False, bico_mode=False
        )
