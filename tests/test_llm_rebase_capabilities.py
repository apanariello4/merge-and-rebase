from __future__ import annotations

import pytest

from merge_and_rebase.rebase.capabilities import check_pair, is_same_size, needs_depth_upsize
from merge_and_rebase.rebase.model_families.base import ModelFamilyMetadata


def _meta(family="llama", hidden=4096, intermediate=11008, layers=32, heads=32):
    return ModelFamilyMetadata(
        family=family,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
    )


def test_same_size_allowed() -> None:
    src = _meta()
    tgt = _meta()
    check_pair("identity", src, tgt)
    check_pair("gradfix", src, tgt)
    check_pair("theseus", src, tgt)
    check_pair("bico", src, tgt)


def test_cross_size_allowed_for_thesus() -> None:
    src = _meta(hidden=4096, intermediate=11008)
    tgt = _meta(hidden=8192, intermediate=28672)
    check_pair("theseus", src, tgt)
    check_pair("bico", src, tgt)


def test_cross_size_rejected_for_identity() -> None:
    src = _meta(hidden=4096)
    tgt = _meta(hidden=8192)
    with pytest.raises(ValueError, match="cross-size"):
        check_pair("identity", src, tgt)


def test_cross_size_rejected_for_gradfix() -> None:
    src = _meta(hidden=4096)
    tgt = _meta(hidden=8192)
    with pytest.raises(ValueError, match="cross-size"):
        check_pair("gradfix", src, tgt)


def test_cross_size_rejected_for_orthogonal_shift() -> None:
    src = _meta(hidden=4096)
    tgt = _meta(hidden=8192)
    with pytest.raises(ValueError, match="cross-size"):
        check_pair("orthogonal_shift", src, tgt)


def test_family_mismatch_rejected() -> None:
    src = _meta(family="llama")
    tgt = _meta(family="qwen2")
    with pytest.raises(ValueError, match="family mismatch"):
        check_pair("theseus", src, tgt)


def test_source_deeper_than_target_rejected() -> None:
    src = _meta(layers=40)
    tgt = _meta(layers=32)
    with pytest.raises(ValueError, match="more layers"):
        check_pair("theseus", src, tgt)


def test_source_deeper_than_target_allowed_for_block_extension() -> None:
    src = _meta(layers=40)
    tgt = _meta(layers=32)
    check_pair("theseus", src, tgt, allow_depth_mismatch=True)


def test_transfusion_not_available() -> None:
    src = _meta()
    tgt = _meta()
    with pytest.raises(ValueError, match="not available"):
        check_pair("transfusion", src, tgt)


def test_is_same_size_true() -> None:
    assert is_same_size(_meta(hidden=4096), _meta(hidden=4096))


def test_is_same_size_false_hidden() -> None:
    assert not is_same_size(_meta(hidden=4096), _meta(hidden=8192))


def test_needs_depth_upsize() -> None:
    assert needs_depth_upsize(_meta(layers=32), _meta(layers=40))
    assert not needs_depth_upsize(_meta(layers=40), _meta(layers=32))
