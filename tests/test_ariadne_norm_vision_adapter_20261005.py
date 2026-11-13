"""Pure-helper tests of the vision adapter of the ariadne_norm diagnostics (no model or dataset loading)."""

import pytest
import torch

from merge_and_rebase.eval.diagnostics.ariadne_norm.vision import (
    reference_vector_path,
    strip_tensor_fields,
    token_index_for_batch,
    vision_token_index,
)


def test_token_index_is_batch_major_with_cls_first():
    idx = token_index_for_batch(3, 12)
    assert idx.tolist() == [0, 1, 2, 3] * 3
    batch = [torch.zeros(3, 3, 8, 8), torch.zeros(3, dtype=torch.long)]
    assert torch.equal(vision_token_index(batch, 12), idx)


def test_token_index_rejects_indivisible_rows():
    with pytest.raises(ValueError):
        token_index_for_batch(3, 10)


def test_reference_vector_path_resolution(tmp_path):
    tvs = tmp_path / "transported_tvs"
    assert reference_vector_path(tmp_path, "DTD") is None
    tvs.mkdir()
    assert reference_vector_path(tmp_path, "DTD") is None
    (tvs / "DTD_direct_residual_transported_native.pt").write_bytes(b"x")
    assert reference_vector_path(tmp_path, "DTD") == tvs / "DTD_direct_residual_transported_native.pt"
    (tvs / "Cars_ariadne_transported_native.pt").write_bytes(b"x")
    assert reference_vector_path(tmp_path, "Cars") == tvs / "Cars_ariadne_transported_native.pt"
    (tvs / "DTD_ariadne_transported_native.pt").write_bytes(b"x")
    with pytest.raises(ValueError, match="ambiguous"):
        reference_vector_path(tmp_path, "DTD")


def test_strip_tensor_fields_keeps_scalars():
    rows = [{"position": 0, "bias_correction": [0.1, 0.2], "ridge": 1.5}]
    assert strip_tensor_fields(rows) == [{"position": 0, "ridge": 1.5}]
    assert "bias_correction" in rows[0]
