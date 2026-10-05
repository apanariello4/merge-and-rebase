"""CPU tests of the LLM adapter's pure helpers: within-sequence token index and reference-vector extraction."""

import pytest
import torch

from merge_and_rebase.eval.diagnostics.ariadne_norm.llm import (
    extract_reference_task_vector,
    llm_token_index,
    token_index_from_mask,
)
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY


def test_token_index_right_padded():
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    idx = token_index_from_mask(mask, int(mask.sum()))
    assert idx.tolist() == [0, 1, 2, 0, 0, 1, 2, 3, 4]


def test_token_index_left_padded_starts_at_first_content_token():
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=torch.bool)
    idx = token_index_from_mask(mask, int(mask.sum()))
    assert idx.tolist() == [0, 1, 2, 0, 0, 1, 2, 3, 4]
    # row-major alignment with the capture's ``tokens[mask]``
    ids = torch.arange(15).reshape(3, 5)
    assert ids[mask].tolist() == [2, 3, 4, 9, 10, 11, 12, 13, 14]


def test_token_index_from_batch_and_row_count_check():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    batch = {"input_ids": torch.zeros(2, 3, dtype=torch.long), ROW_MASK_KEY: mask}
    assert llm_token_index(batch, 3).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        llm_token_index(batch, 4)
    with pytest.raises(ValueError):
        llm_token_index({"input_ids": torch.zeros(2, 3)}, 3)


def test_extract_reference_task_vector_recovers_tau_and_selects_keys():
    g = torch.Generator().manual_seed(0)
    base = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 6, generator=g),
        "model.layers.0.mlp.down_proj.bias": torch.zeros(4),
        "model.embed_tokens.weight": torch.randn(5, 4, generator=g),
    }
    tau = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 6, generator=g),
        "model.layers.0.mlp.down_proj.bias": torch.randn(4, generator=g),
    }
    merged = {k: v + 1.0 * tau[k] if k in tau else v.clone() for k, v in base.items()}
    keys = list(tau)
    got = extract_reference_task_vector(merged, base, keys)
    assert set(got) == set(keys)
    for k in keys:
        assert got[k].dtype == torch.float32
        torch.testing.assert_close(got[k], tau[k], rtol=0, atol=1e-6)
    with pytest.raises(KeyError):
        extract_reference_task_vector(merged, base, ["nope.weight"])
