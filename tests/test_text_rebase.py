from __future__ import annotations

import torch

from merge_and_rebase.eval.text_rebase import (
    _copy_selected_parameters,
    align_task_vector,
    process_tokens,
    svd_transport,
)
from merge_and_rebase.rebase.methods.theseus import ActivationStore


def test_process_tokens_interpolates_source_sequence_to_target() -> None:
    source = torch.randn(2, 3, 4)
    target = torch.randn(2, 5, 6)

    source_rows, target_rows = process_tokens(source, target, "interpolate")

    assert source_rows.shape == (10, 4)
    assert target_rows.shape == (10, 6)


def test_svd_transport_supports_different_source_and_target_widths() -> None:
    cov_in = torch.randn(3, 5)
    cov_out = torch.randn(4, 6)
    source_weight = torch.randn(4, 3)

    transported = svd_transport(cov_in, cov_out, source_weight)

    assert transported.shape == (6, 5)
    assert transported.dtype == torch.float32


def test_align_task_vector_uses_input_and_output_activation_maps() -> None:
    source_model = torch.nn.Linear(3, 4)
    target_model = torch.nn.Linear(5, 6)
    in_store = ActivationStore()
    out_store = ActivationStore()
    in_store.update(torch.randn(12, 3), torch.randn(12, 5))
    out_store.update(torch.randn(12, 4), torch.randn(12, 6))

    aligned = align_task_vector(
        target_model,
        {"weight": torch.randn_like(source_model.weight)},
        {".in": in_store, ".out": out_store},
    )

    assert aligned["weight"].shape == target_model.weight.shape
    assert torch.count_nonzero(aligned["weight"]) > 0


def test_align_task_vector_zeroes_layers_without_statistics() -> None:
    target_model = torch.nn.Linear(5, 6)

    aligned = align_task_vector(target_model, {"weight": torch.randn(4, 3)}, {})

    assert torch.equal(aligned["weight"], torch.zeros_like(target_model.weight))


def test_copy_selected_parameters_only_copies_compatible_head_tensors() -> None:
    destination = {
        "encoder.weight": torch.zeros(2, 2),
        "classification_head.weight": torch.zeros(3, 2),
        "classification_head.bias": torch.zeros(3),
    }
    source = {
        "encoder.weight": torch.ones(2, 2),
        "classification_head.weight": torch.ones(3, 2),
        "classification_head.bias": torch.ones(4),
    }

    copied = _copy_selected_parameters(destination, source, ("classification_head",))

    assert copied == ["classification_head.weight"]
    assert torch.equal(destination["classification_head.weight"], torch.ones(3, 2))
    assert torch.equal(destination["classification_head.bias"], torch.zeros(3))
    assert torch.equal(destination["encoder.weight"], torch.zeros(2, 2))
