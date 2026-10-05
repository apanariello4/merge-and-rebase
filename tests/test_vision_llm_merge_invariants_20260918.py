"""Regression tests for the vision/text split preserved by the branch merge.

The vision and text lines of work were developed on separate branches and
reconciled in one merge. Two independent fixes landed in the same Theseus
calibration loop: the fine-tuned-source covariance plumbing (vision) and the
padding-row filter (text). The tests below pin the properties that make those
two fixes coexist, so a later refactor cannot silently drop one of them.
"""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.rebase.methods import theseus as theseus_mod


def test_padding_filter_is_an_exact_identity_without_a_mask() -> None:
    """The vision path passes ``row_mask=None`` and must be untouched by it.

    Vision calibration has no attention mask, so every vision run flows through
    ``_drop_padding_rows`` with ``None``. It has to return the very same tensor
    objects, not merely equal ones, for vision numbers to be unchanged.
    """
    source_rows = torch.randn(7, 4)
    target_rows = torch.randn(7, 5)

    out_source, out_target = theseus_mod._drop_padding_rows(source_rows, target_rows, None)

    assert out_source is source_rows
    assert out_target is target_rows


def test_padding_filter_drops_exactly_the_padded_rows() -> None:
    """The text path must keep only the rows both endpoints mark as content."""
    source_rows = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    target_rows = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    mask = torch.tensor([True, True, False, True, False, True])

    out_source, out_target = theseus_mod._drop_padding_rows(source_rows, target_rows, mask)

    assert torch.equal(out_source, source_rows[mask])
    assert torch.equal(out_target, target_rows[mask])


def test_row_count_mismatch_leaves_pooled_features_alone() -> None:
    """Head-split/pooled features are not one row per token and must pass through."""
    source_rows = torch.randn(3, 4)
    target_rows = torch.randn(3, 5)
    mask = torch.tensor([True, False, True, True, False, True])

    out_source, out_target = theseus_mod._drop_padding_rows(source_rows, target_rows, mask)

    assert out_source is source_rows
    assert out_target is target_rows


def test_deliberate_all_zero_transport_is_allowed() -> None:
    """The vision depth baselines zero every key on purpose; that is not a failure."""
    _, diagnostics = theseus_mod._apply_transforms_to_visual_delta(
        target_visual_base={"class_embedding": torch.zeros(2)},
        visual_delta={"class_embedding": torch.ones(2)},
        transforms_by_key={"class_embedding": theseus_mod._LayerTransform(kind="zero")},
        show_progress=False,
        method_name="theseus",
        device="cpu",
        strict=True,
    )

    assert diagnostics.intentional_zero == 1
    assert diagnostics.actively_transported == 0


def test_accidental_all_zero_transport_raises_even_when_not_strict() -> None:
    """A run that transports nothing by accident must never be returned silently.

    Without a transform the result is a correctly shaped set of zeros that
    downstream code cannot tell apart from a real transport, so this is refused
    regardless of ``strict``.
    """
    with pytest.raises(RuntimeError, match="transported no keys"):
        theseus_mod._apply_transforms_to_visual_delta(
            target_visual_base={"visual.w": torch.zeros(2, 2)},
            visual_delta={"visual.w": torch.ones(2, 2)},
            transforms_by_key={},
            show_progress=False,
            method_name="theseus",
            device="cpu",
            strict=False,
        )
