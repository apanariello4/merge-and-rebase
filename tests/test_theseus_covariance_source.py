"""Regression tests for the Theseus ``covariance_source`` ablation switch.

The switch decides which source endpoint the Procrustes alignment is fitted on:
the base endpoint (the historical behaviour), the fine-tuned endpoint, their
difference, or an interpolation.  Two properties have to hold before any of its
numbers may sit next to the existing tables.

1. ``covariance_source='base'`` must be bit-identical to not passing the field
   at all, so the campaign's pre-existing baseline row stays comparable across
   the code boundary this change introduces.
2. Combining the two accumulated banks must equal streaming the combined rows
   directly.  The combination is only exact because the cross-covariance is
   linear in the source rows, and that linearity is what the tests pin.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from merge_and_rebase.rebase.methods.theseus import (
    _FT_REGISTRY_PREFIX,
    ActivationStore,
    TheseusRebase,
    _combine_activation_registry,
    _covariance_source_coefficients,
)


def _models() -> tuple[nn.Module, nn.Module, nn.Module]:
    torch.manual_seed(0)
    source = nn.Linear(4, 4, bias=False)
    target = nn.Linear(4, 4, bias=False)
    source_ft = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        source_ft.weight.copy_(source.weight + 0.25 * torch.randn_like(source.weight))
    return source, target, source_ft


def _batches() -> list[tuple[torch.Tensor]]:
    torch.manual_seed(1)
    return [(torch.randn(3, 4),), (torch.randn(3, 4),)]


def _prepare(**overrides):
    source, target, source_ft = _models()
    batches = _batches()
    kwargs = dict(
        source_model=source,
        target_model=target,
        source_dataloader=batches,
        target_dataloader=batches,
        device="cpu",
        n_batches=2,
        patch_qkv=False,
        verbose=False,
    )
    if overrides.pop("with_ft", False):
        kwargs["source_model_ft"] = source_ft
    kwargs.update(overrides)
    return TheseusRebase().prepare(**kwargs)


def _covariances(prepared, *, center: bool) -> dict[str, torch.Tensor]:
    return {
        key: store.get_covariance(center=center)
        for key, store in prepared["activation_registry"].items()
        if store.get_covariance(center=center) is not None
    }


def test_default_covariance_source_is_bit_identical_to_explicit_base() -> None:
    implicit = _covariances(_prepare(), center=True)
    explicit = _covariances(_prepare(covariance_source="base"), center=True)

    assert implicit.keys() == explicit.keys()
    assert implicit, "the toy model must produce at least one activation slot"
    for key in implicit:
        assert torch.equal(implicit[key], explicit[key]), key


def test_collecting_the_ft_bank_does_not_perturb_the_base_bank() -> None:
    # The fine-tuned endpoint runs in the same pass on the same batches; if its
    # forward changed the base statistics, every 'base' row of the ablation
    # would be incomparable to the existing tables.
    without_ft = _covariances(_prepare(), center=False)
    with_ft = _covariances(_prepare(with_ft=True, covariance_source="mixture", covariance_mixture_beta=0.0), center=False)

    assert without_ft.keys() == with_ft.keys()
    for key in without_ft:
        assert torch.allclose(without_ft[key], with_ft[key], atol=0.0, rtol=0.0), key


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize(
    ("covariance_source", "beta"),
    [("ft", 0.5), ("delta", 0.5), ("mixture", 0.25), ("mixture", 0.75)],
)
def test_combination_equals_streaming_the_combined_rows(covariance_source: str, beta: float, center: bool) -> None:
    torch.manual_seed(7)
    base_rows = torch.randn(12, 3, dtype=torch.float64)
    ft_rows = torch.randn(12, 3, dtype=torch.float64)
    target_rows = torch.randn(12, 5, dtype=torch.float64)

    base_store, ft_store = ActivationStore(), ActivationStore()
    for start in (0, 6):
        stop = start + 6
        base_store.update(base_rows[start:stop], target_rows[start:stop])
        ft_store.update(ft_rows[start:stop], target_rows[start:stop])

    combined = _combine_activation_registry(
        {"layer.in": base_store, f"{_FT_REGISTRY_PREFIX}layer.in": ft_store},
        covariance_source=covariance_source,
        mixture_beta=beta,
    )

    c_base, c_ft = _covariance_source_coefficients(covariance_source, beta)
    direct = ActivationStore()
    effective_rows = c_base * base_rows + c_ft * ft_rows
    for start in (0, 6):
        stop = start + 6
        direct.update(effective_rows[start:stop], target_rows[start:stop])

    assert torch.allclose(
        combined["layer.in"].get_covariance(center=center),
        direct.get_covariance(center=center),
        atol=1e-10,
    )


def test_coefficients_cover_the_documented_sources() -> None:
    assert _covariance_source_coefficients("base", 0.5) == (1.0, 0.0)
    assert _covariance_source_coefficients("ft", 0.5) == (0.0, 1.0)
    assert _covariance_source_coefficients("delta", 0.5) == (-1.0, 1.0)
    assert _covariance_source_coefficients("mixture", 0.0) == (1.0, 0.0)
    assert _covariance_source_coefficients("mixture", 1.0) == (0.0, 1.0)


def test_non_base_source_requires_the_ft_model() -> None:
    with pytest.raises(ValueError, match="requires source_model_ft"):
        _prepare(covariance_source="delta")


def test_base_source_refuses_an_unused_ft_model() -> None:
    with pytest.raises(ValueError, match="would ignore it"):
        _prepare(with_ft=True, covariance_source="base")


def test_whitening_is_refused_for_a_non_base_source() -> None:
    # The whitening Gram is quadratic in the source rows and therefore is not
    # recoverable from the two accumulated banks; silently whitening the base
    # Gram would misreport the method.
    with pytest.raises(ValueError, match="only implemented for covariance_source"):
        _prepare(with_ft=True, covariance_source="delta", whiten_power=0.25)


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="covariance_source must be one of"):
        _prepare(covariance_source="gradients")


def test_mixture_beta_is_validated_even_when_inactive() -> None:
    with pytest.raises(ValueError, match="covariance_mixture_beta"):
        _prepare(covariance_mixture_beta=1.5)
