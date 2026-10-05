"""Tests for the pre-merge conditioning of transported task vectors.

These operations sit between a hash-checked transported bank and the merger, so
they are exactly the place where a silent change would move a published merge
number without changing any config that the summary reports.  The default must
therefore be a verified identity, and each active mode must be pinned to the
property it claims: equal contributed energy for ``norm_match``, a measured
retained density for ``magnitude_trim``, and a measured retained spectral
energy for ``rank_trim``.
"""

from __future__ import annotations

import math

import pytest
import torch

from merge_and_rebase.merge.tv_conditioning import (
    ConditioningSpec,
    condition_transported_deltas,
    spec_from_config,
)


def _deltas() -> dict[str, dict[str, torch.Tensor]]:
    torch.manual_seed(11)
    return {
        "EuroSAT": {"visual.w": torch.randn(6, 5), "visual.b": torch.randn(6)},
        "GTSRB": {"visual.w": 9.0 * torch.randn(6, 5), "visual.b": 9.0 * torch.randn(6)},
        "Cars": {"visual.w": 0.05 * torch.randn(6, 5), "visual.b": 0.05 * torch.randn(6)},
    }


def _global_norm(delta: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(t.double().pow(2).sum()) for t in delta.values()))


def test_off_is_an_exact_identity() -> None:
    deltas = _deltas()
    out, diagnostics = condition_transported_deltas(deltas, ConditioningSpec())

    assert diagnostics["spec"] == {"mode": "off"}
    for task, delta in deltas.items():
        for key, tensor in delta.items():
            assert torch.equal(out[task][key], tensor)


def test_norm_match_global_equalises_the_energy_each_task_contributes() -> None:
    deltas = _deltas()
    out, diagnostics = condition_transported_deltas(deltas, ConditioningSpec(mode="norm_match", scope="global"))

    norms = [_global_norm(out[task]) for task in deltas]
    reference = diagnostics["reference_norm"]
    assert reference == pytest.approx(sum(_global_norm(d) for d in deltas.values()) / len(deltas))
    for norm in norms:
        assert norm == pytest.approx(reference, rel=1e-6)
    # Rescaling is a pure per-task gain: direction must be untouched.
    scale = diagnostics["scale"]["Cars"]
    assert torch.allclose(out["Cars"]["visual.w"], deltas["Cars"]["visual.w"] * scale)


def test_norm_match_geomean_reference_differs_from_the_mean_on_a_skewed_set() -> None:
    deltas = _deltas()
    _, mean_diag = condition_transported_deltas(deltas, ConditioningSpec(mode="norm_match", reference="mean"))
    _, geo_diag = condition_transported_deltas(deltas, ConditioningSpec(mode="norm_match", reference="geomean"))

    assert geo_diag["reference_norm"] < mean_diag["reference_norm"]


def test_norm_match_per_tensor_equalises_every_key() -> None:
    deltas = _deltas()
    out, _ = condition_transported_deltas(deltas, ConditioningSpec(mode="norm_match", scope="per_tensor"))

    for key in ("visual.w", "visual.b"):
        norms = [float(out[task][key].double().norm()) for task in deltas]
        assert norms[0] == pytest.approx(norms[1], rel=1e-6)
        assert norms[0] == pytest.approx(norms[2], rel=1e-6)


def test_magnitude_trim_keeps_the_requested_density_of_the_largest_entries() -> None:
    deltas = _deltas()
    spec = ConditioningSpec(mode="magnitude_trim", scope="global", density=0.25)
    out, diagnostics = condition_transported_deltas(deltas, spec)

    for task in deltas:
        stats = diagnostics["per_task"][task]
        assert stats["achieved_density"] == pytest.approx(0.25, abs=0.02)
        surviving = torch.cat([out[task][key].reshape(-1) for key in sorted(out[task])])
        original = torch.cat([deltas[task][key].reshape(-1) for key in sorted(deltas[task])])
        kept = surviving != 0
        # Every retained entry is untouched, and no dropped entry is larger
        # than a retained one.
        assert torch.equal(surviving[kept], original[kept])
        assert original[kept].abs().min() >= original[~kept].abs().max()


def test_magnitude_trim_rescale_restores_the_expected_magnitude() -> None:
    deltas = _deltas()
    plain, _ = condition_transported_deltas(
        deltas, ConditioningSpec(mode="magnitude_trim", density=0.5, rescale=False)
    )
    rescaled, _ = condition_transported_deltas(
        deltas, ConditioningSpec(mode="magnitude_trim", density=0.5, rescale=True)
    )

    assert torch.allclose(rescaled["EuroSAT"]["visual.w"], plain["EuroSAT"]["visual.w"] * 2.0)


def test_rank_trim_truncates_matrices_and_leaves_vectors_alone() -> None:
    deltas = _deltas()
    spec = ConditioningSpec(mode="rank_trim", density=0.4)
    out, diagnostics = condition_transported_deltas(deltas, spec)

    for task in deltas:
        assert torch.equal(out[task]["visual.b"], deltas[task]["visual.b"])
        # The reconstruction is stored back in the delta's float32 dtype, so
        # the discarded directions survive as rounding noise rather than exact
        # zeros; compare them against the leading singular value.
        singular = torch.linalg.svdvals(out[task]["visual.w"].double())
        assert float(singular[2:].max()) < 1e-5 * float(singular[0])
        retained = diagnostics["per_task"][task]["retained_energy_fraction"]
        assert 0.0 < retained < 1.0
        assert diagnostics["per_task"][task]["truncated_tensors"] == 1.0


def test_spec_from_config_defaults_to_off_and_rejects_typos() -> None:
    assert spec_from_config(None).mode == "off"
    assert spec_from_config({}).mode == "off"
    assert spec_from_config({"mode": "norm_match", "scope": "per_tensor"}).scope == "per_tensor"

    with pytest.raises(ValueError, match="Unknown tv_conditioning fields"):
        spec_from_config({"mode": "norm_match", "scopes": "global"})
    with pytest.raises(ValueError, match="density must be in"):
        spec_from_config({"mode": "magnitude_trim", "density": 0.0})
    with pytest.raises(ValueError, match="mode must be one of"):
        spec_from_config({"mode": "ties"})


def test_mean_removal_at_full_strength_centres_the_set() -> None:
    deltas = _deltas()
    out, diagnostics = condition_transported_deltas(deltas, ConditioningSpec(mode="mean_removal", strength=1.0))

    for key in ("visual.w", "visual.b"):
        centred = torch.stack([out[task][key].double() for task in deltas]).mean(dim=0)
        assert float(centred.abs().max()) < 1e-6
    assert diagnostics["strength"] == 1.0
    assert diagnostics["centroid_norm"] > 0.0


def test_mean_removal_at_partial_strength_removes_that_fraction_of_the_centroid() -> None:
    deltas = _deltas()
    full, _ = condition_transported_deltas(deltas, ConditioningSpec(mode="mean_removal", strength=1.0))
    half, _ = condition_transported_deltas(deltas, ConditioningSpec(mode="mean_removal", strength=0.5))

    for task in deltas:
        expected = 0.5 * (deltas[task]["visual.w"].double() + full[task]["visual.w"].double())
        assert torch.allclose(half[task]["visual.w"].double(), expected, atol=1e-6)


def test_mean_removal_strength_is_validated() -> None:
    with pytest.raises(ValueError, match="strength must be in"):
        spec_from_config({"mode": "mean_removal", "strength": 1.5})
