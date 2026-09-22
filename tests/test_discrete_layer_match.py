from __future__ import annotations

import torch
from torch import nn

from merge_and_rebase.rebase.discrete_layer_match import (
    DiscreteLayerPairing,
    build_discrete_indexed_model,
    discrete_layer_pairing,
    reindex_state_dict,
)


def test_exact_sequences_extend():
    assert discrete_layer_pairing(12, 24) == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
    ]


def test_exact_sequences_shrink():
    assert discrete_layer_pairing(24, 12) == [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23]


def test_exact_sequences_same_arch():
    assert discrete_layer_pairing(12, 12) == list(range(12))


def test_identity_for_equal_depths():
    for depth in (1, 12, 24):
        assert discrete_layer_pairing(depth, depth) == list(range(depth))


def test_target_depth_one_special_case():
    for source_depth in (1, 12, 24):
        assert discrete_layer_pairing(source_depth, 1) == [0]


def test_endpoints_when_target_depth_greater_than_one():
    for source_depth, target_depth in [(12, 24), (24, 12), (12, 12), (1, 5)]:
        pairing = discrete_layer_pairing(source_depth, target_depth)
        assert pairing[0] == 0
        assert pairing[-1] == source_depth - 1


def test_monotonic_non_decreasing():
    for source_depth, target_depth in [(12, 24), (24, 12), (12, 12), (5, 17), (17, 5)]:
        pairing = discrete_layer_pairing(source_depth, target_depth)
        assert all(a <= b for a, b in zip(pairing, pairing[1:], strict=False))


def test_extend_case_near_uniform_duplication():
    pairing = discrete_layer_pairing(12, 24)
    counts: dict[int, int] = {}
    for idx in pairing:
        counts[idx] = counts.get(idx, 0) + 1
    assert set(counts.keys()) == set(range(12))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_shrink_case_no_reuse():
    pairing = discrete_layer_pairing(24, 12)
    assert len(set(pairing)) == 12


class TestDiscreteLayerPairingDataclass:
    def test_compute_matches_pure_function(self):
        dlp = DiscreteLayerPairing.compute(12, 24)
        assert dlp.source_depth == 12
        assert dlp.target_depth == 24
        assert dlp.pairing == tuple(discrete_layer_pairing(12, 24))

    def test_frozen(self):
        dlp = DiscreteLayerPairing.compute(12, 12)
        try:
            dlp.source_depth = 99  # type: ignore[misc]
        except AttributeError:
            pass
        else:
            raise AssertionError("DiscreteLayerPairing should be frozen")


def _fake_state_dict(num_blocks: int) -> dict[str, torch.Tensor]:
    sd = {"visual.class_embedding": torch.full((3,), -1.0)}
    for i in range(num_blocks):
        sd[f"transformer.resblocks.{i}.mlp.c_proj.weight"] = torch.full((2, 2), float(i))
    return sd


def test_reindex_state_dict_extend():
    source_sd = _fake_state_dict(4)
    pairing = DiscreteLayerPairing.compute(4, 8)
    out = reindex_state_dict(source_sd, pairing)

    for j, source_idx in enumerate(pairing.pairing):
        key = f"transformer.resblocks.{j}.mlp.c_proj.weight"
        assert key in out
        assert torch.equal(out[key], source_sd[f"transformer.resblocks.{source_idx}.mlp.c_proj.weight"])

    assert "visual.class_embedding" in out
    assert torch.equal(out["visual.class_embedding"], source_sd["visual.class_embedding"])

    block_indices = set()
    for key in out:
        if key.startswith("transformer.resblocks."):
            block_indices.add(int(key.split(".")[2]))
    assert block_indices == set(range(8))


def test_reindex_state_dict_shrink():
    source_sd = _fake_state_dict(8)
    pairing = DiscreteLayerPairing.compute(8, 4)
    out = reindex_state_dict(source_sd, pairing)

    for j, source_idx in enumerate(pairing.pairing):
        key = f"transformer.resblocks.{j}.mlp.c_proj.weight"
        assert torch.equal(out[key], source_sd[f"transformer.resblocks.{source_idx}.mlp.c_proj.weight"])

    block_indices = {int(k.split(".")[2]) for k in out if k.startswith("transformer.resblocks.")}
    assert block_indices == set(range(4))
    assert torch.equal(out["visual.class_embedding"], source_sd["visual.class_embedding"])


class _FakeVisualTransformer(nn.Module):
    def __init__(self, num_blocks: int):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for i in range(num_blocks):
            linear = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                linear.weight.fill_(float(i))
            self.resblocks.append(linear)


class _FakeVisual(nn.Module):
    def __init__(self, num_blocks: int):
        super().__init__()
        self.transformer = _FakeVisualTransformer(num_blocks)


class _FakeModel(nn.Module):
    def __init__(self, num_blocks: int):
        super().__init__()
        self.visual = _FakeVisual(num_blocks)


def test_build_discrete_indexed_model_extend():
    source_model = _FakeModel(4)
    pairing = DiscreteLayerPairing.compute(4, 8)

    result = build_discrete_indexed_model(source_model, pairing)

    assert len(result.visual.transformer.resblocks) == 8
    for j, source_idx in enumerate(pairing.pairing):
        expected = source_model.visual.transformer.resblocks[source_idx].weight
        actual = result.visual.transformer.resblocks[j].weight
        assert torch.equal(actual, expected)

    # Original untouched.
    assert len(source_model.visual.transformer.resblocks) == 4
    for i, block in enumerate(source_model.visual.transformer.resblocks):
        assert torch.equal(block.weight, torch.full((2, 2), float(i)))


def test_build_discrete_indexed_model_shrink():
    source_model = _FakeModel(8)
    pairing = DiscreteLayerPairing.compute(8, 4)

    result = build_discrete_indexed_model(source_model, pairing)

    assert len(result.visual.transformer.resblocks) == 4
    for j, source_idx in enumerate(pairing.pairing):
        expected = source_model.visual.transformer.resblocks[source_idx].weight
        actual = result.visual.transformer.resblocks[j].weight
        assert torch.equal(actual, expected)

    assert len(source_model.visual.transformer.resblocks) == 8
    for i, block in enumerate(source_model.visual.transformer.resblocks):
        assert torch.equal(block.weight, torch.full((2, 2), float(i)))


def test_build_discrete_indexed_model_deepcopy_isolation():
    source_model = _FakeModel(4)
    pairing = DiscreteLayerPairing.compute(4, 4)

    result = build_discrete_indexed_model(source_model, pairing)

    with torch.no_grad():
        result.visual.transformer.resblocks[0].weight.fill_(999.0)

    assert torch.equal(
        source_model.visual.transformer.resblocks[0].weight,
        torch.zeros(2, 2),
    )
