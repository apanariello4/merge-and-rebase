from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.target_informed_runtime import (
    capture_residual_references,
    capture_tokens,
    complete_residuals,
    load_cache,
    paired_calibration,
    parse_target_shared_config,
    projection_transforms,
    save_cache,
    scale_completion,
)
from merge_and_rebase.eval.target_residual_completion import ResidualCompletionConfig


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 2)), ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(width * 2, width)),
        ]))
        self.ls_2 = nn.Identity()

    def forward(self, x):
        return x + self.ls_2(self.mlp(x))


class Visual(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = nn.Linear(4, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class Model(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def loader():
    return DataLoader(TensorDataset(torch.randn(6, 5, 4), torch.arange(6)), batch_size=2, shuffle=False)


def test_pairing_rejects_unrelated_images_even_with_same_labels():
    first, other = loader(), loader()
    with pytest.raises(ValueError, match="identities"):
        paired_calibration(first, other, num_batches=2)
    a, b, metadata = paired_calibration(first, first, num_batches=2, seed=17)
    assert metadata["indices"] == torch.randperm(6, generator=torch.Generator().manual_seed(17)).tolist()[:4]
    assert all(torch.equal(x[0], y[0]) for x, y in zip(a, b, strict=True))


def test_capture_cleans_hooks_and_preserves_model_on_failure():
    model = Model(3, 2).train()
    bad_batches = [(torch.ones(2, 5, 8), torch.zeros(2))]
    with pytest.raises(RuntimeError):
        capture_tokens(model, bad_batches, {"out": (1, "boundary")}, "cpu")
    assert model.training
    assert not model.visual.transformer.resblocks[1]._forward_hooks


def test_sequential_completion_changes_only_added_projections_and_restores_model():
    torch.manual_seed(12)
    source, target = Model(3, 2).eval(), Model(5, 4).eval()
    source_ft = deepcopy(source)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.1)
        source_ft.visual.transformer.resblocks[1].mlp.c_proj.weight.sub_(0.08)
    data = loader()
    refs = capture_residual_references(source, source_ft, target, data, data, num_batches=3, seed=0, device="cpu")
    baseline_state = {k: v.clone() for k, v in target.state_dict().items()}
    baseline_delta = {k: torch.zeros_like(v) for k, v in baseline_state.items()}
    layout = {"inserted_blocks": [{"position": 1, "source_orig_idx": 0}, {"position": 3, "source_orig_idx": 1}]}
    transforms = {}
    for j in [1, 3]:
        transforms[j] = {"t_in": torch.linalg.qr(torch.randn(10, 6)).Q.T,
                         "t_out": torch.linalg.qr(torch.randn(5, 3)).Q.T}
    # Exercise fixed nontrivial LayerScale as well as ordinary identity scale.
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(5))
        def forward(self, x):
            return x * self.gamma
    target.visual.transformer.resblocks[3].ls_2 = Scale()
    baseline_state = {k: v.clone() for k, v in target.state_dict().items()}
    baseline_delta = {k: torch.zeros_like(v) for k, v in baseline_state.items()}
    config = ResidualCompletionConfig(enabled=True, ridge_relative=0.01)
    src, tgt, diag = complete_residuals(target, baseline_state, baseline_delta, refs, transforms, layout, data, config=config, device="cpu")
    expected = {f"visual.transformer.resblocks.{j}.mlp.c_proj.weight" for j in [1, 3]}
    expected |= {f"visual.transformer.resblocks.{j}.mlp.c_proj.bias" for j in [1, 3]}
    assert set(src) == set(tgt) == expected
    assert len(diag) == 2
    for j, row in zip([1, 3], diag, strict=True):
        key = f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"
        bias_key = f"visual.transformer.resblocks.{j}.mlp.c_proj.bias"
        torch.testing.assert_close(tgt[key], transforms[j]["t_out"].T @ src[key] @ transforms[j]["t_in"])
        torch.testing.assert_close(tgt[bias_key], transforms[j]["t_out"].T @ src[bias_key])
        torch.testing.assert_close(src[bias_key], row["bias_correction"])
        assert row["residual_norm_after"] <= row["residual_norm_before"] + 1e-7
    for k, v in target.state_dict().items():
        torch.testing.assert_close(v, baseline_state[k], rtol=0, atol=0)
    unchanged = scale_completion(baseline_delta, tgt, 0)
    assert all(unchanged[k] is baseline_delta[k] for k in unchanged)
    for key, value in scale_completion(baseline_delta, tgt, 0.5).items():
        torch.testing.assert_close(value, baseline_delta[key] + 0.5 * tgt.get(key, torch.zeros_like(value)))


def test_transform_lookup_and_missing_map():
    tin, tout = torch.randn(6, 10), torch.randn(3, 5)
    key = "transformer.resblocks.1.mlp.c_proj.weight"
    layout = {"inserted_blocks": [{"position": 1}]}
    prepared = {"transforms_by_key": {key: SimpleNamespace(kind="weight", t_in=tin, t_out=tout)}}
    maps = projection_transforms(prepared, layout)
    torch.testing.assert_close(maps[1]["t_out"], tout)
    with pytest.raises(ValueError, match="Missing"):
        projection_transforms({}, layout)


def test_cache_rejects_wrong_provenance_and_overwrite(tmp_path):
    path = tmp_path / "cache.pt"
    identity = {"task": "test", "target_hash": "abc"}
    save_cache(path, {"identity": identity, "delta": torch.ones(2)})
    assert torch.equal(load_cache(path, identity)["delta"], torch.ones(2))
    with pytest.raises(ValueError, match="provenance"):
        load_cache(path, {"task": "test", "target_hash": "different"})
    with pytest.raises(FileExistsError):
        save_cache(path, {})


@pytest.mark.parametrize("raw", [{"target_weight": float("nan")}, {"target_weight": -1}, {"num_batches": 0}, {"added_blocks": "last4"}])
def test_shared_config_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        parse_target_shared_config(raw)
