from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.target_informed_runtime import (
    capture_residual_references,
    capture_resized_joint_source_inputs,
    capture_tokens,
    complete_direct_p1_shared_correction,
    complete_joint_blockwise,
    complete_residuals,
    load_cache,
    paired_calibration,
    parse_target_shared_config,
    projection_transforms,
    save_cache,
    scale_completion,
)
from merge_and_rebase.eval.target_residual_completion import (
    JointCorrectionConfig,
    ResidualCompletionConfig,
    fit_joint_cproj_correction,
)


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


def test_joint_blockwise_completion_is_additive_and_sequential():
    """Option 3 uses zero source residuals and target boundary residuals.

    The full source endpoint effect must not be fitted again: the returned
    correction is added to the already transported task vector, and the target
    model is restored after each run.
    """
    torch.manual_seed(17)
    source, target = Model(3, 2).eval(), Model(5, 4).eval()
    source_ft = deepcopy(source)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.1)
    data = loader()
    refs = capture_residual_references(
        source,
        source_ft,
        target,
        data,
        data,
        num_batches=3,
        seed=0,
        device="cpu",
        capture_joint=True,
    )
    assert refs["source_base_cproj_inputs"]
    assert refs["target_cproj_inputs_by_position"]
    layout = {"inserted_blocks": [{"position": 1, "source_orig_idx": 0}, {"position": 3, "source_orig_idx": 1}]}
    resized_source = Model(3, 4).eval()
    refs = capture_resized_joint_source_inputs(
        resized_source, data, refs, layout, device="cpu"
    )
    assert set(refs["resized_source_cproj_inputs_by_position"]) == {1, 3}
    transforms = {
        pos: {"t_in": torch.linalg.qr(torch.randn(10, 6)).Q.T, "t_out": torch.linalg.qr(torch.randn(5, 3)).Q.T}
        for pos in (1, 3)
    }
    baseline_state = {k: v.clone() for k, v in target.state_dict().items()}
    baseline_delta = {k: torch.zeros_like(v) for k, v in baseline_state.items()}
    config = JointCorrectionConfig(enabled=True, source_weight=1.0, target_weight=1.0, ridge_relative=0.01)
    source_corr, target_corr, diagnostics = complete_joint_blockwise(
        target,
        baseline_state,
        baseline_delta,
        refs,
        transforms,
        layout,
        data,
        config=config,
        device="cpu",
    )
    assert len(diagnostics) == 2
    assert all(row["solver"] == "blockwise_eigh" for row in diagnostics)
    assert all(row["objective_after"] <= row["objective_before"] + 1e-7 for row in diagnostics)
    assert set(source_corr) == set(target_corr) == {
        "visual.transformer.resblocks.1.mlp.c_proj.weight",
        "visual.transformer.resblocks.1.mlp.c_proj.bias",
        "visual.transformer.resblocks.3.mlp.c_proj.weight",
        "visual.transformer.resblocks.3.mlp.c_proj.bias",
    }
    for pos, row in zip((1, 3), diagnostics, strict=True):
        weight_key = f"visual.transformer.resblocks.{pos}.mlp.c_proj.weight"
        bias_key = f"visual.transformer.resblocks.{pos}.mlp.c_proj.bias"
        torch.testing.assert_close(target_corr[weight_key], transforms[pos]["t_out"].T @ source_corr[weight_key] @ transforms[pos]["t_in"])
        torch.testing.assert_close(target_corr[bias_key], transforms[pos]["t_out"].T @ source_corr[bias_key])
        torch.testing.assert_close(source_corr[bias_key], row["bias_correction"])
        torch.testing.assert_close(target_corr[bias_key], row["transported_bias_correction"])
        assert row["bias_regularized"] is False
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, baseline_state[key], rtol=0, atol=0)


def test_joint_blockwise_disabled_is_a_true_noop():
    target = Model(5, 1).eval()
    baseline_state = {key: value.clone() for key, value in target.state_dict().items()}
    source, correction, diagnostics = complete_joint_blockwise(
        target,
        baseline_state,
        {},
        {},
        {},
        {},
        loader(),
        config=JointCorrectionConfig(),
        device="cpu",
    )
    assert source == correction == {}
    assert diagnostics == []
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, baseline_state[key], rtol=0, atol=0)


def test_direct_p1_shared_correction_is_sequential_and_restores_models():
    torch.manual_seed(29)
    native_source = Model(3, 2).eval()
    native_ft = deepcopy(native_source)
    target = Model(5, 4).eval()
    with torch.no_grad():
        native_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.12)
        native_ft.visual.transformer.resblocks[1].mlp.c_proj.weight.sub_(0.07)
    data = loader()
    refs = capture_residual_references(
        native_source,
        native_ft,
        target,
        data,
        data,
        num_batches=3,
        seed=0,
        device="cpu",
        capture_joint=True,
    )
    resized_base = Model(3, 4).eval()
    resized_ft = deepcopy(resized_base)
    with torch.no_grad():
        resized_ft.visual.transformer.resblocks[1].mlp.c_proj.weight.add_(0.09)
        resized_ft.visual.transformer.resblocks[1].mlp.c_proj.bias.add_(0.02)
        resized_ft.visual.transformer.resblocks[3].mlp.c_proj.weight.sub_(0.06)
    source_base_before = {key: value.clone() for key, value in resized_base.state_dict().items()}
    source_ft_before = {key: value.clone() for key, value in resized_ft.state_dict().items()}
    target_before = {key: value.clone() for key, value in target.state_dict().items()}
    baseline_delta = {key: torch.zeros_like(value) for key, value in target_before.items()}
    layout = {
        "inserted_blocks": [
            {"position": 1, "source_orig_idx": 0},
            {"position": 3, "source_orig_idx": 1},
        ]
    }
    transforms = {
        pos: {
            "t_in": torch.linalg.qr(torch.randn(10, 6)).Q.T,
            "t_out": torch.linalg.qr(torch.randn(5, 3)).Q.T,
        }
        for pos in (1, 3)
    }
    corrections, diagnostics = complete_direct_p1_shared_correction(
        resized_base,
        resized_ft,
        target,
        target_before,
        baseline_delta,
        refs,
        transforms,
        layout,
        data,
        data,
        config=JointCorrectionConfig(
            enabled=True,
            source_weight=1.0,
            target_weight=0.1,
            ridge_relative=0.01,
        ),
        device="cpu",
    )
    assert len(diagnostics) == 2
    assert all(row["objective_after"] <= row["objective_before"] + 1e-7 for row in diagnostics)
    assert all(row["target_intercept"] is False for row in diagnostics)
    assert set(corrections) == {
        "visual.transformer.resblocks.1.mlp.c_proj.weight",
        "visual.transformer.resblocks.1.mlp.c_proj.bias",
        "visual.transformer.resblocks.3.mlp.c_proj.weight",
        "visual.transformer.resblocks.3.mlp.c_proj.bias",
    }
    for model, before in (
        (resized_base, source_base_before),
        (resized_ft, source_ft_before),
        (target, target_before),
    ):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_joint_solver_matches_explicit_affine_objective_and_returns_intercept():
    torch.manual_seed(23)
    source_h = torch.randn(7, 2)
    source_target = torch.randn(7, 2)
    target_h = torch.randn(7, 3)
    target_effect = torch.randn(7, 3)
    t_in = torch.randn(2, 3)
    t_out = torch.randn(2, 3)
    ws, wt, ridge = 0.7, 1.4, 0.03
    correction, diag = fit_joint_cproj_correction(
        source_h,
        source_target,
        target_h,
        target_effect,
        t_in,
        t_out,
        source_weight=ws,
        target_weight=wt,
        ridge_relative=ridge,
    )
    x = correction.double().T
    beta = diag["bias_correction"].double()
    at = target_h.double() @ t_in.double().T
    hs_aug = torch.cat((source_h.double(), torch.ones(7, 1)), dim=1)
    at_aug = torch.cat((at, torch.ones(7, 1)), dim=1)
    z = torch.cat((x, beta.unsqueeze(0)), dim=0)
    explicit = ws * ((hs_aug @ z - source_target.double()) ** 2).sum()
    explicit += wt * ((at_aug @ z @ t_out.double() - target_effect.double()) ** 2).sum()
    explicit += diag["ridge"] * (x * x).sum()
    assert abs(float(explicit) - diag["objective_after"]) < 1e-5
    assert diag["objective_after"] <= diag["objective_before"] + 1e-7


def test_joint_solver_fits_intercept_when_feature_banks_are_zero():
    source_h = torch.zeros(5, 2)
    target_h = torch.zeros(5, 3)
    source_target = torch.zeros(5, 2)
    target_effect = torch.tensor([[2.0, -1.0, 0.5]]).repeat(5, 1)
    t_in = torch.zeros(2, 3)
    t_out = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _correction, diag = fit_joint_cproj_correction(
        source_h,
        source_target,
        target_h,
        target_effect,
        t_in,
        t_out,
        source_weight=0.0,
        target_weight=1.0,
        ridge_relative=0.1,
    )
    # Only the target effect's reachable first two coordinates can be fitted;
    # the third coordinate is outside the frozen output map's range.
    torch.testing.assert_close(diag["bias_correction"], torch.tensor([2.0, -1.0]))
    assert diag["target_effect_residual_after"] < diag["target_effect_residual_before"]


def test_all_scope_uses_realized_ancestry_and_updates_every_position_in_order():
    torch.manual_seed(12)
    source, target = Model(3, 2).eval(), Model(5, 4).eval()
    source_ft = deepcopy(source)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.1)
        source_ft.visual.transformer.resblocks[1].mlp.c_proj.weight.sub_(0.08)
    data = loader()
    refs = capture_residual_references(
        source, source_ft, target, data, data, num_batches=3, seed=0, device="cpu", target_scope="all"
    )
    layout = {
        "final_blocks": tuple(
            {"position": pos, "source_orig_idx": pos // 2, "block_kind": "inserted" if pos % 2 else "original"}
            for pos in range(4)
        )
    }
    transforms = {}
    for pos in range(4):
        key = f"transformer.resblocks.{pos}.mlp.c_proj.weight"
        transforms[key] = SimpleNamespace(
            kind="weight", t_in=torch.linalg.qr(torch.randn(10, 6)).Q.T, t_out=torch.linalg.qr(torch.randn(5, 3)).Q.T
        )
    prepared = {"transforms_by_key": transforms}
    baseline_state = {k: v.clone() for k, v in target.state_dict().items()}
    baseline_delta = {k: torch.zeros_like(v) for k, v in baseline_state.items()}
    config = ResidualCompletionConfig(enabled=True, target_scope="all", ridge_relative=0.01)
    maps = projection_transforms(prepared, layout, target_scope="all")
    _src, completed, diagnostics = complete_residuals(
        target, baseline_state, baseline_delta, refs, maps, layout, data, config=config, device="cpu"
    )
    assert [row["position"] for row in diagnostics] == [0, 1, 2, 3]
    assert [row["block_kind"] for row in diagnostics] == ["original", "inserted", "original", "inserted"]
    assert all(row["scope"] == "all" for row in diagnostics)
    assert set(completed) >= {
        f"visual.transformer.resblocks.{pos}.mlp.c_proj.{kind}"
        for pos in range(4)
        for kind in ("weight", "bias")
    }
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, baseline_state[key], rtol=0, atol=0)


def test_all_scope_is_deterministic_and_rejects_missing_position_transforms():
    torch.manual_seed(12)
    source, target = Model(3, 2).eval(), Model(5, 4).eval()
    source_ft = deepcopy(source)
    data = loader()
    refs = capture_residual_references(
        source, source_ft, target, data, data, num_batches=3, seed=7, device="cpu", target_scope="all"
    )
    layout = {
        "final_blocks": tuple(
            {"position": pos, "source_orig_idx": pos // 2, "block_kind": "inserted" if pos % 2 else "original"}
            for pos in range(4)
        )
    }
    def prepared_for(positions):
        return {
            "transforms_by_key": {
                f"transformer.resblocks.{pos}.mlp.c_proj.weight": SimpleNamespace(
                    kind="weight", t_in=torch.eye(6, 10), t_out=torch.eye(3, 5)
                )
                for pos in positions
            }
        }
    with pytest.raises(ValueError, match="Missing fitted c_proj transport at original block position 0"):
        projection_transforms(prepared_for([1, 2, 3]), layout, target_scope="all")
    transforms = projection_transforms(prepared_for(range(4)), layout, target_scope="all")
    state = {k: v.clone() for k, v in target.state_dict().items()}
    delta = {k: torch.zeros_like(v) for k, v in state.items()}
    config = ResidualCompletionConfig(enabled=True, target_scope="all", ridge_relative=0.05)
    first = complete_residuals(target, state, delta, refs, transforms, layout, data, config=config, device="cpu")
    second = complete_residuals(target, state, delta, refs, transforms, layout, data, config=config, device="cpu")
    for a, b in zip(first[:2], second[:2], strict=True):
        assert set(a) == set(b)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert len(first[2]) == len(second[2])
    for row_a, row_b in zip(first[2], second[2], strict=True):
        assert set(row_a) == set(row_b)
        for key in row_a:
            if isinstance(row_a[key], torch.Tensor):
                torch.testing.assert_close(row_a[key], row_b[key], rtol=0, atol=0)
            else:
                assert row_a[key] == row_b[key]


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
