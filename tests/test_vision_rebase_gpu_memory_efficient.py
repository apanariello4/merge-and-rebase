"""Opt-in ``gpu_memory_efficient``: visual-only source fit copies and a CPU-parked source CLIP.

The flag may only move whole modules between devices; every fitted vector must equal the flag-off one exactly
(``torch.equal``). Toy CLIP-like models carry a text tower so the "full copy" vs "visual-only copy" difference is real.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.vision_rebase import _run_direct_residual_fit, method_stages
from merge_and_rebase.eval.vision_rebase.memory import (
    VisualOnlyModel,
    memory_efficient_enabled,
    park_on_cpu,
    source_fit_copy,
    temporarily_on,
    visual_only_view,
)
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Clip(torch.nn.Module):
    """open_clip-shaped: a visual tower plus a text tower that the fit never reads."""

    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)
        self.token_embedding = torch.nn.Embedding(16, width)
        self.text_proj = torch.nn.Linear(width, width)

    def encode_image(self, x):
        return self.visual(x)

    def encode_text(self, tokens):
        return self.text_proj(self.token_embedding(tokens).mean(dim=1))


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _fit(source_base, source_ft, config, *, target_depth=4):
    torch.manual_seed(5)
    target = _Clip(5, target_depth).eval()
    target_sd = {k: v.clone() for k, v in target.state_dict().items()}
    data = _loader(seed=13)
    pairing = DiscreteLayerPairing.compute(source_depth=len(source_base.visual.transformer.resblocks), target_depth=target_depth)
    delta, _t, diag, extra = _run_direct_residual_fit(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        target_base_sd=target_sd,
        source_loader=data,
        target_loader=data,
        pairing=pairing,
        config=config,
        device="cpu",
    )
    return delta, diag, extra


def _models():
    torch.manual_seed(11)
    base = _Clip(5, 2).eval()
    return base, _tuned_copy(base, seed=12)


def _assert_deep_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            _assert_deep_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_deep_equal(x, y)
    else:
        assert a == b


def _assert_same(a, b):
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


_CONFIGS = {
    "resident_default": dict(),
    "streaming": dict(activation_storage="streaming"),
    "streaming_chunk1": dict(activation_storage="streaming", streaming_position_chunk=1),
    "streaming_c_proj_ridge": dict(activation_storage="streaming", components=("mlp.c_proj",), ridge_relative=0.1),
}


@pytest.mark.parametrize("name", list(_CONFIGS))
def test_visual_only_source_copies_give_identical_fit(name):
    config = DirectResidualConfig(**{"num_batches": 3, "ridge_relative": 0.05, **_CONFIGS[name]})
    base, ft = _models()
    ref_delta, ref_diag, _ = _fit(deepcopy(base), deepcopy(ft), config)

    base_view = source_fit_copy(base, memory_efficient=True)
    ft_view = source_fit_copy(ft, memory_efficient=True)
    assert isinstance(base_view, VisualOnlyModel)
    assert not hasattr(base_view, "text_proj")
    delta, diag, _ = _fit(base_view, ft_view, config)

    _assert_same(ref_delta, delta)
    _assert_deep_equal(ref_diag, diag)


def test_visual_only_view_shares_the_visual_tower_and_training_mode():
    base, _ = _models()
    full = deepcopy(base)
    view = visual_only_view(full)
    assert view.visual is full.visual
    assert view.training == full.training
    assert [k for k in view.state_dict()] == [k for k in full.state_dict() if k.startswith("visual.")]


def test_source_fit_copy_default_is_the_plain_deepcopy_with_loaders_applied():
    base, ft = _models()
    copy_ = source_fit_copy(base, memory_efficient=False, loaders=(lambda m: m.load_state_dict(ft.state_dict()),))
    assert type(copy_) is _Clip and copy_ is not base
    _assert_same(dict(copy_.state_dict()), dict(ft.state_dict()))
    # the template itself is untouched
    assert not torch.equal(base.visual.transformer.resblocks[0].mlp.c_proj.weight, ft.visual.transformer.resblocks[0].mlp.c_proj.weight)


def test_flag_defaults_off():
    assert memory_efficient_enabled({}) is False
    assert memory_efficient_enabled({"gpu_memory_efficient": False}) is False
    assert memory_efficient_enabled({"gpu_memory_efficient": True}) is True


def test_temporarily_on_restores_placement_and_is_noop_when_disabled():
    model = _Clip(5, 2)
    park_on_cpu(model)
    with temporarily_on(model, "cpu", enabled=True):
        assert next(model.parameters()).device.type == "cpu"
    with temporarily_on(model, "meta", enabled=False):
        assert next(model.parameters()).device.type == "cpu"


# ---- DirectFitStage glue: which models reach the fit ----


def _stage_run(monkeypatch, tmp_path, *, flag, ariadne_cfg=None):
    base, ft = _models()
    ckpt = tmp_path / "ft.pt"
    torch.save({k: v for k, v in ft.state_dict().items() if k.startswith("visual.")}, ckpt)
    seen = {}

    def fake_fit(**kwargs):
        seen.update(kwargs)
        extra = {
            "realization_by_position": None,
            "task_vector_stats": None,
            "alignment_diagnostics": {},
            "calibration": None,
            "tv_scaling": None,
            "fidelity_holdout": None,
        }
        timing = {"alignment_calibration": {}, "correction_fit": {}, "cost_phases": {}}
        return {}, timing, [], extra

    monkeypatch.setattr(method_stages, "_run_direct_residual_fit", fake_fit)
    cfg_obj = ariadne_cfg or DirectResidualConfig(num_batches=3)
    env = SimpleNamespace(
        cfg={"gpu_memory_efficient": True} if flag else {},
        resolved=SimpleNamespace(ariadne_cfg=cfg_obj),
        source_depth=2,
        target_depth=4,
        clf_source=SimpleNamespace(model=base),
        clf_target=SimpleNamespace(model=_Clip(5, 4)),
        source_base_sd=dict(base.state_dict()),
        tuned_by_task={"T": str(ckpt)},
        target_base_sd={},
        device="cpu",
    )
    loaders = SimpleNamespace(train=_loader(), test=_loader(), val=_loader())
    task = SimpleNamespace(
        task="T",
        source_loaders=loaders,
        loaders=loaders,
        classnames=["a"],
        source_build_cfg_task=None,
        build_cfg_task=None,
        ctx=SimpleNamespace(source_text_features=None, target_text_features=None),
    )
    stage = method_stages.DirectFitStage(
        env, task_contexts={}, tasks=["T"], merge_weights=[1.0], calibration_ctx=None, calibration_meta={}
    )
    stage._transport = SimpleNamespace(run=lambda *_a, **_k: SimpleNamespace(transported_delta=None))
    stage.run(env, task, None)
    return base, ft, seen


def test_stage_default_hands_full_source_clips_to_the_fit(monkeypatch, tmp_path):
    base, ft, seen = _stage_run(monkeypatch, tmp_path, flag=False)
    for key in ("source_base_model", "source_ft_model"):
        assert type(seen[key]) is _Clip
        assert seen[key] is not base
    _assert_same(dict(seen["source_base_model"].state_dict()), dict(base.state_dict()))
    _assert_same(dict(seen["source_ft_model"].state_dict()), dict(ft.state_dict()))


def test_stage_flag_hands_visual_only_copies_with_identical_weights(monkeypatch, tmp_path):
    base, ft, seen = _stage_run(monkeypatch, tmp_path, flag=True)
    for key, ref in (("source_base_model", base), ("source_ft_model", ft)):
        model = seen[key]
        assert isinstance(model, VisualOnlyModel)
        for name, value in model.state_dict().items():
            assert torch.equal(value, ref.state_dict()[name]), name
    # the resident template is not aliased by a fit copy
    assert seen["source_base_model"].visual is not base.visual


@pytest.mark.parametrize(
    "kwargs", [dict(procrustes_source="gradient"), dict(residual_target="native_oracle", activation_storage="streaming")]
)
def test_stage_flag_rejects_modes_that_read_the_full_source_clip(monkeypatch, tmp_path, kwargs):
    with pytest.raises(NotImplementedError, match="gpu_memory_efficient"):
        _stage_run(monkeypatch, tmp_path, flag=True, ariadne_cfg=DirectResidualConfig(num_batches=3, **kwargs))
