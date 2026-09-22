from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch.nn as nn

from merge_and_rebase.eval import vision_rebase
from merge_and_rebase.eval.vision_rebase import _build_rebase_prepared


class _DeviceTrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_to_device = None

    def to(self, device):
        self.last_to_device = str(device)
        return super().to(device)


def _model(depth: int) -> nn.Module:
    model = _DeviceTrackingModel()
    model.visual = SimpleNamespace(transformer=SimpleNamespace(resblocks=nn.ModuleList()))
    model.visual.transformer.resblocks.extend(nn.Linear(1, 1) for _ in range(depth))
    return model


class _PreparedTheseusStub:
    def __init__(self) -> None:
        self.source_model = None
        self.target_model = None

    def prepare(self, **kwargs):
        self.source_model = kwargs["source_model"]
        self.target_model = kwargs["target_model"]
        return kwargs


class _PreparedBiCoStub(_PreparedTheseusStub):
    pass


@pytest.mark.parametrize("source_depth", [12, 24])
def test_theseus_uses_depth_matched_source_after_block_extension(source_depth: int) -> None:
    target_depth = 24 if source_depth == 12 else 12
    source_base_model_task = _model(target_depth)
    clf_source = SimpleNamespace(model=_model(source_depth))
    clf_target = SimpleNamespace(model=_model(target_depth))
    method = _PreparedTheseusStub()

    prepared = _build_rebase_prepared(
        method_name="theseus",
        method=method,
        method_params={},
        cfg={"seed": 33},
        device="cpu",
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        theseus_like_method=True,
        bico_mode=False,
        run_block_extension_prestep=True,
        clf_source=clf_source,
        clf_target=clf_target,
        classnames=[],
        loaders=SimpleNamespace(train=None),
        source_loaders=SimpleNamespace(train=None),
        build_cfg_task=None,
        source_build_cfg_task=None,
        task_source_base_sd={},
        target_base_sd={},
        task_delta={},
        source_base_model_task=source_base_model_task,
        transfusion_prepared=None,
    )

    assert prepared["source_model"] is method.source_model
    assert len(method.source_model.visual.transformer.resblocks) == target_depth
    assert len(method.target_model.visual.transformer.resblocks) == target_depth
    assert len(method.source_model.visual.transformer.resblocks) != source_depth
    assert method.source_model is not source_base_model_task
    assert prepared["seed"] == 33


def test_theseus_keeps_raw_source_fallback_without_block_extension() -> None:
    source_model = _model(12)
    target_model = _model(12)
    method = _PreparedTheseusStub()

    prepared = _build_rebase_prepared(
        method_name="theseus",
        method=method,
        method_params={"seed": 17},
        cfg={"seed": 33},
        device="cpu",
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        theseus_like_method=True,
        bico_mode=False,
        run_block_extension_prestep=False,
        clf_source=SimpleNamespace(model=source_model),
        clf_target=SimpleNamespace(model=target_model),
        classnames=[],
        loaders=SimpleNamespace(train=None),
        source_loaders=SimpleNamespace(train=None),
        build_cfg_task=None,
        source_build_cfg_task=None,
        task_source_base_sd={},
        target_base_sd={},
        task_delta={},
        source_base_model_task=None,
        transfusion_prepared=None,
    )

    assert len(method.source_model.visual.transformer.resblocks) == 12
    assert len(method.target_model.visual.transformer.resblocks) == 12
    assert prepared["seed"] == 17


def test_same_architecture_prepare_accepts_no_source_activation_plan() -> None:
    """The ordinary equal-depth path intentionally supplies no BRACE plan."""
    source_model = _model(12)
    target_model = _model(12)
    method = _PreparedTheseusStub()

    prepared = _build_rebase_prepared(
        method_name="theseus",
        method=method,
        method_params={},
        cfg={"seed": 33},
        device="cpu",
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        theseus_like_method=True,
        bico_mode=False,
        run_block_extension_prestep=False,
        clf_source=SimpleNamespace(model=source_model),
        clf_target=SimpleNamespace(model=target_model),
        classnames=[],
        loaders=SimpleNamespace(train=None),
        source_loaders=SimpleNamespace(train=None),
        build_cfg_task=None,
        source_build_cfg_task=None,
        task_source_base_sd={},
        target_base_sd={},
        task_delta={},
        source_base_model_task=None,
        transfusion_prepared=None,
        source_activation_plan=None,
    )

    assert prepared["source_activation_plan"] is None


def test_theseus_places_isolated_calibration_models_on_requested_device() -> None:
    method = _PreparedTheseusStub()

    _build_rebase_prepared(
        method_name="theseus",
        method=method,
        method_params={},
        cfg={"seed": 33},
        device="meta",
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        theseus_like_method=True,
        bico_mode=False,
        run_block_extension_prestep=True,
        clf_source=SimpleNamespace(model=_model(1)),
        clf_target=SimpleNamespace(model=_model(1)),
        classnames=[],
        loaders=SimpleNamespace(train=None),
        source_loaders=SimpleNamespace(train=None),
        build_cfg_task=None,
        source_build_cfg_task=None,
        task_source_base_sd={},
        target_base_sd={},
        task_delta={},
        source_base_model_task=_model(1),
        transfusion_prepared=None,
    )

    assert method.source_model.last_to_device == "meta"
    assert method.target_model.last_to_device == "meta"


@pytest.mark.parametrize("source_depth", [12, 24])
def test_bico_uses_an_isolated_depth_matched_source_after_block_extension(source_depth: int) -> None:
    target_depth = 24 if source_depth == 12 else 12
    corrected_source = _model(target_depth)
    method = _PreparedBiCoStub()

    prepared = _build_rebase_prepared(
        method_name="bico",
        method=method,
        method_params={},
        cfg={"seed": 33},
        device="cpu",
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        theseus_like_method=False,
        bico_mode=True,
        run_block_extension_prestep=True,
        clf_source=SimpleNamespace(model=_model(source_depth), normalize=True),
        clf_target=SimpleNamespace(model=_model(target_depth), normalize=True),
        classnames=[],
        loaders=SimpleNamespace(train=None),
        source_loaders=SimpleNamespace(train=None),
        build_cfg_task=None,
        source_build_cfg_task=None,
        task_source_base_sd={},
        target_base_sd={},
        task_delta={},
        source_base_model_task=corrected_source,
        transfusion_prepared=None,
    )

    assert prepared["source_model"] is method.source_model
    assert len(method.source_model.visual.transformer.resblocks) == target_depth
    assert len(method.target_model.visual.transformer.resblocks) == target_depth
    assert method.source_model is not corrected_source
    assert prepared["seed"] == 33


def test_main_initializes_brace_diagnostic_collectors_before_the_task_loop() -> None:
    tree = ast.parse(inspect.getsource(vision_rebase.main))
    main_fn = tree.body[0]
    assert isinstance(main_fn, ast.FunctionDef)
    try_block = next(node for node in main_fn.body if isinstance(node, ast.Try))
    task_loop_index = next(
        index
        for index, node in enumerate(try_block.body)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "task"
    )
    initialized = {
        node.target.id
        for node in try_block.body[:task_loop_index]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert {
        "source_lmc_rows",
        "cross_task_lmc_rows",
        "all_task_lmc_rows",
        "corrected_ft_states",
        "corrected_ft_templates",
        "independent_base_by_task",
        "independent_ft_by_task",
    } <= initialized
