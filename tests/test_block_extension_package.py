"""The ``rebase.block_extension`` package is the home of the BRACE config/schedule layers;
``eval.block_extension`` must re-export the very same objects and the package must not
import any ``merge_and_rebase.eval`` module."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

_CONFIG_NAMES = (
    "TargetSharedCorrection",
    "BlockExtensionConfig",
    "block_extension_protocol",
    "resolve_block_extension_config",
    "calibration_dataset_spec",
    "select_loader",
    "_warn_unknown_block_extension_params",
    "_ANNOTATION_PARAMS",
    "_MISPLACED_TOP_LEVEL_KEYS",
    "_as_correction_scope",
    "_as_target_shared_correction",
    "_as_inserted_block_mode",
    "_as_transport_activation_mode",
    "_as_reference_capture",
    "_as_optional_int",
    "_as_optional_str",
    "_as_optional_calibration_dataset",
    "_as_optional_dict_float",
)

_COMPLETION_CONFIG_NAMES = (
    "ResidualCompletionConfig",
    "JointCorrectionConfig",
    "parse_residual_completion_config",
    "parse_joint_correction_config",
    "validate_residual_completion_depth_direction",
    "_DEFAULT_COMPONENTS",
    "_ALL_COMPONENT_NAMES",
)

_SCHEDULE_NAMES = (
    "spread_anchor_schedule",
    "balanced_collapse_spans",
    "disjoint_collapse_schedule",
    "plan_inserted_positions",
    "build_extension_layout",
    "build_reduction_layout",
)

_PACKAGE_MODULES = ("config", "completion_config", "schedules", "adapters", "core", "vision", "decoder")


@pytest.mark.parametrize("name", _CONFIG_NAMES)
def test_config_names_are_reexported_by_identity(name):
    from merge_and_rebase.eval import block_extension as old
    from merge_and_rebase.rebase.block_extension import config as new

    assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize("name", _COMPLETION_CONFIG_NAMES)
def test_completion_config_names_are_reexported_by_identity(name):
    from merge_and_rebase.eval import target_residual_completion as old
    from merge_and_rebase.rebase.block_extension import completion_config as new

    assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize("name", _SCHEDULE_NAMES)
def test_schedule_names_are_reexported_by_identity(name):
    schedules = pytest.importorskip("merge_and_rebase.rebase.block_extension.schedules")
    from merge_and_rebase.eval import block_extension as old

    assert getattr(old, name) is getattr(schedules, name)


_VISION_SHIM_NAMES = {
    "BlockExtender": "vision",
    "run_block_extension": "vision",
    "BlockExtenderCore": "core",
    "_deterministic_calibration_loader": "core",
    "_iter_with_progress": "core",
    "_InProjCapture": "adapters",
}

_DECODER_SHIM_NAMES = {
    "DecoderBlockExtender": "decoder",
    "run_block_extension_llm": "decoder",
    "_DECODER_COMPONENTS": "decoder",
    "BlockExtenderCore": "core",
    "BlockExtensionConfig": "config",
    "_deterministic_calibration_loader": "core",
    "_iter_with_progress": "core",
    "_run_decoder_forward": "adapters",
    "_get_layers": "adapters",
    "_get_final_norm": "adapters",
    "build_extension_layout": "schedules",
    "build_reduction_layout": "schedules",
    "decoder_collapse_schedule": "schedules",
    "decoder_locate_collapse_pos": "schedules",
}


@pytest.mark.parametrize("name", sorted(_VISION_SHIM_NAMES))
def test_vision_shim_names_are_reexported_by_identity(name):
    import importlib

    from merge_and_rebase.eval import block_extension as old

    new = importlib.import_module(f"merge_and_rebase.rebase.block_extension.{_VISION_SHIM_NAMES[name]}")
    assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize("name", sorted(_DECODER_SHIM_NAMES))
def test_decoder_shim_names_are_reexported_by_identity(name):
    import importlib

    from merge_and_rebase.eval import block_extension_llm as old

    new = importlib.import_module(f"merge_and_rebase.rebase.block_extension.{_DECODER_SHIM_NAMES[name]}")
    assert getattr(old, name) is getattr(new, name)


def test_decoder_adapter_uses_the_single_decoder_forward_helpers():
    """The adapter has no private copy of the forward/layer helpers; the eval shim re-exports the same ones."""
    from merge_and_rebase.eval import block_extension_llm as shim
    from merge_and_rebase.rebase.block_extension import adapters

    for name in ("_run_decoder_forward", "_get_layers", "_get_final_norm"):
        assert getattr(shim, name) is getattr(adapters, name)


def test_shims_define_no_extender_classes():
    from merge_and_rebase.eval import block_extension, block_extension_llm

    assert block_extension.BlockExtender.__module__ == "merge_and_rebase.rebase.block_extension.vision"
    assert block_extension_llm.DecoderBlockExtender.__module__ == "merge_and_rebase.rebase.block_extension.decoder"


@pytest.mark.parametrize("module", _PACKAGE_MODULES)
def test_package_module_imports_nothing_from_eval(module):
    if importlib.util.find_spec(f"merge_and_rebase.rebase.block_extension.{module}") is None:
        pytest.skip(f"{module} not created yet")
    code = (
        f"import sys, merge_and_rebase.rebase.block_extension.{module}; "
        "bad=[m for m in sys.modules if m.startswith('merge_and_rebase.eval')]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
