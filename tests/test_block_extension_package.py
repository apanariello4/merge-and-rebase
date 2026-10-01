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

_PACKAGE_MODULES = ("config", "completion_config", "schedules", "adapters", "core")


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
