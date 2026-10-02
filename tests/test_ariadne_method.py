"""AriadneRebase: registry/alias, capabilities, label, preset and hash-level equivalence.

"direct_residual" is the historical name of Ariadne: it must resolve to the very same
registered method object, take the same capability/label/dispatch decisions, and the
class-based ``prepare`` must reproduce the byte-pinned outputs of the legacy
``vision_rebase._run_direct_residual_fit`` wrapper (golden suite cases).
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields

import pytest
import torch

from merge_and_rebase.eval import vision_rebase
from merge_and_rebase.rebase import capabilities, run_config
from merge_and_rebase.rebase.methods.ariadne import (
    AriadneRebase,
    DirectResidualConfig,
    parse_direct_residual_config,
    resolve_direct_residual_preset,
)
from merge_and_rebase.rebase.methods.ariadne.method import AriadnePrepared
from merge_and_rebase.rebase.model_families.base import ModelFamilyMetadata
from merge_and_rebase.rebase.registry import canonical_method_name, get_method, list_methods
from merge_and_rebase.rebase.runtime import format_rebase_method_label
from tests.golden._hashing import deterministic_cpu, hash_json, hash_tensor_dict
from tests.golden.test_release_golden_hashes import _MAIN, DEPTHS, EXPECTED, _dr_setup
from tests.test_vision_rebase_direct_residual_dispatch import _run_main_with_cfg

# ---- registry / alias --------------------------------------------------------------


def test_registry_alias_is_the_same_object():
    method = get_method("ariadne")
    assert isinstance(method, AriadneRebase)
    assert get_method("direct_residual") is method
    assert method.name == "ariadne"
    assert canonical_method_name("direct_residual") == "ariadne"
    assert canonical_method_name("ariadne") == "ariadne"
    assert canonical_method_name("theseus") == "theseus"
    names = list_methods()
    assert "ariadne" in names and "direct_residual" in names
    # unchanged for the other methods
    for other in ("bico", "gradfix", "identity", "theseus", "theseus_gqa", "transfusion"):
        assert other in names


def test_unknown_method_still_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown rebase method"):
        get_method("not_a_method")


def test_transport_is_not_available():
    for name in ("ariadne", "direct_residual"):
        with pytest.raises(NotImplementedError, match="prepare"):
            get_method(name).transport(source_base={}, target_base={}, delta={})


# ---- capabilities / label ----------------------------------------------------------


def test_capabilities_same_for_both_names():
    for name in ("ariadne", "direct_residual"):
        assert capabilities.supports_cross_size(name) is True
        assert capabilities.is_text_supported(name) is False
    # existing entries are unchanged
    assert capabilities.supports_cross_size("theseus") and capabilities.is_text_supported("theseus")
    assert not capabilities.supports_cross_size("identity")
    assert not capabilities.is_text_supported("transfusion")
    with pytest.raises(ValueError, match="Unknown rebase method"):
        capabilities.is_text_supported("nope")


def _meta():
    return ModelFamilyMetadata(
        family="qwen2",
        num_hidden_layers=2,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def test_check_pair_rejects_text_with_ariadne_message():
    messages = []
    for name in ("ariadne", "direct_residual"):
        with pytest.raises(ValueError, match="Ariadne for text/decoder models is not available yet") as exc:
            capabilities.check_pair(name, _meta(), _meta())
        messages.append(str(exc.value).replace(f"'{name}'", "'<m>'"))
    assert messages[0] == messages[1]
    # transfusion keeps its generic message
    with pytest.raises(ValueError, match="not available for text/decoder rebasing in v1"):
        capabilities.check_pair("transfusion", _meta(), _meta())


def test_label_is_ariadne_for_both_names():
    assert format_rebase_method_label("ariadne", {}) == "Ariadne"
    assert format_rebase_method_label("direct_residual", {"num_batches": 3}) == "Ariadne"
    assert format_rebase_method_label("identity", {}) == "identity"


# ---- preset ------------------------------------------------------------------------

# Resolved config recorded by the published D-arm run
# results/rebase/dr_feature_sweep_20260923/ref_grid_D/dr_shrink_seed33_nb2_eb_bb_D_ref
_PUBLISHED_MAIN_ARM = {
    "activation_storage": "streaming",
    "backfit_max_iters": 20,
    "backfit_tol": 0.0001,
    "block_split": "none",
    "cascade_order": "independent",
    "component_target": "block_boundary",
    "components": ["mlp.c_proj"],
    "exact_form": True,
    "merge_mode": "per_task_then_merge",
    "missing_bias": "error",
    "num_batches": 2,
    "procrustes_source": "activation",
    "realization_diagnostics": False,
    "residual_target": "transported_delta",
    "ridge_estimator": "empirical_bayes",
    "ridge_relative": 0.01,
    "seed": 33,
    "streaming_position_chunk": None,
    "strength": 1.0,
    "tv_scaling": "none",
    "tv_scaling_iters": 3,
}


def test_preset_changes_exactly_three_fields():
    cfg = parse_direct_residual_config({"preset": "ariadne"})
    default = DirectResidualConfig()
    changed = {f.name for f in fields(cfg) if getattr(cfg, f.name) != getattr(default, f.name)}
    assert changed == {"components", "activation_storage", "ridge_estimator"}
    assert cfg.components == ("mlp.c_proj",)
    assert cfg.activation_storage == "streaming"
    assert cfg.ridge_estimator == "empirical_bayes"
    assert cfg == DirectResidualConfig(
        components=("mlp.c_proj",), activation_storage="streaming", ridge_estimator="empirical_bayes"
    )


def test_preset_explicit_keys_override_and_budgets_are_not_preset():
    cfg = parse_direct_residual_config({"preset": "ariadne", "ridge_estimator": "fixed_relative", "num_batches": 4})
    assert cfg.ridge_estimator == "fixed_relative"
    assert cfg.num_batches == 4 and cfg.seed == DirectResidualConfig().seed
    assert cfg.activation_storage == "streaming"
    cfg = parse_direct_residual_config({"preset": "ariadne", "components": ["attn.out_proj", "mlp.c_proj"]})
    assert cfg.components == ("attn.out_proj", "mlp.c_proj")


def test_unknown_preset_lists_valid_presets():
    with pytest.raises(ValueError, match=r"valid presets: \['ariadne'\]"):
        parse_direct_residual_config({"preset": "nope"})
    with pytest.raises(ValueError, match="valid presets"):
        resolve_direct_residual_preset({"preset": 3})
    assert resolve_direct_residual_preset({"preset": "ariadne"}) == "ariadne"
    assert resolve_direct_residual_preset({"ridge_relative": 0.1}) is None
    assert resolve_direct_residual_preset(None) is None
    # unknown keys are still refused next to a preset
    with pytest.raises(ValueError, match="unknown direct_residual fields"):
        parse_direct_residual_config({"preset": "ariadne", "bogus": 1})


def test_preset_has_no_config_field_and_defaults_unchanged():
    assert "preset" not in {f.name for f in fields(DirectResidualConfig)}
    assert "preset" not in asdict(parse_direct_residual_config({"preset": "ariadne"}))
    default = DirectResidualConfig()
    assert (default.components, default.activation_storage, default.ridge_estimator) == (
        ("attn.out_proj", "mlp.c_proj"),
        "resident",
        "fixed_relative",
    )


def test_preset_reproduces_published_main_arm():
    cfg = parse_direct_residual_config({"preset": "ariadne", "num_batches": 2, "seed": 33})
    resolved = json.loads(json.dumps(asdict(cfg)))
    for key, value in _PUBLISHED_MAIN_ARM.items():
        assert resolved[key] == value, key
    # fields added after that run keep the dataclass defaults
    defaults = json.loads(json.dumps(asdict(DirectResidualConfig())))
    for key in set(resolved) - set(_PUBLISHED_MAIN_ARM):
        assert resolved[key] == defaults[key], key


# ---- vision_rebase dispatch -------------------------------------------------------


@pytest.mark.parametrize("name", ["direct_residual", "ariadne"])
def test_dispatch_is_identical_for_both_names(monkeypatch, tmp_path, name):
    def _boom(*args, **kwargs):
        raise AssertionError("must not be called for Ariadne")

    monkeypatch.setattr(run_config, "resolve_block_extension_config", _boom)
    monkeypatch.setattr(run_config, "get_method", _boom)
    exc = _run_main_with_cfg(
        monkeypatch, tmp_path, {"method": name, "block_extension_params": {"this_field_does_not_exist": 1}}
    )
    assert "must not be called" not in str(exc)
    assert "tuned checkpoints" in str(exc)


@pytest.mark.parametrize("name", ["direct_residual", "ariadne"])
def test_params_alias_and_conflict(monkeypatch, tmp_path, name):
    exc = _run_main_with_cfg(
        monkeypatch,
        tmp_path,
        {"method": name, "ariadne_params": {"preset": "ariadne"}, "direct_residual_params": {"preset": "ariadne"}},
    )
    assert "alias 'ariadne_params'" in str(exc)
    # the alias key is parsed like direct_residual_params (a bad preset is reported from it)
    exc = _run_main_with_cfg(monkeypatch, tmp_path, {"method": name, "ariadne_params": {"preset": "nope"}})
    assert "valid presets" in str(exc)
    exc = _run_main_with_cfg(monkeypatch, tmp_path, {"method": name, "direct_residual_params": {"preset": "nope"}})
    assert "valid presets" in str(exc)
    # a valid ariadne_params proceeds to the next real validation
    exc = _run_main_with_cfg(monkeypatch, tmp_path, {"method": name, "ariadne_params": {"preset": "ariadne"}})
    assert "tuned checkpoints" in str(exc)


# ---- hash-level equivalence with the pinned legacy wrapper ------------------------


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize(
    ("case", "overrides"),
    [
        ("dr_orchestration_main_streaming_eb", _MAIN),
        ("dr_orchestration_od_resident_fixed_relative", {}),
    ],
)
def test_prepare_reproduces_pinned_orchestration_hashes(case, overrides, direction):
    with deterministic_cpu(seed=0):
        source_base, source_ft, target_base, (source_loader, target_loader), pairing, target_base_sd = _dr_setup(
            *DEPTHS[direction]
        )
        config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, **overrides)
        prepared = get_method("ariadne").prepare(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target_base,
            target_base_sd=target_base_sd,
            source_loader=source_loader,
            target_loader=target_loader,
            pairing=pairing,
            config=config,
            device="cpu",
            family_adapter=None,
        )
        assert isinstance(prepared, AriadnePrepared)
        assert set(prepared.timing) == {"alignment_calibration", "correction_fit", "cost_phases"}
        assert prepared.task_vector
        assert hash_tensor_dict(prepared.task_vector) == EXPECTED[f"{case}:{direction}:task_vector"]
        summary = hash_json({"diagnostics": prepared.diagnostics, "extra": prepared.extra})
        assert summary == EXPECTED[f"{case}:{direction}:summary"]
        # apply(): default is exactly what prepare produced; strength=1 rescale is the unit vector
        assert get_method("direct_residual").apply(prepared) is prepared.task_vector
        unit = get_method("ariadne").apply(prepared, strength=1.0)
        assert hash_tensor_dict(unit) == EXPECTED[f"{case}:{direction}:task_vector"]
        assert get_method("ariadne").apply(prepared, strength=0.0) == {}
        half = get_method("ariadne").apply(prepared, strength=0.5)
        assert all(torch.equal(half[k], 0.5 * prepared.unit_task_vector[k]) for k in half)
        with pytest.raises(ValueError, match="does not accept an input delta"):
            get_method("ariadne").apply(prepared, delta={})


def test_legacy_wrappers_delegate_to_prepare():
    with deterministic_cpu(seed=0):
        source_base, source_ft, target_base, (source_loader, target_loader), pairing, target_base_sd = _dr_setup(
            *DEPTHS["extend"]
        )
        config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, **_MAIN)
        delta, timing, diagnostics, extra = vision_rebase._run_direct_residual_fit(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target_base,
            target_base_sd=target_base_sd,
            source_loader=source_loader,
            target_loader=target_loader,
            pairing=pairing,
            config=config,
            device="cpu",
        )
        assert set(timing) == {"alignment_calibration", "correction_fit", "cost_phases"}
        assert hash_tensor_dict(delta) == EXPECTED["dr_orchestration_main_streaming_eb:extend:task_vector"]
        assert (
            hash_json({"diagnostics": diagnostics, "extra": extra})
            == EXPECTED["dr_orchestration_main_streaming_eb:extend:summary"]
        )
