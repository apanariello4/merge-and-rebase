"""Tests for method="direct_residual"'s top-level dispatch in vision_rebase.py.

Direct Residual bypasses ARIADNE's config gates entirely: it must never call
`resolve_block_extension_config`/`get_method` (it is not in the rebase method
registry -- see `rebase/registry.py` -- and its own config schema has no
overlap with `resolve_block_extension_config`'s ARIADNE-shaped one). These
tests confirm that structural guarantee by monkeypatching those two functions
to raise, then confirming a `method="direct_residual"` run reaches a *later*
config-validation error instead of the monkeypatched one.

`vision_rebase.main()` cannot be run end-to-end offline (it downloads/builds
real OpenCLIP models), so every test here drives `main()` only up to the
first config-validation failure that precedes model construction -- the same
"fail fast" checks the plan requires -- following the "no network past this
point" constraint documented in CLAUDE.md ("Offline compute"). This mirrors
`tests/test_direct_target_p1.py` and `tests/test_direct_residual_fit.py`'s
synthetic-model fixture convention for the one test that does need to run
Direct Residual's actual capture/fit pipeline (`_run_direct_residual_fit`).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval import vision_rebase
from merge_and_rebase.eval.direct_residual import DirectResidualConfig
from merge_and_rebase.eval.vision_rebase import _run_direct_residual_fit
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing


def _run_main_with_cfg(monkeypatch, tmp_path, cfg: dict) -> Exception:
    """Invoke vision_rebase.main() with `cfg` written to a temp config file.

    Returns the raised exception (every cfg used here is expected to fail
    before any network/model-download step).
    """
    cfg = dict(cfg)
    cfg.setdefault("logging", {"local_log_dir": str(tmp_path / "logs")})
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr("sys.argv", ["vision_rebase", "--config", str(config_path)])
    with pytest.raises(Exception) as excinfo:
        vision_rebase.main()
    return excinfo.value


@pytest.mark.parametrize(
    "endpoint_construction",
    ["sequential_source_endpoints", "sequential_delta_on_synthesized_base"],
)
def test_saved_sequential_vector_loader_checks_provenance(tmp_path, endpoint_construction):
    key = "visual.transformer.resblocks.0.mlp.c_proj.weight"
    base = {key: torch.zeros(2, 2)}
    vector = {key: torch.ones(2, 2)}
    config = DirectResidualConfig(
        endpoint_construction=endpoint_construction, components=("mlp.c_proj",),
        activation_storage="resident",
    )
    path = tmp_path / "DTD_direct_residual_transported_native.pt"
    torch.save(vector, path)
    metadata = {
        "task": "DTD",
        "endpoint_construction": config.endpoint_construction,
        "target_base_sha256": vision_rebase._state_dict_sha256(base),
        "vector_sha256": vision_rebase._state_dict_sha256(vector),
        "calibration_seed": config.seed,
        "num_batches": config.num_batches,
        "direct_residual_config": json.loads(json.dumps(asdict(config))),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    loaded, _ = vision_rebase._load_saved_sequential_tv(tmp_path, "DTD", base, config)
    assert torch.equal(loaded[key], vector[key])
    with pytest.raises(ValueError, match="target_base_sha256"):
        vision_rebase._load_saved_sequential_tv(tmp_path, "DTD", {key: torch.ones(2, 2)}, config)


def test_direct_residual_never_calls_ariadne_config_gates(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise AssertionError("resolve_block_extension_config must not be called for method='direct_residual'")

    def _boom_get_method(*args, **kwargs):
        raise AssertionError("get_method must not be called for method='direct_residual'")

    monkeypatch.setattr(vision_rebase, "resolve_block_extension_config", _boom)
    monkeypatch.setattr(vision_rebase, "get_method", _boom_get_method)

    exc = _run_main_with_cfg(monkeypatch, tmp_path, {"method": "direct_residual"})

    # Neither monkeypatched guard fired; execution instead reached the next
    # real validation error further down main() (missing tuned checkpoints),
    # proving direct_residual's dispatch never touches either function.
    assert "resolve_block_extension_config must not be called" not in str(exc)
    assert "get_method must not be called" not in str(exc)
    assert "tuned checkpoints" in str(exc)


def test_invalid_ariadne_only_option_is_inert_for_direct_residual(monkeypatch, tmp_path):
    """An ARIADNE-only config field that would normally be rejected is simply
    never read for method="direct_residual", since resolve_block_extension_config
    is never invoked on cfg at all -- not even with a filtered/sanitized cfg.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("resolve_block_extension_config must not be called for method='direct_residual'")

    monkeypatch.setattr(vision_rebase, "resolve_block_extension_config", _boom)

    exc = _run_main_with_cfg(
        monkeypatch,
        tmp_path,
        {
            "method": "direct_residual",
            # Nonsense ARIADNE-only knob: would raise inside
            # resolve_block_extension_config if it were ever parsed.
            "block_extension_params": {"this_field_does_not_exist": 123},
        },
    )
    assert "resolve_block_extension_config must not be called" not in str(exc)
    assert "tuned checkpoints" in str(exc)


def test_direct_residual_method_object_has_no_registry_entry():
    from merge_and_rebase.rebase.registry import list_methods

    assert "direct_residual" not in list_methods()


# ---- _run_direct_residual_fit: the real capture -> desired-effect -> fit ->
# scale pipeline used by both vision_rebase.py's per-task and
# merge_in_source_then_fit dispatch branches. Fixture duplicated from
# tests/test_direct_residual_fit.py per this suite's no-cross-test-import
# convention. ----


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


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    from copy import deepcopy

    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def test_run_direct_residual_fit_returns_scaled_delta_and_timing_brackets():
    torch.manual_seed(11)
    source_base = _Model(5, 2).eval()
    source_ft = _tuned_copy(source_base, seed=12)
    target_base = _Model(5, 4).eval()
    data = _loader(seed=13)
    pairing = DiscreteLayerPairing.compute(source_depth=2, target_depth=4)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, strength=1.0)

    delta, timing, diagnostics, extra = _run_direct_residual_fit(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=target_base_sd,
        source_loader=data,
        target_loader=data,
        pairing=pairing,
        config=config,
        device="cpu",
    )

    assert delta  # nonzero strength -> a nonempty correction dict
    assert all(key.endswith(("c_proj.weight", "c_proj.bias", "out_proj.weight", "out_proj.bias")) for key in delta)
    # realization_diagnostics defaults to False: both extras absent.
    assert extra["realization_by_position"] is None
    assert extra["task_vector_stats"] is None
    # alignment_diagnostics is always populated, regardless of
    # realization_diagnostics or residual_target -- analysis-only, one row
    # per target position; see compute_alignment_diagnostics. Computed by a
    # separate, untimed call, outside both timing/peak-memory brackets.
    assert set(extra["alignment_diagnostics"]) == set(range(pairing.target_depth))
    for row in extra["alignment_diagnostics"].values():
        assert set(row) == {
            "procrustes_error_norm",
            "procrustes_relative_error",
            "delta_target_norm",
            "endpoint_minus_delta_over_delta",
            "procrustes_error_in_range_norm",
            "procrustes_error_out_of_range_norm",
            "mean_offset_norm",
            "source_dim",
            "target_dim",
        }
    assert extra["tv_scaling"] is None
    assert set(extra) == {
        "realization_by_position", "task_vector_stats", "alignment_diagnostics", "calibration", "tv_scaling",
        "fidelity_holdout",
    }
    assert set(timing) == {"alignment_calibration", "correction_fit", "cost_phases"}


    cost = timing["cost_phases"]
    assert set(cost["phases"]) == {"activation_collection", "transformation", "transport"}
    assert cost["phases"]["activation_collection"]["seconds"] > 0.0
    assert cost["phases"]["transformation"]["seconds"] > 0.0
    assert cost["phases"]["transport"]["segments"] == 1
    # alignment diagnostics run, and are excluded from every phase
    assert cost["excluded_seconds"] > 0.0
    for bracket in ("alignment_calibration", "correction_fit"):
        seconds_key = f"{bracket}_seconds"
        memory_key = f"{bracket}_peak_memory_bytes"
        rss_key = f"{bracket}_peak_host_rss_bytes"
        assert set(timing[bracket]) == {seconds_key, memory_key, rss_key}
        assert isinstance(timing[bracket][seconds_key], float)
        assert timing[bracket][seconds_key] >= 0.0
        assert isinstance(timing[bracket][memory_key], float)
        assert isinstance(timing[bracket][rss_key], float)
        assert timing[bracket][rss_key] > 0.0
    assert isinstance(diagnostics, list) and diagnostics
    # target_base is restored to its pristine state by fit_direct_residual's
    # try/finally (see direct_residual.py); the model handed back must be
    # bit-identical to the state before the fit ran.
    for key, value in target_base_sd.items():
        assert torch.equal(dict(target_base.state_dict())[key], value)


def test_run_sequential_endpoint_pipeline_returns_new_vector():
    torch.manual_seed(31)
    source_base = _Model(5, 2).eval()
    source_ft = _tuned_copy(source_base, seed=32)
    target = _Model(5, 4).eval()
    data = _loader(seed=33)
    base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    config = DirectResidualConfig(
        num_batches=3, endpoint_construction="sequential_source_endpoints",
        components=("mlp.c_proj",), activation_storage="resident",
    )
    vector, _timing, rows, extra = _run_direct_residual_fit(
        source_base_model=source_base, source_ft_model=source_ft,
        target_model=target, target_base_sd=base_sd, source_loader=data,
        target_loader=data, pairing=DiscreteLayerPairing.compute(2, 4),
        config=config, device="cpu",
    )
    assert vector and all(".mlp.c_proj." in key for key in vector)
    assert {row["endpoint_stage"] for row in rows} == {"pretrained", "finetuned"}
    assert extra["sequential_endpoints"]["pretrained_correction_norm"] > 0
    for key, value in base_sd.items():
        assert torch.equal(target.state_dict()[key], value)


def test_run_synthesized_base_delta_dispatch_preserves_zero_update():
    torch.manual_seed(41)
    source_base = _Model(5, 2).eval()
    source_ft = _Model(5, 2).eval()
    source_ft.load_state_dict(source_base.state_dict(), strict=True)
    target = _Model(5, 4).eval()
    data = _loader(seed=42)
    base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    config = DirectResidualConfig(
        num_batches=3,
        endpoint_construction="sequential_delta_on_synthesized_base",
        components=("mlp.c_proj",),
        activation_storage="resident",
    )
    vector, _timing, rows, extra = _run_direct_residual_fit(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        target_base_sd=base_sd,
        source_loader=data,
        target_loader=data,
        pairing=DiscreteLayerPairing.compute(2, 4),
        config=config,
        device="cpu",
    )
    assert vector and all(torch.count_nonzero(value) == 0 for value in vector.values())
    assert extra["sequential_endpoints"]["pretrained_correction_norm"] > 0
    assert {row["endpoint_stage"] for row in rows} == {"pretrained", "finetuned"}
    for key, value in base_sd.items():
        assert torch.equal(target.state_dict()[key], value)


def test_run_direct_residual_fit_strength_zero_is_native_target_base_control():
    torch.manual_seed(21)
    source_base = _Model(5, 2).eval()
    source_ft = _tuned_copy(source_base, seed=22)
    target_base = _Model(5, 2).eval()
    data = _loader(seed=23)
    pairing = DiscreteLayerPairing.compute(source_depth=2, target_depth=2)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, strength=0.0)

    delta, _timing, _diagnostics, extra = _run_direct_residual_fit(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=target_base_sd,
        source_loader=data,
        target_loader=data,
        pairing=pairing,
        config=config,
        device="cpu",
    )

    assert delta == {}
    assert extra["realization_by_position"] is None
    assert extra["task_vector_stats"] is None
    assert set(extra["alignment_diagnostics"]) == set(range(pairing.target_depth))
    assert extra["tv_scaling"] is None
    assert set(extra) == {
        "realization_by_position", "task_vector_stats", "alignment_diagnostics", "calibration", "tv_scaling",
        "fidelity_holdout",
    }


def test_run_direct_residual_fit_realization_diagnostics_populates_extra():
    """Wiring check: realization_diagnostics=True reaches _run_direct_residual_fit's
    returned ``extra`` with both new artifacts populated per position, and the
    live target model is left bit-identical to its entry state afterward."""
    torch.manual_seed(31)
    source_base = _Model(5, 2).eval()
    source_ft = _tuned_copy(source_base, seed=32)
    target_base = _Model(5, 4).eval()
    data = _loader(seed=33)
    pairing = DiscreteLayerPairing.compute(source_depth=2, target_depth=4)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    config = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, strength=1.0, realization_diagnostics=True,
    )
    before = {k: v.clone() for k, v in target_base.state_dict().items()}

    _delta, _timing, _diagnostics, extra = _run_direct_residual_fit(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=target_base_sd,
        source_loader=data,
        target_loader=data,
        pairing=pairing,
        config=config,
        device="cpu",
    )

    realization = extra["realization_by_position"]
    stats = extra["task_vector_stats"]
    assert set(realization) == set(range(pairing.target_depth))
    for row in realization.values():
        assert row["component_interaction_error"] is not None  # two families requested
        for key in ("block_realized_target_error", "joint_delta_norm_over_desired"):
            assert torch.isfinite(torch.tensor(float(row[key])))
    assert set(stats) == {
        "n_modified_tensors", "n_modified_parameters", "tau_norm",
        "tau_norm_over_touched_base", "tau_norm_over_all_base", "tau_sha256",
    }
    assert stats["n_modified_parameters"] > 0
    for key, value in before.items():
        assert torch.equal(dict(target_base.state_dict())[key], value)
