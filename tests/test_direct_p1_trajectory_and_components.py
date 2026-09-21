"""Tests for the two ``direct_target`` Proposal-1 extensions.

Both are strictly opt-in and are evaluated as separate campaign cells:

* ``target_trajectory="interpolate"`` stops asking every realized target
  position for its ancestor's *whole* effect.  Where one source block is
  realized as several target blocks, the earlier members are asked only for
  the fraction of the step from source boundary ``i-1`` to ``i`` that their
  realized depth corresponds to.
* ``components=["attn.out_proj", "mlp.c_proj"]`` corrects both of the block's
  additive writes into the residual stream instead of only the MLP's.

The load-bearing claims pinned here are the null controls (defaults must be
byte-identical to the historical path), the fact that the two components are
*cascaded and re-measured* rather than solved as one linear system, and that
the interpolated trajectory agrees exactly with the stepped one at the integer
coordinates where they are the same target by construction.

Hash-level comparison is used for the null controls, following
``tests/test_direct_target_p1.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from merge_and_rebase.eval.target_informed_runtime import (
    _materialize_all_scope_references,
    capture_tokens,
    complete_residuals_direct,
    realized_source_coordinates,
)
from merge_and_rebase.eval.target_residual_completion import (
    ResidualCompletionConfig,
    order_components,
    parse_residual_completion_config,
)
from merge_and_rebase.eval.vision_rebase import (
    _maybe_capture_target_residual_references,
    _maybe_complete_target_residual_task_vector,
    _state_dict_sha256,
)

# The fixtures below are self-contained: the suite has no cross-test imports
# and no tests package.


class _Attention(torch.nn.Module):
    """Attention whose ``out_proj`` is a genuinely called submodule.

    Stands in for the patched OpenCLIP attention. The stock
    ``nn.MultiheadAttention`` path -- where torch applies ``out_proj``
    functionally and a hook on it never fires -- is covered separately by
    ``test_stock_multihead_attention_rows_are_recovered_exactly``.
    """

    def __init__(self, width):
        super().__init__()
        self.in_proj = torch.nn.Linear(width, width)
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(torch.tanh(self.in_proj(x)))


class _Block(torch.nn.Module):
    """A faithful residual block: the MLP's input depends on the attention write.

    This ordering is what makes the two components non-separable -- mounting a
    correction on ``out_proj`` moves ``h_mlp`` through ``ln_2`` and the
    nonlinearity -- so the fixture has to reproduce it or the cascade tests
    would pass vacuously.
    """

    def __init__(self, width):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = _Attention(width)
        self.ls_1 = torch.nn.Identity()
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(width, width * 2)), ("gelu", torch.nn.GELU()),
            ("c_proj", torch.nn.Linear(width * 2, width)),
        ]))
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        x = x + self.ls_1(self.attn(self.ln_1(x)))
        return x + self.ls_2(self.mlp(self.ln_2(x)))


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


def _loader():
    return DataLoader(TensorDataset(torch.randn(6, 5, 4), torch.arange(6)), batch_size=2, shuffle=False)


def _fixture():
    """Source depth 2 (un-resized), target depth 4 (already doubled)."""
    torch.manual_seed(12)
    source = _Model(3, 2).eval()
    target = _Model(5, 4).eval()
    source_ft = deepcopy(source)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.2)
        source_ft.visual.transformer.resblocks[1].mlp.c_proj.weight.sub_(0.15)
        source_ft.visual.transformer.resblocks[0].attn.out_proj.weight.add_(0.1)
    data = _loader()
    # The realized chain [orig_0, ins_0, orig_1, ins_1]: two target positions
    # per source block, which is the layout the campaign actually runs.
    layout = {
        "final_blocks": [
            {"position": 0, "source_orig_idx": 0, "block_kind": "original"},
            {"position": 1, "source_orig_idx": 0, "block_kind": "inserted"},
            {"position": 2, "source_orig_idx": 1, "block_kind": "original"},
            {"position": 3, "source_orig_idx": 1, "block_kind": "inserted"},
        ],
        "inserted_blocks": [
            {"position": 1, "source_orig_idx": 0},
            {"position": 3, "source_orig_idx": 1},
        ],
    }
    target_base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    return source, source_ft, target, data, layout, target_base_sd


def _config(**overrides):
    params = {
        "enabled": True,
        "mode": "direct_target",
        "target_scope": "all",
        "ridge_relative": 0.05,
        "num_batches": 3,
        "strength": 1.0,
    }
    params.update(overrides)
    return ResidualCompletionConfig(**params)


def _references(source, source_ft, target, data, config, seed=0):
    return _maybe_capture_target_residual_references(
        config=config, source_base_model=source, source_ft_model=source_ft, target_model=target,
        source_loader=data, target_loader=data, seed=seed, device="cpu",
    )


def _run(config, seed=0):
    source, source_ft, target, data, layout, target_base_sd = _fixture()
    references = _references(source, source_ft, target, data, config, seed=seed)
    corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    return corrections, diagnostics, references, target, target_base_sd, data


# --------------------------------------------------------------------------
# 1. Null controls -- the most important tests
# --------------------------------------------------------------------------


def test_both_new_fields_default_to_the_historical_behaviour():
    config = ResidualCompletionConfig()
    assert config.target_trajectory == "step"
    assert config.components == ("mlp.c_proj",)
    parsed = parse_residual_completion_config({"enabled": True})
    assert parsed.target_trajectory == "step"
    assert parsed.components == ("mlp.c_proj",)


def test_explicit_step_trajectory_is_a_bitwise_no_op():
    omitted, _, *_ = _run(_config())
    explicit, _, *_ = _run(_config(target_trajectory="step"))
    assert _state_dict_sha256(omitted) == _state_dict_sha256(explicit)


def test_explicit_default_components_is_a_bitwise_no_op():
    omitted, _, *_ = _run(_config())
    explicit, _, *_ = _run(_config(components=("mlp.c_proj",)))
    assert _state_dict_sha256(omitted) == _state_dict_sha256(explicit)


def test_both_defaults_together_reproduce_the_current_direct_target_output():
    omitted, diagnostics, *_ = _run(_config())
    explicit, _, *_ = _run(_config(target_trajectory="step", components=("mlp.c_proj",)))
    assert _state_dict_sha256(omitted) == _state_dict_sha256(explicit)
    # ...and it is still a c_proj-only task vector.
    assert all(".mlp.c_proj." in key for key in omitted)
    assert all(row["component"] == "mlp.c_proj" for row in diagnostics)


# --------------------------------------------------------------------------
# 2. Feature 3 -- depth-interpolated desired effect
# --------------------------------------------------------------------------


def test_fractional_coordinates_come_from_the_realized_layout():
    doubled = [{"position": p, "source_orig_idx": p // 2} for p in range(24)]
    coordinates = realized_source_coordinates(doubled)
    assert [coordinates[p] for p in range(6)] == [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    assert coordinates[23] == 11.0


def test_undoubled_groups_get_integer_coordinates_and_are_an_exact_no_op():
    """A layout that duplicated nothing must give integers, i.e. step == interpolate."""
    untouched = [{"position": p, "source_orig_idx": p} for p in range(4)]
    coordinates = realized_source_coordinates(untouched)
    assert [coordinates[p] for p in range(4)] == [0.0, 1.0, 2.0, 3.0]


def test_uneven_groups_split_their_own_step():
    """Three realized positions for one source block split it in thirds."""
    entries = [{"position": 0, "source_orig_idx": 0}, {"position": 1, "source_orig_idx": 0},
               {"position": 2, "source_orig_idx": 0}, {"position": 3, "source_orig_idx": 1}]
    coordinates = realized_source_coordinates(entries)
    assert coordinates[0] == pytest.approx(-1 + 1 / 3)
    assert coordinates[1] == pytest.approx(-1 + 2 / 3)
    assert coordinates[2] == 0.0
    assert coordinates[3] == 1.0


def test_interpolation_changes_the_target_only_at_fractional_positions():
    """Integer coordinates must agree *exactly*, fractional ones must differ.

    The last member of every ancestry group lands on an integer, where the two
    trajectories are the same target by construction; taking a different code
    path there would silently break the comparison between the two campaign
    cells.
    """
    source, source_ft, target, data, layout, _sd = _fixture()
    references = _references(source, source_ft, target, data, _config())
    entries = layout["final_blocks"]
    stepped, _, _, _ = _materialize_all_scope_references(references, entries, trajectory="step")
    blended, _, _, coordinates = _materialize_all_scope_references(references, entries, trajectory="interpolate")

    for position in (1, 3):  # integer coordinates
        assert coordinates[position] == float(int(coordinates[position]))
        for a, b in zip(stepped[position], blended[position], strict=True):
            assert torch.equal(a, b), f"integer coordinate {position} must be bitwise identical"
    for position in (0, 2):  # fractional coordinates
        assert coordinates[position] != float(int(coordinates[position]))
        assert not all(
            torch.equal(a, b) for a, b in zip(stepped[position], blended[position], strict=True)
        ), f"fractional coordinate {position} should have been re-targeted"


def test_interpolated_first_position_asks_for_strictly_less_than_its_ancestor():
    """Signature 1, on the one pair where it is a theorem rather than a hypothesis.

    At coordinate -0.5 the blend is ``0.5 * Delta_0`` because ``Delta_{-1} = 0``
    by definition, so the demanded effect is strictly smaller than at the
    integer position. Deeper pairs blend two nonzero source effects, where no
    such inequality is guaranteed -- that is the hypothesis the campaign tests,
    and it is deliberately *not* asserted here or in the run.
    """
    source, source_ft, target, data, layout, _sd = _fixture()
    references = _references(source, source_ft, target, data, _config())
    blended, _, _, _ = _materialize_all_scope_references(
        references, layout["final_blocks"], trajectory="interpolate"
    )
    first = sum(float((b.double() ** 2).sum()) for b in blended[0]) ** 0.5
    second = sum(float((b.double() ** 2).sum()) for b in blended[1]) ** 0.5
    assert first < second


def test_interpolated_trajectory_changes_the_fitted_task_vector():
    stepped, _, *_ = _run(_config(target_trajectory="step"))
    blended, _, *_ = _run(_config(target_trajectory="interpolate"))
    assert _state_dict_sha256(stepped) != _state_dict_sha256(blended)
    assert set(stepped) == set(blended), "the trajectory must not change which keys are written"


def test_diagnostics_carry_the_trajectory_and_the_coordinate():
    _corrections, diagnostics, *_ = _run(_config(target_trajectory="interpolate"))
    assert all(row["trajectory"] == "interpolate" for row in diagnostics)
    assert [row["target_coordinate"] for row in diagnostics] == [-0.5, 0.0, 0.5, 1.0]


def test_interpolate_with_inserted_scope_is_refused():
    with pytest.raises(ValueError, match="requires target_scope='all'"):
        parse_residual_completion_config(
            {"enabled": True, "mode": "direct_target", "target_scope": "inserted",
             "target_trajectory": "interpolate"}
        )


def test_interpolate_outside_direct_mode_is_refused():
    with pytest.raises(ValueError, match="requires mode='direct_target'"):
        parse_residual_completion_config(
            {"enabled": True, "target_scope": "all", "target_trajectory": "interpolate"}
        )


# --------------------------------------------------------------------------
# 3. Feature 4 -- attn.out_proj as a second write surface
# --------------------------------------------------------------------------


_BOTH = ("attn.out_proj", "mlp.c_proj")


def test_components_are_fitted_in_block_forward_order_however_they_are_listed():
    assert order_components(["mlp.c_proj", "attn.out_proj"]) == _BOTH
    assert order_components(["attn.out_proj", "mlp.c_proj"]) == _BOTH


def test_both_components_write_exactly_the_two_projections():
    corrections, diagnostics, *_ = _run(_config(components=_BOTH))
    expected = set()
    for position in range(4):
        stem = f"visual.transformer.resblocks.{position}"
        expected |= {
            f"{stem}.attn.out_proj.weight", f"{stem}.attn.out_proj.bias",
            f"{stem}.mlp.c_proj.weight", f"{stem}.mlp.c_proj.bias",
        }
    assert set(corrections) == expected
    assert [row["component"] for row in diagnostics] == list(_BOTH) * 4


def test_the_two_components_are_cascaded_not_fitted_independently():
    """The MLP fit must see a model that already carries Delta_O.

    If the two were solved as one system, or simply fitted against the same
    captured residual, the c_proj correction would be identical to the one the
    c_proj-only run produces. It must not be: mounting Delta_O changes both the
    residual the MLP has left to explain and -- through ln_2 and GELU -- the
    features it explains it with.
    """
    both, _, *_ = _run(_config(components=_BOTH))
    only_mlp, _, *_ = _run(_config(components=("mlp.c_proj",)))
    key = "visual.transformer.resblocks.0.mlp.c_proj.weight"
    assert not torch.allclose(both[key], only_mlp[key], atol=1e-7), (
        "the c_proj fit did not react to the mounted attn correction: the "
        "intra-block re-capture is missing and the cascade is not real"
    )


def _remount(target, target_base_sd, corrections, keys):
    mounted = deepcopy(target)
    state = {k: v.clone() for k, v in target_base_sd.items()}
    for key in keys:
        state[key] = state[key] + corrections[key].to(state[key])
    mounted.load_state_dict(state, strict=True)
    return mounted


def _measured_residual_norm(model, references, data, position):
    meta = references["calibration"]
    batches = list(DataLoader(Subset(data.dataset, meta["indices"]), batch_size=meta["batch_size"],
                              shuffle=False, num_workers=0, collate_fn=data.collate_fn))
    captured = capture_tokens(model, batches, {"out": (position, "boundary")}, "cpu")
    desired, target_outputs, _maps, _coords = _materialize_all_scope_references(
        references,
        [{"position": p, "source_orig_idx": p // 2} for p in range(4)],
        trajectory="step",
    )
    residual_sq = 0.0
    for out, desired_batch, base_out in zip(
        captured["out"], desired[position], target_outputs[position], strict=True
    ):
        error = desired_batch.double() - (out.double() - base_out.double())
        residual_sq += float((error ** 2).sum().item())
    return residual_sq ** 0.5


def test_attn_row_reports_a_measured_post_mount_residual():
    """``residual_norm_after`` is only a linear prediction for attn.out_proj.

    Its input is not independent of the MLP that runs after it, so the row must
    also carry the residual actually measured once the correction is mounted --
    which is what the next component's capture provides for free.
    """
    corrections, diagnostics, references, target, target_base_sd, data = _run(_config(components=_BOTH))
    attn_row = diagnostics[0]
    assert attn_row["component"] == "attn.out_proj"
    assert "measured_residual_norm_after" in attn_row

    mounted = _remount(target, target_base_sd, corrections,
                       ["visual.transformer.resblocks.0.attn.out_proj.weight",
                        "visual.transformer.resblocks.0.attn.out_proj.bias"])
    measured = _measured_residual_norm(mounted, references, data, 0)
    assert measured == pytest.approx(attn_row["measured_residual_norm_after"], rel=1e-6)


def test_the_mlp_row_never_claims_a_measured_residual_it_did_not_take():
    corrections, diagnostics, *_ = _run(_config(components=_BOTH))
    mlp_rows = [row for row in diagnostics if row["component"] == "mlp.c_proj"]
    assert mlp_rows and all("measured_residual_norm_after" not in row for row in mlp_rows)


def test_mlp_solver_residual_is_still_exact_with_both_components():
    """The c_proj exactness argument survives the extra write surface.

    c_proj is still the last operation writing into the residual stream and its
    input still does not depend on its own weight, so its solver objective is
    still the true post-mount residual.
    """
    corrections, diagnostics, references, target, target_base_sd, data = _run(_config(components=_BOTH))
    stem = "visual.transformer.resblocks.0"
    mounted = _remount(target, target_base_sd, corrections, [
        f"{stem}.attn.out_proj.weight", f"{stem}.attn.out_proj.bias",
        f"{stem}.mlp.c_proj.weight", f"{stem}.mlp.c_proj.bias",
    ])
    measured = _measured_residual_norm(mounted, references, data, 0)
    mlp_row = diagnostics[1]
    assert mlp_row["component"] == "mlp.c_proj"
    assert measured == pytest.approx(mlp_row["residual_norm_after"], rel=1e-3)


def test_first_row_still_starts_from_the_untouched_target_base():
    for components in (("mlp.c_proj",), _BOTH):
        _corrections, diagnostics, *_ = _run(_config(components=components))
        first = diagnostics[0]
        assert first["effect_before_norm"] == pytest.approx(0.0, abs=1e-6)
        assert first["relative_residual_before"] == pytest.approx(1.0, rel=1e-5)


def test_out_proj_only_is_refused_because_the_anchor_component_is_missing():
    with pytest.raises(ValueError, match="must contain 'mlp.c_proj'"):
        parse_residual_completion_config(
            {"enabled": True, "mode": "direct_target", "components": ["attn.out_proj"]}
        )


def test_two_components_outside_direct_mode_are_refused():
    with pytest.raises(ValueError, match="require mode='direct_target'"):
        parse_residual_completion_config({"enabled": True, "components": list(_BOTH)})


# --------------------------------------------------------------------------
# 4. The production capture path: a stock nn.MultiheadAttention
# --------------------------------------------------------------------------


class _StockAttentionBlock(torch.nn.Module):
    """OpenCLIP's real shape: ``attn`` is a stock ``nn.MultiheadAttention``.

    torch applies ``out_proj`` functionally inside
    ``F.multi_head_attention_forward``, so a forward hook on the ``out_proj``
    submodule never fires. This is the configuration every vision campaign
    actually runs, because direct_target mode skips the transport fit that
    would otherwise have patched the attention into explicit projections.
    """

    def __init__(self, width):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, 1, batch_first=True)
        self.ls_1 = torch.nn.Identity()
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(width, width * 2)), ("gelu", torch.nn.GELU()),
            ("c_proj", torch.nn.Linear(width * 2, width)),
        ]))
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        normed = self.ln_1(x)
        x = x + self.ls_1(self.attn(normed, normed, normed, need_weights=False)[0])
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class _StockVisual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList(
            [_StockAttentionBlock(width) for _ in range(depth)]
        )

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _StockModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _StockVisual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def test_a_hook_on_stock_multihead_out_proj_would_never_fire():
    """The premise of the workaround, asserted rather than assumed."""
    attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
    fired = []
    attention.out_proj.register_forward_hook(lambda *_: fired.append(1))
    x = torch.randn(2, 5, 8)
    attention(x, x, x, need_weights=False)
    assert fired == [], "torch started calling out_proj; the recompute path can be dropped"


def _stock_fixture():
    """The production shape end to end: stock attention, source 2 -> target 4."""
    torch.manual_seed(21)
    source = _StockModel(4, 2).eval()
    target = _StockModel(6, 4).eval()
    source_ft = deepcopy(source)
    with torch.no_grad():
        source_ft.visual.transformer.resblocks[0].mlp.c_proj.weight.add_(0.2)
        source_ft.visual.transformer.resblocks[1].attn.out_proj.weight.sub_(0.1)
    layout = {
        "final_blocks": [
            {"position": p, "source_orig_idx": p // 2,
             "block_kind": "original" if p % 2 == 0 else "inserted"}
            for p in range(4)
        ]
    }
    return source, source_ft, target, _loader(), layout, {k: v.clone() for k, v in target.state_dict().items()}


@pytest.mark.parametrize("trajectory", ["step", "interpolate"])
def test_both_components_run_end_to_end_against_stock_attention(trajectory):
    """The configuration the vision campaign actually runs.

    ``direct_target`` skips the transport fit, so nothing ever patches the
    attention into explicit projections: the target carries a stock
    ``nn.MultiheadAttention`` whose ``out_proj`` is applied functionally. If the
    recompute path regressed, ``capture_tokens`` would raise rather than fit on
    the wrong features.
    """
    config = _config(components=_BOTH, target_trajectory=trajectory)
    source, source_ft, target, data, layout, target_base_sd = _stock_fixture()
    references = _references(source, source_ft, target, data, config)
    before = _state_dict_sha256(target.state_dict())
    corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    assert _state_dict_sha256(target.state_dict()) == before
    assert [row["component"] for row in diagnostics] == list(_BOTH) * 4
    for position in range(4):
        stem = f"visual.transformer.resblocks.{position}"
        assert f"{stem}.attn.out_proj.weight" in corrections
        assert f"{stem}.attn.out_proj.bias" in corrections
    assert diagnostics[0]["relative_residual_before"] == pytest.approx(1.0, rel=1e-5)
    for row in diagnostics:
        assert row["residual_norm_after"] < row["residual_norm_before"], (row["position"], row["component"])


def test_stock_multihead_attention_rows_are_recovered_exactly():
    """The captured rows must be the ones ``out_proj`` actually consumes.

    Pushing them back through the real projection has to reproduce the
    attention's own output, otherwise the fit would be regressing on the wrong
    features.
    """
    torch.manual_seed(3)
    model = _StockModel(8, 2).eval()
    batches = list(_loader())
    captured = capture_tokens(
        model, batches, {"h": (0, "attn_proj_input"), "y": (0, "attn_proj")}, "cpu",
    )
    projection = model.visual.transformer.resblocks[0].attn.out_proj
    assert len(captured["h"]) == len(batches)
    for rows, output in zip(captured["h"], captured["y"], strict=True):
        replayed = torch.nn.functional.linear(rows, projection.weight, projection.bias)
        torch.testing.assert_close(replayed, output, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# 5. Invariants that must survive both features
# --------------------------------------------------------------------------


def _complete(config):
    source, source_ft, target, data, layout, target_base_sd = _fixture()
    references = _references(source, source_ft, target, data, config)
    return _maybe_complete_target_residual_task_vector(
        config=config,
        references=references,
        prepared=None,  # any transport read would be fatal
        layout=layout,
        target_model=target,
        target_base_sd=target_base_sd,
        transported_delta={},
        target_loader=data,
        device="cpu",
    )


@pytest.mark.parametrize("components", [("mlp.c_proj",), _BOTH])
@pytest.mark.parametrize("trajectory", ["step", "interpolate"])
def test_gamma_zero_reproduces_the_native_target_base_in_every_configuration(components, trajectory):
    completed, diagnostics = _complete(
        _config(strength=0.0, components=components, target_trajectory=trajectory)
    )
    assert diagnostics, "the fit still runs at gamma=0"
    assert completed, "the zero task vector must still carry its keys"
    for value in completed.values():
        assert torch.count_nonzero(value) == 0


@pytest.mark.parametrize("components", [("mlp.c_proj",), _BOTH])
@pytest.mark.parametrize("trajectory", ["step", "interpolate"])
def test_determinism_under_a_fixed_seed(components, trajectory):
    config = _config(components=components, target_trajectory=trajectory)
    first, _, *_ = _run(config, seed=7)
    second, _, *_ = _run(config, seed=7)
    assert _state_dict_sha256(first) == _state_dict_sha256(second)


@pytest.mark.parametrize("components", [("mlp.c_proj",), _BOTH])
@pytest.mark.parametrize("trajectory", ["step", "interpolate"])
def test_the_target_base_is_restored_and_the_fit_reduces_the_residual(components, trajectory):
    config = _config(components=components, target_trajectory=trajectory)
    source, source_ft, target, data, layout, target_base_sd = _fixture()
    references = _references(source, source_ft, target, data, config)
    before = _state_dict_sha256(target.state_dict())
    _corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, data, config=config, device="cpu",
    )
    assert _state_dict_sha256(target.state_dict()) == before, "no hooks or mounted weights may survive"
    for row in diagnostics:
        assert row["residual_norm_after"] < row["residual_norm_before"], (row["position"], row["component"])
