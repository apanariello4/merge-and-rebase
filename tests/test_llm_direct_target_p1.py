"""The transport-free Proposal-1 arm on the HF-decoder path.

``complete_residuals_direct`` and its solver are already pinned on the vision
side (``tests/test_direct_target_p1.py``,
``tests/test_direct_p1_trajectory_and_components.py``).  What was never covered
is the *decoder plumbing* it has to run through: ``self_attn.o_proj`` as a real
``nn.Linear`` write surface, the bias-free projections a materialize pre-pass
has to fill in, and ``llm_rebase``'s own mode branch, which must reach
completion with an empty transported task vector or the arm is not
transport-free at all.

It also covers the two decoder-only knobs the arm grew when the two
independent implementations of it were merged: ``direct_passthrough``, which
decides whether the non-transportable remainder is folded in or dropped, and
``cascade_order``, which decides how (or whether) the per-block fits couple.

The models here are deliberately tiny stand-ins with Qwen's structural
properties -- bias-free ``mlp.down_proj`` and ``self_attn.o_proj``, biased
q/k/v, no LayerScale -- and a genuine width-up, depth-up rebase (8 -> 12 wide,
3 -> 4 deep), because that is the shape of the pair the campaign runs.
"""

from __future__ import annotations

import math
import hashlib

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.eval.target_informed_runtime import (
    capture_residual_references,
    capture_tokens,
    complete_residuals_direct,
    materialize_missing_projection_biases,
    scale_completion,
)
from merge_and_rebase.eval.target_residual_completion import (
    ResidualCompletionConfig,
    parse_residual_completion_config,
)

VOCAB = 32


class _Attn(nn.Module):
    """Qwen's bias convention: q/k/v carry a bias, o_proj does not."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=True)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.width = width

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width)
        return self.o_proj(torch.softmax(scores, dim=-1) @ v)


class _MLP(nn.Module):
    def __init__(self, width: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, inter, bias=False)
        self.up_proj = nn.Linear(width, inter, bias=False)
        self.down_proj = nn.Linear(inter, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, width: int, inter: int) -> None:
        super().__init__()
        self.self_attn = _Attn(width)
        self.mlp = _MLP(width, inter)
        self.input_layernorm = nn.LayerNorm(width)
        self.post_attention_layernorm = nn.LayerNorm(width)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class _Body(nn.Module):
    def __init__(self, width, inter, depth, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList([_Layer(width, inter) for _ in range(depth)])

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class _Decoder(nn.Module):
    """Keys land at ``model.layers.N...``, exactly as ``_DecoderLayout`` expects."""

    def __init__(self, width=8, inter=16, depth=3, vocab=VOCAB, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Body(width, inter, depth, vocab)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)


class _Adapter:
    """The slice of HfDecoderAdapter's surface the runtime actually touches."""

    def transport_scope(self, model):
        return model.model

    def block_count(self, model):
        return len(model.model.layers)

    def extract_calibration_batch(self, batch):
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


class _Texts(Dataset):
    def __init__(self, n=6, seq=5, offset=0):
        self.n = n
        self.seq = seq
        self.offset = offset
        # Identity of the *examples*, not of this tokenization: the source and
        # target loaders tokenize the same texts with different tokenizers.
        self.sample_ids = [f"ex{i}" for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i + self.offset)
        return {
            "input_ids": torch.randint(0, VOCAB, (self.seq,), generator=g),
            "attention_mask": torch.ones(self.seq, dtype=torch.long),
        }


def _collate(rows):
    return {
        "input_ids": torch.stack([r["input_ids"] for r in rows]),
        "attention_mask": torch.stack([r["attention_mask"] for r in rows]),
    }


def _loader(offset=0, batch_size=2):
    return DataLoader(_Texts(offset=offset), batch_size=batch_size, shuffle=False, collate_fn=_collate)


def _batches(n=2, bs=2, seq=5):
    return [
        {
            "input_ids": torch.randint(0, VOCAB, (bs, seq)),
            "attention_mask": torch.ones(bs, seq, dtype=torch.long),
        }
        for _ in range(n)
    ]


def _fixture(*, source_depth=3, target_depth=4, source_width=8, target_width=12):
    """A genuine width-up, depth-up decoder rebase, plus its realized layout."""
    source = _Decoder(width=source_width, inter=2 * source_width, depth=source_depth, seed=0)
    source_ft = _Decoder(width=source_width, inter=2 * source_width, depth=source_depth, seed=0)
    with torch.no_grad():
        # A task vector: perturb the tuned copy away from the base.
        for param in source_ft.parameters():
            param.add_(0.05 * torch.randn_like(param))
    target = _Decoder(width=target_width, inter=2 * target_width, depth=target_depth, seed=7)
    target_base_sd = {k: v.detach().clone() for k, v in target.state_dict().items()}
    # 3 source blocks realized as 4 target positions: block 1 is duplicated,
    # which is what makes the ancestry groups (and so "interpolate") non-trivial.
    ancestry = [0, 1, 1, 2]
    layout = {
        "final_blocks": [
            {"position": pos, "source_orig_idx": src, "block_kind": "inserted" if pos == 2 else "original"}
            for pos, src in enumerate(ancestry)
        ]
    }
    return source, source_ft, target, target_base_sd, layout


def _config(**overrides):
    params = {
        "enabled": True,
        "mode": "direct_target",
        "target_scope": "all",
        "ridge_relative": 0.05,
        "num_batches": 3,
        "strength": 1.0,
        "exact_form": True,
        "missing_bias": "materialize",
    }
    params.update(overrides)
    return ResidualCompletionConfig(**params)


def _references(source, source_ft, target, config, adapter):
    return capture_residual_references(
        source, source_ft, target,
        _loader(offset=0), _loader(offset=100),
        num_batches=config.num_batches, seed=0, device="cpu",
        target_scope=config.target_scope, family_adapter=adapter,
    )


# ---------------------------------------------------------------------------
# 1. The decoder attention write surface
# ---------------------------------------------------------------------------


def test_attn_proj_input_capture_fires_once_per_batch_on_a_decoder():
    """`o_proj` is a real submodule on a decoder, so a plain forward hook fires.

    The vision path needs ``_stock_mha_out_proj_input`` because torch applies
    ``out_proj`` functionally inside ``F.multi_head_attention_forward`` and a
    hook on it never fires at all.  A decoder needs none of that machinery --
    but "needs none" was an inference from the module tree, never a measurement,
    and the layout hook's own docstring called itself untested.  This measures it.
    """
    model = _Decoder(depth=3)
    batches = _batches(n=3)
    out = capture_tokens(
        model, batches,
        {"rows": (1, "attn_proj_input"), "attn": (1, "attn_proj")},
        "cpu", family_adapter=_Adapter(),
    )
    assert len(out["rows"]) == len(batches), "the o_proj hook did not fire once per batch"
    assert len(out["attn"]) == len(batches)
    assert out["rows"][0].shape == out["attn"][0].shape


def test_captured_attn_rows_reproduce_the_projection_output():
    """Pushing the captured rows back through o_proj must give its real output.

    This is the property the fit depends on: the regression features have to be
    exactly what the projection consumes, or the solve optimizes the wrong map.
    """
    model = _Decoder(depth=2)
    batches = _batches(n=2)
    out = capture_tokens(
        model, batches,
        {"rows": (0, "attn_proj_input"), "attn": (0, "attn_proj")},
        "cpu", family_adapter=_Adapter(),
    )
    o_proj = model.model.layers[0].self_attn.o_proj
    for rows, reference in zip(out["rows"], out["attn"], strict=True):
        replayed = torch.nn.functional.linear(rows, o_proj.weight, o_proj.bias)
        assert torch.allclose(replayed, reference, atol=1e-5), (
            "captured rows do not reproduce the attention output through o_proj"
        )


# ---------------------------------------------------------------------------
# 2. Bias materialization over every fitted component
# ---------------------------------------------------------------------------


def test_materialize_covers_every_configured_component():
    """The two-component arm also needs an o_proj bias; Qwen has neither."""
    model = _Decoder(depth=3)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    layout = {"final_blocks": [{"position": i} for i in range(3)]}

    added = materialize_missing_projection_biases(
        model, base_state, layout, family_adapter=_Adapter(),
        components=("attn.out_proj", "mlp.c_proj"),
    )

    assert len(added) == 6, added
    for i in range(3):
        for key in (f"model.layers.{i}.mlp.down_proj.bias", f"model.layers.{i}.self_attn.o_proj.bias"):
            assert key in added
            assert key in base_state
            assert torch.count_nonzero(base_state[key]) == 0
    # ...and the model agrees with the state dict, or the solver's strict
    # restore dies on a key the snapshot never had.
    assert set(base_state) == set(model.state_dict())


def test_materialize_default_is_the_historical_single_component():
    """Existing callers must be unchanged: mlp only, no o_proj bias."""
    model = _Decoder(depth=2)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    added = materialize_missing_projection_biases(
        model, base_state, {"final_blocks": [{"position": i} for i in range(2)]},
        family_adapter=_Adapter(),
    )
    assert added == ["model.layers.0.mlp.down_proj.bias", "model.layers.1.mlp.down_proj.bias"]


# ---------------------------------------------------------------------------
# 3. The direct arm end to end on a decoder
# ---------------------------------------------------------------------------


def test_direct_completion_writes_only_the_decoder_projection_keys():
    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config()
    materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    references = _references(source, source_ft, target, config, adapter)

    corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, _loader(offset=100),
        config=config, device="cpu", family_adapter=adapter,
    )

    assert [row["position"] for row in diagnostics] == [0, 1, 2, 3]
    assert all(row["mode"] == "direct_target" for row in diagnostics)
    expected = set()
    for pos in range(4):
        expected.add(f"model.layers.{pos}.mlp.down_proj.weight")
        expected.add(f"model.layers.{pos}.mlp.down_proj.bias")
    assert set(corrections) == expected
    for key, correction in corrections.items():
        assert correction.shape == target_base_sd[key].shape
        assert torch.isfinite(correction).all()


def test_first_fitted_block_reports_a_relative_residual_of_one():
    """Proves the temporary model started from the untouched target base.

    This is the acceptance gate's check 3, measured on the decoder path rather
    than assumed from the vision one.
    """
    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config()
    materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    references = _references(source, source_ft, target, config, adapter)

    _corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, _loader(offset=100),
        config=config, device="cpu", family_adapter=adapter,
    )

    first = diagnostics[0]
    assert first["effect_before_norm"] == pytest.approx(0.0, abs=1e-6)
    assert first["relative_residual_before"] == pytest.approx(1.0, rel=1e-4)
    # And the fit has to actually improve on its own objective, everywhere.
    for row in diagnostics:
        assert row["residual_norm_after"] < row["residual_norm_before"], row["position"]


def test_two_component_arm_fits_o_proj_as_well_on_a_decoder():
    """`components` becomes reachable in direct mode; the decoder must honour it."""
    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config(components=("attn.out_proj", "mlp.c_proj"))
    materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    references = _references(source, source_ft, target, config, adapter)

    corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, _loader(offset=100),
        config=config, device="cpu", family_adapter=adapter,
    )

    assert {row["component"] for row in diagnostics} == {"attn.out_proj", "mlp.c_proj"}
    assert any(key.endswith("self_attn.o_proj.weight") for key in corrections)
    assert any(key.endswith("self_attn.o_proj.bias") for key in corrections)
    # The attention rows are fitted before the MLP ones in every block, and the
    # MLP capture re-measures what o_proj actually left behind.
    attn_rows = [row for row in diagnostics if row["component"] == "attn.out_proj"]
    assert all("measured_residual_norm_after" in row for row in attn_rows)


def test_strength_zero_is_an_exact_native_target_base_control():
    """gamma=0 must leave the target bit-identical, materialized biases included."""
    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config()
    added = materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    assert added, "the fixture must exercise the materialize path"
    snapshot = {k: v.detach().clone() for k, v in target.state_dict().items()}
    references = _references(source, source_ft, target, config, adapter)

    corrections, _diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, _loader(offset=100),
        config=config, device="cpu", family_adapter=adapter,
    )

    # The solver restores the model it was handed...
    after = target.state_dict()
    assert set(after) == set(snapshot)
    for key, value in snapshot.items():
        assert torch.equal(after[key], value), f"{key} was not restored"

    # ...and at gamma=0 the task vector is identically zero against the explicit
    # zero baseline the llm/vision callers scale with.
    zero_baseline = {k: torch.zeros_like(v) for k, v in corrections.items()}
    scaled = scale_completion(zero_baseline, corrections, 0.0)
    assert set(scaled) == set(corrections)
    assert all(torch.count_nonzero(v) == 0 for v in scaled.values())

    # At gamma=1 it is not, or the control would be vacuous.
    unit = scale_completion(zero_baseline, corrections, 1.0)
    assert any(torch.count_nonzero(v) > 0 for v in unit.values())


def test_interpolate_trajectory_is_reachable_on_the_decoder():
    """The duplicated ancestry group must get fractional depth coordinates."""
    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config(target_trajectory="interpolate")
    materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    references = _references(source, source_ft, target, config, adapter)

    _corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, _loader(offset=100),
        config=config, device="cpu", family_adapter=adapter,
    )

    coordinates = {row["position"]: row["target_coordinate"] for row in diagnostics}
    # Source block 1 is realized at positions 1 and 2, so it splits the step
    # from boundary 0 to boundary 1 in half.
    assert coordinates == {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}
    assert all(row["trajectory"] == "interpolate" for row in diagnostics)


# ---------------------------------------------------------------------------
# 4. llm_rebase's mode branch
# ---------------------------------------------------------------------------


def test_llm_helper_refuses_a_non_empty_delta_in_direct_mode():
    """The empty-delta contract is what makes the arm transport-free at all."""
    from merge_and_rebase.eval.llm_rebase import _maybe_complete_target_residual_task_vector

    with pytest.raises(ValueError, match="empty transported task vector"):
        _maybe_complete_target_residual_task_vector(
            config=_config(),
            references={"calibration": {}},
            prepared=None,
            layout={"final_blocks": [{"position": 0, "source_orig_idx": 0}]},
            target_model=None,
            target_base_sd={},
            transported_delta={"model.layers.0.mlp.down_proj.weight": torch.zeros(2, 2)},
            target_loader=None,
            family_adapter=None,
            device="cpu",
        )


def test_llm_helper_returns_a_task_vector_with_no_transported_keys():
    """End to end through llm_rebase's own helper: the fit is the whole vector."""
    from merge_and_rebase.eval.llm_rebase import _maybe_complete_target_residual_task_vector

    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config()
    materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    references = _references(source, source_ft, target, config, adapter)

    completed, diagnostics = _maybe_complete_target_residual_task_vector(
        config=config,
        references=references,
        prepared=None,  # no fitted transport exists, and none may be required
        layout=layout,
        target_model=target,
        target_base_sd=target_base_sd,
        transported_delta={},
        target_loader=_loader(offset=100),
        family_adapter=adapter,
        device="cpu",
        materialized_bias_keys=set(),
    )

    assert diagnostics and len(diagnostics) == 4
    # Only the fitted residual-writing projections; nothing transported, and in
    # particular no attention/gate/up/embedding keys came along for the ride.
    assert set(completed) == {
        f"model.layers.{pos}.mlp.down_proj.{suffix}"
        for pos in range(4)
        for suffix in ("weight", "bias")
    }
    assert any(torch.count_nonzero(v) > 0 for v in completed.values())


# ---------------------------------------------------------------------------
# 5. The cascade_order ablation
#
# The ablation shipped without tests. What makes it meaningful is a measurable
# difference in how the per-block fits couple, so that is what these pin:
# `independent` breaks the coupling entirely, `top_bottom` reverses it, and the
# default keeps the historical order.
# ---------------------------------------------------------------------------


def _diagnostics_for(cascade_order, **overrides):
    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config(cascade_order=cascade_order, **overrides)
    materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    references = _references(source, source_ft, target, config, adapter)
    corrections, diagnostics = complete_residuals_direct(
        target, target_base_sd, references, layout, _loader(offset=100),
        config=config, device="cpu", family_adapter=adapter,
    )
    return corrections, diagnostics


def test_cascade_order_default_visits_blocks_bottom_up():
    _corrections, diagnostics = _diagnostics_for("bottom_top")
    assert [row["position"] for row in diagnostics] == [0, 1, 2, 3]


def test_cascade_order_top_bottom_visits_blocks_deepest_first():
    """Only the coupling direction changes; every block is still fitted once."""
    _corrections, diagnostics = _diagnostics_for("top_bottom")
    assert [row["position"] for row in diagnostics] == [3, 2, 1, 0]
    # The first block visited still sees the untouched base, which is what keeps
    # the `index == 0` assertion inside the solver valid under reversal.
    assert diagnostics[0]["effect_before_norm"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics[0]["relative_residual_before"] == pytest.approx(1.0, rel=1e-4)


def test_cascade_order_independent_leaves_every_block_seeing_the_base():
    """No mount between fits, so E_j == D_j everywhere -- that is the whole point.

    Under a real cascade only the first block reports an untouched base; the
    rest inherit whatever upstream corrections left behind (measured above 1.0
    in practice). `independent` must flatten that to 1.0 for all of them.
    """
    _corrections, diagnostics = _diagnostics_for("independent")
    for row in diagnostics:
        assert row["effect_before_norm"] == pytest.approx(0.0, abs=1e-6), row["position"]
        assert row["relative_residual_before"] == pytest.approx(1.0, rel=1e-4), row["position"]
    # And the fit still has to improve on its own objective everywhere.
    for row in diagnostics:
        assert row["residual_norm_after"] < row["residual_norm_before"], row["position"]


def test_cascade_order_changes_the_fitted_correction():
    """If the three orders produced the same weights the ablation would be vacuous."""
    bottom_top, _ = _diagnostics_for("bottom_top")
    independent, _ = _diagnostics_for("independent")
    assert set(bottom_top) == set(independent)
    key = "model.layers.3.mlp.down_proj.weight"
    assert not torch.allclose(bottom_top[key], independent[key], atol=1e-6), (
        "the deepest block is the one furthest downstream of the cascade; if its "
        "correction is unchanged, the coupling is not doing anything"
    )


def test_cascade_order_is_validated():
    assert ResidualCompletionConfig().cascade_order == "bottom_top"
    for name in ("bottom_top", "top_bottom", "independent"):
        assert parse_residual_completion_config(
            {"enabled": True, "mode": "direct_target", "cascade_order": name}
        ).cascade_order == name
    with pytest.raises(ValueError, match="cascade_order must be"):
        parse_residual_completion_config({"enabled": True, "cascade_order": "sideways"})


# ---------------------------------------------------------------------------
# 6. The passthrough decision
#
# Decoder-only: that path splits a task vector into a transportable body and a
# remainder (embeddings, per-layer norms, lm_head) which vision has no analogue
# for. The policy is three-way -- transport folds, direct drops by default, and
# direct_passthrough=true opts back into folding -- so it is pinned directly
# rather than through a full run.
# ---------------------------------------------------------------------------


def test_direct_passthrough_defaults_off_and_requires_direct_mode():
    assert ResidualCompletionConfig().direct_passthrough is False
    assert parse_residual_completion_config({"enabled": True}).direct_passthrough is False
    cfg = parse_residual_completion_config(
        {"enabled": True, "mode": "direct_target", "direct_passthrough": True}
    )
    assert cfg.direct_passthrough is True
    with pytest.raises(ValueError, match="direct_passthrough=true requires"):
        parse_residual_completion_config({"enabled": True, "direct_passthrough": True})


def _passthrough_pair(base):
    """A shape-compatible passthrough key and a shape-incompatible one."""
    compatible = "model.embed_tokens.weight"
    return {
        compatible: torch.ones_like(base[compatible]),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(3, 3),
    }


def test_passthrough_is_dropped_when_not_carried():
    from merge_and_rebase.eval.llm_rebase import _apply_passthrough_delta

    _s, _sf, _t, target_base_sd, _layout = _fixture()
    passthrough = _passthrough_pair(target_base_sd)

    out, skipped, dropped = _apply_passthrough_delta(
        {}, passthrough, target_base_sd, carry=False
    )

    assert out == {}
    assert skipped == [], "nothing is even considered for shape when the policy is drop"
    assert dropped == set(passthrough)


def test_passthrough_is_folded_when_carried():
    from merge_and_rebase.eval.llm_rebase import _apply_passthrough_delta

    _s, _sf, _t, target_base_sd, _layout = _fixture()
    passthrough = _passthrough_pair(target_base_sd)

    out, skipped, dropped = _apply_passthrough_delta(
        {}, passthrough, target_base_sd, carry=True
    )

    assert "model.embed_tokens.weight" in out
    torch.testing.assert_close(
        out["model.embed_tokens.weight"], torch.ones_like(target_base_sd["model.embed_tokens.weight"])
    )
    # The shape-incompatible key is reported, not silently swallowed.
    assert skipped == ["model.layers.0.self_attn.q_proj.weight"]
    assert dropped == set()


def test_passthrough_does_not_overwrite_the_fitted_correction():
    """The body is written first; a passthrough key must not clobber a fitted one."""
    from merge_and_rebase.eval.llm_rebase import _apply_passthrough_delta

    _s, _sf, _t, target_base_sd, _layout = _fixture()
    key = "model.layers.0.mlp.down_proj.weight"
    fitted = torch.full_like(target_base_sd[key], 0.5)

    out, _skipped, _dropped = _apply_passthrough_delta(
        {key: fitted}, {}, target_base_sd, carry=True
    )

    torch.testing.assert_close(out[key], fitted)


def test_disabled_completion_returns_the_delta_untouched():
    """A disabled config is a no-op on both arms, direct mode included."""
    from merge_and_rebase.eval.llm_rebase import _maybe_complete_target_residual_task_vector

    delta = {"model.layers.0.mlp.down_proj.weight": torch.zeros(2, 2)}
    completed, diagnostics = _maybe_complete_target_residual_task_vector(
        config=_config(enabled=False),
        references={"calibration": {}},
        prepared=None,
        layout={"final_blocks": [{"position": 0, "source_orig_idx": 0}]},
        target_model=None,
        target_base_sd={},
        transported_delta=delta,
        target_loader=None,
        family_adapter=None,
        device="cpu",
        materialized_bias_keys=set(),
    )
    assert completed is delta, "a disabled run must return the same object, not a copy"
    assert diagnostics is None


def test_transport_mode_does_not_reach_the_direct_solver():
    """The mode branch must actually select; both arms sharing one helper is the risk.

    `transport_residual` needs fitted (t_in, t_out) maps, which `prepared=None`
    cannot supply. If the mode branch were ever deleted, this config would fall
    through to the transport-free solver and quietly succeed.
    """
    from merge_and_rebase.eval.llm_rebase import _maybe_complete_target_residual_task_vector

    source, source_ft, target, target_base_sd, layout = _fixture()
    adapter = _Adapter()
    config = _config(mode="transport_residual")
    materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=adapter, components=config.components
    )
    references = _references(source, source_ft, target, config, adapter)

    with pytest.raises((AttributeError, TypeError, ValueError, KeyError)):
        _maybe_complete_target_residual_task_vector(
            config=config,
            references=references,
            prepared=None,
            layout=layout,
            target_model=target,
            target_base_sd=target_base_sd,
            transported_delta={},
            target_loader=_loader(offset=100),
            family_adapter=adapter,
            device="cpu",
            materialized_bias_keys=set(),
        )


# ---------------------------------------------------------------------------
# 7. Building the tuned body from its own config
#
# A tuned body is normally the same architecture as its base, so building it as a
# deepcopy of the source and overwriting the weights is right. It stops being right
# when the checkpoint carries its own config: every parameter shape still matches, so
# the load succeeds and nothing downstream complains, but the tuned weights then run
# under the SOURCE's positional geometry and every captured activation is wrong.
#
# This is not hypothetical. Qwen ships every Math model with rope_theta=1e4 /
# max_position_embeddings=4096 and every general model with rope_theta=1e6, so the
# general-base + Math-tuned pairing -- the only one that isolates the *math* task
# vector rather than an instruct one -- lands exactly on it.
# ---------------------------------------------------------------------------


class _Cfg:
    """Stand-in for an HF PretrainedConfig: attribute access is all that is used."""

    def __init__(self, **kw):
        self.model_type = kw.pop("model_type", "qwen2")
        for k, v in kw.items():
            setattr(self, k, v)


class _ModelWithConfig(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def _patch_autoconfig(monkeypatch, mapping):
    """Route AutoConfig.from_pretrained to a dict of ref -> config (or raise)."""
    import transformers

    class _AutoConfig:
        @staticmethod
        def from_pretrained(ref, **_kw):
            if ref not in mapping:
                raise OSError(f"no config for {ref!r}")
            return mapping[ref]

    monkeypatch.setattr(transformers, "AutoConfig", _AutoConfig)


def test_matching_configs_report_no_mismatch(monkeypatch):
    """The no-op guarantee: an ordinary base/instruct pair keeps the deepcopy path."""
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    shared = dict(rope_theta=1000000.0, max_position_embeddings=32768)
    _patch_autoconfig(monkeypatch, {"org/Instruct": _Cfg(**shared)})
    source = _ModelWithConfig(_Cfg(**shared))

    assert _tuned_config_mismatch("org/Instruct", source) == {}


def test_rope_theta_difference_is_detected(monkeypatch):
    """The real case: Qwen Math (1e4) against a general base (1e6)."""
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    _patch_autoconfig(monkeypatch, {
        "Qwen/Qwen2.5-Math-1.5B": _Cfg(rope_theta=10000.0, max_position_embeddings=4096),
    })
    source = _ModelWithConfig(_Cfg(rope_theta=1000000.0, max_position_embeddings=131072))

    mismatch = _tuned_config_mismatch("Qwen/Qwen2.5-Math-1.5B", source)

    assert set(mismatch) == {"rope_theta", "max_position_embeddings"}
    assert mismatch["rope_theta"] == {"source": 1000000.0, "tuned": 10000.0}
    # Carried along for the record once something real already fired.
    assert mismatch["max_position_embeddings"] == {"source": 131072, "tuned": 4096}


def test_rope_normalized_into_rope_scaling_is_still_detected(monkeypatch):
    """transformers>=5 drops the flat rope_theta and nests it under rope_scaling.

    Measured on this install: Qwen2.5-1.5B reports rope_theta=<absent> and
    rope_scaling={'rope_theta': 1e6, 'rope_type': 'default'}. A guard watching only
    the field the JSON config names would compare None to None and wave the Math
    pairing through, silently running its weights under the wrong geometry.
    """
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    _patch_autoconfig(monkeypatch, {
        "Qwen/Qwen2.5-Math-1.5B": _Cfg(rope_scaling={"rope_theta": 10000, "rope_type": "default"}),
    })
    source = _ModelWithConfig(_Cfg(rope_scaling={"rope_theta": 1000000.0, "rope_type": "default"}))

    mismatch = _tuned_config_mismatch("Qwen/Qwen2.5-Math-1.5B", source)

    assert "rope_scaling" in mismatch, "normalized rope_theta went undetected"


def test_context_length_alone_is_not_a_mismatch(monkeypatch):
    """max_position_embeddings must NOT trigger a rebuild on its own.

    Real base/tuned pairs disagree about declared context length -- Qwen2.5-1.5B
    says 131072 and Qwen2.5-1.5B-Instruct says 32768 -- while being exactly the
    ordinary same-architecture pairing the deepcopy path is built for. Treating it
    as a mismatch would rebuild every instruct arm, and would make the guard REFUSE
    every existing theseus/bico config that pairs those two.
    """
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    _patch_autoconfig(monkeypatch, {
        "Qwen/Qwen2.5-1.5B-Instruct": _Cfg(rope_theta=1000000.0, max_position_embeddings=32768),
    })
    source = _ModelWithConfig(_Cfg(rope_theta=1000000.0, max_position_embeddings=131072))

    assert _tuned_config_mismatch("Qwen/Qwen2.5-1.5B-Instruct", source) == {}


def test_bare_state_dict_refs_are_never_treated_as_config_carrying(monkeypatch):
    """A .pt/.safetensors checkpoint is weights FOR the source architecture.

    It has no config of its own to disagree with, so it must keep the historical
    deepcopy path even if AutoConfig would happen to resolve the string.
    """
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    _patch_autoconfig(monkeypatch, {"x.safetensors": _Cfg(rope_theta=1.0)})
    source = _ModelWithConfig(_Cfg(rope_theta=1000000.0))

    for ref in ("x.pt", "x.bin", "x.safetensors", "x.ckpt", "x.pth"):
        assert _tuned_config_mismatch(ref, source) == {}


def test_unreadable_config_falls_back_to_the_deepcopy_path(monkeypatch):
    """A PEFT adapter dir or anything AutoConfig cannot read must not break.

    "No detectable mismatch" is the safe answer: it preserves every ref shape that
    worked before this check existed.
    """
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    _patch_autoconfig(monkeypatch, {})  # every lookup raises
    source = _ModelWithConfig(_Cfg(rope_theta=1000000.0))

    assert _tuned_config_mismatch("org/some-adapter", source) == {}


def test_config_without_model_type_is_ignored(monkeypatch):
    """AutoConfig can return something shapeless; only a real model config counts."""
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    _patch_autoconfig(monkeypatch, {"org/weird": _Cfg(model_type=None, rope_theta=1.0)})
    source = _ModelWithConfig(_Cfg(rope_theta=1000000.0))

    assert _tuned_config_mismatch("org/weird", source) == {}


def test_source_without_a_config_is_ignored(monkeypatch):
    """Bare nn.Modules (the test decoders here) have no .config to compare against."""
    from merge_and_rebase.eval.llm_rebase import _tuned_config_mismatch

    _patch_autoconfig(monkeypatch, {"org/tuned": _Cfg(rope_theta=10000.0)})

    assert _tuned_config_mismatch("org/tuned", _Decoder(depth=2)) == {}


def test_every_behavioural_key_is_shape_invisible():
    """Each watched key must be one a state-dict load CANNOT catch.

    That is the whole justification for the check: if a mismatch showed up as a shape
    error we would not need it. A key that changes parameter shapes does not belong
    here -- it would be caught anyway, and listing it would imply false coverage.
    """
    from merge_and_rebase.eval.llm_rebase import _BEHAVIOURAL_CONFIG_KEYS

    shape_bearing = {
        "hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
        "num_attention_heads", "num_key_value_heads",
    }
    assert not (set(_BEHAVIOURAL_CONFIG_KEYS) & shape_bearing)
    assert "rope_theta" in _BEHAVIOURAL_CONFIG_KEYS


# ---------------------------------------------------------------------------
# Padding mask: text batches are right-padded to a fixed length, and without
# mask_padding every pad position becomes a fitted row.
# ---------------------------------------------------------------------------

_LENGTHS = (3, 5, 2, 4, 5, 1)


class _PaddedTexts(_Texts):
    """Right-padded like the real calibration loader: example i has lengths[i] real tokens."""

    def __init__(self, *, offset=0, lengths=_LENGTHS):
        super().__init__(n=len(lengths), offset=offset)
        self.lengths = lengths

    def __getitem__(self, i):
        row = super().__getitem__(i)
        mask = torch.zeros(self.seq, dtype=torch.long)
        mask[: self.lengths[i]] = 1
        row["input_ids"] = row["input_ids"] * mask
        row["attention_mask"] = mask
        return row


def _padded_loader(offset=0, batch_size=2, lengths=_LENGTHS):
    return DataLoader(
        _PaddedTexts(offset=offset, lengths=lengths), batch_size=batch_size, shuffle=False, collate_fn=_collate
    )


def _padded_references(source, source_ft, target, config, adapter, *, target_lengths=_LENGTHS):
    return capture_residual_references(
        source, source_ft, target,
        _padded_loader(offset=0), _padded_loader(offset=100, lengths=target_lengths),
        num_batches=config.num_batches, seed=0, device="cpu",
        target_scope=config.target_scope, family_adapter=adapter, mask_padding=config.mask_padding,
    )


def test_mask_padding_defaults_off_and_is_validated():
    assert parse_residual_completion_config({"enabled": True}).mask_padding is False
    assert parse_residual_completion_config({"enabled": True, "mask_padding": True}).mask_padding is True
    with pytest.raises(TypeError, match="mask_padding must be bool"):
        parse_residual_completion_config({"enabled": True, "mask_padding": "yes"})


def test_capture_keeps_only_real_tokens_when_masked():
    model = _Decoder(depth=2)
    batches = list(_padded_loader(batch_size=3))
    unmasked = capture_tokens(model, batches, {"b": (1, "boundary")}, "cpu", family_adapter=_Adapter())
    masked = capture_tokens(
        model, batches, {"b": (1, "boundary")}, "cpu", family_adapter=_Adapter(), mask_padding=True
    )
    assert unmasked["b"][0].shape[:2] == (3, 5)
    assert masked["b"][0].shape[:2] == (1, sum(_LENGTHS[:3]))
    assert masked["b"][1].shape[:2] == (1, sum(_LENGTHS[3:]))
    # The surviving rows are exactly the real positions, in order.
    mask = batches[0]["attention_mask"].bool()
    torch.testing.assert_close(masked["b"][0][0], unmasked["b"][0][mask])


def test_masked_references_record_real_and_padded_row_counts():
    adapter = _Adapter()
    source, source_ft, target, _sd, _layout = _fixture()
    config = _config(mask_padding=True, num_batches=3)
    refs = _padded_references(source, source_ft, target, config, adapter)
    meta = refs["calibration"]
    assert meta["mask_padding"] is True
    assert meta["real_rows"] == sum(_LENGTHS)
    assert meta["padded_rows"] == len(_LENGTHS) * 5
    for bank in refs["target_base_outputs_by_position"].values():
        assert sum(b.shape[1] for b in bank) == sum(_LENGTHS)


def test_masked_references_refuse_mismatched_real_token_counts():
    """Two tokenizers splitting the text differently cannot be paired row for row."""
    adapter = _Adapter()
    source, source_ft, target, _sd, _layout = _fixture()
    config = _config(mask_padding=True, num_batches=3)
    shifted = (4, 5, 2, 4, 5, 1)
    with pytest.raises(ValueError, match="same real-token counts"):
        _padded_references(source, source_ft, target, config, adapter, target_lengths=shifted)


def test_masked_direct_completion_fits_on_real_rows_only():
    """The solver consumes the packed banks, and pad rows no longer move the fit."""
    adapter = _Adapter()
    results = {}
    for mask_padding in (False, True):
        source, source_ft, target, target_base_sd, layout = _fixture()
        config = _config(mask_padding=mask_padding, num_batches=3)
        materialize_missing_projection_biases(
            target, target_base_sd, layout, family_adapter=adapter, components=config.components
        )
        refs = _padded_references(source, source_ft, target, config, adapter)
        corrections, diagnostics = complete_residuals_direct(
            target, target_base_sd, refs, layout, _padded_loader(offset=100),
            config=config, device="cpu", family_adapter=adapter,
        )
        assert diagnostics and abs(diagnostics[0]["relative_residual_before"] - 1.0) < 1e-4
        for row in diagnostics:
            assert row["residual_norm_after"] < row["residual_norm_before"]
        results[mask_padding] = corrections
    key = next(iter(results[True]))
    assert not torch.allclose(results[True][key], results[False][key])


# --------------------------------------------------------------------------
# 5. cascade_order: "top_bottom" is not a third setting
# --------------------------------------------------------------------------


def _digest(corrections):
    sha = hashlib.sha256()
    for key in sorted(corrections):
        sha.update(key.encode())
        sha.update(corrections[key].detach().numpy().tobytes())
    return sha.hexdigest()


def _cascade_fixture():
    """The fixture these cascade pins were written against on the other branch.

    Two source blocks realized as four target positions, only the source
    down_proj weights perturbed, and the MLP bias already materialized.
    """
    torch.manual_seed(11)
    source = _Decoder(depth=2).eval()
    source_ft = _Decoder(depth=2).eval()
    source_ft.load_state_dict(source.state_dict())
    with torch.no_grad():
        source_ft.model.layers[0].mlp.down_proj.weight.add_(0.2)
        source_ft.model.layers[1].mlp.down_proj.weight.sub_(0.15)
    target = _Decoder(width=10, inter=20, depth=4).eval()
    data = _loader()
    config = ResidualCompletionConfig(
        enabled=True, mode="direct_target", target_scope="all", ridge_relative=0.05,
        num_batches=2, missing_bias="materialize",
    )
    references = capture_residual_references(
        source, source_ft, target, data, data,
        num_batches=config.num_batches, seed=0, device="cpu",
        target_scope="all", family_adapter=_Adapter(),
    )
    layout = {
        "final_blocks": tuple(
            {"position": p, "source_orig_idx": p // 2,
             "block_kind": "inserted" if p % 2 else "original"}
            for p in range(4)
        ),
        "inserted_blocks": (),
    }
    target_base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    added = materialize_missing_projection_biases(target, target_base_sd, layout, family_adapter=_Adapter())
    assert added, "the decoder fixture must need bias materialization, or this is not the real path"
    return config, references, layout, target, target_base_sd, data


def _fit_with_order(order):
    config, references, layout, target, base, data = _cascade_fixture()
    config = ResidualCompletionConfig(**{**config.__dict__, "cascade_order": order})
    corrections, diagnostics = complete_residuals_direct(
        target, base, references, layout, data,
        config=config, device="cpu", family_adapter=_Adapter(),
    )
    return _digest(corrections), {r["position"]: r["relative_residual_before"] for r in diagnostics}


def test_top_bottom_is_byte_identical_to_independent():
    """Reversing the visit order gives the uncoupled fit, not a different coupling.

    The option reads as though fitting deepest-first lets later fits invalidate
    what earlier ones assumed. It cannot: a correction mounted at block k changes
    activations only *above* k, and block j's capture -- its projection input and
    its own boundary -- depends only on blocks <= j. So deepest-first leaves every
    block measuring a pristine upstream, which is exactly what `independent`
    produces by never mounting at all.

    Pinned at hash level because the two settings look different in a config, and
    a campaign that grids both spends real compute reproducing one cell in
    another. A genuinely different coupling would need a re-measured second
    sweep, not a reversed order.
    """
    top_digest, top_before = _fit_with_order("top_bottom")
    ind_digest, ind_before = _fit_with_order("independent")
    assert top_digest == ind_digest
    for position, value in top_before.items():
        assert value == pytest.approx(1.0, abs=1e-4), position
        assert ind_before[position] == pytest.approx(1.0, abs=1e-4), position


def test_bottom_top_really_does_couple():
    """Negative control: the default order must not collapse to the uncoupled fit."""
    bottom_digest, bottom_before = _fit_with_order("bottom_top")
    ind_digest, _ = _fit_with_order("independent")
    assert bottom_digest != ind_digest, "the sequential cascade has stopped coupling anything"
    assert [p for p, v in bottom_before.items() if abs(v - 1.0) > 1e-4], "the cascade is inert"


# --------------------------------------------------------------------------
# 6. materialize_missing_projection_biases must cover every requested write
#    surface, not only mlp.c_proj
# --------------------------------------------------------------------------


def _fresh_target_and_layout():
    """A target model + layout with no bias materialized yet.

    Deliberately not `_cascade_fixture()`: that helper already materializes the MLP
    bias as part of its own setup (and asserts on it), so calling the
    materializer again on its output would find the key already present and
    report nothing added -- exactly the false pass that would have hidden
    this bug.
    """
    target = _Decoder(width=10, inter=20, depth=4).eval()
    base = {k: v.clone() for k, v in target.state_dict().items()}
    layout = {
        "final_blocks": tuple(
            {"position": p, "source_orig_idx": p // 2,
             "block_kind": "inserted" if p % 2 else "original"}
            for p in range(4)
        ),
        "inserted_blocks": (),
    }
    return target, base, layout


def test_bias_materialization_defaults_to_mlp_only():
    """The historical, single-write-surface behaviour must be unchanged."""
    target, base, layout = _fresh_target_and_layout()
    added = materialize_missing_projection_biases(target, base, layout, family_adapter=_Adapter())
    assert added and all("mlp.down_proj" in key for key in added)
    assert not any("self_attn" in key for key in added)


def test_bias_materialization_covers_attn_out_proj_when_requested():
    """Regression test: components=[attn.out_proj, mlp.c_proj] left o_proj bias-free.

    materialize_missing_projection_biases only ever materialized the MLP
    projection's bias. With both write surfaces enabled, completion tried to
    write an intercept onto `self_attn.o_proj.bias` and found it did not exist
    -- caught when a real campaign cell (components=["attn.out_proj",
    "mlp.c_proj"]) actually exercised the combination and raised
    "missing_bias='materialize' requires ... o_proj.bias to exist".
    """
    target, base, layout = _fresh_target_and_layout()
    added = materialize_missing_projection_biases(
        target, base, layout, family_adapter=_Adapter(),
        components=("attn.out_proj", "mlp.c_proj"),
    )
    down_proj_keys = {k for k in added if "mlp.down_proj" in k}
    o_proj_keys = {k for k in added if "self_attn.o_proj" in k}
    assert down_proj_keys, "the MLP write surface must still get its bias"
    assert o_proj_keys, "the attention write surface must also get its bias"
    for key in down_proj_keys | o_proj_keys:
        assert key in base and torch.count_nonzero(base[key]) == 0
