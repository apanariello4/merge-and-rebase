"""The transport-free Proposal-1 arm on the HF-decoder path.

``complete_residuals_direct`` and its solver are already pinned on the vision
side (``tests/test_direct_target_p1.py``,
``tests/test_direct_p1_trajectory_and_components.py``).  What was never covered
is the *decoder plumbing* it has to run through: ``self_attn.o_proj`` as a real
``nn.Linear`` write surface, the bias-free projections a materialize pre-pass
has to fill in, and ``llm_rebase``'s own mode branch, which must reach
completion with an empty transported task vector or the arm is not
transport-free at all.

The models here are deliberately tiny stand-ins with Qwen's structural
properties -- bias-free ``mlp.down_proj`` and ``self_attn.o_proj``, biased
q/k/v, no LayerScale -- and a genuine width-up, depth-up rebase (8 -> 12 wide,
3 -> 4 deep), because that is the shape of the pair the campaign runs.
"""

from __future__ import annotations

import math

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
from merge_and_rebase.eval.target_residual_completion import ResidualCompletionConfig

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
