"""The transport-free Proposal-1 arm on the decoder/LLM path.

`llm_rebase` wired only the transport-aware completion: `complete_residuals`
never inspects `config.mode`, and the only mode guard lives inside
`complete_residuals_direct`, which that path did not import. A
`mode="direct_target"` config therefore passed validation and silently ran the
transport-aware solve, reporting a plausible number for a different method.

These tests pin the decoder plumbing of the direct arm. The solver itself is
already covered by `tests/test_direct_target_p1.py` and is architecture-
agnostic, so what needed new coverage is: the decoder activation capture the
two-component path depends on, and the LLM-only contract around the transported
delta (empty in direct mode) and the `passthrough` remainder that vision has no
analogue for.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.eval.llm_rebase import (
    _maybe_complete_target_residual_task_vector_direct,
)
from merge_and_rebase.eval.target_informed_runtime import (
    capture_residual_references,
    capture_tokens,
    complete_residuals_direct,
    materialize_missing_projection_biases,
)
from merge_and_rebase.eval.target_residual_completion import (
    ResidualCompletionConfig,
    parse_residual_completion_config,
)

VOCAB = 32


class _Attn(nn.Module):
    """`o_proj` is a real `nn.Linear` submodule, unlike CLIP's fused MHA.

    That is the whole reason the decoder needs none of the vision path's
    `out_proj` recompute machinery: a forward hook on this module actually
    fires. These tests exist to prove that rather than assume it.
    """

    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        q, v = self.q_proj(x), self.v_proj(x)
        weights = torch.softmax(q @ q.transpose(-2, -1) * q.shape[-1] ** -0.5, dim=-1)
        return self.o_proj(weights @ v)


class _MLP(nn.Module):
    def __init__(self, width, inter):
        super().__init__()
        self.gate_proj = nn.Linear(width, inter, bias=False)
        self.up_proj = nn.Linear(width, inter, bias=False)
        self.down_proj = nn.Linear(inter, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, width, inter):
        super().__init__()
        self.self_attn = _Attn(width)
        self.mlp = _MLP(width, inter)

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


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
    def __init__(self, width=8, inter=16, depth=3, vocab=VOCAB):
        super().__init__()
        self.model = _Body(width, inter, depth, vocab)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)


class _Adapter:
    def transport_scope(self, model):
        return model.model

    def block_count(self, model):
        return len(model.model.layers)

    def extract_calibration_batch(self, batch):
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


class _TextDataset(Dataset):
    def __init__(self, n=8, seq=5):
        generator = torch.Generator().manual_seed(4)
        self.rows = [torch.randint(0, VOCAB, (seq,), generator=generator) for _ in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _collate(rows):
    ids = torch.stack(rows)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _loader():
    return DataLoader(_TextDataset(), batch_size=2, shuffle=False, collate_fn=_collate)


def _batches(n=2, bs=2, seq=5):
    generator = torch.Generator().manual_seed(1)
    return [
        {
            "input_ids": torch.randint(0, VOCAB, (bs, seq), generator=generator),
            "attention_mask": torch.ones(bs, seq, dtype=torch.long),
        }
        for _ in range(n)
    ]


# --------------------------------------------------------------------------
# 1. Decoder activation capture for the second write surface
# --------------------------------------------------------------------------


def test_attn_proj_input_hook_fires_once_per_batch_on_a_decoder():
    model = _Decoder().eval()
    batches = _batches()
    out = capture_tokens(
        model, batches, {"h": (1, "attn_proj_input"), "o": (1, "attn_proj")}, "cpu",
        family_adapter=_Adapter(),
    )
    assert set(out) == {"h", "o"}
    for values in out.values():
        assert len(values) == len(batches)


def test_captured_attn_rows_reproduce_the_o_proj_output():
    """The captured features must be the ones `o_proj` actually consumes.

    A hook that fires but on the wrong tensor would give a silently wrong
    design matrix, which is exactly the failure the vision path had to work
    around for its fused attention.
    """
    model = _Decoder().eval()
    batches = _batches()
    out = capture_tokens(
        model, batches, {"h": (1, "attn_proj_input"), "o": (1, "attn_proj")}, "cpu",
        family_adapter=_Adapter(),
    )
    o_proj = model.model.layers[1].self_attn.o_proj
    for rows, value in zip(out["h"], out["o"], strict=True):
        replayed = torch.nn.functional.linear(rows, o_proj.weight.detach(), None)
        torch.testing.assert_close(replayed, value, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# 2. Config surface
# --------------------------------------------------------------------------


def test_direct_passthrough_defaults_off_and_requires_direct_mode():
    assert ResidualCompletionConfig().direct_passthrough is False
    assert parse_residual_completion_config({"enabled": True}).direct_passthrough is False
    cfg = parse_residual_completion_config(
        {"enabled": True, "mode": "direct_target", "direct_passthrough": True}
    )
    assert cfg.direct_passthrough is True
    with pytest.raises(ValueError, match="direct_passthrough=true requires"):
        parse_residual_completion_config({"enabled": True, "direct_passthrough": True})


# --------------------------------------------------------------------------
# 3. The direct completion helper on a decoder
# --------------------------------------------------------------------------


def _fixture(strength=1.0, direct_passthrough=False):
    torch.manual_seed(11)
    source = _Decoder(depth=2).eval()
    source_ft = _Decoder(depth=2).eval()
    source_ft.load_state_dict(source.state_dict())
    with torch.no_grad():
        source_ft.model.layers[0].mlp.down_proj.weight.add_(0.2)
        source_ft.model.layers[1].mlp.down_proj.weight.sub_(0.15)
    target = _Decoder(width=10, inter=20, depth=4).eval()
    data = _loader()
    # Decoder MLP projections are bias-free, so the exact form's intercept has
    # nowhere to land until a zero bias is materialized -- in the model and the
    # base state together, exactly as llm_rebase does before completion runs.
    config = ResidualCompletionConfig(
        enabled=True, mode="direct_target", target_scope="all", ridge_relative=0.05,
        num_batches=2, strength=strength, direct_passthrough=direct_passthrough,
        missing_bias="materialize",
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
    added = materialize_missing_projection_biases(
        target, target_base_sd, layout, family_adapter=_Adapter()
    )
    assert added, "the decoder fixture must need bias materialization, or this is not the real path"
    return config, references, layout, target, target_base_sd, data


def _run(config, references, layout, target, target_base_sd, data, *, passthrough=None):
    return _maybe_complete_target_residual_task_vector_direct(
        config=config, references=references, layout=layout, target_model=target,
        target_base_sd=target_base_sd, transported_delta={},
        passthrough_delta=passthrough or {}, target_loader=data,
        family_adapter=_Adapter(), device="cpu",
    )


def test_direct_arm_writes_only_down_proj_keys_and_no_transported_keys():
    config, references, layout, target, base, data = _fixture()
    delta, diagnostics, skipped = _run(config, references, layout, target, base, data)
    assert diagnostics is not None and len(diagnostics) == 4
    assert skipped == []
    assert delta, "the direct arm must produce a task vector of its own"
    assert all("mlp.down_proj" in key for key in delta), sorted(delta)
    # The materialized intercept must actually reach the task vector.
    assert any(key.endswith(".bias") for key in delta)
    # Nothing that a transport would have produced may appear.
    assert not any("self_attn" in key or "embed_tokens" in key for key in delta)


def test_first_fitted_block_reports_an_untouched_base():
    config, references, layout, target, base, data = _fixture()
    _delta, diagnostics, _ = _run(config, references, layout, target, base, data)
    assert diagnostics[0]["relative_residual_before"] == pytest.approx(1.0, abs=1e-4)
    assert diagnostics[0]["effect_before_norm"] == pytest.approx(0.0, abs=1e-6)
    assert all(row["mode"] == "direct_target" for row in diagnostics)


def test_gamma_zero_is_an_exact_native_target_base_control():
    config, references, layout, target, base, data = _fixture(strength=0.0)
    delta, diagnostics, _ = _run(config, references, layout, target, base, data)
    assert diagnostics is not None and len(diagnostics) == 4, "the fit still runs at gamma=0"
    assert delta, "the zero task vector still carries its keys"
    for value in delta.values():
        assert torch.count_nonzero(value) == 0


def test_a_non_empty_transported_delta_is_refused():
    config, references, layout, target, base, data = _fixture()
    with pytest.raises(ValueError, match="requires an empty transported task vector"):
        _maybe_complete_target_residual_task_vector_direct(
            config=config, references=references, layout=layout, target_model=target,
            target_base_sd=base, transported_delta={"model.layers.0.mlp.down_proj.weight": torch.zeros(10, 20)},
            passthrough_delta={}, target_loader=data, family_adapter=_Adapter(), device="cpu",
        )


def test_transport_residual_mode_is_refused_by_the_direct_helper():
    config, references, layout, target, base, data = _fixture()
    config = ResidualCompletionConfig(**{**config.__dict__, "mode": "transport_residual"})
    with pytest.raises(ValueError, match="requires mode='direct_target'"):
        _run(config, references, layout, target, base, data)


# --------------------------------------------------------------------------
# 4. The passthrough decision
# --------------------------------------------------------------------------


def _passthrough(base):
    """A shape-compatible key and a shape-incompatible one."""
    compatible = "model.embed_tokens.weight"
    return {
        compatible: torch.ones_like(base[compatible]),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(3, 3),
    }


def test_passthrough_is_dropped_by_default():
    config, references, layout, target, base, data = _fixture()
    delta, _diag, skipped = _run(
        config, references, layout, target, base, data, passthrough=_passthrough(base)
    )
    assert "model.embed_tokens.weight" not in delta
    assert skipped == [], "nothing is even considered when the flag is off"


def test_passthrough_is_carried_when_the_flag_is_set():
    config, references, layout, target, base, data = _fixture(direct_passthrough=True)
    delta, _diag, skipped = _run(
        config, references, layout, target, base, data, passthrough=_passthrough(base)
    )
    assert "model.embed_tokens.weight" in delta
    torch.testing.assert_close(delta["model.embed_tokens.weight"], torch.ones_like(base["model.embed_tokens.weight"]))
    # The shape-incompatible key is reported, not silently swallowed.
    assert skipped == ["model.layers.0.self_attn.q_proj.weight"]


def test_disabled_completion_returns_the_delta_untouched():
    config, references, layout, target, base, data = _fixture()
    config = ResidualCompletionConfig(**{**config.__dict__, "enabled": False})
    delta, diagnostics, skipped = _run(config, references, layout, target, base, data)
    assert delta == {} and diagnostics is None and skipped == []


# --------------------------------------------------------------------------
# 5. cascade_order: "top_bottom" is not a third setting
# --------------------------------------------------------------------------


def _digest(corrections):
    sha = hashlib.sha256()
    for key in sorted(corrections):
        sha.update(key.encode())
        sha.update(corrections[key].detach().numpy().tobytes())
    return sha.hexdigest()


def _fit_with_order(order):
    config, references, layout, target, base, data = _fixture()
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

    Deliberately not `_fixture()`: that helper already materializes the MLP
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
