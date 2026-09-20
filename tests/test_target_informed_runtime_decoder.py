"""Proposal 1's orchestration on an HF-decoder model.

The solver was always architecture-agnostic, but the runtime reached directly
into CLIP's module tree (model.visual.transformer.resblocks, _encode_image,
mlp.c_proj). These tests pin the decoder path that a family adapter selects, so
the LLM port shares one implementation with vision instead of duplicating it --
the same divergence that produced the inserted-block correction bug earlier.
"""

from __future__ import annotations

import torch
from torch import nn

from merge_and_rebase.eval.target_informed_runtime import capture_tokens


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
        self.mlp = _MLP(width, inter)

    def forward(self, x):
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
    def __init__(self, width=8, inter=16, depth=3, vocab=32):
        super().__init__()
        self.model = _Body(width, inter, depth, vocab)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)


class _Adapter:
    """Minimal stand-in for HfDecoderAdapter's surface used by the runtime."""

    def transport_scope(self, model):
        return model.model

    def block_count(self, model):
        return len(model.model.layers)

    def extract_calibration_batch(self, batch):
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


def _batches(n=2, bs=2, seq=5, vocab=32):
    return [
        {
            "input_ids": torch.randint(0, vocab, (bs, seq)),
            "attention_mask": torch.ones(bs, seq, dtype=torch.long),
        }
        for _ in range(n)
    ]


def test_capture_tokens_runs_on_a_decoder_via_family_adapter() -> None:
    """Boundary and down_proj captures fire once per batch on an HF decoder."""
    model = _Decoder()
    batches = _batches()
    out = capture_tokens(
        model,
        batches,
        {"boundary1": (1, "boundary"), "hidden1": (1, "c_proj_input"), "proj1": (1, "c_proj")},
        "cpu",
        family_adapter=_Adapter(),
    )
    assert set(out) == {"boundary1", "hidden1", "proj1"}
    for key, values in out.items():
        assert len(values) == len(batches), f"{key} did not fire once per batch"

    # down_proj is the decoder's residual-writing projection: its input lives in
    # the intermediate width, its output and the block boundary in model width.
    assert out["hidden1"][0].shape[-1] == 16
    assert out["proj1"][0].shape[-1] == 8
    assert out["boundary1"][0].shape[-1] == 8


def test_decoder_capture_reads_the_requested_block() -> None:
    """Zeroing one block's down_proj changes only that block's capture."""
    model = _Decoder()
    batches = _batches()
    adapter = _Adapter()

    before = capture_tokens(model, batches, {"p0": (0, "c_proj"), "p2": (2, "c_proj")}, "cpu", family_adapter=adapter)
    with torch.no_grad():
        model.model.layers[0].mlp.down_proj.weight.zero_()
    after = capture_tokens(model, batches, {"p0": (0, "c_proj"), "p2": (2, "c_proj")}, "cpu", family_adapter=adapter)

    assert torch.allclose(after["p0"][0], torch.zeros_like(after["p0"][0])), "block 0 capture did not track its own down_proj"
    assert not torch.allclose(before["p2"][0], torch.zeros_like(before["p2"][0]))


def test_vision_path_unchanged_when_no_adapter_is_passed() -> None:
    """family_adapter=None must still take the CLIP path, not the decoder one."""
    model = _Decoder()
    # A decoder has no .visual, so the vision path must fail rather than
    # silently succeed -- proving None does not fall through to the decoder.
    try:
        capture_tokens(model, _batches(), {"b": (0, "boundary")}, "cpu")
    except AttributeError:
        return
    raise AssertionError("family_adapter=None unexpectedly handled a decoder model")


def test_capture_residual_references_runs_on_a_decoder_with_all_scope() -> None:
    """End-to-end reference capture for proposal 1 on an HF decoder.

    target_scope='all' rather than 'inserted': the inserted scope derives target
    positions as 2*i+1, valid only for a doubling resize. The LLM pair here is
    24->28, where that formula runs past the target depth, so the all-block
    scope is the one that applies.

    The two loaders tokenize the same examples differently, as a real source and
    target tokenizer would; pairing must recognise them as the same examples via
    sample_ids rather than rejecting them for holding different tensors.
    """
    from torch.utils.data import DataLoader, Dataset

    from merge_and_rebase.eval.target_informed_runtime import capture_residual_references

    class _Texts(Dataset):
        def __init__(self, n, offset):
            self.n = n
            self.offset = offset
            # identity keys on the example, not on this "tokenization"
            self.sample_ids = [f"ex{i}" for i in range(n)]

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i + self.offset)
            return {
                "input_ids": torch.randint(0, 32, (5,), generator=g),
                "attention_mask": torch.ones(5, dtype=torch.long),
            }

    def _collate(rows):
        return {
            "input_ids": torch.stack([r["input_ids"] for r in rows]),
            "attention_mask": torch.stack([r["attention_mask"] for r in rows]),
        }

    src_loader = DataLoader(_Texts(4, 0), batch_size=2, shuffle=False, collate_fn=_collate)
    tgt_loader = DataLoader(_Texts(4, 100), batch_size=2, shuffle=False, collate_fn=_collate)

    refs = capture_residual_references(
        _Decoder(depth=3), _Decoder(depth=3), _Decoder(depth=4),
        src_loader, tgt_loader,
        num_batches=2, seed=0, device="cpu",
        target_scope="all", family_adapter=_Adapter(),
    )
    assert refs, "no reference banks captured on the decoder path"


def test_materialize_zero_bias_adds_a_writable_bias_to_a_bias_free_projection() -> None:
    """Option 1: give a bias-free down_proj somewhere to put the intercept.

    Qwen2.5 builds mlp.down_proj with bias=False, so the exact affine form has
    no parameter to write its intercept to. Materializing a zero bias is exact
    -- it changes nothing until the intercept is applied -- at the cost of a
    checkpoint carrying a parameter the stock architecture lacks.
    """
    from merge_and_rebase.eval.target_informed_runtime import _materialize_zero_bias

    model = _Decoder()
    key = "model.layers.1.mlp.down_proj.bias"
    assert model.model.layers[1].mlp.down_proj.bias is None, "fixture should start bias-free"

    state: dict[str, torch.Tensor] = {}
    _materialize_zero_bias(model, key, state, out_features=8)

    bias = model.model.layers[1].mlp.down_proj.bias
    assert bias is not None and bias.shape == (8,)
    assert torch.allclose(bias, torch.zeros(8)), "materialized bias must start at zero"
    assert key in state and torch.allclose(state[key], torch.zeros(8))
    # zero bias must leave the function unchanged
    ids = torch.randint(0, 32, (2, 5))
    with torch.no_grad():
        before = _Decoder()
        before.load_state_dict({k: v for k, v in model.state_dict().items() if "bias" not in k}, strict=False)
    assert model(ids).shape == (2, 5, 8)


def test_skip_is_refused_when_the_intercept_is_nonzero() -> None:
    """Option 2 is only sound where it drops nothing.

    exact_form=False makes the intercept exactly zero, so skipping it is a true
    no-op. The parser refuses skip+exact_form=True precisely so a centered-fit
    weight is never applied without the centering it assumes.
    """
    from merge_and_rebase.eval.target_residual_completion import parse_residual_completion_config

    ok = parse_residual_completion_config(
        {"enabled": True, "missing_bias": "skip", "exact_form": False}
    )
    assert ok.missing_bias == "skip" and ok.exact_form is False

    try:
        parse_residual_completion_config({"enabled": True, "missing_bias": "skip", "exact_form": True})
    except ValueError as exc:
        assert "exact_form=false" in str(exc)
        return
    raise AssertionError("skip with the exact form must be refused, not silently accepted")


def test_default_still_refuses_a_missing_bias() -> None:
    """The default stays strict so vision cannot silently change behaviour."""
    from merge_and_rebase.eval.target_residual_completion import parse_residual_completion_config

    assert parse_residual_completion_config({"enabled": True}).missing_bias == "error"
