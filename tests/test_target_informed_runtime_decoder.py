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
