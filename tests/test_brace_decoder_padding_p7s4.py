"""S4b: BRACE decoder correction drops padding rows unconditionally (decision D-P7b).

References and per-step captures follow attention_mask only, so the correction is invariant to the
amount of padding (max_length) and the vision BRACE path is untouched.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, tiny_qwen2

from merge_and_rebase.eval.llm_rebase import _build_text_calibration_loader
from merge_and_rebase.rebase.block_extension.decoder import DecoderBlockExtender
from merge_and_rebase.rebase.model_families import infer_family


def _pair(layers=2):
    base = tiny_qwen2(layers=layers, seed=21)
    ft = deepcopy(base)
    gen = torch.Generator().manual_seed(22)
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
    return base, ft


def _loader(side, max_length):
    tok = local_tokenizer(CALIB_TEXTS, side)
    return _build_text_calibration_loader(tokenizer=tok, texts=CALIB_TEXTS, batch_size=2, max_length=max_length)


def _extender(base, ft):
    return DecoderBlockExtender(base, ft, infer_family(base), device="cpu", verbose=False, show_progress=False)


@pytest.mark.parametrize("side", ["right", "left"])
def test_references_hold_content_rows_only(side):
    n_content = sum(len(t.split()) for t in CALIB_TEXTS)
    for max_length in (12, 24):
        base, ft = _pair()
        ext = _extender(base, ft)
        loader = _loader(side, max_length)
        ext.capture_reference_inputs(loader, 3)
        ext._capture_component_references(loader, 3)
        for name in ("base", "ft"):
            refs = ext.reference_inputs[name]
            assert refs
            for key, rows in refs.items():
                assert rows.shape[0] == n_content, (name, key, max_length)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("direction", ["extend", "shrink"])
def test_correction_is_invariant_to_max_length(side, direction):
    source_depth, target_depth = (2, 4) if direction == "extend" else (4, 2)
    states = {}
    for max_length in (12, 24):
        base, ft = _pair(source_depth)
        ext = _extender(base, ft)
        depth = ext.extend_and_calibrate(
            loader=_loader(side, max_length),
            n_batches=3,
            strategy="interpolate_per_weight",
            target_layers_total=target_depth,
            insertion_order="bottom-top",
            extension_density="spread",
            skip_correction=False,
            # The tiny fixture has fewer content rows than hidden units, so the unregularised fit is ill-conditioned
            # and amplifies float noise; a ridge prior makes the (mathematically identical) fits comparable.
            ridge_identity=10.0,
        )
        assert depth == target_depth
        states[max_length] = (base.state_dict(), ft.state_dict())
    for which in (0, 1):
        a, b = states[12][which], states[24][which]
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=2e-3, atol=2e-5, msg=lambda m, k=key: f"{k}: {m}")


# S4b regenerates only brace_decoder_* / brace_x_decoder_* / llm_rebase_* pins. This guards the vision BRACE
# pins: sha256 over the sorted "key hash" lines of every brace_vision* / brace_x_vision* entry in the two golden
# tables, computed on the parent commit (246 entries).
_VISION_BRACE_PIN_COUNT = 246
_VISION_BRACE_PIN_DIGEST = "88d1c38b26e350ed10a7df73b802d906f59c5182b37908abcba706b84f69a81b"


def test_vision_brace_hashes_are_frozen():
    golden = Path(__file__).parent / "golden"
    items = {}
    for name in ("test_brace_extra_golden.py", "test_release_golden_hashes.py"):
        for line in (golden / name).read_text().splitlines():
            m = re.match(r'\s+"(brace_(?:x_)?vision[^"]*)": "([0-9a-f]{64})",', line)
            if m:
                items[m.group(1)] = m.group(2)
    assert len(items) == _VISION_BRACE_PIN_COUNT
    digest = hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(items.items())).encode()).hexdigest()
    assert digest == _VISION_BRACE_PIN_DIGEST
