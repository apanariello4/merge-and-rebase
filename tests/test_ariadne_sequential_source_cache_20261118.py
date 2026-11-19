"""``cache_source_captures`` (decoder sequential fit with one model resident) equals the lockstep path.

Only the target changes from block to block, so the source base/FT boundaries and the pristine-target boundaries are
captured once (one model at a time) and replayed. Pinned on tiny random Qwen2 decoders on CPU (float64): task vectors
and rows equal the lockstep reference (rows gain only ``source_captures``), with and without ``desired_fn``, for
extension / reduction / same depth, a source that is the target object, and padded batches; no deep copy of the
target is made; the sources are never moved once the block loop starts; the default (False / vision) is unchanged.
"""

from __future__ import annotations

import pytest
import torch
from _llm_fixtures import perturbed_copy, tiny_qwen2
from test_ariadne_sequential_fast_20261008 import TEXTS, _setup, _sha, _text_loader

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne import streaming
from merge_and_rebase.rebase.methods._ariadne.config import resolve_ariadne_decoder_config
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_sequential_streaming,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.model_families import infer_family


def _desired(pos, i, b, f, out):
    # a position-dependent, non-linear stand-in for the anchor decode (same token grid, width mapped to the target's)
    proj = torch.linspace(-1.0, 1.0, b.shape[-1] * out.shape[-1], dtype=b.dtype).reshape(b.shape[-1], out.shape[-1])
    return torch.tanh(f - b) @ proj * (1.0 + 0.1 * pos)


def _run(source_layers, target_layers, *, use_desired, same_object=False, padding_side="right", extra=None):
    src, ft, tgt, fam, base_sd = _setup(source_layers, target_layers, dtype=torch.float64, missing_bias="skip",
                                        same_object=same_object)
    params = {"depth_fit": "sequential", "missing_bias": "skip", "exact_form": False, "num_batches": len(TEXTS) // 2, **(extra or {})}
    cfg = resolve_ariadne_decoder_config(params, fam)
    pairing = DiscreteLayerPairing.compute(source_layers, target_layers)
    loader = _text_loader(padding_side)
    prepared = prepare_direct_residual_streaming(
        src, tgt, loader, loader, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu", family_adapter=fam,
    )  # fmt: skip
    kw = {"desired_fn": _desired} if use_desired else {}
    args = (tgt, base_sd, src, ft, prepared, pairing)
    before = _sha(dict(tgt.state_dict()))
    ref = fit_direct_residual_sequential_streaming(*args, config=cfg, device="cpu", family_adapter=fam, **kw)
    new = fit_direct_residual_sequential_streaming(
        *args, config=cfg, device="cpu", family_adapter=fam, cache_source_captures=True, **kw
    )
    assert _sha(dict(tgt.state_dict())) == before
    return ref, new


def _assert_same(ref, new):
    (tau_ref, rows_ref), (tau_new, rows_new) = ref, new
    assert set(tau_ref) == set(tau_new)
    for key in tau_ref:
        assert torch.equal(tau_ref[key], tau_new[key]), key
    assert len(rows_ref) == len(rows_new)
    for a, b in zip(rows_ref, rows_new, strict=True):
        b = dict(b)
        assert b.pop("source_captures") == "host_precomputed"
        assert set(a) == set(b)
        for field in a:
            if isinstance(a[field], torch.Tensor):
                assert torch.equal(a[field], b[field]), field
            else:
                assert a[field] == b[field], field


@pytest.mark.parametrize("use_desired", [False, True], ids=["transported", "desired_fn"])
@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("layers", [(2, 3), (3, 2), (3, 3)], ids=["ext", "red", "same"])
def test_cache_equals_lockstep(layers, padding_side, use_desired):
    _assert_same(*_run(*layers, use_desired=use_desired, padding_side=padding_side))


def test_cache_equals_lockstep_source_is_target():
    _assert_same(*_run(3, 3, use_desired=False, same_object=True))


def test_cache_off_target_and_strength():
    extra = {"sequential_subtract": "off_target", "sequential_strength": 0.7}
    _assert_same(*_run(2, 3, use_desired=True, extra=extra))


def test_no_deepcopy_and_sources_not_moved_in_block_loop(monkeypatch):
    src, ft, tgt, fam, base_sd = _setup(2, 3, dtype=torch.float64, missing_bias="skip")
    cfg = resolve_ariadne_decoder_config(
        {"depth_fit": "sequential", "missing_bias": "skip", "exact_form": False, "num_batches": len(TEXTS) // 2}, fam
    )
    pairing = DiscreteLayerPairing.compute(2, 3)
    loader = _text_loader("right")
    prepared = prepare_direct_residual_streaming(
        src, tgt, loader, loader, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu", family_adapter=fam,
    )  # fmt: skip

    def boom(*a, **k):
        raise AssertionError("target must not be deep-copied")

    monkeypatch.setattr(streaming, "deepcopy", boom)
    moved = []
    at_call = []
    state = {"in_loop": False}
    orig_to = torch.nn.Module.to

    def spy(self, *a, **k):
        if state["in_loop"] and self in (src, ft):
            moved.append("source")
        return orig_to(self, *a, **k)

    monkeypatch.setattr(torch.nn.Module, "to", spy)

    def desired(pos, i, b, f, out):
        state["in_loop"] = True
        at_call.append(len(moved))
        return torch.zeros_like(out)

    fit_direct_residual_sequential_streaming(
        tgt, base_sd, src, ft, prepared, pairing, config=cfg, device="cpu", family_adapter=fam,
        desired_fn=desired, cache_source_captures=True,
    )  # fmt: skip
    # no source `.to` between the first and the last block-loop call (the closing restore is a no-op on the CPU)
    assert at_call and len(set(at_call)) == 1


def test_default_unchanged_and_vision_rejected():
    (tau_a, rows_a), (_, _) = _run(2, 3, use_desired=False)
    assert all("source_captures" not in r for r in rows_a)  # lockstep rows carry no new field
    src = tiny_qwen2(layers=2, hidden=16, inter=48, seed=0)
    with pytest.raises(ValueError, match="decoder-only"):
        fit_direct_residual_sequential_streaming(
            src, {}, src, perturbed_copy(src, scale=0.05, seed=1), {}, DiscreteLayerPairing.compute(2, 2),
            config=resolve_ariadne_decoder_config({"depth_fit": "sequential"}, infer_family(src)),
            device="cpu", cache_source_captures=True,
        )  # fmt: skip


@pytest.mark.parametrize("cache", [False, True])
def test_sequential_fit_never_mutates_the_base_state(cache):
    """``current_state`` shares CPU tensors with ``target_base_state`` (no full clone): the fit must not write them."""
    src, ft, tgt, fam, base_sd = _setup(2, 3, dtype=torch.float64, missing_bias="skip")
    cfg = resolve_ariadne_decoder_config(
        {"depth_fit": "sequential", "missing_bias": "skip", "exact_form": False, "num_batches": len(TEXTS) // 2}, fam
    )
    pairing = DiscreteLayerPairing.compute(2, 3)
    loader = _text_loader("right")
    prepared = prepare_direct_residual_streaming(
        src, tgt, loader, loader, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu", family_adapter=fam,
    )  # fmt: skip
    before = {k: v.clone() for k, v in base_sd.items()}
    tau, _ = fit_direct_residual_sequential_streaming(
        tgt, base_sd, src, ft, prepared, pairing, config=cfg, device="cpu", family_adapter=fam,
        cache_source_captures=cache,
    )
    assert tau
    assert all(torch.equal(before[k], base_sd[k]) for k in before)
