"""Host-RAM cuts of the LLM direct fit, both bit-identical: the streamed source task vector (norms only) and the
shared fp32 base tensors of a round trip (source == target)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import tiny_llama

from merge_and_rebase.eval.llm_rebase.context import target_base_state_dict
from merge_and_rebase.eval.llm_rebase.merge import _delta_norm, norm_match_transported
from merge_and_rebase.eval.llm_rebase.stages import NoPrestep, _delta_is_norm_only
from merge_and_rebase.merge.runtime import to_cpu_fp32
from merge_and_rebase.merge.task_vectors import TaskVector, streamed_delta_from_checkpoints
from merge_and_rebase.models.text_lm import TextBuildConfig
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.model_families import infer_family


def _sds(seed=0):
    g = torch.Generator().manual_seed(seed)
    base = {f"w{i}": torch.randn(7 + i, 5, generator=g) for i in range(6)}
    base["ids"] = torch.arange(4)  # non-float: filtered
    tuned = {k: (v + 0.01 * torch.randn(v.shape, generator=g) if v.is_floating_point() else v) for k, v in base.items()}
    return base, tuned


def test_streamed_delta_norms_are_bit_identical():
    base, tuned = _sds()
    full = TaskVector.from_checkpoints(base, tuned, strict=False).delta
    st = streamed_delta_from_checkpoints(base, tuned, strict=False)
    assert list(st) == list(full) and len(st) == len(full)
    assert _delta_norm(st) == _delta_norm(full)
    keys = {"w1", "w3"}
    assert _delta_norm(st, keys) == _delta_norm(full, keys)
    transported = {"w0": torch.ones(3)}
    out_f, n_f = norm_match_transported(transported, corrected_delta=full, reference_delta=full,
                                        transport_keys=None, norm_match=None)
    out_s, n_s = norm_match_transported(transported, corrected_delta=st or {}, reference_delta=st or {},
                                        transport_keys=None, norm_match=None)
    assert n_f == n_s and out_f.keys() == out_s.keys()


def test_streamed_delta_keeps_only_requested_and_fails_loudly():
    base, tuned = _sds(1)
    full = TaskVector.from_checkpoints(base, tuned, strict=False).delta
    st = streamed_delta_from_checkpoints(base, tuned, keep=["w2"])
    assert torch.equal(st["w2"], full["w2"])
    with pytest.raises(KeyError):
        st["w0"]
    with pytest.raises(KeyError):
        list(st.items())


@pytest.mark.parametrize("direct,copy,expected", [(True, False, True), (True, True, False), (False, False, False)])
def test_delta_is_norm_only_gate(direct, copy, expected):
    env = SimpleNamespace(resolved=SimpleNamespace(direct_fit=direct,
                                                   ariadne_cfg=SimpleNamespace(copy_shape_matching_source_deltas=copy)))
    assert _delta_is_norm_only(env) is expected
    assert _delta_is_norm_only(SimpleNamespace(resolved=SimpleNamespace(direct_fit=True, ariadne_cfg=None))) is False


def test_noprestep_direct_fit_norms_equal_full_vector():
    src = tiny_llama(layers=2, seed=0)
    base = to_cpu_fp32(src.state_dict())
    tuned = {k: v + 0.02 * torch.randn(v.shape, generator=torch.Generator().manual_seed(len(k))) for k, v in base.items()}
    rt = SimpleNamespace(source_llm=SimpleNamespace(model=src), source_base_sd=base, full_fp_keys=None,
                         load_tuned=lambda **kw: tuned, source_build_cfg=None, tp_keys=())
    task = SimpleNamespace(ctx=SimpleNamespace(ckpt_ref="x"))

    def run(direct):
        env = SimpleNamespace(runtime=rt, resolved=SimpleNamespace(
            direct_fit=direct, ariadne_cfg=SimpleNamespace(copy_shape_matching_source_deltas=False)))
        return NoPrestep().run(env, task, None).task_delta

    full, st = run(False), run(True)
    assert not hasattr(full, "sq_norms") and hasattr(st, "sq_norms")
    assert _delta_norm(full) == _delta_norm(st)
    assert list(full) == list(st)


def _cfg(**kw):
    # device "cuda" is only a config value here (no model is built); sharing is disabled for CPU-built models
    kw = {"device": "cuda", **kw}
    return TextBuildConfig(model_name_or_path="org/m", dtype=None, **kw)


def test_round_trip_shares_tensors_and_equals_independent_copy():
    a, b = tiny_llama(layers=2, seed=3), tiny_llama(layers=2, seed=3)
    src_sd = to_cpu_fp32(a.state_dict())
    ref = to_cpu_fp32(b.state_dict())
    tgt_sd = target_base_state_dict(_cfg(), _cfg(), src_sd, b)
    assert tgt_sd is not src_sd and list(tgt_sd) == list(ref)
    assert all(torch.equal(tgt_sd[k], ref[k]) for k in ref)
    assert all(tgt_sd[k] is src_sd[k] for k in ref)  # same tensors, new dict
    # missing_bias='materialize' adds keys to the target's dict only
    fam = infer_family(b)
    added = materialize_missing_projection_biases(b, tgt_sd, family_adapter=fam, components=("mlp.c_proj",))
    assert added and all(k in tgt_sd and k not in src_sd for k in added)
    assert set(src_sd) == set(ref)
    assert all(torch.equal(src_sd[k], ref[k]) for k in ref)


def test_no_sharing_for_different_cfg_or_base_ckpt():
    a, b = tiny_llama(layers=2, seed=3), tiny_llama(layers=2, seed=3)
    src_sd = to_cpu_fp32(a.state_dict())
    for tgt_cfg, ck in ((TextBuildConfig(model_name_or_path="org/other", device="cpu"), (None, None)),
                        (_cfg(), ("a.pt", None)), (_cfg(), (None, "b.pt"))):
        out = target_base_state_dict(_cfg(), tgt_cfg, src_sd, b, base_ckpts=ck)
        assert all(out[k] is not src_sd[k] for k in src_sd)
        assert all(torch.equal(out[k], src_sd[k]) for k in src_sd)
    # identical configs but CPU-built models or a sequence-classification head: independent copy
    for same in (_cfg(device="cpu"), _cfg(model_kind="sequence_classification")):
        out = target_base_state_dict(same, same, src_sd, b)
        assert all(out[k] is not src_sd[k] for k in src_sd)
        assert all(torch.equal(out[k], src_sd[k]) for k in src_sd)
