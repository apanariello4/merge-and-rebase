"""Characterization of the THESEUS/BiCo hybrid path in ``llm_rebase.main`` for tied embeddings and Qwen3 QK-norm.

Hybrid = transport the family adapter's transportable body keys, identity-pass everything else
(`llm_rebase.main`, branch ``method_name in ("theseus", "theseus_gqa", "bico")``). The passthrough rule is
inline in ``main()``; ``_hybrid_passthrough`` below mirrors it line for line (copy only keys whose shape equals
the target's) so that rule is pinned until S10 extracts it, at which point this helper should call the real one.

Pinned facts (Phase 7 design finding 8):
- ``q_norm``/``k_norm`` are not in the decoder transportable suffixes -> never transported;
- they are passed through only when a same-named target key exists with the same shape;
- a Qwen2 source has no q_norm, so Qwen2 -> Qwen3 produces no q_norm delta at all;
- embeddings / lm_head (tied) are excluded from transport and pass through only on matching shape.
"""

from __future__ import annotations

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_qwen2, tiny_qwen3

import merge_and_rebase.rebase.methods  # noqa: F401  (registers methods)
from merge_and_rebase.eval.llm_rebase import _build_text_calibration_loader
from merge_and_rebase.merge.runtime import to_cpu_fp32
from merge_and_rebase.merge.task_vectors import TaskVector
from merge_and_rebase.models.grad_recipes import causal_lm_recipe
from merge_and_rebase.rebase.model_families import infer_family
from merge_and_rebase.rebase.registry import get_method

_TIED = {"model.embed_tokens.weight", "lm_head.weight"}
_QK_SUFFIXES = ("self_attn.q_norm.weight", "self_attn.k_norm.weight")


def _is_qk_norm(key: str) -> bool:
    return key.endswith(_QK_SUFFIXES)


def _loader():
    tok = local_tokenizer(CALIB_TEXTS, "right")
    return _build_text_calibration_loader(tokenizer=tok, texts=CALIB_TEXTS, batch_size=2, max_length=12)


def _hybrid_passthrough(passthrough_delta, target_base_sd):
    """Mirror of llm_rebase.main's passthrough loop: returns (copied, skipped_keys)."""
    out, skipped = {}, []
    for k, v in passthrough_delta.items():
        if k in target_base_sd and tuple(v.shape) == tuple(target_base_sd[k].shape):
            out[k] = v.to(dtype=target_base_sd[k].dtype, device="cpu")
        else:
            skipped.append(k)
    return out, skipped


def _hybrid(source_base_model, target_model, method_name):
    """Same calls llm_rebase.main issues per task (no depth change): split, transport body, pass the rest."""
    source_ft = perturbed_copy(source_base_model)
    adapter = infer_family(source_base_model)
    source_base = to_cpu_fp32(source_base_model.state_dict())
    delta = TaskVector.from_checkpoints(source_base, to_cpu_fp32(source_ft.state_dict()), strict=False).delta
    transport_keys = set(adapter.transportable_keys(source_base))
    body = {k: v for k, v in delta.items() if k in transport_keys}
    rest = {k: v for k, v in delta.items() if k not in transport_keys}
    target_base = to_cpu_fp32(target_model.state_dict())
    kwargs = dict(
        source_model=source_base_model,
        target_model=target_model,
        source_dataloader=_loader(),
        target_dataloader=_loader(),
        family_adapter=adapter,
        device="cpu",
        seq_align="interpolate",
        n_batches=3,
        verbose=False,
        show_progress=False,
    )
    if method_name == "bico":
        kwargs.update(
            source_recipe=causal_lm_recipe(device="cpu"),
            target_recipe=causal_lm_recipe(device="cpu"),
        )
    transported = get_method(method_name).transport(
        source_base=source_base, target_base=target_base, delta=body, strict=False, prepared=None, **kwargs
    )
    passed, skipped = _hybrid_passthrough(rest, target_base)
    return dict(
        delta=delta,
        transport_keys=transport_keys,
        rest=rest,
        target_base=target_base,
        transported=transported,
        passed=passed,
        skipped=skipped,
    )


def _wide_qwen3(seed=2):
    return tiny_qwen3(hidden=48, heads=6, inter=96, seed=seed)


def test_adapter_excludes_qk_norm_and_tied_embeddings_from_transport():
    model = tiny_qwen3(seed=1)
    state = to_cpu_fp32(model.state_dict())
    assert _TIED <= set(state)  # tied model still exposes both keys
    assert any(_is_qk_norm(k) for k in state)
    keys = infer_family(model).transportable_keys(state)
    assert not any(_is_qk_norm(k) for k in keys)
    assert not (_TIED & keys)
    assert "model.norm.weight" in keys
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    # Qwen2 has no qk-norm and the same exclusions.
    q2_keys = infer_family(tiny_qwen2(seed=1)).transportable_keys(to_cpu_fp32(tiny_qwen2(seed=1).state_dict()))
    assert not (_TIED & q2_keys)


@pytest.mark.parametrize("method", ["theseus", "bico"])
def test_qwen3_to_qwen3_qk_norm_not_transported_but_passed_through_on_shape_match(method):
    out = _hybrid(tiny_qwen3(seed=1), _wide_qwen3(), method)
    qk = {k for k in out["delta"] if _is_qk_norm(k)}
    assert len(qk) == 4  # 2 layers x (q_norm, k_norm)
    assert not (qk & out["transport_keys"]) and not (qk & set(out["transported"]))
    assert not any(_is_qk_norm(k) for k in out["transported"])
    # head_dim is 8 on both sides, so the (8,) deltas pass through unchanged.
    assert qk <= set(out["passed"])
    for k in qk:
        torch.testing.assert_close(out["passed"][k], out["delta"][k])
    # Transport covers exactly the body keys the target has, with target shapes.
    assert set(out["transported"]) <= out["transport_keys"]
    assert "model.norm.weight" in out["transported"]
    for k, v in out["transported"].items():
        assert tuple(v.shape) == tuple(out["target_base"][k].shape)
    # Tied embedding / lm_head: wider target hidden -> shape mismatch -> skipped, not transported.
    assert set(out["skipped"]) == _TIED
    assert not (_TIED & set(out["passed"])) and not (_TIED & set(out["transported"]))


def test_same_width_tied_embeddings_pass_through_identically_to_both_keys():
    out = _hybrid(tiny_qwen3(seed=1), tiny_qwen3(seed=2), "theseus")
    assert out["skipped"] == []
    assert _TIED <= set(out["passed"])
    torch.testing.assert_close(out["passed"]["model.embed_tokens.weight"], out["passed"]["lm_head.weight"])
    assert any(_is_qk_norm(k) for k in out["passed"])


def test_qwen2_to_qwen3_has_no_qk_norm_delta():
    out = _hybrid(tiny_qwen2(seed=1), _wide_qwen3(), "theseus")
    assert not any(_is_qk_norm(k) for k in out["delta"])
    final = {**out["transported"], **out["passed"]}
    assert not any(_is_qk_norm(k) for k in final)
    # ...so the target's own q_norm/k_norm remain exactly at the target base.
    assert any(_is_qk_norm(k) for k in out["target_base"])
    assert set(out["skipped"]) == _TIED


def test_qwen3_to_qwen2_qk_norm_delta_has_no_target_key_and_is_skipped():
    out = _hybrid(tiny_qwen3(seed=1), tiny_qwen2(hidden=48, heads=6, inter=96, seed=2), "theseus")
    qk = {k for k in out["delta"] if _is_qk_norm(k)}
    assert len(qk) == 4
    assert qk <= set(out["skipped"])
    assert not any(_is_qk_norm(k) for k in {**out["transported"], **out["passed"]})
