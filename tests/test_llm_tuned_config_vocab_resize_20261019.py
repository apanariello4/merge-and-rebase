"""A fine-tune with a larger vocabulary (e.g. MergeBench/Llama-3.2-3B_math: 128320 vs 128256) builds under its own
config: alignment drops the reshaped embedding, and the tuned-config model keeps the ref's own one (2026-10-19)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import perturbed_copy, tiny_qwen2

from merge_and_rebase.eval.llm_rebase.stages import LlmTaskContext, build_task_models
from merge_and_rebase.io.text_checkpoints import (
    build_model_with_tuned_config,
    load_aligned_tuned_from_ref,
    tuned_config_mismatch,
)
from merge_and_rebase.models.text_lm import TextBuildConfig


def _resized_tuned(tmp_path, base, extra=8):
    tuned = perturbed_copy(base)
    tuned.resize_token_embeddings(base.config.vocab_size + extra)
    path = tmp_path / "tuned_big_vocab"
    tuned.save_pretrained(path)
    return str(path), tuned


def _env(base, ctx):
    runtime = SimpleNamespace(
        task_contexts={"a": ctx},
        source_llm=SimpleNamespace(model=base),
        target_llm=SimpleNamespace(model=base),
        source_base_sd={k: v.detach().clone() for k, v in base.state_dict().items()},
        source_build_cfg=TextBuildConfig(model_name_or_path="unused", device="cpu", dtype="fp32"),
        load_tuned=load_aligned_tuned_from_ref,
    )
    return SimpleNamespace(
        plan=SimpleNamespace(task_block_extension_prestep=False, task_discrete_layer_match_prestep=False),
        resolved=SimpleNamespace(direct_fit=True),
        runtime=runtime,
    )


def test_vocab_resized_finetune_keeps_its_own_embedding(tmp_path):
    base = tiny_qwen2()
    ref, tuned = _resized_tuned(tmp_path, base)
    overrides = tuned_config_mismatch(ref, base.config)
    assert "vocab_size" in overrides
    ctx = LlmTaskContext(ckpt_ref=ref, index=0, config_overrides=overrides)
    models = build_task_models(_env(base, ctx), "a")
    ft = models.source_ft.state_dict()
    for k, v in tuned.state_dict().items():
        assert torch.equal(ft[k], v), k
    assert ft["model.embed_tokens.weight"].shape[0] == base.config.vocab_size + 8


def test_strict_check_unchanged_without_base_shapes(tmp_path):
    base = tiny_qwen2()
    ref, _ = _resized_tuned(tmp_path, base)
    base_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    build_cfg = TextBuildConfig(model_name_or_path="unused", device="cpu", dtype="fp32")
    aligned = load_aligned_tuned_from_ref(ckpt_ref=ref, base_sd=base_sd, build_cfg=build_cfg, model=base)
    assert "model.embed_tokens.weight" not in aligned
    with pytest.raises(RuntimeError, match="do not cover"):
        build_model_with_tuned_config(ref, build_cfg, dict(aligned))
    shapes = {k: tuple(v.shape) for k, v in base_sd.items()}
    build_model_with_tuned_config(ref, build_cfg, dict(aligned), base_shapes=shapes)


def test_same_shape_gap_still_raises(tmp_path):
    base = tiny_qwen2()
    ref, _ = _resized_tuned(tmp_path, base)
    base_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    build_cfg = TextBuildConfig(model_name_or_path="unused", device="cpu", dtype="fp32")
    aligned = dict(load_aligned_tuned_from_ref(ckpt_ref=ref, base_sd=base_sd, build_cfg=build_cfg, model=base))
    dropped = next(k for k in aligned if "down_proj" in k)
    del aligned[dropped]
    with pytest.raises(RuntimeError, match="do not cover"):
        build_model_with_tuned_config(
            ref, build_cfg, aligned, base_shapes={k: tuple(v.shape) for k, v in base_sd.items()}
        )
