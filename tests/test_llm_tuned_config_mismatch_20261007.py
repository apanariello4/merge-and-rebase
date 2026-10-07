"""A tuned body with its own HF config (RoPE, window, ...) must not run under the source base's config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import perturbed_copy, tiny_qwen2

from merge_and_rebase.eval.llm_rebase import stages
from merge_and_rebase.eval.llm_rebase.context import _task_contexts
from merge_and_rebase.eval.llm_rebase.stages import LlmTaskContext, build_task_models
from merge_and_rebase.io.text_checkpoints import (
    diff_computation_configs,
    load_aligned_tuned_from_ref,
    tuned_config_mismatch,
)
from merge_and_rebase.models.text_lm import TextBuildConfig


def _save_tuned(tmp_path, base, name, **config_changes):
    tuned = perturbed_copy(base)
    for k, v in config_changes.items():
        if k == "rope_theta":
            tuned.config.rope_parameters["rope_theta"] = v
        else:
            setattr(tuned.config, k, v)
    path = tmp_path / name
    tuned.save_pretrained(path)
    return str(path), tuned


def test_detection_ignores_bookkeeping_and_reports_computation_fields(tmp_path):
    base = tiny_qwen2()
    same, _ = _save_tuned(tmp_path, base, "same")
    assert tuned_config_mismatch(same, base.config) == {}

    bookkeeping = deepcopy_config(base.config)
    bookkeeping._name_or_path = "somewhere/else"
    bookkeeping.transformers_version = "0.0.1"
    bookkeeping.torch_dtype = torch.bfloat16
    bookkeeping.architectures = ["Other"]
    assert diff_computation_configs(base.config, bookkeeping) == {}

    other, _ = _save_tuned(tmp_path, base, "other", rope_theta=500000.0, max_position_embeddings=32, rms_norm_eps=1e-5)
    diff = tuned_config_mismatch(other, base.config)
    # max_position_embeddings only sizes the cache of an unscaled RoPE: not a computation difference
    assert set(diff) == {"rope_theta", "rms_norm_eps"}
    assert diff["rope_theta"] == [base.config.rope_parameters["rope_theta"], 500000.0]
    # a raw state-dict file has no config of its own
    pt = tmp_path / "w.pt"
    torch.save(base.state_dict(), pt)
    assert tuned_config_mismatch(str(pt), base.config) == {}


def deepcopy_config(config):
    from copy import deepcopy

    return deepcopy(config)


def test_transporting_methods_raise_unless_allowed(tmp_path):
    base = tiny_qwen2()
    other, _ = _save_tuned(tmp_path, base, "other", rope_theta=500000.0)
    kwargs = {"source_config": base.config}
    with pytest.raises(ValueError, match="allow_tuned_config_mismatch"):
        _task_contexts(["a"], [other], transports_delta=True, **kwargs)
    with pytest.warns(RuntimeWarning, match="tuned config"):
        ctx = _task_contexts(["a"], [other], transports_delta=True, allow_mismatch=True, **kwargs)["a"]
    assert set(ctx.config_overrides) == {"rope_theta"}
    # activation-based (direct fit) runs without the flag but still records the override
    with pytest.warns(RuntimeWarning):
        assert _task_contexts(["a"], [other], transports_delta=False, **kwargs)["a"].config_overrides
    # equal config: no error, no override
    same, _ = _save_tuned(tmp_path, base, "same")
    assert _task_contexts(["a"], [same], transports_delta=True, **kwargs)["a"].config_overrides == {}


def _env(base, ctx):
    runtime = SimpleNamespace(
        task_contexts={"a": ctx},
        source_llm=SimpleNamespace(model=base),
        target_llm=SimpleNamespace(model=base),
        source_base_sd={k: v.detach().clone() for k, v in base.state_dict().items()},
        source_build_cfg=TextBuildConfig(model_name_or_path="unused", device="cpu"),
        load_tuned=load_aligned_tuned_from_ref,
    )
    return SimpleNamespace(
        plan=SimpleNamespace(task_block_extension_prestep=True, task_discrete_layer_match_prestep=False),
        resolved=SimpleNamespace(direct_fit=False),
        runtime=runtime,
    )


def test_ft_forward_model_uses_tuned_config_with_identical_weights(tmp_path):
    base = tiny_qwen2()
    other, tuned = _save_tuned(tmp_path, base, "other", rope_theta=500000.0)
    ctx = LlmTaskContext(ckpt_ref=other, index=0, config_overrides=tuned_config_mismatch(other, base.config))
    models = build_task_models(_env(base, ctx), "a")
    assert models.source_ft.config.rope_parameters["rope_theta"] == 500000.0
    assert models.source_base.config.rope_parameters["rope_theta"] == base.config.rope_parameters["rope_theta"]
    # run-time record: the live forward models' geometry, as written to the summary
    geometry = ctx.forward_geometry
    assert geometry["source_ft"]["rope_theta"] == 500000.0
    assert geometry["source_base"]["rope_theta"] == base.config.rope_parameters["rope_theta"] != 500000.0
    assert geometry["target"]["rope_theta"] == base.config.rope_parameters["rope_theta"]
    for k, v in tuned.state_dict().items():
        assert torch.equal(models.source_ft.state_dict()[k], v), k
    ids = torch.arange(8).unsqueeze(0)
    with torch.no_grad():
        # same weights, geometry of the tuned config: matches the tuned model, differs from the wrongly-configured one
        from transformers import AutoModelForCausalLM

        reference = AutoModelForCausalLM.from_pretrained(other).eval()  # the tuned ref as HF itself builds it
        assert torch.allclose(models.source_ft(ids).logits, reference(ids).logits, atol=1e-6)
        assert not torch.allclose(
            models.source_ft(ids).logits, deepcopy_config_model(base, tuned)(ids).logits, atol=1e-6
        )


def test_equal_config_path_is_unchanged(tmp_path, monkeypatch):
    base = tiny_qwen2()
    same, tuned = _save_tuned(tmp_path, base, "same")

    def _boom(*a, **k):
        raise AssertionError("no extra model build for equal configs")

    monkeypatch.setattr(stages, "build_model_with_tuned_config", _boom)
    ctx = LlmTaskContext(ckpt_ref=same, index=0)
    assert ctx.config_overrides == {}
    models = build_task_models(_env(base, ctx), "a")
    assert models.source_ft is not base and models.source_ft.config == base.config
    for k, v in tuned.state_dict().items():
        assert torch.equal(models.source_ft.state_dict()[k], v), k


def test_real_qwen_math_vs_base_config_if_cached():
    from transformers import AutoConfig

    try:
        base = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
        math = AutoConfig.from_pretrained("Qwen/Qwen2.5-Math-1.5B", local_files_only=True)
    except Exception as exc:
        pytest.skip(f"configs not cached ({exc!r})")
    diff = diff_computation_configs(base, math)
    # rope_theta is the only computation difference; max_position_embeddings / max_window_layers are inert here
    # (default RoPE, sliding window off on both sides)
    assert diff == {"rope_theta": [1000000.0, 10000]}
    for name in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
        try:
            instruct = AutoConfig.from_pretrained(f"{name}-Instruct", local_files_only=True)
            plain = AutoConfig.from_pretrained(name, local_files_only=True)
        except Exception:
            continue
        assert diff_computation_configs(plain, instruct) == {}, name


def test_window_and_scaled_rope_differences_still_count():
    """The inert keys become effective when the window is on, or the RoPE is scaled."""
    from transformers import Qwen2Config

    def cfg(**kw):
        return Qwen2Config(num_hidden_layers=4, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, **kw)

    assert diff_computation_configs(cfg(max_window_layers=2), cfg(max_window_layers=3)) == {}
    assert diff_computation_configs(cfg(max_position_embeddings=64), cfg(max_position_embeddings=128)) == {}
    on = {"use_sliding_window": True, "sliding_window": 16}
    assert "layer_types" in diff_computation_configs(cfg(max_window_layers=2, **on), cfg(max_window_layers=3, **on))
    assert "sliding_window" in diff_computation_configs(cfg(), cfg(**on))


def deepcopy_config_model(base, tuned):
    """Tuned weights under the SOURCE geometry: the silent bug this guard prevents."""
    from copy import deepcopy

    wrong = deepcopy(base)
    wrong.load_state_dict(tuned.state_dict())
    return wrong.eval()


def test_instruct_pair_differences_are_not_a_mismatch():
    from transformers import Qwen2Config

    a = Qwen2Config(eos_token_id=151643, bos_token_id=151643).to_dict()
    a["use_mrope"] = False
    b = dict(a, eos_token_id=151645, use_mrope=None, _name_or_path="Qwen/Qwen2.5-Math-1.5B-Instruct")
    assert (
        diff_computation_configs(Qwen2Config(**a), Qwen2Config(**{k: v for k, v in b.items() if k != "use_mrope"}))
        == {}
    )
    assert diff_computation_configs(SimpleNamespace(to_dict=lambda: a), SimpleNamespace(to_dict=lambda: b)) == {}
