"""``anchors`` on decoder LLMs that share a tokenizer (AGENT_NEXT_PHASE.md L1): rows = tokens, tiny Llama models.

Pairing of identical token rows (hash guard), the shared ridge bit-identical to Ariadne's decoder fit, the bridge to
Ariadne (task rows + full-rank Procrustes), permutation equivariance (pairwise and GPA), the text-only option guard,
and the LLM ``DirectFitStage`` smoke run.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_gemma2, tiny_llama

from merge_and_rebase.data.llm_calibration import build_text_calibration_loader
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.methods._ariadne.capture import (
    ROW_MASK_KEY,
    _first_content_position,
    capture_paired_boundary_activations,
    iter_capture_tokens,
)
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods.anchors import (
    AnchorsConfig,
    AnchorsRebase,
    AnchorState,
    compile_desired_effects,
    resolve_anchors_decoder_config,
)
from merge_and_rebase.rebase.methods.anchors.method import restrict_to_depth
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase, compute_desired_effects
from merge_and_rebase.rebase.model_families import infer_family

WORDS = sorted({w for t in CALIB_TEXTS for w in t.split()})
# Generic pool: 32 sequences of 5-10 words of the fixture vocabulary (8 batches of 4, of which the anchor and
# registration batches are drawn), all within max_length 12, so the right padding varies per sequence.
POOL_TEXTS = [" ".join(WORDS[(7 * i + 3 * k) % len(WORDS)] for k in range(5 + i % 6)) for i in range(32)]
# Task calibration: 3 batches of 4 sequences of 8-10 words, ~90 content rows (more than the widest model, so the
# full-rank Procrustes of the bridge is unique).
TASK_TEXTS = [" ".join(WORDS[(5 * i + 2 * k + 1) % len(WORDS)] for k in range(8 + i % 3)) for i in range(24)]
MAX_LENGTH = 12


def _tok(texts=CALIB_TEXTS):
    return local_tokenizer(texts)


def _pool(tok=None, max_length=MAX_LENGTH):
    return build_text_calibration_loader(
        tokenizer=tok or _tok(), texts=POOL_TEXTS, batch_size=4, max_length=max_length
    )


def _task(tok=None):
    return build_text_calibration_loader(tokenizer=tok or _tok(), texts=TASK_TEXTS, batch_size=4, max_length=MAX_LENGTH)


def _cfg(**kw):
    base = dict(
        anchor_pool="wikitext103",
        num_anchors=16,
        anchor_batches=2,
        registration_batches=2,
        num_batches=3,
        components=("mlp.c_proj",),
        ridge_estimator="empirical_bayes",
        missing_bias="materialize",
    )
    base.update(kw)
    return AnchorsConfig(**base)


def _with_bias(model, fam):
    """A copy with zero down_proj biases (missing_bias='materialize') and its state dict."""
    model = deepcopy(model)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    materialize_missing_projection_biases(model, sd, family_adapter=fam, components=("mlp.c_proj",))
    return model, sd


def _setup(source_layers=2, target_layers=3, target_hidden=32):
    src = tiny_llama(layers=source_layers, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    fam = infer_family(src)
    tgt, sd = _with_bias(tiny_llama(layers=target_layers, hidden=target_hidden, inter=2 * target_hidden, seed=2), fam)
    return src, ft, tgt, sd, fam, DiscreteLayerPairing.compute(source_layers, target_layers)


def _prepare(cfg, src, ft, tgt, sd, fam, pairing, *, state=None, source_id="A", target_id="B"):
    return AnchorsRebase().prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd=sd,
        source_loader=_task(),
        target_loader=_task(),
        pairing=pairing,
        config=cfg,
        device="cpu",
        source_anchor_loader=_pool(),
        target_anchor_loader=_pool(),
        source_model_id=source_id,
        target_model_id=target_id,
        state=state,
        family_adapter=fam,
    )


def _ariadne_cfg(**kw):
    base = dict(
        components=("mlp.c_proj",),
        ridge_estimator="empirical_bayes",
        activation_storage="resident",
        num_batches=3,
        missing_bias="materialize",
        exclude_first_token=True,
    )
    base.update(kw)
    return DirectResidualConfig(**base)


# ---------------------------------------------------------------- frame: identical token rows


def test_text_frame_rows_are_content_tokens_identical_in_both_models():
    src, _, tgt, sd, fam, _ = _setup()
    cfg = _cfg()
    state = AnchorState(cfg)
    a = state.register_model(src, "A", _pool(), device="cpu", family_adapter=fam)
    b = state.register_model(tgt, "B", _pool(_tok()), device="cpu", family_adapter=fam)  # its own tokenizer object
    frame = state.frame
    assert len(frame["anchor_rows"]) == 16 and frame["anchor_rows"] == sorted(frame["anchor_rows"])
    assert set(frame["pool"]["tokens"]) == {"token_rows_sha256", "content_rows", "exclude_first_token"}
    assert a.meta["rows"] == "tokens" and a.meta["max_length"] == MAX_LENGTH
    assert a.depth == 2 and b.depth == 3 and a.p[0].shape == (16, 32)
    # every anchor row is a content token, never padding nor the first token of its sequence; and the stored frame
    # is the model's own hidden state at that (sequence, position), centred and scaled
    batches, _, _ = state._draw(_pool(), fam)  # the seeded pool draw, as registration made it (checked vs the frame)
    masks = torch.cat([bt["attention_mask"].bool() for bt in batches[: cfg.anchor_batches]])
    first = _first_content_position(masks)
    for s, p in frame["anchor_rows"]:
        assert masks[s, p] and not first[s, p]
    plain = [{k: v for k, v in bt.items() if k != ROW_MASK_KEY} for bt in batches[: cfg.anchor_batches]]
    hidden = torch.cat([v["0"] for v in iter_capture_tokens(src, plain, {"0": (0, "boundary")}, "cpu",
                                                            family_adapter=fam)])
    rows = torch.stack([hidden[s, p] for s, p in frame["anchor_rows"]]).double()
    torch.testing.assert_close(a.p[0].double(), ((rows - a.mu[0]) / a.c[0]).float().double(), rtol=0, atol=0)


def test_text_frame_refuses_different_tokens():
    src, *_ = _setup()
    fam = infer_family(src)
    state = AnchorState(_cfg())
    state.register_model(src, "A", _pool(), device="cpu", family_adapter=fam)
    with pytest.raises(ValueError, match="token rows differ"):  # same texts, another vocabulary order
        state.register_model(src, "C", _pool(_tok(list(reversed(CALIB_TEXTS)))), device="cpu", family_adapter=fam)
    with pytest.raises(ValueError, match="token rows differ"):  # same tokenizer, another max_length
        state.register_model(src, "D", _pool(max_length=16), device="cpu", family_adapter=fam)
    assert not state.has_model("C") and not state.has_model("D")


# ---------------------------------------------------------------- shared ridge = Ariadne's decoder ridge


def test_text_shared_ridge_reproduces_ariadne_bitwise():
    """Fed Ariadne's decoder D (BOS and padding excluded), the anchors compile returns Ariadne's task vector."""
    src, ft, tgt, sd, fam, pairing = _setup()
    cfg = _cfg()
    ariadne_cfg = _ariadne_cfg()
    from dataclasses import replace

    assert replace(cfg.ridge_config(), exclude_first_token=True) == ariadne_cfg
    ariadne = AriadneRebase().prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd=sd,
        source_loader=_task(),
        target_loader=_task(),
        pairing=pairing,
        config=ariadne_cfg,
        device="cpu",
        family_adapter=fam,
    )
    captured = capture_paired_boundary_activations(
        src, ft, tgt, _task(), _task(), pairing, num_batches=3, seed=cfg.seed, device="cpu",
        family_adapter=fam, exclude_first_token=True,
    )
    desired = compute_desired_effects(captured, pairing)
    tv, _ = compile_desired_effects(tgt, sd, captured, desired, pairing, config=cfg, device="cpu", family_adapter=fam)
    assert set(tv) == set(ariadne.unit_task_vector)
    assert any(k.endswith("mlp.down_proj.bias") for k in tv)
    for key, value in tv.items():
        assert torch.equal(value, ariadne.unit_task_vector[key]), key


# ---------------------------------------------------------------- bridge and equivariance


@pytest.mark.parametrize("layers,hidden", [((2, 3), 48), ((3, 2), 32)])
def test_text_task_rows_full_rank_procrustes_is_ariadne(layers, hidden):
    """Row 0 of the bridge on text: task rows + procrustes_cross at full rank + no rescaling = Ariadne's decoder fit."""
    src, ft, tgt, sd, fam, pairing = _setup(*layers, target_hidden=hidden)
    cfg = _cfg(map_rows="task", spectral_filter="procrustes_cross", rescale_target=False)
    anchors = _prepare(cfg, src, ft, tgt, sd, fam, pairing).unit_task_vector
    ariadne = AriadneRebase().prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd=sd,
        source_loader=_task(),
        target_loader=_task(),
        pairing=pairing,
        config=_ariadne_cfg(),
        device="cpu",
        family_adapter=fam,
    ).unit_task_vector
    assert set(anchors) == set(ariadne)
    for key in ariadne:
        torch.testing.assert_close(anchors[key], ariadne[key], rtol=1e-4, atol=1e-5)


def _permuted(model, perm):
    """The same function with its residual stream permuted (``x' = x[..., perm]``): RMSNorm commutes with it."""
    out = deepcopy(model)
    m = out.model
    with torch.no_grad():
        m.embed_tokens.weight.copy_(m.embed_tokens.weight[:, perm])  # tied: also the LM head
        m.norm.weight.copy_(m.norm.weight[perm])
        for layer in m.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm.weight.copy_(norm.weight[perm])
            att, mlp = layer.self_attn, layer.mlp
            for lin in (att.q_proj, att.k_proj, att.v_proj, mlp.gate_proj, mlp.up_proj):
                lin.weight.copy_(lin.weight[:, perm])
            for lin in (att.o_proj, mlp.down_proj):
                lin.weight.copy_(lin.weight[perm])
                if lin.bias is not None:
                    lin.bias.copy_(lin.bias[perm])
    return out


@pytest.mark.parametrize("frame_map", ["pairwise", "gpa"])
def test_text_permuted_target_receives_the_permuted_task_vector(frame_map):
    src = tiny_llama(layers=2, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    fam = infer_family(src)
    perm = torch.randperm(32, generator=torch.Generator().manual_seed(4))
    same, sd_same = _with_bias(src, fam)
    moved, sd_moved = _with_bias(_permuted(src, perm), fam)
    pairing = DiscreteLayerPairing.compute(2, 2)
    # GPA needs T > d (the maps are otherwise free outside the anchors' span); the kernel decode does not
    cfg = _cfg(frame_map="gpa", anchor_batches=4, num_anchors=48) if frame_map == "gpa" else _cfg()
    state = AnchorState(cfg)
    a = _prepare(cfg, src, ft, same, sd_same, fam, pairing, state=state, source_id="A", target_id="A")
    b = _prepare(cfg, src, ft, moved, sd_moved, fam, pairing, state=state, source_id="A", target_id="A_perm")
    for j in range(2):
        w, bias = f"model.layers.{j}.mlp.down_proj.weight", f"model.layers.{j}.mlp.down_proj.bias"
        torch.testing.assert_close(b.unit_task_vector[w], a.unit_task_vector[w][perm], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(b.unit_task_vector[bias], a.unit_task_vector[bias][perm], rtol=1e-4, atol=1e-6)
    assert any(float(v.abs().max()) > 0 for v in a.unit_task_vector.values())
    assert b.extra["anchors"]["rows"] == "tokens"
    if frame_map == "gpa":
        assert b.extra["anchors"]["gpa_universe"]["fit_models"] == ["A", "A_perm"]
    else:
        assert all(row["anchors_kernel_rel_err"] < 1e-5 for row in b.diagnostics)


def test_text_norm_filter_with_no_flagged_rows_is_the_unfiltered_transport():
    src, ft, tgt, sd, fam, pairing = _setup()
    plain = _prepare(_cfg(), src, ft, tgt, sd, fam, pairing).unit_task_vector
    loose = _prepare(_cfg(anchor_filter="norm", anchor_filter_k=1e6), src, ft, tgt, sd, fam, pairing).unit_task_vector
    for key in plain:
        torch.testing.assert_close(loose[key], plain[key], rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- option guard


def test_decoder_config_defaults_and_rejections():
    fam = infer_family(tiny_llama())
    cfg = resolve_anchors_decoder_config({"frame_map": "gpa", "num_anchors": 512}, fam)
    assert cfg.anchor_pool == "wikitext103" and cfg.missing_bias == "materialize" and cfg.frame_map == "gpa"
    assert AnchorsConfig().anchor_pool == "tiny_imagenet" and AnchorsConfig().missing_bias == "error"  # vision
    for bad in (
        {"library": "operator", "frame_map": "gpa"},
        {"synthesis": "ridge_preserve"},
        {"update_rank": 4},
        {"depth_fit": "sequential"},
        {"depth_fit": "spread", "measure_depths": [0.5]},
        {"anchor_unit": "cls"},
        {"anchor_positions": "patch"},
        {"anchor_sampling": "whole_images"},
        {"map_rows": "task", "map_rows_per_image": 2},
        {"components": ["mlp.c_proj", "attn.out_proj"]},
        {"anchor_pool": "tiny_imagenet"},
    ):
        with pytest.raises(ValueError):
            resolve_anchors_decoder_config(bad, fam)
    with pytest.raises(ValueError, match="family adapter"):
        resolve_anchors_decoder_config({}, None)
    with pytest.raises(NotImplementedError, match="post-MLP"):
        resolve_anchors_decoder_config({}, infer_family(tiny_gemma2()))


def test_prepare_rejects_text_pool_without_adapter_and_vision_options_with_one():
    src, ft, tgt, sd, fam, pairing = _setup()
    with pytest.raises(ValueError, match="text pool"):
        _prepare(_cfg(), src, ft, tgt, sd, None, pairing)
    with pytest.raises(ValueError, match="anchor_pool"):
        _prepare(_cfg(anchor_pool="tiny_imagenet"), src, ft, tgt, sd, fam, pairing)
    with pytest.raises(ValueError, match="library"):
        _prepare(_cfg(library="operator", frame_map="gpa"), src, ft, tgt, sd, fam, pairing)


def test_restrict_to_depth_reads_decoder_layer_keys():
    tv = {f"model.layers.{j}.mlp.down_proj.weight": torch.ones(2, 2) for j in range(3)}
    out = restrict_to_depth(tv, 3, 0.6)
    assert [float(out[f"model.layers.{j}.mlp.down_proj.weight"].sum()) for j in range(3)] == [4.0, 4.0, 0.0]


# ---------------------------------------------------------------- LLM DirectFitStage


def test_llm_direct_fit_stage_runs_anchors(monkeypatch):
    from merge_and_rebase.eval.llm_rebase import anchors_pool as pool_mod
    from merge_and_rebase.eval.llm_rebase.method_stages import DirectFitStage

    monkeypatch.setattr(pool_mod, "load_pool_texts", lambda pool: list(POOL_TEXTS))
    src, ft, tgt, sd, fam, pairing = _setup(2, 3)
    target = tiny_llama(layers=3, seed=2)
    target_sd = {k: v.detach().clone() for k, v in target.state_dict().items()}
    cfg = resolve_anchors_decoder_config(
        {"num_anchors": 16, "anchor_batches": 2, "registration_batches": 2, "num_batches": 3, "frame_map": "gpa"}, fam
    )
    from dataclasses import replace

    resolved = SimpleNamespace(anchors_cfg=cfg, ariadne_cfg=replace(cfg.ridge_config(), exclude_first_token=True))
    tok = _tok()
    runtime = SimpleNamespace(
        cfg={
            "source_model_name_or_path": "org/src-model",
            "target_model_name_or_path": "org/tgt-model",
            "anchors_pool": {"batch_size": 4, "max_length": MAX_LENGTH, "n_sequences": len(POOL_TEXTS)},
        },
        family_adapter=fam,
        source_llm=SimpleNamespace(model=src, tokenizer=tok),
        target_llm=SimpleNamespace(model=target, tokenizer=_tok()),
        target_base_sd=target_sd,
        _calibration=lambda: SimpleNamespace(texts=TASK_TEXTS),
        calib_batch_size=2,
        calib_max_length=MAX_LENGTH,
        source_depth=2,
        target_depth=3,
        device="cpu",
        task_contexts={"math": None},
    )
    stage = DirectFitStage()
    env = SimpleNamespace(runtime=runtime, resolved=resolved)
    pre = SimpleNamespace(source_base_model=src, source_ft_model=ft, task_delta={})
    task = SimpleNamespace(task="math", ctx=SimpleNamespace(index=0))
    result = stage.run(env, task, pre)
    keys = set(result.transported_delta)
    assert keys == {f"model.layers.{j}.mlp.down_proj.{p}" for j in range(3) for p in ("weight", "bias")}
    report = stage.report()
    record = report["per_task"]["math"]
    assert record["anchors"]["rows"] == "tokens"
    assert record["anchors"]["frame_pool"]["tokens"]["exclude_first_token"] is True
    assert record["anchors"]["pool"]["n_sequences"] == len(POOL_TEXTS)
    assert record["anchors"]["pool"]["max_length"] == MAX_LENGTH and record["anchors"]["pool"]["batch_size"] == 4
    assert record["anchors"]["gpa_universe"]["fit_models"] == ["org_src-model", "org_tgt-model"]
    assert report["pairing"]["pairing"] == list(pairing.pairing)
    # the second task reuses the registered frame (no second registration)
    state = stage._anchors_state
    assert set(state.models) == {"org_src-model", "org_tgt-model"}
    stage.run(env, SimpleNamespace(task="math2", ctx=SimpleNamespace(index=0)),
              SimpleNamespace(source_base_model=src, source_ft_model=ft, task_delta={}))
    assert stage._anchors_state is state
