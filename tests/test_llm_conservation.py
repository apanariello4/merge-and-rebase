"""LLM conservation check (eval/diagnostics/llm_conservation.py): fixed MMLU subset, exact zero difference at tau = 0,
base cache, summary parsing and the CLI, with a toy decoder and a deterministic stand-in for the lm-eval harness."""

from __future__ import annotations

import json

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, tiny_llama

from merge_and_rebase.eval.diagnostics import llm_conservation as lc


def _fake_harness(calls):
    """Scores from the model's own logits on fixed inputs: any weight change moves them, identical weights repeat
    them bit for bit (same CPU kernels)."""

    @torch.no_grad()
    def harness(*, tasks, model, tokenizer, device, num_fewshot, batch_size, samples):
        calls.append({"tasks": list(tasks), "num_fewshot": dict(num_fewshot), "samples": dict(samples),
                      "batch_size": batch_size})
        ids = torch.arange(12).reshape(2, 6) % 60
        logits = model(input_ids=ids).logits.double()
        out = {}
        for i, t in enumerate(tasks):
            v = float(torch.sigmoid(logits[..., i % logits.shape[-1]].mean()))
            out[f"{t}_acc"] = v
            if t == lc.ARC_TASK:
                out[f"{t}_acc_norm"] = float(torch.sigmoid(logits.mean()))
        return out

    return harness


def _tau(model, scale):
    g = torch.Generator().manual_seed(5)
    tau = {}
    for j, layer in enumerate(model.model.layers):
        w = layer.mlp.down_proj.weight
        tau[f"model.layers.{j}.mlp.down_proj.weight"] = scale * torch.randn(w.shape, generator=g)
        tau[f"model.layers.{j}.mlp.down_proj.bias"] = scale * torch.randn(w.shape[0], generator=g)
    return tau


SUBSET = {"mmlu_anatomy": [0, 3], "mmlu_virology": [1]}


def test_mmlu_subset_is_fixed_and_deterministic():
    data = json.loads(lc.MMLU_SUBSET_PATH.read_text())
    assert data["n"] == 500 and data["seed"] == 0 and len(data["sizes"]) == 57 and data["population_size"] == 14042
    subset = lc.load_mmlu_subset()
    assert subset == lc.load_mmlu_subset() == lc.mmlu_subset(data["sizes"], n=500, seed=0)
    assert sum(len(v) for v in subset.values()) == 500
    for task, idx in subset.items():
        assert task.startswith("mmlu_") and idx == sorted(set(idx))
        assert all(0 <= i < data["sizes"][task] for i in idx)
    assert lc.mmlu_subset(data["sizes"], n=500, seed=1) != subset
    # simple random sample over the pool: a subject's share follows its size (largest subject has the most picks)
    largest = max(data["sizes"], key=data["sizes"].get)
    assert len(subset[largest]) == max(len(v) for v in subset.values())


def test_tampered_subset_file_is_refused(tmp_path):
    data = json.loads(lc.MMLU_SUBSET_PATH.read_text())
    task = sorted(data["samples"])[0]
    data["samples"][task] = data["samples"][task][1:]
    bad = tmp_path / "subset.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        lc.load_mmlu_subset(bad)


def test_zero_update_gives_exactly_zero_difference_and_the_base_is_cached(tmp_path):
    model = tiny_llama(layers=2, seed=0).eval()
    tok = local_tokenizer(CALIB_TEXTS)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []
    kw = dict(target="toy/llama", dtype="float32", device="cpu", cache_dir=tmp_path, subset=SUBSET,
              harness=_fake_harness(calls))
    zero = lc.evaluate_conservation(model, tok, _tau(model, 0.0), 0.7, **kw)
    assert zero["zero_biases_added"] == 2 and not zero["base_cached"]
    assert all(v == 0.0 for v in zero["delta_points"].values())
    assert zero["base"] == zero["rebased"]
    # the harness call: ARC (0-shot, full test) and the MMLU subjects (5-shot) with their fixed doc indices
    assert calls[0]["tasks"] == [lc.ARC_TASK, "mmlu_anatomy", "mmlu_virology"]
    assert calls[0]["num_fewshot"] == {lc.ARC_TASK: 0, "mmlu_anatomy": 5, "mmlu_virology": 5}
    assert calls[0]["samples"] == SUBSET and calls[0]["batch_size"] == 2
    # mmlu500_acc pools the questions: subject accuracies weighted by their question counts
    by = zero["base"]["mmlu500_acc_by_subject"]
    assert zero["base"]["mmlu500_acc"] == pytest.approx((2 * by["mmlu_anatomy"] + by["mmlu_virology"]) / 3)
    # the model is left at its base weights (plus the zero biases)
    for k, v in original.items():
        assert torch.equal(model.state_dict()[k], v)
    assert all(float(model.state_dict()[k].abs().max()) == 0.0 for k in model.state_dict() if k.endswith("down_proj.bias"))

    calls.clear()
    moved = lc.evaluate_conservation(model, tok, _tau(model, 0.05), 0.7, **kw)
    assert moved["base_cached"] and len(calls) == 1  # base read from the cache, only the rebased model scored
    assert moved["base"] == zero["base"] and moved["base_cache"] == zero["base_cache"]
    assert any(v != 0.0 for v in moved["delta_points"].values())
    assert moved["zero_biases_added"] == 0  # already materialized on this model object
    assert set(moved["se_points_base"]) == set(lc.HEADLINE)


def test_update_shape_mismatch_and_unknown_keys_are_refused():
    model = tiny_llama(layers=2, seed=0)
    with pytest.raises(KeyError):
        lc.ScaledUpdate(model, {"model.layers.9.mlp.down_proj.weight": torch.zeros(32, 64)})
    with pytest.raises(ValueError):
        lc.ScaledUpdate(model, {"model.layers.0.mlp.down_proj.weight": torch.zeros(64, 32)})


def _summary(tmp_path, tv_path):
    summary = {
        "best_alpha": 0.3,
        "transported_tv": {"path": str(tv_path), "sha256": "abc", "sidecar": str(tmp_path / "missing.json")},
        "run_logging": {"metadata": {"resolved_config": {"target_model_name_or_path": "toy/llama",
                                                          "dtype": "float32"}}},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(summary))
    return path


def test_main_reads_the_run_summary_and_writes_next_to_it(tmp_path, monkeypatch):
    import merge_and_rebase.eval.llm_rebase.harness as harness_mod

    model = tiny_llama(layers=2, seed=0).eval()
    tv_path = tmp_path / "tv.pt"
    torch.save(_tau(model, 0.05), tv_path)
    summary = _summary(tmp_path, tv_path)
    assert lc.run_inputs(summary) == {"tv": str(tv_path), "alpha": 0.3, "target": "toy/llama", "dtype": "float32",
                                      "tv_sha256": "abc"}
    loads = []
    monkeypatch.setattr(lc, "load_target", lambda target, dtype, device: loads.append((target, dtype, device))
                        or (model, local_tokenizer(CALIB_TEXTS)))
    calls = []
    monkeypatch.setattr(harness_mod, "run", _fake_harness(calls))
    monkeypatch.setattr(lc, "load_mmlu_subset", lambda: dict(SUBSET))
    result = lc.main([str(summary), "--device", "cpu"])
    out = tmp_path / "run_conservation.json"
    assert out.exists() and json.loads(out.read_text())["alpha"] == 0.3
    assert loads == [("toy/llama", "float32", "cpu")] and len(calls) == 2
    assert result["tv"] == str(tv_path) and result["tv_sha256"] == "abc"
    assert list(tmp_path.glob("llm_conservation_base_*.json"))
    # explicit flags override the summary; without a summary --out is required
    with pytest.raises(SystemExit):
        lc.main(["--tv", str(tv_path), "--alpha", "0.1", "--target", "toy/llama"])
    with pytest.raises(ValueError, match="transported_tv"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"best_alpha": 1.0}))
        lc.run_inputs(bad)
