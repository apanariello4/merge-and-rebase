"""harness_split (ifeval_k_v1): the fixed IFEval selection/test split, its config key and the full test curve."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch

from merge_and_rebase.eval.llm_rebase import alpha_search as asr
from merge_and_rebase.eval.llm_rebase import harness_split as hs
from merge_and_rebase.eval.llm_rebase.summary import assemble_harness_summary

CFG = {"harness_split": "ifeval_k_v1", "harness_tasks": ["ifeval"]}


def test_split_is_a_disjoint_partition_and_reproducible():
    s = hs.load_harness_split("ifeval_k_v1")
    sel, test, cal = set(s["selection"]), set(s["test"]), set(s["calibration"])
    assert (len(sel), len(test), len(cal)) == (150, 351, 40)
    assert not (sel & test) and not (sel & cal) and not (test & cal)
    assert sel | test | cal == set(range(541))
    assert s["selection"] == sorted(s["selection"]) and s["test"] == sorted(s["test"])
    again = hs.build_split_file("ifeval_k_v1")
    assert json.loads(hs.SPLITS["ifeval_k_v1"].read_text()) == again  # the stored file is what the generator writes
    assert hs.make_split(**hs.IFEVAL_K_V1_PARAMS)["selection"] == s["selection"]
    assert hs.make_split(**{**hs.IFEVAL_K_V1_PARAMS, "seed": 1})["selection"] != s["selection"]


def test_split_equals_the_old_alpha_split_eval_construction():
    """alpha_split_eval.py: calibration eval_samples = sorted(order[40:]) with random.Random(0).shuffle(order); then
    torch.randperm(501, seed 0): first 150 selection, rest test (checked against resolve_calibration_texts on IFEval too)."""
    order = list(range(541))
    random.Random(0).shuffle(order)
    docs = sorted(order[40:])
    perm = torch.randperm(len(docs), generator=torch.Generator().manual_seed(0)).tolist()
    s = hs.load_harness_split("ifeval_k_v1")
    assert s["selection"] == sorted(docs[i] for i in perm[:150])
    assert s["test"] == sorted(docs[i] for i in perm[150:])


def test_tampered_split_file_is_refused(tmp_path, monkeypatch):
    data = json.loads(hs.SPLITS["ifeval_k_v1"].read_text())
    data["test"][0], data["selection"][0] = data["selection"][0], data["test"][0]
    bad = tmp_path / "ifeval_k_v1.json"
    bad.write_text(json.dumps(data))
    monkeypatch.setitem(hs.SPLITS, "ifeval_k_v1", bad)
    with pytest.raises(ValueError, match="does not match"):
        hs.load_harness_split("ifeval_k_v1")


def test_config_key_resolves_to_selection_and_test():
    sel, test, record = hs.resolve_harness_split(CFG)
    s = hs.load_harness_split("ifeval_k_v1")
    assert sel == {"ifeval": s["selection"]} and test == {"ifeval": s["test"]}
    assert record["name"] == "ifeval_k_v1" and record["n_selection"] == 150 and record["sha256"] == s["sha256"]
    assert hs.resolve_harness_split({"harness_tasks": ["ifeval"]}) is None  # default: old behaviour
    assert hs.resolve_harness_split({"harness_split": None}) is None
    with pytest.raises(ValueError, match="remove"):
        hs.resolve_harness_split({**CFG, "harness_samples": {"ifeval": [1]}})
    with pytest.raises(ValueError, match="is a split of 'ifeval'"):
        hs.resolve_harness_split({**CFG, "harness_tasks": ["gsm8k"]})
    with pytest.raises(ValueError, match="unknown harness_split"):
        hs.resolve_harness_split({"harness_split": "nope"})


def test_calibration_must_not_use_split_documents():
    sel, test, _ = hs.resolve_harness_split(CFG)
    both = {"ifeval": sel["ifeval"] + test["ifeval"]}
    s = hs.load_harness_split("ifeval_k_v1")
    hs.check_calibration_disjoint(both, {"ifeval": sorted(set(range(541)) - set(s["calibration"]))})  # n_calib 40: fine
    order = list(range(541))
    random.Random(0).shuffle(order)
    with pytest.raises(ValueError, match="calibration documents"):  # n_calib 400 (the old Qwen runs): leaks
        hs.check_calibration_disjoint(both, {"ifeval": sorted(order[400:])})


def _run_test_slice(monkeypatch, cfg, alphas):
    import merge_and_rebase.eval.llm_rebase.harness as harness_mod

    calls = []

    def fake_run(**kw):
        calls.append({"alpha_marker": float(next(iter(state.values()))), "samples": kw["samples"]})
        return {"ifeval_prompt_level_strict_acc": 0.1 * state["alpha"]}

    state = {"alpha": 0.0}
    monkeypatch.setattr(harness_mod, "run", fake_run)
    monkeypatch.setattr(asr, "apply_delta", lambda base, scaled: {"alpha": scaled["w"]})
    monkeypatch.setattr(asr, "load_into_model", lambda model, sd, strict=False: state.update(sd))
    out = asr.score_harness_test_slice(
        cfg=cfg, merged_delta={"w": 1.0}, target_base_sd={}, target_llm=SimpleNamespace(model=None, tokenizer=None),
        harness_tasks_resolved=["ifeval"], device="cpu", harness_num_fewshot=0, harness_batch_size=1,
        harness_samples={"ifeval": hs.load_harness_split("ifeval_k_v1")["selection"]}, best_alpha=0.5, alphas=alphas)
    return out, calls


def test_test_slice_default_scores_best_alpha_once_and_split_is_used(monkeypatch):
    (res, samples, curve), calls = _run_test_slice(monkeypatch, dict(CFG), [0.0, 0.5, 1.0])
    assert curve is None and len(calls) == 1 and res == {"ifeval_prompt_level_strict_acc": pytest.approx(0.05)}
    assert samples == {"ifeval": hs.load_harness_split("ifeval_k_v1")["test"]} == calls[0]["samples"]


def test_full_test_curve_scores_every_alpha_and_reuses_best(monkeypatch):
    (res, _, curve), calls = _run_test_slice(monkeypatch, {**CFG, "harness_test_curve": True}, [0.0, 0.5, 1.0])
    assert len(calls) == 3 and sorted(curve) == [0.0, 0.5, 1.0]
    assert curve[1.0]["ifeval_prompt_level_strict_acc"] == pytest.approx(0.1)
    assert res == curve[0.5]


def test_summary_keys_are_additive():
    kw = dict(ignored_block_extension_fields=[], calibration_provenance=None, method_name="m", best_alpha=0.5,
              best_harness_results={"t": 1.0}, harness_test_results={"t": 0.9}, test_samples={"t": [1]},
              baseline_harness_results=None, run_block_extension_prestep=False, harness_results_by_alpha={0.5: {"t": 1.0}},
              delta_stats=None, task_vector_report=None, search_planner=SimpleNamespace(search_summary=lambda: {}),
              harness_search_results=[], saved_merged_path=None)
    off = assemble_harness_summary(**kw)
    assert "harness_results_test_by_alpha" not in off and "harness_split" not in off
    on = assemble_harness_summary(**kw, harness_test_results_by_alpha={0.5: {"t": 0.9}, 0.25: {"t": 0.8}},
                                  harness_split={"name": "ifeval_k_v1"})
    assert list(on["harness_results_test_by_alpha"]) == ["0.25", "0.5"] and on["harness_split"]["name"] == "ifeval_k_v1"
    assert {k: v for k, v in on.items() if k not in ("harness_results_test_by_alpha", "harness_split")} == off
