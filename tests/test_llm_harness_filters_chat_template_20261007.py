"""Harness: filter-aware metric keys, opt-in chat template, opt-in generation dump (lm-eval mocked)."""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch.nn as nn

from merge_and_rebase.eval.llm_rebase import harness


class _Tok:
    def __init__(self, chat_template: str | None = "tmpl") -> None:
        self.chat_template = chat_template


def _install_fake_lm_eval(monkeypatch, calls: list[dict], results: dict) -> None:
    def simple_evaluate(**kwargs):
        calls.append(kwargs)
        return results

    cfg_mod = types.ModuleType("lm_eval.config.task")
    cfg_mod.TaskConfig = type("TaskConfig", (), {"to_dict": lambda self, keep_callable=False: {}})
    monkeypatch.setitem(sys.modules, "lm_eval", types.SimpleNamespace(simple_evaluate=simple_evaluate))
    monkeypatch.setitem(sys.modules, "lm_eval.config", types.ModuleType("lm_eval.config"))
    monkeypatch.setitem(sys.modules, "lm_eval.config.task", cfg_mod)


GSM8K = {
    "results": {
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,strict-match": 0.25,
            "exact_match_stderr,strict-match": 0.01,
            "exact_match,flexible-extract": 0.75,
            "exact_match_stderr,flexible-extract": 0.01,
        }
    }
}
IFEVAL = {"results": {"ifeval": {"prompt_level_strict_acc,none": 0.5, "prompt_level_strict_acc_stderr,none": 0.0}}}


def test_gsm8k_filters_kept_distinct():
    out = harness._extract_metrics(GSM8K)
    assert out == {"gsm8k_exact_match_strict-match": 0.25, "gsm8k_exact_match_flexible-extract": 0.75}
    assert harness.score_by_task(out, ["gsm8k"]) == pytest.approx(0.5)


def test_none_filter_keys_unchanged():
    assert harness._extract_metrics(IFEVAL) == {"ifeval_prompt_level_strict_acc": 0.5}


_OLD_KEYS = {"model", "model_args", "tasks", "num_fewshot", "batch_size", "device", "limit"}


def test_flag_off_call_is_unchanged(monkeypatch):
    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, IFEVAL)
    harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu", num_fewshot=3)
    assert set(calls[0]) == _OLD_KEYS


@pytest.mark.parametrize("n_shot", [5, 0])
def test_chat_template_kwargs(monkeypatch, n_shot):
    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, IFEVAL)
    harness.run(
        ["ifeval"],
        nn.Linear(1, 1),
        _Tok(),
        device="cpu",
        num_fewshot=n_shot,
        apply_chat_template=True,
        system_instruction="sys",
    )
    kw = calls[0]
    assert kw["apply_chat_template"] is True
    assert kw["fewshot_as_multiturn"] is (n_shot > 0)
    assert kw["system_instruction"] == "sys"
    assert "log_samples" not in kw


def test_chat_template_requires_tokenizer_template(monkeypatch):
    _install_fake_lm_eval(monkeypatch, [], IFEVAL)
    with pytest.raises(ValueError, match="chat_template"):
        harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu", apply_chat_template=True)


def test_dump_generations(monkeypatch, tmp_path):
    results = {
        "results": GSM8K["results"],
        "samples": {
            "gsm8k": [
                {
                    "doc_id": 7,
                    "arguments": [["Q: 1+1?", {"until": ["\n"]}]],
                    "resps": [["raw 2"]],
                    "filtered_resps": ["2"],
                    "filter": "strict-match",
                    "metrics": ["exact_match"],
                    "exact_match": 1.0,
                    "target": "2",
                }
            ],
            "arc_easy": [{"doc_id": 0, "arguments": [["ctx", " A"]], "metrics": ["acc"], "acc": 1.0}],
        },
    }
    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, results)
    monkeypatch.setattr(harness, "_DUMP_COUNTER", iter(range(4, 100)))
    harness.run(["gsm8k"], nn.Linear(1, 1), _Tok(), device="cpu", dump_generations_dir=str(tmp_path / "d"))
    assert calls[0]["log_samples"] is True
    files = sorted(p.name for p in (tmp_path / "d").iterdir())
    assert files == ["004_gsm8k.jsonl"]
    rec = json.loads((tmp_path / "d" / files[0]).read_text().splitlines()[0])
    assert rec["doc_id"] == 7 and rec["prompt"] == "Q: 1+1?"
    assert rec["response"] == ["2"] and rec["raw_response"] == [["raw 2"]]
    assert rec["metrics"] == {"exact_match": 1.0} and rec["filter"] == "strict-match"
