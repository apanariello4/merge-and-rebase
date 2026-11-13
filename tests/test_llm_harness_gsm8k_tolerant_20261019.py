"""Bundled gsm8k_cot_tolerant task (opt-in harness_include_path="bundled"): same prompts, few-shot context and
generation settings as lm-eval's gsm8k_cot; only the final-answer reader is format tolerant (2026-10-19)."""

from __future__ import annotations

import re

import pytest
import torch.nn as nn
from test_llm_harness_filters_chat_template_20261007 import _OLD_KEYS, IFEVAL, _install_fake_lm_eval, _Tok

from merge_and_rebase.eval.llm_rebase import harness


def test_include_path_is_opt_in(monkeypatch):
    calls: list[dict] = []
    _install_fake_lm_eval(monkeypatch, calls, IFEVAL)
    monkeypatch.setattr(harness, "_task_manager", lambda p: ("tm", p))
    harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu")
    assert set(calls[0]) == _OLD_KEYS
    harness.run(["ifeval"], nn.Linear(1, 1), _Tok(None), device="cpu", include_path="bundled")
    assert calls[1]["task_manager"] == ("tm", "bundled")


def test_bad_include_path_raises(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        harness._task_manager(str(tmp_path / "missing"))


def _tasks():
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import get_task_dict

    tm = harness._task_manager("bundled")
    try:
        return get_task_dict(["gsm8k_cot", "gsm8k_cot_tolerant"], tm)
    except Exception as exc:  # dataset not cached
        pytest.skip(f"gsm8k not available offline: {exc}")


def test_tolerant_task_matches_gsm8k_cot_except_reader():
    tasks = _tasks()
    ref, tol = tasks["gsm8k_cot"], tasks["gsm8k_cot_tolerant"]
    ref_cfg, tol_cfg = ref.config.to_dict(), tol.config.to_dict()
    for key in ("dataset_path", "dataset_name", "test_split", "doc_to_text", "doc_to_target", "num_fewshot",
                "generation_kwargs", "metric_list", "fewshot_config", "output_type"):
        assert ref_cfg.get(key) == tol_cfg.get(key), key
    docs_ref, docs_tol = list(ref.eval_docs), list(tol.eval_docs)
    assert len(docs_ref) == len(docs_tol) == 1319
    for i in (0, 7, 1318):
        assert ref.fewshot_context(docs_ref[i], num_fewshot=8) == tol.fewshot_context(docs_tol[i], num_fewshot=8)
    assert [f["name"] for f in tol_cfg["filter_list"]] == ["tolerant-extract"]


def test_tolerant_reader_formats():
    tasks = _tasks()
    pattern = re.compile(tasks["gsm8k_cot_tolerant"].config.to_dict()["filter_list"][0]["filter"][0]["regex_pattern"])
    cases = {
        "So 2 + 1 = 3. The answer is 3.": "3",
        "The answer is: $3$.": "3",
        "The answer is $\\boxed{3}$.": "3",
        "The answer is: 1,000.": "1,000",
        "The answer is $\\boxed{-2.5}$": "-2.5",
    }
    for text, want in cases.items():
        assert pattern.search(text).group(1) == want, text
    assert pattern.search("The answer is three.") is None
