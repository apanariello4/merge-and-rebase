"""Every config under configs/examples parses with the real resolvers and states its budget fields explicitly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merge_and_rebase.eval.llm_rebase.run_config import resolve_llm_method
from merge_and_rebase.rebase.config_schema import canonicalize
from merge_and_rebase.rebase.methods._ariadne.config import parse_direct_residual_config
from merge_and_rebase.rebase.run_config import resolve_run_config

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "configs" / "examples").glob("*.json"))


def test_examples_exist():
    assert len(EXAMPLES) >= 6


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_config_parses(path):
    cfg = dict(canonicalize(json.loads(path.read_text())))  # the examples use the canonical names
    params = cfg.get("ariadne_params") or cfg.get("method_params") or {}
    # Per-experiment budget fields are never implied: every example states num_batches and seed.
    assert "num_batches" in params and "seed" in params, path.name
    assert "seed" in cfg
    if cfg["method"] == "ariadne":
        parse_direct_residual_config(cfg["ariadne_params"])
    if "suite" in cfg:
        resolved = resolve_run_config(cfg)
        assert resolved.method_name == cfg["method"]
    else:
        method_name, _, _ = resolve_llm_method(cfg)
        assert method_name == cfg["method"]
