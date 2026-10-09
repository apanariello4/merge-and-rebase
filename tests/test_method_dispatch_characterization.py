"""Pin how every registered rebase method is dispatched (refactor safety net for the method taxonomy).

For each registered method and both entrypoints (vision, llm) the resolved run config decides which stage runs and
which depth prestep applies. The table below was captured before the MethodFamily / traits refactor; any change to
it must be deliberate. Regenerate with ``GOLDEN_PRINT_DISPATCH=1 pytest -s tests/test_method_dispatch_characterization.py``.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

from merge_and_rebase.rebase.config_schema import canonicalize
from merge_and_rebase.rebase.registry import list_methods
from merge_and_rebase.utils.helpers import load_json

ROOT = Path(__file__).resolve().parents[1]
VISION_BASE = ROOT / "configs" / "examples" / "vision8_theseus_b16_to_l14.json"
LLM_BASE = ROOT / "configs" / "examples" / "qwen2.5_0.5b_to_1.5b_theseus.json"
DEPTHS = {"vision": (12, 24), "llm": (24, 28)}

# fmt: off
EXPECTED = {
    "ariadne/vision": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "brace", "direct_fit": True, "method_name": "ariadne", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "ariadne/llm": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "none", "direct_fit": True, "method_name": "ariadne", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "bico/vision": {"bico_mode": True, "depth_prestep_method": True, "depth_rule": "discrete_index_match", "direct_fit": False, "method_name": "bico", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": True, "theseus_mode": False, "transfusion_mode": False},
    "bico/llm": {"bico_mode": True, "depth_prestep_method": True, "depth_rule": "discrete_index_match", "direct_fit": False, "method_name": "bico", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": True, "theseus_mode": False, "transfusion_mode": False},
    "direct_residual/vision": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "brace", "direct_fit": True, "method_name": "direct_residual", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "direct_residual/llm": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "none", "direct_fit": True, "method_name": "direct_residual", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "gradfix/vision": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "brace", "direct_fit": False, "method_name": "gradfix", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "gradfix/llm": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "none", "direct_fit": False, "method_name": "gradfix", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "identity/vision": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "brace", "direct_fit": False, "method_name": "identity", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "identity/llm": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "none", "direct_fit": False, "method_name": "identity", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "orthogonal_shift/vision": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "brace", "direct_fit": False, "method_name": "orthogonal_shift", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "orthogonal_shift/llm": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "none", "direct_fit": False, "method_name": "orthogonal_shift", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "theseus/vision": {"bico_mode": False, "depth_prestep_method": True, "depth_rule": "brace", "direct_fit": False, "method_name": "theseus", "run_block_extension_prestep": True, "run_discrete_layer_match_prestep": False, "theseus_mode": True, "transfusion_mode": False},
    "theseus/llm": {"bico_mode": False, "depth_prestep_method": True, "depth_rule": "brace", "direct_fit": False, "method_name": "theseus", "run_block_extension_prestep": True, "run_discrete_layer_match_prestep": False, "theseus_mode": True, "transfusion_mode": False},
    "theseus_gqa/vision": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "brace", "direct_fit": False, "method_name": "theseus_gqa", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "theseus_gqa/llm": {"bico_mode": False, "depth_prestep_method": True, "depth_rule": "brace", "direct_fit": False, "method_name": "theseus_gqa", "run_block_extension_prestep": True, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": False},
    "transfusion/vision": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "brace", "direct_fit": False, "method_name": "transfusion", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": True},
    "transfusion/llm": {"bico_mode": False, "depth_prestep_method": False, "depth_rule": "none", "direct_fit": False, "method_name": "transfusion", "run_block_extension_prestep": False, "run_discrete_layer_match_prestep": False, "theseus_mode": False, "transfusion_mode": True},
}
# fmt: on


def _method_default_depth(cfg: dict) -> dict:
    """The base example states THESEUS's explicit depth rule; this pin swaps the method, so use the per-method default."""
    params = dict(cfg.get("block_extension_params") or {})
    params.pop("depth_rule", None)
    cfg["block_extension_params"] = params
    cfg["depth_defaults"] = "method"
    return cfg


def _vision(method: str) -> dict:
    from merge_and_rebase.rebase.run_config import resolve_run_config

    cfg = _method_default_depth(dict(canonicalize(load_json(VISION_BASE))))
    cfg.update(method=method, method_params={}, block_extension_enabled=True)
    resolved = resolve_run_config(cfg)
    plan = resolved.bind(*DEPTHS["vision"])
    return _record(resolved, plan)


def _llm(method: str) -> dict:
    from merge_and_rebase.eval.llm_rebase.run_config import resolve_llm_method, resolve_llm_run_config
    from merge_and_rebase.rebase.block_extension import config as be_config

    cfg = _method_default_depth(dict(canonicalize(load_json(LLM_BASE))))
    cfg.update(method=method, method_params={}, block_extension_enabled=True)
    method_name, method_obj, method_params = resolve_llm_method(cfg)
    enabled, be_cfg = be_config.resolve_block_extension_config(cfg)
    resolved = resolve_llm_run_config(
        cfg,
        method=method_obj,
        method_name=method_name,
        method_params=method_params,
        block_extension_enabled=enabled,
        block_extension_cfg=be_cfg,
        device="cpu",
        eval_before_rebase_only=False,
    )
    plan = resolved.bind(*DEPTHS["llm"])
    return _record(resolved, plan)


def _record(resolved, plan) -> dict:
    return {
        "method_name": resolved.method_name,
        "direct_fit": bool(resolved.direct_fit),
        "theseus_mode": bool(resolved.theseus_mode),
        "bico_mode": bool(resolved.bico_mode),
        "transfusion_mode": bool(resolved.transfusion_mode),
        "depth_prestep_method": bool(resolved.depth_prestep_method),
        "depth_rule": resolved.depth_rule.kind,
        "run_block_extension_prestep": plan.run_block_extension_prestep,
        "run_discrete_layer_match_prestep": plan.run_discrete_layer_match_prestep,
    }


def _outcome(modality: str, method: str) -> dict | str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (_vision if modality == "vision" else _llm)(method)
    except Exception as exc:  # the error type is part of the dispatch contract (unsupported pairings)
        return f"error:{type(exc).__name__}"


CASES = [(m, d) for m in sorted(list_methods()) for d in ("vision", "llm")]


@pytest.mark.parametrize("method,modality", CASES)
def test_method_dispatch_is_pinned(method, modality):
    actual = _outcome(modality, method)
    if os.environ.get("GOLDEN_PRINT_DISPATCH"):
        print(f"\nDISPATCH {json.dumps([method, modality])}: {json.dumps(actual, sort_keys=True)},")
        return
    assert f"{method}/{modality}" in EXPECTED, f"no pinned dispatch for {method}/{modality}: {actual}"
    assert actual == EXPECTED[f"{method}/{modality}"]
