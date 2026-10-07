"""Pin the fully resolved run config of every example / reference config (refactor safety net).

Config refactors (key renames, nesting, aliases, method taxonomy) must leave what a config *means* unchanged. For each
config under ``configs/examples`` this test resolves it with the real resolvers, binds the run plan at fixed depths
and compares the normalized result with ``tests/golden/resolved_configs.json``. Regenerate deliberately with
``GOLDEN_CAPTURE_RESOLVED=1 pytest tests/test_resolved_config_snapshot.py`` and review the diff.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path

import pytest

from merge_and_rebase.eval.llm_rebase.run_config import resolve_llm_method, resolve_llm_run_config
from merge_and_rebase.rebase.block_extension import config as be_config
from merge_and_rebase.rebase.config_schema import canonicalize
from merge_and_rebase.rebase.run_config import resolve_run_config
from merge_and_rebase.utils.helpers import load_json

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "configs" / "examples"
SNAPSHOT = ROOT / "tests" / "golden" / "resolved_configs.json"
# (source_depth, target_depth) pairs each config is bound at: equal, extend, shrink.
DEPTHS = {"vision": [(12, 12), (12, 24), (24, 12)], "llm": [(24, 24), (24, 28), (28, 24)]}


def _configs() -> list[str]:
    paths = sorted(EXAMPLES.glob("*.json")) + sorted((EXAMPLES / "reference").glob("*/*.yaml"))
    return [str(p.relative_to(EXAMPLES)) for p in paths]


def _modality(rel: str, cfg: dict) -> str:
    if rel.startswith("reference/"):
        return rel.split("/")[1]
    return "llm" if "model_arch" in cfg or "tuned_bodies" in cfg or "qwen" in rel else "vision"


def _normalize(obj):
    """JSON-stable view: dataclasses by field, enums by value, objects by type name (never by id/address)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _normalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return _normalize(obj.value)
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_normalize(v) for v in obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return f"<{type(obj).__module__}.{type(obj).__qualname__}>"


def _resolve(rel: str) -> dict:
    cfg = dict(canonicalize(load_json(EXAMPLES / rel)))  # the examples use the canonical names
    modality = _modality(rel, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg.setdefault("block_extension_enabled", True)  # as both CLIs do
        if modality == "vision":
            resolved = resolve_run_config(cfg)
        else:
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
        plans = {f"{s}->{t}": _normalize(resolved.bind(s, t)) for s, t in DEPTHS[modality]}
    view = _normalize(resolved)
    view.pop("cfg", None)  # the raw input dict: it changes with every key rename, its meaning must not
    return {"modality": modality, "resolved": view, "plans": plans}


def _capture(rel: str, value: dict) -> None:
    data = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    data[rel] = value
    SNAPSHOT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("rel", _configs())
def test_resolved_config_snapshot(rel):
    actual = json.loads(json.dumps(_resolve(rel)))
    if os.environ.get("GOLDEN_CAPTURE_RESOLVED"):
        _capture(rel, actual)
        return
    expected = json.loads(SNAPSHOT.read_text()).get(rel)
    assert expected is not None, f"no snapshot for {rel}; capture with GOLDEN_CAPTURE_RESOLVED=1"
    assert actual == expected, f"{rel}: resolved config changed\n{_diff(expected, actual)}"


def _diff(expected, actual, path: str = "") -> str:
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                lines.append(f"  removed {path}{key}")
            elif key not in expected:
                lines.append(f"  added   {path}{key} = {actual[key]!r}")
            elif expected[key] != actual[key]:
                lines.append(_diff(expected[key], actual[key], f"{path}{key}."))
        return "\n".join(line for line in lines if line)
    return f"  changed {path.rstrip('.')}: {expected!r} -> {actual!r}"
