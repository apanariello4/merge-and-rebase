"""Drift guard for configs/examples/reference/{vision,llm}/<method>.yaml (annotated reference configs).

Each file must (1) parse with the real resolvers without warnings, (2) have exactly the active keys of its
configs/examples/*.json twin, (3) list every key the code accepts in each nested group, (4) state the true code
default on every commented "# key: value" line, and (5) carry the [BRACE-only] / [vision-only] status tags.
Top-level keys have no single source of truth in code: each listed one must at least appear as a string literal in
the modality's entrypoint modules.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import re
import warnings
from pathlib import Path

import pytest
import yaml

import merge_and_rebase
from merge_and_rebase.eval.llm_rebase.run_config import resolve_llm_method, resolve_llm_run_config
from merge_and_rebase.rebase.block_extension import config as be_config
from merge_and_rebase.rebase.config_schema import canonicalize, legacy_location, legacy_value
from merge_and_rebase.rebase.methods import bico as bico_module
from merge_and_rebase.rebase.methods import theseus as theseus_module
from merge_and_rebase.rebase.methods._ariadne import config as ariadne_config
from merge_and_rebase.rebase.run_config import resolve_run_config
from merge_and_rebase.run_logging import DEFAULT_LOGGING_CONFIG
from merge_and_rebase.utils.helpers import load_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs" / "examples" / "reference"
SRC = Path(merge_and_rebase.__file__).resolve().parent

TWINS = {
    "vision/ariadne": "vision8_ariadne_b16_to_l14.json",
    "vision/theseus": "vision8_theseus_b16_to_l14.json",
    "vision/bico": "vision8_bico_b16_to_l14.json",
    "llm/ariadne": "qwen2.5_0.5b_to_1.5b_ariadne.json",
    "llm/theseus": "qwen2.5_0.5b_to_1.5b_theseus.json",
    "llm/theseus_gqa": "qwen2.5_0.5b_to_1.5b_theseus.json",
    "llm/bico": "qwen2.5_0.5b_to_1.5b_bico.json",
}
TOP_LEVEL_SOURCES = {
    "vision": [
        "eval/vision_rebase",
        "rebase/run_config.py",
        "rebase/merge_modes.py",
        "rebase/runtime.py",
        "rebase/block_extension/config.py",
    ],
    "llm": [
        "eval/llm_rebase",
        "data/llm_calibration.py",
        "hyperparam_search.py",
        "rebase/run_config.py",
        "rebase/block_extension/config.py",
    ],
}
# Passed by the stages; a duplicate in method_params is a TypeError, so the reference files must not list them.
INJECTED = {
    "self",
    "source_model",
    "target_model",
    "source_model_ft",
    "source_dataloader",
    "target_dataloader",
    "source_recipe",
    "target_recipe",
    "target_base",
    "delta",
    "device",
    "family_adapter",
    "source_activation_plan",
}

_CANONICAL_BLOCKS = {"models", "method", "depth_alignment", "merge", "alpha", "save", "load"}
_KEY_LINE = re.compile(r"^(?P<ind> *)(?P<hash># )?(?P<key>[a-z_][a-z0-9_]*):(?P<rest>.*)$")


@dataclasses.dataclass
class Entry:
    path: tuple[str, ...]
    commented: bool
    raw_value: str
    comment: str

    @property
    def is_header(self) -> bool:
        return self.raw_value == ""

    @property
    def value(self):
        return yaml.safe_load(self.raw_value)


def parse_entries(text: str) -> list[Entry]:
    """Every key line, active or commented, with its nesting path (indentation-based)."""
    entries: list[Entry] = []
    stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        m = _KEY_LINE.match(line)
        if m is None:
            continue
        indent = len(m["ind"])
        rest = m["rest"]
        hit = re.search(r"\s#", rest)
        value, comment = (rest[: hit.start()], rest[hit.end() :]) if hit else (rest, "")
        value = value.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(k for _, k in stack) + (m["key"],)
        entries.append(Entry(path, bool(m["hash"]), value, comment))
        if value == "":
            stack.append((indent, m["key"]))
    return entries


def _cases():
    return sorted(TWINS)


def _load(case: str) -> tuple[str, dict, list[Entry]]:
    """The file as written: canonical config and canonical entry paths."""
    path = REFERENCE / f"{case}.yaml"
    return path.read_text(), load_json(path), parse_entries(path.read_text())


def _legacy_entries(case: str) -> list[Entry]:
    """The file's entries at their legacy flat paths (where the code's dataclasses and keyword names live)."""
    direct_fit = case.endswith("ariadne")
    out = []
    for e in _load(case)[2]:
        if e.is_header and e.path[-1] in ("data", "gradient") and len(e.path) == 3:
            continue  # canonical-only sub-blocks: their leaves map to flat legacy keys
        path = legacy_location(e.path, direct_fit=direct_fit)
        raw = e.raw_value
        if not e.is_header:
            value = legacy_value(e.path, e.value)
            if value != e.value:
                raw = json.dumps(value)
        out.append(Entry(path, e.commented, raw, e.comment))
    return out


def _group(entries: list[Entry], *prefix: str) -> dict[str, Entry]:
    n = len(prefix)
    return {e.path[n]: e for e in entries if len(e.path) == n + 1 and e.path[:n] == prefix}


def _norm(value):
    return list(value) if isinstance(value, tuple) else value


def _kwargs_defaults(module) -> dict[str, object]:
    """``kwargs.pop/get("name", default)`` reads in a method module (keys the signature does not name)."""
    found = {}
    for name, default in re.findall(r"kwargs\.(?:pop|get)\(\"([a-z_]+)\"(?:, ([^)]*))?\)", inspect.getsource(module)):
        found[name] = ast.literal_eval(default) if default else None
    return found


def _method_params_contract(method: str, modality: str) -> dict[str, object]:
    cls, module = (
        (bico_module.BiCoRebase, bico_module) if method == "bico" else (theseus_module.TheseusRebase, theseus_module)
    )
    sig = inspect.signature(cls.prepare)
    accepted = {
        name: p.default
        for name, p in sig.parameters.items()
        if name not in INJECTED and p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    for name, default in _kwargs_defaults(module).items():
        if name not in INJECTED:
            accepted.setdefault(name, default)
    for legacy in ("n_batches", "patch_qkv"):  # legacy aliases of num_batches / split_qkv (rebase/config_schema.py)
        accepted.pop(legacy, None)
    return accepted


# ---------------------------------------------------------------- (1) real resolvers, no warnings
def _resolve_and_check(case: str, cfg: dict, *, allow_brace_only_warning: bool = False) -> None:
    modality, method = case.split("/")
    cfg = dict(canonicalize(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if allow_brace_only_warning:
            warnings.filterwarnings("ignore", message=r".*ignores the BRACE-only", category=RuntimeWarning)
        if modality == "vision":
            cfg = dict(cfg)
            cfg.setdefault("block_extension_enabled", True)  # as the CLI does
            resolved = resolve_run_config(cfg)
            assert resolved.method_name == method
            plan = resolved.bind(12, 24)
            if method == "theseus":
                assert plan.run_block_extension_prestep and resolved.block_extension_cfg.skip_correction is True
            if method == "bico":
                assert plan.run_discrete_layer_match_prestep and not plan.run_block_extension_prestep
            if method == "ariadne":
                assert resolved.ariadne_preset == "ariadne"
        else:
            method_name, method_obj, method_params = resolve_llm_method(cfg)
            assert method_name == method
            cfg = dict(cfg)
            cfg.setdefault("block_extension_enabled", True)  # as context.build does
            enabled, be_cfg = be_config.resolve_block_extension_config(cfg)
            assert be_config.warn_decoder_ignored_fields(cfg.get("block_extension_params")) == []
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
            plan = resolved.bind(24, 28)
            if method in ("theseus", "theseus_gqa"):
                assert plan.run_block_extension_prestep and resolved.block_extension_cfg.skip_correction is True
            if method == "bico":
                assert plan.run_discrete_layer_match_prestep and not plan.run_block_extension_prestep
            if method == "ariadne":
                ari = ariadne_config.resolve_ariadne_decoder_config(cfg["ariadne_params"], object())
                assert ari.components == ("mlp.c_proj",) and ari.missing_bias == "materialize"


@pytest.mark.parametrize("case", _cases())
def test_reference_config_resolves(case):
    _resolve_and_check(case, _load(case)[1])


@pytest.mark.parametrize("case", _cases())
def test_uncommenting_one_line_keeps_the_run(case):
    """Each commented line, uncommented alone at its stated value, still resolves to the same depth plan.

    Under BiCo's discrete_index_match a set BRACE-only key warns that it is inert (documented in the files).
    """
    _, cfg, entries = _load(case)
    leaves = [e for e in entries if e.commented and not e.is_header and e.path[0] != "logging"]
    assert leaves
    for entry in leaves:
        variant = copy.deepcopy(cfg)
        node = variant
        for key in entry.path[:-1]:
            node = node.setdefault(key, {})
        node[entry.path[-1]] = entry.value
        _resolve_and_check(case, variant, allow_brace_only_warning=case.endswith("bico"))


# ---------------------------------------------------------------- (2) same run as the JSON twin
@pytest.mark.parametrize("case", _cases())
def test_reference_config_matches_example(case):
    _, cfg, _ = _load(case)
    twin = json.loads((ROOT / "configs" / "examples" / TWINS[case]).read_text())
    if case == "llm/theseus_gqa":
        twin["method"]["name"] = "theseus_gqa"
    assert cfg == twin


# ---------------------------------------------------------------- (3) complete, (4) true defaults
@pytest.mark.parametrize("case", [c for c in _cases() if not c.endswith("ariadne")])
def test_block_extension_params_complete_and_defaults(case):
    entries = _legacy_entries(case)
    group = _group(entries, "block_extension_params")
    fields = {f.name: f for f in dataclasses.fields(be_config.BlockExtensionConfig)}
    assert set(fields) | {"depth_rule"} <= set(group), sorted(set(fields) - set(group))
    assert set(group) <= set(fields) | {"depth_rule"}
    for key, entry in group.items():
        if not entry.commented or entry.is_header:
            continue
        if key == "depth_rule":
            assert entry.value == "method_default"
        elif "[default: method]" in entry.comment:
            continue  # checked in test_reference_config_resolves
        else:
            assert entry.value == fields[key].default, key
    tsc = _group(entries, "block_extension_params", "target_shared_correction")
    if case.startswith("vision"):
        tsc_fields = {f.name: f for f in dataclasses.fields(be_config.TargetSharedCorrection)}
        assert set(tsc) == set(tsc_fields) | {"enabled"}
        for key, entry in tsc.items():
            if key == "enabled":
                assert entry.value is True
            elif tsc_fields[key].default is not dataclasses.MISSING:
                assert entry.value == tsc_fields[key].default, key


@pytest.mark.parametrize("case", ["vision/ariadne", "llm/ariadne"])
def test_ariadne_params_complete_and_defaults(case):
    entries = _legacy_entries(case)
    group = _group(entries, "ariadne_params")
    fields = {f.name: f for f in dataclasses.fields(ariadne_config.DirectResidualConfig)}
    assert set(group) == set(fields) | {"preset"}, sorted((set(fields) | {"preset"}) ^ set(group))
    preset = ariadne_config._PRESETS["ariadne"]
    for key, entry in group.items():
        if not entry.commented:
            continue
        if "[default: preset]" in entry.comment:
            expected = preset[key]
        elif "[default: decoder]" in entry.comment:
            expected = ariadne_config.DECODER_DEFAULTS[key]
        else:
            expected = fields[key].default
        assert entry.value == _norm(expected), key
    if case == "llm/ariadne":
        for key, _ in ariadne_config._DECODER_REQUIRED:
            assert "[vision-only]" in group[key].comment, key


@pytest.mark.parametrize("case", [c for c in _cases() if not c.endswith("ariadne")])
def test_method_params_complete_and_defaults(case):
    modality, method = case.split("/")
    entries = _legacy_entries(case)
    group = _group(entries, "method_params")
    contract = _method_params_contract(method, modality)
    assert set(group) == set(contract), sorted(set(group) ^ set(contract))
    for key, entry in group.items():
        if entry.commented:
            assert entry.value == contract[key], key


# ---------------------------------------------------------------- top-level keys exist in code
@pytest.mark.parametrize("case", _cases())
def test_top_level_keys_are_read_by_the_entrypoint(case):
    modality = case.split("/")[0]
    entries = _legacy_entries(case)
    text = "\n".join(
        p.read_text()
        for rel in TOP_LEVEL_SOURCES[modality]
        for p in ((SRC / rel).glob("*.py") if (SRC / rel).is_dir() else [SRC / rel])
    )
    keys = {e.path[0] for e in entries if len(e.path) == 1 and not (e.is_header and e.path[0] in _CANONICAL_BLOCKS)}
    missing = sorted(k for k in keys if f'"{k}"' not in text and f"'{k}'" not in text)
    assert not missing, missing


def test_logging_defaults():
    for case in ("vision/theseus", "llm/theseus"):
        _, _, entries = _load(case)
        group = _group(entries, "logging")
        assert {k: e.value for k, e in group.items()} == DEFAULT_LOGGING_CONFIG


# ---------------------------------------------------------------- (5) status tags
@pytest.mark.parametrize("case", [c for c in _cases() if not c.endswith("ariadne")])
def test_status_tags(case):
    entries = _legacy_entries(case)
    group = _group(entries, "block_extension_params")
    for key in be_config._BRACE_ONLY_FIELDS:
        assert "[BRACE-only]" in group[key].comment, key
    if case.startswith("llm"):
        for key in be_config._DECODER_IGNORED_FIELDS:
            assert "[vision-only]" in group[key].comment, key
