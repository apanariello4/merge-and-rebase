"""Canonical (nested) run-config schema and the single table of the legacy flat keys it replaces.

The resolvers read the legacy flat keys (``ResolvedRunConfig`` and the LLM context are unchanged): ``canonicalize``
translates a canonical config to that flat form. Legacy keys stay accepted; ``canonicalize`` reports which ones a
config used so the entrypoints can warn. Switching the legacy names off later means emptying ``LEGACY_KEYS``.

Canonical layout (every block optional; unlisted keys pass through unchanged)::

    method:          {name, params: {..., data: {...}, gradient: {...}}}
    depth_alignment: {rule, enabled, defaults, brace: {..., correction_endpoint, data: {...}}}
    models:          {source: {model, pretrained}, target: {model, pretrained}, tuned_checkpoints}
    merge:           {mode, method, params, weights, base_construction, global_alpha_search}
    alpha:           {value, search, min, max, step, selection, split, patience}
    save:            {merged, transported_task_vectors, transported_task_vectors_policy,
                      transported_artifacts, legacy_task_vector_layouts}
    load:            {transported_task_vectors}
    humanize_classnames: bool          (legacy ``no_humanize`` with the opposite meaning)

``method.params`` holds the method's own parameters: ``ariadne_params`` for Ariadne, ``method_params`` otherwise.
The diagnostics keys (``eval_before_rebase``, ``source_lmc_*``, ``source_only``, ...) keep their current names.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

#: (canonical dotted path, legacy flat key). Paths are relative to the config root.
LEGACY_KEYS: tuple[tuple[str, str], ...] = (
    ("method.name", "method"),
    ("method.params.data.source", "transport_calibration_data"),
    ("method.params.data.protocol", "transport_calibration_protocol"),
    ("method.params.data.num_batches", "transport_calibration_batches"),
    ("method.params.data.split", "transport_calibration_split"),
    ("method.params.data.max_samples", "transport_calibration_max_samples"),
    ("method.params.gradient.batch_size", "grad_batch_size"),
    ("method.params.gradient.images_per_class", "grad_imgs_per_class"),
    ("method.params.gradient.num_batches", "grad_num_batches"),
    ("depth_alignment.enabled", "block_extension_enabled"),
    ("depth_alignment.defaults", "depth_defaults"),
    ("models.source.model", "source_clip_model"),
    ("models.source.pretrained", "source_clip_pretrained"),
    ("models.target.model", "target_clip_model"),
    ("models.target.pretrained", "target_clip_pretrained"),
    ("models.tuned_checkpoints", "tuned_ckpts"),
    ("merge.mode", "merge_mode"),
    ("merge.method", "merge_method"),
    ("merge.params", "merge_params"),
    ("merge.weights", "weights"),
    ("merge.base_construction", "base_construction"),
    ("merge.global_alpha_search", "global_alpha_search"),
    ("alpha.value", "alpha"),
    ("alpha.search", "alpha_search"),
    ("alpha.min", "alpha_min"),
    ("alpha.max", "alpha_max"),
    ("alpha.step", "alpha_step"),
    ("alpha.selection", "alpha_selection"),
    ("alpha.split", "alpha_search_split"),
    ("alpha.patience", "alpha_patience"),
    ("save.merged", "save_merged"),
    ("save.transported_task_vectors", "save_transported_tvs_dir"),
    ("save.transported_task_vectors_policy", "save_transported_tvs"),
    ("save.transported_artifacts", "save_transported_artifacts"),
    ("save.legacy_task_vector_layouts", "save_transported_tvs_legacy"),
    ("load.transported_task_vectors", "load_direct_residual_tvs_dir"),
)
#: Legacy keys whose content moves under ``method.params`` (the method's own parameter dict).
LEGACY_METHOD_PARAM_KEYS = ("method_params", "ariadne_params", "direct_residual_params", "mask_mode", "vote")
#: ``depth_alignment.brace`` <-> ``block_extension_params``: renamed keys inside the block.
BRACE_RENAMES: tuple[tuple[str, str], ...] = (
    ("data.dataset", "calibration_dataset"),
    ("data.task", "calibration_task"),
    ("data.split", "calibration_split"),
    ("data.num_batches", "n_batches_act"),
    ("correction_endpoint", "lmc_mode"),
)
#: ``correction_endpoint`` value <-> legacy ``lmc_mode`` value.
CORRECTION_ENDPOINTS = {"per_endpoint": "independent", "base": "shared", "finetuned": "shared_ft"}
#: Legacy ``depth_alignment`` string -> ``depth_alignment.rule``.
LEGACY_DEPTH_ALIGNMENT = {"ariadne": "brace", "discrete_index_match": "discrete_index_match"}

_DIRECT_FIT_NAMES = ("ariadne", "direct_residual")
_BLOCKS = ("method", "depth_alignment", "models", "merge", "alpha", "save", "load")


class InternalConfig(dict):
    """A config already translated to the flat form the resolvers read (``canonicalize`` passes it through)."""


def _get(cfg: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _pop(cfg: dict[str, Any], path: str) -> tuple[bool, Any]:
    parts = path.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return False, None
    return True, node.pop(parts[-1])


def _set(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _is_canonical_block(key: str, value: Any) -> bool:
    # ``method`` / ``depth_alignment`` are also legacy keys, but legacy values are strings.
    return key in _BLOCKS and isinstance(value, Mapping)


def legacy_keys_used(cfg: Mapping[str, Any]) -> list[str]:
    """Legacy flat keys present in a user config (the deprecation warning lists them)."""
    if isinstance(cfg, InternalConfig):
        return []
    used = [legacy for _, legacy in LEGACY_KEYS if legacy in cfg and not _is_canonical_block(legacy, cfg[legacy])]
    used += [k for k in LEGACY_METHOD_PARAM_KEYS if k in cfg]
    if "depth_alignment" in cfg and not isinstance(cfg["depth_alignment"], Mapping):
        used.append("depth_alignment")
    if "block_extension_params" in cfg:
        used.append("block_extension_params")
    if "no_humanize" in cfg:
        used.append("no_humanize")
    method_params = cfg.get("method_params")
    if isinstance(method_params, Mapping) and "n_batches" in method_params:
        used.append("method_params.n_batches")
    return used


def canonicalize(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Translate a canonical (or legacy, or mixed) config to the flat form the resolvers read.

    A legacy key and its canonical spelling may not both be set. A config that is already flat is returned as is;
    otherwise a new dict is returned (``cfg`` is not modified).
    """
    if isinstance(cfg, InternalConfig):
        return cfg
    if "humanize_classnames" not in cfg and not any(_is_canonical_block(k, v) for k, v in cfg.items()):
        return cfg  # already flat: returned as is (same object), so legacy configs behave exactly as before
    src = _deep_copy(dict(cfg))
    out: dict[str, Any] = {k: v for k, v in src.items() if not _is_canonical_block(k, v)}
    canon = {k: v for k, v in src.items() if _is_canonical_block(k, v)}

    def put(legacy: str, value: Any, canonical_path: str) -> None:
        if legacy in out:
            raise ValueError(f"config sets both '{canonical_path}' and its legacy spelling '{legacy}'; use one")
        out[legacy] = value

    for path, legacy in LEGACY_KEYS:
        found, value = _pop(canon, path)
        if found:
            put(legacy, value, path)

    method = canon.get("method") or {}
    params = method.pop("params", None) if isinstance(method, dict) else None
    if isinstance(params, dict):
        for block in ("data", "gradient"):  # emptied by the LEGACY_KEYS pass above
            if params.get(block) == {}:
                params.pop(block)
    if params is not None:
        name = str(out.get("method", ""))
        put("ariadne_params" if name in _DIRECT_FIT_NAMES else "method_params", params, "method.params")

    depth = canon.get("depth_alignment") or {}
    if isinstance(depth, dict):
        rule_found, rule = _pop(depth, "rule")
        brace = depth.pop("brace", None)
        if brace is not None or rule_found:
            params_out: dict[str, Any] = dict(brace or {})
            for new, old in BRACE_RENAMES:
                found, value = _pop(params_out, new)
                if found:
                    if old in params_out:
                        raise ValueError(
                            f"depth_alignment.brace sets both '{new}' and its legacy spelling '{old}'; use one"
                        )
                    if new == "correction_endpoint":
                        if value not in CORRECTION_ENDPOINTS:
                            raise ValueError(
                                f"depth_alignment.brace.correction_endpoint must be one of {sorted(CORRECTION_ENDPOINTS)}"
                            )
                        value = CORRECTION_ENDPOINTS[value]
                    params_out[old] = value
            if isinstance(params_out.get("data"), dict) and not params_out["data"]:
                params_out.pop("data")
            if rule_found:
                params_out["depth_rule"] = rule
            put("block_extension_params", params_out, "depth_alignment.brace")

    if "humanize_classnames" in out:
        if "no_humanize" in out:
            raise ValueError("config sets both 'humanize_classnames' and its legacy inverse 'no_humanize'; use one")
        out["no_humanize"] = not bool(out.pop("humanize_classnames"))

    leftovers = {k: v for k, v in canon.items() if _non_empty(v)}
    if leftovers:
        raise ValueError(f"unknown canonical config keys: {_paths(leftovers)}")
    return InternalConfig(out)


def _non_empty(value: Any) -> bool:
    return not isinstance(value, Mapping) or any(_non_empty(v) for v in value.values())


def _paths(tree: Mapping[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in tree.items():
        if isinstance(value, Mapping) and value:
            out += _paths(value, f"{prefix}{key}.")
        elif _non_empty(value):
            out.append(f"{prefix}{key}")
    return out


def to_canonical(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a legacy flat config in the canonical layout (inverse of ``canonicalize`` on legacy configs)."""
    src = _deep_copy(dict(cfg))
    out: dict[str, Any] = {}
    for path, legacy in LEGACY_KEYS:
        if legacy in src and not _is_canonical_block(legacy, src[legacy]):
            _set(out, path, src.pop(legacy))
    sources = [key for key in ("method_params", "ariadne_params", "direct_residual_params") if key in src]
    non_empty = [key for key in sources if src[key]]
    if len(non_empty) > 1:
        raise ValueError(f"config sets more than one method parameter dict: {non_empty}")
    if sources:
        chosen = src[non_empty[0]] if non_empty else src[sources[0]]
        for key in sources:
            src.pop(key)
        params = out.setdefault("method", {}).setdefault("params", {})
        for key, value in dict(chosen).items():
            if key in params:
                raise ValueError(f"method parameter '{key}' collides with a canonical method.params entry")
            params[key] = value
    for key in ("mask_mode", "vote"):  # GradFix's top-level fallbacks
        if key in src:
            _set(out, f"method.params.{key}", src.pop(key))
    legacy_rule = src.pop("depth_alignment", None) if isinstance(src.get("depth_alignment"), str) else None
    brace = src.pop("block_extension_params", None)
    if brace is not None or legacy_rule is not None:
        brace = dict(brace or {})
        rule = brace.pop("depth_rule", None)
        if legacy_rule is not None:
            rule = rule if rule is not None else LEGACY_DEPTH_ALIGNMENT.get(legacy_rule, legacy_rule)
        for new, old in BRACE_RENAMES:
            if old in brace:
                value = brace.pop(old)
                if new == "correction_endpoint":
                    value = {v: k for k, v in CORRECTION_ENDPOINTS.items()}.get(value, value)
                _set(brace, new, value)
        if rule is not None:
            _set(out, "depth_alignment.rule", rule)
        if brace:
            _set(out, "depth_alignment.brace", brace)
    if "no_humanize" in src:
        out["humanize_classnames"] = not bool(src.pop("no_humanize"))
    out.update(src)
    return out


def load_run_config(cfg: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
    """Entrypoint helper: warn once about legacy key names, then ``canonicalize``. Returns ``(config, legacy_keys)``."""
    legacy = legacy_keys_used(cfg)
    if legacy:
        warnings.warn(
            f"config uses legacy key names {legacy}; they still work. Canonical names: "
            "merge_and_rebase/rebase/config_schema.py (LEGACY_KEYS).",
            FutureWarning,
            stacklevel=2,
        )
    return canonicalize(cfg), legacy
