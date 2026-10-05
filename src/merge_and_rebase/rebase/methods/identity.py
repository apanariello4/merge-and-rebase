from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ...merge.task_vectors import TaskVector
from ..base import TensorDict
from ..registry import register

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_MODULE_SCOPE_TOKENS = {"mlp": ".mlp.", "attn": ".self_attn."}


def _filter_layer_keys(
    keys: list[str],
    target_base: Mapping[str, torch.Tensor],
    *,
    module_scope: str | None,
    exclude_first_last_layer: bool,
) -> list[str]:
    """Restrict `keys` to a module type (mlp/attn) and/or drop the first and
    last decoder layer -- a cheap way to ask "what does JUST the MLPs (or
    JUST the attention) contribute" without touching embeddings, final norm,
    or the boundary layers that tend to carry input/output-specific work.
    """
    if module_scope is not None and module_scope not in _MODULE_SCOPE_TOKENS:
        raise ValueError(f"module_scope must be one of {sorted(_MODULE_SCOPE_TOKENS)} or None, got {module_scope!r}")

    layer_indices = {int(m.group(1)) for k in target_base if (m := _LAYER_RE.match(k))}
    last_layer = max(layer_indices) if layer_indices else None

    out: list[str] = []
    for k in keys:
        m = _LAYER_RE.match(k)
        if module_scope is not None:
            # Non-layer params (embeddings, final norm, ...) never belong to
            # an mlp/attn scope.
            if m is None or _MODULE_SCOPE_TOKENS[module_scope] not in k:
                continue
        if exclude_first_last_layer and m is not None:
            idx = int(m.group(1))
            if idx == 0 or idx == last_layer:
                continue
        out.append(k)
    return out


@dataclass(frozen=True)
class IdentityTransport:
    """
    No-op transport: Δ' = Δ (restricted to compatible shared keys).

    Optional `module_scope` ("mlp" | "attn") and `exclude_first_last_layer`
    kwargs (settable via a config's `method_params`) additionally restrict
    which keys carry the delta -- everything left out simply keeps the
    target's own weight (see merge/runtime.apply_delta), so e.g.
    module_scope="mlp" + exclude_first_last_layer=true adds the task vector
    to nothing but the inner layers' MLP weights.
    """

    name: str = "identity"

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        module_scope: str | None = None,
        exclude_first_last_layer: bool = False,
        **kwargs,
    ) -> TensorDict:
        keys = TaskVector.common_keys(source_base, [target_base, delta])

        if strict and set(keys) != set(delta.keys()):
            missing = sorted(set(delta.keys()) - set(keys))
            raise KeyError(f"Target/source base missing keys from delta. Example: {missing[:10]}")

        if module_scope is not None or exclude_first_last_layer:
            keys = _filter_layer_keys(
                list(keys),
                target_base,
                module_scope=module_scope,
                exclude_first_last_layer=exclude_first_last_layer,
            )

        out: TensorDict = {}
        for k in keys:
            out[k] = delta[k].to(dtype=target_base[k].dtype, device=target_base[k].device)
        return out


register(IdentityTransport())
