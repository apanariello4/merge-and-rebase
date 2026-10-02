"""Zero-bias materialization for bias-free decoder projections (``missing_bias='materialize'`` only)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def _resolve(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


@torch.no_grad()
def materialize_missing_projection_biases(
    model: nn.Module,
    base_sd: dict,
    *,
    family_adapter,
    components: Iterable[str],
    positions: Iterable[int] | None = None,
) -> list[str]:
    """Add a zero bias to each bias-free ``components`` projection (down_proj; o_proj if requested).

    Applied to ``model`` and ``base_sd`` alike (dtype/device follow the weight) so that strict
    ``load_state_dict`` snapshots/restores agree. Behaviourally a no-op until an intercept is written.
    ``positions=None`` means every layer. Idempotent; returns the added state-dict keys. Never called
    for ``missing_bias='skip'``, which keeps the stock architecture.
    """
    canonical = family_adapter.CANONICAL_COMPONENTS
    pos_list = range(len(family_adapter.layers(model))) if positions is None else [int(p) for p in positions]
    added: list[str] = []
    for pos in pos_list:
        for component in components:
            suffix = canonical[component]
            weight_key = family_adapter.param_key(pos, f"{suffix}.weight")
            bias_key = f"{weight_key[: -len('.weight')]}.bias"
            module = _resolve(model, weight_key[: -len(".weight")])
            if getattr(module, "bias", None) is None:
                w = module.weight
                module.bias = nn.Parameter(torch.zeros(w.shape[0], dtype=w.dtype, device=w.device))
            if bias_key not in base_sd:
                w = base_sd[weight_key]
                base_sd[bias_key] = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
                added.append(bias_key)
    return added
