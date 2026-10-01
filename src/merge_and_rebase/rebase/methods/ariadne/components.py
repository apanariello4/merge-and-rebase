"""Component-name constants and the block-forward ordering shared by every Ariadne fit."""

from __future__ import annotations

#: Residual-writing projections, in the order a block executes them.
COMPONENT_FORWARD_ORDER: tuple[str, ...] = ("attn.out_proj", "mlp.c_proj")


#: Internal (non-residual-writing) components reachable only in output_* modes.
INTERNAL_COMPONENTS: tuple[str, ...] = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")


#: Canonical block-forward evaluation order across all six component names.
#: Relative order of attn.out_proj and mlp.c_proj is unchanged from
#: COMPONENT_FORWARD_ORDER, so any set drawn only from the historical two
#: names orders identically to before.
CANONICAL_COMPONENT_ORDER: tuple[str, ...] = (
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.out_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


def order_components(components) -> tuple[str, ...]:
    """Return ``components`` in block-forward order.

    The config names a *set* of write surfaces; the fit order is a property of
    the architecture, not of how the config happened to list them. Any name
    from ``CANONICAL_COMPONENT_ORDER`` (residual-writing or internal) is
    accepted; unknown names are silently dropped, matching the historical
    behaviour of filtering against a fixed order tuple.
    """
    selected = set(components)
    return tuple(name for name in CANONICAL_COMPONENT_ORDER if name in selected)
