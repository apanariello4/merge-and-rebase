"""One printed line saying which depth handling a run executes, in the config's own plain names.

Built from the resolved config and the bound plan, so it reports what actually runs (equal depths, a disabled depth
step and a method without one included), not what the config merely says.
"""

from __future__ import annotations

from typing import Any


def _layers(values: list[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def describe_depth_handling(resolved: Any, plan: Any, source_depth: int, target_depth: int) -> str:
    """``Depth handling: <method> -> <rule> (<what it does to source layers>)``."""
    method = str(resolved.method_name)
    head = f"Depth handling: {method} -> "
    if resolved.direct_fit:
        from .discrete_layer_match import DiscreteLayerPairing
        from .methods._ariadne.alignment import apply_depth_pairing_override

        pairing_rule = str(resolved.ariadne_cfg.depth_pairing)
        pairing = apply_depth_pairing_override(DiscreteLayerPairing.compute(source_depth, target_depth), pairing_rule)
        return (
            f"{head}no layers added; depth_pairing={pairing_rule} fits each target layer from a source layer "
            f"({source_depth} -> {target_depth}: target layer j <- source layer {_layers(list(pairing.pairing))})"
        )
    rule = plan.depth_alignment.rule
    if source_depth == target_depth:
        return f"{head}none (source and target both have {source_depth} layers)"
    if rule == "none":
        return (
            f"{head}none (no depth step runs for this method or configuration; {source_depth} -> {target_depth} layers)"
        )
    if rule == "discrete_index_match":
        from .discrete_layer_match import DiscreteLayerPairing

        pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
        return (
            f"{head}index_match (target layer j copies source layer round(j*(Ds-1)/(Dt-1)), no blending; "
            f"{source_depth} -> {target_depth}: {_layers(list(pairing.pairing))})"
        )
    cfg = resolved.block_extension_cfg
    plain = cfg.extension_strategy == "interpolate_per_weight" and cfg.skip_correction
    name = (
        "interpolate_layers"
        if plain
        else f"brace (extension_strategy={cfg.extension_strategy}, skip_correction={cfg.skip_correction})"
    )
    if target_depth < source_depth:
        return f"{head}{name} (merges groups of source layers, {source_depth} -> {target_depth}; collapse_schedule={cfg.collapse_schedule})"
    what = "blended copies" if cfg.extension_strategy == "interpolate_per_weight" else "copies"
    if cfg.insertion_order == "random":
        where = "at random positions"
    else:
        from .block_extension.core import BlockExtenderCore

        schedule = BlockExtenderCore._build_duplication_schedule(
            curr_layers=source_depth,
            n_needed=target_depth - source_depth,
            insertion_order=cfg.insertion_order,
            extension_density=cfg.extension_density,
        )
        where = f"of source layers {_layers(sorted(schedule))}"
    return f"{head}{name} (inserts {what} {where}, {source_depth} -> {target_depth})"
