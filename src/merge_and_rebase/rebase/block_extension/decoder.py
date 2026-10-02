"""HF decoder BRACE block extender: ``DecoderBlockExtender`` and ``run_block_extension_llm``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from ..model_families import accessors as fam
from .adapters import (
    DECODER_COMPONENTS,
    DecoderAdapter,
    _content_rows,
    _get_final_norm,
    _get_layers,
    _run_decoder_forward,
)
from .config import BlockExtensionConfig
from .core import BlockExtenderCore, EagerProvider, _deterministic_calibration_loader, _iter_with_progress
from .schedules import (
    build_extension_layout,
    build_reduction_layout,
    decoder_collapse_schedule,
    decoder_locate_collapse_pos,
)

_DECODER_COMPONENTS = tuple(spec.name for spec in DECODER_COMPONENTS)


class DecoderBlockExtender(BlockExtenderCore):
    _LOG_PREFIX = "block_extension_llm"
    _EXPECTED_PREFIX = "Expected: "
    _SHRINK_HONOURS_PER_WEIGHT_MODE = False

    def __init__(
        self,
        model_base: nn.Module,
        model_ft: nn.Module,
        family_adapter: Any,
        device: str | torch.device,
        *,
        verbose: bool = True,
        show_progress: bool = True,
    ):
        self.model_base = model_base
        self.model_ft = model_ft
        self.family_adapter = family_adapter
        self.adapter = DecoderAdapter(family_adapter)
        self.device = device
        self.reference_inputs: dict[str, dict[str, torch.Tensor]] = {"base": {}, "ft": {}}
        self.verbose = bool(verbose)
        self.show_progress = bool(show_progress)
        self._component_ridge: dict[str, float] | None = None
        # Realized block chain, set by the extend path; None when no extension
        # ran (shrink, or depth already matching).
        self.realized_layout: dict[str, Any] | None = None

    def _make_reference_provider(self) -> EagerProvider:
        return EagerProvider(self)

    def _collapse_output_refs(self, span_end: int, orig_depth: int) -> tuple[str, None]:
        # B19: the span's own last block input (not its output boundary, as vision uses) is the target.
        return f"{span_end}.input", None

    def _finalize_reduction(self, chain_base: list[dict[str, Any]]) -> None:
        # Publish the realized many-to-one ancestry for consumers such as direct-target P1.  Shrink has no
        # inserted positions; every target position is addressed by the source span it absorbed.
        self.realized_layout = build_reduction_layout(chain_base)

    def _finalize_extension(self, chain_base: list[dict[str, Any]]) -> None:
        # Describe the realized chain with the same builder vision uses, so a consumer (e.g. residual completion)
        # addresses inserted positions by recorded ancestry rather than assuming a doubling depth pattern --
        # this extension is 24->28, not a doubling.
        self.realized_layout = build_extension_layout(chain_base)

    @staticmethod
    def _build_collapse_schedule(
        curr_layers: int,
        n_to_remove: int,
        insertion_order: str,
        extension_density: str,
    ) -> list[int]:
        return decoder_collapse_schedule(curr_layers, n_to_remove, insertion_order, extension_density)

    @staticmethod
    def _locate_collapse_pos(chain: list[dict[str, Any]], anchor_orig_idx: int) -> int:
        return decoder_locate_collapse_pos(chain, anchor_orig_idx)

    @torch.no_grad()
    def capture_reference_inputs(self, loader: Iterable[Any], n_batches: int):
        for name, model in [("base", self.model_base), ("ft", self.model_ft)]:
            self._vprint(f"capture reference inputs ({name}) with n_batches={n_batches}")
            model.eval()
            store: dict[str, list[torch.Tensor]] = defaultdict(list)
            masks: list[torch.Tensor] = []
            hooks: list[Any] = []

            layers = _get_layers(model, self.family_adapter)
            for i in range(len(layers)):
                hooks.append(layers[i].register_forward_hook(self._store_input_hook(store, f"{i}.input")))

            final_norm = _get_final_norm(model, self.family_adapter)
            hooks.append(final_norm.register_forward_hook(self._store_input_hook(store, "final.input")))

            it = iter(loader)
            for _ in _iter_with_progress(
                range(n_batches),
                total=n_batches,
                desc=f"block_extension_llm.capture.{name}",
                enabled=self.show_progress,
            ):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                inputs = self.family_adapter.extract_calibration_batch(batch)
                _run_decoder_forward(model, inputs, self.device)
                masks.append(fam.content_mask(self.family_adapter, batch))

            for h in hooks:
                h.remove()

            refs: dict[str, torch.Tensor] = {}
            for key, tensors in store.items():
                # Padding rows never enter a reference (rows follow attention_mask only).
                refs[key] = _content_rows(tensors, masks, what=f"reference '{key}'")
            self.reference_inputs[name] = refs

    @torch.no_grad()
    def _capture_component_references(self, loader: Iterable[Any], n_batches: int):
        for name, model in [("base", self.model_base), ("ft", self.model_ft)]:
            self._vprint(f"capture component references ({name}) with n_batches={n_batches}")
            model.eval()
            store: dict[str, list[torch.Tensor]] = defaultdict(list)
            masks: list[torch.Tensor] = []
            hooks: list[Any] = []

            layers = _get_layers(model, self.family_adapter)
            for i in range(len(layers)):
                block = layers[i]
                comps = fam.block_components(self.family_adapter, block)
                hooks_spec = (
                    (comps["input_layernorm"], "input_layernorm_output"),
                    (fam.attn_module(self.family_adapter, block), "attn_output"),
                    (comps["q_proj"], "q_proj_output"),
                    (comps["k_proj"], "k_proj_output"),
                    (comps["v_proj"], "v_proj_output"),
                    (comps["post_attention_layernorm"], "post_attn_ln_output"),
                    (comps["gate_proj"], "gate_proj_output"),
                    (comps["up_proj"], "up_proj_output"),
                    (comps["down_proj"], "down_proj_output"),
                )
                for module, ref_name in hooks_spec:
                    hooks.append(module.register_forward_hook(self._store_output_hook(store, f"{i}.{ref_name}")))

            it = iter(loader)
            for _ in _iter_with_progress(
                range(n_batches),
                total=n_batches,
                desc=f"block_extension_llm.capture_components.{name}",
                enabled=self.show_progress,
            ):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                inputs = self.family_adapter.extract_calibration_batch(batch)
                _run_decoder_forward(model, inputs, self.device)
                masks.append(fam.content_mask(self.family_adapter, batch))

            for h in hooks:
                h.remove()

            refs: dict[str, torch.Tensor] = {}
            for key, tensors in store.items():
                # Padding rows never enter a reference (rows follow attention_mask only).
                refs[key] = _content_rows(tensors, masks, what=f"reference '{key}'")
            self.reference_inputs[name].update(refs)

    @torch.no_grad()
    def extend_and_calibrate(
        self,
        *,
        loader: Iterable[Any],
        n_batches: int,
        strategy: str = "interpolate",
        dampening_factor: float = 1.0,
        blocks_to_add: int | None = None,
        target_layers_total: int | None = None,
        insertion_order: str = "bottom-top",
        extension_density: str = "spread",
        skip_correction: bool = False,
        skip_final_ln: bool = False,
        ridge_identity: float = 0.0,
        n_cascade_iters: int = 1,
        share_ft_refs: bool = False,
        component_ridge: dict[str, float] | None = None,
        lmc_mode: str = "independent",
    ) -> int:
        if not skip_correction:
            loader = _deterministic_calibration_loader(loader, n_batches)
        if strategy in ("interpolate", "duplicate"):
            raise ValueError(
                f"extension_strategy '{strategy}' (non per-weight) is disabled. Use '{strategy}_per_weight' instead."
            )
        if strategy in (
            "per_weight",
            "per-weight",
            "interpolate_per_weight",
            "interpolate-per-weight",
            "duplicate_per_weight",
            "duplicate-per-weight",
        ):
            per_weight_mode = "duplicate" if strategy in ("duplicate_per_weight", "duplicate-per-weight") else "cascade"
            n_needed = self._resolve_depth_delta(
                len(_get_layers(self.model_base, self.family_adapter)), blocks_to_add, target_layers_total
            )
            per_weight_fn = self._shrink_per_weight if n_needed < 0 else self._extend_per_weight
            return per_weight_fn(
                loader=loader,
                n_batches=n_batches,
                dampening_factor=dampening_factor,
                blocks_to_add=blocks_to_add,
                target_layers_total=target_layers_total,
                insertion_order=insertion_order,
                extension_density=extension_density,
                ridge_identity=ridge_identity,
                per_weight_mode=per_weight_mode,
                n_cascade_iters=n_cascade_iters,
                share_ft_refs=share_ft_refs,
                skip_correction=skip_correction,
                component_ridge=component_ridge,
                lmc_mode=lmc_mode,
            )
        if strategy == "shrink":
            return self._shrink_per_weight(
                loader=loader,
                n_batches=n_batches,
                dampening_factor=dampening_factor,
                blocks_to_add=blocks_to_add,
                target_layers_total=target_layers_total,
                insertion_order=insertion_order,
                extension_density=extension_density,
                ridge_identity=ridge_identity,
                per_weight_mode="cascade",
                n_cascade_iters=n_cascade_iters,
                share_ft_refs=share_ft_refs,
                skip_correction=skip_correction,
                component_ridge=component_ridge,
                lmc_mode=lmc_mode,
            )
        raise ValueError(
            "Unsupported extension_strategy "
            f"'{strategy}'. Expected: interpolate, per_weight, shrink, interpolate_per_weight, duplicate_per_weight."
        )


def run_block_extension_llm(
    *,
    source_base_model: nn.Module,
    source_ft_model: nn.Module,
    calibration_loader: Iterable[Any],
    target_layers_total: int | None,
    config: BlockExtensionConfig,
    family_adapter: Any,
    device: str | torch.device,
    layout_out: dict[str, Any] | None = None,
) -> int:
    """Resize the pair in place and return the realized depth.

    ``layout_out``, when given, is updated with the realized block chain
    (positions, which blocks were inserted, and the neighbour each was
    initialized from). It is an out-parameter rather than a changed return
    type so existing callers stay byte-identical.
    """
    extender = DecoderBlockExtender(
        source_base_model,
        source_ft_model,
        family_adapter,
        device,
        verbose=bool(config.verbose),
        show_progress=bool(config.show_progress),
    )
    resolved_target_layers_total = (
        target_layers_total if target_layers_total is not None else config.target_layers_total
    )
    final_depth = extender.extend_and_calibrate(
        loader=calibration_loader,
        n_batches=config.n_batches_act,
        strategy=config.extension_strategy,
        dampening_factor=float(config.dampening_factor),
        blocks_to_add=config.blocks_to_add,
        target_layers_total=resolved_target_layers_total,
        insertion_order=config.insertion_order,
        extension_density=config.extension_density,
        skip_correction=bool(config.skip_correction),
        skip_final_ln=bool(config.skip_final_ln),
        ridge_identity=float(config.ridge_identity),
        n_cascade_iters=int(config.n_cascade_iters),
        share_ft_refs=bool(config.share_ft_refs),
        component_ridge=config.component_ridge,
        lmc_mode=str(config.lmc_mode),
    )
    if layout_out is not None and extender.realized_layout is not None:
        layout_out.update(extender.realized_layout)
    return final_depth
