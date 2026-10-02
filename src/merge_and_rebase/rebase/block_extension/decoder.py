"""HF decoder BRACE block extender: ``DecoderBlockExtender`` and ``run_block_extension_llm``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from .adapters import DECODER_COMPONENTS, DecoderAdapter, _get_final_norm, _get_layers, _run_decoder_forward
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

            for h in hooks:
                h.remove()

            refs: dict[str, torch.Tensor] = {}
            for key, tensors in store.items():
                refs[key] = torch.cat(tensors, dim=0).flatten(0, 1)
            self.reference_inputs[name] = refs

    @torch.no_grad()
    def _capture_component_references(self, loader: Iterable[Any], n_batches: int):
        for name, model in [("base", self.model_base), ("ft", self.model_ft)]:
            self._vprint(f"capture component references ({name}) with n_batches={n_batches}")
            model.eval()
            store: dict[str, list[torch.Tensor]] = defaultdict(list)
            hooks: list[Any] = []

            layers = _get_layers(model, self.family_adapter)
            for i in range(len(layers)):
                block = layers[i]
                hooks.append(
                    block.input_layernorm.register_forward_hook(
                        self._store_output_hook(store, f"{i}.input_layernorm_output")
                    )
                )
                hooks.append(block.self_attn.register_forward_hook(self._store_output_hook(store, f"{i}.attn_output")))
                hooks.append(
                    block.self_attn.q_proj.register_forward_hook(self._store_output_hook(store, f"{i}.q_proj_output"))
                )
                hooks.append(
                    block.self_attn.k_proj.register_forward_hook(self._store_output_hook(store, f"{i}.k_proj_output"))
                )
                hooks.append(
                    block.self_attn.v_proj.register_forward_hook(self._store_output_hook(store, f"{i}.v_proj_output"))
                )
                hooks.append(
                    block.post_attention_layernorm.register_forward_hook(
                        self._store_output_hook(store, f"{i}.post_attn_ln_output")
                    )
                )
                hooks.append(
                    block.mlp.gate_proj.register_forward_hook(self._store_output_hook(store, f"{i}.gate_proj_output"))
                )
                hooks.append(
                    block.mlp.up_proj.register_forward_hook(self._store_output_hook(store, f"{i}.up_proj_output"))
                )
                hooks.append(
                    block.mlp.down_proj.register_forward_hook(self._store_output_hook(store, f"{i}.down_proj_output"))
                )

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

            for h in hooks:
                h.remove()

            refs: dict[str, torch.Tensor] = {}
            for key, tensors in store.items():
                refs[key] = torch.cat(tensors, dim=0).flatten(0, 1)
            self.reference_inputs[name].update(refs)

    def _set_layers(self, model: nn.Module, new_layers: list[nn.Module]) -> None:
        scope = self.family_adapter.transport_scope(model)
        scope.layers = nn.ModuleList(new_layers)
        self._set_depth(model, scope, len(scope.layers))
        self._reindex_layers(model, scope.layers)

    @staticmethod
    def _set_depth(model: nn.Module, scope: nn.Module, depth: int) -> None:
        # HF decoders iterate `self.layers[: self.config.num_hidden_layers]`, so
        # a longer ModuleList alone does nothing: the appended blocks never run.
        # Everything downstream still sees them (state_dict reports them, deltas
        # are computed over them), which makes the truncation invisible -- the
        # model simply behaves as if it were never extended.
        for holder in (model, scope):
            config = getattr(holder, "config", None)
            if config is None:
                continue
            if getattr(config, "num_hidden_layers", None) == depth:
                continue
            config.num_hidden_layers = depth

    @staticmethod
    def _resolve_layer_types(model: nn.Module, layers: nn.ModuleList) -> list[str] | None:
        """Grow `config.layer_types` to the new depth, keeping it authoritative.

        Models with alternating attention patterns key off this list, and the
        reindex below reads it positionally. Left short, every layer past the
        original depth keeps whichever `attention_type` it was duplicated with
        while the config claims a shorter model.
        """
        config = getattr(model, "config", None)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            return None
        resolved = list(layer_types)
        if len(resolved) >= len(layers):
            return resolved[: len(layers)]
        for idx in range(len(resolved), len(layers)):
            own = getattr(layers[idx], "attention_type", None)
            resolved.append(own if own is not None else resolved[-1])
        config.layer_types = resolved
        return resolved

    @staticmethod
    def _reindex_layers(model: nn.Module, layers: nn.ModuleList) -> None:
        # Each decoder layer's attention module caches its own `layer_idx`
        # (set at construction) to key into the shared KV cache during a
        # forward pass. Duplicating/reordering layers without updating it
        # leaves two layers pointing at the same cache slot: the second one
        # to run has its `update()` call concatenate onto the first's
        # leftover keys/values, silently doubling the sequence length the
        # rest of that layer's attention sees (crashes as a seq-length
        # mismatch against the attention mask, or worse, doesn't crash).
        layer_types = DecoderBlockExtender._resolve_layer_types(model, layers)
        for new_idx, layer in enumerate(layers):
            for holder in (layer, getattr(layer, "self_attn", None)):
                if holder is not None and hasattr(holder, "layer_idx"):
                    holder.layer_idx = new_idx
            if layer_types is not None and hasattr(layer, "attention_type") and new_idx < len(layer_types):
                layer.attention_type = layer_types[new_idx]

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
