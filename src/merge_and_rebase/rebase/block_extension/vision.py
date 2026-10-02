"""OpenCLIP vision BRACE block extender: ``BlockExtender`` and ``run_block_extension``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ...models.vision_utils import _encode_image
from .adapters import VisionAdapter, _InProjCapture
from .config import (
    BlockExtensionConfig,
    TargetSharedCorrection,
    _as_correction_scope,
    _as_inserted_block_mode,
    _as_reference_capture,
)
from .core import BlockExtenderCore, LazyProvider, _deterministic_calibration_loader
from .schedules import (
    build_extension_layout,
    build_reduction_layout,
    plan_inserted_positions,
    vision_collapse_schedule,
    vision_locate_collapse_pos,
)


class BlockExtender(BlockExtenderCore):
    def __init__(
        self,
        model_base: nn.Module,
        model_ft: nn.Module,
        device: str | torch.device,
        *,
        verbose: bool = True,
        show_progress: bool = True,
        diagnostic_collector: Any | None = None,
        diagnostic_mode: str = "independent",
        target_model: nn.Module | None = None,
    ):
        self.model_base = model_base
        self.model_ft = model_ft
        self.adapter = VisionAdapter()
        # Pretrained target backbone, used only by the target-informed
        # correction option; ``None`` for every standard ARIADNE path.
        self.target_model = target_model
        self.device = device
        self.reference_inputs: dict[str, dict[str, torch.Tensor]] = {"base": {}, "ft": {}}
        # Populated by the extension paths with the realized block layout, so a
        # downstream transport method can address inserted positions by index.
        self.extension_layout: dict[str, Any] | None = None
        # Target-side component banks keyed by final chain position, plus the
        # sample count they were captured over (needed to recover tokens/sample).
        self._target_reference_banks: dict[int, torch.Tensor] = {}
        self._target_reference_samples = 0
        self.verbose = bool(verbose)
        self.show_progress = bool(show_progress)
        self._ridge_weight = 1e-6
        # Diagnostics are an optional side channel and are off by default.
        self.diagnostic_collector = diagnostic_collector
        self.diagnostic_mode = str(diagnostic_mode)
        self._diagnostic_context: dict[str, Any] | None = None

    def _make_reference_provider(self) -> LazyProvider:
        return LazyProvider(self)

    def _record_correction(self, endpoint: str, component: str, W: torch.Tensor, b: torch.Tensor) -> None:
        collector = self.diagnostic_collector
        context = self._diagnostic_context
        if collector is None or context is None:
            return
        collector.record_map(
            mode=self.diagnostic_mode,
            endpoint=endpoint,
            structural_step=context["structural_step"],
            final_block=context["final_block"],
            source_block=context["source_block"],
            component=component,
            W=W,
            b=b,
        )

    @staticmethod
    def _inner_block(block: nn.Module) -> nn.Module:
        return block.block if hasattr(block, "block") else block

    @torch.no_grad()
    def _capture_per_weight_reference_subset(
        self,
        reference_models: Mapping[str, nn.Module],
        *,
        endpoints: Sequence[str],
        block_indices: Sequence[int],
        input_indices: Sequence[int | str],
        loader: Iterable[Any],
        n_batches: int,
    ) -> None:
        """Capture only references needed by the current structural step.

        The former eager path retained every component from every block for
        both endpoints.  At 40 calibration batches that scales to hundreds of
        gigabytes.  This lazy path is mathematically equivalent: pristine
        endpoint copies provide the same references, while tensors from the
        previous structural step are released before the next capture.
        """
        eager = getattr(self, "_reference_capture", "lazy") == "eager"
        if eager:
            cached = getattr(self, "_eager_reference_cache", None)
            if cached is not None:
                self.reference_inputs = cached
                return
            probe = reference_models.get("base") or next(iter(reference_models.values()))
            all_blocks = tuple(range(len(probe.visual.transformer.resblocks)))
            endpoints = ("base", "ft")
            block_indices = all_blocks
            input_indices = all_blocks + ("final",)

        # Drop the previous step before allocating the next reference window.
        self.reference_inputs = {"base": {}, "ft": {}}
        unique_blocks = tuple(dict.fromkeys(int(index) for index in block_indices))
        unique_inputs = tuple(dict.fromkeys(input_indices))
        for name in endpoints:
            model = reference_models[name]
            model.to(self.device)
            model.eval()
            store: dict[str, list[torch.Tensor]] = defaultdict(list)
            hooks: list[Any] = []
            caps: list[tuple[int, _InProjCapture]] = []
            try:
                for index in unique_blocks:
                    inner = self._inner_block(model.visual.transformer.resblocks[index])
                    hooks.append(
                        inner.ln_1.register_forward_hook(self._store_output_hook(store, f"{index}.ln_1_output"))
                    )
                    hooks.append(
                        inner.attn.register_forward_hook(self._store_output_hook(store, f"{index}.attn_output"))
                    )
                    hooks.append(
                        inner.ln_2.register_forward_hook(self._store_output_hook(store, f"{index}.ln_2_output"))
                    )
                    hooks.append(
                        inner.mlp.c_fc.register_forward_hook(self._store_output_hook(store, f"{index}.c_fc_output"))
                    )
                    hooks.append(
                        inner.mlp.c_proj.register_forward_hook(self._store_output_hook(store, f"{index}.c_proj_output"))
                    )
                    caps.append((index, _InProjCapture(inner.attn)))

                for index in unique_inputs:
                    if index == "final":
                        module = model.visual.ln_post
                        key = "final.input"
                    else:
                        module = model.visual.transformer.resblocks[int(index)]
                        key = f"{int(index)}.input"
                    hooks.append(module.register_forward_hook(self._store_input_hook(store, key)))

                iterator = iter(loader)
                consumed = 0
                for _ in range(n_batches):
                    try:
                        images, _ = next(iterator)
                    except StopIteration:
                        raise ValueError(
                            f"BRACE calibration loader exhausted after {consumed} batches; requested {n_batches}."
                        ) from None
                    _encode_image(model, images.to(self.device))
                    consumed += 1
            finally:
                for hook in hooks:
                    hook.remove()
                for _, cap in caps:
                    cap.restore()
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for index, cap in caps:
                for q, k, v in cap.outputs:
                    store[f"{index}.q_output"].append(q)
                    store[f"{index}.k_output"].append(k)
                    store[f"{index}.v_output"].append(v)

            refs: dict[str, torch.Tensor] = {}
            for key in list(store):
                tensors = store.pop(key)
                refs[key] = torch.cat(tensors, dim=0).flatten(0, 1)
                del tensors
            self.reference_inputs[name] = refs

        if eager:
            self._eager_reference_cache = self.reference_inputs

    @torch.no_grad()
    def _capture_target_component_references(
        self,
        *,
        positions: Sequence[int],
        loader: Iterable[Any],
        n_batches: int,
    ) -> dict[int, torch.Tensor]:
        """Capture the pretrained target model's c_proj outputs at ``positions``.

        Runs on the same frozen calibration loader the corrections use, so row
        ``i`` of a target bank and row ``i`` of a source reference bank come
        from the same image. Banks are kept token-shaped ``(N, T_target, D_L)``
        because the target's patch grid differs from the source's and has to be
        resampled before the two can be paired.
        """
        target = self.target_model
        if target is None:
            raise RuntimeError(
                "target_shared_correction requires the pretrained target model; "
                "pass target_model= to run_block_extension."
            )
        blocks = target.visual.transformer.resblocks
        buffers: dict[int, list[torch.Tensor]] = {int(p): [] for p in positions}
        for position in buffers:
            if not 0 <= position < len(blocks):
                raise ValueError(f"Target block position {position} is outside the target depth {len(blocks)}.")

        original_device = next(target.parameters()).device
        handles: list[Any] = []
        n_samples = 0
        try:
            target.to(self.device).eval()
            for position in buffers:
                inner = self._inner_block(blocks[position])
                handles.append(inner.mlp.c_proj.register_forward_hook(self._store_output_hook(buffers, position)))
            iterator = iter(loader)
            for _ in range(int(n_batches)):
                try:
                    images, _ = next(iterator)
                except StopIteration:
                    break
                n_samples += int(images.shape[0])
                _encode_image(target, images.to(self.device))
        finally:
            for handle in handles:
                handle.remove()
            target.to(original_device)

        banks: dict[int, torch.Tensor] = {}
        for position, chunks in buffers.items():
            if chunks:
                banks[position] = torch.cat(chunks, dim=0)
        self._target_reference_samples = n_samples
        self._vprint(
            f"captured target component references for {len(banks)} positions over {n_samples} calibration samples"
        )
        return banks

    @staticmethod
    def _blend_target_reference(
        source_ref: torch.Tensor,
        target_bank: torch.Tensor | None,
        n_samples: int,
        target_weight: float,
    ) -> torch.Tensor:
        """Return ``(Y_S + eta * Y_{L->S}) / (1 + eta)``.

        ``Y_{L->S}`` is the target bank expressed in source coordinates through
        a centred rectangular Procrustes map fitted on this block's paired rows.
        The map is orthogonal, so the backprojection keeps only the part of the
        wider target representation that source coordinates can carry; it does
        not import the whole target activation. Normalising by ``1 + eta``
        keeps the blended target's scale comparable to the source target, so
        the identity ridge is not silently re-weighted by ``eta``.
        """
        if target_bank is None or target_weight <= 0.0:
            return source_ref
        if n_samples <= 0:
            raise ValueError("Target reference blending needs a positive calibration sample count.")

        from ..methods.theseus import _compute_procrustes_map_from_cov, _interp_2d_tokens

        rows = int(source_ref.shape[0])
        if rows % n_samples != 0:
            raise ValueError(
                f"Source reference rows ({rows}) are not a multiple of the calibration "
                f"sample count ({n_samples}); cannot recover tokens per sample."
            )
        source_tokens = rows // n_samples
        if int(target_bank.shape[0]) != n_samples:
            raise ValueError(f"Target bank holds {int(target_bank.shape[0])} samples, expected {n_samples}.")

        target_tokens = target_bank.float()
        if int(target_tokens.shape[1]) != source_tokens:
            # Patch grids differ (14x14 vs 16x16 at 224px); resample the target
            # grid onto the source grid, keeping the CLS token separate.
            target_tokens = _interp_2d_tokens(target_tokens, source_tokens)
        target_rows = target_tokens.reshape(-1, target_tokens.shape[-1])
        source_rows = source_ref.float()
        if target_rows.shape[0] != source_rows.shape[0]:
            raise ValueError(
                f"Row counts disagree after token resampling: source {source_rows.shape[0]}, "
                f"target {target_rows.shape[0]}."
            )
        target_rows = target_rows.to(source_rows.device)

        mu_source = source_rows.mean(dim=0, keepdim=True)
        mu_target = target_rows.mean(dim=0, keepdim=True)
        cov = (source_rows - mu_source).T @ (target_rows - mu_target)
        q_map = _compute_procrustes_map_from_cov(cov).to(source_rows.device)
        backprojected = (target_rows - mu_target) @ q_map.T + mu_source

        blended = (source_rows + float(target_weight) * backprojected) / (1.0 + float(target_weight))
        return blended.to(dtype=source_ref.dtype)

    @torch.no_grad()
    def _zero_block_output_projections(self, block: nn.Module):
        """Turn an inserted block into an exact identity on the residual stream.

        With both output projections zeroed (weight and bias), the attention
        and MLP branches contribute nothing and the block returns its input
        unchanged, so the expanded model computes exactly the original
        function. This is the residual-identity depth baseline: it separates
        the cost of extra depth from the cost of the computation ARIADNE puts
        in the inserted block.
        """
        self.adapter.zero_output_projections(block)

    @staticmethod
    def _build_collapse_schedule(
        curr_layers: int,
        n_to_remove: int,
        insertion_order: str,
        extension_density: str,
    ):
        return vision_collapse_schedule(curr_layers, n_to_remove, insertion_order, extension_density)

    @staticmethod
    def _locate_collapse_pos(chain: list[dict[str, Any]], anchor_orig_idx: int) -> int:
        return vision_locate_collapse_pos(chain, anchor_orig_idx)

    @torch.no_grad()
    def extend_and_calibrate(
        self,
        *,
        loader: Iterable[Any],
        n_batches: int,
        strategy: str,
        dampening_factor: float,
        blocks_to_add: int | None,
        target_layers_total: int | None,
        insertion_order: str,
        extension_density: str,
        collapse_schedule: str = "cascade",
        skip_correction: bool,
        skip_final_ln: bool,
        ridge_identity: float = 0.0,
        ridge_weight: float = 1e-6,
        n_cascade_iters: int = 1,
        share_ft_refs: bool = False,
        component_ridge: dict[str, float] | None = None,
        lmc_mode: str = "independent",
        reference_capture: str = "lazy",
        inserted_block_mode: str = "ariadne",
        correction_scope: str = "inserted",
        insertion_target_mode: str = "direct",
        target_shared_correction: TargetSharedCorrection | None = None,
    ) -> int:
        self._reference_capture = _as_reference_capture(reference_capture)
        self._eager_reference_cache: dict[str, dict[str, torch.Tensor]] | None = None
        self._ridge_weight = float(ridge_weight)
        if self._ridge_weight < 0.0:
            raise ValueError("ridge_weight must be >= 0.")
        curr_layers = len(self.model_base.visual.transformer.resblocks)
        if insertion_target_mode not in ("direct", "residual"):
            raise ValueError(
                f"Unsupported insertion_target_mode '{insertion_target_mode}'. "
                "Expected 'direct' (published behaviour) or 'residual'."
            )
        if not skip_correction:
            loader = _deterministic_calibration_loader(loader, n_batches)
        n_needed = self._resolve_depth_delta(curr_layers, blocks_to_add, target_layers_total)

        common_kwargs = dict(
            loader=loader,
            n_batches=n_batches,
            dampening_factor=dampening_factor,
            blocks_to_add=blocks_to_add,
            target_layers_total=target_layers_total,
            insertion_order=insertion_order,
            extension_density=extension_density,
            collapse_schedule=collapse_schedule,
            ridge_identity=ridge_identity,
            n_cascade_iters=n_cascade_iters,
            share_ft_refs=share_ft_refs,
            skip_correction=skip_correction,
            component_ridge=component_ridge,
            lmc_mode=lmc_mode,
        )
        inserted_block_mode = _as_inserted_block_mode(inserted_block_mode)
        if inserted_block_mode != "ariadne":
            if n_needed < 0:
                raise ValueError(
                    f"inserted_block_mode='{inserted_block_mode}' is an extension baseline; "
                    "block shrink has no inserted block to make an identity."
                )
            if not skip_correction:
                raise ValueError(f"inserted_block_mode='{inserted_block_mode}' requires skip_correction=True.")
        correction_scope = _as_correction_scope(correction_scope)
        if correction_scope != "inserted":
            if n_needed < 0:
                raise ValueError(
                    f"correction_scope='{correction_scope}' is an extension option; "
                    "block shrink has no insertion to repair around."
                )
            if skip_correction:
                raise ValueError(f"correction_scope='{correction_scope}' requires skip_correction=False.")
        if target_shared_correction is not None and target_shared_correction.active:
            if n_needed < 0:
                raise ValueError(
                    "target_shared_correction is an extension option; block shrink has no "
                    "inserted block whose correction target could be blended."
                )
            if skip_correction:
                raise ValueError("target_shared_correction requires skip_correction=False.")
        common_kwargs["inserted_block_mode"] = inserted_block_mode
        common_kwargs["correction_scope"] = correction_scope
        common_kwargs["target_shared_correction"] = target_shared_correction
        shrink_kwargs = {
            k: v
            for k, v in common_kwargs.items()
            if k not in {"inserted_block_mode", "correction_scope", "target_shared_correction"}
        }
        # The collapse schedule only exists for the reduction direction; the
        # extension path has no spans to partition.
        common_kwargs.pop("collapse_schedule", None)
        if strategy == "interpolate_per_weight":
            if n_needed < 0:
                return self._shrink_per_weight(per_weight_mode="cascade", **shrink_kwargs)
            return self._extend_per_weight(
                per_weight_mode="cascade", insertion_target_mode=insertion_target_mode, **common_kwargs
            )
        if strategy == "duplicate_per_weight":
            if n_needed < 0:
                return self._shrink_per_weight(per_weight_mode="duplicate", **shrink_kwargs)
            return self._extend_per_weight(
                per_weight_mode="duplicate", insertion_target_mode=insertion_target_mode, **common_kwargs
            )

        if strategy == "interpolate":
            raise ValueError(
                "Vision extension_strategy='interpolate' is no longer supported; use 'interpolate_per_weight'."
            )
        if strategy == "duplicate":
            raise ValueError(
                "Vision extension_strategy='duplicate' is no longer supported; use 'duplicate_per_weight'."
            )
        if n_needed < 0:
            raise ValueError(
                "Block shrink is currently supported only for duplicate_per_weight and interpolate_per_weight strategies. "
                f"Got: {strategy}"
            )
        raise ValueError(
            "Unsupported vision extension_strategy. Expected 'interpolate_per_weight' or "
            f"'duplicate_per_weight'. Got: {strategy}"
        )

    @staticmethod
    def _original_block_position(chain: Sequence[Mapping[str, Any]], orig_idx: int) -> int:
        """Current chain position of an original (non-inserted) block.

        ``orig_idx`` repeats across a block's inserted descendants, so the
        lookup must reject inserted entries.
        """
        for position, item in enumerate(chain):
            if not bool(item.get("inserted", False)) and int(item["orig_idx"]) == orig_idx:
                return position
        raise ValueError(f"Original block {orig_idx} is not present in the current chain.")

    @staticmethod
    def _original_blocks_to_correct(
        chain: Sequence[Mapping[str, Any]], *, insert_pos: int, scope: str
    ) -> tuple[int, ...]:
        """Original block indices to correct after inserting at ``insert_pos``."""
        if scope == "inserted":
            return ()
        above = [
            int(item["orig_idx"])
            for position, item in enumerate(chain)
            if position > insert_pos and not bool(item.get("inserted", False))
        ]
        if scope == "iterative_all":
            return tuple(above)
        if scope == "interleaved_once":
            # Only the block immediately above the insertion. Over the whole
            # bottom-to-top schedule this corrects every original block exactly
            # once, with no block ever corrected twice.
            return tuple(above[:1])
        raise ValueError(
            f"Unsupported correction_scope '{scope}'. Expected 'inserted', 'interleaved_once', or 'iterative_all'."
        )

    def _before_extension_loop(self, schedule, curr_layers, loader, n_batches, target_shared_correction) -> None:
        # Final positions are needed before the loop: the correction step only knows the position in the
        # partially built chain, but a target-side reference must be addressed by the block's position in the
        # finished model, which is what the target backbone's own depth indexes.
        self._inserted_positions = plan_inserted_positions(curr_layers, schedule)
        self._target_reference_banks = {}
        self._target_reference_samples = 0
        self._target_weight = 0.0
        if target_shared_correction is not None and target_shared_correction.active:
            self._target_weight = float(target_shared_correction.target_weight)
            target_batches = target_shared_correction.num_batches or n_batches
            self._target_reference_banks = self._capture_target_component_references(
                positions=self._inserted_positions,
                loader=loader,
                n_batches=target_batches,
            )

    def _target_reference_for(self, step: int) -> tuple[torch.Tensor | None, float]:
        return self._target_reference_banks.get(self._inserted_positions[step - 1]), self._target_weight

    def _init_inserted_block(self, dup_base: nn.Module, dup_ft: nn.Module, inserted_block_mode: str) -> None:
        if inserted_block_mode in {"residual_identity", "residual_identity_inert"}:
            self._zero_block_output_projections(dup_base)
            self._zero_block_output_projections(dup_ft)
        if inserted_block_mode == "residual_identity_inert":
            # Give the FT endpoint the base endpoint's inserted block, so the inserted position's task vector is
            # zero on every parameter and not only on the output projections. The base endpoint is untouched, so
            # this arm and 'residual_identity' fit byte-identical transport maps and differ in exactly one
            # quantity: the delta being transported.
            dup_ft.load_state_dict(dup_base.state_dict())

    def _post_insert_correction(self, chain_base, insert_pos, correction_scope, **block_kwargs) -> None:
        # Repair the original blocks the insertion just disturbed. The bottom-to-top schedule visits each original
        # block as an insertion source exactly once, so correcting the block immediately above the new insertion
        # touches every original block once and always fits against an input that is already final: everything
        # below it has been edited for the last time, and later edits land above it. A pass that instead corrected
        # original blocks *below* an already-fitted inserted block would silently invalidate that block's
        # absorbed correction.
        for original_idx in self._original_blocks_to_correct(chain_base, insert_pos=insert_pos, scope=correction_scope):
            self._correct_one_block(
                position=self._original_block_position(chain_base, original_idx),
                ref_block_idx=original_idx,
                **block_kwargs,
            )

    def _collapse_output_refs(self, span_end: int, orig_depth: int) -> tuple[str, int | str]:
        # The span's output boundary is the input of the block above it (the final norm for the last block).
        if span_end + 1 >= orig_depth:
            return "final.input", "final"
        return f"{span_end + 1}.input", span_end + 1

    def _finalize_reduction(self, chain_base) -> None:
        # The extension path publishes its realized layout for downstream target-informed code; the reduction path
        # must too, or a caller that passes ``layout_out`` silently receives an empty dict.
        self.extension_layout = build_reduction_layout(chain_base)
        self.reference_inputs = {"base": {}, "ft": {}}

    def _finalize_extension(self, chain_base) -> None:
        self.extension_layout = build_extension_layout(chain_base)
        self._target_reference_banks = {}
        self.reference_inputs = {"base": {}, "ft": {}}


@torch.no_grad()
def run_block_extension(
    *,
    source_base_model: nn.Module,
    source_ft_model: nn.Module,
    calibration_loader: Iterable[Any],
    target_layers_total: int | None,
    config: BlockExtensionConfig,
    device: str | torch.device,
    diagnostic_collector: Any | None = None,
    layout_out: dict[str, Any] | None = None,
    target_model: nn.Module | None = None,
) -> int:
    """Resize ``source_base_model``/``source_ft_model`` in place.

    ``layout_out``, when given, is cleared and filled with the realized block
    layout (see :func:`build_extension_layout`). Callers that need to address
    inserted positions downstream read it instead of re-deriving the schedule,
    which would diverge for ``insertion_order='random'``.
    """
    extender = BlockExtender(
        source_base_model,
        source_ft_model,
        device,
        verbose=bool(config.verbose),
        show_progress=bool(config.show_progress),
        diagnostic_collector=diagnostic_collector,
        diagnostic_mode=str(config.lmc_mode),
        target_model=target_model,
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
        collapse_schedule=config.collapse_schedule,
        skip_correction=bool(config.skip_correction),
        skip_final_ln=bool(config.skip_final_ln),
        ridge_identity=float(config.ridge_identity),
        ridge_weight=float(config.ridge_weight),
        n_cascade_iters=int(config.n_cascade_iters),
        share_ft_refs=bool(config.share_ft_refs),
        component_ridge=config.component_ridge,
        lmc_mode=str(config.lmc_mode),
        reference_capture=str(config.reference_capture),
        inserted_block_mode=str(config.inserted_block_mode),
        target_shared_correction=config.target_shared_correction,
        correction_scope=str(config.correction_scope),
        insertion_target_mode=str(config.insertion_target_mode),
    )
    if layout_out is not None:
        layout_out.clear()
        if extender.extension_layout is not None:
            layout_out.update(extender.extension_layout)
    return final_depth
