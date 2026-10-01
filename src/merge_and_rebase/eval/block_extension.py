from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from itertools import islice
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, Subset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from ..models.vision_utils import _encode_image
from ..rebase.block_extension.adapters import _InProjCapture  # noqa: F401  (re-exported for existing importers)
from ..rebase.block_extension.config import (  # noqa: F401  (re-exported for existing importers)
    _ANNOTATION_PARAMS,
    _MISPLACED_TOP_LEVEL_KEYS,
    BlockExtensionConfig,
    TargetSharedCorrection,
    _as_correction_scope,
    _as_inserted_block_mode,
    _as_optional_calibration_dataset,
    _as_optional_dict_float,
    _as_optional_int,
    _as_optional_str,
    _as_reference_capture,
    _as_target_shared_correction,
    _as_transport_activation_mode,
    _warn_unknown_block_extension_params,
    block_extension_protocol,
    calibration_dataset_spec,
    resolve_block_extension_config,
    select_loader,
)
from ..rebase.block_extension.schedules import (  # noqa: F401  (re-exported for existing importers)
    balanced_collapse_spans,
    build_extension_layout,
    build_reduction_layout,
    disjoint_collapse_schedule,
    plan_inserted_positions,
    spread_anchor_schedule,
    vision_collapse_schedule,
    vision_locate_collapse_pos,
)

logger = logging.getLogger(__name__)


def _deterministic_calibration_loader(loader, n_batches: int):
    """Freeze a randomized DataLoader for the extender's repeated passes.

    Correction/reference fitting makes several passes over the same calibration
    examples. A RandomSampler would produce different rows on each pass and
    pair unrelated activations. Preserve the sampler's first calibration
    window, then replay it sequentially.
    """

    if not isinstance(loader, DataLoader) or loader.batch_size is None:
        return loader
    if isinstance(loader.sampler, SequentialSampler):
        return loader

    n_items = max(0, int(n_batches)) * int(loader.batch_size)
    indices = list(islice(iter(loader.sampler), n_items))
    frozen_dataset = Subset(loader.dataset, indices)
    return DataLoader(
        frozen_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        persistent_workers=bool(
            getattr(loader, "persistent_workers", False) and loader.num_workers > 0
        ),
    )


class BlockExtender:
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

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(f"[block_extension] {message}")

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

    def _record_corrections(self, endpoint: str, corrections: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        for component, (W, b) in corrections.items():
            self._record_correction(endpoint, component, W, b)

    @staticmethod
    def _inner_block(block: nn.Module) -> nn.Module:
        return block.block if hasattr(block, "block") else block

    @staticmethod
    def _store_input_hook(store: dict[str, list[torch.Tensor]], key: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any):
            if inputs and inputs[0] is not None:
                store[key].append(inputs[0].detach().cpu())

        return hook

    @staticmethod
    def _store_output_hook(store: dict[str, list[torch.Tensor]], key: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            out = output[0] if isinstance(output, tuple) else output
            if out is not None:
                store[key].append(out.detach().cpu())

        return hook

    def _fit_ridge(
        self,
        A: torch.Tensor,
        T: torch.Tensor,
        lambda_reg: float | None = None,
        ridge_id: float = 0.0,
        ridge_target: torch.Tensor | None = None,
    ):
        A = A.float()
        T = T.float()

        mu_A = A.mean(dim=0)
        mu_T = T.mean(dim=0)

        A_c = A - mu_A
        T_c = T - mu_T

        dim_in = A.shape[1]
        resolved_lambda_reg = self._ridge_weight if lambda_reg is None else float(lambda_reg)
        reg = resolved_lambda_reg + ridge_id
        cov = A_c.T @ A_c
        cov = cov + reg * torch.eye(dim_in, device=A.device, dtype=A.dtype)
        if ridge_target is not None:
            ridge_target = ridge_target.float().to(A.device)
            rhs = A_c.T @ T_c + ridge_id * ridge_target
        else:
            rhs = A_c.T @ T_c + ridge_id * torch.eye(dim_in, device=A.device, dtype=A.dtype)

        try:
            W_T = torch.linalg.solve(cov, rhs)
        except RuntimeError:
            W_T = torch.linalg.pinv(cov) @ rhs

        b = mu_T - mu_A @ W_T
        return W_T.T, b

    @staticmethod
    def _match_rows(A: torch.Tensor, T: torch.Tensor):
        n = min(A.shape[0], T.shape[0])
        return A[:n], T[:n].to(device=A.device, non_blocking=True)

    @staticmethod
    def _resolve_depth_delta(curr_layers: int, blocks_to_add: int | None, target_layers_total: int | None) -> int:
        if blocks_to_add is not None:
            n_needed = int(blocks_to_add)
        elif target_layers_total is not None:
            target_layers_total = int(target_layers_total)
            if target_layers_total < 1:
                raise ValueError(f"target_layers_total must be >= 1. Got: {target_layers_total}")
            n_needed = target_layers_total - curr_layers
        else:
            n_needed = 0

        final_depth = curr_layers + n_needed
        if final_depth < 1:
            raise ValueError(f"Requested final depth must be >= 1. Got: {final_depth}")
        return n_needed

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
                        inner.ln_1.register_forward_hook(
                            self._store_output_hook(store, f"{index}.ln_1_output")
                        )
                    )
                    hooks.append(
                        inner.attn.register_forward_hook(
                            self._store_output_hook(store, f"{index}.attn_output")
                        )
                    )
                    hooks.append(
                        inner.ln_2.register_forward_hook(
                            self._store_output_hook(store, f"{index}.ln_2_output")
                        )
                    )
                    hooks.append(
                        inner.mlp.c_fc.register_forward_hook(
                            self._store_output_hook(store, f"{index}.c_fc_output")
                        )
                    )
                    hooks.append(
                        inner.mlp.c_proj.register_forward_hook(
                            self._store_output_hook(store, f"{index}.c_proj_output")
                        )
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
                            f"BRACE calibration loader exhausted after {consumed} batches; "
                            f"requested {n_batches}."
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
    def _capture_single_input(self, model: nn.Module, target: int | str, loader: Iterable[Any], n_batches: int):
        model.eval()
        buffers: list[torch.Tensor] = []

        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any):
            if inputs and inputs[0] is not None:
                buffers.append(inputs[0].detach().cpu())

        if target == "final":
            handle = model.visual.ln_post.register_forward_hook(hook)
        else:
            handle = model.visual.transformer.resblocks[target].register_forward_hook(hook)

        it = iter(loader)
        for _ in range(n_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                break
            _encode_image(model, images.to(self.device))

        handle.remove()

        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    @torch.no_grad()
    def _capture_component_output(self, model: nn.Module, block_idx: int, component: str, loader: Iterable[Any], n_batches: int):
        model.eval()
        buffers: list[torch.Tensor] = []
        block = model.visual.transformer.resblocks[block_idx]
        inner = self._inner_block(block)

        if component == "ln_1":
            target = inner.ln_1
        elif component == "attn":
            target = inner.attn
        elif component == "ln_2":
            target = inner.ln_2
        elif component == "c_fc":
            target = inner.mlp.c_fc
        elif component == "c_proj":
            target = inner.mlp.c_proj
        elif component in ("q", "k", "v"):
            cap = _InProjCapture(inner.attn)
            it = iter(loader)
            for _ in range(n_batches):
                try:
                    images, _ = next(it)
                except StopIteration:
                    break
                _encode_image(model, images.to(self.device))
            cap.restore()
            slice_idx = {"q": 0, "k": 1, "v": 2}[component]
            for tensors in cap.outputs:
                buffers.append(tensors[slice_idx])
            if not buffers:
                return torch.empty(0)
            return torch.cat(buffers, dim=0).flatten(0, 1)
        else:
            raise ValueError(f"Unsupported component '{component}'. Expected ln_1, attn, ln_2, c_fc, c_proj, q, k, or v.")

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            out = output[0] if isinstance(output, tuple) else output
            if out is not None:
                buffers.append(out.detach().cpu())

        handle = target.register_forward_hook(hook)
        it = iter(loader)
        for _ in range(n_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                break
            _encode_image(model, images.to(self.device))

        handle.remove()

        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    @torch.no_grad()
    def _interpolate_block_weights(self, target_block: nn.Module, source_block: nn.Module, alpha: float = 0.5):
        target_inner = self._inner_block(target_block)
        source_inner = self._inner_block(source_block)
        source_params = dict(source_inner.named_parameters())
        for name_t, p_t in target_inner.named_parameters():
            if name_t.startswith("aligner."):
                continue
            p_s = source_params.get(name_t)
            if p_s is not None and p_t.shape == p_s.shape:
                p_t.copy_((1.0 - alpha) * p_t + alpha * p_s)

    @torch.no_grad()
    def _dampen_block_output(self, block: nn.Module, factor: float):
        inner = self._inner_block(block)
        if hasattr(inner, "attn") and hasattr(inner.attn, "out_proj"):
            inner.attn.out_proj.weight.mul_(factor)
        if hasattr(inner, "mlp") and hasattr(inner.mlp, "c_proj"):
            inner.mlp.c_proj.weight.mul_(factor)

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
                raise ValueError(
                    f"Target block position {position} is outside the target depth {len(blocks)}."
                )

        original_device = next(target.parameters()).device
        handles: list[Any] = []
        n_samples = 0
        try:
            target.to(self.device).eval()
            for position in buffers:
                inner = self._inner_block(blocks[position])
                handles.append(
                    inner.mlp.c_proj.register_forward_hook(self._store_output_hook(buffers, position))
                )
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
            f"captured target component references for {len(banks)} positions "
            f"over {n_samples} calibration samples"
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

        from ..rebase.methods.theseus import _compute_procrustes_map_from_cov, _interp_2d_tokens

        rows = int(source_ref.shape[0])
        if rows % n_samples != 0:
            raise ValueError(
                f"Source reference rows ({rows}) are not a multiple of the calibration "
                f"sample count ({n_samples}); cannot recover tokens per sample."
            )
        source_tokens = rows // n_samples
        if int(target_bank.shape[0]) != n_samples:
            raise ValueError(
                f"Target bank holds {int(target_bank.shape[0])} samples, expected {n_samples}."
            )

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
        inner = self._inner_block(block)
        projections = []
        if hasattr(inner, "attn") and hasattr(inner.attn, "out_proj"):
            projections.append(inner.attn.out_proj)
        if hasattr(inner, "mlp") and hasattr(inner.mlp, "c_proj"):
            projections.append(inner.mlp.c_proj)
        if len(projections) != 2:
            raise ValueError(
                "residual_identity requires a block exposing attn.out_proj and mlp.c_proj; "
                f"found {len(projections)} output projections on {type(inner).__name__}."
            )
        for projection in projections:
            projection.weight.zero_()
            if getattr(projection, "bias", None) is not None:
                projection.bias.zero_()

    def _get_ridge(self, component: str, default: float) -> float:
        cr = getattr(self, "_component_ridge", None)
        if cr is None:
            return default
        return float(cr.get(component, default))

    @staticmethod
    def _reference_endpoint_names(lmc_mode: str, share_ft_refs: bool) -> tuple[str, ...]:
        if lmc_mode == "shared":
            return ("ft",) if share_ft_refs else ("base",)
        if lmc_mode == "shared_ft" or share_ft_refs:
            return ("ft",)
        return ("base", "ft")

    @torch.no_grad()
    def _correct_block_weights_cascade(
        self,
        model_name: str,
        model: nn.Module,
        insert_pos: int,
        src_idx: int,
        loader: Iterable[Any],
        n_batches: int,
        ridge_identity: float = 0.0,
        n_iters: int = 1,
        ref_source: str | None = None,
        component_ridge: dict[str, float] | None = None,
        lmc_store: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        lmc_targets: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        insertion_target_mode: str = "direct",
        target_reference: torch.Tensor | None = None,
        target_weight: float = 0.0,
    ):
        block = model.visual.transformer.resblocks[insert_pos]
        inner = self._inner_block(block)
        ref_key = ref_source if ref_source is not None else model_name
        refs = self.reference_inputs[ref_key]
        dim_qkv = inner.attn.in_proj_weight.shape[0] // 3
        self._component_ridge = component_ridge

        def _ridge_target(comp: str) -> torch.Tensor | None:
            if lmc_targets is None or comp not in lmc_targets:
                return None
            W_base, _ = lmc_targets[comp]
            if comp in ("ln_1", "ln_2"):
                return torch.diag(torch.diag(W_base))
            return W_base

        for _ in range(n_iters):
            # Step 1: ln_1 — element-wise (diagonal) absorption
            cur = self._capture_component_output(model, insert_pos, "ln_1", loader, n_batches)
            ref = refs.get(f"{src_idx}.ln_1_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("ln_1", ridge_identity), ridge_target=_ridge_target("ln_1"))
                    if lmc_store is not None:
                        lmc_store["ln_1"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "ln_1", W, b)
                    d = torch.diag(W).to(inner.ln_1.weight.device, dtype=inner.ln_1.weight.dtype)
                    b = b.to(inner.ln_1.bias.device, dtype=inner.ln_1.bias.dtype)
                    inner.ln_1.weight.mul_(d)
                    inner.ln_1.bias.copy_(d * inner.ln_1.bias + b)

            # Step 2: q_proj — slice 0 of fused in_proj
            cur = self._capture_component_output(model, insert_pos, "q", loader, n_batches)
            ref = refs.get(f"{src_idx}.q_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("q", ridge_identity), ridge_target=_ridge_target("q"))
                    if lmc_store is not None:
                        lmc_store["q"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "q", W, b)
                    W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
                    b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
                    w_slice = inner.attn.in_proj_weight.data[:dim_qkv].clone()
                    b_slice = inner.attn.in_proj_bias.data[:dim_qkv].clone()
                    inner.attn.in_proj_weight.data[:dim_qkv] = W @ w_slice
                    inner.attn.in_proj_bias.data[:dim_qkv] = W @ b_slice + b

            # Step 3: k_proj — slice 1 of fused in_proj
            cur = self._capture_component_output(model, insert_pos, "k", loader, n_batches)
            ref = refs.get(f"{src_idx}.k_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("k", ridge_identity), ridge_target=_ridge_target("k"))
                    if lmc_store is not None:
                        lmc_store["k"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "k", W, b)
                    W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
                    b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
                    w_slice = inner.attn.in_proj_weight.data[dim_qkv:2*dim_qkv].clone()
                    b_slice = inner.attn.in_proj_bias.data[dim_qkv:2*dim_qkv].clone()
                    inner.attn.in_proj_weight.data[dim_qkv:2*dim_qkv] = W @ w_slice
                    inner.attn.in_proj_bias.data[dim_qkv:2*dim_qkv] = W @ b_slice + b

            # Step 4: v_proj — slice 2 of fused in_proj
            cur = self._capture_component_output(model, insert_pos, "v", loader, n_batches)
            ref = refs.get(f"{src_idx}.v_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("v", ridge_identity), ridge_target=_ridge_target("v"))
                    if lmc_store is not None:
                        lmc_store["v"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "v", W, b)
                    W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
                    b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
                    w_slice = inner.attn.in_proj_weight.data[2*dim_qkv:3*dim_qkv].clone()
                    b_slice = inner.attn.in_proj_bias.data[2*dim_qkv:3*dim_qkv].clone()
                    inner.attn.in_proj_weight.data[2*dim_qkv:3*dim_qkv] = W @ w_slice
                    inner.attn.in_proj_bias.data[2*dim_qkv:3*dim_qkv] = W @ b_slice + b

            # Step 5: attn output (after out_proj) — full absorption into out_proj
            cur = self._capture_component_output(model, insert_pos, "attn", loader, n_batches)
            ref = refs.get(f"{src_idx}.attn_output")
            if insertion_target_mode == "residual":
                # Pin the post-attention residual stream to the source block's
                # y_s = x_s + a_s instead of matching a_s alone, absorbing the
                # gap between the inserted block's input and x_s exactly once.
                ref_input = refs.get(f"{src_idx}.input")
                cur_input = self._capture_single_input(model, insert_pos, loader, n_batches)
                if ref is not None and ref_input is not None and cur.numel() > 0 and cur_input.numel() > 0:
                    n = min(cur.shape[0], cur_input.shape[0], ref.shape[0], ref_input.shape[0])
                    A = cur[:n]
                    T = ref_input[:n].to(A.device) + ref[:n].to(A.device) - cur_input[:n].to(A.device)
                else:
                    A = T = torch.empty(0)
            elif ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
            else:
                A = T = torch.empty(0)
            if A.numel() > 0 and T.numel() > 0:
                W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("out_proj", ridge_identity), ridge_target=_ridge_target("out_proj"))
                if lmc_store is not None:
                    lmc_store["out_proj"] = (W.clone(), b.clone())
                self._record_correction(model_name, "out_proj", W, b)
                W = W.to(inner.attn.out_proj.weight.device, dtype=inner.attn.out_proj.weight.dtype)
                b = b.to(inner.attn.out_proj.bias.device, dtype=inner.attn.out_proj.bias.dtype)
                inner.attn.out_proj.weight.copy_(W @ inner.attn.out_proj.weight)
                inner.attn.out_proj.bias.copy_(W @ inner.attn.out_proj.bias + b)

            # Step 6: ln_2 — element-wise (diagonal) absorption
            cur = self._capture_component_output(model, insert_pos, "ln_2", loader, n_batches)
            ref = refs.get(f"{src_idx}.ln_2_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("ln_2", ridge_identity), ridge_target=_ridge_target("ln_2"))
                    if lmc_store is not None:
                        lmc_store["ln_2"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "ln_2", W, b)
                    d = torch.diag(W).to(inner.ln_2.weight.device, dtype=inner.ln_2.weight.dtype)
                    b = b.to(inner.ln_2.bias.device, dtype=inner.ln_2.bias.dtype)
                    inner.ln_2.weight.mul_(d)
                    inner.ln_2.bias.copy_(d * inner.ln_2.bias + b)

            # Step 7: mlp.c_fc (before GELU) — full absorption into c_fc
            cur = self._capture_component_output(model, insert_pos, "c_fc", loader, n_batches)
            ref = refs.get(f"{src_idx}.c_fc_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("c_fc", ridge_identity), ridge_target=_ridge_target("c_fc"))
                    if lmc_store is not None:
                        lmc_store["c_fc"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "c_fc", W, b)
                    W = W.to(inner.mlp.c_fc.weight.device, dtype=inner.mlp.c_fc.weight.dtype)
                    b = b.to(inner.mlp.c_fc.bias.device, dtype=inner.mlp.c_fc.bias.dtype)
                    inner.mlp.c_fc.weight.copy_(W @ inner.mlp.c_fc.weight)
                    inner.mlp.c_fc.bias.copy_(W @ inner.mlp.c_fc.bias + b)

            # Step 8: mlp.c_proj (after GELU, before ls_2) — full absorption into c_proj
            cur = self._capture_component_output(model, insert_pos, "c_proj", loader, n_batches)
            ref = refs.get(f"{src_idx}.c_proj_output")
            if ref is not None and target_reference is not None and target_weight > 0.0:
                ref = self._blend_target_reference(
                    ref, target_reference, self._target_reference_samples, target_weight
                )
            if insertion_target_mode == "residual":
                # Target the source block's OUTPUT x_{s+1} = x_s + a_s + m_s,
                # minus the stream actually reaching the MLP, which is the block
                # input plus the already-corrected attention output. Subtracting
                # only the block input here would consume the input gap a second
                # time and invert the block (see the shrink path, which has
                # always subtracted cur_attn).
                ref_input = refs.get(f"{src_idx}.input")
                ref_attn = refs.get(f"{src_idx}.attn_output")
                cur_input = self._capture_single_input(model, insert_pos, loader, n_batches)
                cur_attn = self._capture_component_output(model, insert_pos, "attn", loader, n_batches)
                if (
                    ref is not None and ref_input is not None and ref_attn is not None
                    and cur.numel() > 0 and cur_input.numel() > 0 and cur_attn.numel() > 0
                ):
                    n = min(
                        cur.shape[0], cur_input.shape[0], cur_attn.shape[0],
                        ref.shape[0], ref_input.shape[0], ref_attn.shape[0],
                    )
                    A = cur[:n]
                    T = (
                        ref_input[:n].to(A.device) + ref_attn[:n].to(A.device) + ref[:n].to(A.device)
                        - cur_input[:n].to(A.device) - cur_attn[:n].to(A.device)
                    )
                else:
                    A = T = torch.empty(0)
            elif ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
            else:
                A = T = torch.empty(0)
            if A.numel() > 0 and T.numel() > 0:
                W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("c_proj", ridge_identity), ridge_target=_ridge_target("c_proj"))
                if lmc_store is not None:
                    lmc_store["c_proj"] = (W.clone(), b.clone())
                self._record_correction(model_name, "c_proj", W, b)
                W = W.to(inner.mlp.c_proj.weight.device, dtype=inner.mlp.c_proj.weight.dtype)
                b = b.to(inner.mlp.c_proj.bias.device, dtype=inner.mlp.c_proj.bias.dtype)
                inner.mlp.c_proj.weight.copy_(W @ inner.mlp.c_proj.weight)
                inner.mlp.c_proj.bias.copy_(W @ inner.mlp.c_proj.bias + b)

    @torch.no_grad()
    def _correct_collapsed_block_weights_cascade(
        self,
        model_name: str,
        model: nn.Module,
        block_idx: int,
        span_start_idx: int,
        span_end_idx: int,
        output_ref_key: str,
        loader: Iterable[Any],
        n_batches: int,
        ridge_identity: float = 0.0,
        n_iters: int = 1,
        ref_source: str | None = None,
        component_ridge: dict[str, float] | None = None,
        lmc_store: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        lmc_targets: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        block = model.visual.transformer.resblocks[block_idx]
        inner = self._inner_block(block)
        ref_key = ref_source if ref_source is not None else model_name
        refs = self.reference_inputs[ref_key]
        dim_qkv = inner.attn.in_proj_weight.shape[0] // 3
        self._component_ridge = component_ridge

        def _ridge_target(comp: str) -> torch.Tensor | None:
            if lmc_targets is None or comp not in lmc_targets:
                return None
            W_base, _ = lmc_targets[comp]
            if comp in ("ln_1", "ln_2"):
                return torch.diag(torch.diag(W_base))
            return W_base

        for _ in range(n_iters):
            # Step 1: ln_1 tracks the start of the collapsed span.
            cur = self._capture_component_output(model, block_idx, "ln_1", loader, n_batches)
            ref = refs.get(f"{span_start_idx}.ln_1_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("ln_1", ridge_identity), ridge_target=_ridge_target("ln_1"))
                    if lmc_store is not None:
                        lmc_store["ln_1"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "ln_1", W, b)
                    d = torch.diag(W).to(inner.ln_1.weight.device, dtype=inner.ln_1.weight.dtype)
                    b = b.to(inner.ln_1.bias.device, dtype=inner.ln_1.bias.dtype)
                    inner.ln_1.weight.mul_(d)
                    inner.ln_1.bias.copy_(d * inner.ln_1.bias + b)

            # Step 2: q_proj follows the first block input distribution.
            cur = self._capture_component_output(model, block_idx, "q", loader, n_batches)
            ref = refs.get(f"{span_start_idx}.q_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("q", ridge_identity), ridge_target=_ridge_target("q"))
                    if lmc_store is not None:
                        lmc_store["q"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "q", W, b)
                    W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
                    b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
                    w_slice = inner.attn.in_proj_weight.data[:dim_qkv].clone()
                    b_slice = inner.attn.in_proj_bias.data[:dim_qkv].clone()
                    inner.attn.in_proj_weight.data[:dim_qkv] = W @ w_slice
                    inner.attn.in_proj_bias.data[:dim_qkv] = W @ b_slice + b

            # Step 3: k_proj follows the first block input distribution.
            cur = self._capture_component_output(model, block_idx, "k", loader, n_batches)
            ref = refs.get(f"{span_start_idx}.k_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("k", ridge_identity), ridge_target=_ridge_target("k"))
                    if lmc_store is not None:
                        lmc_store["k"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "k", W, b)
                    W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
                    b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
                    w_slice = inner.attn.in_proj_weight.data[dim_qkv:2*dim_qkv].clone()
                    b_slice = inner.attn.in_proj_bias.data[dim_qkv:2*dim_qkv].clone()
                    inner.attn.in_proj_weight.data[dim_qkv:2*dim_qkv] = W @ w_slice
                    inner.attn.in_proj_bias.data[dim_qkv:2*dim_qkv] = W @ b_slice + b

            # Step 4: v_proj follows the first block input distribution.
            cur = self._capture_component_output(model, block_idx, "v", loader, n_batches)
            ref = refs.get(f"{span_start_idx}.v_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("v", ridge_identity), ridge_target=_ridge_target("v"))
                    if lmc_store is not None:
                        lmc_store["v"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "v", W, b)
                    W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
                    b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
                    w_slice = inner.attn.in_proj_weight.data[2*dim_qkv:3*dim_qkv].clone()
                    b_slice = inner.attn.in_proj_bias.data[2*dim_qkv:3*dim_qkv].clone()
                    inner.attn.in_proj_weight.data[2*dim_qkv:3*dim_qkv] = W @ w_slice
                    inner.attn.in_proj_bias.data[2*dim_qkv:3*dim_qkv] = W @ b_slice + b

            # Step 5: attn output is corrected to reproduce the last removed block's post-attn residual.
            cur = self._capture_component_output(model, block_idx, "attn", loader, n_batches)
            cur_input = self._capture_single_input(model, block_idx, loader, n_batches)
            ref_input = refs.get(f"{span_end_idx}.input")
            ref_attn = refs.get(f"{span_end_idx}.attn_output")
            if ref_input is not None and ref_attn is not None and cur.numel() > 0 and cur_input.numel() > 0:
                n = min(cur.shape[0], cur_input.shape[0], ref_input.shape[0], ref_attn.shape[0])
                A = cur[:n]
                T = ref_input[:n] + ref_attn[:n] - cur_input[:n]
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("out_proj", ridge_identity), ridge_target=_ridge_target("out_proj"))
                    if lmc_store is not None:
                        lmc_store["out_proj"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "out_proj", W, b)
                    W = W.to(inner.attn.out_proj.weight.device, dtype=inner.attn.out_proj.weight.dtype)
                    b = b.to(inner.attn.out_proj.bias.device, dtype=inner.attn.out_proj.bias.dtype)
                    inner.attn.out_proj.weight.copy_(W @ inner.attn.out_proj.weight)
                    inner.attn.out_proj.bias.copy_(W @ inner.attn.out_proj.bias + b)

            # Step 6: ln_2 tracks the tail block after the attn residual has been matched.
            cur = self._capture_component_output(model, block_idx, "ln_2", loader, n_batches)
            ref = refs.get(f"{span_end_idx}.ln_2_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("ln_2", ridge_identity), ridge_target=_ridge_target("ln_2"))
                    if lmc_store is not None:
                        lmc_store["ln_2"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "ln_2", W, b)
                    d = torch.diag(W).to(inner.ln_2.weight.device, dtype=inner.ln_2.weight.dtype)
                    b = b.to(inner.ln_2.bias.device, dtype=inner.ln_2.bias.dtype)
                    inner.ln_2.weight.mul_(d)
                    inner.ln_2.bias.copy_(d * inner.ln_2.bias + b)

            # Step 7: c_fc tracks the tail block MLP hidden state.
            cur = self._capture_component_output(model, block_idx, "c_fc", loader, n_batches)
            ref = refs.get(f"{span_end_idx}.c_fc_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("c_fc", ridge_identity), ridge_target=_ridge_target("c_fc"))
                    if lmc_store is not None:
                        lmc_store["c_fc"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "c_fc", W, b)
                    W = W.to(inner.mlp.c_fc.weight.device, dtype=inner.mlp.c_fc.weight.dtype)
                    b = b.to(inner.mlp.c_fc.bias.device, dtype=inner.mlp.c_fc.bias.dtype)
                    inner.mlp.c_fc.weight.copy_(W @ inner.mlp.c_fc.weight)
                    inner.mlp.c_fc.bias.copy_(W @ inner.mlp.c_fc.bias + b)

            # Step 8: c_proj is corrected against the final target output of the removed span.
            cur = self._capture_component_output(model, block_idx, "c_proj", loader, n_batches)
            cur_input = self._capture_single_input(model, block_idx, loader, n_batches)
            cur_attn = self._capture_component_output(model, block_idx, "attn", loader, n_batches)
            ref = refs.get(output_ref_key)
            if ref is not None and cur.numel() > 0 and cur_input.numel() > 0 and cur_attn.numel() > 0:
                n = min(cur.shape[0], cur_input.shape[0], cur_attn.shape[0], ref.shape[0])
                A = cur[:n]
                T = ref[:n] - cur_input[:n] - cur_attn[:n]
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("c_proj", ridge_identity), ridge_target=_ridge_target("c_proj"))
                    if lmc_store is not None:
                        lmc_store["c_proj"] = (W.clone(), b.clone())
                    self._record_correction(model_name, "c_proj", W, b)
                    W = W.to(inner.mlp.c_proj.weight.device, dtype=inner.mlp.c_proj.weight.dtype)
                    b = b.to(inner.mlp.c_proj.bias.device, dtype=inner.mlp.c_proj.bias.dtype)
                    inner.mlp.c_proj.weight.copy_(W @ inner.mlp.c_proj.weight)
                    inner.mlp.c_proj.bias.copy_(W @ inner.mlp.c_proj.bias + b)

    @torch.no_grad()
    def _apply_block_corrections(
        self,
        model: nn.Module,
        insert_pos: int,
        corrections: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ):
        block = model.visual.transformer.resblocks[insert_pos]
        inner = self._inner_block(block)
        dim_qkv = inner.attn.in_proj_weight.shape[0] // 3

        # Step 1: ln_1
        if "ln_1" in corrections:
            W, b = corrections["ln_1"]
            d = torch.diag(W).to(inner.ln_1.weight.device, dtype=inner.ln_1.weight.dtype)
            b = b.to(inner.ln_1.bias.device, dtype=inner.ln_1.bias.dtype)
            inner.ln_1.weight.mul_(d)
            inner.ln_1.bias.copy_(d * inner.ln_1.bias + b)

        # Step 2: q_proj
        if "q" in corrections:
            W, b = corrections["q"]
            W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
            b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
            w_slice = inner.attn.in_proj_weight.data[:dim_qkv].clone()
            b_slice = inner.attn.in_proj_bias.data[:dim_qkv].clone()
            inner.attn.in_proj_weight.data[:dim_qkv] = W @ w_slice
            inner.attn.in_proj_bias.data[:dim_qkv] = W @ b_slice + b

        # Step 3: k_proj
        if "k" in corrections:
            W, b = corrections["k"]
            W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
            b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
            w_slice = inner.attn.in_proj_weight.data[dim_qkv:2*dim_qkv].clone()
            b_slice = inner.attn.in_proj_bias.data[dim_qkv:2*dim_qkv].clone()
            inner.attn.in_proj_weight.data[dim_qkv:2*dim_qkv] = W @ w_slice
            inner.attn.in_proj_bias.data[dim_qkv:2*dim_qkv] = W @ b_slice + b

        # Step 4: v_proj
        if "v" in corrections:
            W, b = corrections["v"]
            W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
            b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
            w_slice = inner.attn.in_proj_weight.data[2*dim_qkv:3*dim_qkv].clone()
            b_slice = inner.attn.in_proj_bias.data[2*dim_qkv:3*dim_qkv].clone()
            inner.attn.in_proj_weight.data[2*dim_qkv:3*dim_qkv] = W @ w_slice
            inner.attn.in_proj_bias.data[2*dim_qkv:3*dim_qkv] = W @ b_slice + b

        # Step 5: out_proj
        if "out_proj" in corrections:
            W, b = corrections["out_proj"]
            W = W.to(inner.attn.out_proj.weight.device, dtype=inner.attn.out_proj.weight.dtype)
            b = b.to(inner.attn.out_proj.bias.device, dtype=inner.attn.out_proj.bias.dtype)
            inner.attn.out_proj.weight.copy_(W @ inner.attn.out_proj.weight)
            inner.attn.out_proj.bias.copy_(W @ inner.attn.out_proj.bias + b)

        # Step 6: ln_2
        if "ln_2" in corrections:
            W, b = corrections["ln_2"]
            d = torch.diag(W).to(inner.ln_2.weight.device, dtype=inner.ln_2.weight.dtype)
            b = b.to(inner.ln_2.bias.device, dtype=inner.ln_2.bias.dtype)
            inner.ln_2.weight.mul_(d)
            inner.ln_2.bias.copy_(d * inner.ln_2.bias + b)

        # Step 7: c_fc
        if "c_fc" in corrections:
            W, b = corrections["c_fc"]
            W = W.to(inner.mlp.c_fc.weight.device, dtype=inner.mlp.c_fc.weight.dtype)
            b = b.to(inner.mlp.c_fc.bias.device, dtype=inner.mlp.c_fc.bias.dtype)
            inner.mlp.c_fc.weight.copy_(W @ inner.mlp.c_fc.weight)
            inner.mlp.c_fc.bias.copy_(W @ inner.mlp.c_fc.bias + b)

        # Step 8: c_proj
        if "c_proj" in corrections:
            W, b = corrections["c_proj"]
            W = W.to(inner.mlp.c_proj.weight.device, dtype=inner.mlp.c_proj.weight.dtype)
            b = b.to(inner.mlp.c_proj.bias.device, dtype=inner.mlp.c_proj.bias.dtype)
            inner.mlp.c_proj.weight.copy_(W @ inner.mlp.c_proj.weight)
            inner.mlp.c_proj.bias.copy_(W @ inner.mlp.c_proj.bias + b)

    @staticmethod
    def _build_duplication_schedule(
        curr_layers: int,
        n_needed: int,
        insertion_order: str,
        extension_density: str,
    ):
        if n_needed <= 0:
            return []

        if insertion_order == "bottom-top":
            priority = list(range(curr_layers))
        elif insertion_order == "top-bottom":
            priority = list(range(curr_layers - 1, -1, -1))
        elif insertion_order == "random":
            priority = list(range(curr_layers))
            np.random.shuffle(priority)
        else:
            raise ValueError(
                "Unsupported insertion_order. Expected one of: bottom-top, top-bottom, random. "
                f"Got: {insertion_order}"
            )

        if not priority:
            return []

        if extension_density == "clump":
            return [priority[0]] * n_needed
        if extension_density == "spread_mod":
            n_gaps = curr_layers - 1
            return [i % n_gaps for i in range(n_needed)]
        if extension_density != "spread":
            raise ValueError(
                "Unsupported extension_density. Expected one of: spread, spread_mod, clump. "
                f"Got: {extension_density}"
            )

        # Once there is at least one duplicate per block every block is an
        # anchor anyway; below that keep the final block out of the anchor set.
        # Duplicating it puts an extra full block update directly before the
        # output norm with no later layer to absorb it, which is far more
        # destructive than any other placement (Qwen2.5-1.5B 28 -> 36,
        # interpolate, no correction: wikitext-2 ppl 1847 with it vs 28 without).
        n_positions = curr_layers if n_needed >= curr_layers else max(1, curr_layers - 1)
        return spread_anchor_schedule(n_needed, n_positions, insertion_order)

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
                raise ValueError(
                    f"inserted_block_mode='{inserted_block_mode}' requires skip_correction=True."
                )
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
            k: v for k, v in common_kwargs.items()
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
                "Vision extension_strategy='interpolate' is no longer supported; "
                "use 'interpolate_per_weight'."
            )
        if strategy == "duplicate":
            raise ValueError(
                "Vision extension_strategy='duplicate' is no longer supported; "
                "use 'duplicate_per_weight'."
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
            f"Unsupported correction_scope '{scope}'. "
            "Expected 'inserted', 'interleaved_once', or 'iterative_all'."
        )

    @torch.no_grad()
    def _correct_one_block(
        self,
        *,
        reference_models: Mapping[str, nn.Module],
        position: int,
        ref_block_idx: int,
        loader: Iterable[Any],
        n_batches: int,
        ridge_identity: float,
        n_cascade_iters: int,
        share_ft_refs: bool,
        component_ridge: dict[str, float] | None,
        lmc_mode: str,
        insertion_target_mode: str = "direct",
        target_reference: torch.Tensor | None = None,
        target_weight: float = 0.0,
    ) -> None:
        """Capture references for one block and fit/absorb its correction.

        ``position`` is the block's index in the current chain; ``ref_block_idx``
        is the original block whose pristine activations are the target. For an
        inserted block those differ; for an original block being repaired they
        refer to the same block at its shifted position.
        """
        self._capture_per_weight_reference_subset(
            reference_models,
            endpoints=self._reference_endpoint_names(lmc_mode, share_ft_refs),
            block_indices=(ref_block_idx,),
            input_indices=(),
            loader=loader,
            n_batches=n_batches,
        )
        base_ref = "ft" if share_ft_refs else None
        common = dict(
            ridge_identity=ridge_identity,
            n_iters=n_cascade_iters,
            component_ridge=component_ridge,
            insertion_target_mode=insertion_target_mode,
            target_reference=target_reference,
            target_weight=target_weight,
        )
        if lmc_mode == "independent":
            self._correct_block_weights_cascade(
                "base", self.model_base, position, ref_block_idx, loader, n_batches,
                ref_source=base_ref, **common,
            )
            self._correct_block_weights_cascade(
                "ft", self.model_ft, position, ref_block_idx, loader, n_batches, **common,
            )
        elif lmc_mode == "steer":
            base_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            self._correct_block_weights_cascade(
                "base", self.model_base, position, ref_block_idx, loader, n_batches,
                ref_source=base_ref, lmc_store=base_corrections, **common,
            )
            self._correct_block_weights_cascade(
                "ft", self.model_ft, position, ref_block_idx, loader, n_batches,
                lmc_targets=base_corrections, **common,
            )
        elif lmc_mode == "shared":
            base_corrections = {}
            self._correct_block_weights_cascade(
                "base", self.model_base, position, ref_block_idx, loader, n_batches,
                ref_source=base_ref, lmc_store=base_corrections, **common,
            )
            self._apply_block_corrections(self.model_ft, position, base_corrections)
            self._record_corrections("ft", base_corrections)
        elif lmc_mode == "shared_ft":
            ft_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            self._correct_block_weights_cascade(
                "ft", self.model_ft, position, ref_block_idx, loader, n_batches,
                lmc_store=ft_corrections, **common,
            )
            self._apply_block_corrections(self.model_base, position, ft_corrections)
            self._record_corrections("base", ft_corrections)
        else:
            raise ValueError(
                f"Unsupported lmc_mode '{lmc_mode}'. "
                "Expected 'independent', 'steer', 'shared', or 'shared_ft'."
            )

    @torch.no_grad()
    def _extend_per_weight(
        self,
        *,
        loader: Iterable[Any],
        n_batches: int,
        dampening_factor: float,
        blocks_to_add: int | None,
        target_layers_total: int | None,
        insertion_order: str,
        extension_density: str,
        ridge_identity: float = 0.0,
        per_weight_mode: str = "cascade",
        n_cascade_iters: int = 1,
        share_ft_refs: bool = False,
        skip_correction: bool = False,
        component_ridge: dict[str, float] | None = None,
        lmc_mode: str = "independent",
        inserted_block_mode: str = "ariadne",
        correction_scope: str = "inserted",
        insertion_target_mode: str = "direct",
        target_shared_correction: TargetSharedCorrection | None = None,
    ) -> int:
        if per_weight_mode not in {"cascade", "duplicate"}:
            raise ValueError(f"Unsupported per_weight_mode '{per_weight_mode}'. Expected 'cascade' or 'duplicate'.")
        self._vprint(f"starting per-weight extension (mode={per_weight_mode})")
        if not skip_correction:
            reference_models: dict[str, nn.Module] = {
                "base": deepcopy(self.model_base).cpu(),
                "ft": deepcopy(self.model_ft).cpu(),
            }
            self._vprint("created pristine CPU reference endpoints for lazy capture")
        else:
            reference_models = {}
            self._vprint("skip_correction enabled: skipping reference capture")

        curr_layers = len(self.model_base.visual.transformer.resblocks)
        n_needed = self._resolve_depth_delta(curr_layers, blocks_to_add, target_layers_total)

        if n_needed <= 0:
            logger.info("Block extension: no extension needed.")
            self._vprint("no extension needed")
            return curr_layers

        schedule = self._build_duplication_schedule(
            curr_layers=curr_layers,
            n_needed=n_needed,
            insertion_order=insertion_order,
            extension_density=extension_density,
        )

        logger.info("Block extension planned duplications: %s", schedule)
        self._vprint(f"planned duplications: {schedule}")

        # Final positions are needed before the loop: the correction step only
        # knows the position in the partially built chain, but a target-side
        # reference must be addressed by the block's position in the finished
        # model, which is what the target backbone's own depth indexes.
        inserted_positions = plan_inserted_positions(curr_layers, schedule)
        self._target_reference_banks = {}
        self._target_reference_samples = 0
        target_weight = 0.0
        if target_shared_correction is not None and target_shared_correction.active:
            target_weight = float(target_shared_correction.target_weight)
            target_batches = target_shared_correction.num_batches or n_batches
            self._target_reference_banks = self._capture_target_component_references(
                positions=inserted_positions,
                loader=loader,
                n_batches=target_batches,
            )

        orig_base = list(self.model_base.visual.transformer.resblocks)
        orig_ft = list(self.model_ft.visual.transformer.resblocks)

        chain_base = [{"mod": b, "orig_idx": i, "inserted": False} for i, b in enumerate(orig_base)]
        chain_ft = [{"mod": b, "orig_idx": i, "inserted": False} for i, b in enumerate(orig_ft)]

        step_iter = _iter_with_progress(
            enumerate(schedule, start=1),
            total=len(schedule),
            desc="block_extension.per_weight",
            enabled=self.show_progress,
        )
        for step, src_idx in step_iter:
            logger.info("Block extension step %d/%d. Source block: %d", step, len(schedule), src_idx)
            self._vprint(f"step {step}/{len(schedule)} source_block={src_idx}")

            dup_base = deepcopy(orig_base[src_idx])
            dup_ft = deepcopy(orig_ft[src_idx])

            # The neighbour is defined for every init mode: ``cascade`` blends
            # its weights in, and the interpolated-activation baseline reads
            # its activations. The last block has no successor and is its own
            # neighbour, matching the clamp used for the weight midpoint.
            src_next = min(src_idx + 1, len(orig_base) - 1)
            if per_weight_mode == "cascade":
                self._interpolate_block_weights(dup_base, orig_base[src_next], alpha=0.5)
                self._interpolate_block_weights(dup_ft, orig_ft[src_next], alpha=0.5)

            if dampening_factor < 1.0:
                self._dampen_block_output(dup_base, dampening_factor)
                self._dampen_block_output(dup_ft, dampening_factor)

            if inserted_block_mode in {"residual_identity", "residual_identity_inert"}:
                self._zero_block_output_projections(dup_base)
                self._zero_block_output_projections(dup_ft)
            if inserted_block_mode == "residual_identity_inert":
                # Give the FT endpoint the base endpoint's inserted block, so
                # the inserted position's task vector is zero on every
                # parameter and not only on the output projections. The base
                # endpoint is untouched, so this arm and 'residual_identity'
                # fit byte-identical transport maps and differ in exactly one
                # quantity: the delta being transported.
                dup_ft.load_state_dict(dup_base.state_dict())

            insert_pos = -1
            for i, item in enumerate(chain_base):
                if item["orig_idx"] == src_idx:
                    insert_pos = i
            insert_pos += 1

            inserted_meta = {"orig_idx": src_idx, "inserted": True, "neighbour_orig_idx": src_next}
            chain_base.insert(insert_pos, {"mod": dup_base, **inserted_meta})
            chain_ft.insert(insert_pos, {"mod": dup_ft, **inserted_meta})

            self.model_base.visual.transformer.resblocks = nn.ModuleList([x["mod"] for x in chain_base])
            self.model_ft.visual.transformer.resblocks = nn.ModuleList([x["mod"] for x in chain_ft])

            if not skip_correction:
                # Original-block repairs below keep their ordinary source
                # targets; only the inserted block's target is blended.
                self._correct_one_block(
                    reference_models=reference_models,
                    position=insert_pos,
                    ref_block_idx=src_idx,
                    loader=loader,
                    n_batches=n_batches,
                    ridge_identity=ridge_identity,
                    n_cascade_iters=n_cascade_iters,
                    share_ft_refs=share_ft_refs,
                    component_ridge=component_ridge,
                    lmc_mode=lmc_mode,
                    insertion_target_mode=insertion_target_mode,
                    target_reference=self._target_reference_banks.get(inserted_positions[step - 1]),
                    target_weight=target_weight,
                )
                # Repair the original blocks the insertion just disturbed. The
                # bottom-to-top schedule visits each original block as an
                # insertion source exactly once, so correcting the block
                # immediately above the new insertion touches every original
                # block once and always fits against an input that is already
                # final: everything below it has been edited for the last time,
                # and later edits land above it. A pass that instead corrected
                # original blocks *below* an already-fitted inserted block
                # would silently invalidate that block's absorbed correction.
                for original_idx in self._original_blocks_to_correct(
                    chain_base, insert_pos=insert_pos, scope=correction_scope
                ):
                    self._correct_one_block(
                        reference_models=reference_models,
                        position=self._original_block_position(chain_base, original_idx),
                        ref_block_idx=original_idx,
                        loader=loader,
                        n_batches=n_batches,
                        ridge_identity=ridge_identity,
                        n_cascade_iters=n_cascade_iters,
                        share_ft_refs=share_ft_refs,
                        component_ridge=component_ridge,
                        lmc_mode=lmc_mode,
                        insertion_target_mode=insertion_target_mode,
                    )

        final_depth = len(self.model_base.visual.transformer.resblocks)
        self.extension_layout = build_extension_layout(chain_base)
        self._target_reference_banks = {}
        self.reference_inputs = {"base": {}, "ft": {}}
        reference_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._vprint(f"per-weight extension completed. final_depth={final_depth}")
        return final_depth

    @torch.no_grad()
    def _shrink_per_weight(
        self,
        *,
        loader: Iterable[Any],
        n_batches: int,
        dampening_factor: float,
        blocks_to_add: int | None,
        target_layers_total: int | None,
        insertion_order: str,
        extension_density: str,
        collapse_schedule: str = "cascade",
        ridge_identity: float = 0.0,
        per_weight_mode: str = "cascade",
        n_cascade_iters: int = 1,
        share_ft_refs: bool = False,
        skip_correction: bool = False,
        component_ridge: dict[str, float] | None = None,
        lmc_mode: str = "independent",
    ) -> int:
        if per_weight_mode not in {"cascade", "duplicate"}:
            raise ValueError(f"Unsupported per_weight_mode '{per_weight_mode}'. Expected 'cascade' or 'duplicate'.")
        self._vprint(f"starting per-weight shrink (mode={per_weight_mode})")
        if not skip_correction:
            reference_models: dict[str, nn.Module] = {
                "base": deepcopy(self.model_base).cpu(),
                "ft": deepcopy(self.model_ft).cpu(),
            }
            self._vprint("created pristine CPU reference endpoints for lazy capture")
        else:
            reference_models = {}
            self._vprint("skip_correction enabled: skipping reference capture")

        curr_layers = len(self.model_base.visual.transformer.resblocks)
        n_needed = self._resolve_depth_delta(curr_layers, blocks_to_add, target_layers_total)

        if n_needed >= 0:
            logger.info("Block shrink: no shrink needed.")
            self._vprint("no shrink needed")
            return curr_layers

        n_to_remove = -n_needed
        if collapse_schedule == "disjoint_spans":
            schedule = disjoint_collapse_schedule(curr_layers, n_to_remove, insertion_order)
        elif collapse_schedule == "cascade":
            schedule = self._build_collapse_schedule(
                curr_layers=curr_layers,
                n_to_remove=n_to_remove,
                insertion_order=insertion_order,
                extension_density=extension_density,
            )
        else:
            raise ValueError(
                f"Unsupported collapse_schedule '{collapse_schedule}'. "
                "Expected 'cascade' or 'disjoint_spans'."
            )

        logger.info("Block shrink planned collapses: %s", schedule)
        self._vprint(f"planned collapses: {schedule}")

        orig_base = list(self.model_base.visual.transformer.resblocks)
        orig_ft = list(self.model_ft.visual.transformer.resblocks)
        orig_depth = len(orig_base)

        chain_base = [{"mod": b, "orig_idxs": (i,)} for i, b in enumerate(orig_base)]
        chain_ft = [{"mod": b, "orig_idxs": (i,)} for i, b in enumerate(orig_ft)]

        step_iter = _iter_with_progress(
            enumerate(schedule, start=1),
            total=len(schedule),
            desc="block_extension.shrink_per_weight",
            enabled=self.show_progress,
        )
        for step, anchor_orig_idx in step_iter:
            collapse_pos = self._locate_collapse_pos(chain_base, anchor_orig_idx)
            left_base = chain_base[collapse_pos]
            right_base = chain_base[collapse_pos + 1]
            left_ft = chain_ft[collapse_pos]
            right_ft = chain_ft[collapse_pos + 1]
            merged_orig_idxs = tuple(left_base["orig_idxs"] + right_base["orig_idxs"])
            logger.info(
                "Block shrink step %d/%d. Merge span %s + %s -> %s",
                step,
                len(schedule),
                left_base["orig_idxs"],
                right_base["orig_idxs"],
                merged_orig_idxs,
            )
            self._vprint(
                f"step {step}/{len(schedule)} merge_spans={left_base['orig_idxs']}+{right_base['orig_idxs']} -> {merged_orig_idxs}"
            )

            merged_base = deepcopy(left_base["mod"])
            merged_ft = deepcopy(left_ft["mod"])
            if per_weight_mode == "cascade":
                self._interpolate_block_weights(merged_base, right_base["mod"], alpha=0.5)
                self._interpolate_block_weights(merged_ft, right_ft["mod"], alpha=0.5)

            if dampening_factor < 1.0:
                self._dampen_block_output(merged_base, dampening_factor)
                self._dampen_block_output(merged_ft, dampening_factor)

            chain_base[collapse_pos : collapse_pos + 2] = [{"mod": merged_base, "orig_idxs": merged_orig_idxs}]
            chain_ft[collapse_pos : collapse_pos + 2] = [{"mod": merged_ft, "orig_idxs": merged_orig_idxs}]

            self.model_base.visual.transformer.resblocks = nn.ModuleList([x["mod"] for x in chain_base])
            self.model_ft.visual.transformer.resblocks = nn.ModuleList([x["mod"] for x in chain_ft])

            if not skip_correction:
                span_start_idx = merged_orig_idxs[0]
                span_end_idx = merged_orig_idxs[-1]
                output_ref_key = "final.input" if span_end_idx + 1 >= orig_depth else f"{span_end_idx + 1}.input"
                output_input_index: int | str = (
                    "final" if span_end_idx + 1 >= orig_depth else span_end_idx + 1
                )
                self._capture_per_weight_reference_subset(
                    reference_models,
                    endpoints=self._reference_endpoint_names(lmc_mode, share_ft_refs),
                    block_indices=(span_start_idx, span_end_idx),
                    input_indices=(span_end_idx, output_input_index),
                    loader=loader,
                    n_batches=n_batches,
                )
                self._diagnostic_context = {
                    "structural_step": step,
                    "final_block": collapse_pos,
                    "source_block": span_start_idx,
                }
                base_ref = "ft" if share_ft_refs else None
                if lmc_mode == "independent":
                    self._correct_collapsed_block_weights_cascade(
                        "base",
                        self.model_base,
                        collapse_pos,
                        span_start_idx,
                        span_end_idx,
                        output_ref_key,
                        loader,
                        n_batches,
                        ridge_identity=ridge_identity,
                        n_iters=n_cascade_iters,
                        ref_source=base_ref,
                        component_ridge=component_ridge,
                    )
                    self._correct_collapsed_block_weights_cascade(
                        "ft",
                        self.model_ft,
                        collapse_pos,
                        span_start_idx,
                        span_end_idx,
                        output_ref_key,
                        loader,
                        n_batches,
                        ridge_identity=ridge_identity,
                        n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                    )
                elif lmc_mode == "steer":
                    base_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    self._correct_collapsed_block_weights_cascade(
                        "base",
                        self.model_base,
                        collapse_pos,
                        span_start_idx,
                        span_end_idx,
                        output_ref_key,
                        loader,
                        n_batches,
                        ridge_identity=ridge_identity,
                        n_iters=n_cascade_iters,
                        ref_source=base_ref,
                        component_ridge=component_ridge,
                        lmc_store=base_corrections,
                    )
                    self._correct_collapsed_block_weights_cascade(
                        "ft",
                        self.model_ft,
                        collapse_pos,
                        span_start_idx,
                        span_end_idx,
                        output_ref_key,
                        loader,
                        n_batches,
                        ridge_identity=ridge_identity,
                        n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                        lmc_targets=base_corrections,
                    )
                elif lmc_mode == "shared":
                    base_corrections = {}
                    self._correct_collapsed_block_weights_cascade(
                        "base",
                        self.model_base,
                        collapse_pos,
                        span_start_idx,
                        span_end_idx,
                        output_ref_key,
                        loader,
                        n_batches,
                        ridge_identity=ridge_identity,
                        n_iters=n_cascade_iters,
                        ref_source=base_ref,
                        component_ridge=component_ridge,
                        lmc_store=base_corrections,
                    )
                    self._apply_block_corrections(self.model_ft, collapse_pos, base_corrections)
                    self._record_corrections("ft", base_corrections)
                elif lmc_mode == "shared_ft":
                    ft_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    self._correct_collapsed_block_weights_cascade(
                        "ft",
                        self.model_ft,
                        collapse_pos,
                        span_start_idx,
                        span_end_idx,
                        output_ref_key,
                        loader,
                        n_batches,
                        ridge_identity=ridge_identity,
                        n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                        lmc_store=ft_corrections,
                    )
                    self._apply_block_corrections(self.model_base, collapse_pos, ft_corrections)
                    self._record_corrections("base", ft_corrections)
                else:
                    raise ValueError(
                        f"Unsupported lmc_mode '{lmc_mode}'. "
                        "Expected 'independent', 'steer', 'shared', or 'shared_ft'."
                    )

        final_depth = len(self.model_base.visual.transformer.resblocks)
        # The extension path publishes its realized layout for downstream
        # target-informed code; the reduction path must too, or a caller that
        # passes ``layout_out`` silently receives an empty dict.
        self.extension_layout = build_reduction_layout(chain_base)
        self.reference_inputs = {"base": {}, "ft": {}}
        reference_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._vprint(f"per-weight shrink completed. final_depth={final_depth}")
        return final_depth


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
    resolved_target_layers_total = target_layers_total if target_layers_total is not None else config.target_layers_total
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


def _iter_with_progress(iterable: Any, *, total: int, desc: str, enabled: bool) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
