from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ...models.patch_openclip_attention import merge_openclip_vit_attn
from ..base import TensorDict
from ..registry import register
from . import theseus as _t

logger = logging.getLogger(__name__)

_VISUAL_PREFIX = "visual."
_ZERO_KEYS = {"class_embedding", "positional_embedding", "conv1.weight"}


class _BiCoHook:
    """Register forward hooks for input activations and backward hooks for output gradients."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        scope: torch.nn.Module | None = None,
        projection_mode: str = "gradient",
    ):
        self.model = scope if scope is not None else _t._visual_module(model)
        # "gradient" keeps the legacy behaviour (proj's in-map is read from the
        # ln_post output-gradient store). "tokens"/"pooled" capture the actual
        # activation that feeds visual.proj, pre- and post-pooling respectively.
        self.projection_mode = str(projection_mode)
        self.projection_input: torch.Tensor | None = None
        self.inputs: dict[str, torch.Tensor] = {}
        self.in_grads: dict[str, torch.Tensor] = {}
        self.out_grads: dict[str, torch.Tensor] = {}
        self._forward_handles: list[Any] = []
        self._backward_handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._forward_handles.append(
            self.model.register_forward_hook(self._make_forward_hook(""))
        )
        self._backward_handles.append(
            self.model.register_full_backward_hook(self._make_backward_hook(""))
        )
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if list(module.parameters(recurse=False)):
                self._forward_handles.append(
                    module.register_forward_hook(self._make_forward_hook(name))
                )
                self._backward_handles.append(
                    module.register_full_backward_hook(self._make_backward_hook(name))
                )

    def _make_forward_hook(self, name: str):
        def hook_fn(module, inputs, output):
            inp = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else inputs
            if torch.is_tensor(inp):
                self.inputs[name] = inp.detach().cpu()
            if self.projection_mode != "gradient" and name == "ln_post" and torch.is_tensor(output):
                projection_input = output.detach()
                # ln_post runs before pooling unless final_ln_after_pool is set,
                # so a 3-D output still has to be pooled to become proj's input.
                if self.projection_mode == "pooled" and projection_input.ndim == 3:
                    pooled = getattr(self.model, "_global_pool", None)
                    if pooled is None:
                        raise ValueError("BiCo pooled projection capture requires visual._global_pool.")
                    projection_input, _ = pooled(projection_input)
                self.projection_input = projection_input.cpu()
        return hook_fn

    def _make_backward_hook(self, name: str):
        def hook_fn(module, grad_input, grad_output):
            if grad_output is not None and grad_output[0] is not None:
                self.out_grads[name] = grad_output[0].detach().cpu()
            if grad_input is not None and grad_input[0] is not None:
                self.in_grads[name] = grad_input[0].detach().cpu()
        return hook_fn

    def clear(self) -> None:
        self.projection_input = None
        self.inputs.clear()
        self.in_grads.clear()
        self.out_grads.clear()

    def remove(self) -> None:
        for handle in self._forward_handles:
            handle.remove()
        for handle in self._backward_handles:
            handle.remove()


def _calibration_primary_input(batch: Any, family_adapter: Any = None) -> torch.Tensor:
    """Return a tensor only for calibration shape checks/input gradients.

    Decoder-family batches carry token IDs instead of the image tensor expected
    by Theseus' legacy visual helper.
    """
    if family_adapter is not None:
        inputs = family_adapter.extract_calibration_batch(batch)
        input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        if not torch.is_tensor(input_ids):
            raise TypeError("Family-adapter calibration batches must provide tensor 'input_ids'.")
        return input_ids
    return _t._extract_model_inputs(batch)


def _calibration_row_mask(
    source_batch: Any,
    target_batch: Any,
    family_adapter: Any,
) -> torch.Tensor | None:
    """Non-padding row mask for a text calibration pair, or None for vision.

    Text batches are padded to a fixed length, so a short prompt is mostly pad
    tokens whose activations say nothing about how the two models represent
    content. Theseus drops those rows before accumulating its covariances; BiCo
    fits the same kind of map from the same activations and needs the same
    treatment. Vision batches have no padding, so this returns None and every
    row is kept, leaving the vision path byte-identical.
    """
    if family_adapter is None:
        return None
    masks: list[torch.Tensor | None] = []
    for batch in (source_batch, target_batch):
        inputs = family_adapter.extract_calibration_batch(batch)
        mask = inputs.get("attention_mask") if isinstance(inputs, Mapping) else None
        masks.append(mask if torch.is_tensor(mask) else None)
    return _t._content_row_mask(masks[0], masks[1])


def _collect_batch(
    model: torch.nn.Module,
    recipe,
    batch: Any,
    hook: _BiCoHook,
    *,
    device: torch.device,
    mark_inputs_grad: bool = False,
    family_adapter: Any = None,
) -> None:
    """Run forward + backward on one model/batch and populate hook."""
    model.to(device)
    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(True):
        if mark_inputs_grad:
            inputs = _calibration_primary_input(batch, family_adapter)
            if inputs.is_floating_point():
                inputs.requires_grad_(True)
        loss, _ = recipe(model, batch)
        if loss.dim() > 0:
            loss = loss.sum()
        loss.backward()
    model.zero_grad(set_to_none=True)


def collect_bilinear_statistics(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    source_recipe,
    target_recipe,
    *,
    device: str | torch.device,
    seq_align: str,
    n_batches: int | None,
    seed: int = 0,
    batch_size: int | None = None,
    store_grams: bool = False,
    family_adapter: Any = None,
    projection_mode: str = "gradient",
    source_activation_plan: _t.InterpolatedBlockActivations | None = None,
) -> dict[str, _t.ActivationStore]:
    """
    Collect input activation statistics and output-gradient statistics.

    Source and target are processed sequentially on GPU to minimise peak memory.
    Only one model is on GPU at a time.

    Returns a dict with keys:
      {module_name}.in  -> ActivationStore (input activations)
      {module_name}.out -> ActivationStore (output gradients)
    """
    if family_adapter is not None:
        source_scope = family_adapter.transport_scope(source_model)
        target_scope = family_adapter.transport_scope(target_model)
    else:
        source_scope = None
        target_scope = None

    registry: dict[str, _t.ActivationStore] = {}
    source_hook = _BiCoHook(source_model, scope=source_scope, projection_mode=projection_mode)
    target_hook = _BiCoHook(target_model, scope=target_scope, projection_mode=projection_mode)
    dev = _t._resolve_device(device)

    cpu_device = torch.device("cpu")

    # Move both models off GPU initially
    source_model.to(cpu_device)
    target_model.to(cpu_device)

    try:
        iterator = _t._iter_random_dataset_batches(
            source_dataloader,
            target_dataloader,
            n_batches=n_batches,
            seed=seed,
            batch_size=batch_size,
        )
        if iterator is None:
            iterator = zip(source_dataloader, target_dataloader, strict=True)

        consumed_batches = 0
        for idx, (source_batch, target_batch) in enumerate(iterator):
            if n_batches is not None and idx >= n_batches:
                break

            source_inputs = _calibration_primary_input(source_batch, family_adapter)
            target_inputs = _calibration_primary_input(target_batch, family_adapter)
            if source_inputs.shape[0] != target_inputs.shape[0]:
                raise ValueError(
                    "BiCo calibration expects aligned batch sizes. "
                    f"Got {source_inputs.shape[0]} and {target_inputs.shape[0]}."
                )
            source_labels = source_batch[1] if isinstance(source_batch, (tuple, list)) and len(source_batch) > 1 else None
            target_labels = target_batch[1] if isinstance(target_batch, (tuple, list)) and len(target_batch) > 1 else None
            if torch.is_tensor(source_labels) and torch.is_tensor(target_labels):
                if source_labels.shape != target_labels.shape or not torch.equal(
                    source_labels.detach().cpu(), target_labels.detach().cpu()
                ):
                    raise ValueError("BiCo calibration loaders are not label-aligned.")
            del source_inputs, target_inputs

            row_mask = _calibration_row_mask(source_batch, target_batch, family_adapter)

            # Source: forward + backward on GPU
            source_hook.clear()
            _collect_batch(
                source_model, source_recipe, source_batch, source_hook,
                device=dev, family_adapter=family_adapter,
            )
            source_model.to(cpu_device)
            torch.cuda.empty_cache()

            # Target: forward + backward on GPU
            target_hook.clear()
            _collect_batch(
                target_model, target_recipe, target_batch, target_hook,
                device=dev, family_adapter=family_adapter,
            )
            target_model.to(cpu_device)
            torch.cuda.empty_cache()

            if source_activation_plan is not None:
                # BiCo's bilinear statistics pair input activations with output
                # gradients, so the baseline substitutes both sides of the
                # inserted position, not just the forward activations.
                source_activation_plan.apply(source_hook.inputs)
                source_activation_plan.apply(source_hook.out_grads)

            if source_hook.projection_input is not None and target_hook.projection_input is not None:
                src_rows, tgt_rows = _t._align_features(
                    source_hook.projection_input, target_hook.projection_input, mode=seq_align
                )
                store = registry.setdefault(
                    "__projection__.in",
                    _t.ActivationStore(store_a_gram=store_grams, store_b_gram=store_grams),
                )
                store.update(src_rows, tgt_rows)

            # Align and update registries (all tensors are on CPU from hooks)
            common_inputs = set(source_hook.inputs.keys()) & set(target_hook.inputs.keys())
            for key in common_inputs:
                src_rows, tgt_rows = _t._align_features(
                    source_hook.inputs[key], target_hook.inputs[key], mode=seq_align
                )
                src_rows, tgt_rows = _t._drop_padding_rows(src_rows, tgt_rows, row_mask)
                reg_key = f"{key}.in"
                store = registry.setdefault(
                    reg_key,
                    _t.ActivationStore(store_a_gram=store_grams, store_b_gram=store_grams),
                )
                store.update(src_rows, tgt_rows)

            common_grads = set(source_hook.out_grads.keys()) & set(target_hook.out_grads.keys())
            for key in common_grads:
                src_rows, tgt_rows = _t._align_features(
                    source_hook.out_grads[key], target_hook.out_grads[key], mode=seq_align
                )
                src_rows, tgt_rows = _t._drop_padding_rows(src_rows, tgt_rows, row_mask)
                reg_key = f"{key}.out"
                store = registry.setdefault(
                    reg_key,
                    _t.ActivationStore(store_a_gram=store_grams, store_b_gram=store_grams),
                )
                store.update(src_rows, tgt_rows)

            consumed_batches += 1

        if n_batches is not None and consumed_batches < int(n_batches):
            raise ValueError(
                f"BiCo calibration loaders exhausted after {consumed_batches} batches; requested {int(n_batches)}."
            )

    finally:
        source_hook.remove()
        target_hook.remove()
        source_model.to(cpu_device)
        target_model.to(cpu_device)

    return registry


def collect_gradin_statistics(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    source_recipe,
    target_recipe,
    *,
    device: str | torch.device,
    seq_align: str,
    n_batches: int | None,
    seed: int = 0,
    batch_size: int | None = None,
    store_grams: bool = False,
    family_adapter: Any = None,
    projection_mode: str = "gradient",
    source_activation_plan: _t.InterpolatedBlockActivations | None = None,
) -> dict[str, _t.ActivationStore]:
    """
    Like collect_bilinear_statistics, but fills .in using input-side gradients
    (grad_input[0]) instead of forward activations.

    Falls back to forward activations for modules where grad_input[0] is None
    (e.g. the first layer whose input does not carry gradient).

    Returns a dict with keys:
      {module_name}.in  -> ActivationStore (input gradients, or forward activations)
      {module_name}.out -> ActivationStore (output gradients, dL/dy)
    """
    if family_adapter is not None:
        source_scope = family_adapter.transport_scope(source_model)
        target_scope = family_adapter.transport_scope(target_model)
    else:
        source_scope = None
        target_scope = None

    registry: dict[str, _t.ActivationStore] = {}
    source_hook = _BiCoHook(source_model, scope=source_scope)
    target_hook = _BiCoHook(target_model, scope=target_scope)
    dev = _t._resolve_device(device)

    cpu_device = torch.device("cpu")

    source_model.to(cpu_device)
    target_model.to(cpu_device)

    try:
        iterator = _t._iter_random_dataset_batches(
            source_dataloader,
            target_dataloader,
            n_batches=n_batches,
            seed=seed,
            batch_size=batch_size,
        )
        if iterator is None:
            iterator = zip(source_dataloader, target_dataloader, strict=True)

        for idx, (source_batch, target_batch) in enumerate(iterator):
            if n_batches is not None and idx >= n_batches:
                break

            source_inputs = _calibration_primary_input(source_batch, family_adapter)
            target_inputs = _calibration_primary_input(target_batch, family_adapter)
            if source_inputs.shape[0] != target_inputs.shape[0]:
                raise ValueError(
                    "BiCo gradin calibration expects aligned batch sizes. "
                    f"Got {source_inputs.shape[0]} and {target_inputs.shape[0]}."
                )
            del source_inputs, target_inputs

            row_mask = _calibration_row_mask(source_batch, target_batch, family_adapter)

            # Source: forward + backward on GPU with inputs marked grad
            source_hook.clear()
            _collect_batch(
                source_model, source_recipe, source_batch, source_hook,
                device=dev, mark_inputs_grad=True, family_adapter=family_adapter,
            )
            source_model.to(cpu_device)
            torch.cuda.empty_cache()

            # Target: forward + backward on GPU with inputs marked grad
            target_hook.clear()
            _collect_batch(
                target_model, target_recipe, target_batch, target_hook,
                device=dev, mark_inputs_grad=True, family_adapter=family_adapter,
            )
            target_model.to(cpu_device)
            torch.cuda.empty_cache()

            if source_activation_plan is not None:
                source_activation_plan.apply(source_hook.in_grads)
                source_activation_plan.apply(source_hook.inputs)
                source_activation_plan.apply(source_hook.out_grads)

            # Collect .in from grad_input, fallback to forward inputs
            all_keys = set(source_hook.in_grads.keys())
            all_keys |= set(source_hook.inputs.keys())
            all_keys &= set(target_hook.in_grads.keys()) | set(target_hook.inputs.keys())

            for key in all_keys:
                if key in source_hook.in_grads and key in target_hook.in_grads:
                    src_rows, tgt_rows = _t._align_features(
                        source_hook.in_grads[key], target_hook.in_grads[key], mode=seq_align
                    )
                elif key in source_hook.inputs and key in target_hook.inputs:
                    src_rows, tgt_rows = _t._align_features(
                        source_hook.inputs[key], target_hook.inputs[key], mode=seq_align
                    )
                else:
                    continue
                src_rows, tgt_rows = _t._drop_padding_rows(src_rows, tgt_rows, row_mask)
                reg_key = f"{key}.in"
                store = registry.setdefault(
                    reg_key,
                    _t.ActivationStore(store_a_gram=store_grams, store_b_gram=store_grams),
                )
                store.update(src_rows, tgt_rows)

            # Collect .out from output gradients (same as bico)
            common_grads = set(source_hook.out_grads.keys()) & set(target_hook.out_grads.keys())
            for key in common_grads:
                src_rows, tgt_rows = _t._align_features(
                    source_hook.out_grads[key], target_hook.out_grads[key], mode=seq_align
                )
                src_rows, tgt_rows = _t._drop_padding_rows(src_rows, tgt_rows, row_mask)
                reg_key = f"{key}.out"
                store = registry.setdefault(
                    reg_key,
                    _t.ActivationStore(store_a_gram=store_grams, store_b_gram=store_grams),
                )
                store.update(src_rows, tgt_rows)

    finally:
        source_hook.remove()
        target_hook.remove()
        source_model.to(cpu_device)
        target_model.to(cpu_device)

    return registry


@dataclass(frozen=True)
class BiCoRebase:
    """
    BiCo (Bilinear Coordinate Alignment) rebase method.

    Transport:  delta_B = R_out^T @ delta_A @ R_in
      R_in  = Procrustes map from source→target input activations
      R_out = Procrustes map from source→target output (dL/dy) gradients

    Uses GradFix-style gradient objective for the output-side map.
    """

    name: str = "bico"
    _collect_fn = staticmethod(collect_bilinear_statistics)  # class-level, override in subclasses

    def prepare(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        source_dataloader: Iterable[Any],
        target_dataloader: Iterable[Any],
        source_recipe,
        target_recipe,
        target_base: Mapping[str, torch.Tensor] | None = None,
        delta: Mapping[str, torch.Tensor] | None = None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        verbose: bool = True,
        show_progress: bool = True,
        family_adapter: Any = None,
        source_activation_plan: _t.InterpolatedBlockActivations | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        split_qkv = kwargs.pop("split_qkv", None)
        if split_qkv is not None:
            patch_qkv = bool(split_qkv)
        transform_granularity = str(kwargs.pop("transform_granularity", "param")).strip().lower()
        if transform_granularity not in {"param", "module_type", "block", "global"}:
            raise ValueError("transform_granularity must be one of: param, module_type, block, global")
        if transform_granularity != "param":
            raise ValueError("BiCo transform_granularity support currently requires 'param'.")
        projection_input = str(kwargs.pop("projection_input", "gradient")).strip().lower()
        if projection_input not in {"gradient", "tokens", "pooled"}:
            raise ValueError("BiCo projection_input must be one of: gradient, tokens, pooled.")
        device_transform = str(kwargs.pop("device_transform", "cpu")).strip().lower()
        if device_transform not in {"cpu", "gpu"}:
            raise ValueError("device_transform must be one of: cpu, gpu")
        svd_device = device if device_transform == "gpu" else "cpu"
        del kwargs

        if n_batches is None:
            n_batches = num_batches
        whiten_power = float(whiten_power)
        whiten_eps = float(whiten_eps)
        if not (0.0 <= whiten_power <= 0.5):
            raise ValueError("BiCo whiten_power must be in [0, 0.5].")
        if whiten_eps <= 0.0:
            raise ValueError("BiCo whiten_eps must be > 0.")
        log_prefix = f"[{self.name}]"

        if verbose:
            print(
                f"{log_prefix} prepare: start "
                f"(seq_align={seq_align}, center_acts={bool(center_acts)}, "
                f"whiten_power={whiten_power}, n_batches={n_batches}, seed={int(seed)}, "
                f"transform_granularity={transform_granularity}, device_transform={device_transform}, "
                f"projection_input={projection_input})"
            )

        patched_source = 0
        patched_target = 0
        if patch_qkv:
            if verbose:
                print(f"{log_prefix} prepare: patching fused qkv blocks if needed")
            patched_source = _t._split_fused_qkv_if_needed(source_model)
            patched_target = _t._split_fused_qkv_if_needed(target_model)
            if patched_source > 0 or patched_target > 0:
                logger.info(
                    "%s prepare: split fused qkv attention blocks (source=%d, target=%d)",
                    self.name,
                    patched_source,
                    patched_target,
                )
        elif verbose:
            print(f"{log_prefix} prepare: patch_qkv disabled")

        activation_registry: dict[str, _t.ActivationStore] = {}
        transforms_by_key: dict[str, _t._LayerTransform] = {}
        precompute_diag = _t._PrecomputeDiagnostics(
            slots=0,
            usable=0,
            intentional_zero=0,
            incomplete=0,
            unsupported=0,
            skipped_not_in_target=0,
            examples={},
            assigned_keys=0,
            shared_transform_count=0,
            shared_group_count=0,
        )
        split_fused_qkv = bool(patch_qkv and (patched_source > 0 or patched_target > 0))
        unpatched_source = 0
        unpatched_target = 0

        try:
            if verbose:
                print(f"{log_prefix} prepare: collecting bilinear statistics (input activations + output gradients)")

            activation_registry = self._collect_fn(
                source_model,
                target_model,
                source_dataloader,
                target_dataloader,
                source_recipe,
                target_recipe,
                device=device,
                seq_align=seq_align,
                n_batches=n_batches,
                seed=int(seed),
                batch_size=batch_size,
                store_grams=whiten_power > 0.0,
                projection_mode=projection_input,
                family_adapter=family_adapter,
                source_activation_plan=source_activation_plan,
            )
            if verbose:
                print(f"{log_prefix} prepare: collected activation+gradient entries = {len(activation_registry)}")

            if target_base is not None and delta is not None:
                if verbose:
                    print(f"{log_prefix} prepare: precomputing per-layer transforms")

                if family_adapter is not None:
                    tp_keys = family_adapter.transportable_keys(target_base)
                    visual_key_map = {k: k for k in delta if k in tp_keys}
                    target_visual_base = {k: target_base[k] for k in visual_key_map.values() if k in target_base}
                    visual_delta = {k: delta[k] for k in visual_key_map if k in target_visual_base}
                else:
                    visual_key_map = _t._visual_delta_keys(delta)
                    target_visual_base = _t._visual_state_dict(target_base)
                    visual_delta = {
                        stripped_key: delta[original_key]
                        for stripped_key, original_key in visual_key_map.items()
                        if stripped_key in target_visual_base
                    }

                if split_fused_qkv and family_adapter is None:
                    target_visual_base = _t._split_fused_qkv_state(target_visual_base)
                    visual_delta = _t._split_fused_qkv_state(visual_delta)

                transforms_by_key, precompute_diag = _t._precompute_transforms(
                    target_model=target_model,
                    target_visual_base=target_visual_base,
                    visual_delta=visual_delta,
                    activation_registry=activation_registry,
                    center_acts=bool(center_acts),
                    whiten_power=whiten_power,
                    whiten_eps=whiten_eps,
                    transform_granularity=transform_granularity,
                    show_progress=bool(show_progress),
                    method_name=self.name,
                    svd_device=svd_device,
                    family_adapter=family_adapter,
                    projection_in_key=(
                        "ln_post.out" if projection_input == "gradient" else "__projection__.in"
                    ),
                )
                _t._report_precompute_diagnostics(
                    method_name=self.name,
                    diagnostics=precompute_diag,
                    verbose=bool(verbose),
                )
                if verbose:
                    print(f"{log_prefix} prepare: computed usable transforms = {precompute_diag.usable}")
            elif verbose:
                print(f"{log_prefix} prepare: target_base/delta missing, skipping transform precompute")

        finally:
            if patch_qkv and (patched_source > 0 or patched_target > 0):
                try:
                    unpatched_source = int(merge_openclip_vit_attn(_t._visual_module(source_model)))
                    unpatched_target = int(merge_openclip_vit_attn(_t._visual_module(target_model)))
                    if verbose:
                        print(
                            f"{log_prefix} prepare: recomposed fused qkv blocks "
                            f"(source={unpatched_source}, target={unpatched_target})"
                        )
                except Exception as exc:
                    logger.warning("%s prepare: failed to recompose patched attention blocks: %s", self.name, exc)

        if verbose:
            print(f"{log_prefix} prepare: done")

        return {
            "activation_registry": activation_registry,
            "transforms_by_key": transforms_by_key,
            "precompute_diagnostics": {
                "slots": precompute_diag.slots,
                "usable": precompute_diag.usable,
                "intentional_zero": precompute_diag.intentional_zero,
                "incomplete": precompute_diag.incomplete,
                "unsupported": precompute_diag.unsupported,
                "skipped_not_in_target": precompute_diag.skipped_not_in_target,
                "examples": dict(precompute_diag.examples),
                "assigned_keys": precompute_diag.assigned_keys,
                "shared_transform_count": precompute_diag.shared_transform_count,
                "shared_group_count": precompute_diag.shared_group_count,
            },
            "split_fused_qkv": split_fused_qkv,
            "n_batches": n_batches,
            "patched_source_blocks": patched_source,
            "patched_target_blocks": patched_target,
            "unpatched_source_blocks": unpatched_source,
            "unpatched_target_blocks": unpatched_target,
            "whiten_power": whiten_power,
            "transform_granularity": transform_granularity,
            "device_transform": device_transform,
            "compute_device": _t._resolve_device(device) if device_transform == "gpu" else torch.device("cpu"),
        }

    def apply(
        self,
        prepared: Mapping[str, Any],
        *,
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        verbose: bool = True,
        show_progress: bool = True,
        family_adapter: Any = None,
        zero_attention_delta: bool = False,
        **kwargs,
    ) -> TensorDict:
        del kwargs
        log_prefix = f"[{self.name}]"

        if verbose:
            print(f"{log_prefix} apply: start")

        transforms_by_key = prepared.get("transforms_by_key", None)
        if transforms_by_key is None:
            raise ValueError("BiCo prepared payload is missing 'transforms_by_key'.")

        if family_adapter is not None:
            tp_keys = family_adapter.transportable_keys(target_base)
            visual_key_map = {k: k for k in delta if k in tp_keys}
            target_visual_base_work = {k: target_base[k] for k in visual_key_map.values() if k in target_base}
            visual_delta_work = {k: delta[k] for k in visual_key_map if k in target_visual_base_work}
            split_fused_qkv = False
            out_of_scope_keys = tuple(k for k in delta if k not in tp_keys and k in target_base)
            skipped_not_in_target_keys = tuple(k for k in visual_key_map if k not in target_base)
        else:
            visual_key_map = _t._visual_delta_keys(delta)
            target_visual_base = _t._visual_state_dict(target_base)

            visual_delta = {
                stripped_key: delta[original_key]
                for stripped_key, original_key in visual_key_map.items()
            }

            split_fused_qkv = bool(prepared.get("split_fused_qkv", False))
            if split_fused_qkv:
                target_visual_base_work = _t._split_fused_qkv_state(target_visual_base)
                visual_delta_work = _t._split_fused_qkv_state(
                    {key: value for key, value in visual_delta.items() if key in target_visual_base}
                )
            else:
                target_visual_base_work = target_visual_base
                visual_delta_work = {key: value for key, value in visual_delta.items() if key in target_visual_base}
            has_visual_keys = any(key.startswith(_VISUAL_PREFIX) for key in delta)
            out_of_scope_keys = tuple(
                key for key in delta
                if has_visual_keys and not key.startswith(_VISUAL_PREFIX) and key in target_base
            )
            skipped_not_in_target_keys = tuple(
                original_key
                for stripped_key, original_key in visual_key_map.items()
                if stripped_key not in target_visual_base and original_key not in out_of_scope_keys
            )

        if strict and not visual_delta_work:
            raise ValueError("BiCo did not find any visual delta keys to transport.")

        compute_device = prepared.get("compute_device", "cpu")
        aligned_visual, apply_diag = _t._apply_transforms_to_visual_delta(
            target_visual_base=target_visual_base_work,
            visual_delta=visual_delta_work,
            transforms_by_key=transforms_by_key,
            show_progress=bool(show_progress),
            method_name=self.name,
            device=compute_device,
            strict=bool(strict),
            out_of_scope_keys=out_of_scope_keys,
            skipped_not_in_target_keys=skipped_not_in_target_keys,
        )

        if split_fused_qkv:
            aligned_visual = _t._merge_split_qkv_state(aligned_visual, reference=target_visual_base)

        out: TensorDict = {}
        processed: set[str] = set()

        for stripped_key, original_key in visual_key_map.items():
            if original_key not in target_base:
                continue
            if stripped_key in aligned_visual:
                out[original_key] = aligned_visual[stripped_key].to(
                    dtype=target_base[original_key].dtype,
                    device=target_base[original_key].device,
                )
            else:
                out[original_key] = torch.zeros_like(target_base[original_key], device=target_base[original_key].device)
            processed.add(original_key)

        for key in delta:
            if key in processed or key not in target_base:
                continue
            out[key] = torch.zeros_like(target_base[key], device=target_base[key].device)

        if zero_attention_delta:
            # Ablation: reproduce the transport coverage of the legacy
            # InputAlignedBlock path, where the split-qkv patch landed on the
            # wrapper while forward ran through the unhooked original block, so
            # attention never produced calibration statistics and its delta was
            # written as zeros.
            zeroed = 0
            for key in out:
                if ".attn." in key:
                    out[key] = torch.zeros_like(out[key])
                    zeroed += 1
            if verbose:
                print(f"{log_prefix} apply: zero_attention_delta zeroed {zeroed} attention keys")

        if strict:
            expected_keys = {key for key in visual_key_map.values() if key in target_base}
            missing = sorted(expected_keys - set(out.keys()))
            if missing:
                raise KeyError(f"BiCo did not transport all delta keys. Example: {missing[:10]}")

        if verbose:
            _t._report_apply_diagnostics(method_name=self.name, diagnostics=apply_diag, verbose=True)
            print(f"{log_prefix} apply: done (transported_keys={len(out)})")

        return out

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        source_model: torch.nn.Module | None = None,
        target_model: torch.nn.Module | None = None,
        source_dataloader: Iterable[Any] | None = None,
        target_dataloader: Iterable[Any] | None = None,
        source_recipe=None,
        target_recipe=None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        prepared: Mapping[str, Any] | None = None,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        verbose: bool = True,
        show_progress: bool = True,
        family_adapter: Any = None,
        **kwargs,
    ) -> TensorDict:
        del source_base
        log_prefix = f"[{self.name}]"

        if n_batches is None:
            n_batches = num_batches

        prepared_payload: Mapping[str, Any]
        if prepared is None:
            if source_model is None or target_model is None:
                raise ValueError("BiCo transport requires both source_model and target_model.")
            if source_dataloader is None or target_dataloader is None:
                raise ValueError("BiCo transport requires both source_dataloader and target_dataloader.")
            if source_recipe is None or target_recipe is None:
                raise ValueError("BiCo transport requires both source_recipe and target_recipe.")

            prepared_payload = self.prepare(
                source_model=source_model,
                target_model=target_model,
                source_dataloader=source_dataloader,
                target_dataloader=target_dataloader,
                source_recipe=source_recipe,
                target_recipe=target_recipe,
                target_base=target_base,
                delta=delta,
                device=device,
                seq_align=seq_align,
                center_acts=bool(center_acts),
                whiten_power=float(whiten_power),
                whiten_eps=float(whiten_eps),
                n_batches=n_batches,
                seed=int(seed),
                batch_size=batch_size,
                patch_qkv=patch_qkv,
                verbose=bool(verbose),
                show_progress=bool(show_progress),
                family_adapter=family_adapter,
                **kwargs,
            )
        else:
            prepared_payload = prepared
            if verbose:
                print(f"{log_prefix} transport: using provided prepared payload")

        return self.apply(
            prepared_payload,
            target_base=target_base,
            delta=delta,
            strict=bool(strict),
            verbose=bool(verbose),
            show_progress=bool(show_progress),
            family_adapter=family_adapter,
            zero_attention_delta=bool(kwargs.get("zero_attention_delta", False)),
        )


register(BiCoRebase())


@dataclass(frozen=True)
class BiCoGradInRebase(BiCoRebase):
    """
    Variant of BiCo where T_in is computed from input-side gradients
    (grad_input[0]) instead of forward activations.

    Falls back to forward activations for modules whose grad_input is None
    (e.g. the first layer whose input has no gradient requirement).
    """

    name: str = "bico_gradin"
    _collect_fn = staticmethod(collect_gradin_statistics)


register(BiCoGradInRebase())
