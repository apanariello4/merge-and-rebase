"""Per-family component adapters for BRACE block extension.

An adapter isolates everything that differs between the OpenCLIP vision transformer and
HF decoder families: where the blocks live, which components are corrected and in which
order, how a component's output is captured, how a fitted ``(W, b)`` correction is applied
to a block, and which collapse-schedule policy the family uses.

Every operation is a verbatim lift of the original ``BlockExtender`` /
``DecoderBlockExtender`` method, including the known vision/decoder divergences (see
``CollapseSchedulePolicy`` and ``dampen_block_output``). The extenders delegate to the
adapter, so there is a single copy of each operation; the pre-refactor behaviour is pinned
by the golden hashes and ``tests/test_block_extension_adapters.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.vision_utils import _encode_image
from .schedules import (
    build_extension_layout,
    build_reduction_layout,
    decoder_collapse_schedule,
    decoder_locate_collapse_pos,
    vision_collapse_schedule,
    vision_locate_collapse_pos,
)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    ref_key: str
    kind: Literal["norm_diag", "linear", "fused_slice"]
    slice_index: int | None = None
    residual_aware: bool = False
    blendable_target: bool = False
    # Module name the extender's ``_capture_component_output`` hooks when it differs from ``name``.
    capture_name: str | None = None
    # Position in the residual stream: ``attn_out`` / ``mlp_out`` components can be fitted against a
    # residual-stream target (inputs and already-corrected attention output subtracted).
    stream_role: Literal["attn_out", "mlp_out"] | None = None
    # Which end of a collapsed span the component's reference is read from (``start``: the first block of the
    # span, ``end``: the last one).
    span_anchor: Literal["start", "end"] = "end"


@dataclass(frozen=True)
class AdapterCapabilities:
    residual_target: bool
    target_informed: bool
    correction_scope: bool
    lazy_reference_capture: bool
    zero_projection_baselines: bool
    collapse_disjoint_spans: bool


@dataclass(frozen=True)
class CollapseSchedulePolicy:
    """Family-specific collapse schedule and anchor lookup.

    Vision: ``spread_mod`` is a linspace of anchors honouring ``insertion_order`` and
    ``locate`` is range containment clamped to ``len(chain) - 2`` (B11/B12 reference side).
    Decoder: ``spread_mod`` is ``i % (D - 1)`` ignoring ``insertion_order`` and ``locate`` is
    plain membership without the clamp (so ``clump`` + ``top-bottom`` can raise IndexError
    downstream). These differences are preserved, never unified.
    """

    build: Callable[[int, int, str, str], list[int]]
    locate: Callable[[list[dict[str, Any]], int], int]


class ComponentAdapter(Protocol):
    family: str
    components: Sequence[ComponentSpec]
    capabilities: AdapterCapabilities

    def layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_layers(self, model: nn.Module, new_layers: list[nn.Module]) -> None: ...

    def inner_block(self, block: nn.Module) -> nn.Module: ...

    def capture_component_output(
        self,
        model: nn.Module,
        block_idx: int,
        component: str,
        loader: Iterable[Any],
        n_batches: int,
        device: str | torch.device,
    ) -> torch.Tensor: ...

    def capture_component(
        self,
        model: nn.Module,
        block_idx: int,
        spec: ComponentSpec,
        loader: Iterable[Any],
        n_batches: int,
        device: str | torch.device,
    ) -> torch.Tensor: ...

    def capture_block_input(
        self,
        model: nn.Module,
        target: int | str,
        loader: Iterable[Any],
        n_batches: int,
        device: str | torch.device,
    ) -> torch.Tensor: ...

    def apply_correction(self, block: nn.Module, spec: ComponentSpec, W: torch.Tensor, b: torch.Tensor) -> None: ...

    def interpolate_block_weights(self, target_block: nn.Module, source_block: nn.Module, alpha: float) -> None: ...

    def dampen_block_output(self, block: nn.Module, factor: float) -> None: ...

    def zero_output_projections(self, block: nn.Module) -> None: ...

    def collapse_policy(self) -> CollapseSchedulePolicy: ...

    def build_layout(self, chain: Sequence[Mapping[str, Any]], *, reduction: bool = False) -> dict[str, Any]: ...


def _run_decoder_forward(model: nn.Module, batch: Mapping[str, Any], device: str | torch.device) -> None:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)


def _get_layers(model: nn.Module, family_adapter: Any) -> nn.ModuleList:
    return family_adapter.transport_scope(model).layers


def _get_final_norm(model: nn.Module, family_adapter: Any) -> nn.Module:
    return family_adapter.transport_scope(model).norm


class _InProjCapture:
    """Patch ``nn.MultiheadAttention.forward`` to record the fused q/k/v projections on CPU."""

    def __init__(self, attn: nn.Module):
        self.attn = attn
        self.outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._orig_forward = attn.forward
        attn.forward = self._patched_forward

    def _patched_forward(self, query, key=None, value=None, **kwargs):
        qkv = F.linear(query, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        self.outputs.append((q.detach().cpu(), k.detach().cpu(), v.detach().cpu()))
        return self._orig_forward(query, key=key, value=value, **kwargs)

    def restore(self):
        self.attn.forward = self._orig_forward


VISION_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("ln_1", "ln_1_output", "norm_diag", span_anchor="start"),
    ComponentSpec("q", "q_output", "fused_slice", slice_index=0, span_anchor="start"),
    ComponentSpec("k", "k_output", "fused_slice", slice_index=1, span_anchor="start"),
    ComponentSpec("v", "v_output", "fused_slice", slice_index=2, span_anchor="start"),
    ComponentSpec(
        "out_proj", "attn_output", "linear", residual_aware=True, capture_name="attn", stream_role="attn_out"
    ),
    ComponentSpec("ln_2", "ln_2_output", "norm_diag"),
    ComponentSpec("c_fc", "c_fc_output", "linear"),
    ComponentSpec(
        "c_proj", "c_proj_output", "linear", residual_aware=True, blendable_target=True, stream_role="mlp_out"
    ),
)

DECODER_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("input_layernorm", "input_layernorm_output", "norm_diag", span_anchor="start"),
    ComponentSpec("q_proj", "q_proj_output", "linear", span_anchor="start"),
    ComponentSpec("k_proj", "k_proj_output", "linear", span_anchor="start"),
    ComponentSpec("v_proj", "v_proj_output", "linear", span_anchor="start"),
    ComponentSpec("o_proj", "attn_output", "linear", stream_role="attn_out"),
    ComponentSpec("post_attention_layernorm", "post_attn_ln_output", "norm_diag"),
    ComponentSpec("gate_proj", "gate_proj_output", "linear"),
    ComponentSpec("up_proj", "up_proj_output", "linear"),
    ComponentSpec("down_proj", "down_proj_output", "linear", stream_role="mlp_out"),
)


class VisionAdapter:
    family = "openclip_vision"
    components = VISION_COMPONENTS
    capabilities = AdapterCapabilities(
        residual_target=True,
        target_informed=True,
        correction_scope=True,
        lazy_reference_capture=True,
        zero_projection_baselines=True,
        collapse_disjoint_spans=True,
    )

    def layers(self, model: nn.Module) -> nn.ModuleList:
        return model.visual.transformer.resblocks

    def set_layers(self, model: nn.Module, new_layers: list[nn.Module]) -> None:
        model.visual.transformer.resblocks = nn.ModuleList(new_layers)

    def inner_block(self, block: nn.Module) -> nn.Module:
        return block.block if hasattr(block, "block") else block

    @torch.no_grad()
    def capture_block_input(self, model, target, loader, n_batches, device):
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
            _encode_image(model, images.to(device))

        handle.remove()

        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    def capture_component(self, model, block_idx, spec, loader, n_batches, device):
        # ``out_proj`` is corrected against the attention module's output (ref key ``attn_output``).
        component = "attn" if spec.name == "out_proj" else spec.name
        return self.capture_component_output(model, block_idx, component, loader, n_batches, device)

    @torch.no_grad()
    def capture_component_output(self, model, block_idx, component, loader, n_batches, device):
        model.eval()
        buffers: list[torch.Tensor] = []
        block = model.visual.transformer.resblocks[block_idx]
        inner = self.inner_block(block)

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
                _encode_image(model, images.to(device))
            cap.restore()
            slice_idx = {"q": 0, "k": 1, "v": 2}[component]
            for tensors in cap.outputs:
                buffers.append(tensors[slice_idx])
            if not buffers:
                return torch.empty(0)
            return torch.cat(buffers, dim=0).flatten(0, 1)
        else:
            raise ValueError(
                f"Unsupported component '{component}'. Expected ln_1, attn, ln_2, c_fc, c_proj, q, k, or v."
            )

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
            _encode_image(model, images.to(device))

        handle.remove()

        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    @torch.no_grad()
    def apply_correction(self, block, spec, W, b):
        inner = self.inner_block(block)
        name = spec.name
        if spec.kind == "norm_diag":
            ln = inner.ln_1 if name == "ln_1" else inner.ln_2
            d = torch.diag(W).to(ln.weight.device, dtype=ln.weight.dtype)
            b = b.to(ln.bias.device, dtype=ln.bias.dtype)
            ln.weight.mul_(d)
            ln.bias.copy_(d * ln.bias + b)
        elif spec.kind == "fused_slice":
            dim_qkv = inner.attn.in_proj_weight.shape[0] // 3
            lo = spec.slice_index * dim_qkv
            hi = (spec.slice_index + 1) * dim_qkv
            W = W.to(inner.attn.in_proj_weight.device, dtype=inner.attn.in_proj_weight.dtype)
            b = b.to(inner.attn.in_proj_bias.device, dtype=inner.attn.in_proj_bias.dtype)
            w_slice = inner.attn.in_proj_weight.data[lo:hi].clone()
            b_slice = inner.attn.in_proj_bias.data[lo:hi].clone()
            inner.attn.in_proj_weight.data[lo:hi] = W @ w_slice
            inner.attn.in_proj_bias.data[lo:hi] = W @ b_slice + b
        else:
            linear = {"out_proj": inner.attn.out_proj, "c_fc": inner.mlp.c_fc, "c_proj": inner.mlp.c_proj}[name]
            W = W.to(linear.weight.device, dtype=linear.weight.dtype)
            b = b.to(linear.bias.device, dtype=linear.bias.dtype)
            linear.weight.copy_(W @ linear.weight)
            linear.bias.copy_(W @ linear.bias + b)

    @torch.no_grad()
    def interpolate_block_weights(self, target_block, source_block, alpha=0.5):
        target_inner = self.inner_block(target_block)
        source_inner = self.inner_block(source_block)
        source_params = dict(source_inner.named_parameters())
        for name_t, p_t in target_inner.named_parameters():
            if name_t.startswith("aligner."):
                continue
            p_s = source_params.get(name_t)
            if p_s is not None and p_t.shape == p_s.shape:
                p_t.copy_((1.0 - alpha) * p_t + alpha * p_s)

    @torch.no_grad()
    def dampen_block_output(self, block, factor):
        # Weights only: the vision extender does not scale the output-projection biases.
        inner = self.inner_block(block)
        if hasattr(inner, "attn") and hasattr(inner.attn, "out_proj"):
            inner.attn.out_proj.weight.mul_(factor)
        if hasattr(inner, "mlp") and hasattr(inner.mlp, "c_proj"):
            inner.mlp.c_proj.weight.mul_(factor)

    @torch.no_grad()
    def zero_output_projections(self, block):
        inner = self.inner_block(block)
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

    def collapse_policy(self) -> CollapseSchedulePolicy:
        return CollapseSchedulePolicy(build=vision_collapse_schedule, locate=vision_locate_collapse_pos)

    def build_layout(self, chain, *, reduction=False):
        return build_reduction_layout(chain) if reduction else build_extension_layout(chain)


class DecoderAdapter:
    family = "hf_decoder"
    components = DECODER_COMPONENTS
    capabilities = AdapterCapabilities(
        residual_target=False,
        target_informed=False,
        correction_scope=False,
        lazy_reference_capture=False,
        zero_projection_baselines=False,
        collapse_disjoint_spans=False,
    )

    def __init__(self, family_adapter: Any):
        self.family_adapter = family_adapter

    def layers(self, model: nn.Module) -> nn.ModuleList:
        return _get_layers(model, self.family_adapter)

    def final_norm(self, model: nn.Module) -> nn.Module:
        return _get_final_norm(model, self.family_adapter)

    def set_layers(self, model: nn.Module, new_layers: list[nn.Module]) -> None:
        # Depth/config/KV-cache reindexing stays with ``DecoderBlockExtender`` (_set_depth, _reindex_layers).
        scope = self.family_adapter.transport_scope(model)
        scope.layers = nn.ModuleList(new_layers)

    def inner_block(self, block: nn.Module) -> nn.Module:
        return block

    def _forward(self, model: nn.Module, batch: Any, device: str | torch.device) -> None:
        _run_decoder_forward(model, self.family_adapter.extract_calibration_batch(batch), device)

    @torch.no_grad()
    def capture_block_input(self, model, target, loader, n_batches, device):
        model.eval()
        buffers: list[torch.Tensor] = []

        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any):
            if inputs and inputs[0] is not None:
                buffers.append(inputs[0].detach().cpu())

        if target == "final":
            handle = self.final_norm(model).register_forward_hook(hook)
        else:
            handle = self.layers(model)[target].register_forward_hook(hook)

        it = iter(loader)
        for _ in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            self._forward(model, batch, device)

        handle.remove()
        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    def capture_component(self, model, block_idx, spec, loader, n_batches, device):
        return self.capture_component_output(model, block_idx, spec.name, loader, n_batches, device)

    @torch.no_grad()
    def capture_component_output(self, model, block_idx, component, loader, n_batches, device):
        model.eval()
        buffers: list[torch.Tensor] = []
        block = self.layers(model)[block_idx]

        target_map = {
            "input_layernorm": block.input_layernorm,
            "attn": block.self_attn,
            "post_attention_layernorm": block.post_attention_layernorm,
            "gate_proj": block.mlp.gate_proj,
            "up_proj": block.mlp.up_proj,
            "down_proj": block.mlp.down_proj,
        }

        if component in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj = getattr(block.self_attn, component)
        elif component in target_map:
            proj = target_map[component]
        else:
            raise ValueError(
                f"Unsupported component '{component}'. Expected one of: {', '.join(s.name for s in self.components)}"
            )

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            out = output[0] if isinstance(output, tuple) else output
            if out is not None:
                buffers.append(out.detach().cpu())

        handle = proj.register_forward_hook(hook)
        it = iter(loader)
        for _ in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            self._forward(model, batch, device)

        handle.remove()
        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    @torch.no_grad()
    def apply_correction(self, block, spec, W, b):
        name = spec.name
        if spec.kind == "norm_diag":
            ln = getattr(block, name)
            d = torch.diag(W).to(ln.weight.device, dtype=ln.weight.dtype)
            ln.weight.mul_(d)
            if hasattr(ln, "bias") and ln.bias is not None:
                b = b.to(ln.bias.device, dtype=ln.bias.dtype)
                ln.bias.copy_(d * ln.bias + b)
        else:
            if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                linear = getattr(block.self_attn, name)
            else:
                linear = getattr(block.mlp, name)
            W = W.to(linear.weight.device, dtype=linear.weight.dtype)
            linear.weight.copy_(W @ linear.weight)
            if linear.bias is not None:
                b = b.to(linear.bias.device, dtype=linear.bias.dtype)
                linear.bias.copy_(W @ linear.bias + b)

    @torch.no_grad()
    def interpolate_block_weights(self, target_block, source_block, alpha=0.5):
        source_params = dict(source_block.named_parameters())
        for name_t, p_t in target_block.named_parameters():
            if name_t.startswith("aligner."):
                continue
            p_s = source_params.get(name_t)
            if p_s is not None and p_t.shape == p_s.shape:
                p_t.copy_((1.0 - alpha) * p_t + alpha * p_s)

    @torch.no_grad()
    def dampen_block_output(self, block, factor):
        # Weights AND biases: unlike the vision extender, the decoder also scales the biases.
        if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
            block.self_attn.o_proj.weight.mul_(factor)
            if hasattr(block.self_attn.o_proj, "bias") and block.self_attn.o_proj.bias is not None:
                block.self_attn.o_proj.bias.mul_(factor)
        if hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"):
            block.mlp.down_proj.weight.mul_(factor)
            if hasattr(block.mlp.down_proj, "bias") and block.mlp.down_proj.bias is not None:
                block.mlp.down_proj.bias.mul_(factor)

    def zero_output_projections(self, block):
        raise NotImplementedError("The decoder extender has no residual-identity (zero-projection) baseline.")

    def collapse_policy(self) -> CollapseSchedulePolicy:
        return CollapseSchedulePolicy(build=decoder_collapse_schedule, locate=decoder_locate_collapse_pos)

    def build_layout(self, chain, *, reduction=False):
        return build_reduction_layout(chain) if reduction else build_extension_layout(chain)
