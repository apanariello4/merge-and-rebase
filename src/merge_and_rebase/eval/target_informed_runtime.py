"""Model-side orchestration for target-informed BRACE proposal 1 (target residual
completion): capturing native reference banks, fitting the residual correction
that completes each inserted block's projection, and caching the resulting
task-vector deltas.

Native reference banks are captured before resizing. Transport remains fitted
on the resized base; completion only changes the task vector. Cache files are
tensor-only dictionaries so evaluation never has to refit a completed transport.

Proposal 2 (target-informed shared correction) is NOT implemented here. Its
authoritative, tested implementation lives in `block_extension.py`
(`TargetSharedCorrection`, `_capture_target_component_references`,
`_blend_target_reference`), keyed off the nested
`block_extension_params.target_shared_correction` config group. The
`TargetSharedConfig`/`parse_target_shared_config` pair in this module is a
separate, older top-level-config validator used by `validate_target_protocol`
and `cache_identity`; it does not drive any execution path in this module.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset

from ..rebase.methods.theseus import _interp_2d_tokens, _to_tokens
from .block_extension import _encode_image
from .target_residual_completion import (
    JointCorrectionConfig,
    ResidualCompletionConfig,
    ResidualSufficientStatistics,
    centered_rectangular_procrustes,
    fit_joint_cproj_correction,
    order_components,
)


# Config-validation scaffolding only: block_extension.py holds the authoritative
# proposal-2 (target-informed shared correction) implementation and its own config
# parsing (`_as_target_shared_correction`, keyed off
# `block_extension_params.target_shared_correction`). This dataclass/parser pair
# validates the separate top-level `target_shared_correction` config key that
# `validate_target_protocol` and `cache_identity` still consult; it has no
# execution path of its own in this module.
@dataclass(frozen=True)
class TargetSharedConfig:
    enabled: bool = False
    added_blocks: str = "all"
    component: str = "c_proj"
    target_weight: float = 0.0
    num_batches: int = 5


def parse_target_shared_config(raw: Mapping[str, Any] | None) -> TargetSharedConfig:
    if raw is None:
        return TargetSharedConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("target_shared_correction must be a mapping")
    unknown = set(raw) - set(TargetSharedConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown target_shared_correction fields: {sorted(unknown)}")
    cfg = TargetSharedConfig(**raw)
    if not isinstance(cfg.enabled, bool):
        raise ValueError("target_shared_correction.enabled must be boolean")
    if cfg.added_blocks != "all" or cfg.component != "c_proj":
        raise ValueError("Target shared correction supports all added c_proj components")
    if isinstance(cfg.num_batches, bool) or not isinstance(cfg.num_batches, int) or cfg.num_batches < 1:
        raise ValueError("target_shared_correction.num_batches must be a positive integer")
    if isinstance(cfg.target_weight, bool) or not isinstance(cfg.target_weight, (int, float)):
        raise ValueError("target_weight must be a finite nonnegative number")
    if not math.isfinite(cfg.target_weight) or cfg.target_weight < 0:
        raise ValueError("target_weight must be a finite nonnegative number")
    return cfg


def validate_target_protocol(cfg, block_cfg, source_depth, target_depth, method_name, merge_mode):
    """Reject unsupported experiments instead of silently dropping their options."""
    from .target_residual_completion import parse_residual_completion_config
    shared = parse_target_shared_config(cfg.get("target_shared_correction"))
    residual = parse_residual_completion_config(cfg.get("target_residual_completion"))
    joint = bool(block_cfg.joint_blockwise_correction.enabled)
    direct_p1 = bool(block_cfg.direct_p1_correction.enabled)
    active = shared.enabled or residual.enabled or joint or direct_p1 or bool(cfg.get("capture_target_residual_reference"))
    if not active:
        return
    if method_name not in {"theseus", "bico"}:
        raise ValueError("Target-informed BRACE supports Theseus and BICO only")
    if source_depth < 1 or target_depth != 2 * source_depth:
        raise ValueError("Target-informed BRACE currently requires doubling the source depth")
    if merge_mode not in {"none", "rebase_then_merge", "brace_transport_then_merge"}:
        raise ValueError("Target-informed BRACE requires per-task transport before merging")
    if not cfg.get("block_extension_enabled", True):
        raise ValueError("Target-informed BRACE requires block extension")
    if (block_cfg.insertion_order != "bottom-top" or block_cfg.extension_density not in {"spread", "spread_mod"}
            or block_cfg.extension_strategy != "duplicate_per_weight"):
        raise ValueError("Target-informed BRACE requires bottom-top spread duplicate insertion")
    if block_cfg.blocks_to_add not in (None, source_depth):
        raise ValueError("blocks_to_add must match the depth-doubling protocol")
    identity_p1 = (
        residual.enabled
        and block_cfg.skip_correction
        and block_cfg.inserted_block_mode == "residual_identity"
    )
    if block_cfg.lmc_mode != "shared":
        raise ValueError("Target-informed methods require lmc_mode='shared'")
    if not identity_p1 and (block_cfg.skip_correction or block_cfg.inserted_block_mode != "ariadne"):
        raise ValueError(
            "Target-informed methods extend the ordinary Shared BRACE arm, except for the explicit "
            "residual_identity + Proposal-1 ablation"
        )
    if identity_p1 and shared.enabled:
        raise ValueError("residual_identity + Proposal 1 cannot also enable target_shared_correction")
    if joint and shared.enabled:
        raise ValueError("Option 3 and target_shared_correction cannot be enabled together")
    if direct_p1 and shared.enabled:
        raise ValueError("Direct P1 correction and target_shared_correction cannot be enabled together")
    if block_cfg.calibration_split != "val" or block_cfg.calibration_dataset is not None or block_cfg.calibration_task:
        raise ValueError("Target-informed references require task-local validation calibration")
    if block_cfg.transport_activation_mode != "model" or block_cfg.share_ft_refs:
        raise ValueError("Target-informed BRACE requires base references and model transport activations")
    if block_cfg.insertion_target_mode != "direct":
        raise ValueError("Target-informed correction uses the published direct component targets")
    if shared.enabled and shared.target_weight > 0 and shared.num_batches != block_cfg.n_batches_act:
        raise ValueError("Target shared references must use the BRACE calibration batch count")
    if str((cfg.get("block_extension_params") or {}).get("calibration_protocol", "task_local")).startswith("vision8_mix"):
        raise ValueError("Mixed-dataset target references are outside this task-local experiment")
    if cfg.get("native_target_tasks") or cfg.get("base_construction", "per_task") != "per_task":
        raise ValueError("Target-informed experiments require all-source tasks on the native target base")
    if cfg.get("patched_attn") or cfg.get("attn_patch_cfg"):
        raise ValueError("Target-informed experiments require the ordinary unpatched ViT checkpoints")
    if cfg.get("load_transported_tvs_dir") and (cfg.get("source_lmc_eval") or cfg.get("cross_task_lmc_pairs") or cfg.get("all_task_lmc_eval")):
        raise ValueError("Cached target vectors cannot supply source endpoint diagnostics")


def _dataset_identity(dataset):
    if isinstance(dataset, Subset):
        return (_dataset_identity(dataset.dataset), tuple(int(i) for i in dataset.indices))
    split = getattr(dataset, "split", None)
    fingerprint = getattr(split, "_fingerprint", None)
    if fingerprint is not None:
        return (str(fingerprint), len(dataset))
    ids = getattr(dataset, "sample_ids", None)
    if ids is not None:
        return tuple(str(i) for i in ids)
    return ("object", id(dataset), len(dataset))


def paired_calibration(source_loader, target_loader, *, num_batches, seed=None):
    """Replay identical dataset indices under the two model preprocessors."""
    if not isinstance(source_loader, DataLoader) or not isinstance(target_loader, DataLoader):
        raise ValueError("Paired calibration requires indexed DataLoaders")
    if _dataset_identity(source_loader.dataset) != _dataset_identity(target_loader.dataset):
        raise ValueError("Source/target calibration image identities or ordering do not match")
    if source_loader.batch_size != target_loader.batch_size or not source_loader.batch_size:
        raise ValueError("Paired calibration requires equal positive batch sizes")
    if seed is None and not isinstance(source_loader.sampler, SequentialSampler):
        raise ValueError("BRACE target references require a sequential calibration loader")
    n = len(source_loader.dataset)
    if not n:
        raise ValueError("Calibration set is empty")
    order = list(range(n)) if seed is None else torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    order = order[:num_batches * source_loader.batch_size]
    kwargs = dict(batch_size=source_loader.batch_size, shuffle=False, num_workers=0, drop_last=False)
    source = DataLoader(Subset(source_loader.dataset, order), collate_fn=source_loader.collate_fn, **kwargs)
    target = DataLoader(Subset(target_loader.dataset, order), collate_fn=target_loader.collate_fn, **kwargs)
    source_batches, target_batches = [], []
    for a, b in zip(source, target, strict=True):
        # Vision batches are (images, labels) and the labels are model-agnostic,
        # so they must match example for example. Text batches are mappings
        # whose contents are tokenizer-specific by construction -- the same
        # prompt yields different ids under the source and target tokenizers --
        # so there is nothing comparable to assert here. The examples are
        # already pinned: both Subsets index the same `order` into datasets
        # whose _dataset_identity had to agree above.
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) > 1 and len(b) > 1:
            if not torch.equal(torch.as_tensor(a[1]), torch.as_tensor(b[1])):
                raise ValueError("Calibration labels disagree for supposedly identical images")
        source_batches.append(a)
        target_batches.append(b)
    metadata = {
        "indices": order, "dataset_identity": repr(_dataset_identity(source_loader.dataset)),
        "requested_batches": num_batches, "actual_batches": len(source_batches),
        "batch_size": source_loader.batch_size, "sampling_seed": seed, "split": "val",
    }
    return source_batches, target_batches, metadata


# --- architecture abstraction -------------------------------------------------
#
# Proposal 1's solver is architecture-agnostic; only this orchestration layer
# reached into CLIP's module tree. These two shims name the four things that
# actually differ between a ViT and an HF decoder -- where the blocks live,
# which projection writes the residual, how to run a batch, and what that
# projection is called in the state dict -- so the same completion runs on both
# without duplicating the orchestration. Vision behaviour is unchanged: passing
# family_adapter=None selects the original CLIP paths verbatim.


class _VisionLayout:
    """CLIP ViT: the original, unchanged code paths."""

    name = "vision"

    def blocks(self, model):
        return model.visual.transformer.resblocks

    def block_count(self, model):
        return len(model.visual.transformer.resblocks)

    def proj_module(self, block):
        inner = block.block if hasattr(block, "block") else block
        return inner.mlp.c_proj

    def attn_module(self, block):
        inner = block.block if hasattr(block, "block") else block
        return inner.attn

    def attn_proj_module(self, block):
        inner = block.block if hasattr(block, "block") else block
        return inner.attn.out_proj

    def block_module(self, block):
        return block.block if hasattr(block, "block") else block

    def forward(self, model, batch, device):
        _encode_image(model, batch[0].to(device))

    def batch_size(self, batch):
        return len(batch[0])

    def token_mask(self, batch):
        """Image batches have no padding: every token is a real patch."""
        return None

    def proj_key(self, pos, *, prefixed):
        key = f"transformer.resblocks.{pos}.mlp.c_proj.weight"
        return f"visual.{key}" if prefixed else key

    def attn_proj_key(self, pos, *, prefixed):
        key = f"transformer.resblocks.{pos}.attn.out_proj.weight"
        return f"visual.{key}" if prefixed else key

    def component_key(self, pos, component, *, prefixed):
        if component == "mlp.c_proj":
            return self.proj_key(pos, prefixed=prefixed)
        if component == "attn.out_proj":
            return self.attn_proj_key(pos, prefixed=prefixed)
        raise ValueError(f"Unsupported completion component {component!r}")

    def component_scale_module(self, block, component):
        """The LayerScale on this component's write into the residual stream.

        ``ls_1`` scales the attention path and ``ls_2`` the MLP path; either may
        be ``nn.Identity``. Both must enter the fit's output map, not be applied
        afterwards.
        """
        inner = self.block_module(block)
        name = "ls_2" if component == "mlp.c_proj" else "ls_1"
        return getattr(inner, name, nn.Identity())


class _DecoderLayout:
    """HF decoder: mlp.down_proj is the residual-writing projection, the
    analogue of CLIP's mlp.c_proj (both are the block's final output map)."""

    name = "decoder"

    def __init__(self, family_adapter):
        self.family_adapter = family_adapter

    def blocks(self, model):
        return self.family_adapter.transport_scope(model).layers

    def block_count(self, model):
        return self.family_adapter.block_count(model)

    def proj_module(self, block):
        return block.mlp.down_proj

    def attn_module(self, block):
        return block.self_attn

    def attn_proj_module(self, block):
        """``self_attn.o_proj`` is the decoder analogue of CLIP's attn.out_proj.

        Implemented for symmetry with the vision path and **untested on this
        iteration**: no decoder campaign has run the two-component completion.
        """
        return block.self_attn.o_proj

    def block_module(self, block):
        return block

    def forward(self, model, batch, device):
        inputs = self.family_adapter.extract_calibration_batch(batch)
        model(**{k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")})

    def batch_size(self, batch):
        inputs = self.family_adapter.extract_calibration_batch(batch)
        return int(inputs["input_ids"].shape[0])

    def token_mask(self, batch):
        """Boolean ``[B, T]`` of real (non-pad) positions, or None if unmasked."""
        mask = self.family_adapter.extract_calibration_batch(batch).get("attention_mask")
        return None if mask is None else mask.bool()

    def proj_key(self, pos, *, prefixed):
        # The decoder state dict is already "model.layers.N..."; there is no
        # second prefix the way vision has "visual.".
        return f"model.layers.{pos}.mlp.down_proj.weight"

    def attn_proj_key(self, pos, *, prefixed):
        return f"model.layers.{pos}.self_attn.o_proj.weight"

    def component_key(self, pos, component, *, prefixed):
        if component == "mlp.c_proj":
            return self.proj_key(pos, prefixed=prefixed)
        if component == "attn.out_proj":
            return self.attn_proj_key(pos, prefixed=prefixed)
        raise ValueError(f"Unsupported completion component {component!r}")

    def component_scale_module(self, block, component):
        """Decoder blocks carry no LayerScale; the write is unscaled."""
        return nn.Identity()


def _layout_for(family_adapter):
    return _VisionLayout() if family_adapter is None else _DecoderLayout(family_adapter)


#: Capture kinds understood by ``capture_tokens``. The ``*_input`` kinds take the
#: hooked module's input, the others its output.
_CAPTURE_KINDS = frozenset({"boundary", "c_proj", "c_proj_input", "attn_proj", "attn_proj_input"})
_INPUT_CAPTURE_KINDS = frozenset({"c_proj_input", "attn_proj_input"})
_ATTN_CAPTURE_KINDS = frozenset({"attn_proj", "attn_proj_input"})

#: Which capture kind supplies each component's regression features.
COMPONENT_INPUT_KIND = {"attn.out_proj": "attn_proj_input", "mlp.c_proj": "c_proj_input"}


def _mha_query_key_value(args, kwargs):
    values = list(args[:3])
    for position, name in enumerate(("query", "key", "value")):
        if position >= len(values):
            values.append(kwargs.get(name))
    query, key, value = values[0], values[1], values[2]
    if query is None:
        raise ValueError("Could not recover the attention query from the captured call")
    return query, key if key is not None else query, value if value is not None else query


def _stock_mha_out_proj_input(module, args, kwargs):
    """Rows that ``out_proj`` consumes inside a stock ``nn.MultiheadAttention``.

    torch applies the output projection *functionally* inside
    ``F.multi_head_attention_forward``, straight from ``out_proj.weight``, so the
    ``out_proj`` submodule is never called and a forward hook on it never fires.
    Rerunning the same attention with an identity output projection recovers
    exactly the rows it would have consumed; the caller verifies that by pushing
    them back through the real projection and comparing against the module's own
    output, so this can never silently capture the wrong tensor.
    """
    query, key, value = _mha_query_key_value(args, kwargs)
    batch_first = bool(getattr(module, "batch_first", False))
    if batch_first:
        query, key, value = (tensor.transpose(1, 0) for tensor in (query, key, value))
    identity = torch.eye(module.embed_dim, dtype=query.dtype, device=query.device)
    shared = {
        "training": module.training,
        "key_padding_mask": kwargs.get("key_padding_mask"),
        "need_weights": False,
        "attn_mask": kwargs.get("attn_mask"),
    }
    if module._qkv_same_embed_dim:
        rows, _ = F.multi_head_attention_forward(
            query, key, value, module.embed_dim, module.num_heads,
            module.in_proj_weight, module.in_proj_bias,
            module.bias_k, module.bias_v, module.add_zero_attn, module.dropout,
            identity, None, **shared,
        )
    else:
        rows, _ = F.multi_head_attention_forward(
            query, key, value, module.embed_dim, module.num_heads,
            None, module.in_proj_bias,
            module.bias_k, module.bias_v, module.add_zero_attn, module.dropout,
            identity, None,
            use_separate_proj_weight=True,
            q_proj_weight=module.q_proj_weight, k_proj_weight=module.k_proj_weight,
            v_proj_weight=module.v_proj_weight, **shared,
        )
    return rows.transpose(1, 0) if batch_first else rows


def _verify_recomputed_attention_input(module, rows, module_output):
    """Push the recomputed rows back through the real projection and compare.

    Cheap (one matmul against a full attention) and it turns any future change in
    torch's attention internals into a loud failure instead of a wrong fit.
    """
    reference = module_output[0] if isinstance(module_output, tuple) else module_output
    replayed = F.linear(rows, module.out_proj.weight, module.out_proj.bias)
    if replayed.shape != reference.shape or not torch.allclose(replayed, reference, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            "Recovered attention rows do not reproduce the attention output through "
            "out_proj; the captured features are not the ones the projection consumes"
        )


@torch.no_grad()
def capture_tokens(model, batches, requests: Mapping[str, tuple[int, str]], device, *, family_adapter=None, mask_padding=False):
    """Capture B,T,D tensors, releasing hooks and restoring placement on errors.

    family_adapter=None keeps the original CLIP paths; passing one selects the
    HF-decoder equivalents (see _DecoderLayout). The "c_proj" capture kinds keep
    their names on both paths -- on a decoder they resolve to mlp.down_proj,
    which plays the same residual-writing role.

    mask_padding=True drops the batch's pad positions and stores each batch as
    ``[1, N_real, D]``. Every consumer flattens rows with ``reshape(-1, D)``, so
    the packed shape is transparent to them; pairing across two models then
    requires both to see the same real-token count, which the caller checks.
    """
    layout = _layout_for(family_adapter)
    blocks = layout.blocks(model)
    training = model.training
    original_device = next(model.parameters()).device
    output = {key: [] for key in requests}
    current_batch = [0]
    current_mask = [None]
    handles = []
    try:
        model.to(device).eval()

        def store(name, tensor):
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            tokens = _to_tokens(tensor.detach(), batch_size=current_batch[0])
            mask = current_mask[0]
            if mask is not None:
                if tuple(mask.shape) != tuple(tokens.shape[:2]):
                    raise RuntimeError(
                        f"Padding mask {tuple(mask.shape)} does not match captured tokens "
                        f"{tuple(tokens.shape[:2])}"
                    )
                tokens = tokens[mask.to(tokens.device)].unsqueeze(0)
            output[name].append(tokens.float().cpu().clone())

        for key, (index, kind) in requests.items():
            if kind not in _CAPTURE_KINDS:
                raise ValueError(f"Unknown capture kind {kind}")
            block = layout.block_module(blocks[index])
            recompute = False
            if kind == "boundary":
                module = block
            elif kind in _ATTN_CAPTURE_KINDS:
                attention = layout.attn_module(blocks[index])
                # A stock nn.MultiheadAttention applies out_proj functionally, so
                # a hook on that submodule would never fire and the "fired once
                # per batch" check below would reject the whole capture. Hook the
                # attention itself and recover the projection's rows exactly.
                recompute = isinstance(attention, nn.MultiheadAttention)
                module = attention if recompute else layout.attn_proj_module(blocks[index])
            else:
                module = layout.proj_module(blocks[index])

            if recompute:
                def hook(mod, args, kwargs, value, *, name=key, capture_kind=kind):
                    if capture_kind == "attn_proj_input":
                        rows = _stock_mha_out_proj_input(mod, args, kwargs)
                        _verify_recomputed_attention_input(mod, rows, value)
                        store(name, rows)
                    else:
                        store(name, value)
                handles.append(module.register_forward_hook(hook, with_kwargs=True))
            else:
                def hook(_m, inputs, value, *, name=key, capture_kind=kind):
                    store(name, inputs[0] if capture_kind in _INPUT_CAPTURE_KINDS else value)
                handles.append(module.register_forward_hook(hook))
        for batch in batches:
            current_batch[0] = layout.batch_size(batch)
            current_mask[0] = layout.token_mask(batch) if mask_padding else None
            layout.forward(model, batch, device)
        if any(len(values) != len(batches) for values in output.values()):
            raise RuntimeError("A requested activation hook did not fire exactly once per batch")
    finally:
        for handle in handles:
            handle.remove()
        model.to(original_device).train(training)
    return output


def _rows(batches):
    return torch.cat([b.reshape(-1, b.shape[-1]) for b in batches], dim=0)


def _aligned(source, target):
    if len(source) != len(target):
        raise ValueError("Reference batch counts do not match")
    result = []
    for a, b in zip(source, target, strict=True):
        if a.shape[0] != b.shape[0]:
            raise ValueError("Reference image counts do not match")
        result.append(_interp_2d_tokens(a, b.shape[1]))
    return result


def capture_residual_references(
    source_base,
    source_ft,
    target_base,
    source_loader,
    target_loader,
    *,
    num_batches,
    seed,
    device,
    target_scope="inserted",
    family_adapter=None,
    capture_joint=False,
    mask_padding=False,
):
    return _capture_residual_references(
        source_base,
        source_ft,
        target_base,
        source_loader,
        target_loader,
        num_batches=num_batches,
        seed=seed,
        device=device,
        target_scope=target_scope,
        family_adapter=family_adapter,
        capture_joint=capture_joint,
        mask_padding=mask_padding,
    )


def _capture_residual_references(
    source_base,
    source_ft,
    target_base,
    source_loader,
    target_loader,
    *,
    num_batches,
    seed,
    device,
    target_scope,
    family_adapter=None,
    capture_joint=False,
    mask_padding=False,
):
    """Capture native source banks and target-position banks.

    The public function keeps the historical inserted-only signature.  The
    all-position path deliberately captures the target model at every final
    position, while retaining native source banks by original index.  The
    realized extension layout is only known after BRACE runs; completion then
    joins these banks by the layout's explicit ancestry instead of assuming an
    odd/even depth pattern.
    """
    if target_scope not in {"inserted", "all"}:
        raise ValueError("target_scope must be 'inserted' or 'all'")
    sb, tb, metadata = paired_calibration(source_loader, target_loader, num_batches=num_batches, seed=seed)
    layout_shim = _layout_for(family_adapter)
    depth = layout_shim.block_count(source_base)
    req = {str(i): (i, "boundary") for i in range(depth)}
    if capture_joint:
        for i in range(depth):
            req[f"{i}.c_proj_input"] = (i, "c_proj_input")
            req[f"{i}.c_proj_output"] = (i, "c_proj")
    capture = dict(family_adapter=family_adapter, mask_padding=mask_padding)
    base = capture_tokens(source_base, sb, req, device, **capture)
    ft = capture_tokens(source_ft, sb, req, device, **capture)
    target_depth = layout_shim.block_count(target_base)
    positions = [2 * i + 1 for i in range(depth)] if target_scope == "inserted" else list(range(target_depth))
    if not positions or max(positions) >= target_depth:
        raise ValueError(
            "Target model depth does not contain the requested target positions: "
            f"scope={target_scope!r}, source_depth={depth}, target_depth={target_depth}."
        )
    target_requests = {str(i): (i, "boundary") for i in positions}
    if capture_joint:
        for i in positions:
            target_requests[f"{i}.c_proj_input"] = (i, "c_proj_input")
            target_requests[f"{i}.c_proj_output"] = (i, "c_proj")
    target = capture_tokens(target_base, tb, target_requests, device, **capture)
    metadata["mask_padding"] = bool(mask_padding)
    if mask_padding:
        # Packed rows are paired row for row, so both sides must keep the same
        # real tokens. With a shared tokenizer they do; with two different
        # tokenizers the counts differ and _aligned would silently interpolate
        # across example boundaries, so refuse instead.
        source_counts = [int(b.shape[1]) for b in base["0"]]
        target_counts = [int(b.shape[1]) for b in target[str(positions[0])]]
        if source_counts != target_counts:
            raise ValueError(
                "mask_padding=True needs source and target to tokenize the calibration text to "
                f"the same real-token counts per batch; got source={source_counts[:4]}... "
                f"target={target_counts[:4]}... (different tokenizers?)"
            )
        masks = [layout_shim.token_mask(batch) for batch in tb]
        metadata["real_rows"] = sum(target_counts)
        metadata["padded_rows"] = sum(int(m.numel()) for m in masks if m is not None)

    # Keep source banks by original index for the all-position path.  The
    # inserted path also materializes the historical dictionaries immediately,
    # preserving its byte-compatible downstream behavior.
    source_base_outputs = {int(i): value for i, value in base.items() if i.isdigit()}
    source_ft_outputs = {int(i): value for i, value in ft.items() if i.isdigit()}
    source_base_cproj_inputs = {
        int(i.split(".", 1)[0]): value for i, value in base.items() if i.endswith(".c_proj_input")
    } if capture_joint else {}
    source_ft_cproj_inputs = {
        int(i.split(".", 1)[0]): value for i, value in ft.items() if i.endswith(".c_proj_input")
    } if capture_joint else {}
    source_base_cproj_outputs = {
        int(i.split(".", 1)[0]): value for i, value in base.items() if i.endswith(".c_proj_output")
    } if capture_joint else {}
    source_ft_cproj_outputs = {
        int(i.split(".", 1)[0]): value for i, value in ft.items() if i.endswith(".c_proj_output")
    } if capture_joint else {}
    target_outputs_by_position = {int(i): value for i, value in target.items() if i.isdigit()}
    target_cproj_inputs_by_position = {
        int(i.split(".", 1)[0]): value for i, value in target.items() if i.endswith(".c_proj_input")
    } if capture_joint else {}
    target_cproj_outputs_by_position = {
        int(i.split(".", 1)[0]): value for i, value in target.items() if i.endswith(".c_proj_output")
    } if capture_joint else {}
    desired, maps, target_outputs = {}, {}, {}
    if target_scope == "inserted":
        for i in range(depth):
            position = 2 * i + 1
            targets = target_outputs_by_position[position]
            source_base_batches = _aligned(source_base_outputs[i], targets)
            source_ft_batches = _aligned(source_ft_outputs[i], targets)
            q, mu_s, mu_t = centered_rectangular_procrustes(
                _rows(source_base_batches).double(), _rows(targets).double()
            )
            q = q.float()
            desired[position] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
            target_outputs[position] = targets
            maps[position] = {"P": q, "source_mean": mu_s.float(), "target_mean": mu_t.float()}
    result = {
        "desired": desired,
        "target_base_outputs": target_outputs,
        "maps": maps,
        "calibration": metadata,
    }
    if capture_joint:
        result.update(
            {
                "source_base_cproj_inputs": source_base_cproj_inputs,
                "source_ft_cproj_inputs": source_ft_cproj_inputs,
                "source_base_cproj_outputs": source_base_cproj_outputs,
                "source_ft_cproj_outputs": source_ft_cproj_outputs,
                "target_cproj_inputs_by_position": target_cproj_inputs_by_position,
                "target_cproj_outputs_by_position": target_cproj_outputs_by_position,
            }
        )
    if target_scope == "all":
        result.update(
            {
                "scope": target_scope,
                "source_base_outputs": source_base_outputs,
                "source_ft_outputs": source_ft_outputs,
                "target_base_outputs_by_position": target_outputs_by_position,
            }
        )
    return result


def projection_transforms(prepared, layout, *, target_scope="inserted", family_adapter=None):
    if target_scope not in {"inserted", "all"}:
        raise ValueError("target_scope must be 'inserted' or 'all'")
    transforms = prepared.get("transforms_by_key", {})
    output = {}
    if target_scope == "inserted":
        entries = layout.get("inserted_blocks")
        if entries is None:
            raise ValueError("Realized extension layout is missing inserted_blocks")
    else:
        entries = layout.get("final_blocks")
        if entries is None:
            raise ValueError("All target scope requires realized layout final_blocks")
    expected_positions = {int(row["position"]) for row in entries}
    if len(expected_positions) != len(entries):
        raise ValueError("Realized extension layout contains duplicate target positions")
    for row in entries:
        pos = int(row["position"])
        shim = _layout_for(family_adapter)
        key = shim.proj_key(pos, prefixed=False)
        transform = transforms.get(key, transforms.get(shim.proj_key(pos, prefixed=True)))
        if transform is None or transform.t_in is None or transform.t_out is None or transform.kind != "weight":
            kind = row.get("block_kind", "inserted")
            raise ValueError(f"Missing fitted c_proj transport at {kind} block position {pos}")
        output[pos] = {"t_in": transform.t_in.detach().float().cpu(), "t_out": transform.t_out.detach().float().cpu()}
    if set(output) != expected_positions:
        raise ValueError(
            "Fitted c_proj transport positions do not exactly match realized target positions: "
            f"expected={sorted(expected_positions)}, found={sorted(output)}"
        )
    return output



def _materialize_zero_bias(model, bias_key, out_features):
    """Add a zero bias to one bias-free projection. Returns True if it added one."""
    module_path = bias_key[: -len(".bias")]
    module = model
    for part in module_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    if getattr(module, "bias", None) is not None:
        return False
    weight = module.weight
    module.bias = nn.Parameter(torch.zeros(out_features, dtype=weight.dtype, device=weight.device))
    return True


def materialize_missing_projection_biases(
    target_model, target_base_state, layout, *, family_adapter=None, components=("mlp.c_proj",)
):
    """Give the target's residual projections a zero bias, in model and state alike.

    The exact affine form transports an intercept onto the projection's bias.
    CLIP's mlp.c_proj has one; an HF decoder's mlp.down_proj does not, so there
    is nowhere to put it. Adding a zero bias is behaviourally a no-op -- it
    changes no output until an intercept is written -- but it has to happen
    before completion runs and has to land in the base state dict too, so the
    merged state, the restore inside completion, and the eval load all agree on
    the model's shape. Doing it here rather than inside the solver is what keeps
    that consistent: the solver snapshots and restores with strict=True.

    ``components`` must list every projection the completion will actually fit,
    i.e. ``ResidualCompletionConfig.components``. The default is the historical
    single-component set, so existing callers are unchanged; the two-component
    direct arm has to pass its own, because an HF decoder's self_attn.o_proj is
    bias-free too (Qwen2/2.5 set ``attention_bias`` only on q/k/v) and the fit
    would otherwise die at the o_proj intercept with the exact form.

    Returns the keys it added, so a run can record that its checkpoint carries
    parameters the stock architecture does not.
    """
    shim = _layout_for(family_adapter)
    entries = layout.get("final_blocks") or layout.get("inserted_blocks") or ()
    added = []
    for row in entries:
        pos = int(row["position"])
        for component in order_components(components):
            weight_key = shim.component_key(pos, component, prefixed=True)
            bias_key = f"{weight_key[: -len('.weight')]}.bias"
            if bias_key in target_base_state:
                continue
            out_features = int(target_base_state[weight_key].shape[0])
            _materialize_zero_bias(target_model, bias_key, out_features)
            target_base_state[bias_key] = torch.zeros(
                out_features, dtype=target_base_state[weight_key].dtype
            )
            added.append(bias_key)
    return added


@torch.no_grad()
def complete_residuals(target_model, target_base_state, baseline_delta, references, transforms, layout, target_loader, *, config: ResidualCompletionConfig, device, family_adapter=None):
    """Fit all blocks at gamma=1, returning a separately scalable correction."""
    if references.get("scope", "inserted") != config.target_scope:
        raise ValueError(
            "Native reference scope does not match residual completion config: "
            f"references={references.get('scope', 'inserted')!r}, config={config.target_scope!r}"
        )
    if config.target_scope == "inserted":
        entries = sorted(layout.get("inserted_blocks", ()), key=lambda row: row["position"])
        block_kind = {int(row["position"]): "inserted" for row in entries}
        desired = references.get("desired", {})
        target_outputs = references.get("target_base_outputs", {})
        maps = references.get("maps", {})
    else:
        entries = sorted(layout.get("final_blocks", ()), key=lambda row: row["position"])
        if not entries:
            raise ValueError("All target scope requires non-empty realized layout final_blocks")
        # The transport arm is deliberately left on the historical trajectory:
        # its solve is defined against fitted (t_in, t_out) maps, and the config
        # parser already refuses "interpolate" outside direct_target mode.
        desired, target_outputs, maps, _coordinates = _materialize_all_scope_references(
            references, entries, trajectory="step"
        )
        block_kind = {int(row["position"]): str(row.get("block_kind", "unknown")) for row in entries}
    expected_positions = {int(row["position"]) for row in entries}
    if config.cascade_order == "top_bottom":
        entries = list(reversed(entries))
    for name, values in (("desired", desired), ("target_base_outputs", target_outputs), ("maps", maps), ("transforms", transforms)):
        if set(values) != expected_positions:
            raise ValueError(
                f"Residual completion {name} keys do not exactly match realized target positions: "
                f"expected={sorted(expected_positions)}, found={sorted(values)}"
            )
    meta = references["calibration"]
    if repr(_dataset_identity(target_loader.dataset)) != meta["dataset_identity"]:
        raise ValueError("Cached reference images do not match this task's validation dataset")
    batches = list(DataLoader(Subset(target_loader.dataset, meta["indices"]), batch_size=meta["batch_size"],
                              shuffle=False, num_workers=0, collate_fn=target_loader.collate_fn))
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections, source_corrections, diagnostics = {}, {}, []
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        for key, delta in baseline_delta.items():
            if key not in current_state or delta.shape != current_state[key].shape:
                raise ValueError(f"Baseline task vector is incompatible at {key}")
            current_state[key] = current_state[key] + delta.to(current_state[key])
        target_model.load_state_dict(current_state, strict=True)
        for row in entries:
            pos = int(row["position"])
            if config.target_scope == "inserted" and pos != 2 * int(row["source_orig_idx"]) + 1:
                raise ValueError("Realized insertion ancestry does not match captured references")
            shim = _layout_for(family_adapter)
            key = shim.proj_key(pos, prefixed=True)
            captured = capture_tokens(
                target_model, batches, {"h": (pos, "c_proj_input"), "out": (pos, "boundary")},
                device, family_adapter=family_adapter,
                mask_padding=bool(meta.get("mask_padding", False)),
            )
            t_in, t_out = transforms[pos]["t_in"], transforms[pos]["t_out"]
            block = shim.block_module(shim.blocks(target_model)[pos])
            scale_module = getattr(block, "ls_2", nn.Identity())
            effective_out = t_out
            if not isinstance(scale_module, nn.Identity):
                scale = getattr(scale_module, "gamma", None)
                if scale is None or scale.ndim != 1 or scale.shape[0] != t_out.shape[1]:
                    raise ValueError("Unsupported non-diagonal target LayerScale")
                effective_out = t_out * scale.detach().cpu().float().unsqueeze(0)
            # The Gram accumulation and eigendecomposed solve are the
            # expensive part (up to d_mlp x d_mlp, e.g. 4096x4096 on a
            # ViT-L/14 target); run them on the model's device rather than
            # CPU. Only the small fitted correction/bias tensors -- the same
            # shape as c_proj.weight/bias -- are moved back to CPU right
            # after, so every downstream line here is unchanged.
            stats = ResidualSufficientStatistics(device=device)
            for h, out, desired_batch, base_out in zip(
                captured["h"], captured["out"], desired[pos], target_outputs[pos], strict=True
            ):
                error = desired_batch - (out - base_out)
                stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), t_in, effective_out)
            correction, diag = stats.solve(ridge_relative=config.ridge_relative, exact_form=config.exact_form)
            correction = correction.cpu()
            diag["bias_correction"] = diag["bias_correction"].cpu()
            transported = t_out.T @ correction @ t_in
            if transported.shape != current_state[key].shape or not torch.isfinite(transported).all():
                raise RuntimeError("Residual completion produced an invalid transported projection")
            source_corrections[key] = correction
            target_corrections[key] = transported
            current_state[key] = current_state[key] + transported.to(current_state[key])
            # Affine intercept (Eq. 8-11): a source-coordinate bias correction,
            # transported through the same output-side map as the weight
            # (t_out.T @ bias_correction -- no t_in, a bias has no input
            # contraction; algebraically identical to
            # theseus._transport_bias's `delta_vec @ t_out` convention for a
            # 1-D vector). When exact_form=False this is exactly zero, so the
            # reduced path stays numerically byte-identical to before this
            # change apart from the (always-present) zero bias key.
            bias_key = f"{key[: -len('.weight')]}.bias"
            bias_correction = diag["bias_correction"]
            if bias_key not in current_state:
                # CLIP's mlp.c_proj always has a bias; HF decoder MLPs do not
                # (Qwen2.5 sets mlp_bias=False), so on a decoder the exact form
                # has nowhere to put its intercept. Measured on this pair the
                # activation banks are strongly off-centre (||mean||/std about
                # 1.2-1.9, largest at the deepest layer), so the intercept is
                # carrying real signal and cannot simply be discarded.
                if config.missing_bias == "materialize":
                    # The caller is responsible for materializing these before
                    # calling in: original_state is snapshotted above and
                    # restored with strict=True, so a parameter added here would
                    # make that restore fail (and would not survive into the
                    # merge or the eval load either).
                    raise RuntimeError(
                        f"missing_bias='materialize' requires {bias_key} to exist on the target "
                        "before residual completion runs; call "
                        "materialize_missing_projection_biases() on the target model and its "
                        "base state dict first"
                    )
                elif config.missing_bias == "skip":
                    # Only reachable with exact_form=False, where the intercept
                    # is exactly zero (the parser refuses the other combination),
                    # so nothing is being dropped.
                    if torch.count_nonzero(bias_correction):
                        raise RuntimeError(
                            "missing_bias='skip' would discard a nonzero intercept at "
                            f"{bias_key}; the weight was fitted on centered banks and is "
                            "not valid without it"
                        )
                    continue
                else:
                    raise RuntimeError(
                        f"Target model is missing the expected bias parameter {bias_key}. "
                        "Decoder MLP projections are bias-free; set "
                        "target_residual_completion.missing_bias to 'materialize' "
                        "(exact, adds the parameter) or 'skip' with exact_form=false."
                    )
            transported_bias = t_out.T @ bias_correction.to(t_out)
            if transported_bias.shape != current_state[bias_key].shape or not torch.isfinite(transported_bias).all():
                raise RuntimeError("Residual completion produced an invalid transported bias")
            source_corrections[bias_key] = bias_correction
            target_corrections[bias_key] = transported_bias
            current_state[bias_key] = current_state[bias_key] + transported_bias.to(current_state[bias_key])
            if config.cascade_order != "independent":
                target_model.load_state_dict(current_state, strict=True)
            diagnostics.append(
                {
                    "scope": config.target_scope,
                    "block_kind": block_kind[pos],
                    "position": pos,
                    "source_orig_idx": row["source_orig_idx"],
                    **diag,
                }
            )
    finally:
        target_model.load_state_dict(original_state, strict=True)
    return source_corrections, target_corrections, diagnostics

@torch.no_grad()
def complete_residuals_direct(
    target_model,
    target_base_state,
    references,
    layout,
    target_loader,
    *,
    config: ResidualCompletionConfig,
    device,
    family_adapter=None,
):
    """Transport-free Proposal 1: synthesize the whole target task vector.

    ``complete_residuals`` completes the residual left over by an already
    transported task vector.  This variant answers a different question: can
    the desired local functional effect be written directly into the target's
    residual-writing projections with **no parameter transport at all**?  The
    temporary model therefore starts at the native target base
    ``theta_t^0`` -- there is no ``tau_t`` -- and the accumulated correction is
    the entire returned task vector.

    Everything upstream of the solve is shared with the transport-aware path
    and is used unchanged: the same paired calibration batches, the same
    pre-resize source reference banks ``B_i^0, B_i^1``, the same centered
    rectangular Procrustes map ``Q_j``, and therefore the same desired effect
    ``D_j = (B_i^1 - B_i^0) Q_j``.  Only the coordinate system of the solve
    differs.  With no transport there is no source coordinate system to
    respect, so the affine ridge of Eq. 8-9 is solved directly in target
    coordinates,

        min_{Delta C_t, beta_t} || (H_j Delta C_t^T + 1 beta_t^T) - E_j ||_F^2
                                + lam ||Delta C_t||_F^2,

    which is exactly the transport-aware objective at ``t_in = I`` and
    ``t_out = I``.  It is deliberately run through the *same*
    ``ResidualSufficientStatistics`` solver rather than a second
    implementation of the normal equations: ``t_in=None`` is the identity
    input map (never materialized), and ``t_out`` carries only the target's
    own LayerScale, exactly as in the transport-aware path.

    The sequential cascade is preserved.  Blocks are fitted in realized-layout
    order against a temporary model that already carries every previously
    fitted correction, so

        E_j = D_j - (T_j^cur - T_j^0)

    and a later block only repairs the effect its predecessors did not already
    produce.  At the first fitted block ``T_j^cur == T_j^0`` by construction,
    so ``E_j == D_j`` exactly; that identity is asserted numerically rather
    than assumed.

    Returns ``(target_corrections, diagnostics)``.  The corrections are an
    ordinary target-space task vector fitted at unit strength; ``config
    .strength`` (gamma) is applied afterwards by ``scale_completion``, so
    ``gamma=0`` reproduces the native target base exactly.  The target model is
    restored to its entry state in a ``finally``; no hooks or modules survive.
    """
    if config.mode != "direct_target":
        raise ValueError(
            f"complete_residuals_direct requires mode='direct_target', got {config.mode!r}"
        )
    if references.get("scope", "inserted") != config.target_scope:
        raise ValueError(
            "Native reference scope does not match residual completion config: "
            f"references={references.get('scope', 'inserted')!r}, config={config.target_scope!r}"
        )
    if config.target_scope == "inserted":
        entries = sorted(layout.get("inserted_blocks", ()), key=lambda row: row["position"])
        block_kind = {int(row["position"]): "inserted" for row in entries}
        desired = references.get("desired", {})
        target_outputs = references.get("target_base_outputs", {})
        maps = references.get("maps", {})
        coordinates = {int(row["position"]): float(row["source_orig_idx"]) for row in entries}
    else:
        entries = sorted(layout.get("final_blocks", ()), key=lambda row: row["position"])
        if not entries:
            raise ValueError("All target scope requires non-empty realized layout final_blocks")
        desired, target_outputs, maps, coordinates = _materialize_all_scope_references(
            references, entries, trajectory=config.target_trajectory
        )
        block_kind = {int(row["position"]): str(row.get("block_kind", "unknown")) for row in entries}
    if config.cascade_order == "top_bottom":
        # Fit deepest-first. The E_j == D_j assertion below keys off `index == 0`
        # (whichever position is visited first still sees the untouched base),
        # so it stays valid; only the coupling direction changes.
        entries = list(reversed(entries))
    expected_positions = {int(row["position"]) for row in entries}
    for name, values in (("desired", desired), ("target_base_outputs", target_outputs), ("maps", maps)):
        if set(values) != expected_positions:
            raise ValueError(
                f"Residual completion {name} keys do not exactly match realized target positions: "
                f"expected={sorted(expected_positions)}, found={sorted(values)}"
            )
    meta = references["calibration"]
    if repr(_dataset_identity(target_loader.dataset)) != meta["dataset_identity"]:
        raise ValueError("Cached reference images do not match this task's validation dataset")
    batches = list(DataLoader(Subset(target_loader.dataset, meta["indices"]), batch_size=meta["batch_size"],
                              shuffle=False, num_workers=0, collate_fn=target_loader.collate_fn))
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections, diagnostics = {}, []
    try:
        # No transported vector: the temporary model *is* the native target
        # base. Asserted rather than assumed, because the whole claim of this
        # arm is that nothing but the fitted correction reaches the target.
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        components = order_components(config.components)
        for index, row in enumerate(entries):
            pos = int(row["position"])
            if config.target_scope == "inserted" and pos != 2 * int(row["source_orig_idx"]) + 1:
                raise ValueError("Realized insertion ancestry does not match captured references")
            shim = _layout_for(family_adapter)
            # Components are fitted in block-forward order and cascaded, never
            # solved jointly. out_proj writes before the MLP, so mounting
            # Delta_O moves the MLP's own input through ln_2 and GELU -- a
            # nonlinearity no single linear system can absorb. Each component's
            # capture below therefore *re-measures* the residual the previous
            # one actually left, exactly as the cross-block cascade does.
            block_rows: list[dict[str, Any]] = []
            for component_index, component in enumerate(components):
                key = shim.component_key(pos, component, prefixed=True)
                captured = capture_tokens(
                    target_model, batches,
                    {"h": (pos, COMPONENT_INPUT_KIND[component]), "out": (pos, "boundary")},
                    device, family_adapter=family_adapter,
                    mask_padding=bool(meta.get("mask_padding", False)),
                )
                width = int(current_state[key].shape[0])
                identity_out = torch.eye(width, dtype=torch.float32)
                scale_module = shim.component_scale_module(shim.blocks(target_model)[pos], component)
                effective_out = identity_out
                if not isinstance(scale_module, nn.Identity):
                    scale = getattr(scale_module, "gamma", None)
                    if scale is None or scale.ndim != 1 or scale.shape[0] != width:
                        raise ValueError("Unsupported non-diagonal target LayerScale")
                    # diag(gamma): the residual stream receives gamma * proj(h),
                    # so the fit must predict through that scaling exactly as the
                    # transport-aware path folds it into t_out. ls_1 scales the
                    # attention write, ls_2 the MLP write.
                    effective_out = identity_out * scale.detach().cpu().float().unsqueeze(0)
                # See the identical comment in complete_residuals(): the Gram
                # accumulation and eigendecomposed solve are the expensive part,
                # so they run on the model's device; the fitted correction/bias
                # are moved back to CPU right after solve() below.
                stats = ResidualSufficientStatistics(device=device)
                desired_sq = 0.0
                effect_sq = 0.0
                for h, out, desired_batch, base_out in zip(
                    captured["h"], captured["out"], desired[pos], target_outputs[pos], strict=True
                ):
                    effect = out - base_out
                    error = desired_batch - effect
                    desired_sq += float((desired_batch.double() ** 2).sum().item())
                    effect_sq += float((effect.double() ** 2).sum().item())
                    stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)
                if index == 0 and component_index == 0:
                    # E_j == D_j at the very first fit: the temporary model is
                    # still the untouched target base there, so the current effect
                    # is identically zero. A nonzero effect here means a stale
                    # reference bank or a mutated base, not a small numerical drift.
                    if effect_sq > 1e-12 * max(desired_sq, 1.0):
                        raise RuntimeError(
                            "Direct completion started from a target model that is not the native "
                            f"base: nonzero pre-fit effect at position {pos} (||T-T0||^2={effect_sq:.3e})"
                        )
                correction, diag = stats.solve(ridge_relative=config.ridge_relative, exact_form=config.exact_form)
                correction = correction.cpu()
                diag["bias_correction"] = diag["bias_correction"].cpu()
                if block_rows:
                    # This capture happened after the previous component was
                    # mounted, so its residual is that component's *true*
                    # post-mount residual. For attn.out_proj the solver's own
                    # residual_norm_after is only a linear prediction; this is
                    # the measured one, and their ratio is the MLP knock-on.
                    block_rows[-1]["measured_residual_norm_after"] = diag["residual_norm_before"]
                # t_in = I and the write-side t_out = I: the fitted matrix already
                # lives in target coordinates, so there is nothing to transport.
                if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
                    raise RuntimeError("Direct residual completion produced an invalid projection")
                target_corrections[key] = correction
                current_state[key] = current_state[key] + correction.to(current_state[key])
                bias_key = f"{key[: -len('.weight')]}.bias"
                bias_correction = diag["bias_correction"]
                skip_bias = False
                if bias_key not in current_state:
                    if config.missing_bias == "materialize":
                        raise RuntimeError(
                            f"missing_bias='materialize' requires {bias_key} to exist on the target "
                            "before residual completion runs; call "
                            "materialize_missing_projection_biases() on the target model and its "
                            "base state dict first"
                        )
                    elif config.missing_bias == "skip":
                        if torch.count_nonzero(bias_correction):
                            raise RuntimeError(
                                "missing_bias='skip' would discard a nonzero intercept at "
                                f"{bias_key}; the weight was fitted on centered banks and is "
                                "not valid without it"
                            )
                        skip_bias = True
                    else:
                        raise RuntimeError(
                            f"Target model is missing the expected bias parameter {bias_key}. "
                            "Decoder MLP projections are bias-free; set "
                            "target_residual_completion.missing_bias to 'materialize' "
                            "(exact, adds the parameter) or 'skip' with exact_form=false."
                        )
                if not skip_bias:
                    bias_delta = bias_correction.to(current_state[bias_key])
                    if bias_delta.shape != current_state[bias_key].shape or not torch.isfinite(bias_delta).all():
                        raise RuntimeError("Direct residual completion produced an invalid bias")
                    target_corrections[bias_key] = bias_correction
                    current_state[bias_key] = current_state[bias_key] + bias_delta
                # Mounted before the next component captures, which is what makes
                # the intra-block cascade real rather than two independent fits.
                # cascade_order="independent" deliberately skips the mount: the
                # model stays at the pristine base for every fit, so each block
                # sees E_j == D_j and no correction can move another's target.
                if config.cascade_order != "independent":
                    target_model.load_state_dict(current_state, strict=True)
                desired_norm = desired_sq ** 0.5
                block_rows.append(
                    {
                        "mode": "direct_target",
                        "scope": config.target_scope,
                        "component": component,
                        "trajectory": config.target_trajectory,
                        "target_coordinate": float(coordinates[pos]),
                        "block_kind": block_kind[pos],
                        "position": pos,
                        "source_orig_idx": row["source_orig_idx"],
                        "desired_norm": desired_norm,
                        "effect_before_norm": effect_sq ** 0.5,
                        # r_j: the fraction of the desired local effect still
                        # missing, before and after this fit. For mlp.c_proj the
                        # "after" value is exact rather than re-measured: it is
                        # the last operation writing into the residual stream in
                        # this block and its input H_j does not depend on its own
                        # weight, so mounting the correction changes the block
                        # output by exactly gamma * (H_j dC^T + beta) -- the same
                        # quantity the solver already evaluated. That reasoning
                        # does NOT hold for attn.out_proj, which is why its row
                        # also carries measured_residual_norm_after. Pinned by
                        # tests/test_direct_target_p1.py and
                        # tests/test_direct_p1_trajectory_and_components.py.
                        "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
                        "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                        "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                        **diag,
                    }
                )
            diagnostics.extend(block_rows)
    finally:
        target_model.load_state_dict(original_state, strict=True)
    return target_corrections, diagnostics

@torch.no_grad()
def complete_joint_blockwise(
    target_model,
    target_base_state,
    baseline_delta,
    references,
    transforms,
    layout,
    target_loader,
    *,
    config: JointCorrectionConfig,
    device,
    family_adapter=None,
):
    """Solve Option 3 once per realized block and return target-space deltas.

    The source term penalizes disturbing the post-ARIADNE inserted block on
    its actual resized-source inputs.  The target term fits the remaining
    boundary-effect error after ordinary transport, using the same frozen
    input/output maps. Corrections are mounted sequentially, so later blocks
    observe earlier repairs. The target model is restored before returning;
    only the returned task-vector dictionaries are consumed by the caller.
    """
    if not config.enabled:
        return {}, {}, []
    entries = sorted(layout.get("inserted_blocks", ()), key=lambda row: row["position"])
    # The caller passes the realized layout.  ``target_scope=all`` is not part
    # of the initial Option-3 schema, but accepting final_blocks here keeps a
    # future parser extension from silently selecting the wrong positions.
    if getattr(config, "target_scope", "inserted") == "all":
        entries = sorted(layout.get("final_blocks", ()), key=lambda row: row["position"])
    if not entries:
        raise ValueError("Joint blockwise correction requires a non-empty realized layout")
    expected_positions = {int(row["position"]) for row in entries}
    if set(transforms) != expected_positions:
        raise ValueError(
            "Joint blockwise transforms do not match realized positions: "
            f"expected={sorted(expected_positions)}, found={sorted(transforms)}"
        )
    metadata = references.get("calibration")
    if metadata is None or repr(_dataset_identity(target_loader.dataset)) != metadata["dataset_identity"]:
        raise ValueError("Joint blockwise references do not match this task's validation dataset")
    batches = list(
        DataLoader(
            Subset(target_loader.dataset, metadata["indices"]),
            batch_size=metadata["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=target_loader.collate_fn,
        )
    )
    source_h_by_position = references.get("resized_source_cproj_inputs_by_position", {})
    target_base_outputs = references.get("target_base_outputs_by_position", {})
    target_h_refs = references.get("target_cproj_inputs_by_position", {})
    # Inserted-only capture stores the same target banks under the historical
    # target_base_outputs key; normalize it here for one common implementation.
    if not target_base_outputs:
        target_base_outputs = references.get("target_base_outputs", {})
    if not source_h_by_position or not target_h_refs:
        raise ValueError("Joint blockwise references are missing c_proj activation banks")

    original_state = {key: value.detach().cpu().clone() for key, value in target_model.state_dict().items()}
    current_state = {key: value.detach().cpu().clone() for key, value in target_base_state.items()}
    for key, delta in baseline_delta.items():
        if key not in current_state or tuple(delta.shape) != tuple(current_state[key].shape):
            raise ValueError(f"Baseline task vector is incompatible at {key}")
        current_state[key] = current_state[key] + delta.to(current_state[key])
    source_corrections, target_corrections, diagnostics = {}, {}, []
    try:
        target_model.load_state_dict(current_state, strict=True)
        for row in entries:
            pos = int(row["position"])
            source_idx = int(row["source_orig_idx"])
            transform = transforms[pos]
            t_in, t_out = transform["t_in"], transform["t_out"]
            key = _layout_for(family_adapter).proj_key(pos, prefixed=True)
            captured = capture_tokens(
                target_model,
                batches,
                {"h": (pos, "c_proj_input"), "out": (pos, "boundary")},
                device,
                family_adapter=family_adapter,
            )
            source_h_rows = _rows(source_h_by_position[pos])
            # X is additive to the already-fitted ARIADNE/transport task
            # vector.  A zero source target preserves the baseline source
            # reconstruction and prevents double-counting the full FT-base
            # effect in the joint objective.
            source_y_rows = torch.zeros(
                source_h_rows.shape[0], int(t_out.shape[0]), dtype=source_h_rows.dtype
            )
            desired_target_batches = references["desired"][pos]
            target_effect_batches = [
                desired - (current - native)
                for desired, current, native in zip(
                    desired_target_batches,
                    captured["out"],
                    target_base_outputs[pos],
                    strict=True,
                )
            ]
            target_h_rows = _rows(captured["h"])
            target_effect_rows = _rows(target_effect_batches)
            block = _layout_for(family_adapter).block_module(_layout_for(family_adapter).blocks(target_model)[pos])
            scale_module = getattr(block, "ls_2", nn.Identity())
            effective_out = t_out
            if not isinstance(scale_module, nn.Identity):
                scale = getattr(scale_module, "gamma", None)
                if scale is None or scale.ndim != 1 or scale.shape[0] != t_out.shape[1]:
                    raise ValueError("Unsupported non-diagonal target LayerScale")
                effective_out = t_out * scale.detach().cpu().float().unsqueeze(0)
            correction, diag = fit_joint_cproj_correction(
                source_h_rows,
                source_y_rows,
                target_h_rows,
                target_effect_rows,
                t_in,
                effective_out,
                source_weight=config.source_weight,
                target_weight=config.target_weight,
                ridge_relative=config.ridge_relative,
            )
            transported = t_out.T @ correction @ t_in
            if key not in current_state or tuple(transported.shape) != tuple(current_state[key].shape):
                raise RuntimeError(f"Joint blockwise correction has invalid shape at {key}")
            source_corrections[key] = correction
            target_corrections[key] = transported
            current_state[key] = current_state[key] + transported.to(current_state[key])

            # The joint solve is affine in source coordinates.  Keep the
            # intercept as a real c_proj.bias task-vector correction and push
            # it through the same frozen output map as the weight.  Vision
            # c_proj has a bias parameter; refusing a missing key prevents a
            # fitted nonzero affine term from being silently discarded.
            bias = diag.get("bias_correction")
            if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
                raise RuntimeError("Joint blockwise solver did not return a source-coordinate bias correction")
            bias_key = f"{key[:-len('.weight')]}.bias"
            if bias_key not in current_state:
                raise RuntimeError(
                    "Joint blockwise affine correction requires a target c_proj.bias parameter; "
                    f"missing {bias_key}"
                )
            transported_bias = t_out.T @ bias.to(t_out)
            if tuple(transported_bias.shape) != tuple(current_state[bias_key].shape) or not torch.isfinite(transported_bias).all():
                raise RuntimeError(f"Joint blockwise correction has invalid bias shape at {bias_key}")
            source_corrections[bias_key] = bias
            target_corrections[bias_key] = transported_bias
            current_state[bias_key] = current_state[bias_key] + transported_bias.to(current_state[bias_key])
            target_model.load_state_dict(current_state, strict=True)
            diagnostics.append(
                {
                    "position": pos,
                    "source_orig_idx": source_idx,
                    "block_kind": row.get("block_kind", "inserted"),
                    "bias_correction": bias,
                    "transported_bias_correction": transported_bias,
                    **diag,
                }
            )
    finally:
        target_model.load_state_dict(original_state, strict=True)
    return source_corrections, target_corrections, diagnostics


@torch.no_grad()
def complete_direct_p1_shared_correction(
    source_base_model,
    source_ft_model,
    target_model,
    target_base_state,
    baseline_delta,
    references,
    transforms,
    layout,
    source_loader,
    target_loader,
    *,
    config: JointCorrectionConfig,
    device,
):
    """Refine ARIADNE's shared c_proj maps with a frozen-map P1 objective.

    For each inserted block, fit an additive output-affine map ``(B, c)``.
    The source term reconstructs the original ARIADNE c_proj reference from
    the current resized-base output. The target term makes the change induced
    in the *shared* base/FT task vector explain the remaining P1 boundary
    residual. Applying ``I+B, c`` to both endpoints preserves a shared affine
    resize; ``c`` cancels from their task vector.
    """
    if not config.enabled:
        return {}, []
    entries = sorted(layout.get("inserted_blocks", ()), key=lambda row: row["position"])
    if not entries:
        raise ValueError("Direct P1 correction requires inserted blocks")
    expected = {int(row["position"]) for row in entries}
    if set(transforms) != expected:
        raise ValueError(f"Direct P1 transforms mismatch: expected={sorted(expected)}, found={sorted(transforms)}")
    metadata = references.get("calibration")
    if metadata is None:
        raise ValueError("Direct P1 references are missing calibration metadata")
    if repr(_dataset_identity(source_loader.dataset)) != metadata["dataset_identity"]:
        raise ValueError("Direct P1 source calibration dataset mismatch")
    if repr(_dataset_identity(target_loader.dataset)) != metadata["dataset_identity"]:
        raise ValueError("Direct P1 target calibration dataset mismatch")

    def _batches(loader):
        return list(DataLoader(
            Subset(loader.dataset, metadata["indices"]),
            batch_size=metadata["batch_size"], shuffle=False, num_workers=0,
            collate_fn=loader.collate_fn,
        ))

    source_batches, target_batches = _batches(source_loader), _batches(target_loader)
    native_source_outputs = references.get("source_base_cproj_outputs", {})
    native_target_outputs = references.get("target_base_outputs_by_position", {}) or references.get("target_base_outputs", {})
    if not native_source_outputs or not native_target_outputs:
        raise ValueError("Direct P1 references are missing source c_proj or target boundary banks")

    shim = _layout_for(None)
    source_base_original = {key: value.detach().cpu().clone() for key, value in source_base_model.state_dict().items()}
    source_ft_original = {key: value.detach().cpu().clone() for key, value in source_ft_model.state_dict().items()}
    target_original = {key: value.detach().cpu().clone() for key, value in target_model.state_dict().items()}
    target_state = {key: value.detach().cpu().clone() for key, value in target_base_state.items()}
    for key, delta in baseline_delta.items():
        target_state[key] = target_state[key] + delta.to(target_state[key])
    target_corrections: dict[str, torch.Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        target_model.load_state_dict(target_state, strict=True)
        for row in entries:
            pos, source_idx = int(row["position"]), int(row["source_orig_idx"])
            t_in, t_out = transforms[pos]["t_in"], transforms[pos]["t_out"]
            source_capture = capture_tokens(
                source_base_model, source_batches, {"out": (pos, "c_proj")}, device
            )["out"]
            current_source = source_capture
            reference_source = _aligned(native_source_outputs[source_idx], source_capture)
            source_rows = _rows(current_source)
            source_residual = _rows([
                reference - current
                for current, reference in zip(current_source, reference_source, strict=True)
            ])

            source_base_block = shim.block_module(shim.blocks(source_base_model)[pos])
            source_ft_block = shim.block_module(shim.blocks(source_ft_model)[pos])
            delta_weight = (source_ft_block.mlp.c_proj.weight - source_base_block.mlp.c_proj.weight).detach().cpu().float()
            delta_bias = (source_ft_block.mlp.c_proj.bias - source_base_block.mlp.c_proj.bias).detach().cpu().float()

            captured_target = capture_tokens(
                target_model, target_batches,
                {"h": (pos, "c_proj_input"), "boundary": (pos, "boundary")},
                device,
            )
            desired_batches = references["desired"][pos]
            target_residual_batches = [
                desired - (current - native)
                for desired, current, native in zip(
                    desired_batches, captured_target["boundary"], native_target_outputs[pos], strict=True
                )
            ]
            h_target = _rows(captured_target["h"])
            z_target = (h_target @ t_in.T) @ delta_weight.T + delta_bias
            target_residual = _rows(target_residual_batches)

            target_block = shim.block_module(shim.blocks(target_model)[pos])
            scale_module = getattr(target_block, "ls_2", nn.Identity())
            effective_out = t_out
            if not isinstance(scale_module, nn.Identity):
                scale = getattr(scale_module, "gamma", None)
                if scale is None or scale.ndim != 1 or scale.shape[0] != t_out.shape[1]:
                    raise ValueError("Unsupported non-diagonal target LayerScale")
                effective_out = t_out * scale.detach().cpu().float().unsqueeze(0)

            identity = torch.eye(z_target.shape[1], dtype=z_target.dtype)
            correction, diag = fit_joint_cproj_correction(
                source_rows, source_residual, z_target, target_residual,
                identity, effective_out,
                source_weight=config.source_weight,
                target_weight=config.target_weight,
                ridge_relative=config.ridge_relative,
                target_intercept=False,
            )
            shared_bias = diag["bias_correction"]
            affine = torch.eye(correction.shape[0], dtype=correction.dtype) + correction
            for block in (source_base_block, source_ft_block):
                projection = block.mlp.c_proj
                weight = projection.weight.detach().cpu().float()
                bias = projection.bias.detach().cpu().float()
                projection.weight.copy_((affine @ weight).to(projection.weight))
                projection.bias.copy_((affine @ bias + shared_bias).to(projection.bias))

            delta_weight_change = correction @ delta_weight
            delta_bias_change = correction @ delta_bias
            weight_key = shim.proj_key(pos, prefixed=True)
            bias_key = f"{weight_key[:-len('.weight')]}.bias"
            transported_weight = t_out.T @ delta_weight_change @ t_in
            transported_bias = t_out.T @ delta_bias_change
            target_corrections[weight_key] = transported_weight
            target_corrections[bias_key] = transported_bias
            target_state[weight_key] = target_state[weight_key] + transported_weight.to(target_state[weight_key])
            target_state[bias_key] = target_state[bias_key] + transported_bias.to(target_state[bias_key])
            target_model.load_state_dict(target_state, strict=True)
            diagnostics.append({
                "position": pos,
                "source_orig_idx": source_idx,
                "shared_affine_bias_norm": float(torch.linalg.norm(shared_bias)),
                "task_bias_change_norm": float(torch.linalg.norm(delta_bias_change)),
                **{key: value for key, value in diag.items() if key != "bias_correction"},
            })
    finally:
        source_base_model.load_state_dict(source_base_original, strict=True)
        source_ft_model.load_state_dict(source_ft_original, strict=True)
        target_model.load_state_dict(target_original, strict=True)
    return target_corrections, diagnostics


@torch.no_grad()
def capture_resized_joint_source_inputs(
    source_model,
    source_loader,
    references,
    layout,
    *,
    device,
    family_adapter=None,
):
    """Attach actual post-ARIADNE inserted-block inputs to joint references.

    Option 3's source penalty is evaluated where its additive correction will
    act.  Native ancestor inputs are useful provenance, but are not a faithful
    substitute after structural insertion and upstream ARIADNE corrections.
    """
    metadata = references.get("calibration")
    if metadata is None or repr(_dataset_identity(source_loader.dataset)) != metadata["dataset_identity"]:
        raise ValueError("Joint source capture does not match the paired calibration dataset")
    entries = sorted(layout.get("inserted_blocks", ()), key=lambda row: row["position"])
    if not entries:
        raise ValueError("Joint source capture requires inserted blocks in the realized layout")
    batches = list(
        DataLoader(
            Subset(source_loader.dataset, metadata["indices"]),
            batch_size=metadata["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=source_loader.collate_fn,
        )
    )
    requests = {str(int(row["position"])): (int(row["position"]), "c_proj_input") for row in entries}
    captured = capture_tokens(source_model, batches, requests, device, family_adapter=family_adapter)
    updated = dict(references)
    updated["resized_source_cproj_inputs_by_position"] = {
        int(position): bank for position, bank in captured.items()
    }
    return updated


def _ancestry_groups(entries):
    """Runs of consecutive realized positions sharing one source ancestor."""
    groups: list[tuple[int, list]] = []
    for row in entries:
        source_idx = int(row["source_orig_idx"])
        if groups and groups[-1][0] == source_idx:
            groups[-1][1].append(row)
        else:
            groups.append((source_idx, [row]))
    return groups


def realized_source_coordinates(entries):
    """Fractional source-depth coordinate of every realized target position.

    A group of ``m`` consecutive positions realizing source block ``i`` splits
    the step from source boundary ``i-1`` to boundary ``i`` into ``m`` equal
    parts, so its ``t``-th member sits at ``i - 1 + (t + 1) / m``.  Two
    consequences matter: the *last* member of every group lands exactly on the
    integer ``i`` (which is where "step" and "interpolate" must agree), and a
    group of size 1 is the integer itself, making the feature an exact no-op on
    any layout that did not duplicate that block.

    Derived from the realized layout's recorded ancestry, never from an assumed
    doubling pattern.
    """
    coordinates = {}
    for source_idx, rows in _ancestry_groups(entries):
        size = len(rows)
        for offset, row in enumerate(rows):
            coordinates[int(row["position"])] = source_idx - 1 + (offset + 1) / size
    return coordinates


def _materialize_all_scope_references(references, entries, *, trajectory="step"):
    """Join native banks to every realized target position by explicit ancestry.

    ``trajectory="step"`` hands each position its ancestor's whole effect, the
    historical protocol.  ``"interpolate"`` blends the two neighbouring source
    effects by the position's fractional depth, so a position realizing the
    first half of source block ``i`` is asked for half the step from ``i-1`` to
    ``i`` rather than for all of it a block early.
    """
    if trajectory not in {"step", "interpolate"}:
        raise ValueError("trajectory must be 'step' or 'interpolate'")
    source_base = references.get("source_base_outputs")
    source_ft = references.get("source_ft_outputs")
    target_by_position = references.get("target_base_outputs_by_position")
    if source_base is None or source_ft is None or target_by_position is None:
        raise ValueError("All target scope requires native source and position-specific target reference banks")
    expected_positions = {int(row["position"]) for row in entries}
    if set(target_by_position) != expected_positions:
        raise ValueError(
            "Position-specific target references do not exactly match realized target positions: "
            f"expected={sorted(expected_positions)}, found={sorted(target_by_position)}"
        )
    coordinates = realized_source_coordinates(entries)
    desired, target_outputs, maps = {}, {}, {}
    for row in entries:
        pos = int(row["position"])
        source_idx = int(row["source_orig_idx"])
        if source_idx not in source_base or source_idx not in source_ft:
            raise ValueError(f"Missing native source reference for ancestry index {source_idx} at target position {pos}")
        targets = target_by_position[pos]
        coordinate = coordinates[pos]
        lower, upper = math.floor(coordinate), math.ceil(coordinate)

        if trajectory == "step" or lower == upper:
            # Integer coordinates are the last member of their ancestry group,
            # where the two trajectories are the same target by construction.
            # Taking the identical code path keeps them bitwise identical rather
            # than merely close.
            base_batches = _aligned(source_base[source_idx], targets)
            ft_batches = _aligned(source_ft[source_idx], targets)
            delta_batches = [f - b for b, f in zip(base_batches, ft_batches, strict=True)]
        else:
            weight = coordinate - lower
            upper_base = _aligned(source_base[upper], targets)
            upper_delta = [
                f - b for b, f in zip(upper_base, _aligned(source_ft[upper], targets), strict=True)
            ]
            if lower < 0:
                # Delta_{-1} = 0: no task effect exists before the first source
                # block. Its *base* bank has no counterpart either -- nothing is
                # captured before block 0 -- so the alignment bank is clamped to
                # B_0^0. That leaves Q fitted half a step deep, the same
                # mismatch "step" carries at every position, instead of against
                # a bank that was never recorded.
                lower_base, lower_delta = upper_base, [torch.zeros_like(b) for b in upper_base]
            else:
                lower_base = _aligned(source_base[lower], targets)
                lower_delta = [
                    f - b for b, f in zip(lower_base, _aligned(source_ft[lower], targets), strict=True)
                ]
            base_batches = [
                (1.0 - weight) * low + weight * high
                for low, high in zip(lower_base, upper_base, strict=True)
            ]
            delta_batches = [
                (1.0 - weight) * low + weight * high
                for low, high in zip(lower_delta, upper_delta, strict=True)
            ]

        # Q is fitted from whichever base bank the desired effect is expressed
        # against, so the alignment always matches the delta being transported.
        q, mu_s, mu_t = centered_rectangular_procrustes(
            _rows(base_batches).double(), _rows(targets).double()
        )
        q = q.float()
        desired[pos] = [delta @ q for delta in delta_batches]
        target_outputs[pos] = targets
        maps[pos] = {"P": q, "source_mean": mu_s.float(), "target_mean": mu_t.float()}
    return desired, target_outputs, maps, coordinates


def scale_completion(baseline, completion, strength):
    if strength == 0:
        return dict(baseline)
    result = dict(baseline)
    for key, correction in completion.items():
        if key not in baseline or baseline[key].shape != correction.shape:
            raise ValueError(f"Completion key does not match baseline: {key}")
        result[key] = baseline[key] + strength * correction.to(baseline[key])
    return result


def cache_identity(cfg, task, target_hash):
    from .block_extension import resolve_block_extension_config
    checkpoint = Path(cfg["tuned_ckpts"][task]).resolve()
    stat = checkpoint.stat()
    keys = ("source_clip_model", "source_clip_pretrained", "target_clip_model", "target_clip_pretrained",
            "method", "method_params", "seed", "val_fraction", "batch_size", "dtype")
    protocol = {k: cfg.get(k) for k in keys}
    protocol["block_extension_params"] = asdict(resolve_block_extension_config(cfg)[1])
    shared = parse_target_shared_config(cfg.get("target_shared_correction"))
    protocol["target_shared_correction"] = asdict(shared) if shared.enabled and shared.target_weight > 0 else None
    implementation = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for relative in ("eval/block_extension.py", "eval/target_informed_runtime.py", "eval/target_residual_completion.py",
                     "rebase/methods/theseus.py", "rebase/methods/bico.py"):
        implementation.update((root / relative).read_bytes())
    return {"schema_version": 1, "task": task, "target_hash": target_hash,
            "checkpoint": {"path": str(checkpoint), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "sha256": _checkpoint_hash(str(checkpoint), stat.st_size, stat.st_mtime_ns)},
            "implementation_sha256": implementation.hexdigest(), "protocol": protocol}


@lru_cache(maxsize=64)
def _checkpoint_hash(path, size, mtime_ns):
    del size, mtime_ns  # Included in the cache key so changed checkpoints are rehashed.
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_cache(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cached experiment artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_cache(path, expected_identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("identity") != expected_identity:
        raise ValueError(f"Cached task vector provenance does not match the requested experiment: {path}")
    return payload
