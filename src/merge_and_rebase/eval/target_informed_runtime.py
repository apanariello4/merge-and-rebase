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

import contextlib
import copy
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
    CANONICAL_COMPONENT_ORDER,
    COMPONENT_FORWARD_ORDER,
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
    active = (
        shared.enabled or residual.enabled or joint or direct_p1 or bool(cfg.get("capture_target_residual_reference"))
    )
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
    if (
        block_cfg.insertion_order != "bottom-top"
        or block_cfg.extension_density not in {"spread", "spread_mod"}
        or block_cfg.extension_strategy != "duplicate_per_weight"
    ):
        raise ValueError("Target-informed BRACE requires bottom-top spread duplicate insertion")
    if block_cfg.blocks_to_add not in (None, source_depth):
        raise ValueError("blocks_to_add must match the depth-doubling protocol")
    identity_p1 = (
        residual.enabled and block_cfg.skip_correction and block_cfg.inserted_block_mode == "residual_identity"
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
    if str((cfg.get("block_extension_params") or {}).get("calibration_protocol", "task_local")).startswith(
        "vision8_mix"
    ):
        raise ValueError("Mixed-dataset target references are outside this task-local experiment")
    if cfg.get("native_target_tasks") or cfg.get("base_construction", "per_task") != "per_task":
        raise ValueError("Target-informed experiments require all-source tasks on the native target base")
    if cfg.get("patched_attn") or cfg.get("attn_patch_cfg"):
        raise ValueError("Target-informed experiments require the ordinary unpatched ViT checkpoints")
    if cfg.get("load_transported_tvs_dir") and (
        cfg.get("source_lmc_eval") or cfg.get("cross_task_lmc_pairs") or cfg.get("all_task_lmc_eval")
    ):
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
    order = (
        list(range(n)) if seed is None else torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    )
    order = order[: num_batches * source_loader.batch_size]
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
        "indices": order,
        "dataset_identity": repr(_dataset_identity(source_loader.dataset)),
        "requested_batches": num_batches,
        "actual_batches": len(source_batches),
        "batch_size": source_loader.batch_size,
        "sampling_seed": seed,
        "split": "val",
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
        if component == "mlp.c_fc":
            key = f"transformer.resblocks.{pos}.mlp.c_fc.weight"
            return f"visual.{key}" if prefixed else key
        if component in _PACKED_QKV_SLICE:
            # q/k/v share one packed parameter; the caller writes into (and
            # reads out of) a single row slice of it.
            key = f"transformer.resblocks.{pos}.attn.in_proj_weight"
            return f"visual.{key}" if prefixed else key
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

    def mlp_in_module(self, block):
        """``mlp.c_fc``, the MLP's first (pre-GELU) linear projection."""
        inner = block.block if hasattr(block, "block") else block
        return inner.mlp.c_fc


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
        raise NotImplementedError(f"{component!r} is only supported on the vision (stock nn.MultiheadAttention) layout")

    def component_scale_module(self, block, component):
        """Decoder blocks carry no LayerScale; the write is unscaled."""
        return nn.Identity()

    def mlp_in_module(self, block):
        raise NotImplementedError("mlp_input capture is vision-only")


def _layout_for(family_adapter):
    return _VisionLayout() if family_adapter is None else _DecoderLayout(family_adapter)


#: Capture kinds understood by ``capture_tokens``. The ``*_input`` kinds take the
#: hooked module's input, the others its output.  ``attn_input`` is the query
#: argument of the attention call (ln_1's output, shared by q/k/v); ``mlp_input``
#: is the forward-hook input of ``mlp.c_fc`` (shared by c_fc). ``block_input`` is
#: the forward-hook input of the block module itself (the pristine block-level
#: input ``X_j^0`` intra-block backfitting replays against), i.e. the same
#: module as ``"boundary"`` but its input rather than its output. All are
#: vision-layout only.
_CAPTURE_KINDS = frozenset(
    {
        "boundary",
        "c_proj",
        "c_proj_input",
        "attn_proj",
        "attn_proj_input",
        "attn_input",
        "mlp_input",
        "block_input",
    }
)
_INPUT_CAPTURE_KINDS = frozenset({"c_proj_input", "attn_proj_input", "attn_input", "mlp_input", "block_input"})
_ATTN_CAPTURE_KINDS = frozenset({"attn_proj", "attn_proj_input"})
_ATTN_QUERY_KINDS = frozenset({"attn_input"})

#: Which capture kind supplies each component's regression features.
COMPONENT_INPUT_KIND = {
    "attn.out_proj": "attn_proj_input",
    "mlp.c_proj": "c_proj_input",
    "attn.q_proj": "attn_input",
    "attn.k_proj": "attn_input",
    "attn.v_proj": "attn_input",
    "mlp.c_fc": "mlp_input",
}

#: Row index of each packed-QKV component within nn.MultiheadAttention's
#: in_proj_weight [3d,d] / in_proj_bias [3d] (F._in_projection_packed's
#: convention: w_q, w_k, w_v stacked along dim 0).
_PACKED_QKV_SLICE = {"attn.q_proj": 0, "attn.k_proj": 1, "attn.v_proj": 2}


def _component_weight_bias(shim, block, component):
    """Return ``(weight, bias_or_None, row_slice_or_None)`` for one component's
    own linear map, i.e. the raw (pre-LayerScale) projection it applies to its
    captured input: ``A = X @ weight.T + bias``.

    ``row_slice`` is non-``None`` only for the packed q/k/v components, whose
    weight/bias are a row range of the attention's ``in_proj_weight`` /
    ``in_proj_bias`` rather than a standalone parameter.
    """
    inner = shim.block_module(block)
    if component == "mlp.c_proj":
        module = inner.mlp.c_proj
        return module.weight, module.bias, None
    if component == "mlp.c_fc":
        module = inner.mlp.c_fc
        return module.weight, module.bias, None
    if component == "attn.out_proj":
        module = inner.attn.out_proj
        return module.weight, module.bias, None
    if component in _PACKED_QKV_SLICE:
        attn = inner.attn
        if not isinstance(attn, nn.MultiheadAttention):
            raise NotImplementedError(f"{component} requires a stock nn.MultiheadAttention module")
        d = int(attn.embed_dim)
        s = _PACKED_QKV_SLICE[component]
        row_slice = slice(s * d, (s + 1) * d)
        weight = attn.in_proj_weight[row_slice]
        bias = attn.in_proj_bias[row_slice] if attn.in_proj_bias is not None else None
        return weight, bias, row_slice
    raise ValueError(f"Unsupported completion component {component!r}")


def _assert_layerscale_identity(
    shim, block, *, context, requirement="component_target='output_local'/'output_total'"
):
    """Refuse a nontrivial LayerScale under output-target modes (and, via
    ``requirement``, under ``block_split='joint'``, which shares the same
    ``t_out=I`` assumption -- see ``_fit_block_boundary_joint``).

    Output-target fits solve with ``t_out = I``: any nonidentity ``ls_1``/
    ``ls_2`` would silently drop a scale the block actually applies, so the
    target write surface would no longer be unambiguous.
    """
    inner = shim.block_module(block)
    for name in ("ls_1", "ls_2"):
        module = getattr(inner, name, nn.Identity())
        if not isinstance(module, nn.Identity):
            raise ValueError(
                f"{requirement} requires {name}=nn.Identity on {context}; "
                "found a non-identity LayerScale, which the fit's t_out=I would silently ignore"
            )


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
            query,
            key,
            value,
            module.embed_dim,
            module.num_heads,
            module.in_proj_weight,
            module.in_proj_bias,
            module.bias_k,
            module.bias_v,
            module.add_zero_attn,
            module.dropout,
            identity,
            None,
            **shared,
        )
    else:
        rows, _ = F.multi_head_attention_forward(
            query,
            key,
            value,
            module.embed_dim,
            module.num_heads,
            None,
            module.in_proj_bias,
            module.bias_k,
            module.bias_v,
            module.add_zero_attn,
            module.dropout,
            identity,
            None,
            use_separate_proj_weight=True,
            q_proj_weight=module.q_proj_weight,
            k_proj_weight=module.k_proj_weight,
            v_proj_weight=module.v_proj_weight,
            **shared,
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


def _register_capture_hooks(
    layout, blocks, requests: Mapping[str, tuple[int, str]], store, *, family_adapter=None
) -> list:
    """Register one forward hook per capture request, calling ``store(name, tensor)``.

    Every branch (boundary/block_input, the ``nn.MultiheadAttention`` recompute
    path, the self-attention query check, mlp_input, proj, plain ``*_input``
    kinds) is preserved exactly from the historical ``capture_tokens`` body.
    Returns the list of handles the caller must ``.remove()``.
    """
    handles = []
    for key, (index, kind) in requests.items():
        if kind not in _CAPTURE_KINDS:
            raise ValueError(f"Unknown capture kind {kind}")
        block = layout.block_module(blocks[index])
        recompute = False
        query_capture = kind in _ATTN_QUERY_KINDS
        if kind in ("boundary", "block_input"):
            module = block
        elif kind in _ATTN_CAPTURE_KINDS:
            attention = layout.attn_module(blocks[index])
            # A stock nn.MultiheadAttention applies out_proj functionally, so
            # a hook on that submodule would never fire and the "fired once
            # per batch" check below would reject the whole capture. Hook the
            # attention itself and recover the projection's rows exactly.
            recompute = isinstance(attention, nn.MultiheadAttention)
            module = attention if recompute else layout.attn_proj_module(blocks[index])
        elif query_capture:
            if family_adapter is not None:
                raise NotImplementedError("attn_input capture is vision-only")
            module = layout.attn_module(blocks[index])
        elif kind == "mlp_input":
            module = layout.mlp_in_module(blocks[index])
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
        elif query_capture:
            def hook(_mod, args, kwargs, _value, *, name=key):
                query, key_arg, value_arg = _mha_query_key_value(args, kwargs)
                if not (query is key_arg and query is value_arg):
                    raise ValueError(
                        "attn_input capture requires self-attention (query, key and value "
                        "must be the same tensor)"
                    )
                store(name, query)
            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        else:
            def hook(_m, inputs, value, *, name=key, capture_kind=kind):
                store(name, inputs[0] if capture_kind in _INPUT_CAPTURE_KINDS else value)
            handles.append(module.register_forward_hook(hook))
    return handles


def iter_capture_tokens(
    model, batches, requests: Mapping[str, tuple[int, str]], device, *, family_adapter=None, store_device="cpu"
):
    """Yield one ``{key: tensor}`` dict per batch instead of accumulating all of them.

    Same hook/module resolution as ``capture_tokens`` (see ``_register_capture_hooks``),
    but hooks are registered and removed around EACH batch rather than once for the
    whole sweep -- this lets several generators over the same model object run in
    lockstep (e.g. one per source/finetuned/target model) without cross-firing.
    ``store_device="cpu"`` keeps the historical ``tokens.float().cpu().clone()`` store
    op; any other value stores via ``tokens.float().to(store_device).clone()``.
    Restores the model's device/train mode on normal completion, early
    ``.close()``, garbage collection, or an exception raised mid-sweep.
    """
    layout = _layout_for(family_adapter)
    blocks = layout.blocks(model)
    training = model.training
    original_device = next(model.parameters()).device
    try:
        model.to(device).eval()
        for batch in batches:
            batch_size = layout.batch_size(batch)
            values: dict[str, list[torch.Tensor]] = {key: [] for key in requests}

            def store(name, tensor, *, _batch_size=batch_size, _values=values):
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                tokens = _to_tokens(tensor.detach(), batch_size=_batch_size)
                if store_device == "cpu":
                    tokens = tokens.float().cpu().clone()
                else:
                    tokens = tokens.float().to(store_device).clone()
                _values[name].append(tokens)

            handles = _register_capture_hooks(layout, blocks, requests, store, family_adapter=family_adapter)
            try:
                with torch.no_grad():
                    layout.forward(model, batch, device)
            finally:
                for handle in handles:
                    handle.remove()
            if any(len(v) != 1 for v in values.values()):
                raise RuntimeError("A requested activation hook did not fire exactly once per batch")
            yield {key: v[0] for key, v in values.items()}
    finally:
        model.to(original_device).train(training)


@torch.no_grad()
def capture_tokens(model, batches, requests: Mapping[str, tuple[int, str]], device, *, family_adapter=None):
    """Capture B,T,D tensors, releasing hooks and restoring placement on errors.

    family_adapter=None keeps the original CLIP paths; passing one selects the
    HF-decoder equivalents (see _DecoderLayout). The "c_proj" capture kinds keep
    their names on both paths -- on a decoder they resolve to mlp.down_proj,
    which plays the same residual-writing role.
    """
    output = {key: [] for key in requests}
    for batch_values in iter_capture_tokens(model, batches, requests, device, family_adapter=family_adapter):
        for key, tensor in batch_values.items():
            output[key].append(tensor)
    if any(len(values) != len(batches) for values in output.values()):
        raise RuntimeError("A requested activation hook did not fire exactly once per batch")
    return output


def capture_block_gradients(
    model,
    batches,
    requests: Mapping[str, int],
    recipe,
    device,
    *,
    family_adapter=None,
):
    """Capture block-boundary gradients ``dL/dT_j`` for the gradient-aligned
    Procrustes source (``DirectResidualConfig.procrustes_source="gradient"``).

    For every batch, runs one forward+backward pass of
    ``recipe(model, batch) -> (scalar_loss, named_params)`` (a
    ``models.grad_recipes.GradRecipe``, e.g. ``clip_contrastive_recipe`` --
    mean-reduced CE, exactly BiCo's statistic) and stores each requested
    resblock's ``grad_output[0]`` -- the gradient of the loss with respect to
    that block's OUTPUT, i.e. the same tensor `capture_tokens`'s ``"boundary"``
    kind captures in the forward pass -- as CPU float32, in the same
    `_to_tokens` ``[B,T,D]`` token layout. ``requests`` maps an arbitrary key
    to a resblock index (there is only one capture kind here, so no
    ``(index, kind)`` pair is needed).

    Mirrors `rebase.methods.bico._collect_batch`'s forward+backward pattern:
    ``model.zero_grad(set_to_none=True)`` both before and after each batch's
    backward pass, so no parameter gradient is ever retained. Every
    parameter's ``requires_grad`` is temporarily forced ``True`` for the
    capture (so the backward graph reaches every block even when the model is
    normally used frozen/inference-only) and restored -- together with
    ``training`` mode and device placement -- on return, including on any
    exception. This function never mutates a parameter's *value*, only reads
    gradients off the graph; callers that want a belt-and-braces check can
    compare ``model.state_dict()`` before/after (see
    ``tests/test_direct_residual_gradient_procrustes.py``).

    Vision only: raises `NotImplementedError` for a non-``None``
    ``family_adapter``, since block-boundary gradient capture has no decoder
    equivalent yet.
    """
    if family_adapter is not None:
        raise NotImplementedError("capture_block_gradients is vision-only")
    layout = _layout_for(None)
    blocks = layout.blocks(model)
    training = model.training
    original_device = next(model.parameters()).device
    requires_grad_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    output: dict[str, list[torch.Tensor]] = {key: [] for key in requests}
    current_batch = [0]
    handles = []
    try:
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(True)

        def store(name, tensor):
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            tokens = _to_tokens(tensor.detach(), batch_size=current_batch[0])
            output[name].append(tokens.float().cpu().clone())

        for key, index in requests.items():
            block = layout.block_module(blocks[index])

            def hook(_module, _grad_input, grad_output, *, name=key):
                if grad_output is None or grad_output[0] is None:
                    raise RuntimeError(f"Block gradient hook {name!r} produced no output gradient")
                store(name, grad_output[0])

            handles.append(block.register_full_backward_hook(hook))

        for batch in batches:
            current_batch[0] = layout.batch_size(batch)
            model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(True):
                loss, _ = recipe(model, batch)
                if loss.dim() > 0:
                    loss = loss.sum()
                loss.backward()
            model.zero_grad(set_to_none=True)
        if any(len(values) != len(batches) for values in output.values()):
            raise RuntimeError("A requested block gradient hook did not fire exactly once per batch")
    finally:
        for handle in handles:
            handle.remove()
        for name, p in model.named_parameters():
            p.requires_grad_(requires_grad_flags[name])
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
    base = capture_tokens(source_base, sb, req, device, family_adapter=family_adapter)
    ft = capture_tokens(source_ft, sb, req, device, family_adapter=family_adapter)
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
    target = capture_tokens(target_base, tb, target_requests, device, family_adapter=family_adapter)

    # Keep source banks by original index for the all-position path.  The
    # inserted path also materializes the historical dictionaries immediately,
    # preserving its byte-compatible downstream behavior.
    source_base_outputs = {int(i): value for i, value in base.items() if i.isdigit()}
    source_ft_outputs = {int(i): value for i, value in ft.items() if i.isdigit()}
    source_base_cproj_inputs = (
        {int(i.split(".", 1)[0]): value for i, value in base.items() if i.endswith(".c_proj_input")}
        if capture_joint
        else {}
    )
    source_ft_cproj_inputs = (
        {int(i.split(".", 1)[0]): value for i, value in ft.items() if i.endswith(".c_proj_input")}
        if capture_joint
        else {}
    )
    source_base_cproj_outputs = (
        {int(i.split(".", 1)[0]): value for i, value in base.items() if i.endswith(".c_proj_output")}
        if capture_joint
        else {}
    )
    source_ft_cproj_outputs = (
        {int(i.split(".", 1)[0]): value for i, value in ft.items() if i.endswith(".c_proj_output")}
        if capture_joint
        else {}
    )
    target_outputs_by_position = {int(i): value for i, value in target.items() if i.isdigit()}
    target_cproj_inputs_by_position = (
        {int(i.split(".", 1)[0]): value for i, value in target.items() if i.endswith(".c_proj_input")}
        if capture_joint
        else {}
    )
    target_cproj_outputs_by_position = (
        {int(i.split(".", 1)[0]): value for i, value in target.items() if i.endswith(".c_proj_output")}
        if capture_joint
        else {}
    )
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


def capture_source_component_references(
    source_base,
    source_ft,
    source_batches,
    source_indices,
    components,
    device,
    *,
    family_adapter=None,
    capture_ft_inputs=False,
):
    """Capture source-base component input banks and both endpoints' weight
    slices, for the ``component_target='output_local'``/``'output_total'`` fit.

    Shared by any caller that needs component-specific (rather than block-
    boundary) targets -- currently Direct Residual
    (``direct_residual.capture_paired_boundary_activations``) -- so it lives
    here alongside the capture-kind machinery it depends on
    (``COMPONENT_INPUT_KIND``, ``_component_weight_bias``) rather than being
    duplicated at each call site. ARIADNE's own ``_capture_residual_references``
    does not call this: Proposal 1 only ever fits ``component_target=
    'block_boundary'``.

    The source **base** model's own inputs are always captured: the
    ``output_local`` target is ``A(X^s0, W) - A(X^s0, W0)``, which never
    evaluates either model on a fine-tuned input, and ``output_local``'s
    Procrustes alignment (``A^{s0}`` vs the target's own ``A^{t0}``) is shared
    unchanged by ``output_total``. ``source_batches`` must already be the
    paired calibration source batches (e.g. from ``paired_calibration``);
    this function runs no calibration pairing of its own.

    ``capture_ft_inputs=True`` (``component_target='output_total'`` only)
    additionally captures the source **fine-tuned** model's own component
    inputs ``X^{s1}`` on the same ``source_batches`` -- needed because
    ``output_total``'s target is ``A(X^{s1}, W^{ft}) - A(X^{s0}, W^{base})``,
    which evaluates the fine-tuned endpoint on the fine-tuned model's own
    (possibly upstream-drifted) input rather than reusing ``X^{s0}``. When
    ``False`` (the default, ``output_local``'s case), no extra forward pass
    runs and the third return value is ``{}``.

    Returns ``(source_component_inputs, source_component_weights,
    source_component_inputs_ft)``, all keyed by the entries of
    ``source_indices``; ``source_component_inputs_ft`` is empty unless
    ``capture_ft_inputs`` is set.
    """
    if family_adapter is not None:
        raise NotImplementedError("component_target='output_local'/'output_total' is vision-only")
    layout_shim = _layout_for(family_adapter)
    needed_kinds = sorted({COMPONENT_INPUT_KIND[c] for c in components})
    component_req = {f"{i}.{kind}": (i, kind) for i in source_indices for kind in needed_kinds}
    component_base = capture_tokens(source_base, source_batches, component_req, device, family_adapter=family_adapter)
    component_ft = None
    if capture_ft_inputs:
        component_ft = capture_tokens(source_ft, source_batches, component_req, device, family_adapter=family_adapter)
    source_component_inputs, source_component_weights, source_component_inputs_ft = {}, {}, {}
    for i in source_indices:
        source_component_inputs[i] = {
            kind: [t.clone() for t in component_base[f"{i}.{kind}"]] for kind in needed_kinds
        }
        if capture_ft_inputs:
            source_component_inputs_ft[i] = {
                kind: [t.clone() for t in component_ft[f"{i}.{kind}"]] for kind in needed_kinds
            }
        base_block = layout_shim.blocks(source_base)[i]
        ft_block = layout_shim.blocks(source_ft)[i]
        _assert_layerscale_identity(layout_shim, base_block, context=f"source base block {i}")
        _assert_layerscale_identity(layout_shim, ft_block, context=f"source FT block {i}")
        weights = {}
        for component in components:
            base_w, base_b, _slice = _component_weight_bias(layout_shim, base_block, component)
            ft_w, ft_b, _slice2 = _component_weight_bias(layout_shim, ft_block, component)
            weights[component] = {
                "base_weight": base_w.detach().float().cpu().clone(),
                "base_bias": None if base_b is None else base_b.detach().float().cpu().clone(),
                "ft_weight": ft_w.detach().float().cpu().clone(),
                "ft_bias": None if ft_b is None else ft_b.detach().float().cpu().clone(),
            }
        source_component_weights[i] = weights
    return source_component_inputs, source_component_weights, source_component_inputs_ft


def projection_transforms(prepared, layout, *, target_scope="inserted", family_adapter=None):
    _validate_target_informed_layout(layout, target_scope=target_scope)
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


def _validate_target_informed_layout(layout, *, target_scope, target_trajectory=None):
    """Validate the layout contract before a target-informed solve consumes it.

    Extension layouts historically use ``inserted_blocks`` and the odd-position
    ancestry convention.  Reduction layouts have no inserted positions: every
    final target block represents a collapsed source span and its
    ``source_orig_idx`` is the span-end boundary.  Keeping this check at the
    runtime boundary prevents a shrink run from silently falling through an
    extension-only scope or trajectory.

    Layouts produced before the explicit ``direction`` field was added are
    treated as extension layouts for backwards compatibility with cached test
    fixtures and old direct-target caches.
    """
    direction = str(layout.get("direction", "extend"))
    if direction not in {"extend", "shrink"}:
        raise ValueError(
            f"Target-informed completion requires layout.direction to be 'extend' or 'shrink'; got {direction!r}"
        )
    if direction != "shrink":
        return
    if target_scope != "all":
        raise ValueError(
            "Shrink direct_target completion requires target_scope='all': a reduction has no inserted blocks to address"
        )
    if target_trajectory is not None and target_trajectory != "step":
        raise ValueError(
            "Shrink direct_target completion currently requires target_trajectory='step'; "
            "interpolate has no span-aware reduction semantics"
        )
    entries = layout.get("final_blocks")
    if entries is None:
        raise ValueError("Shrink direct_target completion requires realized layout final_blocks")
    for row in entries:
        span = row.get("span_orig_idxs")
        if not span:
            raise ValueError("Shrink direct_target layout entries must record non-empty span_orig_idxs")
        if int(row.get("source_orig_idx", -1)) != int(span[-1]):
            raise ValueError(
                "Shrink direct_target layout source_orig_idx must equal the span's terminal source boundary"
            )


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

    ``components`` defaults to the historical MLP-only behaviour. Direct mode
    can also fit ``attn.out_proj``, whose decoder analogue (``self_attn.o_proj``)
    is exactly as bias-free as ``down_proj`` -- the caller must pass
    ``order_components(config.components)`` when a run's completion config
    requests more than the default, or completion raises
    "missing_bias='materialize' requires ... to exist on the target" for the
    write surface this function never touched.

    Returns the keys it added, so a run can record that its checkpoint carries
    parameters the stock architecture does not.
    """
    shim = _layout_for(family_adapter)
    entries = layout.get("final_blocks") or layout.get("inserted_blocks") or ()
    added = []
    for row in entries:
        pos = int(row["position"])
        for component in components:
            weight_key = shim.component_key(pos, component, prefixed=True)
            bias_key = f"{weight_key[: -len('.weight')]}.bias"
            if bias_key in target_base_state:
                continue
            out_features = int(target_base_state[weight_key].shape[0])
            _materialize_zero_bias(target_model, bias_key, out_features)
            target_base_state[bias_key] = torch.zeros(out_features, dtype=target_base_state[weight_key].dtype)
            added.append(bias_key)
    return added


@torch.no_grad()
def complete_residuals(
    target_model,
    target_base_state,
    baseline_delta,
    references,
    transforms,
    layout,
    target_loader,
    *,
    config: ResidualCompletionConfig,
    device,
    family_adapter=None,
):
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
    for name, values in (
        ("desired", desired),
        ("target_base_outputs", target_outputs),
        ("maps", maps),
        ("transforms", transforms),
    ):
        if set(values) != expected_positions:
            raise ValueError(
                f"Residual completion {name} keys do not exactly match realized target positions: "
                f"expected={sorted(expected_positions)}, found={sorted(values)}"
            )
    meta = references["calibration"]
    if repr(_dataset_identity(target_loader.dataset)) != meta["dataset_identity"]:
        raise ValueError("Cached reference images do not match this task's validation dataset")
    batches = list(
        DataLoader(
            Subset(target_loader.dataset, meta["indices"]),
            batch_size=meta["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=target_loader.collate_fn,
        )
    )
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
                target_model,
                batches,
                {"h": (pos, "c_proj_input"), "out": (pos, "boundary")},
                device,
                family_adapter=family_adapter,
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
            correction, diag = stats.solve(
                ridge_relative=config.ridge_relative,
                ridge_estimator=config.ridge_estimator,
                exact_form=config.exact_form,
            )
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
def _fit_direct_target_position(
    target_model,
    current_state: dict[str, torch.Tensor],
    pos: int,
    source_coordinate: float,
    desired_batches: list,
    target_output_batches: list,
    batches: list,
    components: tuple,
    config,
    device,
    family_adapter=None,
    *,
    assert_pristine_effect: bool = False,
) -> tuple[dict, list]:
    """Fit one target position's residual-writing components (Part A extraction).

    This is the per-position solver body ``complete_residuals_direct`` has
    always executed, pulled out unchanged so a caller with a wholly different
    notion of "which source coordinate feeds this position" -- ARIADNE's
    ancestry row (``row["source_orig_idx"]``) or Direct Residual's flat
    ``DiscreteLayerPairing`` entry -- can share the identical, tested solve.
    Nothing here inspects a realized-extension layout, an ``entries`` list, or
    a ``block_kind``: those stay with whichever caller actually has them.
    ``source_coordinate`` is carried only into the returned diagnostics (as a
    plain float, generic across both callers' provenance schemes); it plays no
    role in the fit itself, exactly as ``row["source_orig_idx"]`` did not in
    the code this was extracted from.

    ``current_state`` and ``target_model`` are mutated in place: each
    component's fitted correction is folded into ``current_state`` before the
    next component is captured, and ``target_model`` is reloaded from it
    whenever ``config.cascade_order != "independent"`` -- the same mount
    timing ``complete_residuals_direct`` has always used, whether the cascade
    crosses component boundaries within one position or (via the caller
    re-invoking this function) block boundaries across positions.
    ``assert_pristine_effect=True`` raises if the very first component's
    measured pre-fit effect is nonzero; pass it only when ``target_model`` is
    known to still be at its untouched base entering this call.

    Returns ``(position_corrections, block_rows)``.  ``position_corrections``
    holds only this position's fitted weight/bias keys, in target coordinates
    at unit strength.  ``block_rows`` mirrors the historical per-component
    diagnostic dict shape *minus* the ARIADNE-only fields (``scope``,
    ``trajectory``, ``target_coordinate``, ``block_kind``, ``source_orig_idx``)
    that only ``complete_residuals_direct`` has to offer; it carries
    ``source_coordinate`` instead, and the caller is responsible for adding
    back whatever provenance fields its own diagnostic contract promises.
    """
    shim = _layout_for(family_adapter)
    position_corrections: dict[str, torch.Tensor] = {}
    block_rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        key = shim.component_key(pos, component, prefixed=True)
        captured = capture_tokens(
            target_model,
            batches,
            {"h": (pos, COMPONENT_INPUT_KIND[component]), "out": (pos, "boundary")},
            device,
            family_adapter=family_adapter,
        )
        width = int(current_state[key].shape[0])
        identity_out = torch.eye(width, dtype=torch.float32)
        scale_module = shim.component_scale_module(shim.blocks(target_model)[pos], component)
        effective_out = identity_out
        if not isinstance(scale_module, nn.Identity):
            scale = getattr(scale_module, "gamma", None)
            if scale is None or scale.ndim != 1 or scale.shape[0] != width:
                raise ValueError("Unsupported non-diagonal target LayerScale")
            # diag(gamma): the residual stream receives gamma * proj(h), so the
            # fit must predict through that scaling exactly as the
            # transport-aware path folds it into t_out. ls_1 scales the
            # attention write, ls_2 the MLP write.
            effective_out = identity_out * scale.detach().cpu().float().unsqueeze(0)
        # See the identical comment in complete_residuals(): the Gram
        # accumulation and eigendecomposed solve are the expensive part, so
        # they run on the model's device; the fitted correction/bias are moved
        # back to CPU right after solve() below.
        stats = ResidualSufficientStatistics(device=device)
        desired_sq = 0.0
        effect_sq = 0.0
        for h, out, desired_batch, base_out in zip(
            captured["h"], captured["out"], desired_batches, target_output_batches, strict=True
        ):
            effect = out - base_out
            error = desired_batch - effect
            desired_sq += float((desired_batch.double() ** 2).sum().item())
            effect_sq += float((effect.double() ** 2).sum().item())
            stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)
        if assert_pristine_effect and component_index == 0:
            # E_j == D_j at the very first fit: the temporary model is still
            # the untouched target base there, so the current effect is
            # identically zero. A nonzero effect here means a stale reference
            # bank or a mutated base, not a small numerical drift.
            if effect_sq > 1e-12 * max(desired_sq, 1.0):
                raise RuntimeError(
                    "Direct completion started from a target model that is not the native "
                    f"base: nonzero pre-fit effect at position {pos} (||T-T0||^2={effect_sq:.3e})"
                )
        correction, diag = stats.solve(
            ridge_relative=config.ridge_relative,
            ridge_estimator=config.ridge_estimator,
            exact_form=config.exact_form,
        )
        correction = correction.cpu()
        diag["bias_correction"] = diag["bias_correction"].cpu()
        if block_rows:
            # This capture happened after the previous component was mounted,
            # so its residual is that component's *true* post-mount residual.
            # For attn.out_proj the solver's own residual_norm_after is only a
            # linear prediction; this is the measured one.
            block_rows[-1]["measured_residual_norm_after"] = diag["residual_norm_before"]
        if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
            raise RuntimeError("Direct residual completion produced an invalid projection")
        position_corrections[key] = correction
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
            position_corrections[bias_key] = bias_correction
            current_state[bias_key] = current_state[bias_key] + bias_delta
        # Mounted before the next component captures, which is what makes the
        # intra-block cascade real rather than two independent fits.
        # cascade_order="independent" deliberately skips the mount: the model
        # stays at the pristine base for every fit, so each block sees
        # E_j == D_j and no correction can move another's target.
        if config.cascade_order != "independent":
            target_model.load_state_dict(current_state, strict=True)
        desired_norm = desired_sq**0.5
        block_rows.append(
            {
                "mode": "direct_target",
                "component": component,
                "position": pos,
                "source_coordinate": float(source_coordinate),
                "desired_norm": desired_norm,
                "effect_before_norm": effect_sq**0.5,
                # r_j: the fraction of the desired local effect still missing,
                # before and after this fit. See the identical comment this
                # was extracted from in complete_residuals_direct's history
                # (git blame) for why mlp.c_proj's "after" value is exact
                # while attn.out_proj's needs the measured re-capture above.
                "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
                "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                **diag,
            }
        )
    return position_corrections, block_rows


def _realization_diagnostic_fields(
    h_batches, correction, bias_for_pred, effective_out, weight_before, desired_norm, residual_norm_after, *, stats=None
):
    """Analysis-only per-component fit fields, gated on ``realization_
    diagnostics`` (never read by any fit -- see the callers). Shared by every
    block_boundary fit path (``_fit_all_positions_independent``,
    ``_fit_block_boundary_backfit``) so the schema matches
    ``_fit_component_outputs_from_contributions``'s own (output_local) fields
    exactly: ``fit_relative_residual``, ``target_norm``, ``update_norm``,
    ``relative_update_norm`` (denominator = the pre-correction weight slice),
    ``realized_target_norm_ratio``.

    ``realized_target_norm_ratio`` is computed EXACTLY from the accumulated
    fit (``H @ correction^T + bias``, pushed through the SAME ``effective_out``
    -- LayerScale -- the solve itself used), reusing the already-resident
    ``h_batches`` bank rather than a second capture sweep.
    """
    update_norm = float(torch.linalg.norm(correction).item())
    weight_norm = float(torch.linalg.norm(weight_before).item())
    if h_batches is None:
        # Streaming path: no resident input bank, so the prediction norm comes from the
        # solver's accumulated statistics (exact up to floating-point summation order).
        if stats is None:
            raise ValueError("realization fields need either h_batches or the accumulated stats")
        pred_sq = _realized_pred_sq_from_stats(stats, correction, bias_for_pred, effective_out)
    else:
        pred_sq = 0.0
        for h in h_batches:
            pred = h.reshape(-1, h.shape[-1]).double() @ correction.double().T + bias_for_pred.double()
            pred = pred @ effective_out.double()
            pred_sq += float((pred**2).sum().item())
    return {
        "fit_relative_residual": (residual_norm_after / desired_norm) if desired_norm else 0.0,
        "target_norm": desired_norm,
        "update_norm": update_norm,
        "relative_update_norm": update_norm / (weight_norm + 1e-12),
        "realized_target_norm_ratio": (pred_sq**0.5) / (desired_norm + 1e-12),
    }


@torch.no_grad()
def _fit_all_positions_independent(
    target_model,
    current_state: dict[str, torch.Tensor],
    positions: list[int],
    source_coordinates: dict[int, float],
    desired_batches: dict[int, list],
    target_output_batches: dict[int, list],
    batches: list,
    components: tuple,
    config,
    device,
    family_adapter=None,
) -> dict[int, tuple[dict, list]]:
    """Fit every ``(position, component)`` pair for ``cascade_order='independent'``
    from ONE shared target forward sweep instead of one sweep per pair.

    Under ``cascade_order="independent"`` the target model is never mutated
    between fits -- the only mount site
    (``if config.cascade_order != "independent": target_model.load_state_dict(...)``
    in ``_fit_direct_target_position``) is unconditionally skipped, both across
    positions and across components within one position. Every ``(position,
    component)`` pair therefore observes the identical, pristine
    ``target_model`` state that ``current_state`` already describes, which
    makes ``_fit_direct_target_position``'s per-pair ``capture_tokens`` calls
    (one full calibration sweep each, up to ``len(positions) *
    len(components)`` of them) redundant: they all capture from the same
    model. This function captures once and reuses the banks for every solve.

    ``capture_tokens`` already accepts a combined ``requests`` dict spanning
    many ``(position, kind)`` pairs and turns it into one set of hooks fired
    by one sweep over ``batches`` (see its docstring/implementation); nothing
    below it needed to change; this function is only a restructuring of the
    *caller* side. One ``"out"`` (boundary) request is registered per
    position, shared by every component at that position -- the block's own
    boundary output does not depend on which component's input is being
    fitted -- and one ``"h"`` request per ``(position, component)`` pair for
    that component's regression features
    (``COMPONENT_INPUT_KIND[component]``).

    Per-``(position, component)`` solving is otherwise byte-identical to
    ``_fit_direct_target_position``'s independent-mode body: same
    ``ResidualSufficientStatistics`` online accumulation (already a streaming
    accumulator; nothing here changes how it accumulates, only how many
    forward sweeps feed it), same ridge solve, same ``missing_bias`` handling,
    same ``block_rows`` diagnostic fields -- including the (under independent
    mode, vacuous but historically populated) ``measured_residual_norm_after``
    field on every component but the position's last: since nothing is ever
    mounted, that field is simply the next component's own pre-fit residual,
    identical in both the old per-pair-capture path and this one.

    The pristine-effect assertion is strengthened relative to
    ``_fit_direct_target_position``: that function can only assert it for
    ``component_index == 0`` of whichever position the caller nominates as
    "first" (a real cascade only ever *starts* pristine). Under true
    independent semantics there is no privileged "first" position -- every
    ``(position, component)`` pair sees the pristine base -- so this function
    asserts it unconditionally for all of them. This is a strictly stronger
    correctness check with no effect on the fitted values themselves: it can
    only ever raise on a bug (a stale reference bank or a mutated base), never
    change a number that was going to be returned.

    Intra-position "replay" (the user's brief step 6, mounting attn.out_proj
    locally before fitting mlp.c_proj) is deliberately NOT implemented: under
    independent mode there is no intra-position mount to replay in the first
    place (the same skipped-mount conditional gates it), so there is no
    sequential attention->MLP effect for a replay to preserve. Building it
    would change independent-mode's numerics, not merely speed it up.

    Returns ``{position: (position_corrections, block_rows)}``, matching what
    ``len(positions)`` separate ``_fit_direct_target_position`` calls would
    each have returned for the identical inputs.
    """
    if config.cascade_order != "independent":
        raise ValueError("_fit_all_positions_independent requires cascade_order='independent'")
    shim = _layout_for(family_adapter)
    requests: dict[str, tuple[int, str]] = {}
    for pos in positions:
        requests[f"{pos}.out"] = (pos, "boundary")
        for component in components:
            requests[f"{pos}.{component}.h"] = (pos, COMPONENT_INPUT_KIND[component])
    captured = capture_tokens(target_model, batches, requests, device, family_adapter=family_adapter)

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        out_batches = captured[f"{pos}.out"]
        position_corrections: dict[str, torch.Tensor] = {}
        block_rows: list[dict[str, Any]] = []
        for component in components:
            key = shim.component_key(pos, component, prefixed=True)
            h_batches = captured[f"{pos}.{component}.h"]
            width = int(current_state[key].shape[0])
            effective_out = _component_effective_out(shim, target_model, pos, component, width)
            stats = ResidualSufficientStatistics(device=device)
            desired_sq = 0.0
            effect_sq = 0.0
            for h, out, desired_batch, base_out in zip(
                h_batches, out_batches, desired_batches[pos], target_output_batches[pos], strict=True
            ):
                effect = out - base_out
                error = desired_batch - effect
                desired_sq += float((desired_batch.double() ** 2).sum().item())
                effect_sq += float((effect.double() ** 2).sum().item())
                stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)
            _finalize_independent_component(
                stats,
                desired_sq,
                effect_sq,
                pos,
                component,
                key,
                current_state,
                effective_out,
                config,
                source_coordinates,
                position_corrections,
                block_rows,
                h_batches=h_batches,
            )
        results[pos] = (position_corrections, block_rows)
    return results


def _component_effective_out(shim, target_model, pos, component, width) -> torch.Tensor:
    """Build the ``effective_out`` (identity, or LayerScale-diagonal) a component's
    solve is pushed through, exactly as ``_fit_all_positions_independent`` inlined it.
    """
    identity_out = torch.eye(width, dtype=torch.float32)
    scale_module = shim.component_scale_module(shim.blocks(target_model)[pos], component)
    if isinstance(scale_module, nn.Identity):
        return identity_out
    scale = getattr(scale_module, "gamma", None)
    if scale is None or scale.ndim != 1 or scale.shape[0] != width:
        raise ValueError("Unsupported non-diagonal target LayerScale")
    return identity_out * scale.detach().cpu().float().unsqueeze(0)


def _finalize_independent_component(
    stats,
    desired_sq,
    effect_sq,
    pos,
    component,
    key,
    current_state,
    effective_out,
    config,
    source_coordinates,
    position_corrections,
    block_rows,
    h_batches=None,
):
    """Everything after one component's accumulation loop in
    ``_fit_all_positions_independent``: the pristine-effect check, the ridge
    solve, state/bookkeeping updates, missing-bias handling, and the block_row
    diagnostic (plus optional realization diagnostics). Mutates
    ``position_corrections``, ``current_state`` and ``block_rows`` in place.
    """
    # See the docstring: under independent mode this holds for every
    # (position, component) pair, not only a historically-first one.
    if effect_sq > 1e-12 * max(desired_sq, 1.0):
        raise RuntimeError(
            "Direct completion started from a target model that is not the native "
            f"base: nonzero pre-fit effect at position {pos} (||T-T0||^2={effect_sq:.3e})"
        )
    correction, diag = stats.solve(
        ridge_relative=config.ridge_relative,
        ridge_estimator=config.ridge_estimator,
        exact_form=config.exact_form,
    )
    correction = correction.cpu()
    diag["bias_correction"] = diag["bias_correction"].cpu()
    weight_before = current_state[key].detach().clone()
    if block_rows:
        # Same bookkeeping as _fit_direct_target_position: the previous
        # component's row records this component's own pre-fit residual
        # under its historical name. Under independent mode nothing
        # mounted in between, so this is not a "post-mount" measurement
        # -- see the docstring -- but it is the identical value the old
        # per-pair-capture path would have recorded.
        block_rows[-1]["measured_residual_norm_after"] = diag["residual_norm_before"]
    if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
        raise RuntimeError("Direct residual completion produced an invalid projection")
    position_corrections[key] = correction
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
        position_corrections[bias_key] = bias_correction
        current_state[bias_key] = current_state[bias_key] + bias_delta
    desired_norm = desired_sq ** 0.5
    block_row = {
        "mode": "direct_target",
        "component": component,
        "position": pos,
        "source_coordinate": float(source_coordinates[pos]),
        "desired_norm": desired_norm,
        "effect_before_norm": effect_sq ** 0.5,
        "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
        "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
        "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
        **diag,
    }
    if bool(getattr(config, "realization_diagnostics", False)):
        bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
        block_row.update(
            _realization_diagnostic_fields(
                h_batches, correction, bias_for_pred, effective_out, weight_before, desired_norm,
                diag["residual_norm_after"], stats=stats,
            )
        )
    block_rows.append(block_row)


def _replay_block_components(shim, local_block, x_batches, components, device):
    """Run ``local_block`` directly (no full-model forward) over ``x_batches``,
    hooking each of ``components`` (``attn.out_proj`` / ``mlp.c_proj`` only) for
    its own input, plus the block's own output.

    ``local_block`` is a standalone (already-unwrapped) block module -- e.g. a
    ``copy.deepcopy`` of ``shim.block_module(...)`` -- called with a single
    positional tensor, mirroring how ``_VisionLayout.forward``/``_encode_image``
    invoke every block during an ordinary full-model sweep. Returns
    ``(component_h_batches, out_batches)``, both CPU float32, same convention as
    ``capture_tokens``.
    """
    handles = []
    component_h: dict[str, list[torch.Tensor]] = {c: [] for c in components}
    try:
        if "attn.out_proj" in components:
            attn = shim.attn_module(local_block)
            if isinstance(attn, nn.MultiheadAttention):
                # A stock nn.MultiheadAttention applies out_proj functionally
                # (see capture_tokens' own docstring): hook the attention
                # itself and recompute the rows it fed the projection.
                def attn_hook(mod, args, kwargs, value, *, store=component_h["attn.out_proj"]):
                    rows = _stock_mha_out_proj_input(mod, args, kwargs)
                    _verify_recomputed_attention_input(mod, rows, value)
                    store.append(rows.detach().float().cpu().clone())

                handles.append(attn.register_forward_hook(attn_hook, with_kwargs=True))
            else:
                # A plain (non-MHA) attention wrapper calls out_proj as an
                # ordinary submodule, so a direct forward hook on it fires
                # normally -- same fallback capture_tokens itself takes.
                proj = shim.attn_proj_module(local_block)

                def proj_hook(_m, inputs, _value, *, store=component_h["attn.out_proj"]):
                    store.append(inputs[0].detach().float().cpu().clone())

                handles.append(proj.register_forward_hook(proj_hook))
        if "mlp.c_proj" in components:
            proj = shim.proj_module(local_block)

            def proj_hook(_m, inputs, _value, *, store=component_h["mlp.c_proj"]):
                store.append(inputs[0].detach().float().cpu().clone())

            handles.append(proj.register_forward_hook(proj_hook))
        out_batches = []
        for x in x_batches:
            out = local_block(x.to(device))
            out_batches.append(out.detach().float().cpu().clone())
        if any(len(v) != len(x_batches) for v in component_h.values()):
            raise RuntimeError("A backfit component hook did not fire exactly once per batch")
        return component_h, out_batches
    finally:
        for h in handles:
            h.remove()


def _mount_component(shim, local_block, component, weight, bias):
    module = shim.attn_proj_module(local_block) if component == "attn.out_proj" else shim.proj_module(local_block)
    module.weight.data.copy_(weight.to(module.weight.dtype).to(module.weight.device))
    if bias is not None:
        if module.bias is None:
            raise RuntimeError(f"{component} has no bias parameter to mount a fitted bias correction onto")
        module.bias.data.copy_(bias.to(module.bias.dtype).to(module.bias.device))


def _mount_all_deltas(shim, local_block, order, base, deltas):
    """Mount ``base[c] + deltas[c]`` for every ``c in order`` onto ``local_block``."""
    for c in order:
        w, b = deltas[c]
        _mount_component(shim, local_block, c, base[c][0] + w, None if base[c][1] is None else base[c][1] + b)


def _backfit_data_fit_sq(shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas):
    """``||D_j - (block_j(X_j^0; deltas mounted) - T_j^0)||_F^2`` -- the (un-linearized,
    measured on the actual local block replay) data-fit term of the safeguarded
    backfit objective ``J(Delta)``. See ``_fit_block_boundary_backfit``'s docstring.

    ``local_block64`` is a dedicated float64 replica of the block (see
    ``_fit_block_boundary_backfit``), used ONLY for this measurement -- never
    for the candidate sub-solves, whose CPU-float32 delta convention is
    unaffected. The accept/backtrack decision this feeds compares two J
    values that can be arbitrarily close near a fixed point (a near-singular
    Gauss-Seidel design, e.g., can leave genuine per-sweep improvements far
    below float32's ~1e-7 relative precision); evaluating in the module's own
    float32 would let ordinary float32 rounding noise in the forward pass
    flip the accept/reject decision and stall the sweep well short of
    convergence -- exactly the kind of numerical noise a *safeguard*
    (whose entire job is a reliable ``<`` comparison) must not be sensitive
    to. ``_mount_component``'s own ``.to(module.weight.dtype)`` upcasts the
    float32 base/delta tensors to float64 automatically since
    ``local_block64``'s parameters are float64.
    """
    _mount_all_deltas(shim, local_block64, order, base, deltas)
    t_all = [local_block64(x.to(device).double()).detach().cpu().clone() for x in x_batches]
    return sum(
        float(((d.double() - (t - t0.double())) ** 2).sum().item())
        for d, t, t0 in zip(d_batches, t_all, t0_batches, strict=True)
    )


def _backfit_ridge_penalty(order, deltas, lambdas):
    """``sum_c lambda_c * ||Delta W_c||_F^2`` -- the exact penalty
    ``ResidualSufficientStatistics.solve`` minimizes for the weight (the bias is
    fit unpenalized; see ``_fit_block_boundary_backfit``'s docstring), evaluated
    with each component's FROZEN round-1 ``lambda_c`` from ``lambdas``. A
    component with no ``lambdas`` entry yet (never solved) contributes 0, which
    is always exact since its delta is still zero at that point.
    """
    total = 0.0
    for c in order:
        w, _ = deltas[c]
        lam = lambdas.get(c)
        if lam:
            total += lam * float((w.double() ** 2).sum().item())
    return total


def _backfit_objective(shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas, lambdas):
    """Returns ``(J, data_fit_sq)`` -- the full safeguarded objective and its
    data-fit term alone (the latter is what feeds the diagnostic ``r`` trace).
    """
    data_fit_sq = _backfit_data_fit_sq(
        shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas
    )
    return data_fit_sq + _backfit_ridge_penalty(order, deltas, lambdas), data_fit_sq


@torch.no_grad()
def _fit_block_boundary_backfit(
    target_model,
    current_state: dict[str, torch.Tensor],
    positions: list[int],
    source_coordinates: dict[int, float],
    desired_batches: dict[int, list],
    target_output_batches: dict[int, list],
    batches: list,
    components: tuple,
    config,
    device,
    family_adapter=None,
) -> dict[int, tuple[dict, list]]:
    """Intra-block Gauss-Seidel backfitting for ``component_target='block_boundary'``,
    ``block_split='backfit'`` (Direct Residual only; see ``DirectResidualConfig``).

    Every position is independent (the upstream target model is pristine, as
    everywhere else under independent mode), and every component within one
    position shares the SAME block-boundary target ``D_j`` -- exactly as
    ``_fit_all_positions_independent`` -- but instead of regressing each
    component independently against the full ``D_j`` (leaving a "double
    target" when more than one component is requested), this refits each
    component against the RESIDUAL left over once every other component's
    current fit is actually mounted and the block is replayed:

        E_c = D_j - (block_j(X_j^0; others mounted, c at its native base) - T_j^0)

    Gauss-Seidel over ``components`` (canonical forward order), from scratch
    every sweep (no warm start -- ``ResidualSufficientStatistics`` solves a
    fresh ridge system each time, exactly as every other direct-target fit
    does).

    Monotone safeguarded backfitting. Each Gauss-Seidel sub-step refits
    component ``c`` against ``E_c`` above, but ``c``'s update also changes
    the OTHER component's effect through the block's own nonlinear path
    (``attn.out_proj -> ln_2 -> GELU -> mlp.c_proj``'s input, on ViT), so the
    sub-step candidate is not an exact block-coordinate minimizer of the true
    objective and nothing prevents it from increasing that objective. This is
    fixed by measuring the true, un-linearized per-block objective on the
    local block replay,

        J(Delta) = ||D_j - (block_j(X_j^0; Delta mounted) - T_j^0)||_F^2
                   + sum_c lambda_c * ||Delta W_c||_F^2,

    and only ever accepting a change that decreases it. ``lambda_c`` is each
    component's ridge coefficient -- ``ResidualSufficientStatistics.solve``'s
    ``diag["ridge"]`` -- FROZEN at the value its round-1 (first sweep) solve
    returns, not recomputed every sweep: the whole point of a monotone
    descent objective is that it is a fixed function of ``Delta``, so a
    ridge that itself drifts sweep to sweep (as it does inside ``solve``,
    since it is a function of that sweep's own H_c statistics, which change
    as other components' deltas move) would make "J decreased" incomparable
    across sweeps. ``lambda_c`` penalizes ``Delta W_c`` (the weight only) at
    the SAME scale ``solve`` itself minimizes: its normal equations are
    ``S_c X G + lambda X = B_c``, i.e. the stationarity condition of
    ``||A X L - E||_F^2 + lambda ||X||_F^2`` with the bias fit unpenalized
    (``solve``'s ``beta`` is derived with no ridge term) -- so
    ``lambda_c * ||Delta W_c||_F^2`` is exactly what that component's own
    sub-solve minimizes, with ``Delta W_c`` the returned weight correction
    itself (``solve`` returns ``x.T``, and the penalty ``lambda ||x||_F^2``
    is transpose-invariant).

    Each sub-step computes the candidate ``Delta_c^new`` exactly as the
    unsafeguarded rule did, then accepts it only if ``J`` decreases;
    otherwise backtracks ``Delta_c = Delta_c^old + eta (Delta_c^new -
    Delta_c^old)`` (weight AND bias together) for ``eta = 1, 1/2, 1/4, ...``
    down to ``1/256`` (an initial full step plus up to 8 halvings); if no
    ``eta`` decreases ``J``, ``Delta_c`` is left at ``Delta_c^old``
    (recorded as an accepted ``eta`` of 0). ``J`` is therefore non-increasing
    by construction, at every sub-step and therefore every sweep. Sweeping
    stops when the relative decrease of ``J`` over a full sweep (measured
    once, with every current delta mounted, after each sweep's Gauss-Seidel
    pass) drops below ``config.backfit_tol``, or at
    ``config.backfit_max_iters`` sweeps. The plain relative residual
    ``r = ||D_j - (block_j(X_j^0; ALL current deltas mounted) - T_j^0)||_F /
    ||D_j||_F`` is still measured and logged every sweep as a diagnostic
    (``backfit_residual_trace``), but no longer drives the stopping rule.

    With a single component, the first sweep's ``E_c`` reduces exactly to
    ``D_j`` (no other component is mounted, so the replay term is
    ``T_j^0 - T_j^0 = 0``), so the fit is byte-for-byte the single-component
    call of ``_fit_all_positions_independent`` PROVIDED the replayed
    ``H_c``/output from the local block copy on the captured pristine
    ``X_j^0`` bitwise reproduce what a direct hook on the live target model
    would have captured -- asserted below (see ``block_replay_bitwise`` in the
    returned diagnostics) rather than assumed. With one component there is
    also nothing to backtrack against on later sweeps: the candidate is
    always accepted at ``eta=1`` (see the docstring of
    ``_fit_block_boundary_backfit``'s test coverage), since a single
    component's own sub-solve is an exact minimizer of ``J`` restricted to
    that component with every OTHER (nonexistent) component fixed.

    The full target model is never mutated: every mount happens on a
    ``copy.deepcopy`` of the block, discarded at the end of each position.
    """
    if family_adapter is not None:
        raise NotImplementedError("block_split='backfit' is vision-only")
    shim = _layout_for(family_adapter)
    residual_writers = set(COMPONENT_FORWARD_ORDER)
    if set(components) - residual_writers:
        raise ValueError(
            "block_split='backfit' only supports residual-writing components "
            f"{sorted(residual_writers)}; internal components (q/k/v/c_fc) act on the block "
            "output nonlinearly and have no linear regression onto a block-boundary target"
        )
    order = order_components(components)
    if not order:
        raise ValueError("components must not be empty")

    # Capture the pristine block input X_j^0 for every position in ONE target
    # sweep -- new capture kind, the forward-hook INPUT of the block module
    # itself (the same module "boundary" hooks for its OUTPUT).
    block_input_requests = {f"{pos}.block_input": (pos, "block_input") for pos in positions}
    block_inputs = capture_tokens(target_model, batches, block_input_requests, device, family_adapter=family_adapter)

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        x_batches = block_inputs[f"{pos}.block_input"]
        t0_batches = target_output_batches[pos]
        d_batches = desired_batches[pos]
        block = shim.blocks(target_model)[pos]
        local_block = copy.deepcopy(shim.block_module(block)).to(device).eval()

        # Device convention: every "delta" tensor this function accumulates
        # (base[c], deltas[c], and everything derived from them) lives on CPU
        # float32, exactly like ResidualSufficientStatistics.solve()'s own
        # output (`correction = correction.cpu()` below) and like
        # _fit_all_positions_independent's `current_state` bookkeeping.
        # `local_block` itself is moved to `device` (mirroring how the live
        # target model would sit on a training device at runtime), so its
        # parameters -- and therefore _component_weight_bias's raw read of
        # them -- are on `device`. Without the explicit `.cpu()` here, `base[c]`
        # would silently inherit that device, and `base[c2][0] + w2` below (w2
        # is always a CPU delta) would mix a CUDA tensor with a CPU one -- a
        # RuntimeError on CUDA that a CPU-only run can never surface, since
        # cpu + cpu never errors regardless of provenance. _mount_component
        # is the only place a base/delta tensor is moved back onto `device`
        # (via its own `.to(module.weight.device)`), so mounting stays correct
        # regardless of what device `local_block` lives on.
        base: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        for c in order:
            w, b, row_slice = _component_weight_bias(shim, local_block, c)
            if row_slice is not None:
                raise RuntimeError("block_split='backfit' components must not be packed (row_slice must be None)")
            base[c] = (w.detach().cpu().clone(), None if b is None else b.detach().cpu().clone())
        scale_modules = {c: shim.component_scale_module(local_block, c) for c in order}
        # A dedicated float64 replica, used ONLY by the monotone safeguard's own
        # J(Delta) measurement (see _backfit_data_fit_sq's docstring) -- never
        # for candidate generation, so the returned corrections' CPU-float32
        # convention is untouched. Deepcopied here while `local_block` is still
        # pristine (nothing has been mounted onto it yet).
        local_block64 = copy.deepcopy(local_block).double().eval()

        def reset_all(local_block=local_block, order=order, base=base):
            for c in order:
                w, b = base[c]
                _mount_component(shim, local_block, c, w, b)

        # Pristine-replay check: X_j^0 through the untouched local copy must
        # reproduce the captured boundary output T_j^0.
        reset_all()
        t_replayed = []
        for x in x_batches:
            t_replayed.append(local_block(x.to(device)).detach().float().cpu().clone())
        block_replay_bitwise = all(torch.equal(rep, ref) for rep, ref in zip(t_replayed, t0_batches, strict=True))
        for rep, ref in zip(t_replayed, t0_batches, strict=True):
            if not torch.allclose(rep, ref, atol=1e-4, rtol=1e-4):
                raise RuntimeError(
                    f"Block replay at position {pos} does not reproduce the pristine boundary "
                    "output within tolerance; X_j^0/local-block-copy mismatch"
                )

        deltas: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
            c: (torch.zeros_like(base[c][0]), None if base[c][1] is None else torch.zeros_like(base[c][1]))
            for c in order
        }
        d_sq_total = sum(float((d.double() ** 2).sum().item()) for d in d_batches)
        desired_norm = d_sq_total**0.5
        residual_trace: list[float] = []
        j_trace: list[float] = []
        converged = False
        n_sweeps = 0
        j_prev = None
        round1_j: float | None = None
        round1_r: float | None = None
        # J(Delta=0) = ||D_j - (T_j^0 - T_j^0)||^2 + 0 = ||D_j||^2 (no ridge penalty
        # at the all-zero start).
        lambdas: dict[str, float] = {}
        per_component_diag: dict[str, dict] = {}
        per_component_h: dict[str, list] = {}
        per_component_effective_out: dict[str, torch.Tensor] = {}
        per_component_eta_history: dict[str, list[float]] = {c: [] for c in order}
        # Backtracking line-search factors: an initial full step, then up to 8
        # halvings (see the docstring). eta=0 (keep the old delta) is the
        # implicit fallback when none of these decrease J.
        backtrack_etas = [1.0] + [1.0 / (2**k) for k in range(1, 9)]
        for sweep in range(1, int(config.backfit_max_iters) + 1):
            n_sweeps = sweep
            for c in order:
                reset_all()
                for c2 in order:
                    if c2 == c:
                        continue
                    w2, b2 = deltas[c2]
                    _mount_component(
                        shim, local_block, c2, base[c2][0] + w2, None if base[c2][1] is None else base[c2][1] + b2
                    )
                component_h, out_batches = _replay_block_components(shim, local_block, x_batches, [c], device)
                h_batches = component_h[c]
                e_batches = [d - (t - t0) for d, t, t0 in zip(d_batches, out_batches, t0_batches, strict=True)]
                width = int(base[c][0].shape[0])
                identity_out = torch.eye(width, dtype=torch.float32)
                scale_module = scale_modules[c]
                effective_out = identity_out
                if not isinstance(scale_module, nn.Identity):
                    scale = getattr(scale_module, "gamma", None)
                    if scale is None or scale.ndim != 1 or scale.shape[0] != width:
                        raise ValueError("Unsupported non-diagonal target LayerScale")
                    effective_out = identity_out * scale.detach().cpu().float().unsqueeze(0)
                stats = ResidualSufficientStatistics(device=device)
                for h, e in zip(h_batches, e_batches, strict=True):
                    stats.update(h.reshape(-1, h.shape[-1]), e.reshape(-1, e.shape[-1]), None, effective_out)
                correction, diag = stats.solve(
                    ridge_relative=config.ridge_relative,
                    ridge_estimator=config.ridge_estimator,
                    exact_form=config.exact_form,
                )
                correction = correction.cpu()
                diag["bias_correction"] = diag["bias_correction"].cpu()
                if correction.shape != base[c][0].shape or not torch.isfinite(correction).all():
                    raise RuntimeError("block_split='backfit' produced an invalid projection")
                per_component_diag[c] = diag
                per_component_h[c] = h_batches
                per_component_effective_out[c] = effective_out

                # Freeze lambda_c at its round-1 (first-solve) value: J must stay
                # a FIXED function of Delta across the whole backfit for "J
                # decreased" to be comparable sweep to sweep (see the docstring).
                # solve()'s own internal ridge is recomputed every sweep from
                # that sweep's H_c -- that only shapes the CANDIDATE proposed
                # below, never the objective the safeguard accepts or rejects
                # against.
                if sweep == 1:
                    lambdas[c] = float(diag["ridge"])

                old_w, old_b = deltas[c]
                candidate_w, candidate_b = correction, diag["bias_correction"]
                trial_deltas = dict(deltas)
                j_old, _ = _backfit_objective(
                    shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, trial_deltas, lambdas
                )
                accepted_eta = 0.0
                accepted_delta = (old_w, old_b)
                for eta in backtrack_etas:
                    trial_w = old_w + eta * (candidate_w - old_w)
                    trial_b = None if old_b is None else old_b + eta * (candidate_b - old_b)
                    trial_deltas[c] = (trial_w, trial_b)
                    j_trial, _ = _backfit_objective(
                        shim,
                        local_block64,
                        device,
                        x_batches,
                        d_batches,
                        t0_batches,
                        order,
                        base,
                        trial_deltas,
                        lambdas,
                    )
                    if j_trial < j_old:
                        accepted_eta = eta
                        accepted_delta = (trial_w, trial_b)
                        break
                deltas[c] = accepted_delta
                per_component_eta_history[c].append(accepted_eta)
            # Measure the full-sweep objective and diagnostic residual with every
            # current delta mounted -- the same mount _backfit_objective performs,
            # done once more here only because we also want the plain (ridge-free)
            # data-fit norm `r` for the diagnostic trace.
            j_now, data_fit_sq = _backfit_objective(
                shim, local_block64, device, x_batches, d_batches, t0_batches, order, base, deltas, lambdas
            )
            r = (data_fit_sq**0.5) / (desired_norm + 1e-12)
            residual_trace.append(r)
            j_trace.append(j_now)
            if sweep == 1:
                round1_j, round1_r = j_now, r
            if j_prev is not None:
                rel_decrease = (j_prev - j_now) / j_prev if j_prev > 0 else 0.0
                if rel_decrease < float(config.backfit_tol):
                    converged = True
                    break
            j_prev = j_now

        position_corrections: dict[str, torch.Tensor] = {}
        block_rows: list[dict[str, Any]] = []
        for c in order:
            key = shim.component_key(pos, c, prefixed=True)
            correction, bias_correction = deltas[c]
            position_corrections[key] = correction
            bias_key = f"{key[: -len('.weight')]}.bias"
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
                        "Set target_residual_completion.missing_bias to 'materialize' (exact, "
                        "adds the parameter) or 'skip' with exact_form=false."
                    )
            if not skip_bias:
                bias_delta = bias_correction.to(current_state[bias_key])
                if bias_delta.shape != current_state[bias_key].shape or not torch.isfinite(bias_delta).all():
                    raise RuntimeError("block_split='backfit' produced an invalid bias")
                position_corrections[bias_key] = bias_delta
            diag = per_component_diag[c]
            block_row = {
                "mode": "direct_target",
                "component": c,
                "component_target": "block_boundary",
                "block_split": "backfit",
                "position": pos,
                "source_coordinate": float(source_coordinates[pos]),
                "desired_norm": desired_norm,
                "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
                "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                "backfit_n_sweeps": n_sweeps,
                "backfit_converged": converged,
                "backfit_residual_trace": list(residual_trace),
                "backfit_j_trace": list(j_trace),
                "backfit_round1_j": round1_j,
                "backfit_round1_r": round1_r,
                "backfit_ridge_lambda": lambdas.get(c),
                "backfit_eta_history": list(per_component_eta_history[c]),
                "block_replay_bitwise": block_replay_bitwise,
                **diag,
            }
            if bool(getattr(config, "realization_diagnostics", False)):
                bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
                block_row.update(
                    _realization_diagnostic_fields(
                        per_component_h[c],
                        correction,
                        bias_for_pred,
                        per_component_effective_out[c],
                        base[c][0],
                        desired_norm,
                        diag["residual_norm_after"],
                    )
                )
            block_rows.append(block_row)
        results[pos] = (position_corrections, block_rows)
    return results


class _JointBlockRidgeStatistics:
    """Streaming Gram/cross accumulator for the closed-form joint (O, D)-stacked
    ridge fit ``block_split='joint'`` uses (see ``_fit_block_boundary_joint``).

    Mirrors ``ResidualSufficientStatistics`` at ``t_in=None, t_out=I`` (the only
    configuration ``block_split='joint'`` ever needs -- LayerScale is asserted
    ``nn.Identity`` before this class is ever touched), generalized from one
    ridge scalar to a per-block-diagonal ridge vector over the STACKED feature
    dimension ``sum(dims)``. Accumulates the ``(d_total, d_total)`` Gram and
    ``(d_total, d_out)`` cross statistics batch by batch in float64, never
    materializing a design matrix with as many rows as calibration tokens (the
    Gram/cross tensors are the only ``O(d^2)``-sized state this class holds --
    ``d_total = d_O + d_D`` is a few thousand for ViT-L/14, not the token
    count).
    """

    def __init__(self, dims: list[int], d_out: int, device=None) -> None:
        if len(dims) < 1 or any(d <= 0 for d in dims) or d_out <= 0:
            raise ValueError("dims and d_out must be positive")
        self.device = device
        self.dims = list(dims)
        self.d_total = sum(dims)
        self.d_out = d_out
        self.gram = torch.zeros(self.d_total, self.d_total, dtype=torch.float64, device=device)
        self.cross = torch.zeros(self.d_total, d_out, dtype=torch.float64, device=device)
        self.sum_a = torch.zeros(self.d_total, dtype=torch.float64, device=device)
        self.sum_e = torch.zeros(d_out, dtype=torch.float64, device=device)
        self.sum_e2 = 0.0
        self.n_rows = 0

    def update(self, h_list: list[torch.Tensor], e: torch.Tensor) -> None:
        if len(h_list) != len(self.dims):
            raise ValueError("h_list must supply one feature bank per stacked component")
        if self.device is not None:
            h_list = [h.to(self.device) for h in h_list]
            e = e.to(self.device)
        for h, d in zip(h_list, self.dims, strict=True):
            if h.ndim != 2 or h.shape[1] != d:
                raise ValueError("component feature bank has an unexpected shape")
        if e.ndim != 2 or e.shape[1] != self.d_out:
            raise ValueError("target bank has an unexpected shape")
        rows = {h.shape[0] for h in h_list} | {e.shape[0]}
        if len(rows) != 1:
            raise ValueError("component feature banks and the target bank must share the row count")
        a = torch.cat([h.to(torch.float64) for h in h_list], dim=1)
        er = e.to(torch.float64)
        if not torch.isfinite(a).all() or not torch.isfinite(er).all():
            raise ValueError("solver inputs must be finite")
        self.gram += a.T @ a
        self.cross += a.T @ er
        self.sum_a += a.sum(dim=0)
        self.sum_e += er.sum(dim=0)
        self.sum_e2 += float((er * er).sum().item())
        self.n_rows += int(a.shape[0])

    def solve(self, lambdas: list[float]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Exact centered normal-equation solve with a block-diagonal ridge.

        Returns ``(x, beta, diag)`` with ``x`` shaped ``(d_total, d_out)`` (the
        stacked, UNTRANSPOSED weight -- callers split it by ``self.dims`` and
        transpose each block to the usual ``(out, in)`` weight convention) and
        ``beta`` the shared ``(d_out,)`` intercept solved jointly with ``x``
        (not ridge-penalized, matching ``ResidualSufficientStatistics.solve``'s
        own convention).

        Equivalent to rescaling features by ``1/sqrt(lambda_c)`` per block and
        solving with unit ridge (the textbook reduction of a block-diagonal
        ridge to a scalar one), but implemented directly on the centered normal
        equations ``(S_c + diag(lambda)) X = B_c`` -- exact in float64, no
        rescale/un-rescale round trip.
        """
        if len(lambdas) != len(self.dims):
            raise ValueError("lambdas must supply one ridge coefficient per stacked component")
        if self.n_rows == 0:
            raise ValueError("cannot solve empty joint block statistics")
        if any((not math.isfinite(float(lam))) or float(lam) < 0 for lam in lambdas):
            raise ValueError("lambdas must be finite and non-negative")
        n = float(self.n_rows)
        mu_a = self.sum_a / n
        mu_e = self.sum_e / n
        s = (self.gram + self.gram.T) * 0.5
        sc = s - torch.outer(self.sum_a, self.sum_a) / n
        bc = self.cross - torch.outer(self.sum_a, mu_e)
        sc = (sc + sc.T) * 0.5
        lam_vec = torch.cat(
            [
                torch.full((d,), float(lam), dtype=torch.float64, device=sc.device)
                for d, lam in zip(self.dims, lambdas, strict=True)
            ]
        )
        reg = sc + torch.diag(lam_vec)
        x = torch.linalg.solve(reg, bc)
        beta = mu_e - x.T @ mu_a
        # Exact total ||A x + 1 beta^T - E||_F^2 from raw (uncentered) sufficient
        # statistics -- same decomposition as ResidualSufficientStatistics.
        # _residual_sq, specialized to t_out=I (g=I, lmat=I).
        predicted_sq = torch.trace(x.T @ s @ x).item()
        cross_term = 2.0 * torch.sum(x * self.cross).item()
        resid_no_bias = self.sum_e2 - cross_term + predicted_sq
        pred_mean = mu_a @ x
        bias_term = 2.0 * n * float((beta @ (pred_mean - mu_e)).item()) + n * float((beta @ beta).item())
        residual_sq = max(0.0, resid_no_bias + bias_term)
        diag = {
            "n_rows": self.n_rows,
            "residual_norm_before": self.sum_e2**0.5,
            "residual_norm_after": residual_sq**0.5,
        }
        return x, beta, diag


@torch.no_grad()
def _fit_block_boundary_joint(
    target_model,
    current_state: dict[str, torch.Tensor],
    positions: list[int],
    source_coordinates: dict[int, float],
    desired_batches: dict[int, list],
    target_output_batches: dict[int, list],
    batches: list,
    components: tuple,
    config,
    device,
    family_adapter=None,
) -> dict[int, tuple[dict, list]]:
    """Closed-form joint ridge fit for ``component_target='block_boundary'``,
    ``block_split='joint'`` (Direct Residual only; see ``DirectResidualConfig``).

    Where ``block_split='none'`` fits every requested component INDEPENDENTLY
    against the same block-boundary target ``D_j`` (double-counting the target
    when more than one component is requested) and ``block_split='backfit'``
    resolves that by iterative Gauss-Seidel replay of the block's own
    nonlinearity, ``'joint'`` instead solves ONE closed-form ridge in a single
    linear-algebra step, under the explicit first-order approximation that the
    MLP does not respond to a change in ``attn.out_proj`` (``J_M = 0``): with
    ``H_O``/``H_D`` the pristine ``attn.out_proj``/``mlp.c_proj`` inputs (the
    SAME captures ``_fit_all_positions_independent`` uses) and
    ``D_j`` the block-boundary desired effect,

        min_{Wo,Wd,beta} ||H_O Wo^T + H_D Wd^T + 1 beta^T - D_j||_F^2
                          + lambda_O ||Wo||_F^2 + lambda_D ||Wd||_F^2.

    Only valid for ``components`` a non-empty subset of
    ``{"attn.out_proj", "mlp.c_proj"}`` (enforced by
    ``direct_residual.parse_direct_residual_config``).

    **lambda_c convention.** Each component's ridge coefficient is EXACTLY the
    value its OWN standalone ``block_split='none'`` fit would use --
    ``ResidualSufficientStatistics.solve(...)``'s own ``diag["ridge"]``,
    computed from that component's own ``H_c`` and the ridge_relative/
    ridge_estimator/exact_form config, penalizing ``Delta W_c`` (the weight
    only; the joint intercept below is unpenalized) at the identical scale
    ``solve`` itself would use it at alone -- see
    ``tests/test_direct_residual_joint.py``'s ``test_lambda_matches_single_
    component_solver`` for the regression pinning this.

    **Bias convention.** The joint intercept is not identifiable between the
    two components' biases (only their sum enters the objective). The WHOLE
    fitted intercept is assigned to ``mlp.c_proj.bias`` -- the block's LAST
    residual writer -- and ``attn.out_proj.bias`` is left untouched (a zero
    delta): ``attn.out_proj``'s bias also feeds the MLP's input on the real
    (nonlinear) block, which this first-order joint model ignores by
    construction, so it must not absorb any share of the block-level
    intercept a purely-linear model derived. With a single requested component
    this convention is moot -- see below.

    **LayerScale.** Asserted ``nn.Identity`` on both ``ls_1``/``ls_2`` before
    any solve (``_assert_layerscale_identity``): a nontrivial LayerScale would
    give the two writers different output gammas, which the stacked-feature
    derivation above assumes away.

    **Single-component reduction.** With one requested component, this
    function does not build a (degenerate, one-block) joint system at all --
    it returns that component's own standalone ``block_split='none'`` solve
    directly (the same ``ResidualSufficientStatistics`` call this function
    computes ``lambda_c`` from in the first place), so single-component
    ``block_split='joint'`` is not merely numerically close to
    ``block_split='none'`` but literally the same function call.

    Every position is independent and pristine (nothing is ever mounted
    between fits, matching every other Direct Residual path), so ``E_j ==
    D_j`` identically and this reuses ``capture_tokens``'s combined-request,
    one-sweep capture exactly like ``_fit_all_positions_independent``.
    """
    shim = _layout_for(family_adapter)
    residual_writers = set(COMPONENT_FORWARD_ORDER)
    if set(components) - residual_writers:
        raise ValueError(
            "block_split='joint' only supports residual-writing components "
            f"{sorted(residual_writers)}; internal components (q/k/v/c_fc) act on the block "
            "output nonlinearly and have no linear regression onto a block-boundary target"
        )
    order = order_components(components)
    if not order:
        raise ValueError("components must not be empty")

    requests: dict[str, tuple[int, str]] = {}
    for pos in positions:
        requests[f"{pos}.out"] = (pos, "boundary")
        for component in order:
            requests[f"{pos}.{component}.h"] = (pos, COMPONENT_INPUT_KIND[component])
    captured = capture_tokens(target_model, batches, requests, device, family_adapter=family_adapter)

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        block = shim.blocks(target_model)[pos]
        _assert_layerscale_identity(shim, block, context=f"position {pos}", requirement="block_split='joint'")
        out_batches = captured[f"{pos}.out"]
        d_batches = desired_batches[pos]
        t0_batches = target_output_batches[pos]

        # The block hasn't been touched yet at this position (independent
        # mode, pristine by construction): the pre-fit effect is the SAME for
        # every component, so it is measured once rather than per component.
        desired_sq = 0.0
        effect_sq = 0.0
        error_batches = []
        for out, desired_batch, base_out in zip(out_batches, d_batches, t0_batches, strict=True):
            effect = out - base_out
            desired_sq += float((desired_batch.double() ** 2).sum().item())
            effect_sq += float((effect.double() ** 2).sum().item())
            error_batches.append(desired_batch - effect)
        if effect_sq > 1e-12 * max(desired_sq, 1.0):
            raise RuntimeError(
                "Direct completion started from a target model that is not the native "
                f"base: nonzero pre-fit effect at position {pos} (||T-T0||^2={effect_sq:.3e})"
            )
        desired_norm = desired_sq**0.5

        h_batches_by_component: dict[str, list[torch.Tensor]] = {}
        widths: dict[str, int] = {}
        lambdas: dict[str, float] = {}
        single_component_fit: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
        for component in order:
            key = shim.component_key(pos, component, prefixed=True)
            h_batches = captured[f"{pos}.{component}.h"]
            h_batches_by_component[component] = h_batches
            width = int(current_state[key].shape[0])
            widths[component] = width
            scale_module = shim.component_scale_module(block, component)
            if not isinstance(scale_module, nn.Identity):
                raise ValueError(f"block_split='joint' requires an identity LayerScale on {component}'s output")
            effective_out = torch.eye(width, dtype=torch.float32)
            stats = ResidualSufficientStatistics(device=device)
            for h, error in zip(h_batches, error_batches, strict=True):
                stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)
            correction, diag = stats.solve(
                ridge_relative=config.ridge_relative,
                ridge_estimator=config.ridge_estimator,
                exact_form=config.exact_form,
            )
            correction = correction.cpu()
            diag["bias_correction"] = diag["bias_correction"].cpu()
            lambdas[component] = float(diag["ridge"])
            single_component_fit[component] = (correction, diag)

        position_corrections: dict[str, torch.Tensor] = {}
        block_rows: list[dict[str, Any]] = []
        if len(order) == 1:
            # See the docstring: the bias-split convention below is only
            # meaningful with both writers present, so a single requested
            # component just IS the standalone block_split='none' fit -- the
            # identical ResidualSufficientStatistics call computed above for
            # lambda_c, reused verbatim rather than resolved.
            (component,) = order
            correction, diag = single_component_fit[component]
            key = shim.component_key(pos, component, prefixed=True)
            weight_before = current_state[key].detach().clone()
            if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
                raise RuntimeError("block_split='joint' produced an invalid projection")
            position_corrections[key] = correction
            current_state[key] = current_state[key] + correction.to(current_state[key])
            bias_key = f"{key[: -len('.weight')]}.bias"
            bias_correction = diag["bias_correction"]
            skip_bias = _apply_bias_correction(config, current_state, bias_key, bias_correction, position_corrections)
            block_row = {
                "mode": "direct_target",
                "component": component,
                "component_target": "block_boundary",
                "block_split": "joint",
                "position": pos,
                "source_coordinate": float(source_coordinates[pos]),
                "desired_norm": desired_norm,
                "effect_before_norm": effect_sq**0.5,
                "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
                "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                "joint_lambda": {component: lambdas[component]},
                "joint_residual_relative": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                **diag,
            }
            if bool(getattr(config, "realization_diagnostics", False)):
                bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
                block_row.update(
                    _realization_diagnostic_fields(
                        h_batches_by_component[component],
                        correction,
                        bias_for_pred,
                        torch.eye(widths[component], dtype=torch.float32),
                        weight_before,
                        desired_norm,
                        diag["residual_norm_after"],
                    )
                )
            block_rows.append(block_row)
            results[pos] = (position_corrections, block_rows)
            continue

        # Genuine joint (>=2 component) stacked solve. `dims` is each
        # component's own INPUT feature width (H_c's last dim -- e.g.
        # mlp.c_proj's input is d_model*mlp_ratio, NOT its output width
        # `widths[c]`, which is d_model like every other residual writer's
        # OUTPUT). d_out is read off the block's own boundary output bank
        # (the shared regression target every component's H_c writes into).
        dims = [int(h_batches_by_component[c][0].shape[-1]) for c in order]
        d_out = int(out_batches[0].shape[-1])
        joint_stats = _JointBlockRidgeStatistics(dims, d_out, device=device)
        for rows in zip(*(h_batches_by_component[c] for c in order), error_batches, strict=True):
            *h_rows, error = rows
            joint_stats.update([h.reshape(-1, h.shape[-1]) for h in h_rows], error.reshape(-1, error.shape[-1]))
        lambda_list = [lambdas[c] for c in order]
        x, beta, joint_diag = joint_stats.solve(lambda_list)
        x = x.cpu()
        beta = beta.cpu().to(torch.float32)

        offsets = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)
        last_writer = order[-1]
        for idx, component in enumerate(order):
            key = shim.component_key(pos, component, prefixed=True)
            correction = x[offsets[idx] : offsets[idx + 1], :].T.to(torch.float32).contiguous()
            weight_before = current_state[key].detach().clone()
            if correction.shape != current_state[key].shape or not torch.isfinite(correction).all():
                raise RuntimeError("block_split='joint' produced an invalid projection")
            position_corrections[key] = correction
            current_state[key] = current_state[key] + correction.to(current_state[key])
            bias_key = f"{key[: -len('.weight')]}.bias"
            # See the docstring: the whole joint intercept goes to the LAST
            # residual writer (mlp.c_proj in the historical two-writer case);
            # every other component's bias gets an exact zero delta.
            bias_correction = beta if component == last_writer else torch.zeros_like(beta)
            skip_bias = _apply_bias_correction(config, current_state, bias_key, bias_correction, position_corrections)
            _single_correction, single_diag = single_component_fit[component]
            block_row = {
                "mode": "direct_target",
                "component": component,
                "component_target": "block_boundary",
                "block_split": "joint",
                "position": pos,
                "source_coordinate": float(source_coordinates[pos]),
                "desired_norm": desired_norm,
                "effect_before_norm": effect_sq**0.5,
                "relative_residual_before": (joint_diag["residual_norm_before"] / desired_norm)
                if desired_norm
                else 0.0,
                "relative_residual_after": (joint_diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                "n_rows": joint_diag["n_rows"],
                "residual_norm_before": joint_diag["residual_norm_before"],
                "residual_norm_after": joint_diag["residual_norm_after"],
                "exact_form": bool(config.exact_form),
                "ridge_estimator": config.ridge_estimator,
                "configured_ridge_relative": float(config.ridge_relative),
                "effective_ridge_relative": single_diag["effective_ridge_relative"],
                "ridge": lambdas[component],
                "correction_norm": float(torch.linalg.norm(correction).item()),
                "bias_norm": float(torch.linalg.norm(bias_correction).item()),
                "bias_correction": bias_correction,
                "joint_lambda": dict(lambdas),
                "joint_residual_relative": (joint_diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
            }
            if bool(getattr(config, "realization_diagnostics", False)):
                bias_for_pred = bias_correction if not skip_bias else torch.zeros(correction.shape[0])
                block_row.update(
                    _realization_diagnostic_fields(
                        h_batches_by_component[component],
                        correction,
                        bias_for_pred,
                        torch.eye(widths[component], dtype=torch.float32),
                        weight_before,
                        desired_norm,
                        joint_diag["residual_norm_after"],
                    )
                )
            block_rows.append(block_row)
        results[pos] = (position_corrections, block_rows)
    return results


def _apply_bias_correction(config, current_state, bias_key, bias_correction, position_corrections) -> bool:
    """Shared ``missing_bias`` handling for a single component's bias delta
    (extracted from the ``block_split in {'none', 'backfit'}`` paths so
    ``block_split='joint'`` follows the exact same ``error``/``materialize``/
    ``skip`` contract). Mutates ``position_corrections`` in place with the
    accepted bias delta (unless skipped) and returns whether the bias was
    skipped.
    """
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
            return True
        else:
            raise RuntimeError(
                f"Target model is missing the expected bias parameter {bias_key}. "
                "Set target_residual_completion.missing_bias to 'materialize' (exact, "
                "adds the parameter) or 'skip' with exact_form=false."
            )
    bias_delta = bias_correction.to(current_state[bias_key])
    if bias_delta.shape != current_state[bias_key].shape or not torch.isfinite(bias_delta).all():
        raise RuntimeError("block_split='joint' produced an invalid bias")
    position_corrections[bias_key] = bias_delta
    current_state[bias_key] = current_state[bias_key] + bias_delta
    return False


@torch.no_grad()
def _fit_component_outputs_from_contributions(
    target_model,
    current_state: dict[str, torch.Tensor],
    positions: list[int],
    position_contributions: dict[int, list[tuple[int, float]]],
    paired_source_index: dict[int, int],
    source_component_inputs: dict[int, dict[str, list[torch.Tensor]]],
    source_component_weights: dict[int, dict[str, dict[str, torch.Tensor | None]]],
    batches: list,
    components: tuple,
    config,
    device,
    family_adapter=None,
    source_component_inputs_ft: dict[int, dict[str, list[torch.Tensor]]] | None = None,
) -> dict[int, tuple[dict, list]]:
    """Fit every position's requested components against a per-component
    target built from one or more weighted source contributions
    (``config.component_target in {'output_local', 'output_total'}``).

    Unlike ``_fit_all_positions_independent`` -- where every requested
    component (``attn.out_proj`` and ``mlp.c_proj`` alike) is regressed onto
    the SAME shared block-boundary target ``D_j`` -- each component ``c``
    here gets its own target. For a residual-writing component (``attn.
    out_proj``/``mlp.c_proj``), position ``j`` may draw on several source
    blocks at once (``position_contributions[j] = [(i, w_i), ...]``, e.g. the
    span-aware shrink rule); for an internal component (q/k/v/c_fc) it always
    draws on exactly the paired source block ``paired_source_index[j]``, at
    the same weight that block carries in ``position_contributions[j]``
    (``1/m_i`` on extend/same_arch, ``1.0`` on shrink -- see the caller, e.g.
    ``direct_residual.position_source_contributions``, for how these weights
    are derived per direction). The
    per-contribution term, for ``component_target='output_local'``, is

        Delta_A_{i,c} = X^{s0}_i (W_c^{s1,i} - W_c^{s0,i})^T + (b_c^{s1,i} - b_c^{s0,i})
                      = A_c(X^{s0}_i, W_c^{s1,i}) - A_c(X^{s0}_i, W_c^{s0,i}),

    i.e. only source block ``i``'s own weight change ("output_local": no
    upstream-induced input drift), evaluated on that block's own base input
    ``X^{s0}_i`` for both endpoints.

    For ``component_target='output_total'`` (``source_component_inputs_ft``
    not ``None``), the fine-tuned endpoint is instead evaluated on the source
    FT model's OWN captured input ``X^{s1}_i`` (which may differ from
    ``X^{s0}_i`` once any upstream block's weights have changed), so the term
    becomes

        Delta_A_{i,c} = A_c(X^{s1}_i, W_c^{s1,i}) - A_c(X^{s0}_i, W_c^{s0,i}),

    i.e. component ``c``'s full realized output change, upstream drift
    included. The alignment map ``Q_{j,i,c}`` (below) is unchanged between
    the two modes: it is always fit from the BASE-input raw output
    ``A_c^{s0}_i = X^{s0}_i (W_c^{s0,i})^T + b_c^{s0,i}``, never the FT one.
    ``Q_{j,i,c}`` is that term's own centered
    rectangular Procrustes map, fitted from source block ``i``'s raw
    component output ``A_c^{s0}_i = X^{s0}_i (W_c^{s0,i})^T + b_c^{s0,i}`` onto
    the target's own pristine raw component output ``A_c^{t0}_j = H (W_c^{t,0,j})^T
    + b_c^{t,0,j}``, where ``H`` is the target's own captured component input
    at position ``j``. The position's target is the weighted sum of every
    contribution's aligned term:

        D_{j,c} = sum_i w_i * aligned(Delta_A_{i,c}) Q_{j,i,c}.

    The regression then solves ``(H, D_{j,c})`` through the same
    ``ResidualSufficientStatistics`` machinery as every other direct-target
    fit, with ``t_in=None`` (no input transport) and ``t_out=I`` (LayerScale
    is asserted Identity by the caller, via ``_assert_layerscale_identity``,
    so there is nothing else to fold into the output map).

    This function never mounts a correction: the target model is captured
    once per position, at the pristine base, and that pristineness is
    asserted against ``current_state`` rather than assumed. Because nothing
    is ever mounted, ``q``/``k``/``v`` corrections at the same position are
    independent regressions that happen to write disjoint row slices of the
    same packed ``in_proj_weight``/``in_proj_bias`` parameter; each is
    accumulated into a per-position zero tensor so untouched slices stay
    exactly zero.

    Returns ``{position: (position_corrections, block_rows)}``, the same
    contract as ``_fit_all_positions_independent``.
    """
    if family_adapter is not None:
        raise NotImplementedError("component_target='output_local' is vision-only")
    shim = _layout_for(family_adapter)
    residual_writers = set(COMPONENT_FORWARD_ORDER)
    diagnose = bool(getattr(config, "realization_diagnostics", False))

    results: dict[int, tuple[dict, list]] = {}
    for pos in positions:
        target_block = shim.blocks(target_model)[pos]
        _assert_layerscale_identity(shim, target_block, context=f"target block {pos}")

        needed_kinds = sorted({COMPONENT_INPUT_KIND[c] for c in components})
        captured = capture_tokens(
            target_model,
            batches,
            {kind: (pos, kind) for kind in needed_kinds},
            device,
            family_adapter=family_adapter,
        )

        position_corrections: dict[str, torch.Tensor] = {}
        block_rows: list[dict[str, Any]] = []
        for component in components:
            key = shim.component_key(pos, component, prefixed=True)
            input_kind = COMPONENT_INPUT_KIND[component]
            h_batches = captured[input_kind]
            if component in residual_writers:
                contributions = position_contributions[pos]
            else:
                # Internal components (q/k/v/c_fc) use the paired source block
                # only, but at the SAME weight that block carries in the
                # residual-writer contribution list for this position
                # (1/m_i on extend/same_arch, 1.0 on shrink) -- not a
                # hardcoded 1.0, which would be wrong under extend, where a
                # source block realized as m_i > 1 target positions must have
                # its local effect split across them.
                paired = paired_source_index[pos]
                weight = next((w for i, w in position_contributions[pos] if i == paired), None)
                if weight is None:
                    raise ValueError(
                        f"Paired source index {paired} is not among position {pos}'s contributions "
                        f"({position_contributions[pos]!r}); internal components only ever use the "
                        "paired block, so it must appear there"
                    )
                contributions = [(paired, weight)]
            if not contributions:
                raise ValueError(f"No source contributions for position {pos}, component {component!r}")

            target_w, target_b, row_slice = _component_weight_bias(shim, target_block, component)
            # Live module parameters may be on any device (e.g. CUDA, if the
            # caller keeps the target model resident there); every captured
            # bank (h_batches, source component inputs/weights) is forced to
            # CPU float32 by capture_tokens/capture_source_component_references.
            # Force these to match before any F.linear/comparison against them.
            target_w = target_w.detach().cpu().float()
            target_b = None if target_b is None else target_b.detach().cpu().float()
            expected = current_state[key]
            if row_slice is not None:
                expected = expected[row_slice]
            if not torch.equal(target_w.detach().cpu().float(), expected.detach().cpu().float()):
                raise RuntimeError(
                    f"component_target='output_local' fit started from a target model that is "
                    f"not the native base at position {pos}, component {component!r}"
                )
            a_t0_batches = [F.linear(h, target_w, target_b) for h in h_batches]

            d_batches = [torch.zeros_like(a) for a in a_t0_batches]
            procrustes_ranks = []
            for source_idx, weight in contributions:
                if source_idx not in source_component_inputs or source_idx not in source_component_weights:
                    raise ValueError(f"Missing captured component references for source block {source_idx}")
                source_x_batches = source_component_inputs[source_idx][input_kind]
                weights = source_component_weights[source_idx][component]
                base_w, base_b = weights["base_weight"], weights["base_bias"]
                ft_w, ft_b = weights["ft_weight"], weights["ft_bias"]
                a0_batches = [F.linear(x, base_w, base_b) for x in source_x_batches]
                if source_component_inputs_ft is not None:
                    # output_total: evaluate the FT endpoint on the source FT
                    # model's OWN captured input X^{s1}_i, not X^{s0}_i -- the
                    # only way this term differs from output_local's.
                    if source_idx not in source_component_inputs_ft:
                        raise ValueError(f"Missing captured FT component references for source block {source_idx}")
                    source_x_ft_batches = source_component_inputs_ft[source_idx][input_kind]
                    ft_a_batches = [F.linear(x, ft_w, ft_b) for x in source_x_ft_batches]
                    delta_batches = [
                        ft_a - F.linear(x, base_w, base_b)
                        for ft_a, x in zip(ft_a_batches, source_x_batches, strict=True)
                    ]
                else:
                    delta_batches = [
                        F.linear(x, ft_w, ft_b) - F.linear(x, base_w, base_b) for x in source_x_batches
                    ]
                aligned_a0 = _aligned(a0_batches, h_batches)
                aligned_delta = _aligned(delta_batches, h_batches)
                q, _mu_s, _mu_t = centered_rectangular_procrustes(
                    _rows(aligned_a0).double(), _rows(a_t0_batches).double()
                )
                q = q.float()
                procrustes_ranks.append(int(torch.linalg.matrix_rank(q.double()).item()))
                for idx, delta in enumerate(aligned_delta):
                    d_batches[idx] = d_batches[idx] + float(weight) * (delta @ q)

            d_out = int(target_w.shape[0])
            identity_out = torch.eye(d_out, dtype=torch.float32)
            stats = ResidualSufficientStatistics(device=device)
            desired_sq = 0.0
            for h, d in zip(h_batches, d_batches, strict=True):
                desired_sq += float((d.double() ** 2).sum().item())
                stats.update(h.reshape(-1, h.shape[-1]), d.reshape(-1, d.shape[-1]), None, identity_out)
            correction, diag = stats.solve(
                ridge_relative=config.ridge_relative,
                ridge_estimator=config.ridge_estimator,
                exact_form=config.exact_form,
            )
            correction = correction.cpu()
            diag["bias_correction"] = diag["bias_correction"].cpu()
            if correction.shape != target_w.shape or not torch.isfinite(correction).all():
                raise RuntimeError("output_local component fit produced an invalid projection")

            if row_slice is not None:
                if key not in position_corrections:
                    position_corrections[key] = torch.zeros_like(current_state[key])
                position_corrections[key][row_slice] = correction
            else:
                position_corrections[key] = correction

            # Packed q/k/v share ``attn.in_proj_weight`` / ``attn.in_proj_bias``
            # (no dot before "weight"/"bias"), unlike a standalone projection's
            # ``<module>.weight`` / ``<module>.bias``.
            if key.endswith("in_proj_weight"):
                bias_key = key[: -len("in_proj_weight")] + "in_proj_bias"
            else:
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
                        "Set target_residual_completion.missing_bias to 'materialize' (exact, "
                        "adds the parameter) or 'skip' with exact_form=false."
                    )
            if not skip_bias:
                bias_delta = bias_correction.to(current_state[bias_key])
                if row_slice is not None:
                    if bias_key not in position_corrections:
                        position_corrections[bias_key] = torch.zeros_like(current_state[bias_key])
                    if (
                        bias_delta.shape != current_state[bias_key][row_slice].shape
                        or not torch.isfinite(bias_delta).all()
                    ):
                        raise RuntimeError("output_local component fit produced an invalid bias")
                    position_corrections[bias_key][row_slice] = bias_delta
                else:
                    if bias_delta.shape != current_state[bias_key].shape or not torch.isfinite(bias_delta).all():
                        raise RuntimeError("output_local component fit produced an invalid bias")
                    position_corrections[bias_key] = bias_delta

            desired_norm = desired_sq**0.5
            block_row: dict[str, Any] = {
                "mode": "direct_target",
                "component": component,
                "component_target": config.component_target,
                "position": pos,
                "source_coordinate": float(paired_source_index[pos]),
                "source_contributions": [(int(i), float(w)) for i, w in contributions],
                "procrustes_ranks": procrustes_ranks,
                "desired_norm": desired_norm,
                "effect_before_norm": 0.0,
                "relative_residual_before": (diag["residual_norm_before"] / desired_norm) if desired_norm else 0.0,
                "relative_residual_after": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                "correction_rank": int(torch.linalg.matrix_rank(correction.double()).item()),
                **diag,
            }
            if diagnose:
                update_norm = float(torch.linalg.norm(correction).item())
                weight_norm = float(torch.linalg.norm(target_w.detach().cpu().float()).item())
                # realized_target_norm_ratio compares the fitted prediction's own
                # norm (not its residual against D) to the target norm: an exact
                # readout of the accumulated fit, one extra pass over the
                # already-resident h_batches/bias (no new capture sweep).
                bias_for_pred = bias_correction if not skip_bias else torch.zeros(d_out)
                pred_sq = sum(
                    float(((h.double() @ correction.double().T + bias_for_pred.double()) ** 2).sum().item())
                    for h in h_batches
                )
                block_row.update(
                    {
                        "fit_relative_residual": (diag["residual_norm_after"] / desired_norm) if desired_norm else 0.0,
                        "target_norm": desired_norm,
                        "update_norm": update_norm,
                        "relative_update_norm": update_norm / (weight_norm + 1e-12),
                        "realized_target_norm_ratio": (pred_sq**0.5) / (desired_norm + 1e-12),
                    }
                )
            block_rows.append(block_row)
        results[pos] = (position_corrections, block_rows)
    return results


def _task_vector_sha256(sd: Mapping[str, torch.Tensor]) -> str:
    """Stable CPU hash of a task-vector-shaped tensor mapping.

    Identical algorithm to ``vision_rebase._state_dict_sha256`` (sorted keys,
    dtype, shape, raw bytes), duplicated here rather than imported to avoid a
    cycle (``vision_rebase`` imports ``direct_residual``, which imports this
    module). Any caller hashing the same dict with either function gets the
    same digest.
    """
    digest = hashlib.sha256()
    for key in sorted(sd):
        value = sd[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()))
    return digest.hexdigest()


def _family_bias_key(key: str) -> str | None:
    """Bias key paired with ``key``, or ``None`` if that projection has no bias."""
    if key.endswith("in_proj_weight"):
        return key[: -len("in_proj_weight")] + "in_proj_bias"
    if key.endswith(".weight"):
        return f"{key[: -len('.weight')]}.bias"
    return None


def compute_direct_residual_task_vector_stats(
    target_corrections: Mapping[str, torch.Tensor],
    target_base_state: Mapping[str, torch.Tensor],
    positions: list[int],
    components: tuple[str, ...] = CANONICAL_COMPONENT_ORDER,
    *,
    family_adapter=None,
) -> dict[str, Any]:
    """Analysis-only task-vector stats for one Direct Residual task vector.

    ``target_corrections`` is the unscaled (unit-strength) task vector
    ``fit_direct_residual`` returns. Every quantity here is derived purely
    from that dict and the pristine target base state -- nothing is
    re-fitted and no forward pass runs.

    ``n_modified_parameters`` counts touched numel per ``(position,
    component)`` pair actually present in ``target_corrections`` -- rather
    than per physical tensor -- so a packed ``in_proj_weight``/``in_proj_bias``
    correction that only ever wrote one q/k/v row-third (e.g. a
    ``component_target='output_local'`` run requesting only ``attn.v_proj``)
    is counted as ``d * d_in (+ d for bias)``, not ``3x`` that. Two distinct
    components can never double-count the same rows: each packed component
    owns a disjoint row slice by construction (``_PACKED_QKV_SLICE``).
    ``n_modified_tensors`` instead counts physical tensors (dict keys), so a
    packed parameter touched by more than one component is counted once.

    ``tau_norm_over_touched_base`` divides by the Frobenius norm of the base
    values at exactly the touched slices (row-aware for packed q/k/v);
    ``tau_norm_over_all_base`` divides by the Frobenius norm of the ENTIRE
    base state dict (every floating-point tensor), giving the task vector's
    size relative to the whole model rather than only what it touched.
    """
    shim = _layout_for(family_adapter)
    tau_norm_sq = 0.0
    for value in target_corrections.values():
        tau_norm_sq += float((value.detach().double() ** 2).sum().item())
    tau_norm = tau_norm_sq**0.5

    all_base_sq = 0.0
    for value in target_base_state.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            all_base_sq += float((value.detach().double() ** 2).sum().item())
    all_base_norm = all_base_sq**0.5

    touched_base_sq = 0.0
    # WARNING: q/k/v share ONE physical state-dict key (in_proj_weight/
    # in_proj_bias). A presence check keyed only off "is this key in
    # target_corrections" cannot by itself tell which of q/k/v were actually
    # fit -- e.g. a v-only run's in_proj_weight key exists with only its
    # v-rows nonzero, and iterating over q/k too (if they were included in
    # `components` despite never having been fit) would double- or triple-
    # count the very rows this function exists to avoid over-counting.
    # Callers MUST pass exactly the fitted family list (e.g.
    # `order_components(config.components)`), never a broader default, once
    # more than one of q/k/v could plausibly be absent.
    n_modified_parameters = 0
    for pos in positions:
        for component in components:
            key = shim.component_key(pos, component, prefixed=True)
            if key not in target_base_state:
                continue
            weight = target_base_state[key]
            bias_key = _family_bias_key(key)
            if component in _PACKED_QKV_SLICE:
                if key not in target_corrections and (bias_key is None or bias_key not in target_corrections):
                    continue
                d = weight.shape[0] // 3
                row_slice = slice(_PACKED_QKV_SLICE[component] * d, (_PACKED_QKV_SLICE[component] + 1) * d)
                w_slice = weight[row_slice]
                n_modified_parameters += w_slice.numel()
                touched_base_sq += float((w_slice.detach().double() ** 2).sum().item())
                if bias_key is not None and bias_key in target_base_state:
                    b_slice = target_base_state[bias_key][row_slice]
                    n_modified_parameters += b_slice.numel()
                    touched_base_sq += float((b_slice.detach().double() ** 2).sum().item())
            else:
                if key not in target_corrections:
                    continue
                n_modified_parameters += weight.numel()
                touched_base_sq += float((weight.detach().double() ** 2).sum().item())
                if bias_key is not None and bias_key in target_base_state:
                    n_modified_parameters += target_base_state[bias_key].numel()
                    touched_base_sq += float((target_base_state[bias_key].detach().double() ** 2).sum().item())
    touched_base_norm = touched_base_sq**0.5

    return {
        "n_modified_tensors": len(target_corrections),
        "n_modified_parameters": int(n_modified_parameters),
        "tau_norm": tau_norm,
        "tau_norm_over_touched_base": tau_norm / (touched_base_norm + 1e-12),
        "tau_norm_over_all_base": tau_norm / (all_base_norm + 1e-12),
        "tau_sha256": _task_vector_sha256(target_corrections),
    }


def _family_delta_state(
    shim,
    component: str,
    positions: list[int],
    target_corrections: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The subset of ``target_corrections`` belonging to one component family,
    zero-padded to full tensor shape (packed q/k/v: zero outside that
    component's own row slice)."""
    out: dict[str, torch.Tensor] = {}
    for pos in positions:
        key = shim.component_key(pos, component, prefixed=True)
        if key not in target_corrections:
            continue
        full = target_corrections[key]
        bias_key = _family_bias_key(key)
        if component in _PACKED_QKV_SLICE:
            d = base_state[key].shape[0] // 3
            row_slice = slice(_PACKED_QKV_SLICE[component] * d, (_PACKED_QKV_SLICE[component] + 1) * d)
            zeroed = torch.zeros_like(base_state[key])
            zeroed[row_slice] = full[row_slice]
            out[key] = zeroed
            if bias_key is not None and bias_key in target_corrections:
                zb = torch.zeros_like(base_state[bias_key])
                zb[row_slice] = target_corrections[bias_key][row_slice]
                out[bias_key] = zb
        else:
            out[key] = full.clone()
            if bias_key is not None and bias_key in target_corrections:
                out[bias_key] = target_corrections[bias_key].clone()
    return out


@torch.no_grad()
def measure_direct_residual_realization(
    target_model,
    target_base_state: Mapping[str, torch.Tensor],
    target_corrections: Mapping[str, torch.Tensor],
    positions: list[int],
    batches: list,
    target_outputs_by_position: Mapping[int, list],
    desired_by_position: Mapping[int, list],
    *,
    device,
    components: tuple[str, ...] = CANONICAL_COMPONENT_ORDER,
    family_adapter=None,
) -> dict[int, dict[str, Any]]:
    """Measure how well the fitted, unit-strength task vector ``tau`` actually
    realizes each position's desired block-boundary effect ``D_j``, on the
    FULL (nonlinear) target model -- as opposed to the fit's own internal
    linear-prediction diagnostics (``_realization_diagnostic_fields``), which
    never run a real forward pass through the block's nonlinearity.

    For the ``joint`` variant (all of ``tau`` mounted at once) and one variant
    per component family actually present in ``tau`` (a row-sliced,
    zero-elsewhere packed correction for q/k/v, the whole weight+bias
    otherwise), this mounts ``target_base_state + tau_variant``, captures the
    block-boundary output at every position in ``positions`` over ``batches``,
    and compares ``delta_j = T_j^variant - T_j^0`` against ``D_j`` (block-
    boundary desired effect; the same one every ``component_target`` fits
    against internally via ``compute_desired_effects``, always available
    regardless of ``component_target='block_boundary'`` vs ``'output_local'``
    since it only depends on the always-captured boundary banks).

    Per position ``j`` this returns:
      * ``block_realized_target_error``: ``||delta_j^joint - D_j||_F /
        (||D_j||_F + eps)``.
      * ``joint_delta_norm_over_desired``: ``||delta_j^joint||_F /
        (||D_j||_F + eps)``.
      * ``component_interaction_error``: ``||delta_j^joint - sum_c
        delta_j^(c)||_F / (||delta_j^joint||_F + eps)``, or ``None`` when only
        one family is present (there is then nothing to compare the joint
        variant against -- it IS that one family).
      * ``per_family_delta_norm_over_desired``: ``{component: ||delta_j^(c)||_F
        / (||D_j||_F + eps)}`` for every family present.

    WARNING on ``components``: q/k/v share ONE physical state-dict key
    (``in_proj_weight``/``in_proj_bias``). Presence-checking a family only by
    "is its key in ``target_corrections``" cannot by itself tell which of
    q/k/v were actually fit -- e.g. a v-only run's ``in_proj_weight`` key
    exists in ``target_corrections`` with only its v-rows nonzero, and if
    ``components`` also named q/k (despite neither ever being fit) they would
    be falsely reported "present" too. Callers MUST pass exactly the fitted
    family list (e.g. ``order_components(config.components)``), never the
    broad ``CANONICAL_COMPONENT_ORDER`` default, whenever more than one of
    q/k/v could plausibly be absent -- see
    ``vision_rebase._run_direct_residual_fit``.

    Memory bound: at most one variant's all-position boundary banks are held
    at a time (mounted, captured, reduced to a per-batch delta, and
    discarded), PLUS three running accumulators that survive across variants:
    the joint variant's own per-position delta bank (needed at the end for
    every family's interaction-error comparison), a running per-position sum
    of every family's delta bank (accumulated in place, one family at a time,
    so the sum of many families never requires holding more than one family's
    banks at once), and scalar running sums of squared norms. This is a
    batch-inner, variant-outer loop (all positions and batches captured
    together per variant, one variant fully finished before the next starts)
    rather than a batch-outer loop, since ``capture_tokens`` already captures
    every requested position from one shared forward sweep and splitting that
    sweep per batch would multiply the number of forward passes by
    ``len(batches)`` for no memory benefit (a single batch's own multi-
    position banks are already the smallest unit ``capture_tokens`` produces).

    The target model's entry state (whatever it held when this function was
    called, expected to be the pristine target base) is restored exactly in a
    ``finally``, and asserted via a state-dict hash before/after -- this
    function must never be observable by a caller as having left any residue
    on ``target_model``.
    """
    shim = _layout_for(family_adapter)
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    entry_hash = _task_vector_sha256(entry_state)
    base_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}

    def mount(delta: Mapping[str, torch.Tensor]) -> None:
        state = dict(base_state)
        for key, value in delta.items():
            state[key] = state[key] + value.to(state[key])
        target_model.load_state_dict(state, strict=True)

    def capture_all(delta: Mapping[str, torch.Tensor]) -> dict[int, list]:
        mount(delta)
        requests = {str(pos): (pos, "boundary") for pos in positions}
        raw = capture_tokens(target_model, batches, requests, device, family_adapter=family_adapter)
        return {pos: raw[str(pos)] for pos in positions}

    try:
        joint_banks = capture_all(target_corrections)
        joint_delta = {
            pos: [v - t0 for v, t0 in zip(joint_banks[pos], target_outputs_by_position[pos], strict=True)]
            for pos in positions
        }
        del joint_banks
        joint_norm_sq = {pos: sum(float((d.double() ** 2).sum().item()) for d in joint_delta[pos]) for pos in positions}

        present_families = [
            c
            for c in components
            if any(shim.component_key(pos, c, prefixed=True) in target_corrections for pos in positions)
        ]

        running_sum = {pos: [torch.zeros_like(d) for d in joint_delta[pos]] for pos in positions}
        family_norm_sq: dict[str, dict[int, float]] = {c: {} for c in present_families}
        for component in present_families:
            delta = _family_delta_state(shim, component, positions, target_corrections, base_state)
            banks = capture_all(delta)
            for pos in positions:
                deltas_c = [v - t0 for v, t0 in zip(banks[pos], target_outputs_by_position[pos], strict=True)]
                family_norm_sq[component][pos] = sum(float((d.double() ** 2).sum().item()) for d in deltas_c)
                for idx, d in enumerate(deltas_c):
                    running_sum[pos][idx] = running_sum[pos][idx] + d
            del banks, delta

        results: dict[int, dict[str, Any]] = {}
        for pos in positions:
            d_batches = desired_by_position[pos]
            d_norm_sq = sum(float((d.double() ** 2).sum().item()) for d in d_batches)
            d_norm = d_norm_sq**0.5
            joint_norm = joint_norm_sq[pos] ** 0.5
            err_sq = sum(
                float(((jd - dd).double() ** 2).sum().item())
                for jd, dd in zip(joint_delta[pos], d_batches, strict=True)
            )
            row: dict[str, Any] = {
                "position": pos,
                "desired_norm": d_norm,
                "joint_delta_norm": joint_norm,
                "block_realized_target_error": (err_sq**0.5) / (d_norm + 1e-12),
                "joint_delta_norm_over_desired": joint_norm / (d_norm + 1e-12),
                "per_family_delta_norm_over_desired": {
                    c: (family_norm_sq[c][pos] ** 0.5) / (d_norm + 1e-12) for c in present_families
                },
            }
            if len(present_families) > 1:
                interaction_sq = sum(
                    float(((jd - rs).double() ** 2).sum().item())
                    for jd, rs in zip(joint_delta[pos], running_sum[pos], strict=True)
                )
                row["component_interaction_error"] = (interaction_sq**0.5) / (joint_norm + 1e-12)
            else:
                row["component_interaction_error"] = None
            results[pos] = row
        return results
    finally:
        target_model.load_state_dict(entry_state, strict=True)
        exit_hash = _task_vector_sha256({k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()})
        if exit_hash != entry_hash:
            raise RuntimeError(
                "measure_direct_residual_realization failed to restore the target model's entry state exactly"
            )


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
        raise ValueError(f"complete_residuals_direct requires mode='direct_target', got {config.mode!r}")
    _validate_target_informed_layout(
        layout,
        target_scope=config.target_scope,
        target_trajectory=config.target_trajectory,
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
    batches = list(
        DataLoader(
            Subset(target_loader.dataset, meta["indices"]),
            batch_size=meta["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=target_loader.collate_fn,
        )
    )
    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections, diagnostics = {}, []
    try:
        # No transported vector: the temporary model *is* the native target
        # base. Asserted rather than assumed, because the whole claim of this
        # arm is that nothing but the fitted correction reaches the target.
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        components = order_components(config.components)
        for row in entries:
            pos = int(row["position"])
            if config.target_scope == "inserted" and pos != 2 * int(row["source_orig_idx"]) + 1:
                raise ValueError("Realized insertion ancestry does not match captured references")
        if config.cascade_order == "independent":
            # No cascade means no ordering constraint and no privileged
            # "first" position: the target model is never mutated between
            # fits (see _fit_all_positions_independent's docstring), so every
            # position's fit observes the same pristine base a per-position
            # loop would redundantly re-capture from scratch every time. Fit
            # all (position, component) pairs from one shared forward sweep
            # instead of up to len(entries) * len(components) of them.
            positions = [int(row["position"]) for row in entries]
            fitted = _fit_all_positions_independent(
                target_model,
                current_state,
                positions,
                coordinates,
                desired,
                target_outputs,
                batches,
                components,
                config,
                device,
                family_adapter=family_adapter,
            )
            entry_by_position = {int(row["position"]): row for row in entries}
            position_block_rows = [(pos, fitted[pos][0], fitted[pos][1]) for pos in positions]
        else:
            position_block_rows = []
            for index, row in enumerate(entries):
                pos = int(row["position"])
                # Components are fitted in block-forward order and cascaded, never
                # solved jointly. out_proj writes before the MLP, so mounting
                # Delta_O moves the MLP's own input through ln_2 and GELU -- a
                # nonlinearity no single linear system can absorb. Each component's
                # capture therefore *re-measures* the residual the previous one
                # actually left, exactly as the cross-block cascade does. The
                # per-position solve itself (capture -> Gram accumulation -> ridge
                # solve -> mount) is shared, unchanged, with Direct Residual's own
                # caller (`direct_residual.fit_direct_residual`) via Part A's
                # `_fit_direct_target_position`; only the ARIADNE-specific
                # provenance bookkeeping below (scope/trajectory/block_kind/
                # source_orig_idx) stays local to this function.
                position_corrections, block_rows = _fit_direct_target_position(
                    target_model,
                    current_state,
                    pos,
                    coordinates[pos],
                    desired[pos],
                    target_outputs[pos],
                    batches,
                    components,
                    config,
                    device,
                    family_adapter=family_adapter,
                    assert_pristine_effect=(index == 0),
                )
                position_block_rows.append((pos, position_corrections, block_rows))
            entry_by_position = {int(row["position"]): row for row in entries}
        for pos, position_corrections, block_rows in position_block_rows:
            row = entry_by_position[pos]
            target_corrections.update(position_corrections)
            for block_row in block_rows:
                block_row["scope"] = config.target_scope
                block_row["trajectory"] = config.target_trajectory
                block_row["target_coordinate"] = block_row.pop("source_coordinate")
                block_row["block_kind"] = block_kind[pos]
                block_row["source_orig_idx"] = row["source_orig_idx"]
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
            source_y_rows = torch.zeros(source_h_rows.shape[0], int(t_out.shape[0]), dtype=source_h_rows.dtype)
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
            bias_key = f"{key[: -len('.weight')]}.bias"
            if bias_key not in current_state:
                raise RuntimeError(
                    f"Joint blockwise affine correction requires a target c_proj.bias parameter; missing {bias_key}"
                )
            transported_bias = t_out.T @ bias.to(t_out)
            if (
                tuple(transported_bias.shape) != tuple(current_state[bias_key].shape)
                or not torch.isfinite(transported_bias).all()
            ):
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
        return list(
            DataLoader(
                Subset(loader.dataset, metadata["indices"]),
                batch_size=metadata["batch_size"],
                shuffle=False,
                num_workers=0,
                collate_fn=loader.collate_fn,
            )
        )

    source_batches, target_batches = _batches(source_loader), _batches(target_loader)
    native_source_outputs = references.get("source_base_cproj_outputs", {})
    native_target_outputs = references.get("target_base_outputs_by_position", {}) or references.get(
        "target_base_outputs", {}
    )
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
            source_capture = capture_tokens(source_base_model, source_batches, {"out": (pos, "c_proj")}, device)["out"]
            current_source = source_capture
            reference_source = _aligned(native_source_outputs[source_idx], source_capture)
            source_rows = _rows(current_source)
            source_residual = _rows(
                [reference - current for current, reference in zip(current_source, reference_source, strict=True)]
            )

            source_base_block = shim.block_module(shim.blocks(source_base_model)[pos])
            source_ft_block = shim.block_module(shim.blocks(source_ft_model)[pos])
            delta_weight = (
                (source_ft_block.mlp.c_proj.weight - source_base_block.mlp.c_proj.weight).detach().cpu().float()
            )
            delta_bias = (source_ft_block.mlp.c_proj.bias - source_base_block.mlp.c_proj.bias).detach().cpu().float()

            captured_target = capture_tokens(
                target_model,
                target_batches,
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
                source_rows,
                source_residual,
                z_target,
                target_residual,
                identity,
                effective_out,
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
            bias_key = f"{weight_key[: -len('.weight')]}.bias"
            transported_weight = t_out.T @ delta_weight_change @ t_in
            transported_bias = t_out.T @ delta_bias_change
            target_corrections[weight_key] = transported_weight
            target_corrections[bias_key] = transported_bias
            target_state[weight_key] = target_state[weight_key] + transported_weight.to(target_state[weight_key])
            target_state[bias_key] = target_state[bias_key] + transported_bias.to(target_state[bias_key])
            target_model.load_state_dict(target_state, strict=True)
            diagnostics.append(
                {
                    "position": pos,
                    "source_orig_idx": source_idx,
                    "shared_affine_bias_norm": float(torch.linalg.norm(shared_bias)),
                    "task_bias_change_norm": float(torch.linalg.norm(delta_bias_change)),
                    **{key: value for key, value in diag.items() if key != "bias_correction"},
                }
            )
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
    updated["resized_source_cproj_inputs_by_position"] = {int(position): bank for position, bank in captured.items()}
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
            raise ValueError(
                f"Missing native source reference for ancestry index {source_idx} at target position {pos}"
            )
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
            upper_delta = [f - b for b, f in zip(upper_base, _aligned(source_ft[upper], targets), strict=True)]
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
                lower_delta = [f - b for b, f in zip(lower_base, _aligned(source_ft[lower], targets), strict=True)]
            base_batches = [
                (1.0 - weight) * low + weight * high for low, high in zip(lower_base, upper_base, strict=True)
            ]
            delta_batches = [
                (1.0 - weight) * low + weight * high for low, high in zip(lower_delta, upper_delta, strict=True)
            ]

        # Q is fitted from whichever base bank the desired effect is expressed
        # against, so the alignment always matches the delta being transported.
        q, mu_s, mu_t = centered_rectangular_procrustes(_rows(base_batches).double(), _rows(targets).double())
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
    keys = (
        "source_clip_model",
        "source_clip_pretrained",
        "target_clip_model",
        "target_clip_pretrained",
        "method",
        "method_params",
        "seed",
        "val_fraction",
        "batch_size",
        "dtype",
    )
    protocol = {k: cfg.get(k) for k in keys}
    protocol["block_extension_params"] = asdict(resolve_block_extension_config(cfg)[1])
    shared = parse_target_shared_config(cfg.get("target_shared_correction"))
    protocol["target_shared_correction"] = asdict(shared) if shared.enabled and shared.target_weight > 0 else None
    implementation = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "eval/block_extension.py",
        "eval/target_informed_runtime.py",
        "eval/target_residual_completion.py",
        "rebase/methods/theseus.py",
        "rebase/methods/bico.py",
    ):
        implementation.update((root / relative).read_bytes())
    return {
        "schema_version": 1,
        "task": task,
        "target_hash": target_hash,
        "checkpoint": {
            "path": str(checkpoint),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _checkpoint_hash(str(checkpoint), stat.st_size, stat.st_mtime_ns),
        },
        "implementation_sha256": implementation.hexdigest(),
        "protocol": protocol,
    }


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


def iter_capture_block_gradients(
    model,
    batches,
    requests: Mapping[str, int],
    recipe,
    device,
    *,
    family_adapter=None,
):
    """Per-batch generator form of `capture_block_gradients` for the streaming path.

    Yields one ``{key: dL/dT_j}`` dict per batch, with exactly the forward+backward,
    ``zero_grad``, store op and ``_to_tokens`` layout of `capture_block_gradients`, but
    registers the backward hooks around EACH batch (like `iter_capture_tokens`) so it can
    run in lockstep with activation generators over the same model object. Deliberately
    not routed through `iter_capture_tokens`: that generator runs its forward under
    ``torch.no_grad()``, while this one needs the backward graph.
    """
    if family_adapter is not None:
        raise NotImplementedError("iter_capture_block_gradients is vision-only")
    layout = _layout_for(None)
    blocks = layout.blocks(model)
    training = model.training
    original_device = next(model.parameters()).device
    requires_grad_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        model.to(device).eval()
        for batch in batches:
            batch_size = layout.batch_size(batch)
            values: dict[str, list[torch.Tensor]] = {key: [] for key in requests}

            def store(name, tensor, *, _batch_size=batch_size, _values=values):
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                tokens = _to_tokens(tensor.detach(), batch_size=_batch_size)
                _values[name].append(tokens.float().cpu().clone())

            handles = []
            for key, index in requests.items():
                block = layout.block_module(blocks[index])

                def hook(_module, _grad_input, grad_output, *, name=key):
                    if grad_output is None or grad_output[0] is None:
                        raise RuntimeError(f"Block gradient hook {name!r} produced no output gradient")
                    store(name, grad_output[0])

                handles.append(block.register_full_backward_hook(hook))
            try:
                for p in model.parameters():
                    p.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(True):
                    loss, _ = recipe(model, batch)
                    if loss.dim() > 0:
                        loss = loss.sum()
                    loss.backward()
                model.zero_grad(set_to_none=True)
            finally:
                for handle in handles:
                    handle.remove()
                for name, p in model.named_parameters():
                    p.requires_grad_(requires_grad_flags[name])
            if any(len(v) != 1 for v in values.values()):
                raise RuntimeError("A requested block gradient hook did not fire exactly once per batch")
            yield {key: v[0] for key, v in values.items()}
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(requires_grad_flags[name])
        model.to(original_device).train(training)


def _realized_pred_sq_from_stats(stats, correction, bias_for_pred, effective_out) -> float:
    """``||(H C^T + 1 b^T) E||_F^2`` from `ResidualSufficientStatistics` alone.

    With ``A = H`` (``t_in=None``, the direct-target convention), the solver accumulates
    ``s = A^T A``, ``sum_a = A^T 1`` and ``n_rows``, so
    ``P^T P = C s C^T + (C sum_a) b^T + b (C sum_a)^T + n b b^T`` and the value is
    ``trace(E^T P^T P E)``. Equals `_realization_diagnostic_fields`'s bank-based sum up to
    floating-point summation order; used where no activation bank is resident (streaming).
    """
    if stats.s is None or stats.sum_a is None:
        raise ValueError("realization fields from stats require an accumulated ResidualSufficientStatistics")
    c = correction.double().to(stats.s.device)
    b = bias_for_pred.double().to(stats.s.device)
    e_out = effective_out.double().to(stats.s.device)
    c_sum_a = c @ stats.sum_a
    ptp = c @ stats.s @ c.T + torch.outer(c_sum_a, b) + torch.outer(b, c_sum_a) + float(stats.n_rows) * torch.outer(b, b)
    return float(torch.trace(e_out.T @ ptp @ e_out).item())


@torch.no_grad()
def measure_direct_residual_realization_streaming(
    target_model,
    target_base_state: Mapping[str, torch.Tensor],
    target_corrections: Mapping[str, torch.Tensor],
    positions: list[int],
    target_batches: list,
    desired_fn,
    source_iters_fn,
    *,
    device,
    components: tuple[str, ...] = CANONICAL_COMPONENT_ORDER,
    family_adapter=None,
) -> dict[int, dict[str, Any]]:
    """Streaming counterpart of `measure_direct_residual_realization` (same row schema).

    Instead of holding every variant's all-position boundary banks, runs ONE lockstep
    sweep over the calibration batches: the pristine target (``target_model`` loaded with
    ``target_base_state``), one deep copy of it per variant (joint ``tau`` plus one per
    component family present, each mounted as ``base + tau_variant``), and whatever source
    generators ``source_iters_fn()`` returns. Per batch, ``desired_fn(k, source_values,
    t0)`` recomputes ``{pos: D_j}`` (the streaming path never keeps a ``D_j`` bank), and
    every reported norm is accumulated as a per-batch sum of squares, so the result
    equals the resident one up to floating-point summation order (the per-batch
    arithmetic is identical; ``D_j`` itself differs only through the Chan-accumulated
    Procrustes map). The entry state of ``target_model`` is restored and hash-checked.
    """
    shim = _layout_for(family_adapter)
    entry_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    entry_hash = _task_vector_sha256(entry_state)
    base_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}

    def mounted_copy(delta: Mapping[str, torch.Tensor]):
        state = dict(base_state)
        for key, value in delta.items():
            state[key] = state[key] + value.to(state[key])
        model = copy.deepcopy(target_model)
        model.load_state_dict(state, strict=True)
        return model

    present_families = [
        c for c in components if any(shim.component_key(pos, c, prefixed=True) in target_corrections for pos in positions)
    ]
    requests = {str(pos): (pos, "boundary") for pos in positions}
    variants = {"joint": target_corrections}
    for component in present_families:
        variants[component] = _family_delta_state(shim, component, positions, target_corrections, base_state)
    copies = {name: mounted_copy(delta) for name, delta in variants.items()}

    d_sq = {pos: 0.0 for pos in positions}
    joint_sq = {pos: 0.0 for pos in positions}
    err_sq = {pos: 0.0 for pos in positions}
    fam_sq = {c: {pos: 0.0 for pos in positions} for c in present_families}
    inter_sq = {pos: 0.0 for pos in positions}
    try:
        target_model.load_state_dict(base_state, strict=True)
        gens = {"__t0__": iter_capture_tokens(target_model, target_batches, requests, device, family_adapter=family_adapter)}
        for name, model in copies.items():
            gens[name] = iter_capture_tokens(model, target_batches, requests, device, family_adapter=family_adapter)
        source_gens = source_iters_fn()
        names = list(gens)
        with contextlib.ExitStack() as stack:
            for gen in list(gens.values()) + list(source_gens.values()):
                stack.enter_context(contextlib.closing(gen))
            zipped = zip(*(gens[n] for n in names), *source_gens.values(), strict=True)
            for k, values in enumerate(zipped):
                by_name = dict(zip(names, values[: len(names)], strict=True))
                source_values = dict(zip(source_gens, values[len(names):], strict=True))
                t0 = by_name["__t0__"]
                desired = desired_fn(k, source_values, t0)
                for pos in positions:
                    base_out = t0[str(pos)]
                    d = desired[pos]
                    jd = by_name["joint"][str(pos)] - base_out
                    d_sq[pos] += float((d.double() ** 2).sum().item())
                    joint_sq[pos] += float((jd.double() ** 2).sum().item())
                    err_sq[pos] += float(((jd - d).double() ** 2).sum().item())
                    running = torch.zeros_like(jd)
                    for component in present_families:
                        dc = by_name[component][str(pos)] - base_out
                        fam_sq[component][pos] += float((dc.double() ** 2).sum().item())
                        running = running + dc
                    if len(present_families) > 1:
                        inter_sq[pos] += float(((jd - running).double() ** 2).sum().item())
        results: dict[int, dict[str, Any]] = {}
        for pos in positions:
            d_norm = d_sq[pos] ** 0.5
            joint_norm = joint_sq[pos] ** 0.5
            row: dict[str, Any] = {
                "position": pos,
                "desired_norm": d_norm,
                "joint_delta_norm": joint_norm,
                "block_realized_target_error": (err_sq[pos] ** 0.5) / (d_norm + 1e-12),
                "joint_delta_norm_over_desired": joint_norm / (d_norm + 1e-12),
                "per_family_delta_norm_over_desired": {
                    c: (fam_sq[c][pos] ** 0.5) / (d_norm + 1e-12) for c in present_families
                },
                "component_interaction_error": (
                    (inter_sq[pos] ** 0.5) / (joint_norm + 1e-12) if len(present_families) > 1 else None
                ),
            }
            results[pos] = row
        return results
    finally:
        del copies
        target_model.load_state_dict(entry_state, strict=True)
        exit_hash = _task_vector_sha256({k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()})
        if exit_hash != entry_hash:
            raise RuntimeError(
                "measure_direct_residual_realization_streaming failed to restore the target model's entry state exactly"
            )
