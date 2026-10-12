"""Activation / gradient capture and paired calibration for Ariadne."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset

from ....utils.cost_accounting import cost_phase
from ...discrete_layer_match import DiscreteLayerPairing
from .._shared import _to_tokens
from .layouts import _ATTN_CAPTURE_KINDS, _ATTN_QUERY_KINDS, _CAPTURE_KINDS, _INPUT_CAPTURE_KINDS, _layout_for


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


def _stable_dataset_identity(dataset) -> str:
    """Process-independent identity string of a calibration dataset (for run metadata).

    Unlike `_dataset_identity` (which may end in ``id(dataset)``, valid only for the in-process same-object
    pairing check), this is content-based (Subset indices / sample ids hashed, split fingerprint) and falls back
    to ``module.Class`` plus length, never an object id, so it is reproducible across processes. It is what
    ``extra["calibration"]["dataset_identity"]`` records and what the later same-dataset checks in
    ``target_informed_runtime`` compare against.
    """
    if isinstance(dataset, Subset):
        indices = hashlib.sha256(repr([int(i) for i in dataset.indices]).encode()).hexdigest()
        return f"Subset(indices_sha256={indices};n={len(dataset.indices)};{_stable_dataset_identity(dataset.dataset)})"
    split = getattr(dataset, "split", None)
    fingerprint = getattr(split, "_fingerprint", None)
    if fingerprint is not None:
        return f"split_fingerprint={fingerprint};n={len(dataset)}"
    ids = getattr(dataset, "sample_ids", None)
    if ids is not None:
        digest = hashlib.sha256(repr([str(i) for i in ids]).encode()).hexdigest()
        return f"sample_ids_sha256={digest};n={len(ids)}"
    cls = type(dataset)
    return f"{cls.__module__}.{cls.__qualname__};n={len(dataset)}"


#: Batch-dict key carrying the combined (source & target) content-row mask of a decoder calibration batch.
#: ``extract_calibration_batch`` ignores it, so model forwards never see it; capture reads it to drop padding rows.
ROW_MASK_KEY = "_ariadne_row_mask"

#: Optional calibration-batch key (opt-in ``calibration_fit_mask``): bool ``[B, T]``, True = token inside a message
#: content. It is NOT in ``extract_calibration_batch``'s whitelist, so no model forward ever receives it;
#: ``_attach_row_masks`` ANDs it into the row mask, which every capture path (independent, streaming, block_carry) reads.
FIT_MASK_KEY = "fit_mask"


def _batch_fit_mask(batch):
    if isinstance(batch, Mapping):
        mask = batch.get(FIT_MASK_KEY)
        if isinstance(mask, torch.Tensor):
            return mask.bool()
    return None


def _batch_row_mask(batch):
    if isinstance(batch, Mapping):
        mask = batch.get(ROW_MASK_KEY)
        if isinstance(mask, torch.Tensor):
            return mask
    return None


def _first_content_position(mask):
    """Bool ``[B, T]`` marking the first ``True`` of every row (all ``False`` for a row without content).

    Taken from the mask itself, so it is the first non-padded position for left AND right padding.
    """
    first = torch.zeros_like(mask)
    has_content = mask.any(dim=1)
    idx = mask.to(torch.uint8).argmax(dim=1)  # first maximum = first True
    rows = torch.nonzero(has_content).squeeze(1)
    first[rows, idx[rows]] = True
    return first


def _attach_row_masks(source_batch, target_batch, family_adapter, exclude_first_token=False, fit_stats=None):
    """Combine each model's own ``content_mask`` (``ms & mt``) once and attach it to BOTH batches.

    ``exclude_first_token`` additionally drops the first non-padded position of every sequence (the attention
    sink) of EACH model, from the one combined mask, so source and target keep identical row sets. Returns
    ``(source, target, n_total, n_content, n_excluded)``; ``n_content`` is the count before the exclusion.

    A ``fit_mask`` (``calibration_fit_mask``) present on the batches is ANDed into the mask (identical on source and
    target, whose token ids are paired); the rows it removes (relative to the padding mask) are added to
    ``fit_stats["n_fit_masked"]`` when a dict is given, and ``n_excluded`` then counts the first-token exclusion only.
    """
    ms = family_adapter.content_mask(source_batch)
    mt = family_adapter.content_mask(target_batch)
    if ms.shape != mt.shape:
        raise ValueError(
            f"Source/target calibration content masks have different shapes {tuple(ms.shape)} vs {tuple(mt.shape)}: "
            "rows cannot be paired after padding removal (tokenizer or max_length mismatch)"
        )
    mask = ms & mt
    n_content = int(mask.sum())
    fs, ft = _batch_fit_mask(source_batch), _batch_fit_mask(target_batch)
    if (fs is None) != (ft is None):
        raise ValueError("calibration_fit_mask: only one of the source/target batches carries a fit_mask")
    if fs is not None:
        if fs.shape != mask.shape or not torch.equal(fs, ft):
            raise ValueError(
                "calibration_fit_mask: source and target fit masks differ (the paired token ids must be identical)"
            )
        mask = mask & fs
        if fit_stats is not None:
            fit_stats["n_fit_masked"] = fit_stats.get("n_fit_masked", 0) + n_content - int(mask.sum())
    n_after_fit = int(mask.sum())
    if exclude_first_token:
        mask = mask & ~_first_content_position(ms) & ~_first_content_position(mt)
    if not bool(mask.any()):
        raise ValueError(
            "Calibration batch has no content rows after padding removal"
            + (" and first-token exclusion" if exclude_first_token else "")
            + (" and the calibration fit mask" if fs is not None else "")
        )
    out = []
    for batch in (source_batch, target_batch):
        wrapped = dict(family_adapter.extract_calibration_batch(batch))
        wrapped[ROW_MASK_KEY] = mask
        out.append(wrapped)
    return out[0], out[1], int(mask.numel()), n_content, n_after_fit - int(mask.sum())


def paired_calibration(
    source_loader, target_loader, *, num_batches, seed=None, family_adapter=None, exclude_first_token=False
):
    """Replay identical dataset indices under the two model preprocessors.

    ``family_adapter`` (HF decoders only) attaches the combined padding mask to every batch pair; the vision path
    (``None``) returns its batches untouched. ``exclude_first_token`` (decoder only) also drops the first non-padded
    position of every sequence from the attached mask; the metadata then carries ``n_rows_excluded_first_token``
    and ``n_rows_content`` counts the rows that remain.
    """
    if exclude_first_token and family_adapter is None:
        raise ValueError("exclude_first_token is decoder-only (needs the family adapter's attention mask)")
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
    n_rows_total = n_rows_content = n_rows_excluded = 0
    fit_stats: dict[str, int] = {}
    for a, b in zip(source, target, strict=True):
        # Vision batches are (images, labels); labels are model-agnostic and must match example for example.
        # Text batches are tokenizer-specific mappings (ids differ by construction), so nothing is comparable;
        # the examples are pinned by the shared `order` and the _dataset_identity check above.
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) > 1 and len(b) > 1:
            if not torch.equal(torch.as_tensor(a[1]), torch.as_tensor(b[1])):
                raise ValueError("Calibration labels disagree for supposedly identical images")
        if family_adapter is not None:
            a, b, total, content, excluded = _attach_row_masks(
                a, b, family_adapter, exclude_first_token, fit_stats
            )
            n_rows_total += total
            n_rows_content += content
            n_rows_excluded += excluded
        source_batches.append(a)
        target_batches.append(b)
    metadata = {
        "indices": order,
        "dataset_identity": _stable_dataset_identity(source_loader.dataset),
        "requested_batches": num_batches,
        "actual_batches": len(source_batches),
        "batch_size": source_loader.batch_size,
        "sampling_seed": seed,
        "split": "val",
    }
    if family_adapter is not None:
        metadata["n_rows_total"] = n_rows_total
        metadata["n_rows_content"] = n_rows_content - n_rows_excluded - fit_stats.get("n_fit_masked", 0)
        # pad_row_fraction counts padding only (n_rows_content here is before first-token exclusion).
        metadata["pad_row_fraction"] = 1.0 - n_rows_content / max(n_rows_total, 1)
        if exclude_first_token:
            metadata["n_rows_excluded_first_token"] = n_rows_excluded
        if "n_fit_masked" in fit_stats:  # only when calibration_fit_mask is on: default metadata is unchanged
            metadata["n_rows_excluded_fit_mask"] = fit_stats["n_fit_masked"]
    return source_batches, target_batches, metadata


def _assert_layerscale_identity(shim, block, *, context, requirement="block_split='joint'"):
    """Refuse a nontrivial LayerScale under ``block_split='joint'`` (see ``_fit_block_boundary_joint``).

    The joint fit solves with ``t_out = I``: a non-identity ``ls_1``/``ls_2`` would be silently dropped.
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

    torch applies the output projection functionally, so the ``out_proj`` submodule is never called and a
    forward hook on it never fires. Rerunning the attention with an identity projection recovers exactly those
    rows; the caller verifies this against the module's own output (`_verify_recomputed_attention_input`).
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


def _stock_mha_head_mixed_input(module, args, kwargs):
    """Per-head attention-mixed inputs ``Z_h = A_h X`` of a stock ``nn.MultiheadAttention`` (self-attention), ``[B, T, H*d]``.

    ``A_h`` are the softmax probabilities (rows sum to 1, ``need_weights=True`` / ``average_attn_weights=False``, i.e. the
    explicit math path) and ``X`` the attention input (ln_1 output). Head ``h`` occupies columns ``h*d:(h+1)*d``. Because the
    rows of ``A_h`` sum to 1, a change ``(dV_h, db_h)`` of head ``h``'s V slice moves the pre-``out_proj`` rows exactly by
    ``Z_h dV_h^T + 1 db_h^T``. The result is ALWAYS batch-first ``[B, T, H*d]``.

    Verified by the hook (`_verify_mixed_input`) against the module's own output and the out-proj-input recomputation.
    """
    query, key, value = _mha_query_key_value(args, kwargs)
    if not (query is key and query is value):
        raise ValueError(
            "attn_mixed_input capture requires self-attention (query, key and value must be the same tensor)"
        )
    if not module._qkv_same_embed_dim or module.bias_k is not None or module.add_zero_attn:
        raise NotImplementedError(
            "attn_mixed_input capture needs a packed-in_proj nn.MultiheadAttention without bias_k"
        )
    if bool(getattr(module, "batch_first", False)):
        query = query.transpose(1, 0)
    identity = torch.eye(module.embed_dim, dtype=query.dtype, device=query.device)
    _, weights = F.multi_head_attention_forward(
        query,
        query,
        query,
        module.embed_dim,
        module.num_heads,
        module.in_proj_weight,
        module.in_proj_bias,
        None,
        None,
        False,
        module.dropout,
        identity,
        None,
        training=module.training,
        key_padding_mask=kwargs.get("key_padding_mask"),
        need_weights=True,
        attn_mask=kwargs.get("attn_mask"),
        average_attn_weights=False,
    )  # weights: [B, H, T, T]
    x = query.transpose(1, 0)  # [B, T, d]
    z = torch.einsum("bhts,bsd->bthd", weights, x)
    return z.reshape(z.shape[0], z.shape[1], -1)


def _v_apply_mixed(module, z):
    """Pre-``out_proj`` rows ``concat_h(Z_h W_V,h^T + b_V,h)`` implied by mixed inputs ``z`` ``[B, T, H*d]``."""
    d = module.embed_dim
    heads = module.num_heads
    wv = module.in_proj_weight[2 * d :].reshape(heads, d // heads, d)
    bv = module.in_proj_bias[2 * d :].reshape(heads, d // heads) if module.in_proj_bias is not None else None
    zh = z.reshape(z.shape[0], z.shape[1], heads, d)
    rows = torch.einsum("bthd,hed->bthe", zh, wv)
    if bv is not None:
        rows = rows + bv
    return rows.reshape(z.shape[0], z.shape[1], -1)


def _verify_mixed_input(module, args, kwargs, mixed, module_output):
    """The V-slice image of the mixed inputs must equal the recomputed out-proj input (and hence the module output)."""
    rows = _v_apply_mixed(module, mixed)
    reference = _stock_mha_out_proj_input(module, args, kwargs)
    batch_first = bool(getattr(module, "batch_first", False))
    if not batch_first:
        reference = reference.transpose(1, 0)
    if rows.shape != reference.shape or not torch.allclose(rows, reference, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            "Per-head mixed attention inputs do not reproduce the out_proj input: the captured Z_h are not "
            "the features the V projection acts on"
        )
    _verify_recomputed_attention_input(module, rows if batch_first else rows.transpose(1, 0), module_output)


def _verify_recomputed_attention_input(module, rows, module_output):
    """Push the recomputed rows back through the real ``out_proj`` and compare (atol/rtol 1e-4).

    Cheap, and turns a future change in torch's attention internals into a loud failure, not a wrong fit.
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

    Returns the handles the caller must ``.remove()``.
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
            # A stock nn.MultiheadAttention applies out_proj functionally, so a hook on that submodule never
            # fires (and the "once per batch" check would reject the capture): hook the attention itself and
            # recover the projection's rows exactly.
            recompute = isinstance(attention, nn.MultiheadAttention)
            if kind == "attn_mixed_input" and not recompute:
                raise NotImplementedError("attn_mixed_input capture requires a stock nn.MultiheadAttention (vision)")
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
                elif capture_kind == "attn_mixed_input":
                    mixed = _stock_mha_head_mixed_input(mod, args, kwargs)
                    _verify_mixed_input(mod, args, kwargs, mixed, value)
                    store(name, mixed)
                else:
                    store(name, value)

            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        elif query_capture:

            def hook(_mod, args, kwargs, _value, *, name=key):
                query, key_arg, value_arg = _mha_query_key_value(args, kwargs)
                if not (query is key_arg and query is value_arg):
                    raise ValueError(
                        "attn_input capture requires self-attention (query, key and value must be the same tensor)"
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
    """Yield one ``{key: [B,T,D] float32 tensor}`` dict per batch instead of accumulating all of them.

    Same hook resolution as `capture_tokens`, but hooks are registered and removed around EACH batch, so several
    generators over the same model object can run in lockstep without cross-firing. ``store_device="cpu"`` stores
    ``tokens.float().cpu().clone()``; any other value stores via ``tokens.float().to(store_device).clone()``.
    Restores the model's device/train mode on completion, early ``.close()``, GC, or an exception mid-sweep.
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

            _row_mask = _batch_row_mask(batch)

            def store(name, tensor, *, _batch_size=batch_size, _values=values, _row_mask=_row_mask):
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                tokens = _to_tokens(tensor.detach(), batch_size=_batch_size)
                if _row_mask is not None:
                    if tuple(tokens.shape[:2]) != tuple(_row_mask.shape):
                        raise ValueError(
                            f"captured tokens {tuple(tokens.shape[:2])} do not match row mask {tuple(_row_mask.shape)}"
                        )
                    # Content rows only, kept as a [1, N, D] bank so every downstream `_rows`/`_aligned` is unchanged.
                    tokens = tokens[_row_mask.to(tokens.device)].unsqueeze(0)
                if store_device == "cpu":
                    tokens = tokens.float().cpu().clone()
                else:
                    tokens = tokens.float().to(store_device).clone()
                _values[name].append(tokens)

            handles = _register_capture_hooks(layout, blocks, requests, store, family_adapter=family_adapter)
            try:
                with torch.no_grad(), cost_phase("activation_collection"):
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
    """Capture ``{key: [per-batch [B,T,D] tensors]}``, releasing hooks and restoring placement on errors.

    ``family_adapter=None`` selects the CLIP paths; passing one selects the HF-decoder equivalents
    (`_DecoderLayout`). The "c_proj" kinds keep their names on both and resolve to ``mlp.down_proj`` on a decoder.
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
    """Capture block-output gradients ``dL/dT_j`` for ``procrustes_source="gradient"``.

    Per batch, runs forward+backward of ``recipe(model, batch) -> (scalar_loss, named_params)`` (a
    ``models.grad_recipes.GradRecipe``, mean-reduced CE as in BiCo) and stores each requested block's
    ``grad_output[0]`` (the same tensor as `capture_tokens`'s ``"boundary"`` kind) as CPU float32 ``[B,T,D]``.
    ``requests`` maps a key to a block index.

    Follows `rebase.methods.bico._collect_batch`: ``zero_grad(set_to_none=True)`` before and after each
    backward, so no parameter gradient is retained. ``requires_grad`` is forced ``True`` for the capture and
    restored, together with train mode and device, on return or exception. Parameter values are never mutated.
    Vision only: ``NotImplementedError`` for a non-``None`` ``family_adapter``.
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
            with torch.set_grad_enabled(True), cost_phase("activation_collection"):
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
                with torch.set_grad_enabled(True), cost_phase("activation_collection"):
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


def capture_paired_boundary_activations(
    source_base_model,
    source_ft_model,
    target_base_model,
    source_loader,
    target_loader,
    pairing: DiscreteLayerPairing,
    *,
    num_batches: int,
    seed: int | None,
    device,
    family_adapter=None,
    procrustes_source: str = "activation",
    source_recipe=None,
    target_recipe=None,
    exclude_first_token: bool = False,
) -> dict[str, Any]:
    """Capture native boundary activations for every position Ariadne fits.

    For every target position ``j`` in ``range(pairing.target_depth)``, captures ``source_base``/``source_ft``
    boundary activations at ``pairing.pairing[j]`` (deduplicated: under extend several target positions share
    one source index) and ``target_base`` activations at ``j``, via `capture_tokens`/`paired_calibration`.
    No model is resized and no correction is fitted; the models are used at their native depths.

    ``procrustes_source="gradient"`` additionally captures block-boundary gradients (``dL/dT_i`` on
    ``source_base_model`` at each distinct paired source index, ``dL/dT_j`` on ``target_base_model`` at every
    position) with `capture_block_gradients`, using ``source_recipe``/``target_recipe`` (both required) on the
    SAME paired batches; ``source_ft_model`` is never differentiated. Stored under
    ``"source_base_gradients"``/``"target_base_gradients"``, keyed by index. Vision only.

    The result also carries the replayed target-side batches (``"target_batches"``) so `fit_direct_residual`
    can re-run forward passes on the partially-corrected target model without the original `target_loader`.
    """
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if pairing.source_depth < 1:
        raise ValueError("pairing.source_depth must be positive")
    if procrustes_source not in {"activation", "gradient"}:
        raise ValueError(f"procrustes_source must be 'activation' or 'gradient', got {procrustes_source!r}")
    if procrustes_source == "gradient" and (source_recipe is None or target_recipe is None):
        raise ValueError("procrustes_source='gradient' requires both source_recipe and target_recipe")
    source_batches, target_batches, metadata = paired_calibration(
        source_loader,
        target_loader,
        num_batches=num_batches,
        seed=seed,
        family_adapter=family_adapter,
        exclude_first_token=exclude_first_token,
    )
    # Deduplicate: under extend many target positions share one source index; capturing it twice would waste
    # compute and risk a second, non-identical bank for the same index if anything upstream were nondeterministic.
    distinct_source_indices = sorted(set(pairing.pairing))
    source_requests = {str(i): (i, "boundary") for i in distinct_source_indices}
    target_requests = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    source_base_raw = capture_tokens(
        source_base_model, source_batches, source_requests, device, family_adapter=family_adapter
    )
    source_ft_raw = capture_tokens(
        source_ft_model, source_batches, source_requests, device, family_adapter=family_adapter
    )
    target_raw = capture_tokens(
        target_base_model, target_batches, target_requests, device, family_adapter=family_adapter
    )
    result = {
        "source_base_outputs": {int(k): v for k, v in source_base_raw.items()},
        "source_ft_outputs": {int(k): v for k, v in source_ft_raw.items()},
        "target_base_outputs_by_position": {int(k): v for k, v in target_raw.items()},
        "target_batches": target_batches,
        "calibration": metadata,
    }
    if procrustes_source == "gradient":
        source_grad_requests = {str(i): i for i in distinct_source_indices}
        target_grad_requests = {str(j): j for j in range(pairing.target_depth)}
        source_base_grad_raw = capture_block_gradients(
            source_base_model,
            source_batches,
            source_grad_requests,
            source_recipe,
            device,
            family_adapter=family_adapter,
        )
        target_base_grad_raw = capture_block_gradients(
            target_base_model,
            target_batches,
            target_grad_requests,
            target_recipe,
            device,
            family_adapter=family_adapter,
        )
        result["source_base_gradients"] = {int(k): v for k, v in source_base_grad_raw.items()}
        result["target_base_gradients"] = {int(k): v for k, v in target_base_grad_raw.items()}
    return result
