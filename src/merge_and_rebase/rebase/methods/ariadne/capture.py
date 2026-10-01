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
from ..theseus import _to_tokens
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

    `_dataset_identity` ends in ``("object", id(dataset), len)`` for datasets that carry neither a split
    fingerprint nor ``sample_ids`` -- right for the in-process source/target pairing check (it requires the
    SAME dataset object), but not reproducible across processes, so it must not be written to run summaries.
    This variant is content-based wherever the dataset exposes content (Subset indices, split fingerprint,
    sample ids -- the latter two and the Subset indices are hashed rather than listed) and otherwise falls
    back to ``module.Class`` plus length, never an object id. It is what ``extra["calibration"]
    ["dataset_identity"]`` records and what the later same-dataset checks in ``target_informed_runtime``
    compare against.
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
        "dataset_identity": _stable_dataset_identity(source_loader.dataset),
        "requested_batches": num_batches,
        "actual_batches": len(source_batches),
        "batch_size": source_loader.batch_size,
        "sampling_seed": seed,
        "split": "val",
    }
    return source_batches, target_batches, metadata


def _assert_layerscale_identity(shim, block, *, context, requirement="block_split='joint'"):
    """Refuse a nontrivial LayerScale under ``block_split='joint'`` (see
    ``_fit_block_boundary_joint``).

    The joint fit solves with ``t_out = I``: any nonidentity ``ls_1``/
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
) -> dict[str, Any]:
    """Capture native boundary activations for every position Direct Residual fits.

    For every target position ``j`` in ``range(pairing.target_depth)``,
    captures ``source_base``/``source_ft`` boundary activations at
    ``pairing.pairing[j]`` -- deduplicated, since under extend many target
    positions share one source index and there is no reason to run the same
    forward pass twice -- and ``target_base`` boundary activations at ``j``
    itself. Uses `capture_tokens`/`paired_calibration` from
    `target_informed_runtime` unchanged: both already operate on
    ``(model, batches, requests)``/``(source_loader, target_loader)``, never
    on an ARIADNE realized-extension layout, so nothing here reaches into
    that machinery.

    No structural resize of any model happens (the three models passed in are
    used exactly as given, at their native depths) and no correction is
    fitted -- this function only captures activation banks.

    ``procrustes_source="gradient"`` (``DirectResidualConfig.procrustes_source``)
    additionally captures block-boundary GRADIENTS -- ``dL/dT_i`` on
    ``source_base_model`` at every distinct paired source index, and
    ``dL/dT_j`` on ``target_base_model`` at every target position -- via
    ``ariadne.capture.capture_block_gradients``, using
    ``source_recipe``/``target_recipe`` (each model's own
    ``models.grad_recipes.clip_contrastive_recipe``, exactly BiCo's
    recipe/statistic) on the SAME paired calibration batches the activation
    banks above use. ``source_ft_model`` is never used for gradients (BiCo
    only ever differentiates through base models). Stored under
    ``"source_base_gradients"``/``"target_base_gradients"``, keyed by index
    like the activation banks. Both recipes are required in gradient mode.
    Vision only: `capture_block_gradients` raises `NotImplementedError` for a
    non-``None`` ``family_adapter``.

    The returned dict also carries the replayed target-side calibration
    batches (under ``"target_batches"``) so `fit_direct_residual` can re-run
    forward passes against the (possibly partially-corrected) target model
    during the solve without needing the original `target_loader` again.

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
        source_loader, target_loader, num_batches=num_batches, seed=seed
    )
    # Deduplicate: under extend, many target positions share one source
    # index, and capturing it twice would be wasted compute and (worse) a
    # second, potentially non-identical activation bank for the same index
    # if anything upstream were ever non-deterministic.
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
