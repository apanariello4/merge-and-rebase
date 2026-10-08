"""Block-carry implementation of the sequential Ariadne depth fit (``sequential_impl="block_carry"``, decoders only).

Same estimator, same arithmetic, fewer forwards than `streaming.fit_direct_residual_sequential_streaming`
(``sequential_impl="reference"``). The reference runs, for EVERY target position j and every calibration batch, four
full backbone forwards (source base, source FT, the current target carrying the updates of blocks < j, and a pristine
deep copy of the target), i.e. ``4 * N * K`` forwards of ``L`` blocks each, although only block j's input and
output change from one position to the next. Here every stream carries its exact block input forward instead:

* current target: the input of block j under the updates of blocks < j is the output of block j - 1 (already
  updated) applied to the input of block j - 1 under the updates of blocks < j - 1, so it is obtained by running
  block j - 1 alone, after its update is mounted, on the carried input (induction on j);
* pristine target: block j still holds its base weights while position j is being fitted (its update is mounted
  only afterwards), so the pristine stream runs on ``target_model`` itself and no deep copy is needed;
* source base / source FT: never mutated; each stream is advanced block by block to the paired index
  ``i = pairing[j]`` (restarted from the embedding if the pairing ever decreases) and the boundary capture is reused
  while ``i`` repeats (target deeper than the source).

Each executed block receives the bit-identical carried tensor (model dtype, stored losslessly on the host) and the
very arguments a full forward gives it: the backbone's own forward builds the embeddings, masks and rotary tables
from the same batch, the blocks before the first requested one are replaced by the identity and the forward is
aborted right after the last requested one. Captures go through the same hooks and the same row-mask / float32
store as `capture.iter_capture_tokens`, and ``D_j``, ``U_j``, ``E_j`` and the statistics are formed with the
reference's own per-batch arithmetic (on the host, in the same order), so the fitted task vector equals the
reference's bit for bit when the kernels are deterministic (pinned on CPU by
tests/test_ariadne_sequential_fast_20261008.py).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from ....utils.cost_accounting import cost_phase
from ...discrete_layer_match import DiscreteLayerPairing
from .._shared import _to_tokens
from .capture import _batch_row_mask, _register_capture_hooks
from .config import DirectResidualConfig, order_components
from .fit import ResidualSufficientStatistics, _finalize_independent_component
from .layouts import COMPONENT_INPUT_KIND, _aligned, _component_effective_out, _layout_for
from .streaming import _check_boundary_fingerprint

logger = logging.getLogger(__name__)

Tensor = torch.Tensor
_NO_INSTANCE_FORWARD = object()


class _StopAfterBlock(Exception):
    """Raised by the hook on the last requested block: the rest of the backbone is not needed."""


def _identity_block_forward(hidden_states, *args, **kwargs):
    return hidden_states


class _BlockRangeRunner:
    """Run blocks ``start..stop`` (inclusive) of a decoder backbone on carried block inputs, one batch at a time.

    ``executed_blocks`` / ``partial_forwards`` count the work done (benchmark and tests).
    """

    def __init__(self, family_adapter, device, carry_device="cpu"):
        self.layout = _layout_for(family_adapter)
        self.family_adapter = family_adapter
        self.device = device
        self.carry_device = carry_device
        self.executed_blocks = 0
        self.partial_forwards = 0

    def run(
        self,
        model,
        batch,
        *,
        start: int,
        stop: int,
        inject: Tensor | None,
        requests: Mapping[str, tuple[int, str]],
        carry: bool,
    ) -> tuple[dict[str, Tensor], Tensor | None]:
        """Return ``(captures, carried)``: the requested captures (exactly as `iter_capture_tokens` stores them, on
        ``device``) and, if ``carry``, block ``stop``'s raw output (model dtype, on ``carry_device``).

        ``inject`` is the exact input of block ``start``; ``None`` is only valid for ``start == 0`` (the model's own
        embedding output is the input).
        """
        if not 0 <= start <= stop:
            raise ValueError(f"invalid block range {start}..{stop}")
        if inject is None and start != 0:
            raise ValueError("a block range not starting at block 0 needs the carried block input")
        layout = self.layout
        blocks = layout.blocks(model)
        batch_size = layout.batch_size(batch)
        row_mask = _batch_row_mask(batch)
        values: dict[str, list[Tensor]] = {key: [] for key in requests}
        device = self.device

        # Same store as capture.iter_capture_tokens(store_device=device).
        def store(name, tensor):
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            tokens = _to_tokens(tensor.detach(), batch_size=batch_size)
            if row_mask is not None:
                if tuple(tokens.shape[:2]) != tuple(row_mask.shape):
                    raise ValueError(
                        f"captured tokens {tuple(tokens.shape[:2])} do not match row mask {tuple(row_mask.shape)}"
                    )
                tokens = tokens[row_mask.to(tokens.device)].unsqueeze(0)
            if device == "cpu":
                tokens = tokens.float().cpu().clone()
            else:
                tokens = tokens.float().to(device).clone()
            values[name].append(tokens)

        carried: list[Tensor] = []
        handles = _register_capture_hooks(layout, blocks, requests, store, family_adapter=self.family_adapter)
        patched: list[tuple[Any, Any]] = []
        try:
            for index in range(start):
                block = blocks[index]
                patched.append((block, block.__dict__.get("forward", _NO_INSTANCE_FORWARD)))
                block.forward = _identity_block_forward
            if inject is not None:
                carried_input = inject.to(device)

                def pre_hook(_module, args, kwargs):
                    original = args[0] if args else kwargs["hidden_states"]
                    if carried_input.shape != original.shape or carried_input.dtype != original.dtype:
                        raise RuntimeError(
                            f"carried block input {tuple(carried_input.shape)}/{carried_input.dtype} does not match "
                            f"the block's input {tuple(original.shape)}/{original.dtype}"
                        )
                    if args:
                        return (carried_input, *args[1:]), kwargs
                    return args, {**kwargs, "hidden_states": carried_input}

                handles.append(blocks[start].register_forward_pre_hook(pre_hook, with_kwargs=True))

            def counter(_module, _inputs):
                self.executed_blocks += 1

            for index in range(start, stop + 1):
                handles.append(blocks[index].register_forward_pre_hook(counter))

            def stop_hook(_module, _inputs, output):
                if carry:
                    out = output[0] if isinstance(output, tuple) else output
                    carried.append(out.detach().to(self.carry_device, copy=True))
                raise _StopAfterBlock

            handles.append(blocks[stop].register_forward_hook(stop_hook))
            self.partial_forwards += 1
            try:
                with torch.no_grad(), cost_phase("activation_collection"):
                    layout.forward(model, batch, device)
            except _StopAfterBlock:
                pass
            else:
                raise RuntimeError(f"the partial forward never reached block {stop}")
        finally:
            for handle in handles:
                handle.remove()
            for block, previous in patched:
                if previous is _NO_INSTANCE_FORWARD:
                    del block.forward
                else:
                    block.forward = previous
        if any(len(v) != 1 for v in values.values()):
            raise RuntimeError("A requested activation hook did not fire exactly once per batch")
        return {key: v[0] for key, v in values.items()}, (carried[0] if carry else None)


class _SourceStream:
    """One frozen source model, carried block by block to the paired boundary of each target position."""

    def __init__(self, model, num_batches: int):
        self.model = model
        self.next_block = 0  # index of the block whose input `inputs` holds
        self.inputs: list[Tensor | None] = [None] * num_batches
        self.boundary_index: int | None = None
        self.boundary: list[Tensor | None] = [None] * num_batches

    def plan(self, i: int) -> tuple[int, int] | None:
        """``None`` to reuse the held boundary ``i``; else the ``(start, stop)`` block range to run."""
        if self.boundary_index == i:
            return None
        start = self.next_block if i >= self.next_block else 0
        return start, i

    def advance(self, runner: _BlockRangeRunner, batch, k: int, i: int, plan: tuple[int, int] | None) -> Tensor:
        if plan is None:
            return self.boundary[k]
        start, stop = plan
        captures, carried = runner.run(
            self.model,
            batch,
            start=start,
            stop=stop,
            inject=None if start == 0 else self.inputs[k],
            requests={str(i): (i, "boundary")},
            carry=True,
        )
        self.inputs[k] = carried
        self.boundary[k] = captures[str(i)].cpu()
        return self.boundary[k]

    def commit(self, i: int, plan: tuple[int, int] | None) -> None:
        if plan is not None:
            self.next_block = i + 1
            self.boundary_index = i


@torch.no_grad()
def fit_direct_residual_sequential_block_carry(
    target_model,
    target_base_state: Mapping[str, Tensor],
    source_base_model,
    source_ft_model,
    prepared: Mapping[str, Any],
    pairing: DiscreteLayerPairing,
    *,
    config: DirectResidualConfig,
    device,
    family_adapter=None,
    runner: _BlockRangeRunner | None = None,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    """Drop-in replacement for `fit_direct_residual_sequential_streaming` on decoders (same inputs, same returns).

    Per position j and batch k: one block-j forward of the current target (captures boundary j and the
    down-projection input), one block-j forward of the pristine stream (boundary j, carried as the next pristine
    input), the source streams advanced to ``pairing[j]``; after the fit and the mount of block j's update, one more
    block-j forward of the current target produces the next carried input. ``runner`` (tests) supplies the counters.
    """
    if family_adapter is None:
        raise ValueError("sequential_impl='block_carry' is decoder-only (the vision path keeps the reference)")
    if pairing.target_depth < 1:
        raise ValueError("pairing.target_depth must be positive")
    if config.depth_fit != "sequential":
        raise ValueError("fit_direct_residual_sequential_block_carry requires depth_fit='sequential'")
    components = order_components(config.components)
    if len(components) != 1:
        raise ValueError("depth_fit='sequential' requires exactly one component")
    component = components[0]
    batches = prepared["target_batches"]
    source_batches = prepared["source_batches"]
    q_by_position = prepared["q_by_position"]
    fingerprints = prepared["fingerprints"]
    positions = list(range(pairing.target_depth))
    for j in positions:
        if j not in q_by_position:
            raise ValueError(f"Missing prepared Procrustes map for position {j}")
    source_coordinates = {j: float(pairing.pairing[j]) for j in positions}
    shim = _layout_for(family_adapter)
    runner = runner if runner is not None else _BlockRangeRunner(family_adapter, device)

    unique_models: dict[int, Any] = {}
    for m in (target_model, source_base_model, source_ft_model):
        unique_models.setdefault(id(m), m)
    originals = {mid: (next(m.parameters()).device, m.training) for mid, m in unique_models.items()}

    original_state = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    target_corrections: dict[str, Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    frozen_copy = None
    try:
        current_state = {k: v.detach().cpu().clone() for k, v in target_base_state.items()}
        target_model.load_state_dict(current_state, strict=True)
        if source_base_model is target_model or source_ft_model is target_model:
            # A source model that is the target object must not see the mounted updates (reference: pristine copy).
            frozen_copy = deepcopy(target_model)
            frozen_copy.load_state_dict({k: v.detach().cpu().clone() for k, v in target_base_state.items()})
        sb_model = frozen_copy if source_base_model is target_model else source_base_model
        sf_model = frozen_copy if source_ft_model is target_model else source_ft_model
        for m in {id(x): x for x in (target_model, sb_model, sf_model) if x is not None}.values():
            m.to(device).eval()

        num_batches = len(batches)
        if len(source_batches) != num_batches:
            raise ValueError("source and target calibration batch counts differ")
        sb_stream = _SourceStream(sb_model, num_batches)
        sf_stream = _SourceStream(sf_model, num_batches)
        current_inputs: list[Tensor | None] = [None] * num_batches
        pristine_inputs: list[Tensor | None] = [None] * num_batches

        for pos in positions:
            i = pairing.pairing[pos]
            q = q_by_position[pos]
            key = shim.component_key(pos, component, prefixed=True)
            width = int(current_state[key].shape[0])
            effective_out = _component_effective_out(shim, target_model, pos, component, width)
            stats = ResidualSufficientStatistics(device=device)
            desired_sq = 0.0
            upstream_sq = 0.0
            out_key, h_key = f"{pos}.out", f"{pos}.{component}.h"
            tgt_requests = {out_key: (pos, "boundary"), h_key: (pos, COMPONENT_INPUT_KIND[component])}
            ref_requests = {out_key: (pos, "boundary")}
            sb_plan, sf_plan = sb_stream.plan(i), sf_stream.plan(i)
            next_pristine: list[Tensor | None] = [None] * num_batches
            for k, (source_batch, batch) in enumerate(zip(source_batches, batches, strict=True)):
                sb_boundary = sb_stream.advance(runner, source_batch, k, i, sb_plan)
                sf_boundary = sf_stream.advance(runner, source_batch, k, i, sf_plan)
                tgt, _ = runner.run(
                    target_model,
                    batch,
                    start=pos,
                    stop=pos,
                    inject=current_inputs[k],
                    requests=tgt_requests,
                    carry=False,
                )
                # Block `pos` still holds its base weights here, so this is the pristine model's block `pos`.
                ref, next_pristine[k] = runner.run(
                    target_model,
                    batch,
                    start=pos,
                    stop=pos,
                    inject=pristine_inputs[k],
                    requests=ref_requests,
                    carry=pos + 1 < len(positions),
                )
                # Per-batch arithmetic identical to fit_direct_residual_sequential_streaming.
                _check_boundary_fingerprint(ref[out_key], fingerprints[(k, pos)], config, pos, k)
                out_cpu = tgt[out_key].cpu()
                b = _aligned([sb_boundary], [out_cpu])[0]
                f = _aligned([sf_boundary], [out_cpu])[0]
                desired = (f - b) @ q
                upstream = out_cpu - ref[out_key].cpu()
                error = desired - upstream
                desired_sq += float((desired.double() ** 2).sum().item())
                upstream_sq += float((upstream.double() ** 2).sum().item())
                h = tgt[h_key].cpu()
                stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), None, effective_out)
            sb_stream.commit(i, sb_plan)
            sf_stream.commit(i, sf_plan)
            pristine_inputs = next_pristine

            position_corrections: dict[str, Tensor] = {}
            block_rows: list[dict[str, Any]] = []
            _finalize_independent_component(
                stats,
                desired_sq,
                0.0,
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
            )
            target_corrections.update(position_corrections)
            for row in block_rows:
                row["source_position"] = row.pop("source_coordinate")
                row["procrustes_source"] = prepared.get("procrustes_source", "activation")
                row["depth_fit"] = "sequential"
                row["upstream_effect_norm"] = upstream_sq**0.5
                diagnostics.append(row)
            target_model.load_state_dict({name: current_state[name] for name in position_corrections}, strict=False)
            if pos + 1 < len(positions):
                # Block `pos` now carries its update: its output on the carried input is the next block's input.
                for k, batch in enumerate(batches):
                    _, current_inputs[k] = runner.run(
                        target_model,
                        batch,
                        start=pos,
                        stop=pos,
                        inject=current_inputs[k],
                        requests={},
                        carry=True,
                    )
        logger.info(
            "Ariadne sequential (block_carry): %d partial forwards, %d executed blocks over %d positions x %d batches",
            runner.partial_forwards,
            runner.executed_blocks,
            len(positions),
            len(batches),
        )
    finally:
        target_model.load_state_dict(original_state, strict=True)
        for mid, m in unique_models.items():
            orig_device, orig_training = originals[mid]
            m.to(orig_device).train(orig_training)
        del frozen_copy
    return target_corrections, diagnostics


__all__ = ["fit_direct_residual_sequential_block_carry"]
