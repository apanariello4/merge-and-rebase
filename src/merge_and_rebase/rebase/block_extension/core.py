"""Shared leaves of the BRACE block extenders.

``BlockExtender`` (OpenCLIP vision) and ``DecoderBlockExtender`` (HF decoder) inherit the pieces that
were byte-identical between them: verbose logging, calibration hooks, the ridge solve, row matching,
depth-delta resolution, per-component ridge lookup and the duplication schedule.

The two known differences are kept as class attributes rather than unified:

* ``_ridge_weight``: the ridge ``_fit_ridge`` falls back to when ``lambda_reg`` is ``None``. The vision
  extender overwrites it from the ``ridge_weight`` config field; the decoder never plumbs that field, so
  its constant ``1e-6`` stays in force.
* ``_EXPECTED_PREFIX`` / ``_LOG_PREFIX``: wording of the validation messages and the log prefix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .adapters import ComponentAdapter, ComponentSpec
from .schedules import spread_anchor_schedule


class EagerProvider:
    """Reference activations of every block of both endpoints, captured once before the structural loop.

    This is the decoder policy. The captures live on the extender (``capture_reference_inputs`` and
    ``_capture_component_references`` fill ``reference_inputs``), so ``ensure`` has nothing left to do.
    """

    def __init__(self, extender: Any):
        self.extender = extender

    def start(self, loader: Iterable[Any], n_batches: int) -> None:
        self.extender.capture_reference_inputs(loader, n_batches)
        self.extender._capture_component_references(loader, n_batches)
        self.extender._vprint("reference activation and component capture completed")

    def ensure(self, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


class LazyProvider:
    """Pristine CPU endpoint copies; only the references the current structural step needs are captured.

    This is the vision policy (``reference_capture`` ``lazy`` and ``eager`` are both handled inside the
    extender's ``_capture_per_weight_reference_subset``).
    """

    def __init__(self, extender: Any):
        self.extender = extender
        self.models: dict[str, nn.Module] = {}

    def start(self, loader: Iterable[Any], n_batches: int) -> None:
        del loader, n_batches
        self.models = {
            "base": deepcopy(self.extender.model_base).cpu(),
            "ft": deepcopy(self.extender.model_ft).cpu(),
        }
        self.extender._vprint("created pristine CPU reference endpoints for lazy capture")

    def ensure(
        self,
        *,
        endpoints: Sequence[str],
        block_indices: Sequence[int],
        input_indices: Sequence[int | str],
        loader: Iterable[Any],
        n_batches: int,
    ) -> None:
        self.extender._capture_per_weight_reference_subset(
            self.models,
            endpoints=endpoints,
            block_indices=block_indices,
            input_indices=input_indices,
            loader=loader,
            n_batches=n_batches,
        )

    def finish(self) -> None:
        self.models.clear()


class BlockExtenderCore:
    _LOG_PREFIX = "block_extension"
    _EXPECTED_PREFIX = "Expected one of: "
    _ridge_weight = 1e-6
    adapter: ComponentAdapter

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(f"[{self._LOG_PREFIX}] {message}")

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

    def _get_ridge(self, component: str, default: float) -> float:
        cr = getattr(self, "_component_ridge", None)
        if cr is None:
            return default
        return float(cr.get(component, default))

    def _record_correction(self, endpoint: str, component: str, W: torch.Tensor, b: torch.Tensor) -> None:
        """Diagnostic side channel; the vision extender overrides it."""

    def _record_corrections(self, endpoint: str, corrections: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        for component, (W, b) in corrections.items():
            self._record_correction(endpoint, component, W, b)

    # ---- references and lmc dispatch -----------------------------------------------------------------------------

    def _make_reference_provider(self) -> EagerProvider | LazyProvider:
        raise NotImplementedError

    def _open_references(
        self, skip_correction: bool, loader: Iterable[Any], n_batches: int
    ) -> EagerProvider | LazyProvider | None:
        if skip_correction:
            self._vprint("skip_correction enabled: skipping reference capture")
            return None
        provider = self._make_reference_provider()
        provider.start(loader, n_batches)
        return provider

    @staticmethod
    def _reference_endpoint_names(lmc_mode: str, share_ft_refs: bool) -> tuple[str, ...]:
        if lmc_mode == "shared":
            return ("ft",) if share_ft_refs else ("base",)
        if lmc_mode == "shared_ft" or share_ft_refs:
            return ("ft",)
        return ("base", "ft")

    def _dispatch_lmc(
        self,
        lmc_mode: str,
        share_ft_refs: bool,
        position: int,
        correct: Callable[..., None],
    ) -> None:
        """Run ``correct(model_name, model, ref_source=, lmc_store=, lmc_targets=)`` for the lmc mode.

        ``independent``: both endpoints fitted on their own references. ``steer``: the ft fit is pulled
        towards the base maps. ``shared`` / ``shared_ft``: one endpoint is fitted and its maps applied to
        the other.
        """
        base_ref = "ft" if share_ft_refs else None
        if lmc_mode == "independent":
            correct("base", self.model_base, ref_source=base_ref)
            correct("ft", self.model_ft)
        elif lmc_mode == "steer":
            base_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            correct("base", self.model_base, ref_source=base_ref, lmc_store=base_corrections)
            correct("ft", self.model_ft, lmc_targets=base_corrections)
        elif lmc_mode == "shared":
            base_corrections = {}
            correct("base", self.model_base, ref_source=base_ref, lmc_store=base_corrections)
            self._apply_block_corrections(self.model_ft, position, base_corrections)
            self._record_corrections("ft", base_corrections)
        elif lmc_mode == "shared_ft":
            ft_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            correct("ft", self.model_ft, lmc_store=ft_corrections)
            self._apply_block_corrections(self.model_base, position, ft_corrections)
            self._record_corrections("base", ft_corrections)
        else:
            raise ValueError(
                f"Unsupported lmc_mode '{lmc_mode}'. Expected 'independent', 'steer', 'shared', or 'shared_ft'."
            )

    @torch.no_grad()
    def _correct_one_block(
        self,
        *,
        provider: EagerProvider | LazyProvider,
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

        ``position`` is the block's index in the current chain; ``ref_block_idx`` is the original block whose
        pristine activations are the target. For an inserted block those differ; for an original block being
        repaired they refer to the same block at its shifted position.
        """
        provider.ensure(
            endpoints=self._reference_endpoint_names(lmc_mode, share_ft_refs),
            block_indices=(ref_block_idx,),
            input_indices=(),
            loader=loader,
            n_batches=n_batches,
        )

        def correct(model_name: str, model: nn.Module, **lmc: Any) -> None:
            self._correct_block_weights_cascade(
                model_name,
                model,
                position,
                ref_block_idx,
                loader,
                n_batches,
                ridge_identity=ridge_identity,
                n_iters=n_cascade_iters,
                component_ridge=component_ridge,
                insertion_target_mode=insertion_target_mode,
                target_reference=target_reference,
                target_weight=target_weight,
                **lmc,
            )

        self._dispatch_lmc(lmc_mode, share_ft_refs, position, correct)

    # ---- correction cascade (one loop over ``adapter.components``) -------------------------------------------------

    @staticmethod
    def _ridge_target(
        spec: ComponentSpec, lmc_targets: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None
    ) -> torch.Tensor | None:
        if lmc_targets is None or spec.name not in lmc_targets:
            return None
        W_base, _ = lmc_targets[spec.name]
        if spec.kind == "norm_diag":
            return torch.diag(torch.diag(W_base))
        return W_base

    def _stream_pair(
        self,
        spec: ComponentSpec,
        cur: torch.Tensor,
        adds: list[torch.Tensor | None],
        model: nn.Module,
        pos: int,
        loader: Iterable[Any],
        n_batches: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual-stream target ``adds[0] + adds[1] + ... - cur_input [- cur_attn]`` against ``cur``.

        The block input (and, for the MLP output, the already-corrected attention output) actually reaching
        the component is subtracted, so the gap between the block input and the reference is absorbed once.
        Captures happen in the order input, then attention.
        """
        subs = [self._capture_single_input(model, pos, loader, n_batches)]
        if spec.stream_role == "mlp_out":
            subs.append(self._capture_component_output(model, pos, "attn", loader, n_batches))
        if cur.numel() == 0 or any(x is None for x in adds) or any(x.numel() == 0 for x in subs):
            return torch.empty(0), torch.empty(0)
        n = min(cur.shape[0], *(x.shape[0] for x in adds), *(x.shape[0] for x in subs))
        A = cur[:n]
        T = adds[0][:n].to(A.device)
        for x in adds[1:]:
            T = T + x[:n].to(A.device)
        for x in subs:
            T = T - x[:n].to(A.device)
        return A, T

    def _fit_and_correct(
        self,
        model_name: str,
        block: nn.Module,
        spec: ComponentSpec,
        A: torch.Tensor,
        T: torch.Tensor,
        ridge_identity: float,
        lmc_store: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
        lmc_targets: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        if not (A.numel() > 0 and T.numel() > 0):
            return
        W, b = self._fit_ridge(
            A,
            T,
            ridge_id=self._get_ridge(spec.name, ridge_identity),
            ridge_target=self._ridge_target(spec, lmc_targets),
        )
        if lmc_store is not None:
            lmc_store[spec.name] = (W.clone(), b.clone())
        self._record_correction(model_name, spec.name, W, b)
        self.adapter.apply_correction(block, spec, W, b)

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
        block = self.adapter.layers(model)[insert_pos]
        refs = self.reference_inputs[ref_source if ref_source is not None else model_name]
        self._component_ridge = component_ridge
        for _ in range(n_iters):
            for spec in self.adapter.components:
                cur = self._capture_component_output(
                    model, insert_pos, spec.capture_name or spec.name, loader, n_batches
                )
                ref = refs.get(f"{src_idx}.{spec.ref_key}")
                if spec.blendable_target and ref is not None and target_reference is not None and target_weight > 0.0:
                    ref = self._blend_target_reference(
                        ref, target_reference, self._target_reference_samples, target_weight
                    )
                if spec.residual_aware and insertion_target_mode == "residual":
                    # Pin the post-attention (out_proj) / block-output (c_proj) residual stream to the source
                    # block's instead of matching the component alone.
                    adds = [refs.get(f"{src_idx}.input")]
                    if spec.stream_role == "mlp_out":
                        adds.append(refs.get(f"{src_idx}.attn_output"))
                    adds.append(ref)
                    A, T = self._stream_pair(spec, cur, adds, model, insert_pos, loader, n_batches)
                elif ref is not None and cur.numel() > 0:
                    A, T = self._match_rows(cur, ref)
                else:
                    A = T = torch.empty(0)
                self._fit_and_correct(model_name, block, spec, A, T, ridge_identity, lmc_store, lmc_targets)

    @torch.no_grad()
    def _apply_block_corrections(
        self,
        model: nn.Module,
        insert_pos: int,
        corrections: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ):
        block = self.adapter.layers(model)[insert_pos]
        for spec in self.adapter.components:
            if spec.name in corrections:
                self.adapter.apply_correction(block, spec, *corrections[spec.name])

    @classmethod
    def _build_duplication_schedule(
        cls,
        curr_layers: int,
        n_needed: int,
        insertion_order: str,
        extension_density: str,
    ) -> list[int]:
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
                f"Unsupported insertion_order. {cls._EXPECTED_PREFIX}bottom-top, top-bottom, random. "
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
                f"Unsupported extension_density. {cls._EXPECTED_PREFIX}spread, spread_mod, clump. "
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
