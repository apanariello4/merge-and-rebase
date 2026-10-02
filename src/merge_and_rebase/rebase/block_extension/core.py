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

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .schedules import spread_anchor_schedule


class BlockExtenderCore:
    _LOG_PREFIX = "block_extension"
    _EXPECTED_PREFIX = "Expected one of: "
    _ridge_weight = 1e-6

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
