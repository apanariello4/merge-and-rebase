from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from .block_extension import BlockExtensionConfig, _deterministic_calibration_loader

logger = logging.getLogger(__name__)

_DECODER_COMPONENTS = (
    "input_layernorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "post_attention_layernorm",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _iter_with_progress(iterable: Any, *, total: int, desc: str, enabled: bool) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _run_decoder_forward(model: nn.Module, batch: Mapping[str, Any], device: str | torch.device) -> None:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)


def _get_layers(model: nn.Module, family_adapter: Any) -> nn.ModuleList:
    scope = family_adapter.transport_scope(model)
    return scope.layers


def _get_final_norm(model: nn.Module, family_adapter: Any) -> nn.Module:
    scope = family_adapter.transport_scope(model)
    return scope.norm


class DecoderBlockExtender:
    def __init__(
        self,
        model_base: nn.Module,
        model_ft: nn.Module,
        family_adapter: Any,
        device: str | torch.device,
        *,
        verbose: bool = True,
        show_progress: bool = True,
    ):
        self.model_base = model_base
        self.model_ft = model_ft
        self.family_adapter = family_adapter
        self.device = device
        self.reference_inputs: dict[str, dict[str, torch.Tensor]] = {"base": {}, "ft": {}}
        self.verbose = bool(verbose)
        self.show_progress = bool(show_progress)
        self._component_ridge: dict[str, float] | None = None

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(f"[block_extension_llm] {message}")

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

    @staticmethod
    def _fit_ridge(
        A: torch.Tensor,
        T: torch.Tensor,
        lambda_reg: float = 1e-6,
        ridge_id: float = 0.0,
        ridge_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        A = A.float()
        T = T.float()

        mu_A = A.mean(dim=0)
        mu_T = T.mean(dim=0)

        A_c = A - mu_A
        T_c = T - mu_T

        dim_in = A.shape[1]
        reg = lambda_reg + ridge_id
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
    def _match_rows(A: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def _build_duplication_schedule(
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
                f"Unsupported insertion_order. Expected: bottom-top, top-bottom, random. Got: {insertion_order}"
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
                f"Unsupported extension_density. Expected: spread, clump. Got: {extension_density}"
            )

        schedule: list[int] = []
        while len(schedule) < n_needed:
            if insertion_order == "random":
                cycle = list(range(curr_layers))
                np.random.shuffle(cycle)
            else:
                cycle = priority
            need = n_needed - len(schedule)
            schedule.extend(cycle[:need])
        return schedule

    @staticmethod
    def _build_collapse_schedule(
        curr_layers: int,
        n_to_remove: int,
        insertion_order: str,
        extension_density: str,
    ) -> list[int]:
        if n_to_remove <= 0:
            return []
        if curr_layers < 2:
            raise ValueError("Cannot collapse blocks when the model depth is less than 2.")

        max_anchor = curr_layers - 2
        if extension_density == "clump":
            if insertion_order == "top-bottom":
                return [max_anchor] * n_to_remove
            if insertion_order == "random":
                return [int(np.random.randint(0, max_anchor + 1)) for _ in range(n_to_remove)]
            if insertion_order != "bottom-top":
                raise ValueError(
                    f"Unsupported insertion_order. Expected: bottom-top, top-bottom, random. Got: {insertion_order}"
                )
            return [0] * n_to_remove

        if extension_density == "spread_mod":
            n_gaps = curr_layers - 1
            return [i % n_gaps for i in range(n_to_remove)]

        if extension_density != "spread":
            raise ValueError(
                f"Unsupported extension_density. Expected: spread, clump. Got: {extension_density}"
            )

        schedule: list[int] = []
        if insertion_order == "top-bottom":
            priority = list(range(max_anchor, -1, -1))
        elif insertion_order == "random":
            priority = list(range(max_anchor + 1))
            np.random.shuffle(priority)
        else:
            priority = list(range(max_anchor + 1))

        while len(schedule) < n_to_remove:
            need = n_to_remove - len(schedule)
            schedule.extend(priority[:need])
            if insertion_order == "random":
                np.random.shuffle(priority)
        return schedule

    @staticmethod
    def _locate_collapse_pos(chain: list[dict[str, Any]], anchor_orig_idx: int) -> int:
        for i, item in enumerate(chain):
            if anchor_orig_idx in item["orig_idxs"]:
                return i
        raise ValueError(f"Could not locate anchor_orig_idx={anchor_orig_idx} in chain.")

    @torch.no_grad()
    def _interpolate_block_weights(self, target_block: nn.Module, source_block: nn.Module, alpha: float = 0.5):
        source_params = dict(source_block.named_parameters())
        for name_t, p_t in target_block.named_parameters():
            if name_t.startswith("aligner."):
                continue
            p_s = source_params.get(name_t)
            if p_s is not None and p_t.shape == p_s.shape:
                p_t.copy_((1.0 - alpha) * p_t + alpha * p_s)

    @torch.no_grad()
    def _dampen_block_output(self, block: nn.Module, factor: float):
        if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
            block.self_attn.o_proj.weight.mul_(factor)
            if hasattr(block.self_attn.o_proj, "bias") and block.self_attn.o_proj.bias is not None:
                block.self_attn.o_proj.bias.mul_(factor)
        if hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"):
            block.mlp.down_proj.weight.mul_(factor)
            if hasattr(block.mlp.down_proj, "bias") and block.mlp.down_proj.bias is not None:
                block.mlp.down_proj.bias.mul_(factor)

    @torch.no_grad()
    def capture_reference_inputs(self, loader: Iterable[Any], n_batches: int):
        for name, model in [("base", self.model_base), ("ft", self.model_ft)]:
            self._vprint(f"capture reference inputs ({name}) with n_batches={n_batches}")
            model.eval()
            store: dict[str, list[torch.Tensor]] = defaultdict(list)
            hooks: list[Any] = []

            layers = _get_layers(model, self.family_adapter)
            for i in range(len(layers)):
                hooks.append(layers[i].register_forward_hook(self._store_input_hook(store, f"{i}.input")))

            final_norm = _get_final_norm(model, self.family_adapter)
            hooks.append(final_norm.register_forward_hook(self._store_input_hook(store, "final.input")))

            it = iter(loader)
            for _ in _iter_with_progress(
                range(n_batches),
                total=n_batches,
                desc=f"block_extension_llm.capture.{name}",
                enabled=self.show_progress,
            ):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                inputs = self.family_adapter.extract_calibration_batch(batch)
                _run_decoder_forward(model, inputs, self.device)

            for h in hooks:
                h.remove()

            refs: dict[str, torch.Tensor] = {}
            for key, tensors in store.items():
                refs[key] = torch.cat(tensors, dim=0).flatten(0, 1)
            self.reference_inputs[name] = refs

    @torch.no_grad()
    def _capture_component_references(self, loader: Iterable[Any], n_batches: int):
        for name, model in [("base", self.model_base), ("ft", self.model_ft)]:
            self._vprint(f"capture component references ({name}) with n_batches={n_batches}")
            model.eval()
            store: dict[str, list[torch.Tensor]] = defaultdict(list)
            hooks: list[Any] = []

            layers = _get_layers(model, self.family_adapter)
            for i in range(len(layers)):
                block = layers[i]
                hooks.append(block.input_layernorm.register_forward_hook(
                    self._store_output_hook(store, f"{i}.input_layernorm_output")
                ))
                hooks.append(block.self_attn.register_forward_hook(
                    self._store_output_hook(store, f"{i}.attn_output")
                ))
                hooks.append(block.post_attention_layernorm.register_forward_hook(
                    self._store_output_hook(store, f"{i}.post_attn_ln_output")
                ))
                hooks.append(block.mlp.gate_proj.register_forward_hook(
                    self._store_output_hook(store, f"{i}.gate_proj_output")
                ))
                hooks.append(block.mlp.up_proj.register_forward_hook(
                    self._store_output_hook(store, f"{i}.up_proj_output")
                ))
                hooks.append(block.mlp.down_proj.register_forward_hook(
                    self._store_output_hook(store, f"{i}.down_proj_output")
                ))

            it = iter(loader)
            for _ in _iter_with_progress(
                range(n_batches),
                total=n_batches,
                desc=f"block_extension_llm.capture_components.{name}",
                enabled=self.show_progress,
            ):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                inputs = self.family_adapter.extract_calibration_batch(batch)
                _run_decoder_forward(model, inputs, self.device)

            for h in hooks:
                h.remove()

            refs: dict[str, torch.Tensor] = {}
            for key, tensors in store.items():
                refs[key] = torch.cat(tensors, dim=0).flatten(0, 1)
            self.reference_inputs[name].update(refs)

    @torch.no_grad()
    def _capture_single_input(self, model: nn.Module, target: int | str, loader: Iterable[Any], n_batches: int) -> torch.Tensor:
        model.eval()
        buffers: list[torch.Tensor] = []

        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any):
            if inputs and inputs[0] is not None:
                buffers.append(inputs[0].detach().cpu())

        if target == "final":
            final_norm = _get_final_norm(model, self.family_adapter)
            handle = final_norm.register_forward_hook(hook)
        else:
            layers = _get_layers(model, self.family_adapter)
            handle = layers[target].register_forward_hook(hook)

        it = iter(loader)
        for _ in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            inputs = self.family_adapter.extract_calibration_batch(batch)
            _run_decoder_forward(model, inputs, self.device)

        handle.remove()
        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    @torch.no_grad()
    def _capture_component_output(
        self, model: nn.Module, block_idx: int, component: str, loader: Iterable[Any], n_batches: int
    ) -> torch.Tensor:
        model.eval()
        buffers: list[torch.Tensor] = []
        layers = _get_layers(model, self.family_adapter)
        block = layers[block_idx]

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
                f"Unsupported component '{component}'. Expected one of: {', '.join(_DECODER_COMPONENTS)}"
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
            inputs = self.family_adapter.extract_calibration_batch(batch)
            _run_decoder_forward(model, inputs, self.device)

        handle.remove()
        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    def _get_ridge(self, component: str, default: float) -> float:
        if self._component_ridge is None:
            return default
        return float(self._component_ridge.get(component, default))

    @torch.no_grad()
    def _correct_rmsnorm(self, ln: nn.Module, W: torch.Tensor, b: torch.Tensor) -> None:
        d = torch.diag(W).to(ln.weight.device, dtype=ln.weight.dtype)
        ln.weight.mul_(d)
        if hasattr(ln, "bias") and ln.bias is not None:
            b = b.to(ln.bias.device, dtype=ln.bias.dtype)
            ln.bias.copy_(d * ln.bias + b)

    @torch.no_grad()
    def _correct_linear(self, linear: nn.Linear, W: torch.Tensor, b: torch.Tensor) -> None:
        W = W.to(linear.weight.device, dtype=linear.weight.dtype)
        linear.weight.copy_(W @ linear.weight)
        if linear.bias is not None:
            b = b.to(linear.bias.device, dtype=linear.bias.dtype)
            linear.bias.copy_(W @ linear.bias + b)

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
    ):
        layers = _get_layers(model, self.family_adapter)
        block = layers[insert_pos]
        ref_key = ref_source if ref_source is not None else model_name
        refs = self.reference_inputs[ref_key]
        self._component_ridge = component_ridge

        def _ridge_target(comp: str) -> torch.Tensor | None:
            if lmc_targets is None or comp not in lmc_targets:
                return None
            W_base, _ = lmc_targets[comp]
            if comp in ("input_layernorm", "post_attention_layernorm"):
                return torch.diag(torch.diag(W_base))
            return W_base

        for _ in range(n_iters):
            # 1: input_layernorm — diagonal absorption
            cur = self._capture_component_output(model, insert_pos, "input_layernorm", loader, n_batches)
            ref = refs.get(f"{src_idx}.input_layernorm_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("input_layernorm", ridge_identity), ridge_target=_ridge_target("input_layernorm"))
                    if lmc_store is not None:
                        lmc_store["input_layernorm"] = (W.clone(), b.clone())
                    self._correct_rmsnorm(block.input_layernorm, W, b)

            # 2: q_proj
            cur = self._capture_component_output(model, insert_pos, "q_proj", loader, n_batches)
            ref = refs.get(f"{src_idx}.attn_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("q_proj", ridge_identity), ridge_target=_ridge_target("q_proj"))
                    if lmc_store is not None:
                        lmc_store["q_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.self_attn.q_proj, W, b)

            # 3: k_proj
            cur = self._capture_component_output(model, insert_pos, "k_proj", loader, n_batches)
            ref = refs.get(f"{src_idx}.attn_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("k_proj", ridge_identity), ridge_target=_ridge_target("k_proj"))
                    if lmc_store is not None:
                        lmc_store["k_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.self_attn.k_proj, W, b)

            # 4: v_proj
            cur = self._capture_component_output(model, insert_pos, "v_proj", loader, n_batches)
            ref = refs.get(f"{src_idx}.attn_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("v_proj", ridge_identity), ridge_target=_ridge_target("v_proj"))
                    if lmc_store is not None:
                        lmc_store["v_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.self_attn.v_proj, W, b)

            # 5: o_proj — corrected against residual
            cur = self._capture_component_output(model, insert_pos, "o_proj", loader, n_batches)
            cur_input = self._capture_single_input(model, insert_pos, loader, n_batches)
            ref_input = refs.get(f"{src_idx}.input")
            ref_attn = refs.get(f"{src_idx}.attn_output")
            if ref_input is not None and ref_attn is not None and cur.numel() > 0 and cur_input.numel() > 0:
                n = min(cur.shape[0], cur_input.shape[0], ref_input.shape[0], ref_attn.shape[0])
                A = cur[:n]
                T = ref_input[:n] + ref_attn[:n] - cur_input[:n]
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("o_proj", ridge_identity), ridge_target=_ridge_target("o_proj"))
                    if lmc_store is not None:
                        lmc_store["o_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.self_attn.o_proj, W, b)

            # 6: post_attention_layernorm — diagonal absorption
            cur = self._capture_component_output(model, insert_pos, "post_attention_layernorm", loader, n_batches)
            ref = refs.get(f"{src_idx}.post_attn_ln_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("post_attention_layernorm", ridge_identity), ridge_target=_ridge_target("post_attention_layernorm"))
                    if lmc_store is not None:
                        lmc_store["post_attention_layernorm"] = (W.clone(), b.clone())
                    self._correct_rmsnorm(block.post_attention_layernorm, W, b)

            # 7: gate_proj
            cur = self._capture_component_output(model, insert_pos, "gate_proj", loader, n_batches)
            ref = refs.get(f"{src_idx}.gate_proj_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("gate_proj", ridge_identity), ridge_target=_ridge_target("gate_proj"))
                    if lmc_store is not None:
                        lmc_store["gate_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.mlp.gate_proj, W, b)

            # 8: up_proj
            cur = self._capture_component_output(model, insert_pos, "up_proj", loader, n_batches)
            ref = refs.get(f"{src_idx}.up_proj_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("up_proj", ridge_identity), ridge_target=_ridge_target("up_proj"))
                    if lmc_store is not None:
                        lmc_store["up_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.mlp.up_proj, W, b)

            # 9: down_proj — corrected against residual
            cur = self._capture_component_output(model, insert_pos, "down_proj", loader, n_batches)
            cur_input = self._capture_single_input(model, insert_pos, loader, n_batches)
            ref_input = refs.get(f"{src_idx}.input")
            ref_down = refs.get(f"{src_idx}.down_proj_output")
            if ref_input is not None and ref_down is not None and cur.numel() > 0 and cur_input.numel() > 0:
                n = min(cur.shape[0], cur_input.shape[0], ref_input.shape[0], ref_down.shape[0])
                A = cur[:n]
                T = ref_input[:n] + ref_down[:n] - cur_input[:n]
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("down_proj", ridge_identity), ridge_target=_ridge_target("down_proj"))
                    if lmc_store is not None:
                        lmc_store["down_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.mlp.down_proj, W, b)

    @torch.no_grad()
    def _apply_block_corrections(
        self,
        model: nn.Module,
        insert_pos: int,
        corrections: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ):
        layers = _get_layers(model, self.family_adapter)
        block = layers[insert_pos]

        if "input_layernorm" in corrections:
            W, b = corrections["input_layernorm"]
            self._correct_rmsnorm(block.input_layernorm, W, b)
        if "q_proj" in corrections:
            W, b = corrections["q_proj"]
            self._correct_linear(block.self_attn.q_proj, W, b)
        if "k_proj" in corrections:
            W, b = corrections["k_proj"]
            self._correct_linear(block.self_attn.k_proj, W, b)
        if "v_proj" in corrections:
            W, b = corrections["v_proj"]
            self._correct_linear(block.self_attn.v_proj, W, b)
        if "o_proj" in corrections:
            W, b = corrections["o_proj"]
            self._correct_linear(block.self_attn.o_proj, W, b)
        if "post_attention_layernorm" in corrections:
            W, b = corrections["post_attention_layernorm"]
            self._correct_rmsnorm(block.post_attention_layernorm, W, b)
        if "gate_proj" in corrections:
            W, b = corrections["gate_proj"]
            self._correct_linear(block.mlp.gate_proj, W, b)
        if "up_proj" in corrections:
            W, b = corrections["up_proj"]
            self._correct_linear(block.mlp.up_proj, W, b)
        if "down_proj" in corrections:
            W, b = corrections["down_proj"]
            self._correct_linear(block.mlp.down_proj, W, b)

    @torch.no_grad()
    def _extend_interpolate(
        self,
        *,
        loader: Iterable[Any],
        n_batches: int,
        dampening_factor: float,
        blocks_to_add: int | None,
        target_layers_total: int | None,
        insertion_order: str,
        extension_density: str,
        skip_final_ln: bool = False,
    ) -> int:
        self._vprint("starting interpolate extension")
        layers_base = _get_layers(self.model_base, self.family_adapter)
        curr_layers = len(layers_base)
        n_needed = self._resolve_depth_delta(curr_layers, blocks_to_add, target_layers_total)

        if n_needed <= 0:
            self._vprint("no extension needed")
            return curr_layers

        schedule = self._build_duplication_schedule(
            curr_layers=curr_layers,
            n_needed=n_needed,
            insertion_order=insertion_order,
            extension_density=extension_density,
        )
        self._vprint(f"planned duplications: {schedule}")

        for model in [self.model_base, self.model_ft]:
            layers = _get_layers(model, self.family_adapter)
            orig_blocks = list(layers)
            chain: list[nn.Module] = list(orig_blocks)

            for src_idx in schedule:
                dup = deepcopy(orig_blocks[src_idx])
                src_next = min(src_idx + 1, len(orig_blocks) - 1)
                self._interpolate_block_weights(dup, orig_blocks[src_next], alpha=0.5)

                if dampening_factor < 1.0:
                    self._dampen_block_output(dup, dampening_factor)

                insert_pos = -1
                for i, blk in enumerate(chain):
                    if blk is orig_blocks[src_idx]:
                        insert_pos = i
                insert_pos += 1
                chain.insert(insert_pos, dup)

            self._set_layers(model, chain)

        if not skip_final_ln:
            self._vprint("correcting final norm")
            for model in [self.model_base, self.model_ft]:
                final_norm = _get_final_norm(model, self.family_adapter)
                layers = _get_layers(model, self.family_adapter)
                if hasattr(final_norm, "weight"):
                    src_weight = self.reference_inputs.get("base", {}).get("final.input")
                    if src_weight is not None:
                        pass  # final norm correction is handled by the transport step

        final_depth = len(_get_layers(self.model_base, self.family_adapter))
        self._vprint(f"interpolate extension completed. final_depth={final_depth}")
        return final_depth

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
    ) -> int:
        if per_weight_mode not in {"cascade", "duplicate"}:
            raise ValueError(f"Unsupported per_weight_mode '{per_weight_mode}'. Expected 'cascade' or 'duplicate'.")
        self._vprint(f"starting per-weight extension (mode={per_weight_mode})")
        if not skip_correction:
            self.capture_reference_inputs(loader, n_batches)
            self._capture_component_references(loader, n_batches)
            self._vprint("reference activation and component capture completed")
        else:
            self._vprint("skip_correction enabled: skipping reference capture")

        layers_base = _get_layers(self.model_base, self.family_adapter)
        curr_layers = len(layers_base)
        n_needed = self._resolve_depth_delta(curr_layers, blocks_to_add, target_layers_total)

        if n_needed <= 0:
            self._vprint("no extension needed")
            return curr_layers

        schedule = self._build_duplication_schedule(
            curr_layers=curr_layers,
            n_needed=n_needed,
            insertion_order=insertion_order,
            extension_density=extension_density,
        )
        self._vprint(f"planned duplications: {schedule}")

        orig_base = list(layers_base)
        orig_ft = list(_get_layers(self.model_ft, self.family_adapter))

        chain_base: list[dict[str, Any]] = [{"mod": b, "orig_idx": i} for i, b in enumerate(orig_base)]
        chain_ft: list[dict[str, Any]] = [{"mod": b, "orig_idx": i} for i, b in enumerate(orig_ft)]

        step_iter = _iter_with_progress(
            enumerate(schedule, start=1),
            total=len(schedule),
            desc="block_extension_llm.per_weight",
            enabled=self.show_progress,
        )
        for step, src_idx in step_iter:
            self._vprint(f"step {step}/{len(schedule)} source_block={src_idx}")

            dup_base = deepcopy(orig_base[src_idx])
            dup_ft = deepcopy(orig_ft[src_idx])

            if per_weight_mode == "cascade":
                src_next = min(src_idx + 1, len(orig_base) - 1)
                self._interpolate_block_weights(dup_base, orig_base[src_next], alpha=0.5)
                self._interpolate_block_weights(dup_ft, orig_ft[src_next], alpha=0.5)

            if dampening_factor < 1.0:
                self._dampen_block_output(dup_base, dampening_factor)
                self._dampen_block_output(dup_ft, dampening_factor)

            insert_pos = -1
            for i, item in enumerate(chain_base):
                if item["orig_idx"] == src_idx:
                    insert_pos = i
            insert_pos += 1

            chain_base.insert(insert_pos, {"mod": dup_base, "orig_idx": src_idx})
            chain_ft.insert(insert_pos, {"mod": dup_ft, "orig_idx": src_idx})

            self._set_layers(self.model_base, [x["mod"] for x in chain_base])
            self._set_layers(self.model_ft, [x["mod"] for x in chain_ft])

            if not skip_correction:
                base_ref = "ft" if share_ft_refs else None
                if lmc_mode == "independent":
                    self._correct_block_weights_cascade(
                        "base", self.model_base, insert_pos, src_idx, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        ref_source=base_ref, component_ridge=component_ridge,
                    )
                    self._correct_block_weights_cascade(
                        "ft", self.model_ft, insert_pos, src_idx, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                    )
                elif lmc_mode == "steer":
                    base_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    self._correct_block_weights_cascade(
                        "base", self.model_base, insert_pos, src_idx, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        ref_source=base_ref, component_ridge=component_ridge,
                        lmc_store=base_corrections,
                    )
                    self._correct_block_weights_cascade(
                        "ft", self.model_ft, insert_pos, src_idx, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                        lmc_targets=base_corrections,
                    )
                elif lmc_mode == "shared":
                    base_corrections = {}
                    self._correct_block_weights_cascade(
                        "base", self.model_base, insert_pos, src_idx, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        ref_source=base_ref, component_ridge=component_ridge,
                        lmc_store=base_corrections,
                    )
                    self._apply_block_corrections(self.model_ft, insert_pos, base_corrections)
                elif lmc_mode == "shared_reverse":
                    ft_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    self._correct_block_weights_cascade(
                        "ft", self.model_ft, insert_pos, src_idx, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                        lmc_store=ft_corrections,
                    )
                    self._apply_block_corrections(self.model_base, insert_pos, ft_corrections)
                else:
                    raise ValueError(
                        f"Unsupported lmc_mode '{lmc_mode}'. Expected 'independent', 'steer', 'shared', or 'shared_reverse'."
                    )

        final_depth = len(_get_layers(self.model_base, self.family_adapter))
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
            self.capture_reference_inputs(loader, n_batches)
            self._capture_component_references(loader, n_batches)
            self._vprint("reference activation and component capture completed")
        else:
            self._vprint("skip_correction enabled: skipping reference capture")

        layers_base = _get_layers(self.model_base, self.family_adapter)
        curr_layers = len(layers_base)
        n_needed = self._resolve_depth_delta(curr_layers, blocks_to_add, target_layers_total)

        if n_needed >= 0:
            self._vprint("no shrink needed")
            return curr_layers

        n_to_remove = -n_needed
        schedule = self._build_collapse_schedule(
            curr_layers=curr_layers,
            n_to_remove=n_to_remove,
            insertion_order=insertion_order,
            extension_density=extension_density,
        )
        self._vprint(f"planned collapses: {schedule}")

        orig_base = list(layers_base)
        orig_ft = list(_get_layers(self.model_ft, self.family_adapter))

        chain_base: list[dict[str, Any]] = [{"mod": b, "orig_idxs": (i,)} for i, b in enumerate(orig_base)]
        chain_ft: list[dict[str, Any]] = [{"mod": b, "orig_idxs": (i,)} for i, b in enumerate(orig_ft)]

        step_iter = _iter_with_progress(
            enumerate(schedule, start=1),
            total=len(schedule),
            desc="block_extension_llm.shrink_per_weight",
            enabled=self.show_progress,
        )
        for _step, anchor_orig_idx in step_iter:
            collapse_pos = self._locate_collapse_pos(chain_base, anchor_orig_idx)
            left_base = chain_base[collapse_pos]
            right_base = chain_base[collapse_pos + 1]
            left_ft = chain_ft[collapse_pos]
            right_ft = chain_ft[collapse_pos + 1]

            merged_base = deepcopy(left_base["mod"])
            merged_ft = deepcopy(left_ft["mod"])

            self._interpolate_block_weights(merged_base, right_base["mod"], alpha=0.5)
            self._interpolate_block_weights(merged_ft, right_ft["mod"], alpha=0.5)

            if dampening_factor < 1.0:
                self._dampen_block_output(merged_base, dampening_factor)
                self._dampen_block_output(merged_ft, dampening_factor)

            merged_origs = tuple(list(left_base["orig_idxs"]) + list(right_base["orig_idxs"]))
            chain_base[collapse_pos] = {"mod": merged_base, "orig_idxs": merged_origs}
            chain_ft[collapse_pos] = {"mod": merged_ft, "orig_idxs": merged_origs}
            del chain_base[collapse_pos + 1]
            del chain_ft[collapse_pos + 1]

            self._set_layers(self.model_base, [x["mod"] for x in chain_base])
            self._set_layers(self.model_ft, [x["mod"] for x in chain_ft])

            if not skip_correction:
                span_start = merged_origs[0]
                span_end = merged_origs[-1]
                output_ref_key = f"{span_end}.input"
                base_ref = "ft" if share_ft_refs else None
                if lmc_mode == "independent":
                    self._correct_collapsed_block_weights_cascade(
                        "base", self.model_base, collapse_pos, span_start, span_end,
                        output_ref_key, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        ref_source=base_ref, component_ridge=component_ridge,
                    )
                    self._correct_collapsed_block_weights_cascade(
                        "ft", self.model_ft, collapse_pos, span_start, span_end,
                        output_ref_key, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                    )
                elif lmc_mode == "steer":
                    base_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    self._correct_collapsed_block_weights_cascade(
                        "base", self.model_base, collapse_pos, span_start, span_end,
                        output_ref_key, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        ref_source=base_ref, component_ridge=component_ridge,
                        lmc_store=base_corrections,
                    )
                    self._correct_collapsed_block_weights_cascade(
                        "ft", self.model_ft, collapse_pos, span_start, span_end,
                        output_ref_key, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                        lmc_targets=base_corrections,
                    )
                elif lmc_mode == "shared":
                    base_corrections = {}
                    self._correct_collapsed_block_weights_cascade(
                        "base", self.model_base, collapse_pos, span_start, span_end,
                        output_ref_key, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        ref_source=base_ref, component_ridge=component_ridge,
                        lmc_store=base_corrections,
                    )
                    self._apply_block_corrections(self.model_ft, collapse_pos, base_corrections)
                elif lmc_mode == "shared_reverse":
                    ft_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    self._correct_collapsed_block_weights_cascade(
                        "ft", self.model_ft, collapse_pos, span_start, span_end,
                        output_ref_key, loader, n_batches,
                        ridge_identity=ridge_identity, n_iters=n_cascade_iters,
                        component_ridge=component_ridge,
                        lmc_store=ft_corrections,
                    )
                    self._apply_block_corrections(self.model_base, collapse_pos, ft_corrections)
                else:
                    raise ValueError(
                        f"Unsupported lmc_mode '{lmc_mode}'. Expected 'independent', 'steer', 'shared', or 'shared_reverse'."
                    )

        final_depth = len(_get_layers(self.model_base, self.family_adapter))
        self._vprint(f"per-weight shrink completed. final_depth={final_depth}")
        return final_depth

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
        layers = _get_layers(model, self.family_adapter)
        block = layers[block_idx]
        ref_key = ref_source if ref_source is not None else model_name
        refs = self.reference_inputs[ref_key]
        self._component_ridge = component_ridge

        def _ridge_target(comp: str) -> torch.Tensor | None:
            if lmc_targets is None or comp not in lmc_targets:
                return None
            W_base, _ = lmc_targets[comp]
            if comp in ("input_layernorm", "post_attention_layernorm"):
                return torch.diag(torch.diag(W_base))
            return W_base

        for _ in range(n_iters):
            # 1: input_layernorm tracks the start of the collapsed span
            cur = self._capture_component_output(model, block_idx, "input_layernorm", loader, n_batches)
            ref = refs.get(f"{span_start_idx}.input_layernorm_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("input_layernorm", ridge_identity), ridge_target=_ridge_target("input_layernorm"))
                    if lmc_store is not None:
                        lmc_store["input_layernorm"] = (W.clone(), b.clone())
                    self._correct_rmsnorm(block.input_layernorm, W, b)

            # 2-4: q/k/v track the start
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                cur = self._capture_component_output(model, block_idx, proj_name, loader, n_batches)
                ref = refs.get(f"{span_start_idx}.attn_output")
                if ref is not None and cur.numel() > 0:
                    A, T = self._match_rows(cur, ref)
                    if A.numel() > 0 and T.numel() > 0:
                        W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge(proj_name, ridge_identity), ridge_target=_ridge_target(proj_name))
                        if lmc_store is not None:
                            lmc_store[proj_name] = (W.clone(), b.clone())
                        self._correct_linear(getattr(block.self_attn, proj_name), W, b)

            # 5: o_proj corrected against the end block's residual
            cur = self._capture_component_output(model, block_idx, "o_proj", loader, n_batches)
            cur_input = self._capture_single_input(model, block_idx, loader, n_batches)
            ref_input = refs.get(f"{span_end_idx}.input")
            ref_attn = refs.get(f"{span_end_idx}.attn_output")
            if ref_input is not None and ref_attn is not None and cur.numel() > 0 and cur_input.numel() > 0:
                n = min(cur.shape[0], cur_input.shape[0], ref_input.shape[0], ref_attn.shape[0])
                A = cur[:n]
                T = ref_input[:n] + ref_attn[:n] - cur_input[:n]
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("o_proj", ridge_identity), ridge_target=_ridge_target("o_proj"))
                    if lmc_store is not None:
                        lmc_store["o_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.self_attn.o_proj, W, b)

            # 6: post_attention_layernorm tracks the end
            cur = self._capture_component_output(model, block_idx, "post_attention_layernorm", loader, n_batches)
            ref = refs.get(f"{span_end_idx}.post_attn_ln_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("post_attention_layernorm", ridge_identity), ridge_target=_ridge_target("post_attention_layernorm"))
                    if lmc_store is not None:
                        lmc_store["post_attention_layernorm"] = (W.clone(), b.clone())
                    self._correct_rmsnorm(block.post_attention_layernorm, W, b)

            # 7: gate_proj tracks the end
            cur = self._capture_component_output(model, block_idx, "gate_proj", loader, n_batches)
            ref = refs.get(f"{span_end_idx}.gate_proj_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("gate_proj", ridge_identity), ridge_target=_ridge_target("gate_proj"))
                    if lmc_store is not None:
                        lmc_store["gate_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.mlp.gate_proj, W, b)

            # 8: up_proj tracks the end
            cur = self._capture_component_output(model, block_idx, "up_proj", loader, n_batches)
            ref = refs.get(f"{span_end_idx}.up_proj_output")
            if ref is not None and cur.numel() > 0:
                A, T = self._match_rows(cur, ref)
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("up_proj", ridge_identity), ridge_target=_ridge_target("up_proj"))
                    if lmc_store is not None:
                        lmc_store["up_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.mlp.up_proj, W, b)

            # 9: down_proj corrected against the final target output
            cur = self._capture_component_output(model, block_idx, "down_proj", loader, n_batches)
            cur_input = self._capture_single_input(model, block_idx, loader, n_batches)
            ref = refs.get(output_ref_key)
            if ref is not None and cur.numel() > 0 and cur_input.numel() > 0:
                n = min(cur.shape[0], cur_input.shape[0], ref.shape[0])
                A = cur[:n]
                T = ref[:n] - cur_input[:n]
                if A.numel() > 0 and T.numel() > 0:
                    W, b = self._fit_ridge(A, T, ridge_id=self._get_ridge("down_proj", ridge_identity), ridge_target=_ridge_target("down_proj"))
                    if lmc_store is not None:
                        lmc_store["down_proj"] = (W.clone(), b.clone())
                    self._correct_linear(block.mlp.down_proj, W, b)

    def _set_layers(self, model: nn.Module, new_layers: list[nn.Module]) -> None:
        scope = self.family_adapter.transport_scope(model)
        scope.layers = nn.ModuleList(new_layers)
        self._reindex_layers(model, scope.layers)

    @staticmethod
    def _reindex_layers(model: nn.Module, layers: nn.ModuleList) -> None:
        # Each decoder layer's attention module caches its own `layer_idx`
        # (set at construction) to key into the shared KV cache during a
        # forward pass. Duplicating/reordering layers without updating it
        # leaves two layers pointing at the same cache slot: the second one
        # to run has its `update()` call concatenate onto the first's
        # leftover keys/values, silently doubling the sequence length the
        # rest of that layer's attention sees (crashes as a seq-length
        # mismatch against the attention mask, or worse, doesn't crash).
        layer_types = getattr(getattr(model, "config", None), "layer_types", None)
        for new_idx, layer in enumerate(layers):
            for holder in (layer, getattr(layer, "self_attn", None)):
                if holder is not None and hasattr(holder, "layer_idx"):
                    holder.layer_idx = new_idx
            if layer_types is not None and hasattr(layer, "attention_type") and new_idx < len(layer_types):
                layer.attention_type = layer_types[new_idx]

    @torch.no_grad()
    def extend_and_calibrate(
        self,
        *,
        loader: Iterable[Any],
        n_batches: int,
        strategy: str = "interpolate",
        dampening_factor: float = 1.0,
        blocks_to_add: int | None = None,
        target_layers_total: int | None = None,
        insertion_order: str = "bottom-top",
        extension_density: str = "spread",
        skip_correction: bool = False,
        skip_final_ln: bool = False,
        ridge_identity: float = 0.0,
        n_cascade_iters: int = 1,
        share_ft_refs: bool = False,
        component_ridge: dict[str, float] | None = None,
        lmc_mode: str = "independent",
    ) -> int:
        if not skip_correction:
            loader = _deterministic_calibration_loader(loader, n_batches)
        if strategy == "interpolate":
            return self._extend_interpolate(
                loader=loader,
                n_batches=n_batches,
                dampening_factor=dampening_factor,
                blocks_to_add=blocks_to_add,
                target_layers_total=target_layers_total,
                insertion_order=insertion_order,
                extension_density=extension_density,
                skip_final_ln=skip_final_ln,
            )
        if strategy in ("per_weight", "per-weight", "interpolate_per_weight", "interpolate-per-weight", "duplicate_per_weight", "duplicate-per-weight"):
            per_weight_mode = "duplicate" if strategy in ("duplicate_per_weight", "duplicate-per-weight") else "cascade"
            n_needed = self._resolve_depth_delta(
                len(_get_layers(self.model_base, self.family_adapter)), blocks_to_add, target_layers_total
            )
            per_weight_fn = self._shrink_per_weight if n_needed < 0 else self._extend_per_weight
            return per_weight_fn(
                loader=loader,
                n_batches=n_batches,
                dampening_factor=dampening_factor,
                blocks_to_add=blocks_to_add,
                target_layers_total=target_layers_total,
                insertion_order=insertion_order,
                extension_density=extension_density,
                ridge_identity=ridge_identity,
                per_weight_mode=per_weight_mode,
                n_cascade_iters=n_cascade_iters,
                share_ft_refs=share_ft_refs,
                skip_correction=skip_correction,
                component_ridge=component_ridge,
                lmc_mode=lmc_mode,
            )
        if strategy == "shrink":
            return self._shrink_per_weight(
                loader=loader,
                n_batches=n_batches,
                dampening_factor=dampening_factor,
                blocks_to_add=blocks_to_add,
                target_layers_total=target_layers_total,
                insertion_order=insertion_order,
                extension_density=extension_density,
                ridge_identity=ridge_identity,
                per_weight_mode="cascade",
                n_cascade_iters=n_cascade_iters,
                share_ft_refs=share_ft_refs,
                skip_correction=skip_correction,
                component_ridge=component_ridge,
                lmc_mode=lmc_mode,
            )
        raise ValueError(
            "Unsupported extension_strategy "
            f"'{strategy}'. Expected: interpolate, per_weight, shrink, interpolate_per_weight, duplicate_per_weight."
        )


def run_block_extension_llm(
    *,
    source_base_model: nn.Module,
    source_ft_model: nn.Module,
    calibration_loader: Iterable[Any],
    target_layers_total: int | None,
    config: BlockExtensionConfig,
    family_adapter: Any,
    device: str | torch.device,
) -> int:
    extender = DecoderBlockExtender(
        source_base_model,
        source_ft_model,
        family_adapter,
        device,
        verbose=bool(config.verbose),
        show_progress=bool(config.show_progress),
    )
    resolved_target_layers_total = target_layers_total if target_layers_total is not None else config.target_layers_total
    return extender.extend_and_calibrate(
        loader=calibration_loader,
        n_batches=config.n_batches_act,
        strategy=config.extension_strategy,
        dampening_factor=float(config.dampening_factor),
        blocks_to_add=config.blocks_to_add,
        target_layers_total=resolved_target_layers_total,
        insertion_order=config.insertion_order,
        extension_density=config.extension_density,
        skip_correction=bool(config.skip_correction),
        skip_final_ln=bool(config.skip_final_ln),
        ridge_identity=float(config.ridge_identity),
        n_cascade_iters=int(config.n_cascade_iters),
        share_ft_refs=bool(config.share_ft_refs),
        component_ridge=config.component_ridge,
        lmc_mode=str(config.lmc_mode),
    )
