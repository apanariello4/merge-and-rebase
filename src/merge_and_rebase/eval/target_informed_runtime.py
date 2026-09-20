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
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset

from ..rebase.methods.theseus import _interp_2d_tokens, _to_tokens
from .block_extension import _encode_image
from .target_residual_completion import (
    ResidualCompletionConfig,
    ResidualSufficientStatistics,
    centered_rectangular_procrustes,
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
    active = shared.enabled or residual.enabled or bool(cfg.get("capture_target_residual_reference"))
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
    if block_cfg.lmc_mode != "shared" or block_cfg.skip_correction or block_cfg.inserted_block_mode != "ariadne":
        raise ValueError("Target-informed methods extend the ordinary Shared BRACE arm")
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

    def block_module(self, block):
        return block.block if hasattr(block, "block") else block

    def forward(self, model, batch, device):
        _encode_image(model, batch[0].to(device))

    def batch_size(self, batch):
        return len(batch[0])

    def proj_key(self, pos, *, prefixed):
        key = f"transformer.resblocks.{pos}.mlp.c_proj.weight"
        return f"visual.{key}" if prefixed else key


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


def _layout_for(family_adapter):
    return _VisionLayout() if family_adapter is None else _DecoderLayout(family_adapter)


@torch.no_grad()
def capture_tokens(model, batches, requests: Mapping[str, tuple[int, str]], device, *, family_adapter=None):
    """Capture B,T,D tensors, releasing hooks and restoring placement on errors.

    family_adapter=None keeps the original CLIP paths; passing one selects the
    HF-decoder equivalents (see _DecoderLayout). The "c_proj" capture kinds keep
    their names on both paths -- on a decoder they resolve to mlp.down_proj,
    which plays the same residual-writing role.
    """
    layout = _layout_for(family_adapter)
    blocks = layout.blocks(model)
    training = model.training
    original_device = next(model.parameters()).device
    output = {key: [] for key in requests}
    current_batch = [0]
    handles = []
    try:
        model.to(device).eval()
        for key, (index, kind) in requests.items():
            block = layout.block_module(blocks[index])
            module = block if kind == "boundary" else layout.proj_module(blocks[index])
            if kind not in {"boundary", "c_proj", "c_proj_input"}:
                raise ValueError(f"Unknown capture kind {kind}")
            def hook(_m, inputs, value, *, name=key, capture_kind=kind):
                tensor = inputs[0] if capture_kind == "c_proj_input" else value
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                tokens = _to_tokens(tensor.detach(), batch_size=current_batch[0])
                output[name].append(tokens.float().cpu().clone())
            handles.append(module.register_forward_hook(hook))
        for batch in batches:
            current_batch[0] = layout.batch_size(batch)
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
    base = capture_tokens(source_base, sb, req, device, family_adapter=family_adapter)
    ft = capture_tokens(source_ft, sb, req, device, family_adapter=family_adapter)
    target_depth = layout_shim.block_count(target_base)
    positions = [2 * i + 1 for i in range(depth)] if target_scope == "inserted" else list(range(target_depth))
    if not positions or max(positions) >= target_depth:
        raise ValueError(
            "Target model depth does not contain the requested target positions: "
            f"scope={target_scope!r}, source_depth={depth}, target_depth={target_depth}."
        )
    target = capture_tokens(target_base, tb, {str(i): (i, "boundary") for i in positions}, device, family_adapter=family_adapter)

    # Keep source banks by original index for the all-position path.  The
    # inserted path also materializes the historical dictionaries immediately,
    # preserving its byte-compatible downstream behavior.
    source_base_outputs = {int(i): value for i, value in base.items()}
    source_ft_outputs = {int(i): value for i, value in ft.items()}
    target_outputs_by_position = {int(i): value for i, value in target.items()}
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
        desired, target_outputs, maps = _materialize_all_scope_references(references, entries)
        block_kind = {int(row["position"]): str(row.get("block_kind", "unknown")) for row in entries}
    expected_positions = {int(row["position"]) for row in entries}
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
            stats = ResidualSufficientStatistics()
            for h, out, desired_batch, base_out in zip(
                captured["h"], captured["out"], desired[pos], target_outputs[pos], strict=True
            ):
                error = desired_batch - (out - base_out)
                stats.update(h.reshape(-1, h.shape[-1]), error.reshape(-1, error.shape[-1]), t_in, effective_out)
            correction, diag = stats.solve(ridge_relative=config.ridge_relative, exact_form=config.exact_form)
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
            if bias_key not in current_state:
                raise RuntimeError(f"Target model is missing the expected bias parameter {bias_key}")
            bias_correction = diag["bias_correction"]
            transported_bias = t_out.T @ bias_correction.to(t_out)
            if transported_bias.shape != current_state[bias_key].shape or not torch.isfinite(transported_bias).all():
                raise RuntimeError("Residual completion produced an invalid transported bias")
            source_corrections[bias_key] = bias_correction
            target_corrections[bias_key] = transported_bias
            current_state[bias_key] = current_state[bias_key] + transported_bias.to(current_state[bias_key])
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


def _materialize_all_scope_references(references, entries):
    """Join native banks to every realized target position by explicit ancestry."""
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
    desired, target_outputs, maps = {}, {}, {}
    for row in entries:
        pos = int(row["position"])
        source_idx = int(row["source_orig_idx"])
        if source_idx not in source_base or source_idx not in source_ft:
            raise ValueError(f"Missing native source reference for ancestry index {source_idx} at target position {pos}")
        targets = target_by_position[pos]
        source_base_batches = _aligned(source_base[source_idx], targets)
        source_ft_batches = _aligned(source_ft[source_idx], targets)
        q, mu_s, mu_t = centered_rectangular_procrustes(
            _rows(source_base_batches).double(), _rows(targets).double()
        )
        q = q.float()
        desired[pos] = [(f - b) @ q for b, f in zip(source_base_batches, source_ft_batches, strict=True)]
        target_outputs[pos] = targets
        maps[pos] = {"P": q, "source_mean": mu_s.float(), "target_mean": mu_t.float()}
    return desired, target_outputs, maps


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
