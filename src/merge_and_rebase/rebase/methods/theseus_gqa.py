from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from ..registry import register
from .theseus import (
    ActivationStore,
    TheseusRebase,
    _LayerTransform,
    _compute_procrustes_map_from_cov,
)

logger = logging.getLogger(__name__)

# Suffix -> (head_kind, side). `side` is which activation ("in" or "out") of
# the module carries the multi-head structure that must be split before
# alignment: q/k/v project the residual stream (no head structure) to a
# concatenated multi-head vector (head structure on the *output*); o_proj
# consumes the concatenated per-head attention output (head structure on the
# *input*) and projects back to the plain residual stream.
_ATTN_ROLE_SUFFIXES: dict[str, tuple[str, str]] = {
    "self_attn.q_proj.weight": ("q", "out"),
    "self_attn.k_proj.weight": ("k", "out"),
    "self_attn.v_proj.weight": ("v", "out"),
    "self_attn.o_proj.weight": ("q", "in"),
}


def _head_assignment(source_heads: int, target_heads: int) -> list[list[int]]:
    """For each target head, the source head(s) it should be aligned from.

    Uses the same proportional ("nearest neighbor") assignment in both
    directions, which reduces to the standard GQA broadcast/pool pattern
    whenever the head counts divide evenly (e.g. 2 KV heads -> 8 KV heads
    assigns target heads 0-3 to source head 0, 4-7 to source head 1, ...,
    exactly matching `repeat_kv`), and degrades gracefully to a nearest-head
    assignment when they don't (e.g. 12 query heads -> 16).
    """
    if source_heads <= 0 or target_heads <= 0:
        raise ValueError("Head counts must be positive.")

    if target_heads >= source_heads:
        return [[int(t * source_heads // target_heads)] for t in range(target_heads)]

    groups: list[list[int]] = [[] for _ in range(target_heads)]
    for s in range(source_heads):
        t = int(s * target_heads // source_heads)
        groups[t].append(s)
    return groups


def _head_block_transform(
    cov: torch.Tensor,
    *,
    source_heads: int,
    target_heads: int,
    source_head_dim: int,
    target_head_dim: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Head-block-aware alternative to a dense whole-vector Procrustes map.

    Rather than one SVD over the entire concatenated multi-head covariance
    (which is free to mix values across head boundaries with no regard for
    RoPE/QK-norm's per-head structure), fits an independent (head_dim x
    head_dim) rotation for each aligned head pair and assembles a block
    transform with zeros everywhere else.
    """
    src_dim, tgt_dim = int(cov.shape[0]), int(cov.shape[1])
    expected_src = source_heads * source_head_dim
    expected_tgt = target_heads * target_head_dim
    if src_dim != expected_src or tgt_dim != expected_tgt:
        raise ValueError(
            f"Head-aware transform shape mismatch: cov={(src_dim, tgt_dim)}, "
            f"expected source={source_heads}x{source_head_dim}={expected_src}, "
            f"target={target_heads}x{target_head_dim}={expected_tgt}."
        )

    # Must live on the same device the per-block rotations land on, since the
    # assembled transform is later matmul'd against transforms the base method
    # produced with this same `device`.
    transform = torch.zeros((src_dim, tgt_dim), dtype=torch.float32, device=torch.device(device))
    assignment = _head_assignment(source_heads, target_heads)

    for tgt_h, src_heads_for_tgt in enumerate(assignment):
        tgt_slice = slice(tgt_h * target_head_dim, (tgt_h + 1) * target_head_dim)
        blocks = [
            cov[src_h * source_head_dim : (src_h + 1) * source_head_dim, tgt_slice]
            for src_h in src_heads_for_tgt
        ]
        pooled = blocks[0] if len(blocks) == 1 else sum(blocks)
        rot = _compute_procrustes_map_from_cov(pooled, device=device)
        for src_h in src_heads_for_tgt:
            src_slice = slice(src_h * source_head_dim, (src_h + 1) * source_head_dim)
            transform[src_slice, tgt_slice] = rot

    return transform


@dataclass(frozen=True)
class TheseusGqaRebase(TheseusRebase):
    """Theseus variant with head-block-aware attention-projection transport.

    Standard Theseus computes one dense Procrustes map over a linear layer's
    entire (concatenated multi-head) output/input vector. That is blind to
    per-head structure, so whenever source and target disagree on head
    count -- e.g. a differing GQA ratio, or plain head-count growth -- the
    resulting transform freely mixes values across head boundaries, which is
    architecturally incoherent given attention applies RoPE (and, on Qwen3,
    QK-RMSNorm) independently per head. This variant instead aligns each
    target head to its corresponding source head(s) -- broadcasting/pooling
    by the head-count ratio, mirroring how `repeat_kv` broadcasts GQA
    key/value heads at attention time -- and assembles a block-structured
    transform for q_proj/k_proj/v_proj/o_proj, leaving every other
    transportable key (MLP, layernorms, final norm) on the original
    whole-vector Theseus path unchanged.

    Kept as a separate method (rather than changing `theseus` in place) so
    existing `method: "theseus"` configs and results are unaffected; opt in
    with `method: "theseus_gqa"`.
    """

    name: str = "theseus_gqa"

    def prepare(self, **kwargs: Any) -> dict[str, Any]:
        prepared = super().prepare(**kwargs)

        family_adapter = kwargs.get("family_adapter")
        source_model = kwargs.get("source_model")
        target_model = kwargs.get("target_model")
        activation_registry: dict[str, ActivationStore] | None = prepared.get("activation_registry")
        transforms_by_key: dict[str, _LayerTransform] | None = prepared.get("transforms_by_key")
        if not (
            family_adapter is not None
            and source_model is not None
            and target_model is not None
            and activation_registry
            and transforms_by_key
        ):
            return prepared

        center_acts = bool(kwargs.get("center_acts", False))
        # Mirror the base method's svd_device exactly: the overridden blocks are
        # matmul'd against the transforms it already produced, so landing on a
        # different device would fail (or silently zero) the whole attention delta.
        device = str(kwargs.get("device", "cuda"))
        svd_device = device if prepared.get("device_transform") == "gpu" else "cpu"

        if float(kwargs.get("whiten_power", 0.0)) > 0.0:
            logger.warning(
                "%s prepare: whiten_power is ignored for head-aware attention blocks; "
                "they use the raw activation cross-covariance.",
                self.name,
            )

        source_meta = family_adapter.metadata(source_model)
        target_meta = family_adapter.metadata(target_model)

        source_head_dim = source_meta.head_dim or 0
        target_head_dim = target_meta.head_dim or 0
        if not source_head_dim or not target_head_dim:
            logger.warning(
                "%s prepare: missing head_dim metadata, falling back to plain Theseus "
                "transforms for attention projections.",
                self.name,
            )
            return prepared

        source_kv_heads = source_meta.num_key_value_heads or source_meta.num_attention_heads
        target_kv_heads = target_meta.num_key_value_heads or target_meta.num_attention_heads
        param_to_module = family_adapter.param_to_module(target_model)

        new_transforms = dict(transforms_by_key)
        overridden_t_out: dict[str, torch.Tensor] = {}
        overridden_weights = 0

        for key, layer_transform in transforms_by_key.items():
            role = None
            for suffix, (head_kind, side) in _ATTN_ROLE_SUFFIXES.items():
                if key.endswith(suffix):
                    role = (head_kind, side)
                    break
            if role is None or layer_transform.kind != "weight":
                continue
            head_kind, side = role
            if side == "out" and layer_transform.t_in is None:
                continue
            if side == "in" and layer_transform.t_out is None:
                continue

            if head_kind in ("k", "v"):
                source_heads, target_heads = source_kv_heads, target_kv_heads
            else:
                source_heads, target_heads = source_meta.num_attention_heads, target_meta.num_attention_heads

            module_name = param_to_module.get(key)
            if module_name is None:
                continue
            store = activation_registry.get(f"{module_name}.{side}")
            if store is None:
                continue
            cov = store.get_covariance(center=center_acts)
            if cov is None:
                continue

            try:
                block_transform = _head_block_transform(
                    cov,
                    source_heads=source_heads,
                    target_heads=target_heads,
                    source_head_dim=source_head_dim,
                    target_head_dim=target_head_dim,
                    device=svd_device,
                )
            except ValueError as exc:
                logger.warning("%s prepare: skipping head-aware transform for '%s': %s", self.name, key, exc)
                continue

            if side == "out":
                new_transforms[key] = _LayerTransform(kind="weight", t_in=layer_transform.t_in, t_out=block_transform)
                overridden_t_out[key] = block_transform
            else:
                new_transforms[key] = _LayerTransform(kind="weight", t_in=block_transform, t_out=layer_transform.t_out)
            overridden_weights += 1

        # Biases share their weight's t_out; keep them in sync with any override.
        overridden_biases = 0
        for key, layer_transform in transforms_by_key.items():
            if layer_transform.kind != "bias" or not key.endswith(".bias"):
                continue
            weight_key = key[: -len(".bias")] + ".weight"
            new_t_out = overridden_t_out.get(weight_key)
            if new_t_out is not None:
                new_transforms[key] = _LayerTransform(kind="bias", t_out=new_t_out)
                overridden_biases += 1

        if bool(kwargs.get("verbose", True)):
            print(
                f"[{self.name}] prepare: head-aware attention transforms = {overridden_weights} weights "
                f"+ {overridden_biases} biases (q/o heads {source_meta.num_attention_heads}->"
                f"{target_meta.num_attention_heads}, kv heads {source_kv_heads}->{target_kv_heads})"
            )
            if overridden_weights == 0:
                print(
                    f"[{self.name}] prepare: WARNING no attention projection matched -- "
                    "this run is equivalent to plain theseus"
                )

        prepared = dict(prepared)
        prepared["transforms_by_key"] = new_transforms
        return prepared


register(TheseusGqaRebase())
