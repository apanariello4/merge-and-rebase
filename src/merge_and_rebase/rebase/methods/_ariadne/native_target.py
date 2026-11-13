"""Loader of the NATIVE fine-tuned target for ``residual_target="native_oracle"`` (diagnostic, vision only)."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch

from ....io.ckpt import load_ckpt, load_into_model


def build_native_target_model(
    target_model: torch.nn.Module,
    target_base_state: Mapping[str, torch.Tensor],
    source_base_state: Mapping[str, torch.Tensor],
    ckpt_path: str,
) -> torch.nn.Module:
    """The fine-tuned NATIVE target, built exactly like ``eval.diagnostics.ariadne_norm.readout.build_native_model``:
    deepcopy of ``target_model``, base state loaded strictly, then the checkpoint loaded loosely. The checkpoint must
    match the target backbone (``_infer_ckpt_base``).
    """
    # Lazy: the Ariadne package must not import the run entrypoints at import time (tests/test_ariadne_import_isolation).
    from ....eval.vision_rebase.merge import _infer_ckpt_base

    raw = load_ckpt(str(ckpt_path))
    inferred = _infer_ckpt_base(raw, source_base_sd=source_base_state, target_base_sd=target_base_state)
    if inferred != "target":
        raise ValueError(f"{ckpt_path}: checkpoint matches the {inferred!r} backbone, not the target")
    native = copy.deepcopy(target_model)
    load_into_model(native, target_base_state, strict=True)
    load_into_model(native, raw, strict=False)
    # A loose load that matched no key would leave the base weights and silently give D*_j = 0.
    changed = [
        k
        for k, v in native.state_dict().items()
        if k in target_base_state and not torch.equal(v.cpu(), target_base_state[k].cpu())
    ]
    if not changed:
        raise ValueError(f"{ckpt_path}: loading the checkpoint changed no target parameter")
    return native.eval()
