"""Transport stage of the LLM rebase run (mirrors the roles of eval/vision_rebase/method_stages.py).

THESEUS / theseus_gqa / BiCo run a hybrid: body keys are transported, the remaining keys pass through to the target
under ``passthrough_to_target``; every other method transports the whole delta directly.
"""

from __future__ import annotations

from typing import Any

import torch

from ...data.llm_calibration import build_text_calibration_loader as _build_text_calibration_loader
from .stages import _DEFAULT_CALIB_BATCHES, _PreparedTaskDelta

HYBRID_METHODS = ("theseus", "theseus_gqa", "bico")


def passthrough_to_target(
    transported_body: dict[str, torch.Tensor],
    passthrough_delta: dict[str, torch.Tensor],
    target_base_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Copy non-transported keys whose shape fits the target; return (merged delta, skipped keys)."""
    out = dict(transported_body)
    skipped_passthrough: list[str] = []
    for k, v in passthrough_delta.items():
        if k in target_base_sd and tuple(v.shape) == tuple(target_base_sd[k].shape):
            out[k] = v.to(dtype=target_base_sd[k].dtype, device="cpu")
        else:
            skipped_passthrough.append(k)
    return out, skipped_passthrough


class TransportMethodStage:
    """One task's transport: hybrid prepare+transport for THESEUS/theseus_gqa/BiCo, plain transport otherwise."""

    def run(
        self,
        rt: Any,
        prepared_task: _PreparedTaskDelta,
        delta: dict[str, torch.Tensor],
        transport_keys: set[str],
    ) -> dict[str, torch.Tensor]:
        method = rt.method
        method_name = rt.method_name
        method_params = rt.method_params
        source_llm = rt.source_llm
        target_llm = rt.target_llm
        _calibration = rt._calibration
        calib_batch_size = rt.calib_batch_size
        calib_max_length = rt.calib_max_length
        calib_n_batches = rt.calib_n_batches
        family_adapter = rt.family_adapter
        device = rt.device
        target_base_sd = rt.target_base_sd
        if method_name in HYBRID_METHODS and transport_keys:
            # Hybrid: transport body keys, identity-pass the rest
            body_delta = {k: v for k, v in delta.items() if k in transport_keys}
            passthrough_delta = {k: v for k, v in delta.items() if k not in transport_keys}

            transport_kwargs = dict(method_params)
            source_calib = _build_text_calibration_loader(
                tokenizer=source_llm.tokenizer,
                texts=_calibration().texts,
                batch_size=calib_batch_size,
                max_length=calib_max_length,
            )
            target_calib = _build_text_calibration_loader(
                tokenizer=target_llm.tokenizer,
                texts=_calibration().texts,
                batch_size=calib_batch_size,
                max_length=calib_max_length,
            )

            if method_name in ("theseus", "theseus_gqa"):
                transport_kwargs.setdefault("seq_align", "interpolate")
                if calib_n_batches is None:
                    transport_kwargs.setdefault("n_batches", _DEFAULT_CALIB_BATCHES)
                shared_kwargs = dict(
                    source_model=prepared_task.source_model,
                    target_model=target_llm.model,
                    source_dataloader=source_calib,
                    target_dataloader=target_calib,
                    family_adapter=family_adapter,
                    device=device,
                )
                transported_body = method.transport(
                    source_base=prepared_task.source_base,
                    target_base=target_base_sd,
                    delta=body_delta,
                    strict=False,
                    **shared_kwargs,
                    **transport_kwargs,
                )
            else:
                from ...models.grad_recipes import causal_lm_recipe

                transport_kwargs.setdefault("seq_align", "interpolate")
                if calib_n_batches is None:
                    transport_kwargs.setdefault("n_batches", _DEFAULT_CALIB_BATCHES)
                shared_kwargs = dict(
                    source_model=prepared_task.source_model,
                    target_model=target_llm.model,
                    source_dataloader=source_calib,
                    target_dataloader=target_calib,
                    source_recipe=causal_lm_recipe(device=device),
                    target_recipe=causal_lm_recipe(device=device),
                    family_adapter=family_adapter,
                    device=device,
                )
                transported_body = method.transport(
                    source_base=prepared_task.source_base,
                    target_base=target_base_sd,
                    delta=body_delta,
                    strict=False,
                    curvature_dataloader=None,
                    **shared_kwargs,
                    **transport_kwargs,
                )
            transported, skipped_passthrough = passthrough_to_target(transported_body, passthrough_delta, target_base_sd)
            if skipped_passthrough:
                print(
                    f"  skipped {len(skipped_passthrough)} passthrough keys with incompatible target shape "
                    f"(sample={skipped_passthrough[:5]})"
                )
        else:
            transported = method.transport(
                source_base=prepared_task.source_base,
                target_base=target_base_sd,
                delta=delta,
                strict=False,
                **method_params,
            )
        return transported


def build_method_stage() -> TransportMethodStage:
    return TransportMethodStage()
