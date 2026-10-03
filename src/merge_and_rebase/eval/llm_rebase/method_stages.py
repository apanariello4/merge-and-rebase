"""Transport stage of the LLM rebase run (mirrors the roles of eval/vision_rebase/method_stages.py).

THESEUS / theseus_gqa / BiCo run a hybrid: body keys are transported, the remaining keys pass through to the target
under ``passthrough_to_target``; every other method transports the whole delta directly.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from ...data.llm_calibration import build_text_calibration_loader as _build_text_calibration_loader
from ...rebase.discrete_layer_match import DiscreteLayerPairing
from ...rebase.methods._ariadne.alignment import apply_depth_pairing_override
from ...rebase.methods._ariadne.biases import materialize_missing_projection_biases
from ...rebase.methods.ariadne import AriadneRebase
from ...rebase.orchestration import MethodResult
from ...rebase.prestep import PrestepKind, PrestepResult, StageEnv, TaskInputs
from .merge import norm_match_transported, resolve_delta_source, resolve_norm_match
from .stages import _DEFAULT_CALIB_BATCHES

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
    """One task's transport: hybrid prepare+transport for THESEUS/theseus_gqa/BiCo, plain transport otherwise.

    ``run`` follows ``rebase.orchestration.MethodStage``: it picks the delta to transport (``transport_delta_source``),
    transports it, norm-matches the result (``delta_norm_match``) and releases the task-local resized model.
    """

    def __init__(self, *, delta_source: str, norm_match: Any) -> None:
        self.delta_source = delta_source
        self.norm_match = norm_match

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> MethodResult:
        rt = env.runtime
        corrected_delta = pre.task_delta
        reference_delta = pre.uncorrected_delta or corrected_delta
        delta = reference_delta if self.delta_source == "uncorrected" else corrected_delta
        transport_keys = pre.transport_keys
        print(f"\n--- '{task.task}' ({task.ctx.index + 1}/{len(rt.task_contexts)}) ---")
        t0 = time.time()
        resized = pre.kind in (PrestepKind.BRACE, PrestepKind.DISCRETE_INDEX)
        if resized:
            pre.source_base_model.to(rt.device)

        transported = self._transport(rt, pre, delta, transport_keys)
        print(f"  transported {len(transported)} keys in {time.time() - t0:.1f}s")

        transported, norms = norm_match_transported(
            transported,
            corrected_delta=corrected_delta,
            reference_delta=reference_delta,
            transport_keys=transport_keys,
            norm_match=self.norm_match,
        )
        if resized:
            # Release each task-local resized model immediately after its matching transport completes.
            pre.source_base_model.to("cpu")
            pre.source_base_model = None
        return MethodResult(transported_delta=transported, task_vector_norms=norms)

    def _transport(
        self,
        rt: Any,
        pre: PrestepResult,
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
                    source_model=pre.source_base_model,
                    target_model=target_llm.model,
                    source_dataloader=source_calib,
                    target_dataloader=target_calib,
                    family_adapter=family_adapter,
                    device=device,
                )
                transported_body = method.transport(
                    source_base=pre.source_base_sd,
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
                    source_model=pre.source_base_model,
                    target_model=target_llm.model,
                    source_dataloader=source_calib,
                    target_dataloader=target_calib,
                    source_recipe=causal_lm_recipe(device=device),
                    target_recipe=causal_lm_recipe(device=device),
                    family_adapter=family_adapter,
                    device=device,
                )
                transported_body = method.transport(
                    source_base=pre.source_base_sd,
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
                source_base=pre.source_base_sd,
                target_base=target_base_sd,
                delta=delta,
                strict=False,
                **method_params,
            )
        return transported


class AriadneStage:
    """One task's Ariadne fit on the native source pair (no prestep, no parameter transport).

    The task vector is only the fitted corrections of the target's residual-writing projections; with
    ``copy_shape_matching_source_deltas`` (ablation, default off) shape-matching source deltas of the other keys are
    added. Missing projection biases are materialized once on the target (``missing_bias="materialize"``).
    """

    #: Recorded in the run summary like the transport stage's; neither applies to a direct fit.
    delta_source = "corrected"
    norm_match = None

    def __init__(self) -> None:
        self.materialized_bias_keys: list[str] | None = None

    def precompute(self, env: StageEnv) -> None:
        """Vision's merge-in-source precompute hook; LLM Ariadne fits per task only."""
        if env.resolved.ariadne_cfg.merge_mode != "per_task_then_merge":
            raise ValueError("LLM Ariadne supports merge_mode='per_task_then_merge' only.")

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> MethodResult:
        rt = env.runtime
        config = env.resolved.ariadne_cfg
        family_adapter = rt.family_adapter
        print(f"\n--- '{task.task}' ({task.ctx.index + 1}/{len(rt.task_contexts)}) [ariadne] ---")
        if config.missing_bias == "materialize" and self.materialized_bias_keys is None:
            self.materialized_bias_keys = list(
                materialize_missing_projection_biases(
                    rt.target_llm.model, rt.target_base_sd, family_adapter=family_adapter, components=config.components
                )
            )
        texts = rt._calibration().texts
        source_loader = _build_text_calibration_loader(
            tokenizer=rt.source_llm.tokenizer, texts=texts, batch_size=rt.calib_batch_size,
            max_length=rt.calib_max_length,
        )
        target_loader = _build_text_calibration_loader(
            tokenizer=rt.target_llm.tokenizer, texts=texts, batch_size=rt.calib_batch_size,
            max_length=rt.calib_max_length,
        )
        pairing = apply_depth_pairing_override(
            DiscreteLayerPairing.compute(int(rt.source_depth), int(rt.target_depth)), config.depth_pairing
        )
        t0 = time.time()
        prepared = AriadneRebase().prepare(
            source_base_model=pre.source_base_model,
            source_ft_model=pre.source_ft_model,
            target_model=rt.target_llm.model,
            target_base_sd=rt.target_base_sd,
            source_loader=source_loader,
            target_loader=target_loader,
            pairing=pairing,
            config=config,
            device=rt.device,
            family_adapter=family_adapter,
        )
        fitted = dict(prepared.task_vector)
        if config.copy_shape_matching_source_deltas and pre.task_delta:
            for key, value in pre.task_delta.items():
                target = rt.target_base_sd.get(key)
                if key not in fitted and target is not None and tuple(target.shape) == tuple(value.shape):
                    fitted[key] = value
        print(f"  ariadne fitted {len(fitted)} keys in {time.time() - t0:.1f}s (pairing {list(pairing.pairing)})")
        pre.source_base_model = None
        pre.source_ft_model = None
        # "source_*" norms are the full source task vector's (Ariadne transports no source keys); "transported" is
        # the fitted vector. No norm matching applies to a direct fit.
        source_delta = pre.task_delta or {}
        transported, norms = norm_match_transported(
            fitted, corrected_delta=source_delta, reference_delta=source_delta, transport_keys=None, norm_match=None
        )
        return MethodResult(transported_delta=transported, task_vector_norms=norms)


def build_method_stage(cfg: Any, *, ariadne: bool = False) -> TransportMethodStage | AriadneStage:
    """Resolves ``transport_delta_source`` / ``delta_norm_match`` (their config errors surface here)."""
    delta_source, norm_match = resolve_delta_source(cfg), resolve_norm_match(cfg)
    if ariadne:
        if delta_source != "corrected" or norm_match not in (None, "none"):
            raise ValueError(
                "Ariadne fits its task vector directly: transport_delta_source and delta_norm_match do not apply."
            )
        return AriadneStage()
    return TransportMethodStage(delta_source=delta_source, norm_match=norm_match)
