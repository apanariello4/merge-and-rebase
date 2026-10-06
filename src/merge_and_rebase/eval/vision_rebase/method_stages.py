"""Method stages of the per-task vision rebase pipeline (Phase 5.9).

``TransportMethodStage`` is the ordinary per-task path: ``_build_rebase_prepared`` followed by
``method.transport`` under one ``PhaseCostRecorder`` (THESEUS, BiCo, TransFusion, gradfix and the
other state-dict transports). ``AriadneStage`` is the Ariadne path: it never calls
``_build_rebase_prepared``/``method.transport`` for a result, owns the ``merge_in_source_then_fit``
precompute and the saved-vector loader, and computes the depth pairing once per run. Bodies are the
former inline blocks of ``main()``, moved verbatim.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import torch

from ...data.vision_loaders import DEFAULT_NUM_WORKERS, DEFAULT_SEED
from ...io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ...merge.methods._common import axpy_state_dict
from ...merge.task_vectors import TaskVector
from ...models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ...rebase.block_extension.config import select_loader
from ...rebase.depth_pairing import spread_duplicate_pairing
from ...rebase.discrete_layer_match import DiscreteLayerPairing
from ...rebase.methods.ariadne import AriadneRebase, apply_depth_pairing_override
from ...rebase.orchestration import AriadneRunRecord, MethodResult
from ...rebase.prestep import PrestepResult, StageEnv, TaskInputs
from ...utils.cost_accounting import PhaseCostRecorder, cost_phase, recording
from ..utils import to_cpu_fp32
from .artifacts import _load_saved_sequential_tv
from .context import _TaskContext
from .merge import _merge_direction
from .stages import _visual_only_filter


def _build_rebase_prepared(
    *,
    method_name: str,
    method: Any,
    method_params: dict[str, Any],
    cfg: dict[str, Any],
    device: str,
    grad_batch_size: int | None,
    grad_imgs_per_class: int | None,
    grad_num_batches: int | None,
    theseus_mode: bool,
    bico_mode: bool,
    run_block_extension_prestep: bool,
    clf_source: OpenClipClassifier,
    clf_target: OpenClipClassifier,
    classnames: list[str],
    loaders: Any,
    source_loaders: Any,
    build_cfg_task: OpenClipBuildConfig,
    source_build_cfg_task: OpenClipBuildConfig,
    task_source_base_sd: dict[str, torch.Tensor],
    target_base_sd: dict[str, torch.Tensor],
    task_delta: dict[str, torch.Tensor],
    source_base_model_task: torch.nn.Module | None,
    transfusion_prepared: dict[str, Any] | None,
    source_text_features: torch.Tensor | None = None,
    target_text_features: torch.Tensor | None = None,
    source_activation_plan: Any | None = None,
) -> Any:
    """Compute the rebase method's prepared state for one task context.

    Shared by the per-task transport path and by merge_then_rebase's single
    post-composition transport. Note that theseus/bico use ``task_delta`` for
    key filtering and shape handling only — values never affect the prepared
    transforms.

    For Theseus and BiCo, ``seed`` controls the deterministic sampling of
    calibration batches used to fit the transport maps.  Default it to the
    run-level seed so a seed sweep actually changes those maps, while allowing
    ``method_params.seed`` to override it when map sampling must be decoupled
    from the validation/test split seed.
    """
    if method_name == "gradfix":
        from ...models.grad_recipes import clip_contrastive_recipe
        from ..utils import build_grad_dataloader

        grad_loader = build_grad_dataloader(
            loaders.train,
            loaders.train.dataset,
            grad_batch_size=grad_batch_size,
            grad_imgs_per_class=grad_imgs_per_class,
            grad_num_batches=grad_num_batches,
            num_workers=int(cfg.get("num_workers", DEFAULT_NUM_WORKERS)),
            seed=int(cfg.get("seed", DEFAULT_SEED)),
        )
        recipe = clip_contrastive_recipe(
            clf_target,
            classnames,
            build_cfg_task,
            device=device,
            reduction="none" if str(method_params.get("vote", "mean")) in {"majority", "max"} else "mean",
        )
        return method.prepare(
            target_model=clf_target.model,
            target_dataloader=grad_loader,
            recipe=recipe,
            device=device,
            **method_params,
        )

    if source_activation_plan is not None and not (theseus_mode or bico_mode):
        raise ValueError(
            "The interpolated-activation baseline only applies to the activation-aligned "
            f"transport methods; method '{method_name}' does not consume activations."
        )

    if theseus_mode:
        theseus_params = dict(method_params)
        transport_seed = int(theseus_params.pop("seed", cfg.get("seed", DEFAULT_SEED)))
        if run_block_extension_prestep:
            if source_base_model_task is None:
                raise RuntimeError("Theseus block-extension preprocess requires the corrected source base model.")
            # BRACE changes the source depth.  Loading this state into the raw
            # source architecture non-strictly drops every inserted block.
            source_model_for_theseus = deepcopy(source_base_model_task)
            load_into_model(source_model_for_theseus, task_source_base_sd, strict=True)
        else:
            source_model_for_theseus = deepcopy(clf_source.model)
            load_into_model(source_model_for_theseus, task_source_base_sd, strict=True)
        target_model_for_theseus = deepcopy(clf_target.model)
        load_into_model(target_model_for_theseus, target_base_sd, strict=True)

        source_depth = len(source_model_for_theseus.visual.transformer.resblocks)
        target_depth = len(target_model_for_theseus.visual.transformer.resblocks)
        if source_depth != target_depth:
            raise ValueError(
                "Theseus calibration models must be depth-matched: "
                f"source_depth={source_depth}, target_depth={target_depth}"
            )

        # Corrected BRACE templates are intentionally retained on CPU to keep
        # the multi-task campaign's resident memory bounded.  Theseus sends
        # calibration inputs to ``device`` but does not own model placement,
        # so place only these isolated working copies immediately before
        # activation collection.
        source_model_for_theseus.to(device).eval()
        target_model_for_theseus.to(device).eval()

        # ``covariance_source`` other than the default fits the alignment on the
        # fine-tuned source endpoint as well as the base one.  That endpoint is
        # the corrected base plus the corrected task vector, which is the same
        # theta_ft_bar the transported task vector is defined against.
        source_ft_model_for_theseus: torch.nn.Module | None = None
        if str(theseus_params.get("covariance_source", "base")).strip().lower() not in {
            "base",
            "source_base",
            "source-base",
        }:
            if task_delta is None:
                raise RuntimeError("A non-default Theseus covariance_source requires the task delta.")
            source_ft_model_for_theseus = deepcopy(source_model_for_theseus)
            load_into_model(
                source_ft_model_for_theseus,
                axpy_state_dict(task_source_base_sd, task_delta, alpha=1.0),
                strict=True,
            )
            source_ft_model_for_theseus.to(device).eval()

        return method.prepare(
            source_model=source_model_for_theseus,
            target_model=target_model_for_theseus,
            source_model_ft=source_ft_model_for_theseus,
            source_dataloader=source_loaders.train,
            target_dataloader=loaders.train,
            target_base=target_base_sd,
            delta=task_delta,
            device=device,
            seed=transport_seed,
            source_activation_plan=source_activation_plan,
            **theseus_params,
        )

    if bico_mode:
        from ...models.grad_recipes import clip_contrastive_recipe

        bico_params = dict(method_params)
        transport_seed = int(bico_params.pop("seed", cfg.get("seed", DEFAULT_SEED)))

        if run_block_extension_prestep:
            if source_base_model_task is None:
                raise RuntimeError("BiCo block-extension preprocess requires the corrected source base model.")
            # BiCo moves models across devices and performs backward passes
            # while collecting statistics; use a strict-loaded copy so the
            # task's corrected endpoint remains immutable for later steps.
            source_model_for_bico = deepcopy(source_base_model_task)
            load_into_model(source_model_for_bico, task_source_base_sd, strict=True)
        else:
            source_model_for_bico = deepcopy(clf_source.model)
            load_into_model(source_model_for_bico, task_source_base_sd, strict=True)
        target_model_for_bico = deepcopy(clf_target.model)
        load_into_model(target_model_for_bico, target_base_sd, strict=True)

        source_depth = len(source_model_for_bico.visual.transformer.resblocks)
        target_depth = len(target_model_for_bico.visual.transformer.resblocks)
        if source_depth != target_depth:
            raise ValueError(
                "BiCo calibration models must be depth-matched: "
                f"source_depth={source_depth}, target_depth={target_depth}"
            )

        source_recipe = clip_contrastive_recipe(
            clf_source,
            classnames,
            source_build_cfg_task,
            device=device,
            text_features=source_text_features,
        )
        target_recipe = clip_contrastive_recipe(
            clf_target,
            classnames,
            build_cfg_task,
            device=device,
            text_features=target_text_features,
        )

        prepared = method.prepare(
            source_model=source_model_for_bico,
            target_model=target_model_for_bico,
            source_dataloader=source_loaders.train,
            target_dataloader=loaders.train,
            source_recipe=source_recipe,
            target_recipe=target_recipe,
            target_base=target_base_sd,
            delta=task_delta,
            device=device,
            seed=transport_seed,
            source_activation_plan=source_activation_plan,
            **bico_params,
        )
        del source_model_for_bico, target_model_for_bico, source_recipe, target_recipe
        return prepared

    return transfusion_prepared


def _run_direct_residual_fit(**kwargs: Any):
    """Run the Ariadne fit (`AriadneRebase.prepare`) under its own `PhaseCostRecorder`.

    Thin wrapper kept for the existing call sites: same keyword arguments as
    `_direct_residual_fit_body` (minus the recorder), returning
    ``(scaled_delta, timing, diagnostics, extra)`` with ``timing["cost_phases"]``
    the recorder summary splitting this fit's wall time, CUDA peak and host peak
    RSS into activation_collection / transformation / transport (analysis-only
    diagnostics are excluded; see utils.cost_accounting). The pipeline itself
    lives in `merge_and_rebase.rebase.methods.ariadne`.
    """
    prepared = AriadneRebase().prepare(**kwargs)
    return prepared.task_vector, prepared.timing, prepared.diagnostics, prepared.extra


def _direct_residual_fit_body(recorder: PhaseCostRecorder, **kwargs: Any):
    """Same fit as `_run_direct_residual_fit`, under a caller-owned recorder (no ``cost_phases``)."""
    prepared = AriadneRebase().prepare(recorder=recorder, **kwargs)
    return prepared.task_vector, prepared.timing, prepared.diagnostics, prepared.extra


class TransportMethodStage:
    """``_build_rebase_prepared`` + ``method.transport`` for one task, timed under one recorder."""

    def __init__(self, *, bypass_ordinary_transport: bool, transport_calibration_ctx: _TaskContext | None) -> None:
        # Proposal-1 transport-free ablation: THESEUS/BiCo are neither fitted nor applied. The
        # question this arm asks is whether the desired functional effect can be written into the
        # target at all without parameter transport, so invoking the transport fit and then
        # discarding its output would only burn GPU hours and blur the claim. Ariadne bypasses it
        # too (its fit is the ``AriadneStage`` body).
        self.bypass_ordinary_transport = bypass_ordinary_transport
        self.transport_calibration_ctx = transport_calibration_ctx

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> MethodResult:
        resolved = env.resolved
        plan = env.plan
        cfg = env.cfg
        device = env.device
        method = resolved.method
        method_params = resolved.method_params
        transport_calibration_ctx = self.transport_calibration_ctx
        bypass_ordinary_transport = self.bypass_ordinary_transport
        task_cost_recorder = PhaseCostRecorder(device)
        with recording(task_cost_recorder):
            prepare_mark = task_cost_recorder.mark()
            prepare_started = time.perf_counter()

            prepared = (
                None
                if bypass_ordinary_transport
                else _build_rebase_prepared(
                    method_name=resolved.method_name,
                    method=method,
                    method_params=method_params,
                    cfg=cfg,
                    device=device,
                    grad_batch_size=resolved.grad_batch_size,
                    grad_imgs_per_class=resolved.grad_imgs_per_class,
                    grad_num_batches=resolved.grad_num_batches,
                    theseus_mode=resolved.theseus_mode,
                    bico_mode=resolved.bico_mode,
                    run_block_extension_prestep=plan.task_block_extension_prestep
                    or plan.task_discrete_layer_match_prestep,
                    clf_source=env.clf_source,
                    clf_target=env.clf_target,
                    classnames=(
                        task.classnames if transport_calibration_ctx is None else transport_calibration_ctx.classnames
                    ),
                    loaders=task.loaders if transport_calibration_ctx is None else transport_calibration_ctx.loaders,
                    source_loaders=(
                        task.source_loaders
                        if transport_calibration_ctx is None
                        else transport_calibration_ctx.source_loaders
                    ),
                    build_cfg_task=(
                        task.build_cfg_task
                        if transport_calibration_ctx is None
                        else transport_calibration_ctx.build_cfg_task
                    ),
                    source_build_cfg_task=(
                        task.source_build_cfg_task
                        if transport_calibration_ctx is None
                        else transport_calibration_ctx.source_build_cfg_task
                    ),
                    task_source_base_sd=pre.source_base_sd,
                    target_base_sd=env.target_base_sd,
                    task_delta=pre.task_delta,
                    source_base_model_task=pre.source_base_model,
                    transfusion_prepared=env.transfusion_prepared,
                    source_activation_plan=pre.activation_plan,
                )
            )

            prepare_seconds = time.perf_counter() - prepare_started
            peak_memory_bytes = task_cost_recorder.peaks_since(prepare_mark)[0]

            transport_started = time.perf_counter()
            with cost_phase("transport"):
                transported_delta = (
                    {}
                    if bypass_ordinary_transport
                    else method.transport(
                        source_base=pre.source_base_sd,
                        target_base=env.target_base_sd,
                        delta=pre.task_delta,
                        strict=resolved.strict_load,
                        prepared=prepared,
                        **method_params,
                    )
                )
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.synchronize()
            transport_timing = {
                "prepare_seconds": prepare_seconds,
                "transport_seconds": time.perf_counter() - transport_started,
                "peak_memory_allocated_bytes": peak_memory_bytes,
            }
            cost_phases = task_cost_recorder.summary()
        return MethodResult(
            transported_delta=transported_delta,
            prepared=prepared,
            transport_timing=transport_timing,
            cost_phases=cost_phases,
        )


class AriadneStage:
    """Ariadne (formerly Direct Residual): paired capture + fit, merged-fit precompute, saved-vector loading."""

    def __init__(
        self,
        env: StageEnv,
        *,
        task_contexts: dict[str, _TaskContext],
        tasks: list[str],
        merge_weights: list[float],
        calibration_ctx: _TaskContext | None,
        calibration_meta: dict[str, Any],
    ) -> None:
        self.cfg = env.resolved.ariadne_cfg
        self.task_contexts = task_contexts
        self.tasks = tasks
        self.merge_weights = merge_weights
        self.calibration_ctx = calibration_ctx
        self.record = AriadneRunRecord(calibration_meta=calibration_meta)
        # Bypassed ordinary transport: only the shared timing bracket (prepare ~ 0, empty transport).
        self._transport = TransportMethodStage(bypass_ordinary_transport=True, transport_calibration_ctx=None)
        self._pairing_cache: DiscreteLayerPairing | None = None
        self._merged_correction: dict[str, torch.Tensor] | None = None
        self._merged_timing: dict[str, dict[str, float]] | None = None

    def pairing(self, env: StageEnv) -> DiscreteLayerPairing:
        """The depth pairing, computed once per run (both legacy call sites yielded the identical value)."""
        if self._pairing_cache is None:
            ancestry = None
            if self.cfg.depth_pairing == "spread_duplicate":
                ancestry = spread_duplicate_pairing(env.source_depth, env.target_depth)
            self._pairing_cache = apply_depth_pairing_override(
                DiscreteLayerPairing.compute(env.source_depth, env.target_depth),
                self.cfg.depth_pairing,
                ancestry=ancestry,
            )
            self.record.record_pairing(self._pairing_cache, self.cfg.depth_pairing)
        return self._pairing_cache

    def precompute(self, env: StageEnv) -> None:
        """merge_in_source_then_fit: merge every task's native delta ONCE, then fit ONE shared correction.

        (A ``DirectResidualConfig`` field, distinct from the top-level ``merge_mode`` cfg key.) The
        per-task path reuses this single cached correction for every task instead of re-fitting, exactly
        as merge_then_brace_then_transport reuses one merged/transported model across tasks.
        """
        direct_residual_cfg = self.cfg
        if direct_residual_cfg.merge_mode != "merge_in_source_then_fit":
            return
        resolved = env.resolved
        source_base_sd = env.source_base_sd
        merge_in_source_tasks = [t for t in self.tasks if t not in env.native_tasks]
        merge_in_source_deltas: list[dict[str, torch.Tensor]] = []
        merge_in_source_weights: list[float] = []
        for idx, t in enumerate(self.tasks):
            if t in env.native_tasks:
                continue
            ckpt_path = str(env.tuned_by_task[t])
            sd = load_ckpt(ckpt_path)
            aligned = align_to_base_keys(sd, source_base_sd)
            if not aligned:
                raise ValueError(
                    f"No tensors from tuned checkpoint aligned to source base keys for task '{t}': {ckpt_path}."
                )
            tuned_sd = to_cpu_fp32(aligned)
            delta = TaskVector.from_checkpoints(
                source_base_sd, tuned_sd, strict=False, key_filter=_visual_only_filter
            ).delta
            merge_in_source_deltas.append(delta)
            merge_in_source_weights.append(self.merge_weights[idx])
        if not merge_in_source_tasks:
            raise ValueError("direct_residual merge_in_source_then_fit requires at least one non-native task.")
        merged_direction = _merge_direction(
            base_sd=source_base_sd,
            deltas=merge_in_source_deltas,
            merge_method_name=resolved.merge.method_name,
            weights=merge_in_source_weights,
            merge_params=resolved.merge.params,
        )
        source_base_model_merged = deepcopy(env.clf_source.model)
        source_ft_model_merged = deepcopy(env.clf_source.model)
        load_into_model(source_base_model_merged, source_base_sd, strict=True)
        load_into_model(
            source_ft_model_merged,
            axpy_state_dict(source_base_sd, merged_direction, alpha=1.0),
            strict=True,
        )
        # The once-only merged fit has no single task to draw loaders from: it uses the shared Ariadne
        # calibration context when one is configured, else the first contributing task's train loaders.
        merged_calibration_task_ctx = (
            self.calibration_ctx if self.calibration_ctx is not None else self.task_contexts[merge_in_source_tasks[0]]
        )
        direct_residual_pairing = self.pairing(env)
        (
            self._merged_correction,
            self._merged_timing,
            direct_residual_merged_diag,
            direct_residual_merged_extra,
        ) = _run_direct_residual_fit(
            source_base_model=source_base_model_merged,
            source_ft_model=source_ft_model_merged,
            target_model=env.clf_target.model,
            target_base_sd=env.target_base_sd,
            source_loader=merged_calibration_task_ctx.source_loaders.train,
            target_loader=merged_calibration_task_ctx.loaders.train,
            pairing=direct_residual_pairing,
            config=direct_residual_cfg,
            device=env.device,
            clf_source=env.clf_source,
            clf_target=env.clf_target,
            classnames=merged_calibration_task_ctx.classnames,
            source_build_cfg_task=merged_calibration_task_ctx.source_build_cfg_task,
            build_cfg_task=merged_calibration_task_ctx.build_cfg_task,
            source_text_features=merged_calibration_task_ctx.source_text_features,
            target_text_features=merged_calibration_task_ctx.target_text_features,
        )
        for t in merge_in_source_tasks:
            self.record.record_fit(t, direct_residual_merged_diag, direct_residual_merged_extra, sequential=False)

    def run(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> MethodResult:
        result = self._transport.run(env, task, pre)
        cfg = env.cfg
        direct_residual_cfg = self.cfg
        t = task.task
        # Ariadne never resizes: capture runs on freshly built native (un-resized) source models. The
        # block-extension prestep never runs for Ariadne (it is not a blockext_like_method).
        pairing = self.pairing(env)
        if cfg.get("load_direct_residual_tvs_dir"):
            transported_delta, loaded_meta = _load_saved_sequential_tv(
                cfg["load_direct_residual_tvs_dir"], t, env.target_base_sd, direct_residual_cfg
            )
            self.record.loaded_vectors[t] = loaded_meta
            result.transported_delta = transported_delta
        elif direct_residual_cfg.merge_mode == "merge_in_source_then_fit":
            if self._merged_correction is None or self._merged_timing is None:
                raise RuntimeError(
                    "Direct Residual merge_in_source_then_fit correction was not precomputed before the per-task loop."
                )
            result.transported_delta = dict(self._merged_correction)
            result.alignment_calibration = dict(self._merged_timing["alignment_calibration"])
            result.correction_fit = dict(self._merged_timing["correction_fit"])
            result.cost_phases = dict(self._merged_timing["cost_phases"])
        else:
            source_base_model_native = deepcopy(env.clf_source.model)
            source_ft_model_native = deepcopy(env.clf_source.model)
            load_into_model(source_base_model_native, env.source_base_sd, strict=True)
            load_into_model(source_ft_model_native, env.source_base_sd, strict=True)
            load_into_model(source_ft_model_native, load_ckpt(str(env.tuned_by_task[t])), strict=False)
            if self.calibration_ctx is not None:
                # Task-independent calibration: the same paired
                # images for every task (see calibration_data).
                direct_residual_source_loader = self.calibration_ctx.source_loaders.train
                direct_residual_target_loader = self.calibration_ctx.loaders.train
            else:
                direct_residual_source_loader = select_loader(
                    "train",
                    train_loader=task.source_loaders.train,
                    test_loader=task.source_loaders.test,
                    val_loader=task.source_loaders.val,
                )
                direct_residual_target_loader = select_loader(
                    "train",
                    train_loader=task.loaders.train,
                    test_loader=task.loaders.test,
                    val_loader=task.loaders.val,
                )
            (
                transported_delta,
                direct_residual_timing,
                task_direct_residual_diag,
                task_direct_residual_extra,
            ) = _run_direct_residual_fit(
                source_base_model=source_base_model_native,
                source_ft_model=source_ft_model_native,
                target_model=env.clf_target.model,
                target_base_sd=env.target_base_sd,
                source_loader=direct_residual_source_loader,
                target_loader=direct_residual_target_loader,
                pairing=pairing,
                config=direct_residual_cfg,
                device=env.device,
                clf_source=env.clf_source,
                clf_target=env.clf_target,
                classnames=task.classnames,
                source_build_cfg_task=task.source_build_cfg_task,
                build_cfg_task=task.build_cfg_task,
                source_text_features=task.ctx.source_text_features,
                target_text_features=task.ctx.target_text_features,
            )
            result.transported_delta = transported_delta
            result.alignment_calibration = direct_residual_timing["alignment_calibration"]
            result.correction_fit = direct_residual_timing["correction_fit"]
            result.cost_phases = direct_residual_timing["cost_phases"]
            self.record.record_fit(t, task_direct_residual_diag, task_direct_residual_extra)
        return result


def build_method_stage(
    env: StageEnv,
    *,
    transport_calibration_ctx: _TaskContext | None,
    task_contexts: dict[str, _TaskContext],
    tasks: list[str],
    merge_weights: list[float],
    ariadne_calibration_ctx: _TaskContext | None,
    ariadne_calibration_meta: dict[str, Any],
) -> TransportMethodStage | AriadneStage:
    """Select the method stage once from the resolved config (Ariadne vs every state-dict transport)."""
    if env.resolved.direct_residual_like:
        return AriadneStage(
            env,
            task_contexts=task_contexts,
            tasks=tasks,
            merge_weights=merge_weights,
            calibration_ctx=ariadne_calibration_ctx,
            calibration_meta=ariadne_calibration_meta,
        )
    return TransportMethodStage(
        bypass_ordinary_transport=False,
        transport_calibration_ctx=transport_calibration_ctx,
    )
