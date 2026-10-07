"""Per-task depth prestep for the LLM rebase run (mirrors the roles of eval/vision_rebase/stages.py).

``NoPrestep`` builds the same-size full-model delta; ``BracePrestep`` resizes a per-task source copy to the target
depth with the decoder block extension and keeps the exact source context for transport. Both implement
``rebase.prestep.DepthPrestep`` and return a ``PrestepResult`` (the resize itself still builds a
``_PreparedTaskDelta``); the per-task loop is ``rebase.orchestration.TaskPipeline``.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, is_dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

import torch

from ...io.ckpt import load_into_model
from ...io.text_checkpoints import build_model_with_tuned_config
from ...merge.runtime import to_cpu_fp32
from ...merge.task_vectors import TaskVector
from ...rebase.block_extension.decoder import run_block_extension_llm
from ...rebase.discrete_layer_match import DiscreteLayerPairing, build_discrete_indexed_decoder
from ...rebase.prestep import PrestepKind, PrestepResult, StageEnv, TaskInputs, TaskModels, select_prestep_kind


@dataclass
class _PreparedTaskDelta:
    """Delta together with the exact source context it was prepared from."""

    delta: dict[str, torch.Tensor]
    source_base: dict[str, torch.Tensor]
    transport_keys: set[str]
    source_model: torch.nn.Module
    # The same task vector resized without correction. Under lmc_mode="shared"
    # the fitted correction W is applied to base and ft alike, so the corrected
    # delta is W @ delta: correction changes the task vector's scale as well as
    # its direction. Keeping the uncorrected delta lets a run transport one and
    # normalize to the other, separating those two effects.
    uncorrected_delta: dict[str, torch.Tensor] | None = None
    # Realized block chain from the resize, needed by residual completion to
    # address inserted positions by ancestry instead of a depth pattern.
    extension_layout: dict[str, Any] | None = None


# Calibration batches used by theseus/bico when a config names neither
# method_params.num_batches nor method_params.n_batches.
_DEFAULT_CALIB_BATCHES = 2


def _config_with_correction_disabled(config: Any) -> Any | None:
    """Same block-extension config with correction off, or None if not derivable.

    Real runs pass a BlockExtensionConfig dataclass. Tests pass lightweight
    stubs, and a stub that cannot express skip_correction simply means no
    uncorrected reference vector is available for that call.
    """
    if is_dataclass(config) and not isinstance(config, type):
        return dataclass_replace(config, skip_correction=True)
    clone = copy(config)
    try:
        clone.skip_correction = True
    except (AttributeError, TypeError):
        return None
    return clone


def _prepare_resized_task_delta(
    *,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    calibration_loader: Any,
    target_layers_total: int,
    config: Any,
    family_adapter: Any,
    device: str,
) -> _PreparedTaskDelta:
    """Resize one task pair and retain the exact source context for transport."""
    # Reference resize with correction disabled, kept so the caller can compare
    # or substitute the uncorrected task vector. This costs a deepcopy and no
    # forward passes: with skip_correction the extension only duplicates blocks,
    # it never captures reference or component activations.
    uncorrected_delta: dict[str, torch.Tensor] | None = None
    reference_config = _config_with_correction_disabled(config)
    if reference_config is not None and not bool(getattr(config, "skip_correction", False)):
        ref_base = deepcopy(source_base_model)
        ref_ft = deepcopy(source_ft_model)
        run_block_extension_llm(
            source_base_model=ref_base,
            source_ft_model=ref_ft,
            calibration_loader=calibration_loader,
            target_layers_total=target_layers_total,
            config=reference_config,
            family_adapter=family_adapter,
            device=device,
        )
        uncorrected_delta = TaskVector.from_checkpoints(
            to_cpu_fp32(ref_base.state_dict()), to_cpu_fp32(ref_ft.state_dict()), strict=False
        ).delta
        del ref_base, ref_ft

    extension_layout: dict[str, Any] = {}
    final_depth = run_block_extension_llm(
        source_base_model=source_base_model,
        source_ft_model=source_ft_model,
        calibration_loader=calibration_loader,
        target_layers_total=target_layers_total,
        config=config,
        family_adapter=family_adapter,
        device=device,
        layout_out=extension_layout,
    )
    if final_depth != target_layers_total:
        raise RuntimeError(
            "Block extension preprocess failed: "
            f"final_depth={final_depth}, expected={target_layers_total}."
        )

    source_base = to_cpu_fp32(source_base_model.state_dict())
    source_ft = to_cpu_fp32(source_ft_model.state_dict())
    task_vector = TaskVector.from_checkpoints(source_base, source_ft, strict=False)
    if uncorrected_delta is None:
        # skip_correction: the corrected and uncorrected resizes are the same run.
        uncorrected_delta = task_vector.delta
    return _PreparedTaskDelta(
        delta=task_vector.delta,
        source_base=source_base,
        transport_keys=set(family_adapter.transportable_keys(source_base)),
        source_model=source_base_model,
        uncorrected_delta=uncorrected_delta,
        extension_layout=extension_layout or None,
    )


@dataclass
class LlmTaskContext:
    """What the LLM stages need to know about one task: the tuned checkpoint reference (and its position)."""

    ckpt_ref: Any
    index: int
    #: ``{config field: [source, tuned]}`` where the tuned ref's own HF config disagrees with the source base's
    #: (RoPE, window, ...); empty (the usual case) when the two agree. Non-empty: the fine-tuned forward model is
    #: built from the tuned config instead of a deepcopy of the source.
    config_overrides: dict[str, list[Any]] = field(default_factory=dict)
    #: Positional geometry read from the live models that run forward passes for this task (source base, source
    #: fine-tuned, target), filled by ``build_task_models``: the run-time evidence that a tuned ref with its own
    #: config really runs under it.
    forward_geometry: dict[str, dict[str, Any]] = field(default_factory=dict)


def model_geometry(model: torch.nn.Module) -> dict[str, Any]:
    """``rope_theta`` / ``max_position_embeddings`` / ``use_sliding_window`` of a live HF model (transformers 4 or 5)."""
    config = model.config
    rope = getattr(config, "rope_parameters", None) or {}
    return {
        "name_or_path": getattr(config, "_name_or_path", None),
        "rope_theta": rope.get("rope_theta", getattr(config, "rope_theta", None)),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "use_sliding_window": getattr(config, "use_sliding_window", None),
    }


def build_task_models(env: StageEnv, task: str) -> TaskModels | None:
    """Per-task source base / fine-tuned copies; the ft copy carries the tuned ckpt.

    Built for the block-extension and discrete-index presteps and for Ariadne (which fits from the native pair).
    """
    plan = env.plan
    if not (plan.task_block_extension_prestep or plan.task_discrete_layer_match_prestep or env.resolved.direct_fit):
        return None
    rt = env.runtime
    ckpt_ref = rt.task_contexts[task].ckpt_ref
    # Each task starts from an immutable source template, then its
    # own copy is resized to the target depth before transport.
    # Keep that depth-matched source model alive below.
    source_base_model_task = deepcopy(rt.source_llm.model)
    overrides = rt.task_contexts[task].config_overrides
    # Equal configs: the source deepcopy carrying the tuned weights is exactly the fine-tuned model. Otherwise the
    # tuned weights would run under the source's RoPE / window geometry, so build it from the tuned ref's own config.
    source_ft_model_task = None if overrides else deepcopy(rt.source_llm.model)

    # Load tuned checkpoint into ft model
    aligned = rt.load_tuned(
        ckpt_ref=ckpt_ref,
        base_sd=rt.source_base_sd,
        build_cfg=rt.source_build_cfg,
        model=source_ft_model_task if source_ft_model_task is not None else rt.source_llm.model,
        prefer_lora_view=False,
    )
    tuned_sd = to_cpu_fp32(aligned) if isinstance(aligned, dict) else {k: v.cpu() for k, v in aligned.items()}
    if overrides:
        changed = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in sorted(overrides.items()))
        print(f"  '{task}': fine-tuned forward model built from the tuned ref's own config ({changed})")
        source_ft_model_task = build_model_with_tuned_config(str(ckpt_ref), rt.source_build_cfg, tuned_sd)
    else:
        load_into_model(source_ft_model_task, tuned_sd, strict=False)
    rt.task_contexts[task].forward_geometry = {
        "source_base": model_geometry(source_base_model_task),
        "source_ft": model_geometry(source_ft_model_task),
        "target": model_geometry(rt.target_llm.model),
    }
    return TaskModels(source_base=source_base_model_task, source_ft=source_ft_model_task)


class BracePrestep:
    """Depth mismatch: resize a per-task source copy to the target depth (block extension)."""

    kind = PrestepKind.BRACE

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        rt = env.runtime
        family_adapter_for_ext = rt.target_family or rt.source_family
        if family_adapter_for_ext is None:
            raise ValueError("Block extension requires a family adapter but none was inferred.")

        prepared_task = _prepare_resized_task_delta(
            source_base_model=models.source_base,
            source_ft_model=models.source_ft,
            calibration_loader=rt.blockext_calib_loader,
            target_layers_total=int(rt.target_depth),
            config=rt.block_extension_cfg,
            family_adapter=family_adapter_for_ext,
            device=rt.device,
        )
        print(f"  block extension completed (source_depth={rt.source_depth} -> {rt.target_depth})")
        # The resized ft model has already been absorbed into the delta; drop it before the observers run so it is
        # not holding device memory while lm-harness runs.
        models.source_ft = None
        return PrestepResult(
            kind=self.kind,
            source_base_sd=prepared_task.source_base,
            task_delta=prepared_task.delta,
            source_base_model=prepared_task.source_model,
            layout=prepared_task.extension_layout or {},
            final_depth=int(rt.target_depth),
            transport_keys=prepared_task.transport_keys,
            uncorrected_delta=prepared_task.uncorrected_delta,
        )

    def load_delta(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> PrestepResult:
        return pre


class NoPrestep:
    """Same depth: the plain full-model task vector of the source pair."""

    kind = PrestepKind.NONE

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        rt = env.runtime
        source_llm = rt.source_llm
        source_base_sd = rt.source_base_sd
        full_fp_keys = rt.full_fp_keys
        aligned = rt.load_tuned(
            ckpt_ref=task.ctx.ckpt_ref,
            base_sd=source_base_sd,
            build_cfg=rt.source_build_cfg,
            model=source_llm.model,
            prefer_lora_view=False,
        )
        tuned_cpu = to_cpu_fp32(aligned) if isinstance(aligned, dict) else {k: v.cpu() for k, v in aligned.items()}

        # Full-model same-size delta
        if full_fp_keys is not None:
            tuned_cpu = {k: v for k, v in tuned_cpu.items() if k in full_fp_keys and k in source_base_sd}
            # Validate shapes
            for k in tuned_cpu:
                if tuple(tuned_cpu[k].shape) != tuple(source_base_sd[k].shape):
                    raise ValueError(
                        f"Source base vs tuned shape mismatch for '{k}': "
                        f"base {tuple(source_base_sd[k].shape)} vs "
                        f"tuned {tuple(tuned_cpu[k].shape)}"
                    )

        tv = TaskVector.from_checkpoints(source_base_sd, tuned_cpu, strict=False)
        return PrestepResult(
            kind=self.kind,
            source_base_sd=source_base_sd,
            task_delta=tv.delta,
            # Ariadne gets the native per-task pair from build_task_models; transport uses the shared source model.
            source_base_model=models.source_base if models is not None else source_llm.model,
            source_ft_model=models.source_ft if models is not None else None,
            transport_keys=set(rt.tp_keys or ()),
        )

    def load_delta(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> PrestepResult:
        return pre


class ExtendedSourceBaseObserver:
    """Scores the resized source base before transport (when requested), then parks it on the CPU.

    This is the "before rebase" reference: the extended source base that BiCo/Theseus read from, not the target base.
    """

    def before(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> None:
        return None

    def after(self, env: StageEnv, task: TaskInputs, models: TaskModels | None, result: PrestepResult) -> None:
        rt = env.runtime
        if rt.run_before_rebase_eval:
            rt._eval_before_rebase(models.source_base, f"extended_source_base:{task.task}")
        models.source_base.to("cpu")


class DiscreteIndexPrestep:
    """Depth mismatch for BiCo: the BiCo-paper discrete index match ``i(j)=round(j(D_s-1)/(D_t-1))``.

    The source base and fine-tuned models are reindexed to the target depth (deep copies, HF-correct layer
    bookkeeping); the task delta and the transport statistics both come from the reindexed source stack.
    """

    kind = PrestepKind.DISCRETE_INDEX

    def run(self, env: StageEnv, task: TaskInputs, models: TaskModels | None) -> PrestepResult:
        rt = env.runtime
        family_adapter = rt.target_family or rt.source_family
        if family_adapter is None:
            raise ValueError("Discrete index match requires a family adapter but none was inferred.")
        pairing = DiscreteLayerPairing.compute(int(rt.source_depth), int(rt.target_depth))
        base = build_discrete_indexed_decoder(models.source_base, pairing, family_adapter)
        ft = build_discrete_indexed_decoder(models.source_ft, pairing, family_adapter)
        models.source_ft = None
        base_sd = to_cpu_fp32(base.state_dict())
        ft_sd = to_cpu_fp32(ft.state_dict())
        del ft
        delta = TaskVector.from_checkpoints(base_sd, ft_sd, strict=False).delta
        print(f"  discrete index match (source_depth={rt.source_depth} -> {rt.target_depth}): {list(pairing.pairing)}")
        return PrestepResult(
            kind=self.kind,
            source_base_sd=base_sd,
            task_delta=delta,
            source_base_model=base,
            layout={"pairing": list(pairing.pairing)},
            final_depth=int(rt.target_depth),
            transport_keys=set(family_adapter.transportable_keys(base_sd)),
        )

    def load_delta(self, env: StageEnv, task: TaskInputs, pre: PrestepResult) -> PrestepResult:
        return pre


def build_prestep(plan: Any) -> BracePrestep | DiscreteIndexPrestep | NoPrestep:
    kind = select_prestep_kind(plan)
    if kind is PrestepKind.BRACE:
        return BracePrestep()
    if kind is PrestepKind.DISCRETE_INDEX:
        return DiscreteIndexPrestep()
    return NoPrestep()


def build_prestep_observers(plan: Any) -> tuple[ExtendedSourceBaseObserver, ...]:
    return (ExtendedSourceBaseObserver(),) if plan.task_block_extension_prestep else ()
