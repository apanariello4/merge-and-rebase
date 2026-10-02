"""Per-task depth prestep for the LLM rebase run (mirrors the roles of eval/vision_rebase/stages.py).

``NoPrestep`` builds the same-size full-model delta; ``BracePrestep`` resizes a per-task source copy to the target
depth with the decoder block extension and keeps the exact source context for transport. Both return a
``_PreparedTaskDelta``; the code is the former ``cli.main`` task loop, moved verbatim.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, is_dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

import torch

from ...io.ckpt import load_into_model
from ...merge.runtime import to_cpu_fp32
from ...merge.task_vectors import TaskVector
from ...rebase.block_extension.decoder import run_block_extension_llm


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
    # Proposal-1 native reference banks, captured before the resize.



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




def build_task_models(rt: Any, ckpt_ref: Any) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Per-task source base / fine-tuned copies; the fine-tuned copy carries the tuned checkpoint."""
    source_llm = rt.source_llm
    source_base_sd = rt.source_base_sd
    source_build_cfg = rt.source_build_cfg
    # Each task starts from an immutable source template, then its
    # own copy is resized to the target depth before transport.
    # Keep that depth-matched source model alive below.
    source_base_model_task = deepcopy(source_llm.model)
    source_ft_model_task = deepcopy(source_llm.model)

    # Load tuned checkpoint into ft model
    aligned = rt.load_tuned(
        ckpt_ref=ckpt_ref,
        base_sd=source_base_sd,
        build_cfg=source_build_cfg,
        model=source_ft_model_task,
        prefer_lora_view=False,
    )
    tuned_sd = to_cpu_fp32(aligned) if isinstance(aligned, dict) else {k: v.cpu() for k, v in aligned.items()}
    load_into_model(source_ft_model_task, tuned_sd, strict=False)
    return source_base_model_task, source_ft_model_task


class BracePrestep:
    """Depth mismatch: resize a per-task source copy to the target depth (block extension)."""

    def run(self, rt: Any, task_label: str, ckpt_ref: Any) -> _PreparedTaskDelta:
        source_base_model_task, source_ft_model_task = build_task_models(rt, ckpt_ref)
        target_family = rt.target_family
        source_family = rt.source_family
        blockext_calib_loader = rt.blockext_calib_loader
        target_depth = rt.target_depth
        block_extension_cfg = rt.block_extension_cfg
        device = rt.device
        source_depth = rt.source_depth
        run_before_rebase_eval = rt.run_before_rebase_eval
        _eval_before_rebase = rt._eval_before_rebase

        family_adapter_for_ext = target_family or source_family
        if family_adapter_for_ext is None:
            raise ValueError("Block extension requires a family adapter but none was inferred.")

        prepared_task = _prepare_resized_task_delta(
            source_base_model=source_base_model_task,
            source_ft_model=source_ft_model_task,
            calibration_loader=blockext_calib_loader,
            target_layers_total=int(target_depth),
            config=block_extension_cfg,
            family_adapter=family_adapter_for_ext,
            device=device,
        )
        print(f"  block extension completed (source_depth={source_depth} -> {target_depth})")
        # The resized ft model has already been absorbed into the delta;
        # drop it before the eval below so it is not holding device
        # memory while lm-harness runs.
        del source_ft_model_task
        if run_before_rebase_eval:
            # Scored here, after extension and before transport: this is
            # the extended source base that BiCo/Theseus will read from.
            _eval_before_rebase(
                source_base_model_task, f"extended_source_base:{task_label}"
            )
        source_base_model_task.to("cpu")
        return prepared_task


class NoPrestep:
    """Same depth: the plain full-model task vector of the source pair."""

    def run(self, rt: Any, task_label: str, ckpt_ref: Any) -> _PreparedTaskDelta:
        source_llm = rt.source_llm
        source_base_sd = rt.source_base_sd
        source_build_cfg = rt.source_build_cfg
        full_fp_keys = rt.full_fp_keys
        tp_keys = rt.tp_keys
        aligned = rt.load_tuned(
            ckpt_ref=ckpt_ref,
            base_sd=source_base_sd,
            build_cfg=source_build_cfg,
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

        tv = TaskVector.from_checkpoints(
            source_base_sd, tuned_cpu, strict=False
        )
        return _PreparedTaskDelta(
            delta=tv.delta,
            source_base=source_base_sd,
            transport_keys=set(tp_keys or ()),
            source_model=source_llm.model,
        )


def build_prestep(run_block_extension_prestep: bool) -> BracePrestep | NoPrestep:
    return BracePrestep() if run_block_extension_prestep else NoPrestep()
