from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from ...models.patch_openclip_attention import merge_openclip_vit_attn
from ...models.vision_utils import _encode_image
from ...utils.cost_accounting import cost_phase
from ..base import TensorDict
from ..registry import register
from ._shared import (  # noqa: F401
    _ACTIVATION_COVARIANCE_MODES,  # noqa: F401
    _DATA_FREE_COVARIANCE_MODES,  # noqa: F401
    _DIAGNOSTIC_EXAMPLE_LIMIT,  # noqa: F401
    _FUSED_IN_PROJ_BIAS,  # noqa: F401
    _FUSED_IN_PROJ_WEIGHT,  # noqa: F401
    _K_PROJ_BIAS,  # noqa: F401
    _K_PROJ_WEIGHT,  # noqa: F401
    _Q_PROJ_BIAS,  # noqa: F401
    _Q_PROJ_WEIGHT,  # noqa: F401
    _RESBLOCK_RE,  # noqa: F401
    _V_PROJ_BIAS,  # noqa: F401
    _V_PROJ_WEIGHT,  # noqa: F401
    _VISUAL_PREFIX,  # noqa: F401
    _ZERO_KEYS,  # noqa: F401
    ActivationStore,  # noqa: F401
    InterpolatedBlockActivations,  # noqa: F401
    _activation_group,  # noqa: F401
    _ActivationHook,  # noqa: F401
    _align_features,  # noqa: F401
    _append_diagnostic_example,  # noqa: F401
    _apply_transforms_to_visual_delta,  # noqa: F401
    _ApplyDiagnostics,  # noqa: F401
    _build_grouped_covariances,  # noqa: F401
    _build_grouped_transforms,  # noqa: F401
    _compute_alignment_map,  # noqa: F401
    _compute_alignment_map_from_matrix_proxies,  # noqa: F401
    _compute_procrustes_map,  # noqa: F401
    _compute_procrustes_map_from_cov,  # noqa: F401
    _content_row_mask,  # noqa: F401
    _drop_padding_rows,  # noqa: F401
    _extract_model_inputs,  # noqa: F401
    _extract_output_tensor,  # noqa: F401
    _freeze_diagnostic_examples,  # noqa: F401
    _has_fused_mha,  # noqa: F401
    _interp_2d_tokens,  # noqa: F401
    _interp_linear_tokens,  # noqa: F401
    _is_square,  # noqa: F401
    _iter_random_dataset_batches,  # noqa: F401
    _iter_with_progress,  # noqa: F401
    _LayerTransform,  # noqa: F401
    _matrix_power_psd,  # noqa: F401
    _merge_split_qkv_state,  # noqa: F401
    _param_to_module,  # noqa: F401
    _partially_whiten_covariance,  # noqa: F401
    _precompute_diagnostics_from_transforms,  # noqa: F401
    _precompute_transforms,  # noqa: F401
    _precompute_transforms_data_free,  # noqa: F401
    _PrecomputeDiagnostics,  # noqa: F401
    _report_apply_diagnostics,  # noqa: F401
    _report_precompute_diagnostics,  # noqa: F401
    _resolve_covariance_mode,  # noqa: F401
    _resolve_device,  # noqa: F401
    _split_fused_qkv_if_needed,  # noqa: F401
    _split_fused_qkv_state,  # noqa: F401
    _standardize_tokens,  # noqa: F401
    _to_tokens,  # noqa: F401
    _transport_bias,  # noqa: F401
    _transport_weight,  # noqa: F401
    _visual_delta_keys,  # noqa: F401
    _visual_module,  # noqa: F401
    _visual_state_dict,  # noqa: F401
    _WrongTransportShape,  # noqa: F401
)

logger = logging.getLogger(__name__)


# Registry keys collected against the fine-tuned source endpoint are stored in
# the same flat dict as the base-endpoint keys so that one activation-cache
# payload still round-trips a whole prepare call.
_FT_REGISTRY_PREFIX = "ft::"
_COVARIANCE_SOURCES = ("base", "ft", "delta", "mixture")


def _cache_dataset_identity(dataset: Any) -> tuple[Any, ...]:
    """Return stable metadata that distinguishes common dataset wrappers/splits."""

    if dataset is None:
        return ("none",)

    identity: list[Any] = [
        type(dataset).__module__,
        type(dataset).__qualname__,
    ]
    for attr in ("_fingerprint", "fingerprint", "image_key", "label_key"):
        value = getattr(dataset, attr, None)
        if value is not None:
            identity.append((attr, str(value)))

    # Hugging Face datasets expose a split fingerprint; torch Subset exposes
    # its parent dataset and selected indices. Include both when available.
    for attr in ("split", "dataset"):
        nested = getattr(dataset, attr, None)
        if nested is not None and nested is not dataset:
            identity.append((attr, _cache_dataset_identity(nested)))

    indices = getattr(dataset, "indices", None)
    if indices is not None:
        try:
            indices = tuple(int(index) for index in indices)
        except (TypeError, ValueError):
            indices = repr(indices)
        identity.append(("indices", indices))
    return tuple(identity)


def _activation_cache_fingerprint(
    *,
    source_model: nn.Module,
    target_model: nn.Module,
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    seq_align: str,
    n_batches: int | None,
    seed: int,
    batch_size: int | None,
    cache_key: str | None,
    whiten_power: float,
    whiten_eps: float,
    source_activation_plan: InterpolatedBlockActivations | None = None,
    source_model_ft: nn.Module | None = None,
) -> str:
    """Fingerprint every input that affects streamed activation statistics."""
    digest = sha256()
    for value in (
        seq_align,
        n_batches,
        seed,
        batch_size,
        cache_key,
        float(whiten_power),
        float(whiten_eps),
        None if source_activation_plan is None else source_activation_plan.fingerprint(),
    ):
        digest.update(repr(value).encode())
    # The combination weights are applied after collection, so they do not
    # enter the fingerprint: one cached payload serves every covariance_source
    # that the same pair of banks supports.
    models = (source_model, target_model) if source_model_ft is None else (source_model, source_model_ft, target_model)
    for model in models:
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    for loader in (source_dataloader, target_dataloader):
        dataset = getattr(loader, "dataset", None)
        digest.update(f"{type(loader).__qualname__}:{type(dataset).__qualname__}".encode())
        for value in (loader, dataset):
            try:
                digest.update(str(len(value)).encode())
            except TypeError:
                digest.update(b"unknown-length")
        digest.update(
            repr(
                (
                    getattr(loader, "batch_size", None),
                    getattr(loader, "drop_last", None),
                    _cache_dataset_identity(dataset),
                )
            ).encode()
        )
    return digest.hexdigest()


def _activation_registry_payload(registry: Mapping[str, ActivationStore]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "store_raw": store.store_raw,
            "store_a_gram": store.store_a_gram,
            "store_b_gram": store.store_b_gram,
            "at_b": store.at_b,
            "at_a": store.at_a,
            "bt_b": store.bt_b,
            "sum_a": store.sum_a,
            "sum_b": store.sum_b,
            "n_samples": store.n_samples,
            "h_a_list": store.h_a_list,
            "h_b_list": store.h_b_list,
        }
        for key, store in registry.items()
    }


def _activation_registry_from_payload(payload: Mapping[str, Mapping[str, Any]]) -> dict[str, ActivationStore]:
    registry: dict[str, ActivationStore] = {}
    for key, values in payload.items():
        store = ActivationStore(
            store_raw=bool(values["store_raw"]),
            store_a_gram=bool(values["store_a_gram"]),
            store_b_gram=bool(values["store_b_gram"]),
        )
        for name in ("at_b", "at_a", "bt_b", "sum_a", "sum_b", "n_samples", "h_a_list", "h_b_list"):
            setattr(store, name, values[name])
        registry[key] = store
    return registry


def _resolve_covariance_source(source: str) -> str:
    key = str(source).strip().lower()
    if key in {"base", "source_base", "source-base"}:
        return "base"
    if key in {"ft", "fine_tuned", "fine-tuned", "source_ft"}:
        return "ft"
    if key in {"delta", "delta_activations", "delta-activations"}:
        return "delta"
    if key in {"mixture", "mix", "interpolate"}:
        return "mixture"
    raise ValueError(f"Theseus covariance_source must be one of {_COVARIANCE_SOURCES}; got {source!r}.")


def _covariance_source_coefficients(covariance_source: str, mixture_beta: float) -> tuple[float, float]:
    """Return ``(c_base, c_ft)`` for the effective source rows ``c_base X_base + c_ft X_ft``.

    Every supported source is an affine combination of the two collected source
    banks, and the cross-covariance against a shared target bank is linear in
    the source rows.  Combining the accumulated statistics is therefore exact:
    no second pass over the calibration data is needed once both banks exist.
    """

    if covariance_source == "base":
        return 1.0, 0.0
    if covariance_source == "ft":
        return 0.0, 1.0
    if covariance_source == "delta":
        return -1.0, 1.0
    if covariance_source == "mixture":
        beta = float(mixture_beta)
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Theseus covariance_mixture_beta must be in [0, 1].")
        return 1.0 - beta, beta
    raise ValueError(f"Unsupported covariance_source {covariance_source!r}.")


def _combine_activation_registry(
    registry: Mapping[str, ActivationStore],
    *,
    covariance_source: str,
    mixture_beta: float,
) -> dict[str, ActivationStore]:
    """Collapse the paired base/FT banks into one registry of combined statistics."""

    base_registry = {key: store for key, store in registry.items() if not key.startswith(_FT_REGISTRY_PREFIX)}
    if covariance_source == "base":
        return base_registry

    c_base, c_ft = _covariance_source_coefficients(covariance_source, mixture_beta)
    combined: dict[str, ActivationStore] = {}
    for key, base_store in base_registry.items():
        ft_store = registry.get(f"{_FT_REGISTRY_PREFIX}{key}")
        if ft_store is None:
            raise ValueError(
                f"covariance_source='{covariance_source}' requires a fine-tuned source bank for '{key}'."
            )
        if base_store.at_b is None or ft_store.at_b is None:
            continue
        if base_store.n_samples != ft_store.n_samples:
            raise ValueError(
                f"Paired activation banks disagree on sample count for '{key}': "
                f"{base_store.n_samples} != {ft_store.n_samples}."
            )
        # Both banks were streamed from the same batches against the same
        # target forward, so the target-side statistics must be identical.
        # A mismatch means the two passes did not see the same calibration
        # rows, which would silently void the comparison.
        if base_store.sum_b is None or ft_store.sum_b is None or not torch.equal(base_store.sum_b, ft_store.sum_b):
            raise ValueError(f"Paired activation banks disagree on target statistics for '{key}'.")

        store = ActivationStore()
        store.at_b = c_base * base_store.at_b + c_ft * ft_store.at_b
        store.sum_a = c_base * base_store.sum_a + c_ft * ft_store.sum_a
        store.sum_b = base_store.sum_b.clone()
        store.n_samples = int(base_store.n_samples)
        combined[key] = store
    return combined


@torch.inference_mode()
def collect_activations(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    *,
    device: str | torch.device,
    seq_align: str,
    n_batches: int | None,
    seed: int = 0,
    batch_size: int | None = None,
    store_raw: bool = False,
    store_a_gram: bool = False,
    store_b_gram: bool = False,
    family_adapter: Any = None,
    source_activation_plan: InterpolatedBlockActivations | None = None,
    source_model_ft: torch.nn.Module | None = None,
) -> dict[str, ActivationStore]:
    """Stream paired source/target activation statistics.

    When ``source_model_ft`` is given, the fine-tuned source endpoint is run on
    the same batches in the same pass and its statistics are stored under the
    ``ft::`` key prefix.  Sharing the pass is what makes the two banks exactly
    comparable: they see identical calibration rows and an identical target
    forward, which ``_combine_activation_registry`` then relies on.
    """

    if family_adapter is not None:
        source_scope = family_adapter.transport_scope(source_model)
        target_scope = family_adapter.transport_scope(target_model)
    else:
        source_scope = None
        target_scope = None

    registry: dict[str, ActivationStore] = {}
    source_hook = _ActivationHook(source_model, scope=source_scope)
    target_hook = _ActivationHook(target_model, scope=target_scope)
    source_ft_hook: _ActivationHook | None = None
    if source_model_ft is not None:
        source_ft_scope = family_adapter.transport_scope(source_model_ft) if family_adapter is not None else None
        source_ft_hook = _ActivationHook(source_model_ft, scope=source_ft_scope)
    dev = _resolve_device(device)

    try:
        iterator = _iter_random_dataset_batches(
            source_dataloader,
            target_dataloader,
            n_batches=n_batches,
            seed=seed,
            batch_size=batch_size,
        )
        if iterator is None:
            iterator = zip(source_dataloader, target_dataloader, strict=True)

        consumed_batches = 0
        for idx, (source_batch, target_batch) in enumerate(iterator):
            if n_batches is not None and idx >= n_batches:
                break

            with cost_phase("activation_collection"):
                if family_adapter is not None:
                    source_inputs = family_adapter.extract_calibration_batch(source_batch)
                    target_inputs = family_adapter.extract_calibration_batch(target_batch)
                    s_inp = source_inputs.get("input_ids", source_batch)
                    t_inp = target_inputs.get("input_ids", target_batch)
                    if isinstance(s_inp, torch.Tensor):
                        s_inp = s_inp.to(dev)
                    if isinstance(t_inp, torch.Tensor):
                        t_inp = t_inp.to(dev)
                    s_attn = source_inputs.get("attention_mask", None)
                    t_attn = target_inputs.get("attention_mask", None)
                    if s_attn is not None:
                        s_attn = s_attn.to(dev)
                    if t_attn is not None:
                        t_attn = t_attn.to(dev)

                    source_model(**({"input_ids": s_inp, "attention_mask": s_attn} if s_attn is not None else {"input_ids": s_inp}))
                    target_model(**({"input_ids": t_inp, "attention_mask": t_attn} if t_attn is not None else {"input_ids": t_inp}))
                    row_mask = _content_row_mask(s_attn, t_attn)
                else:
                    source_imgs = _extract_model_inputs(source_batch).to(dev)
                    target_imgs = _extract_model_inputs(target_batch).to(dev)
                    if source_imgs.shape[0] != target_imgs.shape[0]:
                        raise ValueError(
                            "Theseus calibration expects aligned batch sizes. "
                            f"Got {source_imgs.shape[0]} and {target_imgs.shape[0]}."
                        )
                    source_labels = source_batch[1] if isinstance(source_batch, (tuple, list)) and len(source_batch) > 1 else None
                    target_labels = target_batch[1] if isinstance(target_batch, (tuple, list)) and len(target_batch) > 1 else None
                    if torch.is_tensor(source_labels) and torch.is_tensor(target_labels):
                        if source_labels.shape != target_labels.shape or not torch.equal(
                            source_labels.detach().cpu(), target_labels.detach().cpu()
                        ):
                            raise ValueError("Theseus calibration loaders are not label-aligned.")
                    _encode_image(source_model, source_imgs)
                    if source_model_ft is not None:
                        _encode_image(source_model_ft, source_imgs)
                    _encode_image(target_model, target_imgs)
                    row_mask = None
            consumed_batches += 1

            if source_activation_plan is not None:
                source_activation_plan.apply(source_hook.inputs)
                source_activation_plan.apply(source_hook.outputs)

            def _accumulate(hook: _ActivationHook, *, prefix: str, row_mask=row_mask) -> None:
                for side, source_side, target_side in (
                    ("in", hook.inputs, target_hook.inputs),
                    ("out", hook.outputs, target_hook.outputs),
                ):
                    for key in set(source_side.keys()) & set(target_side.keys()):
                        src_rows, tgt_rows = _align_features(source_side[key], target_side[key], mode=seq_align)
                        src_rows, tgt_rows = _drop_padding_rows(src_rows, tgt_rows, row_mask)
                        registry.setdefault(
                            f"{prefix}{key}.{side}",
                            ActivationStore(
                                store_raw=store_raw,
                                store_a_gram=store_a_gram,
                                store_b_gram=store_b_gram,
                            ),
                        ).update(src_rows, tgt_rows)

            _accumulate(source_hook, prefix="")
            if source_ft_hook is not None:
                _accumulate(source_ft_hook, prefix=_FT_REGISTRY_PREFIX)

            source_hook.clear()
            target_hook.clear()
            if source_ft_hook is not None:
                source_ft_hook.clear()
        if n_batches is not None and consumed_batches < int(n_batches):
            raise ValueError(
                f"Theseus calibration loaders exhausted after {consumed_batches} batches; requested {int(n_batches)}."
            )
    finally:
        source_hook.remove()
        target_hook.remove()
        if source_ft_hook is not None:
            source_ft_hook.remove()

    return registry


@dataclass(frozen=True)
class TheseusRebase:
    """Transport matrix task-vector updates with activation-aligned layer maps.

    ``prepare`` estimates source-to-target coordinate transforms from activations
    or data-free covariance information. ``apply`` reuses those transforms for
    a compatible delta, making alpha sweeps cheaper than rebuilding alignment.
    """

    name: str = "theseus"

    def prepare(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        source_dataloader: Iterable[Any],
        target_dataloader: Iterable[Any],
        target_base: Mapping[str, torch.Tensor] | None = None,
        delta: Mapping[str, torch.Tensor] | None = None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        verbose: bool = True,
        show_progress: bool = True,
        family_adapter: Any = None,
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        covariance_mode: str = "activations",
        covariance_source: str = "base",
        covariance_mixture_beta: float = 0.5,
        source_model_ft: torch.nn.Module | None = None,
        source_activation_plan: InterpolatedBlockActivations | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        activation_cache_dir = kwargs.pop("activation_cache_dir", None)
        activation_cache_mode = str(kwargs.pop("activation_cache_mode", "off")).lower()
        activation_cache_key = kwargs.pop("activation_cache_key", None)
        if activation_cache_mode not in {"off", "auto", "load", "refresh"}:
            raise ValueError("activation_cache_mode must be one of: off, auto, load, refresh")
        if activation_cache_mode != "off" and not activation_cache_dir:
            raise ValueError("activation_cache_dir is required when activation_cache_mode is enabled")
        split_qkv = kwargs.pop("split_qkv", None)
        if split_qkv is not None:
            patch_qkv = bool(split_qkv)
        transform_granularity = str(kwargs.pop("transform_granularity", "param")).strip().lower()
        if transform_granularity not in {"param", "module_type", "block", "global"}:
            raise ValueError("transform_granularity must be one of: param, module_type, block, global")
        device_transform = str(kwargs.pop("device_transform", "cpu")).strip().lower()
        if device_transform not in {"cpu", "gpu"}:
            raise ValueError("device_transform must be one of: cpu, gpu")
        svd_device = device if device_transform == "gpu" else "cpu"
        del kwargs
        #Config fallbacks num_batches -> n_batches
        if n_batches is None:
            n_batches = num_batches
        log_prefix = f"[{self.name}]"

        covariance_mode = _resolve_covariance_mode(covariance_mode)
        covariance_source = _resolve_covariance_source(covariance_source)
        covariance_mixture_beta = float(covariance_mixture_beta)
        whiten_power = float(whiten_power)
        whiten_eps = float(whiten_eps)
        if not 0.0 <= whiten_power <= 0.5:
            raise ValueError("Theseus whiten_power must be in [0, 0.5].")
        if covariance_source != "base":
            if covariance_mode != "activations":
                raise ValueError(
                    "covariance_source only reweights streamed activations and is undefined for "
                    f"covariance_mode='{covariance_mode}'."
                )
            if source_model_ft is None:
                raise ValueError(f"covariance_source='{covariance_source}' requires source_model_ft.")
            if whiten_power > 0.0:
                # Whitening needs the source Gram of the effective rows, which
                # is quadratic and therefore not recoverable from the two
                # accumulated banks.  Refuse rather than whiten the wrong Gram.
                raise ValueError("Theseus whitening is only implemented for covariance_source='base'.")
            if source_activation_plan is not None:
                raise ValueError(
                    "The interpolated-activation baseline substitutes base-endpoint activations and is "
                    f"undefined for covariance_source='{covariance_source}'."
                )
        elif source_model_ft is not None:
            raise ValueError("source_model_ft was supplied but covariance_source='base' would ignore it.")
        # Validate the mixture weight even when it is inactive, so a typo in a
        # config is rejected at prepare time rather than silently ignored.
        _covariance_source_coefficients("mixture", covariance_mixture_beta)
        if source_activation_plan is not None and covariance_mode != "activations":
            raise ValueError(
                "The interpolated-activation baseline substitutes collected activations and is "
                f"undefined for covariance_mode='{covariance_mode}'."
            )
        if whiten_eps <= 0.0:
            raise ValueError("Theseus whiten_eps must be > 0.")

        if verbose:
            print(
                f"{log_prefix} prepare: start "
                f"(seq_align={seq_align}, center_acts={bool(center_acts)}, n_batches={n_batches}, "
                f"seed={int(seed)}, transform_granularity={transform_granularity}, "
                f"device_transform={device_transform}, covariance_mode={covariance_mode}, "
                f"covariance_source={covariance_source}, "
                f"covariance_mixture_beta={covariance_mixture_beta}, "
                f"whiten_power={whiten_power})"
            )

        patched_source = 0
        patched_target = 0
        if patch_qkv:
            if verbose:
                print(f"{log_prefix} prepare: patching fused qkv blocks if needed")
            patched_source = _split_fused_qkv_if_needed(source_model)
            if source_model_ft is not None:
                _split_fused_qkv_if_needed(source_model_ft)
            patched_target = _split_fused_qkv_if_needed(target_model)
            if patched_source > 0 or patched_target > 0:
                logger.info(
                    "%s prepare: split fused qkv attention blocks (source=%d, target=%d)",
                    self.name,
                    patched_source,
                    patched_target,
                )
        elif verbose:
            print(f"{log_prefix} prepare: patch_qkv disabled")

        activation_registry: dict[str, ActivationStore] = {}
        transforms_by_key: dict[str, _LayerTransform] = {}
        precompute_diag = _PrecomputeDiagnostics(
            slots=0,
            usable=0,
            intentional_zero=0,
            incomplete=0,
            unsupported=0,
            skipped_not_in_target=0,
            examples={},
            assigned_keys=0,
            shared_transform_count=0,
            shared_group_count=0,
        )
        split_fused_qkv = bool(patch_qkv and (patched_source > 0 or patched_target > 0))
        unpatched_source = 0
        unpatched_target = 0
        try:
            if covariance_mode == "activations":
                cache_path: Path | None = None
                if activation_cache_mode != "off":
                    fingerprint = _activation_cache_fingerprint(
                        source_model=source_model,
                        target_model=target_model,
                        source_dataloader=source_dataloader,
                        target_dataloader=target_dataloader,
                        seq_align=seq_align,
                        n_batches=n_batches,
                        seed=int(seed),
                        batch_size=batch_size,
                        cache_key=activation_cache_key,
                        whiten_power=whiten_power,
                        whiten_eps=whiten_eps,
                        source_activation_plan=source_activation_plan,
                        source_model_ft=source_model_ft,
                    )
                    cache_path = Path(activation_cache_dir) / f"theseus_activations_{fingerprint}.pt"
                    if activation_cache_mode != "refresh" and cache_path.exists():
                        activation_registry = _activation_registry_from_payload(
                            torch.load(cache_path, map_location="cpu", weights_only=True)
                        )
                        if verbose:
                            print(f"{log_prefix} prepare: loaded activations from {cache_path}")
                    elif activation_cache_mode == "load":
                        raise FileNotFoundError(f"Theseus activation cache not found: {cache_path}")

                if not activation_registry:
                    if verbose:
                        print(f"{log_prefix} prepare: collecting activations")
                    activation_registry = collect_activations(
                        source_model,
                        target_model,
                        source_dataloader,
                        target_dataloader,
                        device=device,
                        seq_align=seq_align,
                        n_batches=n_batches,
                        seed=int(seed),
                        batch_size=batch_size,
                        store_a_gram=whiten_power > 0.0,
                        store_b_gram=whiten_power > 0.0,
                        family_adapter=family_adapter,
                        source_activation_plan=source_activation_plan,
                        source_model_ft=source_model_ft,
                    )
                    if cache_path is not None:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
                        torch.save(_activation_registry_payload(activation_registry), temporary_path)
                        temporary_path.replace(cache_path)
                        if verbose:
                            print(f"{log_prefix} prepare: saved activations to {cache_path}")
                activation_registry = _combine_activation_registry(
                    activation_registry,
                    covariance_source=covariance_source,
                    mixture_beta=covariance_mixture_beta,
                )
                if verbose:
                    print(
                        f"{log_prefix} prepare: collected activation entries = {len(activation_registry)} "
                        f"(covariance_source={covariance_source})"
                    )
            elif verbose:
                print(f"{log_prefix} prepare: skipping activation collection (data-free covariance mode)")

            if target_base is not None and delta is not None:
                if verbose:
                    print(f"{log_prefix} prepare: precomputing per-layer transforms")

                if family_adapter is not None:
                    tp_keys = family_adapter.transportable_keys(target_base)
                    visual_key_map = {k: k for k in delta if k in tp_keys}
                    target_visual_base = {k: target_base[k] for k in visual_key_map.values() if k in target_base}
                    visual_delta = {k: delta[k] for k in visual_key_map if k in target_visual_base}
                else:
                    visual_key_map = _visual_delta_keys(delta)
                    target_visual_base = _visual_state_dict(target_base)
                    visual_delta = {
                        stripped_key: delta[original_key]
                        for stripped_key, original_key in visual_key_map.items()
                        if stripped_key in target_visual_base
                    }

                if split_fused_qkv and family_adapter is None:
                    target_visual_base = _split_fused_qkv_state(target_visual_base)
                    visual_delta = _split_fused_qkv_state(visual_delta)

                if covariance_mode == "activations":
                    transforms_by_key, precompute_diag = _precompute_transforms(
                        target_model=target_model,
                        target_visual_base=target_visual_base,
                        visual_delta=visual_delta,
                        activation_registry=activation_registry,
                        center_acts=bool(center_acts),
                        transform_granularity=transform_granularity,
                        show_progress=bool(show_progress),
                        method_name=self.name,
                        svd_device=svd_device,
                        family_adapter=family_adapter,
                        whiten_power=whiten_power,
                        whiten_eps=whiten_eps,
                    )
                else:
                    if family_adapter is not None:
                        source_visual_base = {k: source_model.state_dict()[k] for k in delta if k in tp_keys}
                    else:
                        source_visual_base = _visual_state_dict(source_model.state_dict())
                    if split_fused_qkv and family_adapter is None:
                        source_visual_base = _split_fused_qkv_state(source_visual_base)
                    transforms_by_key = _precompute_transforms_data_free(
                        source_visual_base=source_visual_base,
                        target_visual_base=target_visual_base,
                        visual_delta=visual_delta,
                        whiten_power=whiten_power,
                        whiten_eps=whiten_eps,
                        show_progress=bool(show_progress),
                        method_name=self.name,
                    )
                    precompute_diag = _precompute_diagnostics_from_transforms(
                        target_visual_base=target_visual_base,
                        visual_delta=visual_delta,
                        transforms_by_key=transforms_by_key,
                    )
                _report_precompute_diagnostics(
                    method_name=self.name,
                    diagnostics=precompute_diag,
                    verbose=bool(verbose),
                )
                if verbose:
                    if covariance_mode == "activations" and transform_granularity != "param":
                        print(
                            f"{log_prefix} prepare: computed usable transforms = {precompute_diag.usable} "
                            f"(shared={precompute_diag.shared_transform_count}, groups={precompute_diag.shared_group_count})"
                        )
                    else:
                        print(f"{log_prefix} prepare: computed usable transforms = {precompute_diag.usable}")
            elif verbose:
                print(f"{log_prefix} prepare: target_base/delta missing, skipping transform precompute")
        finally:
            if patch_qkv and (patched_source > 0 or patched_target > 0):
                try:
                    unpatched_source = int(merge_openclip_vit_attn(_visual_module(source_model)))
                    unpatched_target = int(merge_openclip_vit_attn(_visual_module(target_model)))
                    if verbose:
                        print(
                            f"{log_prefix} prepare: recomposed fused qkv blocks "
                            f"(source={unpatched_source}, target={unpatched_target})"
                        )
                except Exception as exc:
                    logger.warning("%s prepare: failed to recompose patched attention blocks: %s", self.name, exc)

        if verbose:
            print(f"{log_prefix} prepare: done")

        return {
            "activation_registry": activation_registry,
            "transforms_by_key": transforms_by_key,
            "split_fused_qkv": split_fused_qkv,
            "n_batches": n_batches,
            "patched_source_blocks": patched_source,
            "patched_target_blocks": patched_target,
            "unpatched_source_blocks": unpatched_source,
            "unpatched_target_blocks": unpatched_target,
            "transform_granularity": transform_granularity,
            "covariance_mode": covariance_mode,
            "covariance_source": covariance_source,
            "covariance_mixture_beta": covariance_mixture_beta,
            "device_transform": device_transform,
            "compute_device": _resolve_device(device) if device_transform == "gpu" else torch.device("cpu"),
            "precompute_diagnostics": {
                "slots": precompute_diag.slots,
                "usable": precompute_diag.usable,
                "intentional_zero": precompute_diag.intentional_zero,
                "incomplete": precompute_diag.incomplete,
                "unsupported": precompute_diag.unsupported,
                "skipped_not_in_target": precompute_diag.skipped_not_in_target,
                "examples": dict(precompute_diag.examples),
                "assigned_keys": precompute_diag.assigned_keys,
                "shared_transform_count": precompute_diag.shared_transform_count,
                "shared_group_count": precompute_diag.shared_group_count,
            },
        }

    def apply(
        self,
        prepared: Mapping[str, Any],
        *,
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        verbose: bool = True,
        show_progress: bool = True,
        family_adapter: Any = None,
        **kwargs,
    ) -> TensorDict:
        del kwargs
        log_prefix = f"[{self.name}]"

        if verbose:
            print(f"{log_prefix} apply: start")

        transforms_by_key = prepared.get("transforms_by_key", None)
        if transforms_by_key is None:
            raise ValueError("Theseus prepared payload is missing 'transforms_by_key'.")

        if family_adapter is not None:
            tp_keys = family_adapter.transportable_keys(target_base)
            visual_key_map = {k: k for k in delta if k in tp_keys}
            target_visual_base = {k: target_base[k] for k in visual_key_map.values() if k in target_base}
            visual_delta_work = {k: delta[k] for k in visual_key_map if k in target_visual_base}
            target_visual_base_work = target_visual_base
            split_fused_qkv = False
            out_of_scope_keys = tuple(k for k in delta if k not in tp_keys and k in target_base)
            skipped_not_in_target_keys = tuple(k for k in visual_key_map if k not in target_base)
        else:
            visual_key_map = _visual_delta_keys(delta)
            target_visual_base = _visual_state_dict(target_base)

            visual_delta = {
                stripped_key: delta[original_key]
                for stripped_key, original_key in visual_key_map.items()
            }
            has_visual_keys = any(key.startswith(_VISUAL_PREFIX) for key in delta)
            out_of_scope_keys = tuple(
                key for key in delta
                if has_visual_keys and not key.startswith(_VISUAL_PREFIX) and key in target_base
            )
            skipped_not_in_target_keys = tuple(
                original_key
                for stripped_key, original_key in visual_key_map.items()
                if stripped_key not in target_visual_base and original_key not in out_of_scope_keys
            )

            split_fused_qkv = bool(prepared.get("split_fused_qkv", False))
            if split_fused_qkv:
                target_visual_base_work = _split_fused_qkv_state(target_visual_base)
                visual_delta_work = _split_fused_qkv_state(
                    {key: value for key, value in visual_delta.items() if key in target_visual_base}
                )
            else:
                target_visual_base_work = target_visual_base
                visual_delta_work = {key: value for key, value in visual_delta.items() if key in target_visual_base}

            if strict and not visual_delta_work:
                raise ValueError("Theseus did not find any visual delta keys to transport.")

        compute_device = prepared.get("compute_device", "cpu")
        aligned_visual, apply_diag = _apply_transforms_to_visual_delta(
            target_visual_base=target_visual_base_work,
            visual_delta=visual_delta_work,
            transforms_by_key=transforms_by_key,
            show_progress=bool(show_progress),
            method_name=self.name,
            device=compute_device,
            strict=bool(strict),
            out_of_scope_keys=out_of_scope_keys,
            skipped_not_in_target_keys=skipped_not_in_target_keys,
        )

        if not family_adapter and split_fused_qkv:
            aligned_visual = _merge_split_qkv_state(aligned_visual, reference=target_visual_base)

        out: TensorDict = {}
        processed: set[str] = set()

        for stripped_key, original_key in visual_key_map.items():
            if original_key not in target_base:
                continue
            if stripped_key in aligned_visual:
                out[original_key] = aligned_visual[stripped_key].to(
                    dtype=target_base[original_key].dtype,
                    device=target_base[original_key].device,
                )
            else:
                out[original_key] = torch.zeros_like(target_base[original_key], device=target_base[original_key].device)
            processed.add(original_key)

        for key in delta:
            if key in processed or key not in target_base:
                continue
            out[key] = torch.zeros_like(target_base[key], device=target_base[key].device)

        if strict:
            expected_keys = {key for key in visual_key_map.values() if key in target_base}
            missing = sorted(expected_keys - set(out.keys()))
            if missing:
                raise KeyError(f"Theseus did not transport all delta keys. Example: {missing[:10]}")

        if verbose:
            _report_apply_diagnostics(method_name=self.name, diagnostics=apply_diag, verbose=True)
            print(f"{log_prefix} apply: done (transported_keys={len(out)})")

        return out

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        source_model: torch.nn.Module | None = None,
        target_model: torch.nn.Module | None = None,
        source_dataloader: Iterable[Any] | None = None,
        target_dataloader: Iterable[Any] | None = None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        prepared: Mapping[str, Any] | None = None,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        verbose: bool = True,
        show_progress: bool = True,
        family_adapter: Any = None,
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        covariance_mode: str = "activations",
        **kwargs,
    ) -> TensorDict:
        del source_base
        log_prefix = f"[{self.name}]"

        if n_batches is None:
            n_batches = num_batches

        prepared_payload: Mapping[str, Any]
        if prepared is None:
            if source_model is None or target_model is None:
                raise ValueError("Theseus transport requires both source_model and target_model.")
            if _resolve_covariance_mode(covariance_mode) == "activations" and (
                source_dataloader is None or target_dataloader is None
            ):
                raise ValueError("Theseus transport requires both source_dataloader and target_dataloader.")

            prepared_payload = self.prepare(
                source_model=source_model,
                target_model=target_model,
                source_dataloader=source_dataloader,
                target_dataloader=target_dataloader,
                target_base=target_base,
                delta=delta,
                device=device,
                seq_align=seq_align,
                center_acts=bool(center_acts),
                n_batches=n_batches,
                seed=int(seed),
                batch_size=batch_size,
                patch_qkv=patch_qkv,
                verbose=bool(verbose),
                show_progress=bool(show_progress),
                family_adapter=family_adapter,
                whiten_power=float(whiten_power),
                whiten_eps=float(whiten_eps),
                covariance_mode=covariance_mode,
                **kwargs,
            )
        else:
            prepared_payload = prepared
            if verbose:
                print(f"{log_prefix} transport: using provided prepared payload")

        return self.apply(
            prepared_payload,
            target_base=target_base,
            delta=delta,
            strict=bool(strict),
            verbose=bool(verbose),
            show_progress=bool(show_progress),
            family_adapter=family_adapter,
        )


register(TheseusRebase())
