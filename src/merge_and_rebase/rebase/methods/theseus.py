from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from ...models.patch_openclip_attention import merge_openclip_vit_attn, split_openclip_vit_attn
from ..base import TensorDict
from ..registry import register

logger = logging.getLogger(__name__)

_VISUAL_PREFIX = "visual."
_ZERO_KEYS = {"class_embedding", "positional_embedding", "conv1.weight"}
_FUSED_IN_PROJ_WEIGHT = ".attn.in_proj_weight"
_FUSED_IN_PROJ_BIAS = ".attn.in_proj_bias"
_Q_PROJ_WEIGHT = ".attn.q_proj.weight"
_K_PROJ_WEIGHT = ".attn.k_proj.weight"
_V_PROJ_WEIGHT = ".attn.v_proj.weight"
_Q_PROJ_BIAS = ".attn.q_proj.bias"
_K_PROJ_BIAS = ".attn.k_proj.bias"
_V_PROJ_BIAS = ".attn.v_proj.bias"

_ACTIVATION_COVARIANCE_MODES = {"activation", "activations"}
_DATA_FREE_COVARIANCE_MODES = {"data_free", "data-free", "weight", "weights", "weight_space", "weight-space"}

# Registry keys collected against the fine-tuned source endpoint are stored in
# the same flat dict as the base-endpoint keys so that one activation-cache
# payload still round-trips a whole prepare call.
_FT_REGISTRY_PREFIX = "ft::"
_COVARIANCE_SOURCES = ("base", "ft", "delta", "mixture")


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _extract_model_inputs(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("pixel_values", "images", "image", "inputs", "x"):
            value = batch.get(key, None)
            if torch.is_tensor(value):
                return value
    if isinstance(batch, (tuple, list)) and batch:
        first = batch[0]
        if torch.is_tensor(first):
            return first
    raise TypeError("Unsupported batch format for Theseus calibration.")


def _extract_output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError("Unsupported module output while collecting Theseus activations.")


def _encode_image(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image") and callable(model.encode_image):
        return model.encode_image(images)
    if hasattr(model, "visual") and callable(model.visual):
        return model.visual(images)
    return model(images)


def _visual_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.visual if hasattr(model, "visual") else model


def _has_fused_mha(visual: torch.nn.Module) -> bool:
    transformer = getattr(visual, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        return False
    for block in resblocks:
        if isinstance(getattr(block, "attn", None), nn.MultiheadAttention):
            return True
    return False


def _split_fused_qkv_if_needed(model: torch.nn.Module) -> int:
    visual = _visual_module(model)
    if not _has_fused_mha(visual):
        return 0

    ref_param = next(visual.parameters(), None)
    ref_device = ref_param.device if ref_param is not None else torch.device("cpu")
    ref_dtype = ref_param.dtype if ref_param is not None else None

    

    n_patched = int(
        split_openclip_vit_attn(
            visual,
            proj_dropout=0.0,
            attn_impl="softmax",
        )
    )

    if n_patched > 0:
        if ref_dtype is None:
            visual.to(device=ref_device)
        else:
            visual.to(device=ref_device, dtype=ref_dtype)

    return n_patched


def _visual_state_dict(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    visual = {key[len(_VISUAL_PREFIX) :]: value for key, value in sd.items() if key.startswith(_VISUAL_PREFIX)}
    return visual if visual else dict(sd)


def _visual_delta_keys(delta: Mapping[str, torch.Tensor]) -> dict[str, str]:
    visual = {key[len(_VISUAL_PREFIX) :]: key for key in delta if key.startswith(_VISUAL_PREFIX)}
    if visual:
        return visual
    return {key: key for key in delta}


def _split_fused_qkv_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith(_FUSED_IN_PROJ_WEIGHT) and value.ndim == 2 and value.shape[0] % 3 == 0:
            base = key[: -len(_FUSED_IN_PROJ_WEIGHT)]
            c = value.shape[0] // 3
            out[f"{base}{_Q_PROJ_WEIGHT}"] = value[:c, :]
            out[f"{base}{_K_PROJ_WEIGHT}"] = value[c : 2 * c, :]
            out[f"{base}{_V_PROJ_WEIGHT}"] = value[2 * c :, :]
            continue

        if key.endswith(_FUSED_IN_PROJ_BIAS) and value.ndim == 1 and value.shape[0] % 3 == 0:
            base = key[: -len(_FUSED_IN_PROJ_BIAS)]
            c = value.shape[0] // 3
            out[f"{base}{_Q_PROJ_BIAS}"] = value[:c]
            out[f"{base}{_K_PROJ_BIAS}"] = value[c : 2 * c]
            out[f"{base}{_V_PROJ_BIAS}"] = value[2 * c :]
            continue

        out[key] = value
    return out


def _merge_split_qkv_state(
    state: Mapping[str, torch.Tensor],
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = dict(state)

    def _merge_triplet(q_suffix: str, k_suffix: str, v_suffix: str, fused_suffix: str) -> None:
        prefixes: set[str] = set()
        for k in tuple(out.keys()):
            if k.endswith(q_suffix):
                prefixes.add(k[: -len(q_suffix)])
            elif k.endswith(k_suffix):
                prefixes.add(k[: -len(k_suffix)])
            elif k.endswith(v_suffix):
                prefixes.add(k[: -len(v_suffix)])

        for p in prefixes:
            qk = f"{p}{q_suffix}"
            kk = f"{p}{k_suffix}"
            vk = f"{p}{v_suffix}"
            fused = f"{p}{fused_suffix}"
            if qk not in out or kk not in out or vk not in out:
                continue
            if reference is not None and fused not in reference:
                continue

            merged = torch.cat([out[qk], out[kk], out[vk]], dim=0)
            out[fused] = merged
            del out[qk]
            del out[kk]
            del out[vk]

    _merge_triplet(_Q_PROJ_WEIGHT, _K_PROJ_WEIGHT, _V_PROJ_WEIGHT, _FUSED_IN_PROJ_WEIGHT)
    _merge_triplet(_Q_PROJ_BIAS, _K_PROJ_BIAS, _V_PROJ_BIAS, _FUSED_IN_PROJ_BIAS)
    return out


def _is_square(n: int) -> bool:
    if n <= 0:
        return False
    r = int(n**0.5)
    return r * r == n


def _standardize_tokens(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if x.ndim == 1:
        return x.view(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        if x.shape[0] == batch_size:
            return x
        if x.shape[1] == batch_size:
            return x.transpose(0, 1)
        return x
    if x.ndim == 4:
        return x
    return x.reshape(batch_size, -1, x.shape[-1])


def _to_tokens(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    x = _standardize_tokens(x, batch_size=batch_size)
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
    if x.ndim == 3:
        return x
    return x.reshape(batch_size, -1, x.shape[-1])


def _interp_linear_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[1] == target_tokens:
        return tokens
    tokens_t = tokens.transpose(1, 2)
    tokens_t = F.interpolate(tokens_t, size=target_tokens, mode="linear", align_corners=False)
    return tokens_t.transpose(1, 2)


def _interp_2d_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[1] == target_tokens:
        return tokens

    has_cls = _is_square(tokens.shape[1] - 1) and _is_square(target_tokens - 1)
    cls_token: torch.Tensor | None = None
    patch_tokens = tokens
    target_patch_tokens = target_tokens

    if has_cls:
        cls_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]
        target_patch_tokens = target_tokens - 1

    if not _is_square(patch_tokens.shape[1]) or not _is_square(target_patch_tokens):
        resized = _interp_linear_tokens(patch_tokens, target_patch_tokens)
        return torch.cat([cls_token, resized], dim=1) if cls_token is not None else resized

    src_side = int(patch_tokens.shape[1] ** 0.5)
    tgt_side = int(target_patch_tokens ** 0.5)
    x = patch_tokens.reshape(tokens.shape[0], src_side, src_side, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(tgt_side, tgt_side), mode="bilinear", align_corners=False)
    resized = x.permute(0, 2, 3, 1).reshape(tokens.shape[0], tgt_side * tgt_side, patch_tokens.shape[-1])
    return torch.cat([cls_token, resized], dim=1) if cls_token is not None else resized


def _align_features(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_feat.shape[0] != target_feat.shape[0]:
        raise ValueError(
            "Theseus calibration expects aligned batch sizes. "
            f"Got {source_feat.shape[0]} and {target_feat.shape[0]}."
        )

    source_tokens = _to_tokens(source_feat, batch_size=int(source_feat.shape[0]))
    target_tokens = _to_tokens(target_feat, batch_size=int(target_feat.shape[0]))

    if mode == "cls":
        source_tokens = source_tokens[:, :1, :]
        target_tokens = target_tokens[:, :1, :]
    elif mode == "mean":
        source_tokens = source_tokens.mean(dim=1, keepdim=True)
        target_tokens = target_tokens.mean(dim=1, keepdim=True)
    elif mode in {"interpolate2d", "interpolate_2d"}:
        source_tokens = _interp_2d_tokens(source_tokens, int(target_tokens.shape[1]))
    elif mode == "interpolate":
        source_tokens = _interp_linear_tokens(source_tokens, int(target_tokens.shape[1]))

    return source_tokens.reshape(-1, source_tokens.shape[-1]), target_tokens.reshape(-1, target_tokens.shape[-1])


class ActivationStore:
    """Streaming activation statistics with optional Gram and raw storage."""

    def __init__(self, *, store_raw: bool = False, store_a_gram: bool = False, store_b_gram: bool = False) -> None:
        self.store_raw = bool(store_raw)
        self.store_a_gram = bool(store_a_gram)
        self.store_b_gram = bool(store_b_gram)

        self.at_b: torch.Tensor | None = None
        self.at_a: torch.Tensor | None = None
        self.bt_b: torch.Tensor | None = None
        self.sum_a: torch.Tensor | None = None
        self.sum_b: torch.Tensor | None = None
        self.n_samples = 0

        self.h_a_list: list[torch.Tensor] = []
        self.h_b_list: list[torch.Tensor] = []

    def update(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        a = batch_a.detach().cpu().to(torch.float64)
        b = batch_b.detach().cpu().to(torch.float64)

        if self.store_raw:
            self.h_a_list.append(a.float())
            self.h_b_list.append(b.float())

        if self.at_b is None:
            self.at_b = a.T @ b
            self.sum_a = a.sum(dim=0)
            self.sum_b = b.sum(dim=0)
            if self.store_a_gram:
                self.at_a = a.T @ a
            if self.store_b_gram:
                self.bt_b = b.T @ b
        else:
            self.at_b += a.T @ b
            self.sum_a += a.sum(dim=0)
            self.sum_b += b.sum(dim=0)
            if self.store_a_gram and self.at_a is not None:
                self.at_a += a.T @ a
            if self.store_b_gram and self.bt_b is not None:
                self.bt_b += b.T @ b

        self.n_samples += int(a.shape[0])

    def rows(self, *, center: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.store_raw or not self.h_a_list:
            return None, None
        source = torch.cat(self.h_a_list, dim=0)
        target = torch.cat(self.h_b_list, dim=0)
        if center:
            source = source - source.mean(dim=0, keepdim=True)
            target = target - target.mean(dim=0, keepdim=True)
        return source, target

    def get_covariance(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.at_b is None:
            return None
        cov = self.at_b.clone()
        if center:
            assert self.sum_a is not None and self.sum_b is not None
            mu_a = self.sum_a / self.n_samples
            mu_b = self.sum_b / self.n_samples
            cov = cov - self.n_samples * torch.outer(mu_a, mu_b)
        if epsilon > 0 and cov.shape[0] == cov.shape[1]:
            cov = cov + epsilon * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        return cov

    def get_a_gram(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.at_a is None:
            return None
        gram = self.at_a.clone()
        if center:
            assert self.sum_a is not None
            mu_a = self.sum_a / self.n_samples
            gram = gram - self.n_samples * torch.outer(mu_a, mu_a)
        if epsilon > 0 and gram.shape[0] == gram.shape[1]:
            gram = gram + epsilon * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return gram

    def get_b_gram(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.bt_b is None:
            return None
        gram = self.bt_b.clone()
        if center:
            assert self.sum_b is not None
            mu_b = self.sum_b / self.n_samples
            gram = gram - self.n_samples * torch.outer(mu_b, mu_b)
        if epsilon > 0 and gram.shape[0] == gram.shape[1]:
            gram = gram + epsilon * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return gram


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


@dataclass(frozen=True)
class InterpolatedBlockActivations:
    """Source-side activation override for the ARIADNE interpolated-activation baseline.

    ARIADNE inserts a block into the source model and fits it so that its
    activations reproduce a reference bank. This baseline asks what a width
    transport method would do if the inserted position carried no computed
    activations at all: at every component module of an inserted block, the
    source rows are replaced by the midpoint of the same component's rows in
    the two original blocks whose weights initialized it. Target rows and
    original source positions are never touched, so the substitution ablates
    exactly one thing — the inserted block's own forward pass.

    ``entries`` holds ``(inserted_position, left_position, right_position)``
    triples over the extended block list, as produced by
    ``merge_and_rebase.eval.block_extension.build_extension_layout``.
    """

    entries: tuple[tuple[int, int, int], ...]
    block_prefix: str = "transformer.resblocks"

    @classmethod
    def from_extension_layout(
        cls,
        layout: Mapping[str, Any],
        *,
        block_prefix: str = "transformer.resblocks",
    ) -> InterpolatedBlockActivations:
        entries = tuple(
            (int(block["position"]), int(block["source_position"]), int(block["neighbour_position"]))
            for block in layout.get("inserted_blocks", ())
        )
        if not entries:
            raise ValueError(
                "The interpolated-activation baseline needs at least one inserted block; "
                "the recorded extension layout has none."
            )
        return cls(entries=entries, block_prefix=str(block_prefix))

    def fingerprint(self) -> str:
        payload = self.block_prefix + "|" + ";".join(
            f"{position}:{left}:{right}" for position, left, right in self.entries
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def apply(self, store: dict[str, torch.Tensor]) -> None:
        """Replace every inserted-position tensor in ``store`` with the neighbour midpoint."""

        if not self.entries:
            return
        # Snapshot first: inserted positions read only original positions, but
        # a snapshot makes that independent of iteration order.
        captured = dict(store)
        for position, left, right in self.entries:
            prefix = f"{self.block_prefix}.{position}"
            for key in list(store.keys()):
                if key != prefix and not key.startswith(f"{prefix}."):
                    continue
                suffix = key[len(prefix) :]
                left_key = f"{self.block_prefix}.{left}{suffix}"
                right_key = f"{self.block_prefix}.{right}{suffix}"
                left_rows = captured.get(left_key)
                right_rows = captured.get(right_key)
                if left_rows is None or right_rows is None:
                    missing = left_key if left_rows is None else right_key
                    raise KeyError(
                        "Interpolated-activation baseline needs both neighbour activations for "
                        f"'{key}'; '{missing}' was not captured."
                    )
                if left_rows.shape != right_rows.shape:
                    raise ValueError(
                        f"Neighbour activations for '{key}' disagree in shape: "
                        f"{tuple(left_rows.shape)} vs {tuple(right_rows.shape)}."
                    )
                store[key] = (0.5 * (left_rows.float() + right_rows.float())).to(store[key].dtype)


class _ActivationHook:
    def __init__(self, model: torch.nn.Module, *, scope: torch.nn.Module | None = None):
        self.model = scope if scope is not None else _visual_module(model)
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.handles.append(self.model.register_forward_hook(self._make_hook("")))
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if list(module.parameters(recurse=False)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook_fn(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            inp = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else inputs
            if torch.is_tensor(inp):
                self.inputs[name] = inp.detach().cpu()
            try:
                out = _extract_output_tensor(output)
            except TypeError:
                out = None
            if out is not None and torch.is_tensor(out):
                self.outputs[name] = out.detach().cpu()

        return hook_fn

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


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
            consumed_batches += 1

            if source_activation_plan is not None:
                source_activation_plan.apply(source_hook.inputs)
                source_activation_plan.apply(source_hook.outputs)

            def _accumulate(hook: _ActivationHook, *, prefix: str) -> None:
                for side, source_side, target_side in (
                    ("in", hook.inputs, target_hook.inputs),
                    ("out", hook.outputs, target_hook.outputs),
                ):
                    for key in set(source_side.keys()) & set(target_side.keys()):
                        src_rows, tgt_rows = _align_features(source_side[key], target_side[key], mode=seq_align)
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


def _compute_procrustes_map(source_rows: torch.Tensor, target_rows: torch.Tensor, *, center: bool) -> torch.Tensor:
    if center:
        source_rows = source_rows - source_rows.mean(dim=0, keepdim=True)
        target_rows = target_rows - target_rows.mean(dim=0, keepdim=True)
    cov = source_rows.double().T @ target_rows.double()
    u, _, v_h = torch.linalg.svd(cov, full_matrices=False)
    return (u @ v_h).float()


def _compute_procrustes_map_from_cov(cov: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    if device != "cpu":
        cov = cov.to(device=device)
    u, _, v_h = torch.linalg.svd(cov.double(), full_matrices=False)
    return (u @ v_h).float()


def _matrix_power_psd(matrix: torch.Tensor, *, power: float, eps: float) -> torch.Tensor:
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = torch.linalg.eigh(sym)
    powered = evals.clamp_min(float(eps)).pow(float(power))
    return (evecs * powered.unsqueeze(0)) @ evecs.T


def _partially_whiten_covariance(
    cov: torch.Tensor,
    *,
    a_gram: torch.Tensor,
    b_gram: torch.Tensor,
    power: float,
    eps: float,
) -> torch.Tensor:
    if power <= 0.0:
        return cov
    left = _matrix_power_psd(a_gram, power=-power, eps=eps)
    right = _matrix_power_psd(b_gram, power=-power, eps=eps)
    return left @ cov @ right


def _compute_alignment_map(
    store: ActivationStore,
    *,
    center: bool,
    whiten_power: float,
    whiten_eps: float,
) -> torch.Tensor | None:
    cov = store.get_covariance(center=center)
    if cov is None:
        return None
    if whiten_power > 0.0:
        a_gram = store.get_a_gram(center=center, epsilon=whiten_eps)
        b_gram = store.get_b_gram(center=center, epsilon=whiten_eps)
        if a_gram is not None and b_gram is not None:
            cov = _partially_whiten_covariance(
                cov,
                a_gram=a_gram,
                b_gram=b_gram,
                power=whiten_power,
                eps=whiten_eps,
            )
        else:
            logger.warning(
                "Theseus whitening requested but Gram statistics were unavailable; falling back to raw Procrustes."
            )
    return _compute_procrustes_map_from_cov(cov, device="cpu")


def _resolve_covariance_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in _ACTIVATION_COVARIANCE_MODES:
        return "activations"
    if key in _DATA_FREE_COVARIANCE_MODES:
        return "data_free"
    raise ValueError(
        "Theseus covariance_mode must be one of: activations, activation, data_free, data-free, weights, weight_space."
    )


def _compute_alignment_map_from_matrix_proxies(
    source_proxy: torch.Tensor,
    target_proxy: torch.Tensor,
    *,
    side: str,
    whiten_power: float,
    whiten_eps: float,
) -> torch.Tensor:
    source = source_proxy.detach().cpu().to(torch.float64)
    target = target_proxy.detach().cpu().to(torch.float64)

    if side == "input":
        a_gram = source.T @ source
        b_gram = target.T @ target
    elif side == "output":
        a_gram = source @ source.T
        b_gram = target @ target.T
    else:
        raise ValueError(f"Unsupported alignment side '{side}'.")

    u_a, s_a, _ = torch.linalg.svd(a_gram, full_matrices=False)
    u_b, s_b, _ = torch.linalg.svd(b_gram, full_matrices=False)
    rank = min(int(u_a.shape[1]), int(u_b.shape[1]))
    basis_power = 0.5 - float(whiten_power)
    scale_a = s_a[:rank].clamp_min(float(whiten_eps)).pow(basis_power)
    scale_b = s_b[:rank].clamp_min(float(whiten_eps)).pow(basis_power)
    source_basis = u_a[:, :rank] * scale_a.unsqueeze(0)
    target_basis = u_b[:, :rank] * scale_b.unsqueeze(0)
    cov = source_basis @ target_basis.T

    return _compute_procrustes_map_from_cov(cov, device="cpu")


def _transport_weight(delta_weight: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor, *, key: str) -> torch.Tensor:
    if key == "proj":
        return (t_out.T @ delta_weight.T @ t_in).T
    return t_out.T @ delta_weight @ t_in


def _transport_bias(delta_vec: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
    return delta_vec @ t_out


def _param_to_module(visual_model: torch.nn.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for module_name, module in visual_model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            out[full_name] = module_name
    return out


_RESBLOCK_RE = re.compile(r"^transformer\.resblocks\.(\d+)(?:\.(.*))?$")


def _activation_group(module_name: str, *, granularity: str) -> str:
    if granularity == "param":
        return module_name
    if granularity == "global":
        return "global"

    match = _RESBLOCK_RE.match(module_name)
    if match is None:
        return module_name

    block_idx = match.group(1)
    suffix = match.group(2) or ""

    if granularity == "block":
        return f"transformer.resblocks.{block_idx}"

    if granularity == "module_type":
        if suffix:
            return f"transformer.resblocks.*.{suffix}"
        return "transformer.resblocks.*"

    raise ValueError(
        "Unsupported transform_granularity. Expected one of: param, module_type, block, global. "
        f"Got: {granularity}"
    )


def _build_grouped_covariances(
    activation_registry: Mapping[str, ActivationStore],
    *,
    center_acts: bool,
    granularity: str,
) -> dict[tuple[str, str, tuple[int, int]], torch.Tensor]:
    grouped_covariances: dict[tuple[str, str, tuple[int, int]], torch.Tensor] = {}

    for act_key, store in activation_registry.items():
        if act_key.endswith(".in"):
            side = "in"
            module_name = act_key[: -len(".in")]
        elif act_key.endswith(".out"):
            side = "out"
            module_name = act_key[: -len(".out")]
        else:
            continue

        cov = store.get_covariance(center=center_acts)
        if cov is None:
            continue

        group = _activation_group(module_name, granularity=granularity)
        shape_key = (int(cov.shape[0]), int(cov.shape[1]))
        key = (group, side, shape_key)
        if key in grouped_covariances:
            grouped_covariances[key] = grouped_covariances[key] + cov
        else:
            grouped_covariances[key] = cov.clone()

    return grouped_covariances


def _build_grouped_transforms(
    grouped_covariances: Mapping[tuple[str, str, tuple[int, int]], torch.Tensor],
    *,
    show_progress: bool,
    method_name: str,
    device: str = "cpu",
) -> dict[tuple[str, str, tuple[int, int]], torch.Tensor]:
    grouped_transforms: dict[tuple[str, str, tuple[int, int]], torch.Tensor] = {}
    items = _iter_with_progress(
        grouped_covariances.items(),
        total=len(grouped_covariances),
        desc=f"{method_name}.prepare: compute shared transforms",
        enabled=show_progress,
    )
    for key, cov in items:
        grouped_transforms[key] = _compute_procrustes_map_from_cov(cov, device=device)
    return grouped_transforms


def _iter_with_progress(iterable: Any, *, total: int, desc: str, enabled: bool) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _iter_random_dataset_batches(
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    *,
    n_batches: int | None,
    seed: int,
    batch_size: int | None,
) -> Iterable[tuple[Any, Any]] | None:
    source_dataset = getattr(source_dataloader, "dataset", None)
    target_dataset = getattr(target_dataloader, "dataset", None)
    if source_dataset is None or target_dataset is None:
        return None

    try:
        n_source = int(len(source_dataset))
        n_target = int(len(target_dataset))
    except Exception:
        return None

    n_samples = min(n_source, n_target)
    if n_samples <= 0:
        return iter(())

    if batch_size is None:
        source_bs = getattr(source_dataloader, "batch_size", None)
        target_bs = getattr(target_dataloader, "batch_size", None)
        if source_bs is None or target_bs is None:
            return None
        batch_size = min(int(source_bs), int(target_bs))
    else:
        batch_size = int(batch_size)

    if batch_size <= 0:
        return None

    source_collate = getattr(source_dataloader, "collate_fn", None)
    target_collate = getattr(target_dataloader, "collate_fn", None)
    if not callable(source_collate) or not callable(target_collate):
        return None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_samples, generator=generator)

    if n_batches is not None:
        max_items = min(n_samples, int(n_batches) * batch_size)
        perm = perm[:max_items]

    def _iterator() -> Iterable[tuple[Any, Any]]:
        for start in range(0, int(perm.numel()), batch_size):
            indices = perm[start : start + batch_size].tolist()
            source_items = [source_dataset[i] for i in indices]
            target_items = [target_dataset[i] for i in indices]
            yield source_collate(source_items), target_collate(target_items)

    return _iterator()


@dataclass(frozen=True)
class _LayerTransform:
    kind: str
    t_in: torch.Tensor | None = None
    t_out: torch.Tensor | None = None


@dataclass(frozen=True)
class _PrecomputeDiagnostics:
    slots: int
    usable: int
    intentional_zero: int
    incomplete: int
    unsupported: int
    skipped_not_in_target: int
    examples: Mapping[str, tuple[str, ...]]
    assigned_keys: int
    shared_transform_count: int
    shared_group_count: int


@dataclass(frozen=True)
class _ApplyDiagnostics:
    actively_transported: int
    intentional_zero: int
    missing_transform_zero: int
    unsupported_zero: int
    transport_failure_zero: int
    wrong_shape_zero: int
    out_of_scope_zero: int
    skipped_not_in_target: int
    examples: Mapping[str, tuple[str, ...]]
    transformed_weight: int
    transformed_bias: int

    @property
    def missing_transform(self) -> int:
        return self.missing_transform_zero

    @property
    def transport_failures(self) -> int:
        return self.transport_failure_zero

    @property
    def wrong_shape(self) -> int:
        return self.wrong_shape_zero

    @property
    def zero_passthrough(self) -> int:
        return self.intentional_zero

    @property
    def skipped_not_in_target_visual(self) -> int:
        return self.skipped_not_in_target


_DIAGNOSTIC_EXAMPLE_LIMIT = 5


def _append_diagnostic_example(examples: dict[str, list[str]], category: str, key: str) -> None:
    bucket = examples.setdefault(category, [])
    if len(bucket) < _DIAGNOSTIC_EXAMPLE_LIMIT:
        bucket.append(str(key))


def _freeze_diagnostic_examples(examples: Mapping[str, list[str]]) -> dict[str, tuple[str, ...]]:
    return {category: tuple(keys) for category, keys in examples.items() if keys}


def _precompute_diagnostics_from_transforms(
    *,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    transforms_by_key: Mapping[str, _LayerTransform],
) -> _PrecomputeDiagnostics:
    counts = {"intentional_zero": 0, "incomplete": 0, "unsupported": 0, "skipped_not_in_target": 0}
    examples: dict[str, list[str]] = {}
    usable = 0
    for key, delta_source in visual_delta.items():
        if key not in target_visual_base:
            counts["skipped_not_in_target"] += 1
            _append_diagnostic_example(examples, "skipped_not_in_target", key)
            continue
        transform = transforms_by_key.get(key)
        if key in _ZERO_KEYS or (transform is not None and transform.kind == "zero"):
            counts["intentional_zero"] += 1
            _append_diagnostic_example(examples, "intentional_zero", key)
        elif transform is None:
            counts["incomplete"] += 1
            _append_diagnostic_example(examples, "incomplete", key)
        elif transform.kind == "unsupported" or delta_source.ndim not in {1, 2}:
            counts["unsupported"] += 1
            _append_diagnostic_example(examples, "unsupported", key)
        elif (
            (transform.kind == "weight" and delta_source.ndim == 2 and transform.t_in is not None and transform.t_out is not None)
            or (transform.kind == "bias" and delta_source.ndim == 1 and transform.t_out is not None)
        ):
            usable += 1
        else:
            counts["incomplete"] += 1
            _append_diagnostic_example(examples, "incomplete", key)
    return _PrecomputeDiagnostics(
        slots=len(visual_delta),
        usable=usable,
        intentional_zero=counts["intentional_zero"],
        incomplete=counts["incomplete"],
        unsupported=counts["unsupported"],
        skipped_not_in_target=counts["skipped_not_in_target"],
        examples=_freeze_diagnostic_examples(examples),
        assigned_keys=len(transforms_by_key),
        shared_transform_count=0,
        shared_group_count=0,
    )


def _report_precompute_diagnostics(
    *,
    method_name: str,
    diagnostics: _PrecomputeDiagnostics,
    verbose: bool,
) -> None:
    if verbose:
        print(
            f"[{method_name}] prepare: transport slots={diagnostics.slots} "
            f"usable={diagnostics.usable} intentional_zero={diagnostics.intentional_zero} "
            f"incomplete={diagnostics.incomplete} unsupported={diagnostics.unsupported} "
            f"skipped_not_in_target={diagnostics.skipped_not_in_target}"
        )
        if diagnostics.examples:
            print(f"[{method_name}] prepare: transport examples={diagnostics.examples}")
    unexpected = {
        category: count
        for category, count in {
            "incomplete": diagnostics.incomplete,
            "unsupported": diagnostics.unsupported,
            "skipped_not_in_target": diagnostics.skipped_not_in_target,
        }.items()
        if count
    }
    if unexpected:
        logger.warning(
            "[%s] prepare: transport coverage loss %s; examples=%s",
            method_name, unexpected, diagnostics.examples,
        )


def _report_apply_diagnostics(*, method_name: str, diagnostics: _ApplyDiagnostics, verbose: bool) -> None:
    if verbose:
        print(
            f"[{method_name}] apply: diagnostics "
            f"active={diagnostics.actively_transported} "
            f"matrices={diagnostics.transformed_weight} "
            f"vectors={diagnostics.transformed_bias} "
            f"intentional_zero={diagnostics.intentional_zero} "
            f"missing_transform_zero={diagnostics.missing_transform_zero} "
            f"unsupported_zero={diagnostics.unsupported_zero} "
            f"transport_failure_zero={diagnostics.transport_failure_zero} "
            f"wrong_shape_zero={diagnostics.wrong_shape_zero} "
            f"out_of_scope_zero={diagnostics.out_of_scope_zero} "
            f"skipped_not_in_target={diagnostics.skipped_not_in_target}"
        )
        if diagnostics.examples:
            print(f"[{method_name}] apply: transport examples={diagnostics.examples}")


def _precompute_transforms(
    *,
    target_model: torch.nn.Module,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    activation_registry: Mapping[str, ActivationStore],
    center_acts: bool,
    transform_granularity: str,
    show_progress: bool,
    method_name: str,
    svd_device: str = "cpu",
    family_adapter: Any = None,
    whiten_power: float = 0.0,
    whiten_eps: float = 1e-6,
    projection_in_key: str = "ln_post.out",
) -> tuple[dict[str, _LayerTransform], _PrecomputeDiagnostics]:
    transforms_by_key: dict[str, _LayerTransform] = {}
    t_out_cache: dict[str, torch.Tensor] = {}
    examples: dict[str, list[str]] = {}
    intentional_zero = 0
    incomplete = 0
    unsupported = 0
    skipped_not_in_target = 0
    usable = 0
    if family_adapter is not None:
        param_to_module = family_adapter.param_to_module(target_model)
    else:
        visual_model = _visual_module(target_model)
        param_to_module = _param_to_module(visual_model)
    grouped_covariances: dict[tuple[str, str, tuple[int, int]], torch.Tensor] = {}
    grouped_transforms: dict[tuple[str, str, tuple[int, int]], torch.Tensor] = {}

    if transform_granularity != "param":
        grouped_covariances = _build_grouped_covariances(
            activation_registry,
            center_acts=center_acts,
            granularity=transform_granularity,
        )
        grouped_transforms = _build_grouped_transforms(
            grouped_covariances,
            show_progress=show_progress,
            method_name=method_name,
            device=svd_device,
        )

    def _transform_for(
        module_name: str,
        *,
        side: str,
        expected_shape: tuple[int, int],
        fallback_key: str,
    ) -> torch.Tensor | None:
        if transform_granularity == "param":
            store = activation_registry.get(fallback_key)
            if store is None:
                return None
            if whiten_power > 0.0:
                return _compute_alignment_map(
                    store,
                    center=center_acts,
                    whiten_power=whiten_power,
                    whiten_eps=whiten_eps,
                )
            cov = store.get_covariance(center=center_acts)
            if cov is None:
                return None
            if (int(cov.shape[0]), int(cov.shape[1])) != expected_shape:
                return None
            return _compute_procrustes_map_from_cov(cov, device=svd_device)

        group = _activation_group(module_name, granularity=transform_granularity)
        return grouped_transforms.get((group, side, expected_shape))

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.prepare: assign transforms",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            skipped_not_in_target += 1
            _append_diagnostic_example(examples, "skipped_not_in_target", key)
            continue

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            intentional_zero += 1
            _append_diagnostic_example(examples, "intentional_zero", key)
            continue

        module_name = param_to_module.get(key, key.rsplit(".", 1)[0] if "." in key else "")
        if key == "proj":
            in_key = projection_in_key
            out_key = ".out"
            in_module = "ln_post"
            out_module = ""
        else:
            in_key = f"{module_name}.in"
            out_key = f"{module_name}.out"
            in_module = module_name
            out_module = module_name

        if delta_source.ndim == 2:
            target_ref = target_visual_base[key]
            if key == "proj":
                expected_in = (int(delta_source.shape[0]), int(target_ref.shape[0]))
                expected_out = (int(delta_source.shape[1]), int(target_ref.shape[1]))
            else:
                expected_in = (int(delta_source.shape[1]), int(target_ref.shape[1]))
                expected_out = (int(delta_source.shape[0]), int(target_ref.shape[0]))

            t_in = _transform_for(in_module, side="in", expected_shape=expected_in, fallback_key=in_key)
            t_out = _transform_for(out_module, side="out", expected_shape=expected_out, fallback_key=out_key)
            if t_in is not None and t_out is not None:

                transforms_by_key[key] = _LayerTransform(kind="weight", t_in=t_in, t_out=t_out)
                usable += 1
                continue
            transforms_by_key[key] = _LayerTransform(kind="weight")
            incomplete += 1
            _append_diagnostic_example(examples, "incomplete", key)
            continue

        if delta_source.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[:-len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    usable += 1
                    continue

            # Robustness fallback: covers uncommon ordering/edge cases where
            # the bias has no directly available weight transform yet.
            cached_t_out = t_out_cache.get(out_key)
            if cached_t_out is not None:
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=cached_t_out)
                usable += 1
                continue

            target_ref = target_visual_base[key]
            expected_out = (int(delta_source.shape[0]), int(target_ref.shape[0]))
            t_out = _transform_for(out_module, side="out", expected_shape=expected_out, fallback_key=out_key)
            if t_out is not None:
                t_out_cache[out_key] = t_out
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
                usable += 1
                continue
            transforms_by_key[key] = _LayerTransform(kind="bias")
            incomplete += 1
            _append_diagnostic_example(examples, "incomplete", key)
            continue

        transforms_by_key[key] = _LayerTransform(kind="unsupported")
        unsupported += 1
        _append_diagnostic_example(examples, "unsupported", key)

    diagnostics = _PrecomputeDiagnostics(
        slots=len(visual_delta),
        usable=usable,
        intentional_zero=intentional_zero,
        incomplete=incomplete,
        unsupported=unsupported,
        skipped_not_in_target=skipped_not_in_target,
        examples=_freeze_diagnostic_examples(examples),
        assigned_keys=len(transforms_by_key),
        shared_transform_count=(len(grouped_transforms) if transform_granularity != "param" else 0),
        shared_group_count=(len(grouped_covariances) if transform_granularity != "param" else 0),
    )
    return transforms_by_key, diagnostics


def _precompute_transforms_data_free(
    *,
    source_visual_base: Mapping[str, torch.Tensor],
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    whiten_power: float,
    whiten_eps: float,
    show_progress: bool,
    method_name: str,
) -> dict[str, _LayerTransform]:
    transforms_by_key: dict[str, _LayerTransform] = {}

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.prepare: compute data-free transforms",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base or key not in source_visual_base:
            logger.warning(
                "%s prepare: skipping transform for %s because it is absent from the source or target base",
                method_name, key,
            )
            continue

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            continue

        w_src_base = source_visual_base[key].detach().cpu().to(torch.float64)
        w_tgt_base = target_visual_base[key].detach().cpu().to(torch.float64)
        w_delta = delta_source.detach().cpu().to(torch.float64)

        if w_delta.ndim == 2:
            w_src_proxy = w_src_base + w_delta
            if key == "proj":
                t_in = _compute_alignment_map_from_matrix_proxies(
                    w_src_proxy, w_tgt_base, side="output",
                    whiten_power=whiten_power, whiten_eps=whiten_eps,
                )
                t_out = _compute_alignment_map_from_matrix_proxies(
                    w_src_proxy, w_tgt_base, side="input",
                    whiten_power=whiten_power, whiten_eps=whiten_eps,
                )
            else:
                t_in = _compute_alignment_map_from_matrix_proxies(
                    w_src_proxy, w_tgt_base, side="input",
                    whiten_power=whiten_power, whiten_eps=whiten_eps,
                )
                t_out = _compute_alignment_map_from_matrix_proxies(
                    w_src_proxy, w_tgt_base, side="output",
                    whiten_power=whiten_power, whiten_eps=whiten_eps,
                )
            transforms_by_key[key] = _LayerTransform(kind="weight", t_in=t_in, t_out=t_out)
        elif w_delta.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[:-len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    continue

            # Fall back to a one-row weight-space alignment for uncommon
            # standalone vectors whose parent weight transform is unavailable.
            t_out = _compute_alignment_map_from_matrix_proxies(
                w_src_base.unsqueeze(0),
                w_tgt_base.unsqueeze(0),
                side="input",
                whiten_power=whiten_power,
                whiten_eps=whiten_eps,
            )
            transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
        else:
            transforms_by_key[key] = _LayerTransform(kind="unsupported")

    return transforms_by_key


class _WrongTransportShape(ValueError):
    def __init__(self, actual: torch.Size, expected: torch.Size):
        self.actual = tuple(actual)
        self.expected = tuple(expected)
        super().__init__(f"got {self.actual}, expected {self.expected}")


def _apply_transforms_to_visual_delta(
    *,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    transforms_by_key: Mapping[str, _LayerTransform],
    show_progress: bool,
    method_name: str,
    device: str = "cpu",
    strict: bool = False,
    out_of_scope_keys: Iterable[str] = (),
    skipped_not_in_target_keys: Iterable[str] = (),
) -> tuple[TensorDict, _ApplyDiagnostics]:
    aligned: TensorDict = {}
    actively_transported = 0
    intentional_zero = 0
    missing_transform_zero = 0
    unsupported_zero = 0
    transport_failure_zero = 0
    wrong_shape_zero = 0
    out_of_scope_zero = 0
    skipped_not_in_target = 0
    transformed_weight = 0
    transformed_bias = 0
    examples: dict[str, list[str]] = {}

    for key in out_of_scope_keys:
        out_of_scope_zero += 1
        _append_diagnostic_example(examples, "out_of_scope_zero", key)
    for key in skipped_not_in_target_keys:
        skipped_not_in_target += 1
        _append_diagnostic_example(examples, "skipped_not_in_target", key)

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.apply: transport params",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            skipped_not_in_target += 1
            _append_diagnostic_example(examples, "skipped_not_in_target", key)
            continue

        target_ref = target_visual_base[key]
        transported = torch.zeros_like(target_ref, dtype=torch.float32, device=device)

        transform = transforms_by_key.get(key)
        if transform is not None and transform.kind == "zero":
            intentional_zero += 1
            _append_diagnostic_example(examples, "intentional_zero", key)
        elif transform is None:
            missing_transform_zero += 1
            _append_diagnostic_example(examples, "missing_transform_zero", key)
        elif transform.kind == "unsupported" or delta_source.ndim not in {1, 2}:
            unsupported_zero += 1
            _append_diagnostic_example(examples, "unsupported_zero", key)
        elif transform.kind == "weight" and delta_source.ndim == 2 and transform.t_in is not None and transform.t_out is not None:
            try:
                candidate = _transport_weight(
                    delta_source.float().to(device=device), transform.t_in, transform.t_out, key=key
                )
                if candidate.shape != target_ref.shape:
                    raise _WrongTransportShape(candidate.shape, target_ref.shape)
                transported = candidate
                transformed_weight += 1
                actively_transported += 1
            except _WrongTransportShape as exc:
                wrong_shape_zero += 1
                _append_diagnostic_example(examples, "wrong_shape_zero", key)
                logger.warning(
                    "%s transport produced wrong shape for %s: got %s expected %s; zeroing",
                    method_name, key, exc.actual, exc.expected,
                )
            except (RuntimeError, ValueError) as exc:
                transport_failure_zero += 1
                _append_diagnostic_example(examples, "transport_failure_zero", key)
                logger.warning("%s transport failed for %s: %s; zeroing", method_name, key, exc)
        elif transform.kind == "bias" and delta_source.ndim == 1 and transform.t_out is not None:
            try:
                candidate = _transport_bias(delta_source.float().to(device=device), transform.t_out)
                if candidate.shape != target_ref.shape:
                    raise _WrongTransportShape(candidate.shape, target_ref.shape)
                transported = candidate
                transformed_bias += 1
                actively_transported += 1
            except _WrongTransportShape as exc:
                wrong_shape_zero += 1
                _append_diagnostic_example(examples, "wrong_shape_zero", key)
                logger.warning(
                    "%s transport produced wrong shape for %s: got %s expected %s; zeroing",
                    method_name, key, exc.actual, exc.expected,
                )
            except (RuntimeError, ValueError) as exc:
                transport_failure_zero += 1
                _append_diagnostic_example(examples, "transport_failure_zero", key)
                logger.warning("%s vector transport failed for %s: %s; zeroing", method_name, key, exc)
        else:
            missing_transform_zero += 1
            _append_diagnostic_example(examples, "missing_transform_zero", key)

        aligned[key] = transported.to(dtype=target_ref.dtype, device=target_ref.device)

    diagnostics = _ApplyDiagnostics(
        actively_transported=actively_transported,
        intentional_zero=intentional_zero,
        missing_transform_zero=missing_transform_zero,
        unsupported_zero=unsupported_zero,
        transport_failure_zero=transport_failure_zero,
        wrong_shape_zero=wrong_shape_zero,
        out_of_scope_zero=out_of_scope_zero,
        skipped_not_in_target=skipped_not_in_target,
        examples=_freeze_diagnostic_examples(examples),
        transformed_weight=transformed_weight,
        transformed_bias=transformed_bias,
    )
    unexpected = {
        category: count
        for category, count in {
            "missing_transform_zero": missing_transform_zero,
            "unsupported_zero": unsupported_zero,
            "transport_failure_zero": transport_failure_zero,
            "wrong_shape_zero": wrong_shape_zero,
            "skipped_not_in_target": skipped_not_in_target,
        }.items()
        if count
    }
    if unexpected:
        logger.warning(
            "%s transport diagnostics: unexpected loss %s; examples=%s",
            method_name, unexpected, diagnostics.examples,
        )
    if strict and unexpected:
        raise RuntimeError(
            f"{method_name} strict transport failed: "
            f"missing_transform_zero={diagnostics.missing_transform_zero}, "
            f"unsupported_zero={diagnostics.unsupported_zero}, "
            f"transport_failure_zero={diagnostics.transport_failure_zero}, "
            f"wrong_shape_zero={diagnostics.wrong_shape_zero}, "
            f"skipped_not_in_target={diagnostics.skipped_not_in_target}, "
            f"examples={diagnostics.examples}"
        )
    return aligned, diagnostics


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
