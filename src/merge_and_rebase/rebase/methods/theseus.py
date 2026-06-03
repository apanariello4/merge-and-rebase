from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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


class _ActivationHook:
    def __init__(self, model: torch.nn.Module):
        self.model = _visual_module(model)
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
) -> dict[str, ActivationStore]:
    registry: dict[str, ActivationStore] = {}
    source_hook = _ActivationHook(source_model)
    target_hook = _ActivationHook(target_model)
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

        for idx, (source_batch, target_batch) in enumerate(iterator):
            if n_batches is not None and idx >= n_batches:
                break

            source_imgs = _extract_model_inputs(source_batch).to(dev)
            target_imgs = _extract_model_inputs(target_batch).to(dev)
            if source_imgs.shape[0] != target_imgs.shape[0]:
                raise ValueError(
                    "Theseus calibration expects aligned batch sizes. "
                    f"Got {source_imgs.shape[0]} and {target_imgs.shape[0]}."
                )

            _encode_image(source_model, source_imgs)
            _encode_image(target_model, target_imgs)

            common_inputs = set(source_hook.inputs.keys()) & set(target_hook.inputs.keys())
            common_outputs = set(source_hook.outputs.keys()) & set(target_hook.outputs.keys())

            for key in common_inputs:
                src_rows, tgt_rows = _align_features(source_hook.inputs[key], target_hook.inputs[key], mode=seq_align)
                reg_key = f"{key}.in"
                registry.setdefault(
                    reg_key,
                    ActivationStore(
                        store_raw=store_raw,
                        store_a_gram=store_a_gram,
                        store_b_gram=store_b_gram,
                    ),
                ).update(src_rows, tgt_rows)

            for key in common_outputs:
                src_rows, tgt_rows = _align_features(source_hook.outputs[key], target_hook.outputs[key], mode=seq_align)
                reg_key = f"{key}.out"
                registry.setdefault(
                    reg_key,
                    ActivationStore(
                        store_raw=store_raw,
                        store_a_gram=store_a_gram,
                        store_b_gram=store_b_gram,
                    ),
                ).update(src_rows, tgt_rows)

            source_hook.clear()
            target_hook.clear()
    finally:
        source_hook.remove()
        target_hook.remove()

    return registry


def _compute_procrustes_map(source_rows: torch.Tensor, target_rows: torch.Tensor, *, center: bool) -> torch.Tensor:
    if center:
        source_rows = source_rows - source_rows.mean(dim=0, keepdim=True)
        target_rows = target_rows - target_rows.mean(dim=0, keepdim=True)
    cov = source_rows.double().T @ target_rows.double()
    u, _, v_h = torch.linalg.svd(cov, full_matrices=False)
    return (u @ v_h).float()


def _compute_procrustes_map_from_cov(cov: torch.Tensor) -> torch.Tensor:
    u, _, v_h = torch.linalg.svd(cov.double(), full_matrices=False)
    return (u @ v_h).float()


def _transport_weight(delta_weight: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor, *, key: str) -> torch.Tensor:
    if key == "proj" or key.endswith(".proj"):
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
) -> dict[tuple[str, str, tuple[int, int]], torch.Tensor]:
    grouped_transforms: dict[tuple[str, str, tuple[int, int]], torch.Tensor] = {}
    items = _iter_with_progress(
        grouped_covariances.items(),
        total=len(grouped_covariances),
        desc=f"{method_name}.prepare: compute shared transforms",
        enabled=show_progress,
    )
    for key, cov in items:
        grouped_transforms[key] = _compute_procrustes_map_from_cov(cov)
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
    assigned_keys: int
    shared_transform_count: int
    shared_group_count: int


@dataclass(frozen=True)
class _ApplyDiagnostics:
    transformed_weight: int
    transformed_bias: int
    zero_passthrough: int
    missing_transform: int
    transport_failures: int
    wrong_shape: int
    wrong_shape_examples: tuple[str, ...]
    skipped_not_in_target_visual: int


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
) -> tuple[dict[str, _LayerTransform], _PrecomputeDiagnostics]:
    transforms_by_key: dict[str, _LayerTransform] = {}
    t_out_cache: dict[str, torch.Tensor] = {}
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
            cov = store.get_covariance(center=center_acts)
            if cov is None:
                return None
            if (int(cov.shape[0]), int(cov.shape[1])) != expected_shape:
                return None
            return _compute_procrustes_map_from_cov(cov)

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
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            continue

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            continue

        module_name = param_to_module.get(key, key.rsplit(".", 1)[0] if "." in key else "")
        if key == "proj":
            in_key = "ln_post.out"
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
                continue
            transforms_by_key[key] = _LayerTransform(kind="weight")
            continue

        if delta_source.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[:-len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    continue

            # Robustness fallback: covers uncommon ordering/edge cases where
            # the bias has no directly available weight transform yet.
            cached_t_out = t_out_cache.get(out_key)
            if cached_t_out is not None:
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=cached_t_out)
                continue

            target_ref = target_visual_base[key]
            expected_out = (int(delta_source.shape[0]), int(target_ref.shape[0]))
            t_out = _transform_for(out_module, side="out", expected_shape=expected_out, fallback_key=out_key)
            if t_out is not None:
                t_out_cache[out_key] = t_out
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
                continue
            transforms_by_key[key] = _LayerTransform(kind="bias")
            continue

        transforms_by_key[key] = _LayerTransform(kind="unsupported")

    diagnostics = _PrecomputeDiagnostics(
        assigned_keys=len(transforms_by_key),
        shared_transform_count=(len(grouped_transforms) if transform_granularity != "param" else 0),
        shared_group_count=(len(grouped_covariances) if transform_granularity != "param" else 0),
    )
    return transforms_by_key, diagnostics


def _apply_transforms_to_visual_delta(
    *,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    transforms_by_key: Mapping[str, _LayerTransform],
    show_progress: bool,
    method_name: str,
) -> tuple[TensorDict, _ApplyDiagnostics]:
    aligned: TensorDict = {}
    transformed_weight = 0
    transformed_bias = 0
    zero_passthrough = 0
    missing_transform = 0
    transport_failures = 0
    wrong_shape = 0
    wrong_shape_examples: list[str] = []
    skipped_not_in_target_visual = 0

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.apply: transport params",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            skipped_not_in_target_visual += 1
            continue

        target_ref = target_visual_base[key]
        transported = torch.zeros_like(target_ref, dtype=torch.float32, device="cpu")

        transform = transforms_by_key.get(key)
        applied = False
        if transform is not None:
            if transform.kind == "zero":
                zero_passthrough += 1
                applied = True
            if transform.kind == "weight" and delta_source.ndim == 2 and transform.t_in is not None and transform.t_out is not None:
                try:
                    transported = _transport_weight(delta_source.float().cpu(), transform.t_in, transform.t_out, key=key)
                    transformed_weight += 1
                    applied = True
                except RuntimeError as exc:
                    logger.warning("Theseus transport failed for %s: %s", key, exc)
                    transport_failures += 1
            elif transform.kind == "bias" and delta_source.ndim == 1 and transform.t_out is not None:
                try:
                    transported = _transport_bias(delta_source.float().cpu(), transform.t_out)
                    transformed_bias += 1
                    applied = True
                except ValueError as exc:
                    logger.warning("Theseus vector transport failed for %s: %s", key, exc)
                    transport_failures += 1

        if not applied and key in target_visual_base:
            missing_transform += 1

        if transported.shape != target_ref.shape:
            logger.warning(
                "Theseus produced wrong shape for %s: got %s expected %s. Zeroing.",
                key,
                tuple(transported.shape),
                tuple(target_ref.shape),
            )
            wrong_shape += 1
            if len(wrong_shape_examples) < 5:
                wrong_shape_examples.append(key)
            transported = torch.zeros_like(target_ref, dtype=torch.float32, device="cpu")

        aligned[key] = transported.to(dtype=target_ref.dtype, device=target_ref.device)

    diagnostics = _ApplyDiagnostics(
        transformed_weight=transformed_weight,
        transformed_bias=transformed_bias,
        zero_passthrough=zero_passthrough,
        missing_transform=missing_transform,
        transport_failures=transport_failures,
        wrong_shape=wrong_shape,
        wrong_shape_examples=tuple(wrong_shape_examples),
        skipped_not_in_target_visual=skipped_not_in_target_visual,
    )
    return aligned, diagnostics


@dataclass(frozen=True)
class TheseusRebase:
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
        **kwargs,
    ) -> dict[str, Any]:
        split_qkv = kwargs.pop("split_qkv", None)
        if split_qkv is not None:
            patch_qkv = bool(split_qkv)
        transform_granularity = str(kwargs.pop("transform_granularity", "param")).strip().lower()
        if transform_granularity not in {"param", "module_type", "block", "global"}:
            raise ValueError("transform_granularity must be one of: param, module_type, block, global")
        del kwargs
        #Config fallbacks num_batches -> n_batches
        if n_batches is None:
            n_batches = num_batches
        log_prefix = f"[{self.name}]"

        if verbose:
            print(
                f"{log_prefix} prepare: start "
                f"(seq_align={seq_align}, center_acts={bool(center_acts)}, n_batches={n_batches}, "
                f"seed={int(seed)}, transform_granularity={transform_granularity})"
            )

        patched_source = 0
        patched_target = 0
        if patch_qkv:
            if verbose:
                print(f"{log_prefix} prepare: patching fused qkv blocks if needed")
            patched_source = _split_fused_qkv_if_needed(source_model)
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
        precompute_diag = _PrecomputeDiagnostics(assigned_keys=0, shared_transform_count=0, shared_group_count=0)
        split_fused_qkv = bool(patch_qkv and (patched_source > 0 or patched_target > 0))
        unpatched_source = 0
        unpatched_target = 0
        try:
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
            )
            if verbose:
                print(f"{log_prefix} prepare: collected activation entries = {len(activation_registry)}")

            if target_base is not None and delta is not None:
                if verbose:
                    print(f"{log_prefix} prepare: precomputing per-layer transforms")
                visual_key_map = _visual_delta_keys(delta)
                target_visual_base = _visual_state_dict(target_base)
                visual_delta = {
                    stripped_key: delta[original_key]
                    for stripped_key, original_key in visual_key_map.items()
                    if stripped_key in target_visual_base
                }

                if split_fused_qkv:
                    target_visual_base = _split_fused_qkv_state(target_visual_base)
                    visual_delta = _split_fused_qkv_state(visual_delta)

                transforms_by_key, precompute_diag = _precompute_transforms(
                    target_model=target_model,
                    target_visual_base=target_visual_base,
                    visual_delta=visual_delta,
                    activation_registry=activation_registry,
                    center_acts=bool(center_acts),
                    transform_granularity=transform_granularity,
                    show_progress=bool(show_progress),
                    method_name=self.name,
                )
                if verbose:
                    if transform_granularity == "param":
                        print(f"{log_prefix} prepare: computed transforms = {len(transforms_by_key)}")
                    else:
                        print(
                            f"{log_prefix} prepare: computed transforms = {len(transforms_by_key)} "
                            f"(shared={precompute_diag.shared_transform_count}, groups={precompute_diag.shared_group_count})"
                        )
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
            "precompute_diagnostics": {
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
        **kwargs,
    ) -> TensorDict:
        del kwargs
        log_prefix = f"[{self.name}]"

        if verbose:
            print(f"{log_prefix} apply: start")

        transforms_by_key = prepared.get("transforms_by_key", None)
        if transforms_by_key is None:
            raise ValueError("Theseus prepared payload is missing 'transforms_by_key'.")

        visual_key_map = _visual_delta_keys(delta)
        target_visual_base = _visual_state_dict(target_base)

        visual_delta = {
            stripped_key: delta[original_key]
            for stripped_key, original_key in visual_key_map.items()
            if stripped_key in target_visual_base
        }

        split_fused_qkv = bool(prepared.get("split_fused_qkv", False))
        if split_fused_qkv:
            target_visual_base_work = _split_fused_qkv_state(target_visual_base)
            visual_delta_work = _split_fused_qkv_state(visual_delta)
        else:
            target_visual_base_work = target_visual_base
            visual_delta_work = visual_delta

        if strict and not visual_delta_work:
            raise ValueError("Theseus did not find any visual delta keys to transport.")

        aligned_visual, apply_diag = _apply_transforms_to_visual_delta(
            target_visual_base=target_visual_base_work,
            visual_delta=visual_delta_work,
            transforms_by_key=transforms_by_key,
            show_progress=bool(show_progress),
            method_name=self.name,
        )

        if split_fused_qkv:
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
            missing = sorted(set(delta.keys()) - set(out.keys()))
            if missing:
                raise KeyError(f"Theseus did not transport all delta keys. Example: {missing[:10]}")

        if verbose:
            if apply_diag.wrong_shape > 0:
                print(
                    f"{log_prefix} apply: warnings wrong_shape={apply_diag.wrong_shape} "
                    f"examples={list(apply_diag.wrong_shape_examples)}"
                )
            print(
                f"{log_prefix} apply: diagnostics "
                f"weights={apply_diag.transformed_weight} biases={apply_diag.transformed_bias} "
                f"zero={apply_diag.zero_passthrough} "
                f"missing={apply_diag.missing_transform} failures={apply_diag.transport_failures}"
            )
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
            if source_dataloader is None or target_dataloader is None:
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
        )


register(TheseusRebase())
