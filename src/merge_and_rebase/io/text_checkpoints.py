from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..io.peft_helpers import is_peft_adapter_dir_ckpt, load_peft_adapter_dir_components
from ..merge import runtime as _merge_utils
from ..merge.methods._common import default_weights
from ..merge.task_vectors import default_key_filter
from ..models.text_lm import TextBuildConfig

_is_peft_checkpoint = _merge_utils.is_peft_checkpoint
_extract_peft_components = _merge_utils.extract_peft_components
_ensure_peft_cfg_map = _merge_utils.ensure_peft_cfg_map
_get_peft_cfg = _merge_utils.get_peft_cfg
_to_cpu_fp32 = _merge_utils.to_cpu_fp32

_HEAD_FORMAT_MARKERS = ("heads.pt",)


class HeadCheckpointError(ValueError):
    """Raised when a head-only checkpoint is used in a body-loading context."""


def _reject_head_payload(obj: dict[str, Any], ref: str) -> None:
    if isinstance(obj, dict):
        fmt = obj.get("format", "")
        if str(fmt).strip().lower() == "head":
            raise HeadCheckpointError(
                f"Checkpoint '{ref}' has format='head' and contains only head parameters. "
                "Use task_heads / heads.pt for head loading, not body checkpoint paths."
            )
        if "head" in obj and isinstance(obj["head"], dict):
            raise HeadCheckpointError(
                f"Checkpoint '{ref}' contains an embedded head payload (format='head'). "
                "This is an eval-only artifact and cannot be used as a body checkpoint."
            )


@dataclass(frozen=True)
class _LoRAFactors:
    a: torch.Tensor
    b: torch.Tensor
    scale: float


def _lookup_layer_pattern(
    pattern: dict[str, Any] | None,
    *,
    layer_key: str,
    default: Any,
) -> Any:
    if not pattern:
        return default
    candidates = [layer_key]
    if layer_key.startswith("base_model.model."):
        tail = layer_key[len("base_model.model.") :]
        candidates.append(tail)
        candidates.append(f"model.{tail}")
    elif layer_key.startswith("model."):
        tail = layer_key[len("model.") :]
        candidates.append(tail)
        candidates.append(f"base_model.model.{tail}")
    else:
        candidates.append(f"base_model.model.{layer_key}")
        candidates.append(f"model.{layer_key}")
    for k in candidates:
        if k in pattern:
            return pattern[k]
    return default


def _lora_scaling_for_layer(
    *,
    layer_key: str,
    a: torch.Tensor,
    peft_cfg: dict[str, Any],
) -> float:
    rank_pattern = peft_cfg.get("rank_pattern", {}) if isinstance(peft_cfg.get("rank_pattern", {}), dict) else {}
    alpha_pattern = peft_cfg.get("alpha_pattern", {}) if isinstance(peft_cfg.get("alpha_pattern", {}), dict) else {}
    default_alpha = float(peft_cfg.get("lora_alpha", max(1, int(a.shape[0]))))
    use_rslora = bool(peft_cfg.get("use_rslora", False))

    r_eff = int(a.shape[0])
    r_cfg = int(_lookup_layer_pattern(rank_pattern, layer_key=layer_key, default=r_eff))
    if r_cfg <= 0:
        r_cfg = r_eff
    alpha = float(_lookup_layer_pattern(alpha_pattern, layer_key=layer_key, default=default_alpha))
    denom = (r_cfg**0.5) if use_rslora else float(r_cfg)
    return float(alpha / max(1e-12, denom))


def _strip_known_key_prefixes(key: str) -> list[str]:
    out: list[str] = [key]
    queue: list[str] = [key]
    seen: set[str] = set()
    prefixes = ("base_model.model.", "model.", "module.", "clip_model.model.", "clip_model.")
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        for p in prefixes:
            if cur.startswith(p):
                nxt = cur[len(p) :]
                if nxt and nxt not in seen:
                    out.append(nxt)
                    queue.append(nxt)
    uniq: list[str] = []
    seen2: set[str] = set()
    for k in out:
        if k not in seen2:
            uniq.append(k)
            seen2.add(k)
    return uniq


def _aligned_key_from_candidates(
    *,
    candidates: list[str],
    shape: tuple[int, ...],
    base_shapes: dict[str, tuple[int, ...]],
) -> str | None:
    queue = list(candidates)
    seen: set[str] = set()
    while queue:
        k = queue.pop(0)
        if k in seen:
            continue
        seen.add(k)
        if base_shapes.get(k, None) == shape:
            return k

        for p in ("model.", "module.", "clip_model.model.", "clip_model."):
            if k.startswith(p):
                queue.append(k[len(p) :])
        if k.startswith("visual.transformer."):
            queue.append("transformer." + k[len("visual.transformer.") :])
        if k.startswith("transformer."):
            queue.append("visual.transformer." + k[len("transformer.") :])
    return None


def _base_key_candidates_from_lora_prefix(prefix: str) -> list[str]:
    cands: list[str] = []
    for p in _strip_known_key_prefixes(prefix):
        if p.endswith(".base_layer"):
            cands.append(p[: -len(".base_layer")] + ".weight")
        cands.append(f"{p}.weight")
    uniq: list[str] = []
    seen: set[str] = set()
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _base_key_candidates_from_modules_to_save_key(key: str) -> list[str]:
    marker = ".modules_to_save."
    if marker not in key:
        return _strip_known_key_prefixes(key)
    head, rest = key.split(marker, 1)
    parts = rest.split(".")
    if len(parts) >= 2:
        tail = ".".join(parts[1:])
        canonical = f"{head}.{tail}"
    else:
        canonical = head
    return _strip_known_key_prefixes(canonical)


class _LoRAAlignedAdapterView(Mapping[str, torch.Tensor]):
    def __init__(
        self,
        *,
        adapter_ref: str,
        base_sd: Mapping[str, torch.Tensor],
        lora_by_key: dict[str, _LoRAFactors],
        direct_overrides: dict[str, torch.Tensor],
    ) -> None:
        self._adapter_ref = str(adapter_ref)
        self._base_sd = base_sd
        self._lora_by_key = dict(lora_by_key)
        self._direct = {k: v.detach().cpu() for k, v in direct_overrides.items()}
        keys = set(self._direct.keys()).union(self._lora_by_key.keys())
        self._keys = tuple(sorted(keys))

    @property
    def adapter_ref(self) -> str:
        return self._adapter_ref

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str) -> torch.Tensor:
        k = str(key)
        direct = self._direct.get(k, None)
        if direct is not None:
            base = self._base_sd.get(k, None)
            if isinstance(base, torch.Tensor):
                return direct.to(dtype=base.dtype, device="cpu")
            return direct

        factors = self._lora_by_key.get(k, None)
        if factors is None:
            raise KeyError(k)
        base = self._base_sd.get(k, None)
        if not isinstance(base, torch.Tensor):
            raise KeyError(k)

        base_cpu = base.detach().to(device="cpu")
        work_dtype = (
            torch.float32 if base_cpu.dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float64
        )
        a = factors.a.detach().to(device="cpu", dtype=work_dtype)
        b = factors.b.detach().to(device="cpu", dtype=work_dtype)
        delta = torch.matmul(b, a).mul_(float(factors.scale))
        tuned = base_cpu.to(dtype=work_dtype).add_(delta)
        return tuned.to(dtype=base_cpu.dtype)

    def __repr__(self):
        return f"_LoRAAlignedAdapterView(adapter_ref={self._adapter_ref})"

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for k in self._keys:
            yield k, self[k]


def resolve_checkpoint_reference(ckpt_ref: str) -> str:
    resolved_ref = str(ckpt_ref)
    p = Path(resolved_ref)
    if p.exists() and p.is_file():
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
            _reject_head_payload(obj, resolved_ref)
            if is_peft_adapter_dir_ckpt(obj):
                adapter_dir = str(obj["peft_adapter_dir"])
                print(f"Resolved PEFT adapter checkpoint metadata {resolved_ref} -> {adapter_dir}")
                return adapter_dir
        except HeadCheckpointError:
            raise
        except Exception:
            return resolved_ref
    return resolved_ref


def is_adapter_reference(ref: str) -> bool:
    p = Path(ref)
    if p.exists():
        if p.is_file():
            return False
        if p.is_dir():
            return (p / "adapter_config.json").exists()
    if "/" in ref and not ref.endswith((".pt", ".bin", ".safetensors", ".ckpt", ".pth")):
        return True
    return False


def load_peft_components_from_adapter_ref(adapter_ref: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    try:
        from peft import PeftConfig
        from peft.utils import load_peft_weights
    except Exception as e:
        raise ImportError("Loading PEFT adapters requires `peft`.") from e

    peft_cfg_obj = PeftConfig.from_pretrained(adapter_ref)
    cfg_dict = peft_cfg_obj.to_dict() if hasattr(peft_cfg_obj, "to_dict") else dict(peft_cfg_obj.__dict__)
    peft_state = load_peft_weights(adapter_ref, device="cpu")
    if not isinstance(peft_state, dict):
        raise ValueError(f"Invalid PEFT adapter '{adapter_ref}': adapter weights are not a dict.")
    state = {str(k): v.detach().cpu() for k, v in peft_state.items() if torch.is_tensor(v)}
    if not state:
        raise ValueError(f"Invalid PEFT adapter '{adapter_ref}': adapter state has no tensors.")
    return state, {"default": cfg_dict}


def load_peft_components_for_subspace(
    *,
    ckpt_ref: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    resolved_ref = resolve_checkpoint_reference(str(ckpt_ref))
    p = Path(resolved_ref)
    if p.exists() and p.is_file():
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        if is_peft_adapter_dir_ckpt(obj):
            adapter_dir = str(obj["peft_adapter_dir"])
            state, cfg_map = load_peft_adapter_dir_components(adapter_dir)
            return state, cfg_map, adapter_dir
        if isinstance(obj, dict) and isinstance(obj.get("peft_adapter_dir"), str):
            adapter_dir = str(obj["peft_adapter_dir"])
            state, cfg_map = load_peft_adapter_dir_components(adapter_dir)
            return state, cfg_map, adapter_dir
        if _is_peft_checkpoint(obj):
            state, cfg_map = _extract_peft_components(obj)
            return state, cfg_map, resolved_ref
        raise ValueError(f"peft_subspace requires PEFT checkpoints. Got non-PEFT checkpoint payload: {resolved_ref}")

    if is_adapter_reference(resolved_ref):
        state, cfg_map = load_peft_components_from_adapter_ref(resolved_ref)
        return state, cfg_map, resolved_ref

    raise ValueError(f"peft_subspace requires PEFT adapter references or PEFT checkpoints. Got: {ckpt_ref}")


def _build_lora_aligned_adapter_view(
    *,
    adapter_ref: str,
    base_sd: Mapping[str, torch.Tensor],
) -> _LoRAAlignedAdapterView | None:
    try:
        from peft import PeftConfig
        from peft.utils import load_peft_weights
    except Exception as e:
        raise ImportError("LoRA adapter loading requires `peft`.") from e

    peft_cfg_obj = PeftConfig.from_pretrained(adapter_ref)
    peft_cfg = peft_cfg_obj.to_dict() if hasattr(peft_cfg_obj, "to_dict") else dict(peft_cfg_obj.__dict__)
    peft_type_raw = peft_cfg.get("peft_type", "")
    if hasattr(peft_type_raw, "value"):
        peft_type = str(peft_type_raw.value)
    else:
        peft_type = str(peft_type_raw)
    peft_type = peft_type.split(".")[-1].strip().upper()
    if peft_type != "LORA":
        return None
    if bool(peft_cfg.get("use_dora", False)):
        print(f"[warn] Adapter {adapter_ref} uses DoRA; falling back to full materialization.")
        return None

    peft_state = load_peft_weights(adapter_ref, device="cpu")
    if not isinstance(peft_state, dict):
        return None
    state = {str(k): v.detach().cpu() for k, v in peft_state.items() if isinstance(v, torch.Tensor)}
    if any("lora_magnitude_vector" in k for k in state):
        print(f"[warn] Adapter {adapter_ref} has lora_magnitude_vector; falling back to full materialization.")
        return None

    base_shapes = {k: tuple(v.shape) for k, v in base_sd.items() if isinstance(v, torch.Tensor)}
    a_by_prefix: dict[str, torch.Tensor] = {}
    b_by_prefix: dict[str, torch.Tensor] = {}
    direct_overrides: dict[str, torch.Tensor] = {}

    for k, v in state.items():
        prefix: str | None = None
        if ".lora_A." in k and k.endswith(".weight"):
            prefix = k.split(".lora_A.", 1)[0]
            a_by_prefix[prefix] = v
            continue
        if k.endswith(".lora_A.weight"):
            prefix = k[: -len(".lora_A.weight")]
            a_by_prefix[prefix] = v
            continue
        if ".lora_B." in k and k.endswith(".weight"):
            prefix = k.split(".lora_B.", 1)[0]
            b_by_prefix[prefix] = v
            continue
        if k.endswith(".lora_B.weight"):
            prefix = k[: -len(".lora_B.weight")]
            b_by_prefix[prefix] = v
            continue
        if ".modules_to_save." in k:
            candidates = _base_key_candidates_from_modules_to_save_key(k)
            resolved = _aligned_key_from_candidates(
                candidates=candidates, shape=tuple(v.shape), base_shapes=base_shapes
            )
            if resolved is not None:
                direct_overrides[resolved] = v

    lora_by_key: dict[str, _LoRAFactors] = {}
    for prefix in sorted(set(a_by_prefix.keys()).intersection(b_by_prefix.keys())):
        a = a_by_prefix[prefix]
        b = b_by_prefix[prefix]
        if a.ndim != 2 or b.ndim != 2:
            continue
        shape = (int(b.shape[0]), int(a.shape[1]))
        base_candidates = _base_key_candidates_from_lora_prefix(prefix)
        base_key = _aligned_key_from_candidates(candidates=base_candidates, shape=shape, base_shapes=base_shapes)
        if base_key is None:
            continue
        scale = _lora_scaling_for_layer(layer_key=prefix, a=a, peft_cfg=peft_cfg)
        lora_by_key[base_key] = _LoRAFactors(a=a, b=b, scale=scale)

    if not lora_by_key and not direct_overrides:
        return None
    return _LoRAAlignedAdapterView(
        adapter_ref=adapter_ref,
        base_sd=base_sd,
        lora_by_key=lora_by_key,
        direct_overrides=direct_overrides,
    )


def _build_hf_model_for_materialization(
    *,
    build_cfg: TextBuildConfig,
):
    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
        )
    except Exception as e:
        raise ImportError("Hugging Face materialization requires transformers.") from e

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(build_cfg.dtype, None)
    arch = str(build_cfg.model_arch).strip().lower()
    if arch not in {"llama", "qwen", "t5", "auto"}:
        raise ValueError("model_arch must be one of: llama, qwen, t5, auto")
    kind = str(build_cfg.model_kind).strip().lower()
    common = {
        "pretrained_model_name_or_path": build_cfg.model_name_or_path,
        "trust_remote_code": bool(build_cfg.trust_remote_code),
        "torch_dtype": torch_dtype,
    }
    if kind == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            **common,
            num_labels=int(build_cfg.num_labels),
        )
    elif kind == "causal_lm":
        hf_cfg = AutoConfig.from_pretrained(
            build_cfg.model_name_or_path,
            trust_remote_code=bool(build_cfg.trust_remote_code),
        )
        is_encoder_decoder = bool(getattr(hf_cfg, "is_encoder_decoder", False))
        use_seq2seq = (arch == "t5") or (arch == "auto" and is_encoder_decoder)
        if use_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(**common)
        else:
            model = AutoModelForCausalLM.from_pretrained(**common)
    else:
        raise ValueError("model_kind must be one of: causal_lm, sequence_classification")
    return model.to(build_cfg.device)


def materialize_adapter_state_dict(
    *,
    adapter_ref: str,
    build_cfg: TextBuildConfig,
    model: Any | None = None,
) -> dict[str, torch.Tensor]:
    try:
        from peft import PeftModel
    except Exception as e:
        raise ImportError("Adapter materialization from HF requires `peft`.") from e

    owns_model = model is None
    if model is None:
        model = _build_hf_model_for_materialization(build_cfg=build_cfg)
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_ref,
        is_trainable=False,
    )
    if not hasattr(peft_model, "merge_and_unload"):
        raise RuntimeError(f"PEFT model from '{adapter_ref}' does not support merge_and_unload().")
    merged = peft_model.merge_and_unload()
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if torch.is_tensor(v)}

    if owns_model:
        del merged
        del peft_model
        del model
    if torch.cuda.is_available() and str(build_cfg.device).lower() != "cpu":
        torch.cuda.empty_cache()
    return sd


def _is_hf_dense_ref(ref: str) -> bool:
    """Check if a reference is a dense HuggingFace model (not a PEFT adapter)."""
    if ref.endswith((".pt", ".bin", ".safetensors", ".ckpt", ".pth")):
        return False
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(ref, trust_remote_code=False)
        model_type = getattr(config, "model_type", None)
        return model_type is not None
    except Exception:
        return False


def _load_dense_hf_state_dict(hf_ref: str, build_cfg: TextBuildConfig) -> dict[str, torch.Tensor]:
    """Load a dense HF model from a hub ref and return its CPU state dict."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    except Exception as e:
        raise ImportError("Loading dense HF model requires transformers.") from e

    kind = str(build_cfg.model_kind).strip().lower()
    common = {
        "pretrained_model_name_or_path": hf_ref,
        "trust_remote_code": bool(build_cfg.trust_remote_code),
        "torch_dtype": torch.float32,
    }
    if kind == "sequence_classification":
        model = AutoModelForCausalLM.from_pretrained(**common, num_labels=int(build_cfg.num_labels))
    elif kind == "causal_lm":
        hf_cfg = AutoConfig.from_pretrained(hf_ref, trust_remote_code=bool(build_cfg.trust_remote_code))
        is_encoder_decoder = bool(getattr(hf_cfg, "is_encoder_decoder", False))
        arch = str(build_cfg.model_arch).strip().lower()
        use_seq2seq = (arch == "t5") or (arch == "auto" and is_encoder_decoder)
        if use_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(**common)
        else:
            model = AutoModelForCausalLM.from_pretrained(**common)
    else:
        raise ValueError("model_kind must be one of: causal_lm, sequence_classification")

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if torch.is_tensor(v)}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sd


def load_aligned_tuned_from_ref(
    *,
    ckpt_ref: str,
    base_sd: dict[str, torch.Tensor],
    build_cfg: TextBuildConfig,
    model: Any,
    prefer_lora_view: bool = False,
    adapter_view_cache: dict[str, _LoRAAlignedAdapterView] | None = None,
) -> Mapping[str, torch.Tensor]:
    resolved_ref = resolve_checkpoint_reference(str(ckpt_ref))
    used_adapter = False

    # A local directory that isn't a raw checkpoint *file* can never be loaded
    # via torch.load (that's an IsADirectoryError waiting to happen). If it's
    # not a PEFT adapter dir either (is_adapter_reference already covers that,
    # plus HF-hub-id-shaped strings), it's presumably a plain dense HF model
    # dir -- let _is_hf_dense_ref below confirm and route it accordingly
    # instead of falling through to the raw-checkpoint-file branch.
    resolved_path = Path(resolved_ref)
    is_dense_local_dir = resolved_path.is_dir() and not is_adapter_reference(resolved_ref)

    if is_adapter_reference(resolved_ref) or is_dense_local_dir:
        if _is_hf_dense_ref(resolved_ref):
            print(f"Loading dense HF model ref: {resolved_ref}")
            sd = _load_dense_hf_state_dict(resolved_ref, build_cfg)
        else:
            if prefer_lora_view:
                if adapter_view_cache is not None and resolved_ref in adapter_view_cache:
                    view = adapter_view_cache[resolved_ref]
                    print(f"Reusing cached LoRA adapter view: {resolved_ref} ({len(view)} aligned tensors)")
                    return view
                try:
                    view = _build_lora_aligned_adapter_view(adapter_ref=resolved_ref, base_sd=base_sd)
                except Exception as exc:
                    print(
                        f"[warn] LoRA adapter fast-path failed for {resolved_ref}: {exc}. Falling back to full materialization."
                    )
                    view = None
                if view is not None:
                    if adapter_view_cache is not None:
                        adapter_view_cache[resolved_ref] = view
                    print(f"Loaded LoRA adapter view {resolved_ref}: {len(view)} aligned tensors")
                    return view

            print(f"Materializing HF/PEFT adapter into full checkpoint: {resolved_ref}")
            sd = materialize_adapter_state_dict(
                adapter_ref=resolved_ref,
                build_cfg=build_cfg,
                model=model,
            )
            used_adapter = True
    else:
        sd = load_ckpt(resolved_ref)

    try:
        aligned: Mapping[str, torch.Tensor] = align_to_base_keys(sd, base_sd)
        if not aligned:
            raise ValueError(
                f"No tensors from tuned checkpoint aligned to base keys: {resolved_ref}. "
                "Check checkpoint key prefixes and model compatibility."
            )
        print(f"Aligned tuned checkpoint {resolved_ref}: {len(aligned)} tensors")
        return aligned
    finally:
        del sd
        if used_adapter:
            miss, unexp = load_into_model(model, base_sd, strict=False)
            print(f"Restored base model after adapter materialization. missing={miss}, unexpected={unexp}")


class LazyAlignedTunedSequence(Sequence[Mapping[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        tuned_refs: list[str],
        base_sd: dict[str, torch.Tensor],
        build_cfg: TextBuildConfig,
        model: Any,
        force_fp32: bool = True,
        prefer_lora_view: bool = True,
    ) -> None:
        self._refs = [str(x) for x in tuned_refs]
        self._base_sd = base_sd
        self._build_cfg = build_cfg
        self._model = model
        self._force_fp32 = bool(force_fp32)
        self._prefer_lora_view = bool(prefer_lora_view)
        self._index_cache: dict[int, Mapping[str, torch.Tensor]] = {}
        self._adapter_view_cache: dict[str, _LoRAAlignedAdapterView] = {}

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        i = int(idx)
        if i < 0:
            i += len(self._refs)
        if i < 0 or i >= len(self._refs):
            raise IndexError(idx)

        cached = self._index_cache.get(i, None)
        if cached is not None:
            return cached

        aligned = load_aligned_tuned_from_ref(
            ckpt_ref=self._refs[i],
            base_sd=self._base_sd,
            build_cfg=self._build_cfg,
            model=self._model,
            prefer_lora_view=self._prefer_lora_view,
            adapter_view_cache=self._adapter_view_cache,
        )
        if self._force_fp32:
            out = _to_cpu_fp32(aligned)
            del aligned
            return out
        if isinstance(aligned, _LoRAAlignedAdapterView):
            self._index_cache[i] = aligned
        return aligned

    def __repr__(self):
        return f"LazyAlignedTunedSequence(refs={self._refs}, prefer_lora_view={self._prefer_lora_view})"


def load_tuned_sequence_for_preparation(
    *,
    tuned_refs: list[str],
    base_sd: dict[str, torch.Tensor],
    build_cfg: TextBuildConfig,
    model: Any,
    strict_load: bool,
    use_low_memory_prepare: bool,
) -> tuple[Sequence[Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]:
    if use_low_memory_prepare:
        print("Using low-memory lazy checkpoint loading for method preparation.")
        tuned_sds_list: Sequence[Mapping[str, torch.Tensor]] = LazyAlignedTunedSequence(
            tuned_refs=tuned_refs,
            base_sd=base_sd,
            build_cfg=build_cfg,
            model=model,
            force_fp32=False,
            prefer_lora_view=(not strict_load),
        )
        return tuned_sds_list, {k: v.detach().cpu() for k, v in base_sd.items()}

    eager_list: list[dict[str, torch.Tensor]] = []
    for ckpt_ref in tuned_refs:
        aligned = load_aligned_tuned_from_ref(
            ckpt_ref=ckpt_ref,
            base_sd=base_sd,
            build_cfg=build_cfg,
            model=model,
            prefer_lora_view=(not strict_load),
        )
        eager_list.append(_to_cpu_fp32(aligned))
        del aligned
    return eager_list, _to_cpu_fp32(base_sd)


def _resolve_merge_weights(n: int, weights: Any) -> list[float]:
    return [float(w) for w in default_weights(int(n), weights).tolist()]


def prepare_task_arithmetic_streaming(
    *,
    base_sd: dict[str, torch.Tensor],
    tuned_refs: list[str],
    weights: Any,
    strict: bool,
    build_cfg: TextBuildConfig,
    model: Any,
    prefer_lora_view: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not tuned_refs:
        raise ValueError("No tuned checkpoints provided for streaming task_arithmetic.")

    w = _resolve_merge_weights(len(tuned_refs), weights)
    active_keys: set[str] | None = None
    direction: dict[str, torch.Tensor] = {}
    expected_base_keys = {k for k, v in base_sd.items() if default_key_filter(k, v)}

    for i, ckpt_ref in enumerate(tuned_refs):
        aligned = load_aligned_tuned_from_ref(
            ckpt_ref=ckpt_ref,
            base_sd=base_sd,
            build_cfg=build_cfg,
            model=model,
            prefer_lora_view=prefer_lora_view,
        )
        current_keys: set[str] = set()
        wi = float(w[i])

        for k, t in aligned.items():
            b = base_sd.get(k, None)
            if b is None:
                continue
            if not default_key_filter(k, b):
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if t.shape != b.shape:
                continue
            current_keys.add(k)

        if active_keys is None:
            active_keys = set(current_keys)
            for k in active_keys:
                b = base_sd[k]
                t = aligned[k].to(dtype=b.dtype, device="cpu")
                direction[k] = wi * (t - b)
        else:
            shared = active_keys.intersection(current_keys)
            dropped = active_keys - shared
            for k in dropped:
                direction.pop(k, None)
            for k in shared:
                b = base_sd[k]
                d = direction[k]
                t = aligned[k].to(dtype=d.dtype, device="cpu")
                d.add_(wi * (t - b.to(dtype=d.dtype, device="cpu")))
            active_keys = shared

        print(
            f"[stream] processed {i + 1}/{len(tuned_refs)} tuned checkpoints; "
            f"active merged keys={0 if active_keys is None else len(active_keys)}"
        )
        del aligned

    if not active_keys:
        raise RuntimeError("Streaming task_arithmetic found no common mergeable keys across checkpoints.")
    if strict and active_keys != expected_base_keys:
        missing = sorted(expected_base_keys - active_keys)
        raise ValueError(
            "Strict mode: tuned checkpoints do not match base floating-point keyspace.\n"
            f"Missing keys (sample): {missing[:10]}"
        )
    return base_sd, direction
