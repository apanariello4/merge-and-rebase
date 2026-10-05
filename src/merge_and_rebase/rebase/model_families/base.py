from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelFamilyMetadata:
    family: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    is_moe: bool = False


@runtime_checkable
class ModelFamilyAdapter(Protocol):
    name: str

    def metadata(self, model: nn.Module) -> ModelFamilyMetadata:
        ...

    def transport_scope(self, model: nn.Module) -> nn.Module:
        """Return the submodule owning backbone body params (e.g. model.model)."""
        ...

    def transportable_keys(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> set[str]:
        """Return keys in state_dict that should be transported."""
        ...

    def param_to_module(
        self, model: nn.Module
    ) -> dict[str, str]:
        """Map each transportable param key to its parent module name."""
        ...

    def iter_blocks(self, model: nn.Module) -> Iterator[nn.Module]:
        """Yield each transformer block in order."""
        ...

    def block_count(self, model: nn.Module) -> int:
        """Number of hidden layers / transformer blocks."""
        ...

    def extract_calibration_batch(
        self, batch: Any
    ) -> dict[str, torch.Tensor]:
        """Extract input_ids, attention_mask, labels from a batch."""
        ...

    def excluded_keys(self) -> set[str]:
        """Keys that should never be transported (embeddings, lm_head, etc.)."""
        ...

    # ---- decoder layout (single source of truth; see ``accessors`` for duck-typed fallbacks) ----

    def layers(self, model: nn.Module) -> nn.ModuleList:
        """The block list (the live ``ModuleList``, not a copy)."""
        ...

    def set_layers(self, model: nn.Module, blocks: Any) -> None:
        """Install ``blocks`` and keep depth, ``layer_idx`` and ``layer_types`` consistent."""
        ...

    def final_norm(self, model: nn.Module) -> nn.Module:
        ...

    def block_components(self, block: nn.Module) -> dict[str, nn.Module]:
        """Correctable sub-modules of one block, keyed by canonical component name."""
        ...

    def residual_writer(self, block: nn.Module) -> nn.Module:
        """The MLP projection that writes into the residual stream (``mlp.down_proj``)."""
        ...

    def attn_output(self, block: nn.Module) -> nn.Module:
        """The attention projection that writes into the residual stream (``self_attn.o_proj``)."""
        ...

    def param_key(self, position: int, suffix: str) -> str:
        """State-dict key of ``suffix`` (e.g. ``mlp.down_proj.weight``) in block ``position``."""
        ...

    def calibration_forward(self, model: nn.Module, batch: Any, device: Any) -> Any:
        """Backbone-only forward (no labels, ``use_cache=False``); returns the backbone output."""
        ...

    def content_mask(self, batch: Any) -> torch.Tensor:
        """Bool ``[B, T]`` mask of content tokens, from ``attention_mask`` only (never from ids)."""
        ...
