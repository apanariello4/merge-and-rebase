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
