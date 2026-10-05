"""What a family adapter (vision / LLM) hands to the family-agnostic runner, per task."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig

Tensor = torch.Tensor


@dataclass
class DiagnosticInputs:
    """Everything needed to reproduce one task's Ariadne fit of a reference run and to measure it.

    Models are live modules on ``device`` (or movable there); ``target_model`` must hold exactly
    ``target_base_state`` (decoders: after ``materialize_missing_projection_biases``). ``source_loader`` /
    ``target_loader`` are the calibration loaders the reference run passed to ``AriadneRebase.prepare``.
    ``generic_target_batches`` are off-task target batches for N6 (already in the model's input format, with row
    masks attached for decoders). ``token_index(batch, n_rows)`` gives the within-sequence index of each content row
    the target capture yields for ``batch`` (0 = CLS / first content token). ``component_key(j)`` and
    ``source_component_key(i)`` return the prefixed weight key of the fitted projection at a target / source block.
    ``reference_task_vector`` is the run's saved unit-strength task vector (``None`` when only norms can be
    compared); ``reference_rows`` the run's per-position summary rows.
    """

    family: str
    task: str
    device: Any
    config: DirectResidualConfig
    pairing: DiscreteLayerPairing
    source_base_model: torch.nn.Module
    source_ft_model: torch.nn.Module
    target_model: torch.nn.Module
    target_base_state: dict[str, Tensor]
    source_loader: Any
    target_loader: Any
    generic_target_batches: Sequence[Any]
    token_index: Callable[[Any, int], Tensor]
    component_key: Callable[[int], str]
    source_component_key: Callable[[int], str]
    family_adapter: Any = None
    alpha_star: float | None = None
    reference_task_vector: Mapping[str, Tensor] | None = None
    reference_task_vector_tolerance: str = "exact"
    reference_rows: Sequence[Mapping[str, Any]] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
