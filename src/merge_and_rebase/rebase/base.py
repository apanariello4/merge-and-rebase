from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch

TensorDict = dict[str, torch.Tensor]


@runtime_checkable
class TransportMethod(Protocol):
    """
    Minimal API for transporting a task vector delta from source base to target base
    (``MethodFamily.ACTIVATION_ALIGNED`` and ``WEIGHT_SPACE`` in ``rebase.capabilities``).
    """

    name: str

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        **kwargs,
    ) -> TensorDict: ...


@runtime_checkable
class PreparedTransportMethod(Protocol):
    """
    Optional API for rebase methods that have an expensive prepare step
    (e.g. gradient computation) and a cheap apply step (e.g. masking).

    - prepare(): heavy work done once (e.g. compute gradient signs)
    - apply(): cheap masking of a delta using prepared state
    - transport(): convenience pipeline = prepare + apply
    """

    name: str

    def prepare(
        self,
        *,
        target_model: torch.nn.Module,
        target_dataloader: Any,
        device: str = "cuda",
        **kwargs,
    ) -> TensorDict:
        """
        Returns prepared state (e.g. gradient signs dict).
        """
        ...

    def apply(
        self,
        prepared: TensorDict,
        *,
        delta: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> TensorDict:
        """
        Apply the prepared state to mask/transform a delta.
        """
        ...

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        **kwargs,
    ) -> TensorDict: ...


@runtime_checkable
class FitMethod(Protocol):
    """
    A method that fits the target task vector from paired source/target activations instead of transporting the
    source delta (``MethodFamily.DIRECT_FIT``, e.g. Ariadne): ``prepare`` takes live models and calibration loaders.
    """

    name: str

    def prepare(self, **kwargs) -> Any: ...
