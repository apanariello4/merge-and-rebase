from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch

from ..base import TensorDict
from ..task_vectors import TaskVector


def default_weights(n: int, weights: Sequence[float] | None) -> torch.Tensor:
    if weights is None:
        return torch.ones(n, dtype=torch.float32)
    if len(weights) != n:
        raise ValueError("weights length must match tuned checkpoints")
    return torch.tensor([float(w) for w in weights], dtype=torch.float32)


def resolve_merge_weights(n: int, weights: Sequence[float] | None) -> list[float]:
    return [float(weight) for weight in default_weights(int(n), weights).tolist()]


def axpy_state_dict(base: TensorDict, delta: TensorDict, alpha: float) -> TensorDict:
    out: TensorDict = dict(base)
    for k in TaskVector.common_keys(base, [delta]):
        b = base[k]
        d = delta[k].to(dtype=b.dtype, device=b.device)
        out[k] = b + float(alpha) * d
    return out


def get_method_params(kwargs: dict) -> dict:
    method_params = kwargs.get("method_params", {})
    if method_params is None:
        method_params = {}
    if not isinstance(method_params, dict):
        raise ValueError("method_params must be a dict.")
    return method_params


class DirectionMerge:
    """Shared ``apply`` / ``merge`` of the merge methods whose ``prepare`` returns ``(base, merged direction)``:
    the merged model is ``base + alpha * direction``."""

    #: ``merge`` passes its extra ``**kwargs`` (``method_params``, ...) on to ``prepare``.
    forward_merge_kwargs: ClassVar[bool] = True

    def apply(self, prepared: tuple[TensorDict, TensorDict], *, alpha: float, **kwargs: Any) -> TensorDict:
        base, direction = prepared
        return axpy_state_dict(base, direction, alpha=float(alpha))

    def merge(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        strict: bool = False,
        **kwargs: Any,
    ) -> TensorDict:
        extra = kwargs if self.forward_merge_kwargs else {}
        prepared = self.prepare(base=base, tuned=tuned, weights=weights, strict=strict, **extra)  # type: ignore[attr-defined]
        return self.apply(prepared, alpha=float(alpha))
