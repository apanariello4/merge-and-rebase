from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import DirectionMerge, default_weights, get_method_params


@dataclass(frozen=True)
class WeightedAverageMerge(DirectionMerge):
    """Interpolate the base model toward a weighted average of checkpoints.

    The prepared direction is ``avg(tuned) - base`` and ``apply`` returns
    ``base + alpha * direction``. ``method_params["normalize"]`` chooses
    weight-sum normalization (``"sumw"``, default) or task-count
    normalization (``"n"``).
    """

    forward_merge_kwargs: ClassVar[bool] = False  # merge() never passed its **kwargs to prepare()

    name: str = "weighted_average"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        if len(tuned) == 0:
            raise ValueError("tuned must be non-empty")

        method_params = get_method_params(kwargs)
        normalize = str(method_params.get("normalize", "sumw"))

        w = default_weights(len(tuned), weights)
        keys = TaskVector.common_keys(base, tuned)

        if normalize == "sumw":
            denom = float(w.sum().clamp_min(1e-12).item())
        elif normalize == "n":
            denom = float(len(tuned))
        else:
            raise ValueError("normalize must be 'sumw' or 'n'")

        direction: TensorDict = {}
        for k in keys:
            b = base[k]
            acc = torch.zeros_like(b)
            for wi, t in zip(w, tuned, strict=True):
                acc = acc + float(wi) * t[k].to(dtype=acc.dtype, device=acc.device)
            avg = acc / denom
            direction[k] = avg - b

        return base, direction


register(WeightedAverageMerge())
