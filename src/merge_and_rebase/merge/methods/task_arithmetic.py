from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import DirectionMerge, default_weights


@dataclass(frozen=True)
class TaskArithmeticMerge(DirectionMerge):
    """Add weighted task vectors to the shared base checkpoint.

    The prepared direction is ``sum_i w_i * (tuned_i - base)`` and ``apply``
    returns ``base + alpha * direction``. Task Arithmetic has no
    method-specific parameters.
    """

    forward_merge_kwargs: ClassVar[bool] = False  # merge() never passed its **kwargs to prepare()

    name: str = "task_arithmetic"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        w = default_weights(len(tuned), weights)

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]

        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for k in keys:
            acc = torch.zeros_like(base[k])
            for wi, d in zip(w, deltas, strict=True):
                acc = acc + float(wi) * d[k].to(dtype=acc.dtype, device=acc.device)
            direction[k] = acc

        return base, direction


register(TaskArithmeticMerge())
