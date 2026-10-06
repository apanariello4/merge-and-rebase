from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tqdm import tqdm

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import DirectionMerge, get_method_params
from .functional import merge_functional


@dataclass(frozen=True)
class DCMerge(DirectionMerge):
    """Merge dense task deltas through whitened low-rank coordinate covers.

    The method truncates and optionally smooths each matrix spectrum, merges
    their core matrices with ``cover_merge_method``, and maps the result back
    through the coordinate cover. LoRA checkpoints must be materialized into
    dense deltas before this method is called.
    """

    name: str = "dc_merge"

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
        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for key in tqdm(keys, desc="Processing keys"):
            ref = base[key]
            matrices = [delta[key] for delta in deltas]
            direction[key] = merge_functional(
                "dc_merge",
                matrices=matrices,
                weights=weights,
                method_params=method_params,
            ).to(dtype=ref.dtype, device=ref.device)

        return base, direction


register(DCMerge())
