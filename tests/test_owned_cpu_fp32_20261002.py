"""P5.14/B4: base snapshots must not alias a CPU fp32 model."""

from __future__ import annotations

import torch

from merge_and_rebase.eval.vision_rebase.pipeline import _owned_cpu_fp32
from merge_and_rebase.merge.runtime import to_cpu_fp32


def test_to_cpu_fp32_aliases_but_owned_snapshot_does_not():
    model = torch.nn.Linear(3, 2)
    raw = dict(model.state_dict())
    aliased = to_cpu_fp32(raw)
    owned = _owned_cpu_fp32(raw)
    before = {k: v.clone() for k, v in owned.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert any(not torch.equal(aliased[k], before[k]) for k in before), "premise: to_cpu_fp32 returns views on CPU"
    assert all(torch.equal(owned[k], before[k]) for k in before)
    assert all(v.dtype == torch.float32 for v in owned.values())
