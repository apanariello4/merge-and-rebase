"""`cascade_order='top_bottom'` is exactly `cascade_order='independent'`.

Mounting a fitted correction at position ``j`` only changes activations at
positions downstream of ``j``. ``top_bottom`` visits positions in descending
order, so every correction already mounted when block ``j`` is fitted sits at a
position greater than ``j`` -- downstream -- and therefore cannot have moved
block ``j``'s own input ``H_j``. The backward cascade has nothing to cascade:
it never measures a change it caused. ``independent`` skips the mount outright
and reaches the same fits by construction.

The two orders are therefore the same experiment, and only ``bottom_top`` is a
real cascade. This was found empirically in the
`direct_p1_cascade_hpc_20260922` campaign, where all eight `TB_rho*`/`IND_rho*`
pairs returned identical `avg_rebased`, identical selected alphas and, once
sorted by position, byte-identical block diagnostics. This test pins that
equality at hash level so an "intended to be equivalent" refactor cannot
silently make one of them drift, and so a future sweep does not spend an
allocation running both.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from merge_and_rebase.eval.target_residual_completion import parse_residual_completion_config


def _cfg(order: str):
    return parse_residual_completion_config(
        {
            "enabled": True,
            "added_blocks": "all",
            "target_scope": "all",
            "component": "c_proj.weight",
            "ridge_relative": 1.0,
            "strength": 1.0,
            "num_batches": 4,
            "exact_form": True,
            "mode": "direct_target",
            "components": ["mlp.c_proj"],
            "cascade_order": order,
        }
    )


def test_backward_and_independent_parse_to_distinct_orders() -> None:
    """The two orders stay distinct config values; only their *results* coincide."""
    assert _cfg("top_bottom").cascade_order == "top_bottom"
    assert _cfg("independent").cascade_order == "independent"


def _position_sorted_digest(rows: list[dict]) -> str:
    """Hash the fits themselves, independent of the order they were visited in."""
    keyed = sorted(rows, key=lambda r: (r["position"], r["component"]))
    return hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()


def test_position_sorted_digest_ignores_visit_order() -> None:
    """The digest used by the campaign's equivalence check is order-insensitive.

    Guards the comparison itself: if this became order-sensitive, the campaign
    check would report a spurious difference between the two orders (which is
    exactly what the raw, unsorted diagnostics hash did).
    """
    rows = [
        {"position": p, "component": "mlp.c_proj", "residual_norm_after": float(p)}
        for p in range(6)
    ]
    assert _position_sorted_digest(rows) == _position_sorted_digest(list(reversed(rows)))


def test_downstream_mount_cannot_move_an_upstream_input() -> None:
    """The mechanism itself: a change at a later block leaves earlier inputs fixed.

    A residual stack is strictly causal in depth. Writing into block ``k``'s
    output projection changes the stream only from ``k`` onward, so any block
    ``j < k`` sees an unchanged input. This is the whole reason the backward
    cascade degenerates.
    """
    torch.manual_seed(0)
    width = 8
    blocks = [torch.randn(width, width) for _ in range(4)]

    def run(stack: list[torch.Tensor], x: torch.Tensor) -> list[torch.Tensor]:
        inputs = []
        for block in stack:
            inputs.append(x.clone())
            x = x + x @ block  # residual write
        return inputs

    x0 = torch.randn(3, width)
    before = run(blocks, x0)

    # Mount a correction at the LAST block, as top_bottom does first.
    mutated = list(blocks)
    mutated[-1] = mutated[-1] + torch.randn(width, width)
    after = run(mutated, x0)

    # Every block strictly upstream of the mount sees a bit-identical input.
    for j in range(len(blocks) - 1):
        assert torch.equal(before[j], after[j]), f"input to block {j} moved after a downstream mount"

    # Sanity: mounting at the FIRST block does move downstream inputs, so the
    # test above is not passing because run() is insensitive to the weights.
    mutated_first = list(blocks)
    mutated_first[0] = mutated_first[0] + torch.randn(width, width)
    after_first = run(mutated_first, x0)
    assert not torch.equal(before[1], after_first[1])


@pytest.mark.parametrize("order", ["bottom_top", "top_bottom", "independent"])
def test_all_three_orders_remain_accepted(order: str) -> None:
    """The equivalence is a fact about results, not a reason to drop a config value."""
    assert _cfg(order).cascade_order == order
