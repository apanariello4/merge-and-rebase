"""Pure helpers of eval/diagnostics/anchors_kernel_agreement.py."""

from __future__ import annotations

import torch

from merge_and_rebase.eval.diagnostics.anchors_kernel_agreement import centred_kernel, cka, knn_sets, mutual_knn


def test_cka_and_knn_invariant_to_rotation_and_scale():
    g = torch.Generator().manual_seed(0)
    p = torch.randn(40, 6, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    p2 = 3.0 * p @ q + 1.0  # rotation, scale, shift: same centred kernel up to scale
    assert abs(cka(centred_kernel(p), centred_kernel(p2)) - 1) < 1e-12
    assert mutual_knn(knn_sets(p, 5), knn_sets(p2, 5)) == 1.0


def test_unrelated_representations_have_low_agreement():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(200, 8, generator=g, dtype=torch.float64)
    b = torch.randn(200, 8, generator=g, dtype=torch.float64)
    assert cka(centred_kernel(a), centred_kernel(b)) < 0.3
    assert mutual_knn(knn_sets(a, 10), knn_sets(b, 10)) < 0.2
