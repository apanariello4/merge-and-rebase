import torch

from merge_and_rebase.eval.diagnostics.anchors_precision import agreement, model_id


def _kernel(t=40, d=12, seed=0):
    p = torch.randn(t, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return p @ p.T


def test_identical_and_rescaled_kernels_agree_fully():
    k = _kernel()
    assert abs(agreement(k, k, factor=1.5)["agree_energy"] - 1.0) < 1e-12
    assert abs(agreement(k, 7.0 * k, factor=1.5)["agree_energy"] - 1.0) < 1e-12  # trace-normalised


def test_disagreeing_direction_is_excluded():
    u = torch.linalg.qr(torch.randn(10, 10, generator=torch.Generator().manual_seed(1), dtype=torch.float64))[0]
    lam = torch.tensor([5.0, 4, 3, 2, 1, 1, 1, 1, 1, 1], dtype=torch.float64)
    lam_b = lam.clone()
    lam_b[0] = 0.5  # one direction off by 10x, the rest equal up to the trace normalisation
    a, b = u @ torch.diag(lam) @ u.T, u @ torch.diag(lam_b) @ u.T
    res = agreement(a, b, factor=1.5)
    ea, eb = lam / lam.sum(), lam_b / lam_b.sum()
    ok = ((ea / eb) <= 1.5) & ((ea / eb) >= 1 / 1.5)
    mean = (ea + eb) / 2
    assert abs(res["agree_energy"] - float(mean[ok].sum() / mean.sum())) < 1e-10
    assert res["agree_energy"] < 1.0


def test_model_id_matches_vision_rebase():
    assert model_id({"model": "ViT-B-16", "pretrained": "datacomp_xl_s13b_b90k"}) == "ViT-B-16__datacomp_xl_s13b_b90k"
