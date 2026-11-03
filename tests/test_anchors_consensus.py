"""frame_map='consensus' (AGENT_ADDENDUM_CONSENSUS section 2.4)."""

import hashlib

import pytest
import torch

from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, AnchorState
from merge_and_rebase.rebase.methods.anchors.config import parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.core import (
    consensus_frame,
    consensus_kernel,
    consensus_register,
    gpa,
    gpa_transport_map,
    kernel_stats_to_consensus,
    mutual_knn,
    normalised_frame,
)
from merge_and_rebase.rebase.methods.anchors.state import RegisteredModel
from merge_and_rebase.rebase.methods.anchors.universe import fit_consensus

REGISTERS = ("rot", "correct", "zca_rot")


def _rotated_pair(t=60, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(t, d, generator=g, dtype=torch.float64) @ torch.diag(torch.linspace(3.0, 0.5, d, dtype=torch.float64))
    q = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    return p, p @ q, q


@pytest.mark.parametrize("construction", ["mean", "logeuclid", "robust"])
@pytest.mark.parametrize("register", REGISTERS)
def test_exact_rotation_gives_the_gpa_transport(construction, register):
    p_a, p_b, q = _rotated_pair()
    # centred frames: the consensus registers centred anchors, GPA uses the frames as given
    p_a, p_b = p_a - p_a.mean(0), p_b - p_b.mean(0)
    k_star, _ = consensus_kernel([p_a, p_b], construction=construction)
    z, lam = consensus_frame(k_star, 8)
    a_m, _, info_a = consensus_register(p_a, z, lam, register=register, eps=1e-12)
    _, b_k, _ = consensus_register(p_b, z, lam, register=register, eps=1e-12)
    phi = gpa_transport_map(a_m, b_k, 1.0, scale=1.0)
    _, (ga, gb), _ = gpa([p_a, p_b])
    torch.testing.assert_close(phi, gpa_transport_map(ga, gb, 1.0, scale=1.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(phi, q, atol=1e-6, rtol=1e-6)
    if construction != "logeuclid":  # log-Euclid regularises the spectrum: K* is not exactly K^ of the pair
        assert info_a["map_residual"] < 1e-6


def test_single_model_consensus_is_its_own_kernel_and_rot_is_identity():
    p, _, _ = _rotated_pair(seed=3)
    k_star, _ = consensus_kernel([p])
    x, _ = normalised_frame(p)
    torch.testing.assert_close(k_star, x @ x.T, atol=1e-12, rtol=1e-12)
    z, lam = consensus_frame(k_star, 8)
    a, b, _ = consensus_register(p, z, lam, register="rot")
    torch.testing.assert_close(a @ a.T, torch.eye(8, dtype=torch.float64), atol=1e-10, rtol=1e-10)  # orthogonal
    torch.testing.assert_close(gpa_transport_map(a, b, 1.0, scale=1.0), torch.eye(8, dtype=torch.float64),
                               atol=1e-10, rtol=1e-10)
    stats = kernel_stats_to_consensus(p, k_star)
    assert abs(stats["one_minus_cka"]) < 1e-12 and stats["mutual_knn10"] == 1.0


def test_mutual_knn_is_one_for_equal_kernels_and_small_for_unrelated():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(80, 6, generator=g, dtype=torch.float64)
    b = torch.randn(80, 6, generator=g, dtype=torch.float64)
    assert mutual_knn(a @ a.T, a @ a.T) == 1.0
    assert mutual_knn(a @ a.T, b @ b.T) < 0.5


def _digest(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("register", REGISTERS)
def test_frozen_consensus_add_model_keeps_k_star_z_and_maps(tmp_path, register):
    g = torch.Generator().manual_seed(8)
    state = AnchorState(AnchorsConfig(num_anchors=40), tmp_path)
    for name, width, depth in (("A", 5, 2), ("B", 6, 4), ("C", 5, 2)):
        p = [torch.randn(40, width, generator=g) for _ in range(depth)]
        state.models[name] = RegisteredModel(
            name, {"depth": depth}, p, [torch.zeros(width, dtype=torch.float64)] * depth, [1.0] * depth
        )
    kw = {"n_grid": 4, "k": 4, "register": register}
    base = fit_consensus(state, ["A", "B"], **kw)
    root = tmp_path / "universes" / base.version
    k_before = [_digest(k) for k in torch.load(root / "k_star.pt")]
    maps_before = {m: [_digest(a) for a in base.maps[m]] for m in ("A", "B")}
    g_before = [_digest(x) for x in base.g]
    grown = fit_consensus(state, ["A", "B"], add_models=["C"], **kw)
    assert grown.version == base.version and grown.added == ["C"]
    assert [_digest(x) for x in grown.g] == g_before
    assert {m: [_digest(a) for a in grown.maps[m]] for m in ("A", "B")} == maps_before
    assert [_digest(k) for k in torch.load(root / "k_star.pt")] == k_before
    a_c, b_b = grown.block_maps(1, "C", "B")
    assert a_c.shape == (4, 5) and b_b.shape == (4, 6)
    assert (grown.dec is None) == (register == "rot")
    import json

    stats = json.loads((root / "consensus_stats.json").read_text())
    assert set(stats["per_grid"][0]["models"]) == {"A", "B", "C"}
    assert stats["per_grid"][0]["models"]["C"]["added"] is True
    assert "one_minus_cka" in stats["per_grid"][0]["models"]["C"]
    assert fit_consensus(state, ["A", "B", "C"], **kw).version != base.version
    assert fit_consensus(state, ["A", "B"], n_grid=4, k=3, register=register).version != base.version


def test_config_consensus_options_validated():
    cfg = parse_anchors_config({"frame_map": "consensus", "consensus_k": 128, "consensus_register": "correct"})
    assert cfg.frame_map == "consensus" and cfg.consensus_k == 128
    with pytest.raises(ValueError, match="consensus"):
        parse_anchors_config({"frame_map": "gpa", "consensus_k": 128})
    with pytest.raises(ValueError, match="consensus"):
        parse_anchors_config({"frame_map": "consensus", "consensus_register": "free"})
    with pytest.raises(ValueError, match="consensus"):
        parse_anchors_config({"frame_map": "consensus", "gpa_rank": 64})
    parse_anchors_config({"frame_map": "consensus", "gpa_models": ["X"], "gpa_add_models": ["Y"]})
