"""``anchors`` on tiny hand-built ViT-like models: shared ridge, registration state, end-to-end transport.

The block fixture is copied from ``tests/golden/test_release_golden_hashes.py`` (``_Model``/``_dr_*``), with an
independent target width so source and target differ in width as well as depth.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.capture import capture_paired_boundary_activations
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods.anchors import (
    AnchorsConfig,
    AnchorsRebase,
    AnchorState,
    compile_desired_effects,
    decode_desired_effects,
    parse_anchors_config,
)
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase, compute_desired_effects


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


TOKENS = 5  # CLS + 2x2 grid: the canonical grid of these toys


class _IdDataset(TensorDataset):
    """TensorDataset with ``sample_ids``, the content identity the pool-draw check compares (real HF splits
    carry a split fingerprint instead)."""

    def __init__(self, images, labels, sample_ids):
        super().__init__(images, labels)
        self.sample_ids = sample_ids


def _loader(n, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, TOKENS, 4, generator=generator)
    dataset = _IdDataset(images, torch.arange(n) % 6, [f"s{seed}_{i}" for i in range(n)])
    return DataLoader(dataset, batch_size=2, shuffle=False)


def _tuned(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _rotated(model, o):
    """Same function as ``model`` with its residual stream expressed in the basis ``x' = x O``."""
    rot = deepcopy(model)
    v = rot.visual
    with torch.no_grad():
        v.input.weight.copy_(o.T @ model.visual.input.weight)
        v.input.bias.copy_(model.visual.input.bias @ o)
        for blk, ref in zip(v.transformer.resblocks, model.visual.transformer.resblocks, strict=True):
            blk.attn.out_proj.weight.copy_(o.T @ ref.attn.out_proj.weight @ o)
            blk.attn.out_proj.bias.copy_(ref.attn.out_proj.bias @ o)
            blk.mlp.c_fc.weight.copy_(ref.mlp.c_fc.weight @ o)
            blk.mlp.c_proj.weight.copy_(o.T @ ref.mlp.c_proj.weight)
            blk.mlp.c_proj.bias.copy_(ref.mlp.c_proj.bias @ o)
    return rot


def _orthogonal(d, seed):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed)))
    return q


POOL = _loader(8, seed=101)  # 4 batches of 2: anchor_batches=2 + registration_batches=2


def _cfg(**kw):
    base = dict(
        num_anchors=8,
        anchor_batches=2,
        registration_batches=2,
        canonical_tokens=TOKENS,
        num_batches=3,
        components=("mlp.c_proj",),
        ridge_estimator="empirical_bayes",
    )
    base.update(kw)
    return AnchorsConfig(**base)


def _setup(source_depth=2, target_depth=4, source_width=5, target_width=7, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(source_width, source_depth).eval()
    target_base = _Model(target_width, target_depth).eval()
    source_ft = _tuned(source_base, seed + 1)
    data = _loader(6, seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_sd


def _prepare(cfg, setup, *, state=None, source_id="A", target_id="B"):
    source_base, source_ft, target_base, data, pairing, target_sd = setup
    return AnchorsRebase().prepare(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=target_sd,
        source_loader=data,
        target_loader=data,
        pairing=pairing,
        config=cfg,
        device="cpu",
        source_anchor_loader=POOL,
        target_anchor_loader=POOL,
        source_model_id=source_id,
        target_model_id=target_id,
        state=state,
    )


# ---------------------------------------------------------------- config


def test_parse_rejects_unknown_keys_and_bad_values():
    assert parse_anchors_config({"num_anchors": 64, "components": "mlp.c_proj"}).components == ("mlp.c_proj",)
    with pytest.raises(ValueError, match="unknown anchors config keys"):
        parse_anchors_config({"num_anchor": 64})
    with pytest.raises(ValueError, match="anchor_pool"):
        parse_anchors_config({"anchor_pool": "imagenet"})


def test_registry_lists_anchors():
    from merge_and_rebase.rebase.registry import get_method, list_methods

    assert "anchors" in list_methods()
    assert isinstance(get_method("anchors"), AnchorsRebase)


# ---------------------------------------------------------------- shared ridge


@pytest.mark.parametrize("depths", [(2, 4), (4, 2)])
def test_shared_ridge_reproduces_ariadne_bitwise(depths):
    """The anchors compile step IS Ariadne's ridge: fed Ariadne's D it returns Ariadne's task vector."""
    source_base, source_ft, target_base, data, pairing, target_sd = _setup(*depths, target_width=5)
    cfg = _cfg()
    ariadne_cfg = DirectResidualConfig(
        components=("mlp.c_proj",), ridge_estimator="empirical_bayes", activation_storage="resident", num_batches=3
    )
    assert cfg.ridge_config() == ariadne_cfg
    ariadne = AriadneRebase().prepare(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=target_sd,
        source_loader=data,
        target_loader=data,
        pairing=pairing,
        config=ariadne_cfg,
        device="cpu",
    )
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, data, data, pairing, num_batches=3, seed=cfg.seed, device="cpu"
    )
    desired = compute_desired_effects(captured, pairing)
    tv, _ = compile_desired_effects(target_base, target_sd, captured, desired, pairing, config=cfg, device="cpu")
    assert set(tv) == set(ariadne.unit_task_vector)
    for key, value in tv.items():
        assert torch.equal(value, ariadne.unit_task_vector[key]), key


# ---------------------------------------------------------------- end to end


def test_prepare_produces_cproj_weight_and_bias_for_every_target_block():
    setup = _setup()
    out = _prepare(_cfg(), setup)
    depth = setup[4].target_depth
    expected = {f"visual.transformer.resblocks.{j}.mlp.c_proj.{p}" for j in range(depth) for p in ("weight", "bias")}
    assert set(out.unit_task_vector) == expected
    for key, value in out.unit_task_vector.items():
        assert torch.isfinite(value).all(), key
        assert value.shape == setup[5][key].shape
    assert any(float(v.abs().max()) > 0 for v in out.unit_task_vector.values())
    row = out.diagnostics[0]
    assert {"anchors_kernel_cka", "anchors_kernel_rel_err", "anchors_coverage"} <= set(row)
    assert AnchorsRebase().apply(out, strength=0.0) == {}
    half = AnchorsRebase().apply(out, strength=0.5)
    assert all(torch.equal(half[k], 0.5 * out.unit_task_vector[k]) for k in half)


def test_same_model_with_spanning_anchors_matches_ariadne():
    """Target = source, T > d, eps -> 0: decode(encode(E)) = E and Ariadne's Q = I, so both give the same tau."""
    torch.manual_seed(5)
    base = _Model(5, 3).eval()
    ft = _tuned(base, 6)
    data = _loader(6, seed=7)
    pairing = DiscreteLayerPairing.compute(3, 3)
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    cfg = _cfg(num_anchors=12, eps_rel=1e-12)
    setup = (base, ft, base, data, pairing, sd)
    anchors = _prepare(cfg, setup, source_id="A", target_id="A").unit_task_vector
    ariadne = (
        AriadneRebase()
        .prepare(
            source_base_model=base,
            source_ft_model=ft,
            target_model=base,
            target_base_sd=sd,
            source_loader=data,
            target_loader=data,
            pairing=pairing,
            config=cfg.ridge_config(),
            device="cpu",
        )
        .unit_task_vector
    )
    for key in ariadne:
        assert torch.allclose(anchors[key], ariadne[key], atol=1e-4, rtol=1e-3), key


def test_rotated_target_receives_the_rotated_task_vector():
    """A target that is the source in a rotated basis gets exactly the rotated update, with nothing estimated:
    the anchor kernels coincide, so even with few anchors (T < d) and eps > 0 the transport is the rotation."""
    torch.manual_seed(5)
    base = _Model(6, 2).eval()
    ft = _tuned(base, 6)
    o = _orthogonal(6, seed=9)
    rot = _rotated(base, o)
    data = _loader(6, seed=7)
    pairing = DiscreteLayerPairing.compute(2, 2)
    cfg = _cfg(num_anchors=4)
    state = AnchorState(cfg)
    sd_base = {k: v.clone() for k, v in base.state_dict().items()}
    sd_rot = {k: v.clone() for k, v in rot.state_dict().items()}
    same = _prepare(cfg, (base, ft, base, data, pairing, sd_base), state=state, source_id="A", target_id="A")
    moved = _prepare(cfg, (base, ft, rot, data, pairing, sd_rot), state=state, source_id="A", target_id="A_rot")
    for j in range(2):
        w = f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"
        b = f"visual.transformer.resblocks.{j}.mlp.c_proj.bias"
        assert torch.allclose(moved.unit_task_vector[w], o.T @ same.unit_task_vector[w], atol=1e-5)
        assert torch.allclose(moved.unit_task_vector[b], same.unit_task_vector[b] @ o, atol=1e-5)
    for row in moved.diagnostics:
        assert row["anchors_kernel_rel_err"] < 1e-5
        assert row["anchors_kernel_cka"] == pytest.approx(1.0, abs=1e-6)


def test_decode_resamples_through_the_canonical_grid():
    """Source and target on different token grids: D comes back on the target's native grid."""
    torch.manual_seed(0)
    cfg = _cfg(num_anchors=4, canonical_tokens=10)  # CLS + 3x3
    state = AnchorState(cfg)

    class _Reg:
        def __init__(self, d, seed):
            g = torch.Generator().manual_seed(seed)
            self.p = [torch.randn(4, d, generator=g, dtype=torch.float64)]
            self.c = [1.0]

    src, tgt = _Reg(6, 1), _Reg(9, 2)
    pairing = DiscreteLayerPairing.compute(1, 1)
    captured = {
        "source_base_outputs": {0: [torch.zeros(2, 17, 6)]},  # CLS + 4x4
        "source_ft_outputs": {0: [torch.randn(2, 17, 6)]},
        "target_base_outputs_by_position": {0: [torch.zeros(2, 26, 9)]},  # CLS + 5x5
    }
    del state
    desired, diag = decode_desired_effects(captured, pairing, src, tgt, config=cfg)
    assert desired[0][0].shape == (2, 26, 9)
    assert torch.isfinite(desired[0][0]).all()
    assert diag[0]["anchors_source_block"] == 0


# ---------------------------------------------------------------- state


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_state_persists_and_registering_c_recomputes_nothing(tmp_path):
    setup = _setup()
    cfg = _cfg(state_dir=str(tmp_path / "state"))
    first = _prepare(cfg, setup)
    root = tmp_path / "state"
    before = {p.relative_to(root): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    assert (
        Path("frame.json") in before and Path("models/A/blocks.pt") in before and Path("models/B/blocks.pt") in before
    )

    # New session: same task vector, no forward pass needed for registration (models load from disk).
    second = _prepare(cfg, setup, state=AnchorState(cfg, cfg.state_dir))
    for key, value in first.unit_task_vector.items():
        assert torch.equal(value, second.unit_task_vector[key]), key

    # Model C arrives: only models/C/ is written.
    torch.manual_seed(77)
    model_c = _Model(6, 3).eval()
    state = AnchorState(cfg, cfg.state_dir)
    state.register_model(model_c, "C", POOL, device="cpu")
    after = {p.relative_to(root): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    for rel, digest in before.items():
        assert after[rel] == digest, rel
    assert set(after) - set(before) == {Path("models/C/blocks.pt"), Path("models/C/meta.json")}


def test_state_refuses_a_different_frame(tmp_path):
    cfg = _cfg(state_dir=str(tmp_path / "s"))
    AnchorState(cfg, cfg.state_dir).register_model(_Model(5, 2).eval(), "A", POOL, device="cpu")
    other = _cfg(state_dir=str(tmp_path / "s"), num_anchors=6)
    with pytest.raises(ValueError, match="different frame"):
        AnchorState(other, other.state_dir)
    # per-run fields may change on the same state
    AnchorState(_cfg(state_dir=str(tmp_path / "s"), eps_rel=1e-2), str(tmp_path / "s"))


def test_registration_rejects_a_different_pool_draw():
    cfg = _cfg()
    state = AnchorState(cfg)
    state.register_model(_Model(5, 2).eval(), "A", POOL, device="cpu")
    with pytest.raises(ValueError, match="pool"):
        state.register_model(_Model(5, 2).eval(), "B", _loader(8, seed=999), device="cpu")


def test_restrict_to_depth_zeroes_deep_blocks_only():
    from merge_and_rebase.rebase.methods.anchors.method import restrict_to_depth

    tv = {
        f"visual.transformer.resblocks.{j}.mlp.c_proj.{p}": torch.ones(2) for j in range(5) for p in ("weight", "bias")
    }
    assert restrict_to_depth(tv, 5, None) == tv
    out = restrict_to_depth(tv, 5, 0.5)  # s = j / 4: blocks 0, 1 kept (s < 0.5), 2..4 zeroed
    for key, value in out.items():
        j = int(key.split(".")[3])
        assert torch.equal(value, torch.ones(2) if j < 2 else torch.zeros(2))
    with pytest.raises(ValueError):
        parse_anchors_config({"compile_max_depth": 0.0})


def test_primal_decode_matches_kernel_decode_when_t_exceeds_d():
    """T > d_k routes through Phi = (scale/c_m) P_m^T P_k (G_k + eps I)^{-1}; same D as the T x T kernel form."""
    from merge_and_rebase.rebase.methods.anchors.core import decode_rows, encode_rows, kernel, resolve_eps
    from merge_and_rebase.rebase.methods.anchors.method import transport_map
    from merge_and_rebase.rebase.methods.anchors.state import RegisteredModel

    g = torch.Generator().manual_seed(3)
    t, d_m, d_k = 40, 6, 9
    src = RegisteredModel("a", {}, [torch.randn(t, d_m, generator=g)], [torch.zeros(d_m, dtype=torch.float64)], [1.7])
    tgt = RegisteredModel("b", {}, [torch.randn(t, d_k, generator=g)], [torch.zeros(d_k, dtype=torch.float64)], [2.3])
    e = torch.randn(11, d_m, generator=g, dtype=torch.float64)
    for rescale in (True, False):
        cfg = parse_anchors_config({"eps_rel": 0.05, "rescale_target": rescale})
        fn, eps, _ = transport_map(src, 0, tgt, 0, config=cfg)
        p_m, p_k = src.p[0].double(), tgt.p[0].double()
        k_k = kernel(p_k)
        assert eps == pytest.approx(resolve_eps(k_k, 0.05), rel=1e-12)
        scale = 2.3 if rescale else 1.7
        ref = decode_rows(encode_rows(e, p_m, 1.7), p_k, k_k, scale=scale, eps=eps)
        torch.testing.assert_close(fn(e), ref, rtol=1e-9, atol=1e-10)


def test_spectral_filters():
    """Full-rank truncate_target = eps -> 0 Tikhonov; procrustes_cross is a partial isometry; ranks clip."""
    from merge_and_rebase.rebase.methods.anchors.core import spectral_transport_map, transport_map_primal

    g = torch.Generator().manual_seed(7)
    t, d_m, d_k = 60, 6, 9
    p_m = torch.randn(t, d_m, generator=g, dtype=torch.float64)
    p_k = torch.randn(t, d_k, generator=g, dtype=torch.float64)
    full, info = spectral_transport_map(p_m, 1.5, p_k, scale=2.0, mode="truncate_target", rank=None)
    assert info["spectral_rank_used"] == d_k and info["spectral_kept_fraction"] == pytest.approx(1.0)
    torch.testing.assert_close(full, transport_map_primal(p_m, 1.5, p_k, scale=2.0, eps=1e-12), rtol=1e-7, atol=1e-9)
    q, info = spectral_transport_map(p_m, 1.0, p_k, scale=1.0, mode="procrustes_cross", rank=3)
    assert info["spectral_rank_used"] == 3
    torch.testing.assert_close(q.T @ q @ q.T @ q, q.T @ q)  # q^T q is a rank-3 projector
    assert torch.linalg.matrix_rank(q).item() == 3
    _, info = spectral_transport_map(p_m, 1.0, p_k, scale=1.0, mode="truncate_cross", rank=100)
    assert info["spectral_rank_used"] == d_m  # clipped to rank(C) = min(d_m, d_k)
    with pytest.raises(ValueError):
        parse_anchors_config({"spectral_rank": 4})  # tikhonov takes no rank
    cfg = parse_anchors_config({"spectral_filter": "truncate_cross", "spectral_rank": 4})
    assert cfg.spectral_rank == 4


# ---------------------------------------------------------------- sequential depth fit


@pytest.mark.parametrize("depths", [(2, 4), (4, 2)])
def test_sequential_desired_fn_reproducing_procrustes_is_bitwise_the_default(depths):
    """Ariadne's hook: a desired_fn that returns (f - b) Q_j gives exactly the default sequential fit."""
    from merge_and_rebase.rebase.methods._ariadne.layouts import _aligned
    from merge_and_rebase.rebase.methods._ariadne.streaming import (
        fit_direct_residual_sequential_streaming,
        prepare_direct_residual_streaming,
    )

    source_base, source_ft, target_base, data, pairing, target_sd = _setup(*depths)
    cfg = DirectResidualConfig(
        components=("mlp.c_proj",), activation_storage="streaming", depth_fit="sequential", num_batches=3
    )
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=3,
        seed=cfg.seed,
        device="cpu",
        source_ft_model=source_ft,
    )

    def procrustes(pos, i, b, f, out):
        b = _aligned([b], [out])[0]
        f = _aligned([f], [out])[0]
        return (f - b) @ prepared["q_by_position"][pos]

    args = (target_base, target_sd, source_base, source_ft, prepared, pairing)
    ref, ref_rows = fit_direct_residual_sequential_streaming(*args, config=cfg, device="cpu")
    got, got_rows = fit_direct_residual_sequential_streaming(*args, config=cfg, device="cpu", desired_fn=procrustes)
    assert set(ref) == set(got)
    for key in ref:
        assert torch.equal(ref[key], got[key]), key
    assert [r["upstream_effect_norm"] for r in ref_rows] == [r["upstream_effect_norm"] for r in got_rows]


def test_anchors_sequential_runs_and_matches_independent_on_block_zero():
    """Block 0 has no upstream, so its sequential update equals the independent one (up to the resident vs
    streaming accumulation order); deeper blocks differ because they target D_j - U_j."""
    setup = _setup(2, 4)
    ind = _prepare(_cfg(), setup)
    seq = _prepare(_cfg(depth_fit="sequential"), setup)
    assert set(seq.unit_task_vector) == set(ind.unit_task_vector)
    for key, value in seq.unit_task_vector.items():
        assert torch.isfinite(value).all(), key
    w0 = "visual.transformer.resblocks.0.mlp.c_proj.weight"
    torch.testing.assert_close(seq.unit_task_vector[w0], ind.unit_task_vector[w0], rtol=1e-4, atol=1e-6)
    w3 = "visual.transformer.resblocks.3.mlp.c_proj.weight"
    assert not torch.allclose(seq.unit_task_vector[w3], ind.unit_task_vector[w3], atol=1e-6)
    rows = {int(r["position"]): r for r in seq.diagnostics}
    assert set(rows) == {0, 1, 2, 3}
    assert all(r["depth_fit"] == "sequential" and r["anchors_d_norm"] > 0 for r in rows.values())


def test_anchors_sequential_rotated_target_receives_the_rotated_task_vector():
    """The rotation property survives the sequential fit: U_j is measured on the rotated target, so it rotates too."""
    torch.manual_seed(5)
    base = _Model(6, 2).eval()
    ft = _tuned(base, 6)
    o = _orthogonal(6, seed=9)
    rot = _rotated(base, o)
    data = _loader(6, seed=7)
    pairing = DiscreteLayerPairing.compute(2, 2)
    cfg = _cfg(num_anchors=4, depth_fit="sequential")
    state = AnchorState(cfg)
    sd_base = {k: v.clone() for k, v in base.state_dict().items()}
    sd_rot = {k: v.clone() for k, v in rot.state_dict().items()}
    same = _prepare(cfg, (base, ft, base, data, pairing, sd_base), state=state, source_id="A", target_id="A")
    moved = _prepare(cfg, (base, ft, rot, data, pairing, sd_rot), state=state, source_id="A", target_id="A_rot")
    for j in range(2):
        w = f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"
        b = f"visual.transformer.resblocks.{j}.mlp.c_proj.bias"
        assert torch.allclose(moved.unit_task_vector[w], o.T @ same.unit_task_vector[w], atol=1e-5)
        assert torch.allclose(moved.unit_task_vector[b], same.unit_task_vector[b] @ o, atol=1e-5)


# ---------------------------------------------------------------- bridge to Ariadne (map_rows='task')


@pytest.mark.parametrize("depths,widths", [((2, 4), (5, 7)), ((4, 2), (7, 5))])
def test_task_rows_full_rank_procrustes_is_ariadne(depths, widths):
    """Row 0 of the bridge: task calibration rows + procrustes_cross at full rank + no rescaling = Ariadne."""
    setup = _setup(*depths, source_width=widths[0], target_width=widths[1])
    source_base, source_ft, target_base, data, pairing, target_sd = setup
    cfg = _cfg(map_rows="task", spectral_filter="procrustes_cross", rescale_target=False)
    anchors = _prepare(cfg, setup).unit_task_vector
    ariadne = (
        AriadneRebase()
        .prepare(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target_base,
            target_base_sd=target_sd,
            source_loader=data,
            target_loader=data,
            pairing=pairing,
            config=DirectResidualConfig(
                components=("mlp.c_proj",),
                ridge_estimator="empirical_bayes",
                activation_storage="resident",
                num_batches=3,
            ),
            device="cpu",
        )
        .unit_task_vector
    )
    assert set(anchors) == set(ariadne)
    for key in ariadne:
        torch.testing.assert_close(anchors[key], ariadne[key], rtol=1e-4, atol=1e-5)


def test_task_rows_subsampling_is_paired_and_counted():
    from merge_and_rebase.rebase.methods.anchors.method import task_map_frames

    source_base, _, target_base, data, pairing, target_sd = _setup(2, 4)
    cfg = _cfg(map_rows="task", map_rows_per_image=1, map_num_rows=4)
    src, tgt = task_map_frames(source_base, target_base, target_sd, data, data, pairing, config=cfg, device="cpu")
    assert src.meta["n_rows"] == 4 and all(p.shape[0] == 4 for p in src.p + tgt.p)
    full, _ = task_map_frames(
        source_base, target_base, target_sd, data, data, pairing, config=_cfg(map_rows="task"), device="cpu"
    )
    assert full.p[0].shape[0] == 3 * 2 * TOKENS  # num_batches x batch_size x canonical tokens
    with pytest.raises(ValueError):
        parse_anchors_config({"map_num_rows": 4})  # frame rows take no task subsample


def test_gpa_rotated_target_receives_the_rotated_task_vector():
    """frame_map='gpa': a target that is the source in a rotated basis gets exactly the rotated update."""
    torch.manual_seed(5)
    base = _Model(6, 2).eval()
    ft = _tuned(base, 6)
    o = _orthogonal(6, seed=9)
    rot = _rotated(base, o)
    data = _loader(6, seed=7)
    pairing = DiscreteLayerPairing.compute(2, 2)
    cfg = _cfg(num_anchors=8, frame_map="gpa")
    state = AnchorState(cfg)
    sd_base = {k: v.clone() for k, v in base.state_dict().items()}
    sd_rot = {k: v.clone() for k, v in rot.state_dict().items()}
    same = _prepare(cfg, (base, ft, base, data, pairing, sd_base), state=state, source_id="A", target_id="A")
    moved = _prepare(cfg, (base, ft, rot, data, pairing, sd_rot), state=state, source_id="A", target_id="A_rot")
    for j in range(2):
        w = f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"
        assert torch.allclose(moved.unit_task_vector[w], o.T @ same.unit_task_vector[w], atol=1e-5)
    assert moved.extra["anchors"]["gpa_universe"]["fit_models"] == ["A", "A_rot"]


def test_gpa_add_models_target_is_added_not_fitted():
    """G4: the run's target listed in gpa_add_models enters G by a Procrustes to the fixed centre of the others."""
    setup = _setup(2, 2, source_width=6, target_width=6)
    source_base, source_ft, target_base, data, pairing, target_sd = setup
    cfg = _cfg(num_anchors=8, frame_map="gpa", gpa_models=("C",), gpa_add_models=("B",))
    state = AnchorState(cfg)
    torch.manual_seed(3)
    c_model = _Model(6, 2).eval()
    state.register_model(c_model, "C", POOL, device="cpu")
    out = _prepare(cfg, setup, state=state)
    uni = out.extra["anchors"]["gpa_universe"]
    assert uni["fit_models"] == ["A", "C"] and uni["added"] == ["B"]


def test_norm_filter_with_no_flagged_rows_is_the_unfiltered_transport(tmp_path):
    """anchor_filter='norm' with a k so large that nothing is flagged reproduces the unfiltered update; a small k
    flags anchors and is stored once per model."""
    setup = _setup(2, 4)
    plain = _prepare(_cfg(state_dir=str(tmp_path)), setup, state=AnchorState(_cfg(), tmp_path))
    state = AnchorState(_cfg(), tmp_path)
    loose = _prepare(_cfg(anchor_filter="norm", anchor_filter_k=1e6), setup, state=state)
    for key, value in plain.unit_task_vector.items():
        torch.testing.assert_close(loose.unit_task_vector[key], value, rtol=1e-4, atol=1e-6)
    assert (tmp_path / "models" / "A" / "norm_filter_k1e+06.pt").exists()
    tight = state.norm_filter(setup[0], "A", POOL, k=0.5, device="cpu")
    assert bool(tight["anchor_flags"].any())


def test_gpa_with_filter_loads_the_stored_filter_of_extra_models(tmp_path):
    """G3/G4: an extra universe model is not live in the run; its stored norm filter is loaded from the state."""
    setup = _setup(2, 2, source_width=6, target_width=6)
    cfg = _cfg(state_dir=str(tmp_path), frame_map="gpa", gpa_models=("C",), anchor_filter="norm", anchor_filter_k=1e6)
    state = AnchorState(cfg, tmp_path)
    torch.manual_seed(3)
    c_model = _Model(6, 2).eval()
    state.register_model(c_model, "C", POOL, device="cpu")
    state.norm_filter(c_model, "C", POOL, k=1e6, device="cpu")
    fresh = AnchorState(cfg, tmp_path)  # as in a new run: nothing cached in memory
    out = _prepare(cfg, setup, state=fresh)
    assert out.extra["anchors"]["gpa_universe"]["fit_models"] == ["A", "B", "C"]
    with pytest.raises(KeyError):
        AnchorState(cfg, tmp_path).load_norm_filter("C", k=7.0)


@pytest.mark.parametrize("sampling", ["whole_images", "one_per_image"])
def test_anchor_sampling_rows_and_registration(sampling):
    # 2 anchor batches of 2 images = 4 anchor images, TOKENS positions each
    num = 2 * TOKENS if sampling == "whole_images" else 3
    cfg = _cfg(num_anchors=num, anchor_sampling=sampling)
    state = AnchorState(cfg)
    reg = state.register_model(_Model(5, 2).eval(), "A", POOL, device="cpu")
    rows = state.frame["anchor_rows"]
    assert len(rows) == num and rows == sorted(rows)
    images = [r[0] for r in rows]
    if sampling == "whole_images":
        assert len(set(images)) == 2
        assert all(sorted(r[1] for r in rows if r[0] == i) == list(range(TOKENS)) for i in set(images))
    else:
        assert len(set(images)) == num
    assert reg.p[0].shape[0] == num
    assert state.frame["config"]["anchor_sampling"] == sampling


def test_anchor_sampling_default_keeps_the_frame_and_rejects_bad_sizes(tmp_path):
    assert "anchor_sampling" not in _cfg().frame()
    with pytest.raises(ValueError, match="multiple"):
        AnchorState(_cfg(num_anchors=TOKENS + 1, anchor_sampling="whole_images")).register_model(
            _Model(5, 2).eval(), "A", POOL, device="cpu"
        )
    with pytest.raises(ValueError, match="anchor images"):
        AnchorState(_cfg(num_anchors=5, anchor_sampling="one_per_image")).register_model(
            _Model(5, 2).eval(), "A", POOL, device="cpu"
        )
    with pytest.raises(ValueError, match="anchor_sampling"):
        parse_anchors_config({"anchor_sampling": "bogus"})
