"""Rank-truncated Procrustes screen (analysis only): toy-model tests on CPU.

Real ``open_clip`` ``VisionTransformer`` instances at tiny width (source and target with different token counts, widths
and depths), random weights, no network. Fixtures are copied from the readout tests on purpose (no cross-test imports).
"""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from open_clip.transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.diagnostics.ariadne_norm import procrustes_screen as P
from merge_and_rebase.eval.diagnostics.ariadne_norm import runner as RUN
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import _procrustes_from_cross
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.streaming import prepare_direct_residual_streaming

N_CLASSES = 5
OUT_DIM = 7
SRC = dict(image_size=24, patch_size=4, width=8, layers=2, heads=2)
TGT = dict(image_size=16, patch_size=4, width=12, layers=3, heads=3)


class _CLIPLike(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def encode_image(self, x):
        return self.visual(x)


def _vit(*, image_size, patch_size, width, layers, heads, seed):
    torch.manual_seed(seed)
    visual = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        mlp_ratio=2.0,
        ls_init_value=None,
        output_dim=OUT_DIM,
        pool_type="tok",
    )
    return _CLIPLike(visual).eval()


class _Dataset(TensorDataset):
    def __init__(self, images, ids):
        super().__init__(images, torch.arange(len(images)) % N_CLASSES)
        self.sample_ids = ids


def _loader(size, n, seed, ids, batch_size=4):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(_Dataset(torch.randn(n, 3, size, size, generator=g), ids), batch_size=batch_size, shuffle=False)


def _setup(n=12, seed=3):
    source_base = _vit(**SRC, seed=seed)
    target = _vit(**TGT, seed=seed + 1)
    source_ft = deepcopy(source_base)
    g = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        for p in source_ft.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g))
    ids = [str(i) for i in range(n)]
    return (
        source_base,
        source_ft,
        target,
        _loader(SRC["image_size"], n, 20, ids),
        _loader(TGT["image_size"], n, 21, ids),
    )


def _native_model(target, seed=77, scale=0.3):
    native = deepcopy(target)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in native.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return native


def _write_native_ckpt(tmp_path):
    _, _, target, _, _ = _setup()
    sd = {k: v.clone() for k, v in _native_model(target).state_dict().items()}
    sd["logit_scale"] = torch.tensor(4.6)
    path = tmp_path / "native.pt"
    torch.save(sd, path)
    return str(path)


def _fake_inputs():
    source_base, source_ft, target, src_cal, tgt_cal = _setup()
    ids = [str(i) for i in range(10)]
    sizes = SRC["image_size"], TGT["image_size"]
    inp = SimpleNamespace(
        task="Toy",
        device="cpu",
        config=DirectResidualConfig(num_batches=3, seed=7),
        pairing=DiscreteLayerPairing.compute(SRC["layers"], TGT["layers"]),
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        target_base_state={k: v.clone() for k, v in target.state_dict().items()},
        source_loader=src_cal,
        target_loader=tgt_cal,
        family_adapter=None,
        provenance={"toy": True},
    )
    ctx = SimpleNamespace(
        source_loaders=SimpleNamespace(val=_loader(sizes[0], 10, 30, ids)),
        loaders=SimpleNamespace(val=_loader(sizes[1], 10, 32, ids)),
    )
    return inp, {"eval_context": ctx}


def _prepare(inp):
    cfg = inp.config
    return prepare_direct_residual_streaming(
        inp.source_base_model,
        inp.target_model,
        inp.source_loader,
        inp.target_loader,
        inp.pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
        source_ft_model=inp.source_ft_model,
    )


def _bank(inp, prepared, ranks=((("2"), 2), ("4", 4), ("full", 8))):
    crosses = P.cross_sweep(
        source_base_model=inp.source_base_model,
        target_model=inp.target_model,
        source_batches=prepared["source_batches"],
        target_batches=prepared["target_batches"],
        pairing=inp.pairing,
        device="cpu",
    )
    return crosses, P.MapBank(crosses, list(ranks))


# ---- maps ---------------------------------------------------------------------------------------------------------


def test_full_rank_q_equals_method_q_and_gate_in_record():
    inp, _ = _fake_inputs()
    prepared = _prepare(inp)
    crosses, bank = _bank(inp, prepared)
    for j in range(inp.pairing.target_depth):
        want = prepared["activation_q64_by_position"][j]
        assert (bank.q_full64[j] - want).abs().max() <= P.Q_GATE_TOL
        assert (_procrustes_from_cross(crosses["all"][j].cross()).cpu() - want).abs().max() <= P.Q_GATE_TOL
        assert (bank.q_all[j][-1] - want.float()).abs().max() < 1e-6


def test_truncated_map_is_partial_isometry_of_rank_k():
    g = torch.Generator().manual_seed(0)
    cross = torch.randn(8, 12, generator=g, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    for k in (1, 3, 8):
        q = P.truncated_map(u, vh, k)
        proj = q @ q.T
        assert torch.allclose(proj @ proj, proj, atol=1e-12)  # idempotent
        assert torch.allclose(q @ q.T @ q, q, atol=1e-12)  # partial isometry
        assert int(torch.linalg.matrix_rank(q)) == k
        assert float(torch.trace(proj)) == pytest.approx(k)
    assert torch.allclose(P.truncated_map(u, vh, 8), u @ vh)


def test_odd_even_crosses_merge_to_all():
    inp, _ = _fake_inputs()
    prepared = _prepare(inp)
    crosses, _ = _bank(inp, prepared)
    for j in range(inp.pairing.target_depth):
        c_all = crosses["all"][j].cross()
        merged = P.merge_cross(crosses["even"][j], crosses["odd"][j])
        assert (merged - c_all).abs().max() / c_all.abs().max() < 1e-12
        assert crosses["even"][j].n + crosses["odd"][j].n == crosses["all"][j].n


def test_resolve_ranks_full_and_dropped():
    ranks, dropped = P.resolve_ranks([2, 32, "full", 8], 8, 12)
    assert ranks == [("2", 2), ("full", 8)] and dropped == [32]
    with pytest.raises(ValueError):
        P.resolve_ranks([64], 8, 12)


# ---- sweep B vs brute force ---------------------------------------------------------------------------------------


def _all_token_states(model, images, block):
    x = model.visual._embeds(images)
    for blk in model.visual.transformer.resblocks[: block + 1]:
        x = blk(x)
    return x


@torch.no_grad()
def test_scalar_sums_equal_brute_force_and_full_rank_d_equals_runner_desired():
    inp, _ = _fake_inputs()
    native = _native_model(inp.target_model)
    prepared = _prepare(inp)
    _, bank = _bank(inp, prepared)
    sums = P.sweep_sums(
        source_base_model=inp.source_base_model,
        source_ft_model=inp.source_ft_model,
        target_model=inp.target_model,
        native_model=native,
        source_batches=prepared["source_batches"],
        target_batches=prepared["target_batches"],
        pairing=inp.pairing,
        bank=bank,
        device="cpu",
    )
    cache, _ = RUN._cache_sweep(inp, prepared)
    pairing = inp.pairing
    last = pairing.target_depth - 1
    images = [b[0] for b in prepared["target_batches"]]
    assert sums.n_images == 12 and sums.n_batches == 3
    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        ds = torch.cat([c[i] for c in cache])  # [n, T, d_s]
        n_all = torch.cat([_all_token_states(native, x, j) - _all_token_states(inp.target_model, x, j) for x in images])
        # k = full: D identical to the runner's desired effect with the method's Q
        d_runner = torch.cat([RUN._desired(c, pairing, prepared["q_by_position"], j) for c in cache])
        d_full = ds @ bank.q_full[j]
        assert torch.equal(d_full, d_runner)
        for si, scope in enumerate(P.SCOPES):
            sl = (lambda t: t[:, 0]) if scope == "cls" else (lambda t: t.reshape(-1, t.shape[-1]))
            n64 = sl(n_all).double()
            df = sl(d_full).double()
            p = sums.p_terms[j, si]
            assert float(p[0]) == pytest.approx(float((n64**2).sum()), rel=1e-5)
            assert float(p[1]) == pytest.approx(float((df**2).sum()), rel=1e-6)
            assert float(p[2]) == pytest.approx(float((df * n64).sum()), rel=1e-4, abs=1e-6)
            for ri in range(len(bank.ranks)):
                dk = sl(ds @ bank.q_all[j][ri]).double()
                diff = sl(ds @ bank.q_diff[j][ri]).double()
                k = sums.k_terms[j, ri, si]
                assert float(k[0]) == pytest.approx(float((dk**2).sum()), rel=1e-6)
                assert float(k[1]) == pytest.approx(float((dk * n64).sum()), rel=1e-4, abs=1e-6)
                assert float(k[2]) == pytest.approx(float((diff**2).sum()), rel=1e-5)
        # full rank entry of the bank is the full map
        assert float(sums.k_terms[j, -1, 0, 0]) == pytest.approx(float(sums.p_terms[j, 0, 1]), rel=1e-9)
    # per-image CLS cosines at the last block
    ds = torch.cat([c[pairing.pairing[last]] for c in cache])
    n_cls = torch.cat(
        [_all_token_states(native, x, last) - _all_token_states(inp.target_model, x, last) for x in images]
    )[:, 0]
    for ri in range(len(bank.ranks)):
        want = torch.nn.functional.cosine_similarity((ds @ bank.q_all[last][ri])[:, 0], n_cls, dim=-1)
        got = torch.cat(sums.cos_last[ri])
        assert torch.allclose(got.float(), want, atol=1e-5)
    derived = P.derive(sums, bank, has_native=True)
    row = derived[last]["ranks"][-1]["scopes"]["all"]
    assert row["energy_kept"] == pytest.approx(1.0) and row["cos_D_N"] == pytest.approx(
        derived[last]["full_reference"]["all"]["cos_D_N"]
    )


# ---- one task, write-once, models unchanged -----------------------------------------------------------------------


def test_screen_task_with_and_without_native(tmp_path):
    ckpt = _write_native_ckpt(tmp_path)
    inp, sink = _fake_inputs()
    before = P._model_shas(inp, None)
    rec, arrays = P.screen_task(inp, sink, ranks=[2, 4, "full", 64], heldout_max_batches=2, native_ckpt_path=ckpt)
    assert P._model_shas(inp, None) == before and rec["model_state_unchanged"]
    assert [r["label"] for r in rec["ranks"]] == ["2", "4", "full"] and rec["ranks_dropped_above_full"] == [64]
    assert all(g["passed"] for g in rec["gates"].values()) and rec["failed_gates"] == []
    assert rec["heldout"]["n_images"] == 8 and rec["splits"]["heldout"]["n_images"] == 8
    assert rec["splits"]["calibration"]["n_images"] == 12
    blk = rec["splits"]["calibration"]["blocks"][-1]["ranks"]
    assert blk[-1]["mass_share"] == pytest.approx(1.0) and blk[0]["mass_share"] < 1.0
    assert blk[-1]["scopes"]["cls"]["energy_kept"] == pytest.approx(1.0)
    assert set(rec["last_block_cls_cos"]) == {"calibration", "heldout"}
    assert arrays["last_block_cls_cos"]["heldout"].shape == (8, 3)
    assert arrays["singular_values"]["all"].shape == (3, 8)
    assert len(rec["calibration_indices_sha256"]) == 64 and len(rec["heldout"]["indices_sha256"]) == 64
    json.dumps(P._jsonable(rec))

    inp, sink = _fake_inputs()
    rec2, arrays2 = P.screen_task(inp, sink, ranks=[2, "full"], heldout_max_batches=1)
    r = rec2["splits"]["calibration"]["blocks"][0]["ranks"][0]["scopes"]["all"]
    assert "cos_D_N" not in r and "size_D_over_N" not in r and r["energy_kept"] <= 1.0
    assert rec2["last_block_cls_cos"] == {} and "calibration" not in arrays2["last_block_cls_cos"]


def test_main_refuses_existing_out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"family": "vision", "reference_summary": "x", "out_dir": str(out)}))
    with pytest.raises(FileExistsError, match="write-once"):
        P.main(["--config", str(cfg), "--device", "cpu"])
    cfg.write_text(json.dumps({"family": "llm", "reference_summary": "x", "out_dir": str(tmp_path / "new")}))
    with pytest.raises(NotImplementedError):
        P.main(["--config", str(cfg), "--device", "cpu"])
    assert not (tmp_path / "new").exists()
