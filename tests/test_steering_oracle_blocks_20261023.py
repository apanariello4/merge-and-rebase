"""Per-block steering oracle (readout ``steer_mode``): toy CPU tests with real tiny open_clip ViTs, no network."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.diagnostics.ariadne_norm import readout as R
from merge_and_rebase.eval.diagnostics.ariadne_norm import steering_oracle as S
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.layouts import _interp_2d_tokens

N_CLASSES, OUT_DIM = 5, 7
SRC = dict(image_size=24, patch_size=4, width=8, heads=2)
TGT = dict(image_size=16, patch_size=4, width=12, heads=3)


class _CLIPLike(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def encode_image(self, x):
        return self.visual(x)


def _vit(*, layers, seed, image_size, patch_size, width, heads):
    torch.manual_seed(seed)
    visual = VisionTransformer(
        image_size=image_size, patch_size=patch_size, width=width, layers=layers, heads=heads,
        mlp_ratio=2.0, ls_init_value=None, output_dim=OUT_DIM, pool_type="tok",
    )  # fmt: skip
    return _CLIPLike(visual).eval()


class _Classifier:
    normalize = True
    logit_scale = 100.0

    def __init__(self, model):
        self.model = model
        self._zs_text_features = torch.empty(0)

    def build_zeroshot_text_features(self, classnames, build_cfg, *, cache_dir=None, force_rebuild=False):
        g = torch.Generator().manual_seed(5)
        self._zs_text_features = F.normalize(torch.randn(len(classnames), OUT_DIM, generator=g), dim=-1)


class _Dataset(TensorDataset):
    def __init__(self, images, ids):
        super().__init__(images, torch.arange(len(images)) % N_CLASSES)
        self.sample_ids = ids


def _loader(size, n, seed, ids, batch_size=4):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(_Dataset(torch.randn(n, 3, size, size, generator=g), ids), batch_size=batch_size, shuffle=False)


def _tau(model, scale=0.3):
    g = torch.Generator().manual_seed(11)
    return {
        f"visual.transformer.resblocks.{j}.mlp.c_proj.{name}": scale * torch.randn(
            model.state_dict()[f"visual.transformer.resblocks.{j}.mlp.c_proj.{name}"].shape, generator=g
        )
        for j in range(len(model.visual.transformer.resblocks))
        for name in ("weight", "bias")
    }  # fmt: skip


def _fake(src_layers, tgt_layers, seed=3, n=10):
    source_base = _vit(layers=src_layers, seed=seed, **SRC)
    target = _vit(layers=tgt_layers, seed=seed + 1, **TGT)
    source_ft = deepcopy(source_base)
    g = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        for p in source_ft.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g))
    ids = [str(i) for i in range(n)]
    pairing = DiscreteLayerPairing.compute(src_layers, tgt_layers)
    inp = SimpleNamespace(
        task="Toy", device="cpu", config=DirectResidualConfig(num_batches=3, seed=7), pairing=pairing,
        source_base_model=source_base, source_ft_model=source_ft, target_model=target,
        target_base_state={k: v.clone() for k, v in target.state_dict().items()},
        source_loader=_loader(SRC["image_size"], 12, 20, [str(i) for i in range(12)]),
        target_loader=_loader(TGT["image_size"], 12, 21, [str(i) for i in range(12)]),
        family_adapter=None, alpha_star=0.5, reference_task_vector=_tau(target), provenance={"toy": True},
    )  # fmt: skip
    ctx = SimpleNamespace(
        source_loaders=SimpleNamespace(
            val=_loader(SRC["image_size"], n, 30, ids), test=_loader(SRC["image_size"], n, 31, ids)
        ),
        loaders=SimpleNamespace(val=_loader(TGT["image_size"], n, 32, ids), test=_loader(TGT["image_size"], n, 33, ids)),
        classnames=list(range(N_CLASSES)), build_cfg_task=None, source_build_cfg_task=None,
    )  # fmt: skip
    resolved = SimpleNamespace(
        alpha=SimpleNamespace(search_split="val", selection="per_task", patience=3), merge=SimpleNamespace(mode="none")
    )
    summary = {"test_results": {"per_task_rebased": {}, "per_task_baseline_accuracy": {}}}
    sink = {
        "resolved": resolved, "summary": summary, "eval_context": ctx,
        "target_clf": _Classifier(target), "source_clf": _Classifier(source_base),
    }  # fmt: skip
    return inp, sink


GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _steer(inp, sink, mode, **kw):
    return S.steering_oracle_task(inp, sink, steer_mode=mode, alpha_grid=GRID, zs_cache_dir=None, **kw)


def _block_outputs(model, x):
    outs: dict[int, torch.Tensor] = {}
    hs = [
        blk.register_forward_hook(lambda _m, _i, o, j=j: outs.__setitem__(j, (o[0] if isinstance(o, tuple) else o).detach()))
        for j, blk in enumerate(model.visual.transformer.resblocks)
    ]  # fmt: skip
    try:
        with torch.no_grad():
            model.visual(x)
    finally:
        for h in hs:
            h.remove()
    return outs


# ---- references / config -----------------------------------------------------------------------------------------


def test_references_follow_increment_references_and_target_groups():
    assert S.steering_references((0, 0, 1), "block_total") == {0: (None, 1), 1: (None, 1), 2: (None, 1)}
    assert S.steering_references((0, 0, 1), "block_increment") == {0: (None, 2), 1: (None, 2), 2: (0, 1)}
    assert S.steering_references((0, 1, 2), "block_increment") == {0: (None, 1), 1: (0, 1), 2: (1, 1)}
    assert S.source_blocks_needed((0, 0, 1), S.steering_references((0, 0, 1), "block_increment")) == [0, 1]
    with pytest.raises(ValueError, match="non-decreasing"):
        S.steering_references((0, 2, 1), "block_increment")


def test_unknown_mode_rejected_before_anything_is_created(tmp_path):
    with pytest.raises(ValueError, match="steer_mode"):
        S.validate_steer_mode("block_sum")
    assert S.validate_steer_mode("last_block") == "last_block"
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"family": "vision", "reference_summary": "x", "out_dir": str(out), "steer_mode": "nope"})
    )
    with pytest.raises(ValueError, match="steer_mode"):
        R.main(["--config", str(cfg), "--device", "cpu"])
    assert not out.exists()
    with pytest.raises(ValueError, match="last_block"):
        _steer(*_fake(2, 3), "last_block")


# ---- vectors and hooks vs explicit recomputation ------------------------------------------------------------------


def _explicit(inp, x_src, x_tgt, q_all, mode):
    """Independent recomputation from raw block outputs (own hooks, own interpolation)."""
    pairing = [int(i) for i in inp.pairing.pairing]
    bo, fo = _block_outputs(inp.source_base_model, x_src), _block_outputs(inp.source_ft_model, x_src)
    t_tokens = _block_outputs(inp.target_model, x_tgt)[0].shape[1]
    diff = {i: _interp_2d_tokens(fo[i] - bo[i], t_tokens) for i in set(pairing)}
    out = {}
    for j, i in enumerate(pairing):
        if mode == "block_total":
            out[j] = diff[i] @ q_all[j]
            continue
        run = [k for k, s in enumerate(pairing) if s == i]
        before = [k for k in range(run[0])]
        d = diff[i] - (diff[pairing[before[-1]]] if before else 0.0)
        out[j] = d @ q_all[j] / len(run)
    return out


@pytest.mark.parametrize("mode", ["block_total", "block_increment"])
@pytest.mark.parametrize("src_layers,tgt_layers", [(2, 3), (3, 3), (2, 2)])
def test_vectors_and_patched_forward_equal_explicit_recomputation(mode, src_layers, tgt_layers):
    inp, _ = _fake(src_layers, tgt_layers)
    pairing = [int(i) for i in inp.pairing.pairing]
    g = torch.Generator().manual_seed(4)
    q_all = {j: torch.randn(SRC["width"], TGT["width"], generator=g) for j in range(tgt_layers)}
    xs = torch.randn(3, 3, SRC["image_size"], SRC["image_size"], generator=g)
    xt = torch.randn(3, 3, TGT["image_size"], TGT["image_size"], generator=g)
    refs = S.steering_references(pairing, mode)
    req = {i: (i, "boundary") for i in S.source_blocks_needed(pairing, refs)}
    sb = R.capture_batch(inp.source_base_model, xs, req, "cpu")
    sf = R.capture_batch(inp.source_ft_model, xs, req, "cpu")
    dummy = torch.empty(3, _block_outputs(inp.target_model, xt)[0].shape[1], 1)
    v, total = S.steering_vectors(sf, sb, dummy, q_all, pairing, refs)
    want = _explicit(inp, xs, xt, q_all, mode)
    for j in range(tgt_layers):
        assert torch.allclose(v[j], want[j], atol=1e-5, rtol=1e-5)
    # the patched forward: every block output of the steered model = (its own steered input path) + alpha v_j
    alpha = 0.7
    cls = S.forward_with_block_patches(inp.target_model, xt, {j: alpha * v[j] for j in v}, "cpu")
    patches = {j: alpha * v[j] for j in v}
    hs = [
        blk.register_forward_hook(lambda _m, _i, o, j=j: (o[0] if isinstance(o, tuple) else o) + patches[j])
        for j, blk in enumerate(inp.target_model.visual.transformer.resblocks)
    ]  # fmt: skip
    try:
        outs = _block_outputs(inp.target_model, xt)  # recorded after the patch hooks (registered later -> run later)
    finally:
        for h in hs:
            h.remove()
    assert torch.allclose(cls, outs[tgt_layers - 1][:, 0], atol=1e-6)
    plain = _block_outputs(inp.target_model, xt)
    assert torch.allclose(outs[0], plain[0] + alpha * v[0], atol=1e-6)  # first block: nothing upstream
    # alpha = 0 is the hook-free forward, exactly
    zero = S.forward_with_block_patches(inp.target_model, xt, {j: 0.0 * v[j] for j in v}, "cpu")
    assert torch.equal(zero, S.forward_with_block_patches(inp.target_model, xt, {}, "cpu"))


def test_increment_sums_over_a_run_and_telescopes_with_identical_q():
    inp, _ = _fake(2, 3)  # pairing (0, 0, 1): runs {0, 1} -> source 0 and {2} -> source 1
    pairing = [int(i) for i in inp.pairing.pairing]
    assert pairing == [0, 0, 1]
    g = torch.Generator().manual_seed(9)
    q = torch.randn(SRC["width"], TGT["width"], generator=g)
    q_all = dict.fromkeys(range(3), q)
    xs = torch.randn(3, 3, SRC["image_size"], SRC["image_size"], generator=g)
    refs = S.steering_references(pairing, "block_increment")
    req = {i: (i, "boundary") for i in (0, 1)}
    sb = R.capture_batch(inp.source_base_model, xs, req, "cpu")
    sf = R.capture_batch(inp.source_ft_model, xs, req, "cpu")
    dummy = torch.empty(3, _block_outputs(inp.target_model, torch.randn(1, 3, 16, 16))[0].shape[1], 1)
    v, total = S.steering_vectors(sf, sb, dummy, q_all, pairing, refs)
    tol = dict(atol=1e-5, rtol=1e-5)
    assert torch.allclose(v[0] + v[1], total[0], **tol)  # first run: the run's increments sum to its step D
    assert torch.allclose(v[2], total[2] - total[1], **tol)  # step between runs
    assert torch.allclose(v[0] + v[1] + v[2], total[2], **tol)  # telescoping: the stack reproduces the last D


# ---- end to end ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["block_total", "block_increment"])
def test_alpha_zero_is_the_untouched_target_and_record_shape(mode):
    inp, sink = _fake(2, 3)
    rec, arrays = _steer(inp, sink, mode, micro_batch=3)
    assert rec["gates"]["g_alpha0_equals_untouched_target"]["passed"] is True
    assert rec["gates"]["g_alpha0_equals_untouched_target"]["max_abs_cls_diff"] == 0.0
    for s in ("val", "test"):
        assert rec["curves"][s][0] == rec["selection"][f"untouched_target_{s}_acc"]
        assert torch.equal(arrays["preds"][s][0], arrays["untouched_preds"][s])
        assert arrays["preds"][s].shape == (len(GRID), 10)
        assert len(rec["block_norms"][s]["v_over_D"]) == 3
    sel = rec["selection"]
    assert sel["val_acc"] == max(rec["curves"]["val"]) and sel["selected_alpha"] in GRID
    assert sel["test_acc"] == rec["curves"]["test"][sel["selected_index"]]
    if mode == "block_total":
        assert rec["block_norms"]["val"]["v_over_D"] == pytest.approx([1.0, 1.0, 1.0])
    else:  # pairing (0, 0, 1): g = 2, 2, 1 -> blocks 0, 1 carry half of D
        assert rec["block_norms"]["val"]["v_over_D"][:2] == pytest.approx([0.5, 0.5])
    prev = 0 if mode == "block_increment" else None
    assert rec["references"]["2"] == {"source_block": 1, "previous_source_block": prev, "g": 1}


def test_micro_batch_size_does_not_change_the_result():
    a, _ = _steer(*_fake(2, 3), "block_increment", micro_batch=1)
    b, _ = _steer(*_fake(2, 3), "block_increment", micro_batch=64)
    assert a["curves"] == b["curves"]


def test_single_block_block_total_equals_last_block_target_variant():
    inp, sink = _fake(1, 1)
    last, _ = R.readout_task(inp, sink, alpha_grid=GRID, zs_cache_dir=None)
    inp, sink = _fake(1, 1)
    rec, _ = _steer(inp, sink, "block_total")
    assert rec["alphas"] == last["alphas"]
    for s in ("val", "test"):
        assert rec["curves"][s] == last["curves"][s]["target"]
    assert rec["selection"]["selected_alpha"] == last["selection"]["target"]["selected_alpha"]
    assert rec["selection"]["test_acc"] == last["selection"]["target"]["test_acc"]


def test_eval_subset_rows_match_the_readout_subset():
    inp, sink = _fake(2, 3)
    rec, _ = _steer(inp, sink, "block_total", eval_subset_size=6, eval_subset_seed=0)
    want = R.subset_indices(10, 6, 0)[1]
    assert rec["n"] == {"val": 6, "test": 6}
    assert rec["subset"]["splits"]["val"]["indices_sha256"] == want


# ---- default mode untouched ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_extra", [{}, {"steer_mode": "last_block"}])
def test_main_default_calls_readout_task_only_and_writes_the_same_files(tmp_path, monkeypatch, cfg_extra):
    from merge_and_rebase.eval.diagnostics.ariadne_norm import vision as V

    calls = []
    inp, sink = _fake(2, 3)

    def fake_build(cfg, device, context_sink=None):
        context_sink.update(sink)
        yield inp

    def fake_readout(inp_, sink_, **kw):
        calls.append(kw)
        return {"task": "Toy", "gates": {}, "failed_gates": []}, {"x": torch.zeros(1)}

    def boom(*a, **k):
        raise AssertionError("steering oracle must not run in the default mode")

    monkeypatch.setattr(V, "build_vision_inputs", fake_build)
    monkeypatch.setattr(R, "readout_task", fake_readout)
    monkeypatch.setattr(S, "steering_oracle_task", boom)
    cfg = tmp_path / "cfg.json"
    base = {"family": "vision", "reference_summary": "x", "out_dir": str(tmp_path / "o"), "score_target": True}
    cfg.write_text(json.dumps({**base, **cfg_extra}))
    R.main(["--config", str(cfg), "--device", "cpu"])
    assert len(calls) == 1 and calls[0]["score_target"] is True and "steer_mode" not in calls[0]
    assert sorted(p.name for p in (tmp_path / "o" / "Toy").iterdir()) == ["arrays.pt", "readout.json"]
