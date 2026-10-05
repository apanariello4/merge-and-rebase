"""Readout diagnostic (aligned vs orthogonal part of an Ariadne task vector): toy-model tests on CPU.

Real ``open_clip`` ``VisionTransformer`` instances at tiny width (source and target with different token counts, widths
and depths), random weights, no network.
"""

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
from merge_and_rebase.merge.methods._common import axpy_state_dict
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig

N_CLASSES = 5
OUT_DIM = 7


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


class _Classifier:
    """The attributes of ``OpenClipClassifier`` the readout touches."""

    normalize = True
    logit_scale = 100.0

    def __init__(self, model):
        self.model = model
        self._zs_text_features = torch.empty(0)

    def build_zeroshot_text_features(self, classnames, build_cfg, *, cache_dir=None, force_rebuild=False):
        generator = torch.Generator().manual_seed(5)
        self._zs_text_features = F.normalize(torch.randn(len(classnames), OUT_DIM, generator=generator), dim=-1)


class _Dataset(TensorDataset):
    def __init__(self, images, ids):
        super().__init__(images, torch.arange(len(images)) % N_CLASSES)
        self.sample_ids = ids


def _loader(size, n, seed, ids, batch_size=4):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(_Dataset(torch.randn(n, 3, size, size, generator=g), ids), batch_size=batch_size, shuffle=False)


SRC = dict(image_size=24, patch_size=4, width=8, layers=2, heads=2)
TGT = dict(image_size=16, patch_size=4, width=12, layers=3, heads=3)


def _tau(model, seed=11, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    tau = {}
    for j in range(len(model.visual.transformer.resblocks)):
        for name in ("weight", "bias"):
            ref = model.state_dict()[f"visual.transformer.resblocks.{j}.mlp.c_proj.{name}"]
            tau[f"visual.transformer.resblocks.{j}.mlp.c_proj.{name}"] = scale * torch.randn(ref.shape, generator=g)
    return tau


# ---- pure maths ---------------------------------------------------------------------------------------------------


def test_decomposition_identities_and_zero_target_row():
    g = torch.Generator().manual_seed(0)
    delta, d = torch.randn(9, 6, generator=g), torch.randn(9, 6, generator=g)
    d[3] = 0.0
    p = R.decompose(delta, d)
    assert torch.allclose(p["aligned"] + p["orthogonal"], delta.double(), atol=1e-12)
    assert (p["aligned"] * p["orthogonal"]).sum(-1).abs().max() < 1e-12  # orthogonal per image
    assert p["c"][3] == 0 and torch.equal(p["orthogonal"][3], delta[3].double())
    assert torch.allclose(p["c"][0] * d[0].double(), p["aligned"][0])
    # delta = c d + orth with c = <delta, d>/||d||^2
    assert abs(float(p["c"][0]) - float((delta[0] * d[0]).sum() / (d[0] ** 2).sum())) < 1e-6


def test_global_coefficient_is_sum_ratio():
    g = torch.Generator().manual_seed(1)
    delta, d = torch.randn(20, 4, generator=g), torch.randn(20, 4, generator=g)
    assert abs(R.global_coefficient(delta, d) - float((delta * d).sum() / (d * d).sum())) < 1e-6
    assert R.global_coefficient(delta, torch.zeros_like(d)) == 0.0
    # delta exactly aligned with d -> c_g == c == the multiplier
    assert abs(R.global_coefficient(0.37 * d, d) - 0.37) < 1e-6


def test_alpha_selection_first_maximum_and_grid_snapping():
    alphas = [0.0, 0.1, 0.2, 0.3]
    assert R.select_alpha(alphas, [0.5, 0.7, 0.7, 0.6]) == 1  # tie keeps the earlier alpha
    assert R.select_alpha(alphas, [0.5, 0.4, 0.3, 0.2]) == 0
    assert R.select_alpha(alphas, [0.5, 0.5 + 1e-13, 0.4, 0.3]) == 0  # below the tracker's 1e-12
    grid, info = R.resolve_alpha_grid([0.0, 0.1, 0.2, 0.3], 0.20000000298023224)
    assert 0.20000000298023224 in grid and 0.2 not in grid and 1.0 in grid and info["added"] == [1.0]
    default, _ = R.resolve_alpha_grid(None, 0.35)
    assert len(default) == 21 and 0.35 in default and default[0] == 0.0 and default[-1] == 1.0


def test_check_loader_pairing_requires_identical_sequential_views():
    ids = [str(i) for i in range(8)]
    a, b = _loader(24, 8, 0, ids), _loader(16, 8, 1, ids)
    assert R.check_loader_pairing(a, b, "val")["dataset_identity_equal"]
    with pytest.raises(ValueError, match="identities"):
        R.check_loader_pairing(a, _loader(16, 8, 1, ids[::-1]), "val")
    shuffled = DataLoader(b.dataset, batch_size=4, shuffle=True)
    with pytest.raises(ValueError, match="not sequential"):
        R.check_loader_pairing(a, shuffled, "val")
    with pytest.raises(ValueError, match="batch sizes"):
        R.check_loader_pairing(a, _loader(16, 8, 1, ids, batch_size=2), "val")


def test_head_refuses_non_cls_pooling():
    model = _vit(**TGT, seed=1)
    model.visual.pool_type = "avg"
    clf = _Classifier(model)
    clf._zs_text_features = torch.ones(2, OUT_DIM)
    with pytest.raises(NotImplementedError, match="CLS pooling"):
        R.VisionHead.from_classifier(clf, "cpu")


# ---- sweeps on toy open_clip models -------------------------------------------------------------------------------


def _setup(n=10, seed=3):
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


def _cls_after_block(model, images, block):
    visual = model.visual
    x = visual._embeds(images)
    for blk in visual.transformer.resblocks[: block + 1]:
        x = blk(x)
    return x[:, 0]


def test_reference_states_match_manual_forward():
    source_base, source_ft, target, src_loader, tgt_loader = _setup()
    g = torch.Generator().manual_seed(9)
    q = torch.randn(SRC["width"], TGT["width"], generator=g)
    with torch.no_grad():
        out = R.collect_reference_states(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target,
            source_loader=src_loader,
            target_loader=tgt_loader,
            source_block=1,
            target_block=2,
            q=q,
            device="cpu",
        )
        xs = torch.cat([b[0] for b in src_loader])
        xt = torch.cat([b[0] for b in tgt_loader])
        want_d = (_cls_after_block(source_ft, xs, 1) - _cls_after_block(source_base, xs, 1)) @ q
        want_t0 = _cls_after_block(target, xt, 2)
    assert torch.allclose(out["d"], want_d, atol=1e-5)
    assert torch.allclose(out["t0"], want_t0, atol=1e-5)
    assert torch.equal(out["labels"], torch.arange(10) % N_CLASSES)
    with torch.no_grad():
        limited = R.collect_reference_states(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target,
            source_loader=src_loader,
            target_loader=tgt_loader,
            source_block=1,
            target_block=2,
            q=q,
            device="cpu",
            max_batches=1,
        )
    assert limited["t0"].shape[0] == 4


def test_reference_states_refuse_label_mismatch():
    source_base, source_ft, target, src_loader, tgt_loader = _setup()
    bad = DataLoader(
        TensorDataset(torch.randn(10, 3, 16, 16), (torch.arange(10) + 1) % N_CLASSES), batch_size=4, shuffle=False
    )
    with pytest.raises(ValueError, match="different labels"):
        R.collect_reference_states(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target,
            source_loader=src_loader,
            target_loader=bad,
            source_block=1,
            target_block=2,
            q=torch.eye(SRC["width"], TGT["width"]),
            device="cpu",
        )


def test_variants_full_equals_plain_forward_alpha0_is_base_and_orthogonality():
    source_base, source_ft, target, src_loader, tgt_loader = _setup()
    clf = _Classifier(target)
    clf.build_zeroshot_text_features(list(range(N_CLASSES)), None)
    head = R.VisionHead.from_classifier(clf, "cpu")
    q = torch.randn(SRC["width"], TGT["width"], generator=torch.Generator().manual_seed(2))
    ref = R.collect_reference_states(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        source_loader=src_loader,
        target_loader=tgt_loader,
        source_block=1,
        target_block=2,
        q=q,
        device="cpu",
    )
    base_sd = {k: v.clone() for k, v in target.state_dict().items()}
    tau = _tau(target)
    mounted = deepcopy(target)
    base_pred = R.predictions_from_states(head, ref["t0"], "cpu")
    with torch.no_grad():
        plain_base = torch.cat([head.finish(target.encode_image(b[0])) for b in tgt_loader]).argmax(-1)
    assert torch.equal(base_pred, plain_base)  # head(t0) == model forward

    for alpha in (0.0, 0.5, 1.0):
        mounted.load_state_dict(axpy_state_dict(base_sd, tau, alpha), strict=True)
        t1, _ = R.collect_mounted_cls(mounted, tgt_loader, target_block=2, device="cpu", expected_labels=ref["labels"])
        res = R.evaluate_variants(head, ref["t0"], ref["d"], t1, ref["labels"], alpha, 0.1, "cpu")
        with torch.no_grad():
            plain = torch.cat([head.finish(mounted.encode_image(b[0])) for b in tgt_loader])
            full = head.logits(ref["t0"] + res["delta"].float())
        assert (plain - full).abs().max() < 1e-4  # "full" variant == plain forward of the mounted model
        assert torch.equal(res["preds"]["full"], plain.argmax(-1))
        if alpha == 0.0:
            assert float(res["delta"].abs().max()) < 1e-6
            for v in ("full", "aligned", "orthogonal", "target"):
                assert torch.equal(res["preds"][v], base_pred), v  # alpha = 0 gives the base model
        # aligned + orthogonal = delta, aligned . orthogonal = 0, per image
        parts = R.decompose(res["delta"], ref["d"])
        assert torch.allclose(parts["aligned"] + parts["orthogonal"], res["delta"], atol=1e-12)
        assert (parts["aligned"] * parts["orthogonal"]).sum(-1).abs().max() < 1e-9
        assert torch.allclose(res["per_image"]["c"].double(), parts["c"], atol=1e-5)
    assert set(res["preds"]) == set(R.ALPHA_VARIANTS)


# ---- one task end to end (CPU) ------------------------------------------------------------------------------------


def _fake_inputs(*, reference, search_split="val", merge_mode="none"):
    source_base, source_ft, target, src_cal, tgt_cal = _setup(n=12)
    ids = [str(i) for i in range(10)]
    sizes = SRC["image_size"], TGT["image_size"]
    tau = _tau(target)
    target_sd = {k: v.clone() for k, v in target.state_dict().items()}
    pairing = DiscreteLayerPairing.compute(SRC["layers"], TGT["layers"])
    config = DirectResidualConfig(num_batches=3, seed=7)
    inp = SimpleNamespace(
        task="Toy",
        device="cpu",
        config=config,
        pairing=pairing,
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        target_base_state=target_sd,
        source_loader=src_cal,
        target_loader=tgt_cal,
        family_adapter=None,
        alpha_star=0.5,
        reference_task_vector=tau,
        provenance={"toy": True},
    )
    ctx = SimpleNamespace(
        source_loaders=SimpleNamespace(val=_loader(sizes[0], 10, 30, ids), test=_loader(sizes[0], 10, 31, ids)),
        loaders=SimpleNamespace(val=_loader(sizes[1], 10, 32, ids), test=_loader(sizes[1], 10, 33, ids)),
        classnames=list(range(N_CLASSES)),
        build_cfg_task=None,
    )
    resolved = SimpleNamespace(
        alpha=SimpleNamespace(search_split=search_split, selection="per_task", patience=3),
        merge=SimpleNamespace(mode=merge_mode),
    )
    summary = {
        "test_results": {
            "per_task_rebased": {"Toy": reference.get("test_rebased")},
            "per_task_baseline_accuracy": {"Toy": reference.get("test_baseline")},
        },
        "selected_baseline_alpha_by_task": {"Toy": 0.0},
        "baseline_label": "target_zeroshot",
    }
    sink = {
        "resolved": resolved,
        "summary": summary,
        "eval_context": ctx,
        "target_clf": _Classifier(target),
    }
    return inp, sink


def _run(inp, sink, **kw):
    return R.readout_task(inp, sink, alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0], zs_cache_dir=None, **kw)


def test_readout_task_end_to_end_gates_selection_and_shapes():
    inp, sink = _fake_inputs(reference={})
    record, arrays = _run(inp, sink)
    assert record["alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert record["blocks"] == {"target_block": 2, "source_block": 1, "target_depth": 3}
    assert record["pooling"]["pool_type"] == "tok"
    assert record["gates"]["g1_full_equals_plain_forward"]["passed"] is True
    assert set(record["gates"]["g1_full_equals_plain_forward"]["max_abs_logit_diff_by_alpha"]) == {"0.5", "1"}
    assert record["gates"]["g0_alpha0_reproduces_base"]["passed"] is True
    assert record["gates"]["g2_full_test_acc_equals_reference"]["passed"] is None  # no reference number
    assert record["failed_gates"] == []
    for split in ("val", "test"):
        for v in R.ALPHA_VARIANTS:
            assert len(record["curves"][split][v]) == 5
        # alpha = 0: every variant reads out the base model
        for v in R.ALPHA_VARIANTS:
            assert record["curves"][split][v][0] == record["curves"][split]["base"][0]
    # selection: argmax on validation (first maximum), test accuracy read at that alpha
    for v in R.ALPHA_VARIANTS:
        sel = record["selection"][v]
        val_curve = record["curves"]["val"][v]
        assert sel["selected_index"] == R.select_alpha(record["alphas"], val_curve)
        assert sel["test_acc"] == record["curves"]["test"][v][sel["selected_index"]]
        assert sel["test_acc_at_reference_alpha"] == record["curves"]["test"][v][2]
    assert arrays["preds"]["test"]["full"].shape == (5, 10) and arrays["c"]["test"].shape == (5, 10)
    assert arrays["preds"]["test"]["base"].shape == (10,)
    json.dumps(R._jsonable(record))


def test_readout_gate_against_reference_accuracy_and_loud_failure():
    inp, sink = _fake_inputs(reference={})
    record, _ = _run(inp, sink)
    ours = record["curves"]["test"]["full"][2]
    base = record["curves"]["test"]["base"][0]
    inp, sink = _fake_inputs(reference={"test_rebased": ours, "test_baseline": base})
    ok, _ = _run(inp, sink)
    assert ok["gates"]["g2_full_test_acc_equals_reference"]["exact"] is True
    assert ok["gates"]["g3_base_test_acc_equals_reference_baseline"]["passed"] is True
    inp, sink = _fake_inputs(reference={"test_rebased": ours + 0.3, "test_baseline": base})
    bad, _ = _run(inp, sink)
    assert bad["failed_gates"] == ["g2_full_test_acc_equals_reference"]
    assert bad["gates"]["g2_full_test_acc_equals_reference"]["difference"] == pytest.approx(-0.3)
    # a subset run skips the comparisons with the reference
    inp, sink = _fake_inputs(reference={"test_rebased": ours + 0.3})
    sub, _ = _run(inp, sink, max_batches=1)
    assert sub["gates"]["g2_full_test_acc_equals_reference"]["passed"] is None and sub["n"]["test"] == 4


def test_readout_refuses_merged_reference_and_missing_vector():
    inp, sink = _fake_inputs(reference={}, merge_mode="rebase_then_merge")
    with pytest.raises(NotImplementedError, match="merge_mode"):
        _run(inp, sink)
    inp, sink = _fake_inputs(reference={})
    inp.reference_task_vector = None
    with pytest.raises(ValueError, match="no saved unit task vector"):
        _run(inp, sink)


def test_alpha_selected_on_the_reference_search_split():
    inp, sink = _fake_inputs(reference={}, search_split="test")
    record, _ = _run(inp, sink)
    assert all(s["selection_split"] == "test" for k, s in record["selection"].items() if k != "base")
    for v in R.ALPHA_VARIANTS:
        assert record["selection"][v]["selected_index"] == R.select_alpha(record["alphas"], record["curves"]["test"][v])


# ---- write-once ---------------------------------------------------------------------------------------------------


def test_main_refuses_existing_out_dir_and_write_new_never_overwrites(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"family": "vision", "reference_summary": "x", "out_dir": str(out)}))
    with pytest.raises(FileExistsError, match="write-once"):
        R.main(["--config", str(cfg), "--device", "cpu"])
    target = tmp_path / "readout.json"
    R._write_new(target, "a")
    with pytest.raises(FileExistsError):
        R._write_new(target, "b")
    assert target.read_text() == "a"
    cfg.write_text(json.dumps({"family": "llm", "reference_summary": "x", "out_dir": str(tmp_path / "new")}))
    with pytest.raises(NotImplementedError):
        R.main(["--config", str(cfg), "--device", "cpu"])
    assert not (tmp_path / "new").exists()


# ---- optional analyses: native target, all-block comparison, oracle patching, subsets -----------------------------


def _native_model(target, seed=77, scale=0.3):
    native = deepcopy(target)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in native.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return native


def _all_q(seed=4):
    g = torch.Generator().manual_seed(seed)
    return {j: torch.randn(SRC["width"], TGT["width"], generator=g) for j in range(TGT["layers"])}


def test_native_decomposition_and_span_identities():
    g = torch.Generator().manual_seed(2)
    d, n = torch.randn(8, 6, generator=g), torch.randn(8, 6, generator=g)
    rest = torch.randn(8, 6, generator=g)
    sp = R.span_decomposition(2.0 * d - 3.0 * n, d, n)
    assert torch.allclose(sp["coef_d"], torch.full((8,), 2.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(sp["coef_n"], torch.full((8,), -3.0, dtype=torch.float64), atol=1e-6)
    assert sp["rest"].abs().max() < 1e-6
    sp = R.span_decomposition(2.0 * d - 3.0 * n + rest, d, n)
    assert (sp["rest"] * d.double()).sum(-1).abs().max() < 1e-6 and (sp["rest"] * n.double()).sum(-1).abs().max() < 1e-6
    assert torch.allclose(sp["span"] + sp["rest"], (2.0 * d - 3.0 * n + rest).double(), atol=1e-12)
    # collinear d, n: pseudo-inverse keeps the identity span + rest = delta
    sp = R.span_decomposition(rest, d, 2.0 * d)
    assert torch.allclose(sp["span"] + sp["rest"], rest.double(), atol=1e-12)
    # aligned_native + orthogonal_native = delta, orthogonal per image
    parts = R.decompose(rest, n)
    assert torch.allclose(parts["aligned"] + parts["orthogonal"], rest.double(), atol=1e-12)
    assert (parts["aligned"] * parts["orthogonal"]).sum(-1).abs().max() < 1e-12


def test_patch_beta0_is_base_and_patch_touches_one_block_all_tokens():
    _, _, target, _, tgt_loader = _setup()
    x = next(iter(tgt_loader))[0]
    last = TGT["layers"] - 1
    allb = list(range(TGT["layers"]))
    with torch.no_grad():
        plain_cls, plain = R.forward_with_block_patch(target, x, 1, None, "cpu", capture_blocks=allb)
        tokens = plain[1].shape
        zero_cls, zero = R.forward_with_block_patch(target, x, 1, torch.zeros(tokens), "cpu", capture_blocks=allb)
        logits_plain = target.encode_image(x)
        patch = torch.randn(tokens)
        cls, patched = R.forward_with_block_patch(target, x, 1, 0.7 * patch, "cpu", capture_blocks=allb)
        head_cls, _ = R.forward_with_block_patch(target, x, last, 0.7 * patch, "cpu")
    assert torch.equal(zero_cls, plain_cls)  # beta = 0 reproduces the base forward exactly
    assert all(torch.equal(zero[i], plain[i]) for i in allb)
    assert torch.equal(plain_cls, plain[last][:, 0]) and torch.allclose(
        target.visual.ln_post(plain_cls) @ target.visual.proj, logits_plain, atol=1e-5
    )
    assert torch.equal(patched[0], plain[0])  # blocks before j untouched
    assert torch.allclose(patched[1], plain[1] + 0.7 * patch)  # block j: output + patch on ALL tokens
    assert not torch.allclose(patched[last], plain[last])  # downstream blocks see it
    assert torch.allclose(head_cls, plain[last][:, 0] + 0.7 * patch[:, 0], atol=1e-6)  # patching the last block


def test_oracle_last_block_equals_target_variant_and_final_change_is_exact():
    source_base, source_ft, target, src_loader, tgt_loader = _setup()
    q_all = _all_q()
    clf = _Classifier(target)
    clf.build_zeroshot_text_features(list(range(N_CLASSES)), None)
    head = R.VisionHead.from_classifier(clf, "cpu")
    last = TGT["layers"] - 1
    pairing = DiscreteLayerPairing.compute(SRC["layers"], TGT["layers"])
    ref = R.collect_reference_states(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        source_loader=src_loader,
        target_loader=tgt_loader,
        source_block=int(pairing.pairing[last]),
        target_block=last,
        q=q_all[last],
        device="cpu",
    )
    betas = [0.0, 0.5, 1.5]
    out = R.oracle_patching(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        native_model=_native_model(target),
        head=head,
        loaders={"val": (src_loader, tgt_loader)},
        rows={"val": None},
        states={"val": ref},
        n_final={"val": torch.randn_like(ref["t0"])},
        pairing=pairing,
        q_all=q_all,
        blocks=[0, last],
        betas=betas,
        device="cpu",
        select_split="val",
    )
    res = out["results"][f"D_{last}"]
    for beta, acc in zip(betas, res["curves"]["val"], strict=True):
        states = ref["t0"].double() + beta * ref["d"].double()
        want = float((R.predictions_from_states(head, states.float(), "cpu") == ref["labels"]).double().mean())
        assert acc == want  # patch of the last block with beta*D == the "target" variant at alpha = beta
    fc = res["final_change"]["1.5"]["val"]
    assert fc["norm_over_d_final"]["mean"] == pytest.approx(1.5, rel=1e-5)
    assert fc["cos_with_d_final"]["mean"] == pytest.approx(1.0, abs=1e-5)
    # beta = 0 is the base model for every block and kind
    base_acc = float((R.predictions_from_states(head, ref["t0"], "cpu") == ref["labels"]).double().mean())
    assert all(r["curves"]["val"][0] == base_acc for r in out["results"].values())
    assert set(out["results"]) == {"D_0", f"D_{last}", "N_0", f"N_{last}"}


def test_subset_determinism_and_loader_rows():
    idx, sha = R.subset_indices(100, 10, 3)
    idx2, sha2 = R.subset_indices(100, 10, 3)
    assert torch.equal(idx, idx2) and sha == sha2 and len(idx) == 10
    assert torch.equal(idx, idx.sort().values) and len(set(idx.tolist())) == 10
    assert R.subset_indices(100, 10, 4)[1] != sha
    assert R.subset_indices(100, 100, 3) == (None, None) and R.subset_indices(100, None, 3) == (None, None)
    loader = _loader(8, 12, 0, [str(i) for i in range(12)], batch_size=4)
    sub = R.subset_loader(loader, torch.tensor([1, 5, 6, 11]))
    labels = torch.cat([b[1] for b in sub])
    assert labels.tolist() == [1 % N_CLASSES, 5 % N_CLASSES, 6 % N_CLASSES, 11 % N_CLASSES]
    assert R.resolve_oracle_blocks("auto", 24) == [5, 9, 14, 18]
    with pytest.raises(ValueError):
        R.resolve_oracle_blocks([30], 24)


def test_block_sums_match_brute_force_and_boundary_comparison_cls_matches_reference_states():
    g = torch.Generator().manual_seed(8)
    Ds = [torch.randn(3, 5, 4, generator=g) for _ in range(2)]
    Ns = [torch.randn(3, 5, 4, generator=g) for _ in range(2)]
    Ts = [{0.5: torch.randn(3, 5, 4, generator=g)} for _ in range(2)]
    sums = R.BlockSums([0], [0.5])
    for D, N, T in zip(Ds, Ns, Ts, strict=True):
        sums.update(0, D, N, T)
    out = sums.summary()[0]
    D, N, T = torch.cat(Ds).double(), torch.cat(Ns).double(), torch.cat([t[0.5] for t in Ts]).double()
    for scope, (d, n, t) in {
        "all": (D.reshape(-1, 4), N.reshape(-1, 4), T.reshape(-1, 4)),
        "cls": (D[:, 0], N[:, 0], T[:, 0]),
    }.items():
        row = out[scope]
        assert row["cos_D_N"] == pytest.approx(float((d * n).sum() / (d.norm() * n.norm())), rel=1e-10)
        assert row["norm_D_over_norm_N"] == pytest.approx(float(d.norm() / n.norm()), rel=1e-10)
        by = row["by_alpha"][0.5]
        assert by["cos_dT_N"] == pytest.approx(float((t * n).sum() / (t.norm() * n.norm())), rel=1e-10)
        assert by["projection_coefficient_dT_on_N"] == pytest.approx(float((t * n).sum() / (n * n).sum()), rel=1e-10)
        assert by["relative_error_to_N"] == pytest.approx(float((t - n).norm() / n.norm()), rel=1e-10)

    # on a toy: the CLS sums of the last block are the reference-state quantities (D_cls = d, N_cls = n)
    source_base, source_ft, target, src_loader, tgt_loader = _setup()
    q_all = _all_q()
    pairing = DiscreteLayerPairing.compute(SRC["layers"], TGT["layers"])
    last = TGT["layers"] - 1
    native = _native_model(target)
    ref = R.collect_reference_states(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        source_loader=src_loader,
        target_loader=tgt_loader,
        source_block=int(pairing.pairing[last]),
        target_block=last,
        q=q_all[last],
        device="cpu",
    )
    ncls, _ = R.collect_mounted_cls(native, tgt_loader, target_block=last, device="cpu", expected_labels=ref["labels"])
    n_vec = ncls.double() - ref["t0"].double()
    res = R.boundary_comparison(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        native_model=native,
        mounted_models={1.0: native},
        source_loader=src_loader,
        target_loader=tgt_loader,
        pairing=pairing,
        q_all=q_all,
        device="cpu",
        max_batches=None,
        block_group=2,
    )
    row = res["blocks"][last]["cls"]
    assert row["sums"]["dn"] == pytest.approx(float((ref["d"].double() * n_vec).sum()), rel=1e-5)
    assert row["sums"]["dd"] == pytest.approx(float((ref["d"].double() ** 2).sum()), rel=1e-5)
    assert row["by_alpha"][1.0]["relative_error_to_N"] < 1e-6  # mounted == native -> dT = N
    assert row["by_alpha"][1.0]["projection_coefficient_dT_on_N"] == pytest.approx(1.0)
    assert res["batches_used"] == [0, 1, 2] and res["n_images"] == 10
    capped = R.boundary_comparison(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target,
        native_model=native,
        mounted_models={1.0: native},
        source_loader=src_loader,
        target_loader=tgt_loader,
        pairing=pairing,
        q_all=q_all,
        device="cpu",
        max_batches=1,
    )
    assert capped["batches_used"] == [0] and capped["n_images"] == 4


def _write_native_ckpt(tmp_path, matches=True):
    _, _, target, _, _ = _setup(n=12)
    native = _native_model(target) if matches else _vit(**SRC, seed=99)
    sd = {k: v.clone() for k, v in native.state_dict().items()}
    sd["logit_scale"] = torch.tensor(4.6)  # a non-target key
    path = tmp_path / "native.pt"
    torch.save(sd, path)
    return str(path)


def test_readout_task_with_native_oracle_and_subset(tmp_path):
    ckpt = _write_native_ckpt(tmp_path)
    inp, sink = _fake_inputs(reference={})
    record, arrays = _run(
        inp,
        sink,
        native_ckpt_path=ckpt,
        oracle_patch_blocks=[0, 2],
        oracle_beta_grid=[0.0, 1.0],
        boundary_max_batches=2,
        eval_subset_size=6,
        eval_subset_seed=1,
    )
    ex = record["extras"]
    assert record["n"] == {"val": 10, "test": 10}  # main variants keep the full split
    assert ex["subset"]["splits"]["val"]["n"] == 6 and len(ex["subset"]["splits"]["val"]["indices_sha256"]) == 64
    nat = ex["native"]
    assert nat["inferred_base"] == "target" and nat["keys_in_ckpt_not_in_target"] == ["logit_scale"]
    assert nat["non_visual_keys_changed"] == [] and nat["ln_post_or_proj_differs"] is True
    assert nat["ceiling"]["val"]["n"] == 6
    assert 0.0 <= nat["ceiling"]["val"]["native_cls_base_head"] <= 1.0
    assert record["gates"]["g1n_native_head_equals_plain_forward"]["passed"] is True
    assert (
        set(nat["curves"]["val"]) == {"aligned_native", "orthogonal_native"}
        and len(nat["curves"]["val"]["aligned_native"]) == 5
    )
    assert nat["curves"]["val"]["aligned_native"][0] == nat["curves"]["val"]["orthogonal_native"][0]
    assert ex["boundary_comparison"]["batches_used"] == [0, 1] and set(ex["boundary_comparison"]["blocks"]) == {0, 1, 2}
    assert ex["boundary_comparison"]["alphas"] == [0.5, 1.0]
    orc = ex["oracle"]
    assert set(orc["results"]) == {"D_0", "D_2", "N_0", "N_2"}
    base_val = ex["subset"]["reference_curves_same_rows"]["val"]["base"]
    assert all(r["curves"]["val"][0] == base_val for r in orc["results"].values())  # beta=0 == base on the same rows
    assert arrays["native"]["val"]["cos_d_n"].shape == (6,)
    json.dumps(R._jsonable(record))

    # applies_to_all: the main sweeps use the subset and the reference-accuracy gates are skipped
    inp, sink = _fake_inputs(reference={"test_rebased": 0.123})
    rec2, _ = _run(inp, sink, eval_subset_size=6, eval_subset_applies_to_all=True)
    assert rec2["n"] == {"val": 6, "test": 6}
    assert rec2["gates"]["g2_full_test_acc_equals_reference"]["passed"] is None and rec2["failed_gates"] == []
    with pytest.raises(ValueError, match="mutually exclusive"):
        inp, sink = _fake_inputs(reference={})
        _run(inp, sink, eval_subset_size=6, max_batches=1)


def test_native_ckpt_of_the_wrong_backbone_is_refused(tmp_path):
    ckpt = _write_native_ckpt(tmp_path, matches=False)
    inp, sink = _fake_inputs(reference={})
    with pytest.raises(ValueError, match="not the target"):
        _run(inp, sink, native_ckpt_path=ckpt)
