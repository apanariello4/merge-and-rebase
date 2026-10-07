"""Readout score-target analysis (class scores as a shared coordinate system): toy-model tests on CPU.

Real ``open_clip`` ``VisionTransformer`` instances at tiny width (source and target with different token counts, widths
and depths), random weights, no network.
"""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.diagnostics.ariadne_norm import readout as R
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
        source_build_cfg_task=None,
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
        "source_clf": _Classifier(source_base),
    }
    return inp, sink


def _run(inp, sink, **kw):
    return R.readout_task(inp, sink, alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0], zs_cache_dir=None, **kw)


def _native_model(target, seed=77, scale=0.3):
    native = deepcopy(target)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in native.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return native


def _write_native_ckpt(tmp_path):
    _, _, target, _, _ = _setup(n=12)
    sd = {k: v.clone() for k, v in _native_model(target).state_dict().items()}
    sd["logit_scale"] = torch.tensor(4.6)
    path = tmp_path / "native.pt"
    torch.save(sd, path)
    return str(path)


def _score_map(seed=1):
    target = _vit(**TGT, seed=seed)
    clf = _Classifier(target)
    clf.build_zeroshot_text_features(list(range(N_CLASSES)), None)
    return R.ScoreMap(R.VisionHead.from_classifier(clf, "cpu")), R.VisionHead.from_classifier(clf, "cpu")


def _z(n=6, seed=0):
    return torch.randn(n, TGT["width"], generator=torch.Generator().manual_seed(seed)) * 2.0


# ---- maths --------------------------------------------------------------------------------------------------------


def test_score_map_matches_head_and_jacobian_matches_finite_differences():
    sm, head = _score_map()
    z = _z()
    assert (sm.scores(z).float() - head.logits(z)).abs().max() < 1e-4
    jac = sm.jacobian(z)
    assert jac.shape == (6, N_CLASSES, TGT["width"]) and jac.dtype == torch.float64
    h = 1e-6
    eye = torch.eye(TGT["width"], dtype=torch.float64)
    for b in range(2):
        fd = torch.stack(
            [(sm.scores(z[b].double() + h * e) - sm.scores(z[b].double() - h * e)) / (2 * h) for e in eye], dim=-1
        )
        assert (fd - jac[b]).abs().max() < 1e-6 * max(1.0, float(jac[b].abs().max()))


def test_dual_solve_equals_primal_ridge_and_reproduces_centered_score_change():
    sm, _ = _score_map()
    z = _z(5, 3)
    jac = sm.jacobian(z)
    ds = torch.randn(5, N_CLASSES, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    sol = R.score_target_solve(jac, ds)
    jc, dsc, lam = sol["jc"], sol["ds"], sol["lam"]
    assert jc.sum(-2).abs().max() < 1e-12 and dsc.sum(-1).abs().max() < 1e-12  # centred over classes
    assert torch.allclose(lam, 1e-3 * (jc**2).sum((-2, -1)) / N_CLASSES)
    eye = torch.eye(TGT["width"], dtype=torch.float64)
    for b in range(5):
        primal = torch.linalg.solve(jc[b].T @ jc[b] + lam[b] * eye, jc[b].T @ dsc[b])
        assert torch.allclose(sol["delta_z"][b], primal, atol=1e-9, rtol=1e-7)
    # D = 12 > C - 1 = 4: the centred change is reproduced up to the ridge shrinkage G (G + lam I)^-1 (exactly known)
    got = (jc @ sol["delta_z"][..., None])[..., 0]
    gram = jc @ jc.transpose(-1, -2)
    shrunk = (
        gram @ torch.linalg.solve(gram + lam[:, None, None] * torch.eye(N_CLASSES, dtype=torch.float64), dsc[..., None])
    )[..., 0]
    assert torch.allclose(got, shrunk, atol=1e-9)
    assert ((got - dsc).norm(dim=-1) / dsc.norm(dim=-1)).max() < 0.1
    # first order: scores(z + eps dz) - scores(z) = eps J dz, centred = eps ds_c
    eps = 1e-4
    change = sm.scores(z + eps * sol["delta_z"]) - sm.scores(z)
    change_c = change - change.mean(-1, keepdim=True)
    assert ((change_c - eps * got).norm(dim=-1) / (eps * got.norm(dim=-1))).max() < 1e-2  # first order: eps Jc dz
    # the gate agrees, and a wrong Jacobian-independent check fails loudly through the verdict
    gate = R.jacobian_finite_difference_gate(sm, z, sol["delta_z"], "cpu")
    assert gate["passed"] is True and gate["max_relative_error"] < 1e-2
    assert R.jacobian_finite_difference_gate(sm, z, sol["delta_z"] * 1e4, "cpu", tol=1e-9)["passed"] is False


def test_score_target_rows_linearization_error_is_small_for_small_changes():
    sm, _ = _score_map()
    z = _z(4, 7)
    ds = 1e-3 * torch.randn(4, N_CLASSES, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    rows = R.score_target_rows(sm, z, ds, "cpu")
    assert rows["delta_z"].shape == (4, TGT["width"])
    assert rows["linearization_rel_error"].max() < 1e-2


# ---- end to end ---------------------------------------------------------------------------------------------------


def test_score_target_off_by_default_leaves_the_record_unchanged():
    inp, sink = _fake_inputs(reference={})
    off, arrays = _run(inp, sink)
    assert "score_target" not in off["extras"] and "score_target" not in arrays
    assert "g4_score_target_jacobian_vs_finite_difference" not in off["gates"]
    assert set(off) == {
        "task", "family", "alphas", "alpha_grid", "reference", "reference_alpha_index", "n", "subset_max_batches",
        "curves", "selection", "stats", "gates", "failed_gates", "pooling", "blocks", "q", "pairing_checks",
        "mean_d_norm", "provenance", "timings_seconds", "extras",
    }  # fmt: skip
    assert set(off["curves"]["val"]) == set(R.VARIANTS)
    inp, sink = _fake_inputs(reference={})
    on, _ = _run(inp, sink, score_target=True)
    for key in ("curves", "selection", "stats", "n", "q"):
        assert json.dumps(R._jsonable(on[key]), sort_keys=True) == json.dumps(R._jsonable(off[key]), sort_keys=True)


def _check_score_target(record, arrays, native, n_rows=10):
    st = record["extras"]["score_target"]
    assert record["gates"]["g4_score_target_jacobian_vs_finite_difference"]["passed"] is True
    assert record["failed_gates"] == []
    for split in ("val", "test"):
        assert len(st["curves"][split]) == 5
        assert st["curves"][split][0] == record["curves"][split]["base"][0]  # alpha 0 is the base model
        stat = st["stats"][split]
        for key in ("cos_dz_d", "norm_dz_over_norm_d", "linearization_rel_error_alpha1", "fit_rel_residual"):
            assert set(stat[key]) == {"median", "mean"} and stat[key]["median"] == stat[key]["median"]
        assert ("cos_dz_n" in stat) is native
    sel = st["selection"]
    assert sel["selected_index"] == R.select_alpha(record["alphas"], st["curves"]["val"])
    assert sel["test_acc"] == st["curves"]["test"][sel["selected_index"]]
    assert arrays["score_target"]["test"]["preds"].shape == (5, n_rows)
    json.dumps(R._jsonable(record))


def test_readout_task_with_score_target():
    inp, sink = _fake_inputs(reference={})
    record, arrays = _run(inp, sink, score_target=True)
    _check_score_target(record, arrays, native=False)
    # delta_z is the solution for the source's own score change: recompute it independently for the test split
    with torch.no_grad():
        src_loader = sink["eval_context"].source_loaders.test
        x = torch.cat([b[0] for b in src_loader])
        src = sink["source_clf"]
        s = [
            R.zero_shot_logits_from_features(src, m.encode_image(x)).double()
            for m in (inp.source_ft_model, inp.source_base_model)
        ]
    assert (s[0] - s[1]).abs().max() > 0
    assert torch.isfinite(arrays["score_target"]["test"]["delta_z_norm"]).all()


def test_readout_task_with_score_target_native_and_subset(tmp_path):
    ckpt = _write_native_ckpt(tmp_path)
    for applies in (False, True):
        inp, sink = _fake_inputs(reference={})
        record, arrays = _run(
            inp,
            sink,
            score_target=True,
            native_ckpt_path=ckpt,
            boundary_max_batches=1,
            eval_subset_size=6,
            eval_subset_seed=1,
            eval_subset_applies_to_all=applies,
        )
        n_rows = 6 if applies else 10
        _check_score_target(record, arrays, native=True, n_rows=n_rows)
        assert record["n"] == {"val": n_rows, "test": n_rows}
        assert arrays["score_target"]["test"]["preds"].shape == (5, n_rows)
        # same images as the other variants: identical row count in every curve source
        assert arrays["preds"]["test"]["target"].shape == (5, n_rows)
