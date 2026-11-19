"""c_proj-only calibration fine-tuning baseline: trainable set, calibration draw, delta format, budget stop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.eval.vision_rebase.artifacts import _load_precomputed_tv
from merge_and_rebase.finetune import cproj_calib as cc
from merge_and_rebase.finetune._vision_runtime import ImageEncoder
from merge_and_rebase.models.openclip_classifier import OpenClipClassifier
from merge_and_rebase.rebase.methods._ariadne.capture import paired_calibration
from merge_and_rebase.rebase.registry import canonical_method_name

D, NCLS = 8, 3


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(D, 2 * D)
        self.mlp.c_proj = nn.Linear(2 * D, D)

    def forward(self, x):
        return x + self.mlp.c_proj(torch.relu(self.mlp.c_fc(self.ln(x))))


class _Visual(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.inp = nn.Linear(12, D)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_Block() for _ in range(depth)])
        self.proj = nn.Linear(D, D)

    def forward(self, x):
        h = self.inp(x)
        for b in self.transformer.resblocks:
            h = b(h)
        return self.proj(h)


class _TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()
        self.transformer = nn.Linear(D, D)  # stands in for the text tower


def _encoder(seed=0):
    torch.manual_seed(seed)
    clf = OpenClipClassifier(_TinyClip(), tokenizer=None, preprocess=None)
    clf._zs_text_features = torch.nn.functional.normalize(torch.randn(NCLS, D), dim=-1)
    return ImageEncoder(clf)


class _DS(Dataset):
    def __init__(self, n=96, scale=1.0):
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(n, 12, generator=g) * scale
        self.y = torch.randint(0, NCLS, (n,), generator=g)
        self.sample_ids = [str(i) for i in range(n)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _setup(encoder, **kw):
    kw = {"lr": 1e-2, "weight_decay": 0.1, "warmup_steps": 0, "scheduler_name": "constant", "total_steps": 10} | kw
    return cc.configure_cproj_training(encoder, device=torch.device("cpu"), **kw)


def _batches(n=6, bs=8):
    return [(torch.randn(bs, 12), torch.randint(0, NCLS, (bs,))) for _ in range(n)]


def test_only_cproj_changes():
    enc = _encoder()
    before = {k: v.clone() for k, v in enc.clip_model.model.state_dict().items()}
    zs_before = enc.clip_model._zs_text_features.clone()
    opt, sched, info = _setup(enc)
    assert info["trainable_tensors"] == 6 and info["trainable_blocks"] == 3
    losses: list[float] = []
    cc.run_epoch(enc, _batches(), opt, sched, accumulate=2, grad_clip=1.0, device=torch.device("cpu"),
                 global_step=0, losses=losses)
    assert len(losses) == 3  # 6 batches, accumulate 2
    after = enc.clip_model.model.state_dict()
    keys = set(cc.cproj_keys(enc.clip_model.model))
    assert len(keys) == 6
    for k, v in before.items():
        if k in keys:
            assert not torch.equal(v, after[k]), k
        else:
            assert torch.equal(v, after[k]), k
    assert torch.equal(enc.clip_model._zs_text_features, zs_before)


def test_calibration_indices_match_paired_calibration():
    src = DataLoader(_DS(scale=1.0), batch_size=8, shuffle=True)
    tgt = DataLoader(_DS(scale=2.0), batch_size=8, shuffle=True)
    _, _, meta_ref = paired_calibration(src, tgt, num_batches=5, seed=89)
    batches, meta = cc.calibration_batches(tgt, num_batches=5, seed=89)
    assert meta["indices"] == meta_ref["indices"] and len(batches) == 5
    assert cc.indices_sha256(meta["indices"]) == cc.indices_sha256(meta_ref["indices"])
    _, _, other = paired_calibration(src, tgt, num_batches=5, seed=90)
    assert other["indices"] != meta["indices"]


def test_saved_delta_loads_through_precomputed_tv_dir(tmp_path):
    enc = _encoder()
    model = enc.clip_model.model
    keys = cc.cproj_keys(model)
    base = cc.snapshot(model, keys)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    opt, sched, _ = _setup(enc)
    cc.run_epoch(enc, _batches(), opt, sched, accumulate=1, grad_clip=1.0, device=torch.device("cpu"),
                 global_step=0, losses=[])
    delta = cc.extract_delta(model, base, keys)
    path = tmp_path / f"EuroSAT_{canonical_method_name('direct_residual')}_transported_native.pt"
    torch.save(delta, path)
    loaded, meta = _load_precomputed_tv(tmp_path, "EuroSAT", base_sd)
    assert set(loaded) == set(keys) and meta["path"] == str(path)
    for k in keys:
        torch.testing.assert_close(base_sd[k] + loaded[k], model.state_dict()[k])


def test_init_delta_roundtrip():
    enc = _encoder()
    model = enc.clip_model.model
    keys = cc.cproj_keys(model)
    base = cc.snapshot(model, keys)
    init = {k: torch.ones_like(v) for k, v in base.items()}
    cc.apply_init_delta(model, init, 0.5)
    d = cc.extract_delta(model, base, keys)
    for k in keys:
        torch.testing.assert_close(d[k], 0.5 * init[k])


def test_budget_stops_within_one_step():
    enc = _encoder()
    opt, sched, _ = _setup(enc)
    budget = 0.2
    import time

    t0 = time.perf_counter()
    out = cc.train_budget(enc, _batches(n=4), opt, sched, time_budget_s=budget, t0=t0, grad_clip=1.0,
                          device=torch.device("cpu"))
    ends = out["step_end_seconds"]
    assert out["steps"] == len(ends) >= 1
    assert ends[-1] >= budget
    assert len(ends) == 1 or ends[-2] < budget  # the step before the last one was still inside the budget
    assert out["epochs"] == out["steps"] / 4  # cycled over the same fixed 4 batches


def test_converge_keeps_best_val_weights():
    enc = _encoder()
    opt, sched, _ = _setup(enc)
    batches = _batches()
    val = DataLoader(_DS(n=32), batch_size=16)
    keys = cc.cproj_keys(enc.clip_model.model)
    out = cc.train_converge(enc, lambda: batches, val, opt, sched, keys=keys, max_epochs=6, patience=2, accumulate=1,
                            grad_clip=1.0, device=torch.device("cpu"), log=lambda *_: None)
    assert 0 <= out["best_epoch"] <= out["epochs"] <= 6
    assert enc.top1(val, "cpu") == out["best_val_acc"]


def test_cproj_subset_requires_visual_prefix():
    from merge_and_rebase.models.parameter_subsets import resolve_parameter_subset

    class _Text(nn.Module):  # bare CLIP: a text tower with the same block layout, plus the visual one
        def __init__(self):
            super().__init__()
            self.visual = _Visual(depth=2)
            self.transformer = nn.Module()
            self.transformer.resblocks = nn.ModuleList([_Block() for _ in range(2)])

    names = resolve_parameter_subset(_Text(), "c_proj_only").parameter_names
    assert len(names) == 4 and all(n.startswith("visual.transformer.resblocks.") for n in names)


class _RngDS(_DS):
    """Val dataset whose reads consume the global torch RNG (as real augmenting/eval pipelines may)."""

    def __getitem__(self, i):
        torch.rand(1)
        return super().__getitem__(i)


def _fake_build(monkeypatch, rng_val=False):
    import types

    def build(args, batch_size, split_seed):
        enc = _encoder()
        meta = {"split_map": {"train": "train", "test": "test"}, "val_fraction": args.val_fraction, "seed": split_seed}
        loaders = types.SimpleNamespace(
            train=DataLoader(_DS(96), batch_size=batch_size, shuffle=True),
            val=DataLoader((_RngDS if rng_val else _DS)(32), batch_size=16),
            test=DataLoader(_DS(40), batch_size=16),
            sizes={"train": 96, "val": 32, "test": 40},
            meta=meta,
        )
        return enc.clip_model, enc, loaders, 0.0

    monkeypatch.setattr(cc, "_build_task", build)


def _args(tmp_path, *extra):
    return cc.build_parser().parse_args(
        ["--mode", "converge_calib", "--task", "EuroSAT", "--target-model", "ViT-B-16", "--out-dir", str(tmp_path),
         "--device", "cpu", "--num-batches", "4", "--batch-size", "8", "--max-epochs", "5", *extra]
    )


def test_run_end_to_end_converge_calib(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    out = cc.run(_args(tmp_path, "--lr", "1e-2", "--split-seed", "7", "--calib-seed", "5", "--patience", "3"))
    ref_idx = paired_calibration(
        DataLoader(_DS(96), batch_size=8, shuffle=True), DataLoader(_DS(96), batch_size=8, shuffle=True),
        num_batches=4, seed=5)[2]["indices"]
    assert out["calibration_indices_sha256"] == cc.indices_sha256(ref_idx)
    sp = out["split"]
    assert sp["split_seed"] == 7 and sp["calib_seed"] == 5 and sp["val_fraction"] == 0.1
    assert sp["sizes"] == {"train": 96, "val": 32, "test": 40} and sp["meta"]["seed"] == 7
    assert sp["val_identity"]["sha256"] and sp["test_identity"]["sha256"]
    assert out["lr"] == 1e-2 and out["epoch_log"][0]["epoch"] == 0
    tr = out["timed_region"]
    assert "val evaluation" in tr["definition"] and tr["train_seconds"] > 0 and tr["wall_seconds"] >= tr["train_seconds"]
    assert out["val_acc_alpha1"] == out["best_val_acc"] and out["trainable"]["trainable_tensors"] == 6
    base_sd = {k: v.clone() for k, v in _encoder().clip_model.model.state_dict().items()}
    loaded, _ = _load_precomputed_tv(tmp_path, "EuroSAT", base_sd)
    assert set(loaded) == set(cc.cproj_keys(_encoder().clip_model.model))
    assert (tmp_path / "EuroSAT_cproj_calib.json").exists()


def test_run_best_epoch_zero_when_no_improvement(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    out = cc.run(_args(tmp_path, "--lr", "0.0", "--patience", "2"))
    assert out["best_epoch"] == 0 and out["epochs"] == 2
    assert out["best_val_acc"] == out["val_acc_zero_shot"] == out["val_acc_alpha1"]
    base_sd = _encoder().clip_model.model.state_dict()
    loaded, _ = _load_precomputed_tv(tmp_path, "EuroSAT", base_sd)
    assert all(torch.count_nonzero(v) == 0 for v in loaded.values())


def test_run_rejections(tmp_path, monkeypatch):
    import pytest

    _fake_build(monkeypatch)
    with pytest.raises(ValueError, match="val-size"):
        cc.run(_args(tmp_path, "--val-size", "100"))
    a = _args(tmp_path, "--accumulate", "2")
    a.mode, a.time_budget_s = "budget", 1.0
    with pytest.raises(ValueError, match="accumulate"):
        cc.run(a)


def test_selection_final_keeps_last_weights():
    enc = _encoder()
    opt, sched, _ = _setup(enc, lr=0.3)
    batches = _batches()
    val = DataLoader(_DS(n=32), batch_size=16)
    keys = cc.cproj_keys(enc.clip_model.model)
    # an artificially high epoch-0 val: with best_val selection the start weights would win and stop early
    out = cc.train_converge(enc, lambda: batches, val, opt, sched, keys=keys, max_epochs=4, patience=1, accumulate=1,
                            grad_clip=1.0, device=torch.device("cpu"), start_val=2.0, selection="final",
                            log=lambda *_: None)
    assert out["selection"] == "final" and out["best_epoch"] is None and out["epochs"] == 4  # no early stopping
    assert out["val_acc_final"] == out["epoch_log"][-1]["val_acc"] == enc.top1(val, "cpu")
    start = cc.snapshot(_encoder().clip_model.model, keys)
    cur = cc.snapshot(enc.clip_model.model, keys)
    assert any(not torch.equal(start[k], cur[k]) for k in keys)  # final (trained) weights kept, not epoch-0


def test_run_selection_defaults(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    a = _args(tmp_path, "--lr", "1e-2", "--selection", "final", "--max-epochs", "3")
    out = cc.run(a)
    assert out["selection"] == "final" and out["best_epoch"] is None and out["epochs"] == 3
    assert out["val_acc_final"] == out["val_acc_alpha1"]
    assert cc.build_parser().parse_args(["--mode", "converge_full", "--task", "x", "--out-dir", "o"]).selection is None


def test_train_steps_exact_count_and_passes():
    enc = _encoder()
    opt, sched, _ = _setup(enc, lr=1e-2)
    calls: list[int] = []
    loader = DataLoader(_DS(40), batch_size=8, shuffle=True)  # 5 batches / pass; accumulate 2 -> windows span passes
    out = cc.train_steps(enc, loader, DataLoader(_DS(16), batch_size=8), opt, lambda s: (calls.append(s), sched(s)),
                         max_steps=7, eval_every=3, accumulate=2, grad_clip=1.0, device=torch.device("cpu"),
                         log=lambda *_: None)
    assert out["steps"] == 7 and calls == list(range(7)) and len(out["train_loss_steps"]) == 7
    assert out["batches_used"] == 14 and out["epochs"] == 14 / 5  # exactly accumulate batches per step
    assert [v["step"] for v in out["val_log"]] == [0, 3, 6, 7] and out["selection"] == "final"
    assert out["val_acc_final"] == out["val_log"][-1]["val_acc"] and out["best_epoch"] is None


def test_run_converge_full_defaults_and_cosine_total(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    seen = {}
    orig = cc.configure_cproj_training

    def spy(enc, **kw):
        seen.update(kw)
        return orig(enc, **kw)

    monkeypatch.setattr(cc, "configure_cproj_training", spy)
    a = cc.build_parser().parse_args(["--mode", "converge_full", "--task", "EuroSAT", "--target-model", "ViT-L-14",
                                      "--out-dir", str(tmp_path), "--device", "cpu", "--max-steps", "6",
                                      "--val-every", "4", "--warmup-steps", "2"])
    assert a.max_steps == 6
    out = cc.run(a)
    assert seen["total_steps"] == 6 and seen["scheduler_name"] == "cosine" and seen["lr"] == 1e-5
    assert out["steps"] == 6 and out["max_steps"] == 6 and out["selection"] == "final"
    assert out["effective_batch"] == 128 and out["config"]["accumulate"] == 4  # L/14 yaml: 32 x 4
    assert [v["step"] for v in out["val_log"]] == [0, 4, 6] and out["best_epoch"] is None
    assert out["passes_over_data"] == 6 * 4 / len(DataLoader(_DS(96), batch_size=32))
    # defaults: 2000 steps, 500 warmup
    d = cc.build_parser().parse_args(["--mode", "converge_full", "--task", "EuroSAT", "--out-dir", "o"])
    assert d.max_steps is None and d.warmup_steps is None and d.lr == 1e-5
    import pytest

    with pytest.raises(ValueError, match="converge_full"):
        cc.run(_args(tmp_path / "x", "--max-steps", "5"))


def test_converge_full_batch_32x4_for_both_models(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    for i, model in enumerate(("ViT-L-14", "ViT-B-16")):
        a = cc.build_parser().parse_args(["--mode", "converge_full", "--task", "EuroSAT", "--target-model", model,
                                          "--out-dir", str(tmp_path / str(i)), "--device", "cpu", "--max-steps", "1"])
        out = cc.run(a)
        assert out["config"]["batch_size"] == 32 and out["config"]["accumulate"] == 4
        assert out["effective_batch"] == 128


def test_training_order_independent_of_val_every(tmp_path, monkeypatch):
    _fake_build(monkeypatch, rng_val=True)
    deltas = []
    for i, every in enumerate((2, 5)):
        a = cc.build_parser().parse_args(["--mode", "converge_full", "--task", "EuroSAT", "--out-dir", str(tmp_path / str(i)),
                                          "--device", "cpu", "--max-steps", "10", "--val-every", str(every),
                                          "--lr", "1e-2", "--warmup-steps", "2", "--accumulate", "2", "--batch-size", "8"])
        out = cc.run(a)
        assert out["config"]["patience"] is None and "recipe_used" in out["config"]
        assert "val_every-step chunks" in out["timed_region"]["definition"]
        deltas.append(torch.load(tmp_path / str(i) / "EuroSAT_ariadne_transported_native.pt", weights_only=True))
    for k in deltas[0]:
        assert torch.equal(deltas[0][k], deltas[1][k]), k


def test_run_defaults_resolve_to_2000_steps_500_warmup(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    seen = {}
    orig = cc.configure_cproj_training

    def spy(enc, **kw):
        seen.update(kw)
        return orig(enc, **kw)

    def stub_train_steps(encoder, loader, val_loader, opt, sched, **kw):
        seen["train_steps"] = kw
        return {"selection": "final", "val_acc_final": 0.0, "steps": kw["max_steps"], "epochs": 0.0,
                "best_epoch": None, "best_val_acc": None, "train_seconds": 0.0, "eval_seconds": 0.0,
                "train_cuda_peak_bytes": 0.0, "train_host_peak_bytes": 0.0, "val_log": [], "epoch_log": None,
                "train_loss_steps": []}

    monkeypatch.setattr(cc, "configure_cproj_training", spy)
    monkeypatch.setattr(cc, "train_steps", stub_train_steps)
    a = cc.build_parser().parse_args(["--mode", "converge_full", "--task", "EuroSAT", "--out-dir", str(tmp_path),
                                      "--device", "cpu"])
    out = cc.run(a)
    assert seen["total_steps"] == 2000 and seen["warmup_steps"] == 500 and seen["scheduler_name"] == "cosine"
    assert seen["lr"] == 1e-5 and seen["weight_decay"] == 0.1
    ts = seen["train_steps"]
    assert ts["max_steps"] == 2000 and ts["eval_every"] == 500 and ts["accumulate"] == 4 and ts["grad_clip"] == 1.0
    assert out["config"]["recipe_used"]["effective_batch"] == 128


# ---------------------------------------------------------------------------------------------------------------
# label-free knowledge distillation (--objective kd)
# ---------------------------------------------------------------------------------------------------------------


def _kd_args(tmp_path, *extra, mode="converge_calib"):
    a = _args(tmp_path, "--objective", "kd", "--teacher-model", "ViT-B-16", "--teacher-ckpt", str(tmp_path / "t.pt"),
              *extra)
    a.mode = mode
    if mode == "budget":
        a.time_budget_s, a.max_epochs = 0.05, None
    (tmp_path / "t.pt").write_bytes(b"fake-teacher-ckpt")
    return a


def _fake_teacher(monkeypatch, separate_view=False):
    import types

    state = {"teacher": None}

    def build(args, clf, loaders, batch_size, split_seed):
        t = _encoder(seed=5)
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        state["teacher"] = t
        tl = None
        if separate_view:
            tl = types.SimpleNamespace(train=DataLoader(_DS(96, scale=2.0), batch_size=batch_size, shuffle=True),
                                       val=DataLoader(_DS(32, scale=2.0), batch_size=16))
        return t, tl, "separate_teacher_view" if separate_view else "shared_target_view"

    monkeypatch.setattr(cc, "_build_teacher", build)
    return state


def test_kd_loss_hand_example():
    import math

    s = torch.tensor([[0.0, 2.0]])
    t = torch.tensor([[2.0, 0.0]])
    T = 2.0
    ps = torch.softmax(s / T, -1)[0]
    pt = torch.softmax(t / T, -1)[0]
    ref = T * T * sum(float(pt[i]) * (math.log(float(pt[i])) - math.log(float(ps[i]))) for i in range(2))
    assert abs(float(cc.kd_loss(s, t, T)) - ref) < 1e-6
    assert float(cc.kd_loss(t, t, T)) < 1e-7  # student == teacher
    sb, tb = torch.cat([s, t]), torch.cat([t, t])  # batch mean: (ref + 0) / 2
    assert abs(float(cc.kd_loss(sb, tb, T)) - ref / 2) < 1e-6


def test_ce_default_loss_fn_unchanged():
    batches = _batches()
    out = []
    for loss_fn in (None, torch.nn.CrossEntropyLoss()):
        enc = _encoder()
        opt, sched, _ = _setup(enc)
        losses: list[float] = []
        cc.run_epoch(enc, batches, opt, sched, accumulate=2, grad_clip=1.0, device=torch.device("cpu"),
                     global_step=0, losses=losses, loss_fn=loss_fn)
        out.append((losses, cc.snapshot(enc.clip_model.model, cc.cproj_keys(enc.clip_model.model))))
    assert out[0][0] == out[1][0]
    assert all(torch.equal(out[0][1][k], out[1][1][k]) for k in out[0][1])


def test_kd_update_independent_of_labels():
    cpu = torch.device("cpu")
    teacher = _encoder(seed=5)
    batches = _batches()
    shuffled = [(x, y[torch.randperm(len(y))]) for x, y in batches]
    snaps = []
    for bs in (batches, shuffled):
        tl = cc.compute_teacher_logits(teacher, bs, cpu)
        kb = cc.kd_batches(bs, tl)
        assert all(len(b) == 2 and b[1].dtype == torch.float32 and b[1].shape == (8, NCLS) for b in kb)
        enc = _encoder()
        opt, sched, _ = _setup(enc)
        cc.run_epoch(enc, kb, opt, sched, accumulate=1, grad_clip=1.0, device=cpu, global_step=0, losses=[],
                     loss_fn=lambda s, t: cc.kd_loss(s, t, 2.0))
        snaps.append(cc.snapshot(enc.clip_model.model, cc.cproj_keys(enc.clip_model.model)))
    assert all(torch.equal(snaps[0][k], snaps[1][k]) for k in snaps[0])
    assert any(not torch.equal(snaps[0][k], cc.snapshot(_encoder().clip_model.model, [k])[k]) for k in snaps[0])


def test_kd_rejects_missing_or_stray_teacher_flags(tmp_path, monkeypatch):
    import pytest

    _fake_build(monkeypatch)
    with pytest.raises(ValueError, match="requires --teacher-model"):
        cc.run(_args(tmp_path, "--objective", "kd"))
    with pytest.raises(ValueError, match="requires --teacher-model"):
        cc.run(_args(tmp_path, "--objective", "kd", "--teacher-model", "ViT-B-16"))
    with pytest.raises(ValueError, match="only valid with --objective kd"):
        cc.run(_args(tmp_path, "--teacher-model", "ViT-B-16"))
    with pytest.raises(ValueError, match="only valid with --objective kd"):
        cc.run(_args(tmp_path, "--kd-temperature", "3"))
    assert cc.build_parser().parse_args(["--mode", "budget", "--task", "x", "--out-dir", "o"]).objective == "ce"


def test_load_teacher_visual_fails_on_missing_keys():
    import pytest

    model = _TinyClip()
    ft = {k: torch.ones_like(v) for k, v in model.state_dict().items() if k.startswith("visual.")}
    assert cc.load_teacher_visual(model, ft) == len(ft)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in ft.items())
    ft.pop("visual.proj.weight")
    with pytest.raises(ValueError, match="missing"):
        cc.load_teacher_visual(model, ft)


def test_run_kd_converge_calib_caches_teacher_once(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    state = _fake_teacher(monkeypatch)
    calls = []
    orig = cc.compute_teacher_logits
    monkeypatch.setattr(cc, "compute_teacher_logits", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    out = cc.run(_kd_args(tmp_path, "--lr", "1e-2", "--patience", "2", "--calib-seed", "5"))
    assert len(calls) == 1  # once, not per epoch
    assert out["epochs"] >= 2 and out["objective"] == "kd"
    d = out["distillation"]
    assert d["teacher_view"] == "shared_target_view" and d["temperature"] == 2.0 and len(d["teacher_ckpt_sha256"]) == 64
    assert d["teacher_model"] == "ViT-B-16" and 0.0 <= d["teacher_val_top1"] <= 1.0
    assert d["teacher_val_top1"] == state["teacher"].top1(DataLoader(_DS(32), batch_size=16), "cpu")
    ref_idx = paired_calibration(
        DataLoader(_DS(96), batch_size=8, shuffle=True), DataLoader(_DS(96), batch_size=8, shuffle=True),
        num_batches=4, seed=5)[2]["indices"]
    assert out["calibration_indices_sha256"] == cc.indices_sha256(ref_idx)
    assert out["val_acc_alpha1"] == out["best_val_acc"]  # val-accuracy selection unchanged


def test_run_kd_budget_teacher_inside_timed_region_and_separate_view(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    _fake_teacher(monkeypatch, separate_view=True)
    seen = {}
    orig = cc.compute_teacher_logits

    def spy(teacher, batches, device):
        seen["x"] = torch.cat([b[0] for b in batches])
        return orig(teacher, batches, device)

    monkeypatch.setattr(cc, "compute_teacher_logits", spy)
    t_clock = {}
    orig_budget = cc.train_budget

    def budget(*a, **k):
        t_clock["t0"] = k["t0"]
        return orig_budget(*a, **k)

    monkeypatch.setattr(cc, "train_budget", budget)
    out = cc.run(_kd_args(tmp_path, "--lr", "1e-2", mode="budget"))
    assert out["distillation"]["teacher_view"] == "separate_teacher_view"
    # the teacher saw the scale-2 view of the same indices (teacher-preprocessed), not the target-view tensors
    assert out["calibration_meta"]["requested_batches"] == 4
    assert seen["x"].std() > 1.5 * _DS(96).x.std()
    assert out["steps"] >= 1 and out["timed_region"]["wall_seconds"] >= out["timed_region"]["train_seconds"]


def test_run_kd_converge_full_on_the_fly_teacher(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    state = _fake_teacher(monkeypatch)
    n_fwd = []
    monkeypatch.setattr(cc, "_build_teacher", (lambda orig: (lambda *a: (
        lambda r: (r[0].register_forward_hook(lambda *_: n_fwd.append(1)), r)[1])(orig(*a))))(cc._build_teacher))
    a = _args(tmp_path, "--objective", "kd", "--teacher-model", "ViT-B-16", "--teacher-ckpt", str(tmp_path / "t.pt"),
              "--lr", "1e-2")
    (tmp_path / "t.pt").write_bytes(b"x")
    a.mode, a.max_epochs, a.max_steps, a.val_every, a.accumulate = "converge_full", None, 3, 3, 2
    out = cc.run(a)
    assert out["steps"] == 3 and out["distillation"]["teacher_view"] == "shared_target_view"
    assert len(n_fwd) >= 3 * 2  # one teacher forward per training batch (3 steps x 2 accumulated), plus val
    assert state["teacher"] is not None


# ---------------------------------------------------------------------------------------------------------------
# activation-displacement distillation (--objective act)
# ---------------------------------------------------------------------------------------------------------------


class _TokBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(d, 2 * d)
        self.mlp.c_proj = nn.Linear(2 * d, d)

    def forward(self, x):
        return x + self.mlp.c_proj(torch.relu(self.mlp.c_fc(self.ln(x))))


class _TokVisual(nn.Module):
    def __init__(self, depth, d, tokens=4):
        super().__init__()
        self.tokens, self.d = tokens, d
        self.inp = nn.Linear(12, tokens * d)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_TokBlock(d) for _ in range(depth)])
        self.proj = nn.Linear(d, D)

    def forward(self, x):
        h = self.inp(x).reshape(x.shape[0], self.tokens, self.d)
        for b in self.transformer.resblocks:
            h = b(h)
        return self.proj(h.mean(1))


class _TokClip(nn.Module):
    def __init__(self, depth, d, tokens=4):
        super().__init__()
        self.visual = _TokVisual(depth, d, tokens)
        self.transformer = nn.Linear(D, D)


def _tok_model(depth, d, seed, tokens=4):
    torch.manual_seed(seed)
    return _TokClip(depth, d, tokens)


def _tok_encoder(depth=4, d=8, seed=0, tokens=4):
    clf = OpenClipClassifier(_tok_model(depth, d, seed, tokens), tokenizer=None, preprocess=None)
    clf._zs_text_features = torch.nn.functional.normalize(torch.randn(NCLS, D), dim=-1)
    return ImageEncoder(clf)


def _act_loaders(n=64, bs=8):
    return DataLoader(_DS(n), batch_size=bs, shuffle=True)


def _act_targets(target, sbase, sft, **kw):
    ld = _act_loaders()
    return cc.build_act_targets(target, sbase, sft, ld, ld, num_batches=4, seed=5, device=torch.device("cpu"), **kw)


def test_act_q_equals_ariadne_function_and_depth_map():
    from merge_and_rebase.rebase.methods._ariadne.alignment import centered_rectangular_procrustes
    from merge_and_rebase.rebase.methods._ariadne.capture import capture_tokens

    target = _tok_model(4, 8, 0)
    sbase, sft = _tok_model(2, 6, 1), _tok_model(2, 6, 2)  # source depth 2 -> target depth 4 (extend), widths differ
    tg = _act_targets(target, sbase, sft)
    assert tg.pairing == (0, 0, 1, 1) and tg.source_depth == 2  # round(j (Ds-1)/(Dt-1))
    _, tb, meta = paired_calibration(_act_loaders(), _act_loaders(), num_batches=4, seed=5)
    assert meta["indices"] == tg.calibration["indices"]
    s = capture_tokens(sbase, tb, {"0": (0, "boundary"), "1": (1, "boundary")}, torch.device("cpu"))
    t = capture_tokens(target, tb, {str(j): (j, "boundary") for j in range(4)}, torch.device("cpu"))
    for j in range(4):
        x = torch.cat([a for a in s[str(tg.pairing[j])]]).reshape(-1, 6)
        y = torch.cat(t[str(j)]).reshape(-1, 8)
        q, mu_s, mu_t = centered_rectangular_procrustes(x, y)
        torch.testing.assert_close(tg.q0[j], q.float(), atol=2e-5, rtol=1e-4)
        info = tg.q_info[j]
        assert 0 < info["procrustes_residual_rel"] <= 1.5 and len(info["q_sha256"]) == 64
        torch.testing.assert_close(tg.mu_t[j], mu_t.float(), atol=1e-5, rtol=1e-4)
    # cached displacement is S_ft - S_base on the source grid (equal grids here)
    sf = capture_tokens(sft, tb, {"0": (0, "boundary")}, torch.device("cpu"))
    torch.testing.assert_close(tg.d_s[0][0], sf["0"][0] - s["0"][0])
    torch.testing.assert_close(tg.d_s[0][1], sf["0"][0] - s["0"][0])  # block 1 also paired with source 0


def _train_act(target_enc, tg, *, learned=False, q_lr=0.0, steps=3, lr=1e-2, align=0.0, q_init="procrustes", seed=0):
    dev = torch.device("cpu")
    obj = cc.ActObjective(target_enc.clip_model.model, tg, dev, learned=learned, align_weight=align, q_init=q_init,
                          seed=seed)
    opt, sched, _ = _setup(target_enc, lr=lr)
    if learned:
        opt.add_param_group({"params": obj.q_parameters(), "lr": q_lr, "weight_decay": 0.0})
        from merge_and_rebase.finetune.schedulers import build_lr_scheduler

        sched = build_lr_scheduler(opt, name="constant", base_lrs=[lr, q_lr], warmup_length=0, steps=10)
    losses: list[float] = []
    for _ in range(steps):
        cc.run_epoch(target_enc, cc.act_batches(tg), opt, sched, accumulate=1, grad_clip=1.0, device=dev,
                     global_step=0, losses=losses, loss_fn=obj.loss_fn)
    obj.close()
    return obj, losses


def test_act_loss_zero_when_student_matches_and_labels_unused():
    target_enc = _tok_encoder(seed=0)
    sbase, sft = _tok_model(2, 8, 1), _tok_model(2, 8, 2)
    tg = _act_targets(target_enc.clip_model.model, sbase, sft)
    dev = torch.device("cpu")
    obj = cc.ActObjective(target_enc.clip_model.model, tg, dev)
    target_enc.train()
    batches = cc.act_batches(tg)
    # student = base: displacement 0, loss = mean ||D||^2 (> 0)
    l0 = obj.loss_fn(target_enc(batches[0][0]), batches[0][1])
    assert float(l0.detach()) > 0
    # overwrite the cache so that the desired displacement equals the student's own (zero) -> loss 0
    for j in range(len(tg.q0)):
        tg.d_s[0][j].zero_()
    obj._out = {}
    assert float(obj.loss_fn(target_enc(batches[0][0]), batches[0][1])) < 1e-12
    # labels are not part of the batches at all
    assert all(len(b) == 2 and b[1].ndim == 0 and b[1].dtype == torch.long for b in batches)
    obj.close()
    # changing the labels of the loader's data changes nothing (identical caches and loss)
    ds = _DS(64)
    tg2 = _act_targets(_tok_model(4, 8, 0), sbase, sft)
    ds.y = (ds.y + 1) % NCLS
    tg3 = cc.build_act_targets(_tok_model(4, 8, 0), sbase, sft, DataLoader(ds, batch_size=8, shuffle=True),
                               DataLoader(ds, batch_size=8, shuffle=True), num_batches=4, seed=5,
                               device=dev)
    torch.testing.assert_close(tg2.t_base[1], tg3.t_base[1])
    torch.testing.assert_close(tg2.q0[2], tg3.q0[2])


def test_act_gradient_reduces_error_and_only_cproj_moves():
    target_enc = _tok_encoder(seed=0)
    sbase, sft = _tok_model(2, 8, 1), _tok_model(2, 8, 2)
    tg = _act_targets(target_enc.clip_model.model, sbase, sft)
    dev = torch.device("cpu")
    before = {k: v.clone() for k, v in target_enc.clip_model.model.state_dict().items()}
    e0 = cc.act_displacement_errors(target_enc.clip_model.model, tg, tg.q0, dev)
    _train_act(target_enc, tg, steps=30, lr=3e-2)
    e1 = cc.act_displacement_errors(target_enc.clip_model.model, tg, tg.q0, dev)
    assert all(abs(a["rel_error"] - 1.0) < 1e-5 for a in e0)  # base student: displacement 0 -> relative error 1
    assert sum(b["rel_error"] for b in e1) < sum(a["rel_error"] for a in e0)
    keys = set(cc.cproj_keys(target_enc.clip_model.model))
    for k, v in target_enc.clip_model.model.state_dict().items():
        assert (not torch.equal(v, before[k])) == (k in keys), k


def test_act_learned_q_lr_zero_reproduces_procrustes_and_stiefel_holds():
    sbase, sft = _tok_model(2, 6, 1), _tok_model(2, 6, 2)  # ds=6 < dt=8: orthonormal rows
    res = {}
    for name, kw in {"fixed": {}, "learned0": {"learned": True, "q_lr": 0.0}, "learned": {"learned": True, "q_lr": 5e-2, "q_init": "random", "align": 0.5}}.items():
        enc = _tok_encoder(seed=0)
        base = cc.snapshot(enc.clip_model.model, cc.cproj_keys(enc.clip_model.model))
        tg = _act_targets(enc.clip_model.model, sbase, sft)
        obj, _ = _train_act(enc, tg, steps=4, **kw)
        res[name] = (cc.extract_delta(enc.clip_model.model, base, list(base)), obj, tg)
    for k in res["fixed"][0]:
        torch.testing.assert_close(res["learned0"][0][k], res["fixed"][0][k], atol=1e-6, rtol=1e-5)
    obj, tg = res["learned"][1], res["learned"][2]
    moved = False
    for q, q0 in zip(obj.q_matrices(), tg.q0, strict=True):
        qd = q.detach()
        assert qd.shape == q0.shape == (6, 8)
        torch.testing.assert_close(qd @ qd.T, torch.eye(6), atol=1e-4, rtol=0)  # orthonormal rows (ds < dt)
        moved |= float(torch.linalg.norm(qd - q0)) > 1e-4
    assert moved
    stats = cc.act_q_stats(tg, [q.detach() for q in obj.q_matrices()], torch.device("cpu"))
    assert all(s["q_minus_procrustes_frobenius"] >= 0 and s["alignment_residual_rel"] is not None for s in stats)
    s0 = cc.act_q_stats(tg, tg.q0, torch.device("cpu"))
    assert all(s["q_minus_procrustes_frobenius"] == 0 for s in s0)


def test_act_random_stiefel_init_on_manifold_and_seeded():
    for shape in [(6, 8), (8, 6), (7, 7)]:
        a = cc.random_stiefel(shape, torch.Generator().manual_seed(1))
        b = cc.random_stiefel(shape, torch.Generator().manual_seed(2))
        a2 = cc.random_stiefel(shape, torch.Generator().manual_seed(1))
        gram = a @ a.T if shape[0] <= shape[1] else a.T @ a
        torch.testing.assert_close(gram, torch.eye(min(shape)), atol=1e-5, rtol=0)
        assert torch.equal(a, a2) and float(torch.linalg.norm(a - b)) > 0.1
    sbase, sft = _tok_model(2, 6, 1), _tok_model(2, 6, 2)
    enc = _tok_encoder(seed=0)
    tg = _act_targets(enc.clip_model.model, sbase, sft)
    q1 = cc.ActObjective(enc.clip_model.model, tg, torch.device("cpu"), learned=True, q_init="random", seed=1)
    q2 = cc.ActObjective(enc.clip_model.model, tg, torch.device("cpu"), learned=True, q_init="random", seed=2)
    for a, b, q0 in zip(q1.q_matrices(), q2.q_matrices(), tg.q0, strict=True):
        assert float(torch.linalg.norm(a - b)) > 0.1 and float(torch.linalg.norm(a - q0)) > 0.1
        torch.testing.assert_close(a.detach() @ a.detach().T, torch.eye(6), atol=1e-4, rtol=0)
    q1.close()
    q2.close()


def _fake_act(monkeypatch, depth_src=2):
    state = {}

    def build(args, clf, loaders, batch_size, split_seed):
        t = _tok_encoder(depth=depth_src, d=8, seed=2)
        t.eval()
        state["teacher"] = t
        return t, None, "shared_target_view"

    monkeypatch.setattr(cc, "_build_teacher", build)
    monkeypatch.setattr(cc, "_build_source_base", lambda args: _tok_model(depth_src, 8, 1))

    def fake_task(args, batch_size, split_seed):
        import types

        enc = _tok_encoder(depth=4, d=8, seed=0)
        loaders = types.SimpleNamespace(
            train=DataLoader(_DS(96), batch_size=batch_size, shuffle=True),
            val=DataLoader(_DS(32), batch_size=16),
            test=DataLoader(_DS(40), batch_size=16),
            sizes={"train": 96, "val": 32, "test": 40},
            meta={"split_map": {}, "val_fraction": 0.1, "seed": split_seed},
        )
        return enc.clip_model, enc, loaders, 0.0

    monkeypatch.setattr(cc, "_build_task", fake_task)


def _act_args(tmp_path, *extra, mode="converge_calib"):
    a = _args(tmp_path, "--objective", "act", "--teacher-model", "ViT-B-16", "--teacher-ckpt", str(tmp_path / "t.pt"),
              "--lr", "1e-2", *extra)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "t.pt").write_bytes(b"fake")
    a.mode = mode
    if mode == "budget":
        a.time_budget_s, a.max_epochs = 0.05, None
    return a


def test_act_run_end_to_end_converge_calib_and_budget(tmp_path, monkeypatch):
    import pytest

    _fake_act(monkeypatch)
    out = cc.run(_act_args(tmp_path, "--patience", "2"))
    a = out["act"]
    assert out["objective"] == "act" and a["q_mode"] == "procrustes" and a["depth_map"] == [0, 0, 1, 1]
    assert len(a["q_check_procrustes"]) == 4 and len(a["final_displacement_error_per_block"]) == 4
    assert a["cache"]["mode"] == "cpu_resident" and len(a["source"]["ft_ckpt_sha256"]) == 64
    assert out["steps"] >= 1 and (tmp_path / "EuroSAT_cproj_calib.json").exists()
    tmp2 = tmp_path / "b"
    out = cc.run(_act_args(tmp2, mode="budget"))
    assert out["steps"] >= 1 and "act objective" in out["timed_region"]["definition"]
    tmp3 = tmp_path / "c"
    out = cc.run(_act_args(tmp3, "--act-q", "learned", "--act-q-lr", "1e-2", "--act-q-init", "random", "--patience", "2"))
    assert out["act"]["q_mode"] == "learned" and len(out["act"]["q_stats_end"]) == 4
    assert (tmp3 / "EuroSAT_act_Q_learned.pt").exists()
    a2 = _act_args(tmp_path / "d")
    a2.mode, a2.max_epochs, a2.max_steps = "converge_full", None, 2
    with pytest.raises(NotImplementedError):
        cc.run(a2)


def test_act_init_delta_and_option_validation(tmp_path, monkeypatch):
    import pytest

    _fake_act(monkeypatch)
    enc = _tok_encoder(seed=0)
    keys = cc.cproj_keys(enc.clip_model.model)
    init = {k: 0.05 * torch.ones_like(v) for k, v in cc.snapshot(enc.clip_model.model, keys).items()}
    torch.save(init, tmp_path / "init.pt")
    out = cc.run(_act_args(tmp_path, "--init-delta", str(tmp_path / "init.pt"), "--patience", "2"))
    # Q_j computed on the untouched target: same Q hashes as without init delta
    ref = cc.run(_act_args(tmp_path / "r", "--patience", "2"))
    assert [q["q_sha256"] for q in out["act"]["q_check_procrustes"]] == [q["q_sha256"] for q in ref["act"]["q_check_procrustes"]]
    assert out["init_delta"]["sha256"]
    with pytest.raises(ValueError, match="only valid with --objective act"):
        cc.run(_args(tmp_path / "e", "--act-q", "learned"))
    with pytest.raises(ValueError, match="requires --teacher-model"):
        cc.run(_args(tmp_path / "f", "--objective", "act"))
    with pytest.raises(ValueError, match="learned"):
        cc.run(_act_args(tmp_path / "g", "--act-align-weight", "2"))


# ---------------------------------------------------------------------------------------------------------------
# steps mode / --trainable full / --micro-batch (compute-matched baselines)
# ---------------------------------------------------------------------------------------------------------------


def _steps_args(tmp_path, *extra, mode="steps"):
    a = _args(tmp_path, *extra)
    a.mode, a.max_epochs = mode, None
    return a


def _vis_state(enc):
    return {k: v.clone() for k, v in enc.clip_model.model.state_dict().items() if k.startswith("visual.")}


def test_train_fixed_steps_exact_count_cycles_batches():
    enc = _encoder()
    opt, sched, _ = _setup(enc)
    seen = []
    orig = enc.forward

    def spy(x):
        seen.append(x.clone())
        return orig(x)

    enc.forward = spy
    bs = _batches(n=3)
    res = cc.train_fixed_steps(enc, bs, opt, sched, num_steps=7, grad_clip=1.0, device=torch.device("cpu"))
    assert res["steps"] == 7 and len(res["train_loss_steps"]) == 7 and len(res["step_end_seconds"]) == 7
    assert len(seen) == 7
    for i, x in enumerate(seen):
        assert torch.equal(x, bs[i % 3][0])
    with __import__("pytest").raises(ValueError):
        cc.train_fixed_steps(enc, bs, opt, sched, num_steps=0, grad_clip=1.0, device=torch.device("cpu"))


def test_run_steps_ce_deterministic_final_weights(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    outs = []
    for i in range(2):
        d = tmp_path / str(i)
        out = cc.run(_steps_args(d, "--lr", "1e-2", "--num-steps", "5"))
        outs.append((out, torch.load(d / f"EuroSAT_{canonical_method_name('direct_residual')}_transported_native.pt")))
    (o0, d0), (o1, d1) = outs
    assert o0["steps"] == 5 and o0["mode"] == "steps" and len(o0["train_loss_steps"]) == 5
    assert o0["epochs"] == 5 / 4 and o0["best_epoch"] is None and o0["val_log"] is None
    assert o0["timed_region"]["train_seconds"] > 0 and "optimizer steps" in o0["timed_region"]["definition"]
    assert o0["config"]["max_epochs"] is None and o0["trainable"]["trainable_mode"] == "cproj"
    assert set(d0) == set(cc.cproj_keys(_encoder().clip_model.model))
    for k in d0:
        assert torch.equal(d0[k], d1[k])
    assert o0["train_loss_steps"] == o1["train_loss_steps"]


def test_run_steps_matches_budget_prefix_and_rejections(tmp_path, monkeypatch):
    import pytest

    _fake_build(monkeypatch)
    # N steps == N sequential run_epoch steps on the same draw (final weights = weights after the last step)
    out = cc.run(_steps_args(tmp_path / "a", "--lr", "1e-2", "--num-steps", "6"))
    enc = _encoder()
    model = enc.clip_model.model
    keys = cc.cproj_keys(model)
    base = cc.snapshot(model, keys)
    batches, _ = cc.calibration_batches(DataLoader(_DS(96), batch_size=8, shuffle=True), num_batches=4, seed=89)
    opt, sched, _ = _setup(enc)
    cc.train_fixed_steps(enc, batches, opt, sched, num_steps=6, grad_clip=1.0, device=torch.device("cpu"))
    ref = cc.extract_delta(model, base, keys)
    got = torch.load(tmp_path / "a" / f"EuroSAT_{canonical_method_name('direct_residual')}_transported_native.pt")
    for k in keys:
        torch.testing.assert_close(got[k], ref[k])
    assert out["steps"] == 6
    for extra, msg in ((("--time-budget-s", "1"), "num-steps"),):
        with pytest.raises(ValueError, match=msg):
            cc.run(_steps_args(tmp_path / "b", *extra))
    with pytest.raises(ValueError, match="num-steps"):
        cc.run(_steps_args(tmp_path / "c", "--num-steps", "0"))
    with pytest.raises(ValueError, match="num-steps"):
        cc.run(_steps_args(tmp_path / "d", "--num-steps", "2", mode="converge_calib"))
    with pytest.raises(ValueError, match="accumulate"):
        cc.run(_steps_args(tmp_path / "e", "--num-steps", "2", "--accumulate", "2"))


def test_trainable_full_changes_all_visual_and_default_unchanged():
    enc = _encoder()
    before = _vis_state(enc)
    full_all = {k: v.clone() for k, v in enc.clip_model.model.state_dict().items()}
    zs = enc.clip_model._zs_text_features.clone()
    opt, sched, info = _setup(enc, trainable="full")
    assert info["trainable_mode"] == "full"
    n_vis = sum(v.numel() for v in before.values())
    assert info["trainable_params"] == n_vis
    cc.train_fixed_steps(enc, _batches(), opt, sched, num_steps=2, grad_clip=1.0, device=torch.device("cpu"))
    after = enc.clip_model.model.state_dict()
    assert set(cc.visual_float_keys(enc.clip_model.model)) == set(before)
    for k in before:
        assert not torch.equal(before[k], after[k]), k
    for k, v in full_all.items():  # non-visual (text stand-in) untouched
        if not k.startswith("visual."):
            assert torch.equal(v, after[k]), k
    assert torch.equal(enc.clip_model._zs_text_features, zs)
    # default stays c_proj only
    enc2 = _encoder()
    _, _, info2 = _setup(enc2)
    assert info2["trainable_mode"] == "cproj" and info2["trainable_tensors"] == 6
    with __import__("pytest").raises(ValueError):
        _setup(_encoder(), trainable="bogus")


def test_run_steps_trainable_full_saves_all_visual_keys(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    out = cc.run(_steps_args(tmp_path, "--lr", "1e-2", "--num-steps", "3", "--trainable", "full"))
    base_sd = _encoder().clip_model.model.state_dict()
    loaded, _ = _load_precomputed_tv(tmp_path, "EuroSAT", base_sd)
    assert set(loaded) == set(cc.visual_float_keys(_encoder().clip_model.model)) and out["delta_keys"] == len(loaded)
    assert out["trainable"]["trainable_mode"] == "full" and out["config"]["trainable"] == "full"
    assert all(torch.count_nonzero(v) > 0 for v in loaded.values())


def _grad_update(enc_seed_batches, micro, kd, accumulate_base=1):
    enc = _encoder()
    opt, sched, _ = _setup(enc, lr=1e-2)
    bs = enc_seed_batches
    loss_fn = (lambda s, t: cc.kd_loss(s, t, 2.0)) if kd else None
    if kd:
        t_enc = _encoder(seed=5)
        bs = cc.kd_batches(bs, cc.compute_teacher_logits(t_enc, bs, torch.device("cpu")))
    acc = 1
    if micro:
        bs = cc.micro_split(bs, micro)
        acc = bs[0][0].shape[0] and len(bs) // len(enc_seed_batches)
    res = cc.train_fixed_steps(enc, bs, opt, sched, num_steps=3, grad_clip=1.0, device=torch.device("cpu"),
                               loss_fn=loss_fn, accumulate=acc)
    return enc.clip_model.model.state_dict(), res


def test_micro_batch_equals_full_batch_update_ce_and_kd():
    torch.manual_seed(3)
    bs = [(torch.randn(8, 12), torch.randint(0, NCLS, (8,))) for _ in range(3)]
    for kd in (False, True):
        ref, rr = _grad_update(bs, 0, kd)
        for micro in (4, 2, 1):
            got, gr = _grad_update(bs, micro, kd)
            assert gr["steps"] == 3 == rr["steps"]
            for k in ref:
                torch.testing.assert_close(got[k], ref[k], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(gr["train_loss_steps"], rr["train_loss_steps"], rtol=1e-4, atol=1e-6)


def test_micro_split_slices_consistently_and_validates():
    import pytest

    x, y = torch.randn(8, 12), torch.randn(8, NCLS)
    parts = cc.micro_split([(x, y)], 2)
    assert len(parts) == 4 and torch.equal(torch.cat([p[0] for p in parts]), x) and torch.equal(
        torch.cat([p[1] for p in parts]), y)
    with pytest.raises(ValueError, match="divide"):
        cc.micro_split([(x, y)], 3)


def test_run_steps_micro_batch_matches_no_micro_ce_kd_and_rejections(tmp_path, monkeypatch):
    import pytest

    _fake_build(monkeypatch)
    _fake_teacher(monkeypatch)
    for tag, kd in (("ce", False), ("kd", True)):
        res = {}
        for m in ("0", "4"):
            d = tmp_path / f"{tag}{m}"
            if kd:
                d.mkdir(parents=True, exist_ok=True)
                a = _kd_args(d, "--lr", "1e-2", "--num-steps", "5", "--micro-batch", m, mode="steps")
                a.max_epochs = None
            else:
                a = _steps_args(d, "--lr", "1e-2", "--num-steps", "5", "--micro-batch", m)
            out = cc.run(a)
            res[m] = (out, torch.load(d / f"EuroSAT_{canonical_method_name('direct_residual')}_transported_native.pt"))
        assert res["0"][0]["steps"] == res["4"][0]["steps"] == 5
        assert res["4"][0]["config"]["accumulate"] == 1 and res["4"][0]["config"]["effective_batch"] == 8
        for k in res["0"][1]:
            torch.testing.assert_close(res["4"][1][k], res["0"][1][k], rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="divide"):
        cc.run(_steps_args(tmp_path / "bad", "--num-steps", "2", "--micro-batch", "3"))
    with pytest.raises(ValueError, match="micro-batch"):
        cc.run(_steps_args(tmp_path / "bad2", "--num-steps", "2", "--micro-batch", "-1"))


def test_steps_rejects_act(tmp_path, monkeypatch):
    import pytest

    _fake_build(monkeypatch)
    a = _steps_args(tmp_path, "--num-steps", "2", "--objective", "act", "--teacher-model", "ViT-B-16",
                    "--teacher-ckpt", str(tmp_path / "t.pt"))
    with pytest.raises(NotImplementedError):
        cc.run(a)


# ---------------------------------------------------------------------------------------------------------------
# LoRA variants (--trainable lora_cproj / lora_full)
# ---------------------------------------------------------------------------------------------------------------


class _AttnBlock(_Block):
    """Tiny block with an ``nn.MultiheadAttention`` (``in_proj_weight`` is a Parameter, ``out_proj`` a Linear)."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, 2, batch_first=True)

    def forward(self, x):
        h = self.ln(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp.c_proj(torch.relu(self.mlp.c_fc(self.ln(x))))


def _attn_encoder(seed=0):
    torch.manual_seed(seed)
    clip = _TinyClip()
    clip.visual.transformer.resblocks = nn.ModuleList([_AttnBlock() for _ in range(2)])
    clf = OpenClipClassifier(clip, tokenizer=None, preprocess=None)
    clf._zs_text_features = torch.nn.functional.normalize(torch.randn(NCLS, D), dim=-1)
    return ImageEncoder(clf)


def _lora_setup(enc, trainable, **kw):
    lora = cc.attach_lora(enc.clip_model.model, trainable, rank=kw.pop("rank", 4), alpha=kw.pop("alpha", None), seed=3)
    opt, sched, info = _setup(enc, trainable=trainable, lora=lora, **kw)
    return lora, opt, sched, info


def test_lora_init_is_base_and_zero_delta():
    enc = _attn_encoder()
    model = enc.clip_model.model
    x = torch.randn(8, 12)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        ref = enc(x)
    lora, _, _, info = _lora_setup(enc, "lora_full")
    with torch.no_grad():
        assert torch.equal(enc(x), ref)  # B = 0: exactly the base model
    assert all(torch.count_nonzero(v) == 0 for v in lora.deltas().values())
    keys = [k for k in base_sd if k.startswith("visual.")]
    d = cc.lora_delta_dict(lora, base_sd, keys)
    assert all(torch.count_nonzero(v) == 0 for v in d.values())
    assert info["trainable_mode"] == "lora_full" and info["lora_rank"] == 4 and info["lora_scale"] == 1.0


def test_lora_cproj_touches_only_cproj_and_delta_equals_scaled_BA():
    enc = _attn_encoder()
    model = enc.clip_model.model
    base = {k: v.clone() for k, v in model.state_dict().items()}
    keys = cc.cproj_keys(model)  # (state_dict names change once the parametrizations are attached)
    lora, opt, sched, info = _lora_setup(enc, "lora_cproj", alpha=8.0)  # scale 2
    assert info["lora_scale"] == 2.0 and len(lora.names) == 2 and all(n.endswith("mlp.c_proj.weight") for n in lora.names)
    n_params = sum(m.A.numel() + m.B.numel() for m in lora.modules)
    assert info["trainable_params"] == info["lora_trainable_params"] == n_params
    cc.train_fixed_steps(enc, _batches(), opt, sched, num_steps=2, grad_clip=1.0, device=torch.device("cpu"))
    delta = cc.lora_delta_dict(lora, {k: base[k] for k in keys}, keys)
    for n, m in zip(lora.names, lora.modules, strict=True):
        torch.testing.assert_close(delta[n], 2.0 * (m.B @ m.A).detach())
        assert torch.count_nonzero(delta[n]) > 0
    assert all(torch.count_nonzero(v) == 0 for k, v in delta.items() if k not in lora.names)  # biases zero
    # effective weights = base + delta; the frozen base is untouched during training
    for n in lora.names:
        prefix, _, leaf = n.rpartition(".")
        torch.testing.assert_close(getattr(model.get_submodule(prefix), leaf), base[n] + delta[n])
    # nothing but the LoRA factors moved: remove the parametrizations and compare everything with the base
    lora.detach()
    after = model.state_dict()
    assert set(after) == set(base)
    for k in base:
        assert torch.equal(after[k], base[k]), k


def test_lora_full_touches_the_four_matrices_incl_in_proj():
    enc = _attn_encoder()
    model = enc.clip_model.model
    base = {k: v.clone() for k, v in model.state_dict().items()}
    lora, opt, sched, _ = _lora_setup(enc, "lora_full")
    suffixes = {"attn.in_proj_weight", "attn.out_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
    assert len(lora.names) == 2 * 4 and {n.split("resblocks.")[1].split(".", 1)[1] for n in lora.names} == suffixes
    cc.train_fixed_steps(enc, _batches(), opt, sched, num_steps=3, grad_clip=1.0, device=torch.device("cpu"))
    keys = [k for k in base if k.startswith("visual.")]
    delta = cc.lora_delta_dict(lora, base, keys)
    assert set(delta) == set(keys)
    for k, v in delta.items():
        if k in lora.names:
            assert torch.count_nonzero(v) > 0, k
        else:
            assert torch.count_nonzero(v) == 0, k  # biases, LN, patch embed, proj: zero
    lora.detach()
    for k, v in base.items():
        assert torch.equal(model.state_dict()[k], v), k


def test_lora_init_seeded_and_default_scale():
    a = cc.attach_lora(_attn_encoder().clip_model.model, "lora_full", rank=4, seed=11)
    b = cc.attach_lora(_attn_encoder().clip_model.model, "lora_full", rank=4, seed=11)
    c = cc.attach_lora(_attn_encoder().clip_model.model, "lora_full", rank=4, seed=12)
    assert a.alpha == 4.0 and a.scale == 1.0
    assert all(torch.equal(x.A, y.A) for x, y in zip(a.modules, b.modules, strict=True))
    assert not torch.equal(a.modules[0].A, c.modules[0].A)
    bound = 1 / (a.modules[0].A.shape[1] ** 0.5)  # kaiming_uniform(a=sqrt 5): U(-1/sqrt(in), 1/sqrt(in))
    assert a.modules[0].A.abs().max() <= bound + 1e-7


def test_lora_converge_restores_best_factors():
    enc = _attn_encoder()
    lora, opt, sched, _ = _lora_setup(enc, "lora_cproj", lr=1e-1)
    vals = iter([0.5, 0.6, 0.4, 0.3, 0.2])  # best = epoch 1
    enc.top1 = lambda loader, device: next(vals)
    snap = {}

    orig_run_epoch = cc.run_epoch

    def spy(*a, **k):
        out = orig_run_epoch(*a, **k)
        snap.setdefault("n", 0)
        snap["n"] += 1
        if snap["n"] == 1:
            snap["A"] = [m.A.detach().clone() for m in lora.modules]
            snap["B"] = [m.B.detach().clone() for m in lora.modules]
        return out

    cc.run_epoch = spy
    try:
        res = cc.train_converge(enc, lambda: _batches(), None, opt, sched, keys=[], max_epochs=4, patience=2,
                                accumulate=1, grad_clip=1.0, device=torch.device("cpu"), extra_module=lora.module_list)
    finally:
        cc.run_epoch = orig_run_epoch
    assert res["best_epoch"] == 1
    for m, a, b in zip(lora.modules, snap["A"], snap["B"], strict=True):
        assert torch.equal(m.A, a) and torch.equal(m.B, b)
    assert any(torch.count_nonzero(m.B) > 0 for m in lora.modules)


def test_lora_rejections(tmp_path, monkeypatch):
    import pytest

    _fake_build(monkeypatch)
    a = _steps_args(tmp_path, "--num-steps", "2", "--trainable", "lora_cproj", "--objective", "act",
                    "--teacher-model", "ViT-B-16", "--teacher-ckpt", str(tmp_path / "t.pt"))
    with pytest.raises(ValueError, match="LoRA"):
        cc.run(a)
    with pytest.raises(ValueError, match="init-delta"):
        cc.run(_steps_args(tmp_path / "i", "--num-steps", "2", "--trainable", "lora_full", "--init-delta", "x.pt"))
    with pytest.raises(ValueError, match="lora-rank"):
        cc.attach_lora(_attn_encoder().clip_model.model, "lora_full", rank=0)
    with pytest.raises(ValueError):
        _setup(_attn_encoder(), trainable="lora_full")  # LoRA must be attached first


def test_run_steps_lora_saves_cproj_and_full_format_and_summary(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    for tr, keyfn in (("lora_cproj", cc.cproj_keys), ("lora_full", None)):
        d = tmp_path / tr
        out = cc.run(_steps_args(d, "--lr", "1e-2", "--num-steps", "3", "--trainable", tr, "--lora-rank", "2"))
        model = _encoder().clip_model.model
        keys = keyfn(model) if keyfn else cc.visual_float_keys(model)
        loaded, _ = _load_precomputed_tv(d, "EuroSAT", model.state_dict())
        assert set(loaded) == set(keys) and out["delta_keys"] == len(keys)
        t = out["trainable"]
        assert t["trainable_mode"] == tr and t["lora_rank"] == 2 and t["lora_alpha"] == 2.0 and t["lora_scale"] == 1.0
        assert t["lora_trainable_params"] == t["trainable_params"] > 0 and t["delta_frobenius_norm"] > 0
        assert out["timed_region"]["train_seconds"] > 0
        n_nonzero = {k for k, v in loaded.items() if torch.count_nonzero(v) > 0}
        assert n_nonzero == {k for k in keys if k.endswith("c_proj.weight") or (tr == "lora_full" and k.endswith("c_fc.weight"))}


def test_run_steps_lora_micro_batch_matches_and_converge_calib(tmp_path, monkeypatch):
    _fake_build(monkeypatch)
    res = {}
    for m in ("0", "4"):
        d = tmp_path / f"m{m}"
        cc.run(_steps_args(d, "--lr", "1e-2", "--num-steps", "4", "--trainable", "lora_full", "--micro-batch", m))
        res[m] = torch.load(d / f"EuroSAT_{canonical_method_name('direct_residual')}_transported_native.pt")
    for k in res["0"]:
        torch.testing.assert_close(res["4"][k], res["0"][k], rtol=1e-4, atol=1e-6)
    out = cc.run(_args(tmp_path / "cc", "--lr", "1e-2", "--patience", "2", "--trainable", "lora_cproj"))
    assert out["trainable"]["trainable_mode"] == "lora_cproj" and out["best_epoch"] is not None
    # parser default unchanged
    assert _args(tmp_path).trainable == "cproj"
