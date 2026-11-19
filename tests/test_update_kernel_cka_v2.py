"""update_kernel_cka v2: class-stratified chunks, cross-class image null, mean-shift agreement with CIs, the cross-task
matching aggregator and the merge of the new fields. Synthetic, CPU, small."""

from __future__ import annotations

import json
import math

import pytest
import torch

from merge_and_rebase.eval.diagnostics import update_kernel_cka as uk
from merge_and_rebase.eval.diagnostics import update_mean_matching as mm

DT = torch.float64


# ----------------------------------------------------------------------------------------------------------------
# sampling design
# ----------------------------------------------------------------------------------------------------------------


def _imbalanced_labels(n, n_classes, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.arange(1, n_classes + 1, dtype=torch.float64)
    p = p / p.sum()
    lab = torch.multinomial(p, n, replacement=True, generator=g)
    return lab.sort().values.tolist()  # class-sorted, as the old sorted-index selection of a class-sorted dataset


@pytest.mark.parametrize("n,bs,g", [(1024, 32, 16), (1000, 32, 16), (130, 16, 4)])
def test_stratified_chunks_have_the_class_proportions_and_cover_the_sample_once(n, bs, g):
    labels = _imbalanced_labels(n, 7, seed=n)
    sizes = [bs] * (n // bs) + ([n % bs] if n % bs else [])
    order, cid = uk.stratified_chunk_order(labels, sizes, g, seed=5)
    assert sorted(order) == list(range(n))  # every image exactly once
    counts = uk.chunk_image_counts(sizes, g)
    assert [cid.count(c) for c in range(len(counts))] == counts
    assert cid == sorted(cid)  # chunks are consecutive: they coincide with the chunk_of_batch groups of batches
    off = 0
    for b, size in enumerate(sizes):
        assert set(cid[off:off + size]) == {uk.chunk_of_batch(b, len(sizes), g)}
        off += size
    lab = torch.tensor([labels[i] for i in order])
    cidt = torch.tensor(cid)
    tol = 1.0 if len(set(counts)) == 1 else 2.0
    for c in torch.unique(lab):
        total = int((lab == c).sum())
        for k, size in enumerate(counts):
            got = int(((lab == c) & (cidt == k)).sum())
            assert abs(got - total * size / n) <= tol + 1e-9, (int(c), k, got, total * size / n)
    bal = uk.chunk_class_balance([labels[i] for i in order], cid)
    assert bal["max_abs_dev_from_proportional"] <= tol + 1e-9 and bal["mean_classes_per_chunk"] == 7


def test_stratified_order_is_seeded_not_class_sorted_and_the_sorted_baseline_is_far_from_stratified():
    labels = _imbalanced_labels(512, 8, seed=1)
    sizes = [32] * 16
    a = uk.stratified_chunk_order(labels, sizes, 8, seed=3)
    assert a == uk.stratified_chunk_order(labels, sizes, 8, seed=3)  # reproducible
    assert a[0] != uk.stratified_chunk_order(labels, sizes, 8, seed=4)[0]
    lab = [labels[i] for i in a[0]]
    same_adjacent = sum(x == y for x, y in zip(lab, lab[1:], strict=False)) / (len(lab) - 1)
    sorted_adjacent = sum(x == y for x, y in zip(labels, labels[1:], strict=False)) / (len(labels) - 1)
    assert sorted_adjacent > 0.85 and same_adjacent < 0.35  # shuffled inside the chunks, not in class order
    per_batch = [len(set(lab[i:i + 32])) for i in range(0, 512, 32)]
    assert min(per_batch) >= 4  # every batch is mixed (the sorted baseline has 1-3 classes per batch)
    # the sorted baseline: chunk of consecutive batches of the class-sorted list is far from the proportions
    base = uk.chunk_class_balance(labels, [uk.chunk_of_batch(b, 16, 8) for b in range(16) for _ in range(32)])
    new = uk.chunk_class_balance(lab, a[1])
    assert base["max_abs_dev_from_proportional"] > 10 * new["max_abs_dev_from_proportional"]
    assert base["mean_classes_per_chunk"] < new["mean_classes_per_chunk"]


def test_class_derangement_pairs_every_image_with_another_class():
    for seed in range(40):
        g = torch.Generator().manual_seed(seed)
        lab = torch.randint(0, 4, (32,), generator=g)
        sg = uk.class_derangement(lab, g)
        assert sorted(sg.tolist()) == list(range(32))
        assert int((lab[sg] == lab).sum()) == 0  # no same-label pair (also no fixed point)
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    lab = torch.randint(0, 5, (24,), generator=torch.Generator().manual_seed(2))
    assert torch.equal(uk.class_derangement(lab, g1), uk.class_derangement(lab, g2))  # seeded
    # two classes of equal size: the only solution swaps the classes
    lab2 = torch.tensor([0, 1] * 8)
    sg = uk.class_derangement(lab2, torch.Generator().manual_seed(1))
    assert int((lab2[sg] == lab2).sum()) == 0
    # one class: a plain derangement; more than n/2 in one class: the minimum number of same-class pairs, still a permutation
    one = uk.class_derangement(torch.zeros(6, dtype=torch.long), torch.Generator().manual_seed(1))
    assert sorted(one.tolist()) == list(range(6)) and not (one == torch.arange(6)).any()
    skew = torch.tensor([0] * 7 + [1] * 3)
    sg = uk.class_derangement(skew, torch.Generator().manual_seed(1))
    assert sorted(sg.tolist()) == list(range(10)) and int((skew[sg] == skew).sum()) == 2 * 7 - 10


def test_old_image_derangement_is_unchanged():
    sg = uk.image_derangement(10, torch.Generator().manual_seed(0))
    assert sorted(sg.tolist()) == list(range(10)) and not (sg == torch.arange(10)).any()


def test_sampling_option_defaults_to_the_old_behaviour():
    spec = {"model": "m", "pretrained": "p", "experts": {"T": "x"}}
    cfg = uk.parse_config({"source": spec, "target": spec, "task": "T", "state_dir": "s", "out_dir": "o", "controls": ["null"]})
    assert cfg.sampling == "legacy" and cfg.mean_shift is True
    assert uk.parse_config({"source": spec, "target": spec, "task": "T", "state_dir": "s", "out_dir": "o",
                            "controls": ["null"], "sampling": "stratified"}).sampling == "stratified"
    with pytest.raises(ValueError, match="sampling"):
        uk.parse_config({"source": spec, "target": spec, "task": "T", "state_dir": "s", "out_dir": "o", "controls": ["null"], "sampling": "x"})


# ----------------------------------------------------------------------------------------------------------------
# mean-shift agreement through the real streaming pass (fake linear models)
# ----------------------------------------------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, w):
        self.w = w

    def load_state_dict(self, sd, strict=True):
        self.w = sd["w"]


def _fake_capture(model, batches, requests, device, *, family_adapter=None, store_device="cpu"):
    for x, _ in batches:
        yield {key: x @ model.w for key in requests}


def _pass(monkeypatch, *, null_across_classes, n_img=24, bs=6, d=5, t=16, n_tok=5, seed=3, with_other=True):
    from merge_and_rebase.rebase.methods._ariadne import capture

    monkeypatch.setattr(capture, "iter_capture_tokens", _fake_capture)
    g = torch.Generator().manual_seed(seed)
    eye = torch.eye(d, dtype=DT)
    x = torch.randn(n_img, n_tok, d, generator=g, dtype=DT) + 0.5
    lab = torch.randint(0, 3, (n_img,), generator=g)
    batches = [(x[i:i + bs], lab[i:i + bs]) for i in range(0, n_img, bs)]
    w = {tag: eye + 0.4 * torch.randn(d, d, generator=g, dtype=DT) for tag in ("m_t", "k_t", "m_o", "k_o")}
    if not with_other:
        w = {tag: w[tag] for tag in ("m_t", "k_t")}
    p_m, p_k = torch.randn(t, d, generator=g, dtype=DT), torch.randn(t, d, generator=g, dtype=DT)
    frames = {0: uk.PairFrame(0, 0, p_m, p_k, 1.0, 1.0, torch.arange(t) // 2)}
    ctx = {"device": "cpu", "dtype": DT, "m_batches": batches, "k_batches": batches, "m_base": _FakeModel(eye),
           "k_base": _FakeModel(eye), "m_work": _FakeModel(eye), "m_work2": _FakeModel(eye), "k_work": _FakeModel(eye),
           "k_work2": _FakeModel(eye)}
    col = uk.Collector.empty([0])
    pairs = [("true", "m_t", "k_t")]
    if with_other:
        pairs += [("other_task", "m_t", "k_o"), ("other_task_rev", "m_o", "k_t")]
    uk._stream_pass(
        name="t", ctx=ctx, col=col, states={tag: {"w": wt} for tag, wt in w.items()}, frames=frames, n_ref=n_tok,
        row_types=("cls", "patch"), native_tags=list(w), side_tags=list(w), pairs=pairs, collect_base=True,
        chunk_fn=lambda b: b // 2, n_perm_images=3, seed=0, collect_means=True, null_across_classes=null_across_classes,
    )
    return SimpleNamespaceX(col=col, x=x, lab=lab, w=w, p_m=p_m, p_k=p_k, eye=eye, frames=frames, bs=bs)


class SimpleNamespaceX:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _direct_cos(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def test_mean_shift_matches_a_direct_computation_and_its_jackknife(monkeypatch):
    r = _pass(monkeypatch, null_across_classes=False)
    parts = uk.mean_vectors(r.col.means[0])
    res = uk.mean_shift_block(parts, ci=True)
    g = res["n_chunks"]
    assert g == 2 and res["images_per_chunk"] == [12.0, 12.0]  # 4 batches of 6, 2 per chunk

    def enc(tag, kind, rows=slice(None)):
        h = r.x[rows] @ (r.w[tag] - r.eye)  # update
        base = r.x[rows]
        e = {"displacement": h, "base_position": base, "ft_position": base + h}[kind]
        return e @ (r.p_m if tag[0] == "m" else r.p_k).T  # [B, N, T]

    def vec(tag, kind, group, rows):
        e = enc(tag, kind, rows)
        return {"cls": e[:, 0].sum(0), "patch": e[:, 1:].sum((0, 1)), "all": e.sum((0, 1))}[group]

    allrows = slice(None)
    halves = [slice(0, 12), slice(12, 24)]
    for kind in uk.MS_KINDS:
        for group in uk.MS_GROUPS:
            want = _direct_cos(vec("m_t", kind, group, allrows), vec("k_t", kind, group, allrows))
            assert res[kind][group]["value"] == pytest.approx(want, abs=1e-10), (kind, group)
            loo = [_direct_cos(vec("m_t", kind, group, halves[1 - d]), vec("k_t", kind, group, halves[1 - d])) for d in range(g)]
            ref = uk.jackknife_ci(want, loo)
            for key in ("lo", "hi", "se"):
                assert res[kind][group][key] == pytest.approx(ref[key], abs=1e-9)
            if kind == "base_position":
                assert "other_task" not in res[kind][group]  # the base is the same: no comparator
                continue
            other = _direct_cos(vec("m_t", kind, group, allrows), vec("k_o", kind, group, allrows))
            rev = _direct_cos(vec("m_o", kind, group, allrows), vec("k_t", kind, group, allrows))
            e = res[kind][group]
            assert e["other_task"]["value"] == pytest.approx(other, abs=1e-10)
            assert e["other_task_rev"]["value"] == pytest.approx(rev, abs=1e-10)
            assert e["other_task_mean"]["value"] == pytest.approx((other + rev) / 2, abs=1e-10)
            assert e["excess_other_task_mean"]["value"] == pytest.approx(want - (other + rev) / 2, abs=1e-10)
            loo_ex = [
                _direct_cos(vec("m_t", kind, group, halves[1 - d]), vec("k_t", kind, group, halves[1 - d]))
                - (_direct_cos(vec("m_t", kind, group, halves[1 - d]), vec("k_o", kind, group, halves[1 - d]))
                   + _direct_cos(vec("m_o", kind, group, halves[1 - d]), vec("k_t", kind, group, halves[1 - d]))) / 2
                for d in range(g)
            ]
            ref_ex = uk.jackknife_ci(want - (other + rev) / 2, loo_ex)
            assert e["excess_other_task_mean"]["lo"] == pytest.approx(ref_ex["lo"], abs=1e-9)
            assert e["excess_other_task_mean"]["hi"] == pytest.approx(ref_ex["hi"], abs=1e-9)
    # the dataset mean-shift of the 'all' group equals the existing common-mode cosine (same grids: native tokens = matched)
    cm = uk.common_mode(r.frames[0], r.col.native[0])
    assert res["displacement"]["all"]["value"] == pytest.approx(cm["cos_mean_true_task"], abs=1e-10)
    assert res["displacement"]["all"]["other_task"]["value"] == pytest.approx(cm["cos_mean_other_task"], abs=1e-10)
    # ci off: values only
    assert "lo" not in uk.mean_shift_block(parts, ci=False)["displacement"]["all"]


def test_cross_class_null_has_no_same_label_pairs_and_the_default_counts_nothing(monkeypatch):
    on = _pass(monkeypatch, null_across_classes=True)
    assert on.col.null_pairs[1] == 24 * 3 and on.col.null_pairs[0] == 0
    off = _pass(monkeypatch, null_across_classes=False)
    assert off.col.null_pairs == [0, 0]
    # the default pairing is the old one (any other image): identical statistics to the pre-change code path
    again = _pass(monkeypatch, null_across_classes=False)
    assert torch.equal(off.col.perm[0].c, again.col.perm[0].c)
    assert not torch.equal(off.col.perm[0].c, on.col.perm[0].c)


def test_mean_vectors_are_saved_float32_per_chunk_and_small(monkeypatch):
    r = _pass(monkeypatch, null_across_classes=False)
    parts = uk.mean_vectors(r.col.means[0])
    assert set(parts) == {"m_t", "k_t", "m_o", "k_o", "base_m", "base_k"}
    assert parts["m_t"]["cls"].shape == (2, 16) and parts["m_t"]["n"].tolist() == [12.0, 12.0]
    e = (r.x @ (r.w["m_t"] - r.eye)) @ r.p_m.T
    assert torch.allclose(parts["m_t"]["patch"].sum(0), e[:, 1:].sum((0, 1)), atol=1e-9)
    assert torch.allclose(parts["base_k"]["cls"].sum(0), (r.x @ r.p_k.T)[:, 0].sum(0), atol=1e-9)
    # size: 8 (T-vector) tags x 2 groups x chunks x T x 4 bytes x blocks, for the real runs (16 chunks, T <= 2048, 24 blocks)
    assert 7 * 2 * 16 * 2048 * 4 * 24 < 0.1e9


# ----------------------------------------------------------------------------------------------------------------
# matching aggregator
# ----------------------------------------------------------------------------------------------------------------


def _planted(n=8, g=6, t=40, noise=0.3, seed=0, planted=True):
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(n, t, generator=gen, dtype=DT) + 2.0 * torch.randn(1, t, generator=gen, dtype=DT)
    vm = u[:, None] + noise * torch.randn(n, g, t, generator=gen, dtype=DT)
    src = u if planted else torch.randn(n, t, generator=gen, dtype=DT)
    vk = src[:, None] + noise * torch.randn(n, g, t, generator=gen, dtype=DT)
    return vm, vk


def test_matching_planted_diagonal_gives_accuracy_one_and_a_tiny_exact_p():
    vm, vk = _planted()
    tasks = [f"t{i}" for i in range(8)]
    out = mm.block_matching(vm, vk, tasks)
    assert out["top1_row"]["observed"] == 1.0 and out["top1_col"]["observed"] == 1.0
    assert out["top1_row"]["n_correct"] == 8 and out["top1_row"]["chance"] == 0.125
    assert out["top1_row"]["n_matchings"] == math.factorial(8)
    assert out["top1_row"]["p"] == pytest.approx(1 / math.factorial(8))  # only the identity reaches 8/8
    assert out["top1_row"]["null_mean"] == pytest.approx(1 / 8)  # exact: E over matchings = 1/n
    assert out["mean_row_excess"]["p"] == pytest.approx(1 / math.factorial(8))
    assert out["mean_row_excess"]["lo"] > 0 and out["diag_mean"] > out["offdiag_mean"]
    # the matrix and the excess against a direct computation
    c = torch.zeros(8, 8, dtype=DT)
    for i in range(8):
        for j in range(8):
            a, b = vm[i].sum(0), vk[j].sum(0)
            a, b = a - a.mean(), b - b.mean()
            c[i, j] = a @ b / (a.norm() * b.norm())
    assert torch.allclose(torch.tensor(out["C"], dtype=DT), c, atol=1e-12)
    row_ex = c.diagonal() - (c.sum(1) - c.diagonal()) / 7
    assert out["row_excess"]["t3"]["value"] == pytest.approx(float(row_ex[3]), abs=1e-12)
    # jackknife of one row: delete chunk d from all tasks, recompute, formula of jackknife_ci
    loo = []
    for d in range(6):
        keep = [x for x in range(6) if x != d]
        cc = torch.zeros(8, 8, dtype=DT)
        for i in range(8):
            for j in range(8):
                a, b = vm[i][keep].sum(0), vk[j][keep].sum(0)
                a, b = a - a.mean(), b - b.mean()
                cc[i, j] = a @ b / (a.norm() * b.norm())
        loo.append(float(cc[3, 3] - (cc[3].sum() - cc[3, 3]) / 7))
    ref = uk.jackknife_ci(float(row_ex[3]), loo)
    assert out["row_excess"]["t3"]["lo"] == pytest.approx(ref["lo"], abs=1e-10)
    assert out["row_excess"]["t3"]["hi"] == pytest.approx(ref["hi"], abs=1e-10)


def test_matching_shuffled_vectors_are_at_chance():
    accs, ps, exs = [], [], []
    for seed in range(60):
        vm, vk = _planted(seed=100 + seed, planted=False)
        out = mm.block_matching(vm, vk, [f"t{i}" for i in range(8)], ci=False)
        accs.append(out["top1_row"]["observed"])
        ps.append(out["top1_row"]["p"])
        exs.append(out["mean_row_excess"]["observed"])
    assert abs(sum(accs) / len(accs) - 0.125) < 0.05  # chance level
    assert abs(sum(exs) / len(exs)) < 0.1
    assert sum(p < 0.05 for p in ps) <= 8  # the exact null is calibrated (about 5 % expected, loose bound)
    # permuting the task labels of k breaks a planted diagonal
    vm, vk = _planted(seed=7)
    perm = torch.tensor([3, 0, 1, 2, 7, 4, 5, 6])
    out = mm.block_matching(vm, vk[perm], [f"t{i}" for i in range(8)], ci=False)
    assert out["top1_row"]["observed"] == 0.0 and out["top1_row"]["p"] > 0.5


def test_matching_with_more_than_eight_tasks_uses_seeded_random_matchings():
    p = mm.matchings(9, seed=1)
    assert p.shape == (mm.N_PERM_RANDOM + 1, 9) and torch.equal(p[0], torch.arange(9))
    assert torch.equal(p, mm.matchings(9, seed=1)) and not torch.equal(p, mm.matchings(9, seed=2))
    vm, vk = _planted(n=9)
    out = mm.block_matching(vm, vk, [f"t{i}" for i in range(9)], ci=False)
    assert out["top1_row"]["observed"] == 1.0 and out["top1_row"]["p"] < 1e-3


def _fake_run(task, vm, vk, setting_extra=None, block=0):
    """A run dict as ``load_run`` returns it: m_t / k_t displacement vectors, base vectors, two chunks of the tasks."""
    g, t = vm.shape
    zeros = torch.zeros(g, t, dtype=torch.float32)
    tags = {
        "m_t": {"cls": vm.float(), "patch": (0.5 * vm).float()}, "k_t": {"cls": vk.float(), "patch": (0.5 * vk).float()},
        "base_m": {"cls": zeros + 1.0, "patch": zeros + 2.0}, "base_k": {"cls": zeros + 1.0, "patch": zeros + 2.0},
    }
    setting = {"source": {"model": "A", "pretrained": "x", "expert": {"path": f"a_{task}"}},
               "target": {"model": "B", "pretrained": "y", "expert": {"path": f"b_{task}"}},
               "images": {"task": {"n_images": 64, "image_indices_sha256": task}}, "state_dir": "s", "split": "test",
               "split_seed": 89, "batch_size": 16, "sampling": "stratified", "jackknife_chunks_actual": g, "T_stored": t,
               **(setting_extra or {})}
    return {"dir": f"/d/{task}", "setting": setting, "meta": {"task": task, "n_chunks": g, "images_per_chunk": [16.0] * g},
            "blocks": {block: {"tags": tags, "n_images": torch.full((g,), 16.0), "n_anchors": t, "source_block": block}}}


def test_matching_runs_and_checks():
    vm, vk = _planted(n=4, g=4, t=24)
    tasks = ["a", "b", "c", "d"]
    runs = [_fake_run(tk, vm[i], vk[i]) for i, tk in enumerate(tasks)]
    res = mm.matching_runs(runs)
    assert res["setting"]["tasks"] == tasks and len(res["blocks"]) == 1
    e = res["blocks"][0]["kinds"]["displacement"]["all"]
    assert e["top1_row"]["observed"] == 1.0 and e["top1_row"]["n_matchings"] == 24
    # base position vectors are constant across tasks in this fake: C is constant (cosines of equal vectors / NaN centred)
    assert "base_position" in res["blocks"][0]["kinds"] and "ft_position" in res["blocks"][0]["kinds"]
    bad = [dict(r) for r in runs]
    bad[1] = _fake_run("b", vm[1], vk[1], {"state_dir": "other"})
    with pytest.raises(ValueError, match="state_dir"):
        mm.matching_runs(bad)
    with pytest.raises(ValueError, match="duplicate"):
        mm.matching_runs([runs[0], runs[0], runs[1]])
    with pytest.raises(ValueError, match="at least 2"):
        mm.matching_runs(runs[:1])
    assert mm.matching_runs(runs[:2])["blocks"][0]["kinds"]["displacement"]["all"]["top1_row"]["n_matchings"] == 2


def test_matching_cli_writes_json_and_md(tmp_path):
    vm, vk = _planted(n=4, g=4, t=24)
    dirs = []
    for i, tk in enumerate(["a", "b", "c", "d"]):
        run = _fake_run(tk, vm[i], vk[i])
        d = tmp_path / tk
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"setting": run["setting"], "blocks": []}))
        torch.save({"meta": run["meta"], "blocks": run["blocks"]}, d / uk.MEAN_VECTORS_FILE)
        dirs.append(str(d))
    out = tmp_path / "match"
    uk.main(["--matching", *dirs, "--out", str(out)])
    res = json.loads((out / "matching.json").read_text())
    md = (out / "matching.md").read_text()
    assert res["setting"]["tasks"] == ["a", "b", "c", "d"] and res["setting"]["sampling"] == "stratified"
    for token in ("## Procedura", "### displacement (PRIMARIA: spostamento medio dell'update dH), gruppo all", "eccesso medio [IC]", "task-centrato", "Matrice C task-centrata", "SECONDARIA", "riferimento SOLO-IMMAGINI", "Matrice C", "top-1"):
        assert token in md
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "summary.json").write_text(json.dumps({"setting": run["setting"], "blocks": []}))
    with pytest.raises(FileNotFoundError, match="mean-shift vectors"):
        mm.load_run(tmp_path / "x")


# ----------------------------------------------------------------------------------------------------------------
# merge and outputs with the new fields
# ----------------------------------------------------------------------------------------------------------------


def _ms_block(j, seed):
    g = torch.Generator().manual_seed(seed)
    parts = {tag: {"cls": torch.randn(4, 12, generator=g, dtype=DT), "patch": torch.randn(4, 12, generator=g, dtype=DT),
                   "n": torch.full((4,), 8.0, dtype=DT)} for tag in ("m_t", "k_t", "m_o", "k_o", "base_m", "base_k")}
    return parts, {"target_block": j, "source_block": j, "mean_shift": uk.mean_shift_block(parts, ci=True)}


def _setting(blocks_run, tag, preprocess="<fn at 0x7f00aa>"):
    return {"task": "T", "images": {"task": {"n_images": 32, "preprocess": preprocess}}, "sampling": "stratified",
            "mean_vectors_file": uk.MEAN_VECTORS_FILE, "image_null_pairs": {"same_label": 0, "total": 10},
            "blocks_run": blocks_run, "config": {"tag": tag}, "seconds": 1.0}


def test_merge_handles_mean_shift_fields_vectors_and_addresses(tmp_path):
    p0, b0 = _ms_block(0, 1)
    p1, b1 = _ms_block(1, 2)
    merged = uk.merge_summaries([
        {"setting": _setting([0], "a", "<fn at 0x1>"), "blocks": [b0]},
        {"setting": _setting([1], "b", "<fn at 0x2>"), "blocks": [b1]},  # only the object address differs
    ])
    assert [b["target_block"] for b in merged["blocks"]] == [0, 1]
    assert merged["blocks"][1]["mean_shift"]["displacement"]["all"] == b1["mean_shift"]["displacement"]["all"]
    with pytest.raises(ValueError, match="different setting"):
        uk.merge_summaries([{"setting": _setting([0], "a"), "blocks": [b0]},
                            {"setting": {**_setting([1], "b"), "sampling": "legacy"}, "blocks": [b1]}])
    # vector files: union of the blocks, meta must agree, no overlap
    meta = {"task": "T", "n_chunks": 4}
    f0 = {"meta": meta, "blocks": {0: {"tags": {}}}}
    f1 = {"meta": meta, "blocks": {1: {"tags": {}}}}
    assert list(uk.merge_vector_files([f1, f0])["blocks"]) == [0, 1]
    with pytest.raises(ValueError, match="overlapping"):
        uk.merge_vector_files([f0, f0])
    with pytest.raises(ValueError, match="meta differs"):
        uk.merge_vector_files([f0, {"meta": {**meta, "task": "U"}, "blocks": {1: {}}}])
    # merge_dirs end to end: summary, image selection and vectors of both halves
    dirs = []
    for j, (parts, blk) in enumerate(((p0, b0), (p1, b1))):
        d = tmp_path / f"part{j}"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"setting": {**_setting([j], str(j)), "T_stored": 12, "pairing": [0, 1],
                                                              "source": {"model": "A", "pretrained": "x", "n_tokens": 5, "expert": {"path": "a"}},
                                                              "target": {"model": "B", "pretrained": "y", "n_tokens": 5, "expert": {"path": "b"}},
                                                              "state_dir": "s", "anchor_filter": "norm", "anchor_filter_k": 3.0,
                                                              "split": "test", "split_seed": 1, "batch_size": 8, "grid_mode": "direct",
                                                              "debiased": True, "compute_dtype": "float64", "controls": [],
                                                              "permutations": 0, "image_permutations": 0, "control_seed": 0,
                                                              "tf32": {}, "code": {}}, "blocks": [blk]}, default=str))
        (d / uk.SELECTED_IMAGES_FILE).write_text(json.dumps({"hf_indices": [3, 1, 2]}))
        torch.save({"meta": meta, "blocks": {j: {"tags": {t: {"cls": v["cls"].float(), "patch": v["patch"].float()} for t, v in parts.items()}}}},
                   d / uk.MEAN_VECTORS_FILE)
        dirs.append(d)
    out = tmp_path / "merged"
    uk.merge_dirs(dirs, out)
    mv = torch.load(out / uk.MEAN_VECTORS_FILE, weights_only=False)
    assert sorted(mv["blocks"]) == [0, 1] and (out / uk.SELECTED_IMAGES_FILE).exists()
    assert len(json.loads((out / "summary.json").read_text())["blocks"]) == 2
    (dirs[1] / uk.SELECTED_IMAGES_FILE).write_text(json.dumps({"hf_indices": [9]}))
    with pytest.raises(ValueError, match="different image selections"):
        uk.merge_dirs(dirs, tmp_path / "merged2")


def test_md_and_csv_show_the_primary_side_by_side_table_and_the_mean_shift_rows(monkeypatch, tmp_path):
    r = _pass(monkeypatch, null_across_classes=True)
    res = uk.finalize_pair([], r.frames[0], uk.FinalizeOptions(debiased=True, ci=True, null=True, n_perm=3),
                           direct=r.col.direct_block(0))
    res["mean_shift"] = uk.mean_shift_block(uk.mean_vectors(r.col.means[0]), ci=True)
    res.update({"target_block": 0, "source_block": 0})
    setting = {
        "source": {"model": "A", "pretrained": "x", "n_tokens": 5, "expert": {"path": "a"}},
        "target": {"model": "B", "pretrained": "y", "n_tokens": 5, "expert": {"path": "b"}},
        "task": "T", "state_dir": "s", "T_stored": 16, "anchor_filter": "norm", "anchor_filter_k": 3.0, "pairing": [0],
        "split": "test", "sampling": "stratified", "selected_images_file": "selected_images.json", "mean_vectors_file": "mean_vectors.pt",
        "images": {"task": {"n_images": 24, "chunk_class_balance": {"max_abs_dev_from_proportional": 0.8, "n_classes": 3}}},
        "image_null_pairs": {"same_label": 0, "total": 72}, "split_seed": 1, "batch_size": 6, "grid_mode": "direct",
        "debiased": True, "compute_dtype": "float64", "controls": ["null"], "permutations": 0, "image_permutations": 3,
        "control_seed": 0, "code": {}, "tf32": {},
    }
    uk.write_outputs({"setting": setting, "blocks": [res]}, tmp_path)
    md = (tmp_path / "summary.md").read_text()
    assert "PRIMARIA affiancata (gruppo all)" in md and "null immagini (uncentered)" in md and "mean-shift spostamento [IC]" in md
    assert md.index("PRIMARIA affiancata") < md.index("PRIMARIA: coseno grezzo")
    assert "Campionamento `stratified`" in md and "STRATIFICATI" in md and "0 su 72" in md
    assert "Mean-shift, spostamento medio" in md and "Mean-shift, posizione media del fine-tuned" in md
    assert "derangement fra classi diverse" in md
    csv_text = (tmp_path / "blocks.csv").read_text().splitlines()
    head = csv_text[0].split(",")
    ms_rows = [row.split(",") for row in csv_text[1:] if row.split(",")[2].startswith("meanshift_")]
    assert {row[2] for row in ms_rows} == {f"meanshift_{k}" for k in uk.MS_KINDS} and len(ms_rows) == 9
    row = next(x for x in ms_rows if x[2] == "meanshift_displacement" and x[3] == "all")
    assert float(row[head.index("value")]) == pytest.approx(res["mean_shift"]["displacement"]["all"]["value"])
    # an old summary without mean-shift / sampling still renders (legacy text)
    old = {k: v for k, v in res.items() if k != "mean_shift"}
    uk.write_outputs({"setting": {k: v for k, v in setting.items() if k not in ("sampling", "image_null_pairs")}, "blocks": [old]},
                     tmp_path / "old")
    assert "Campionamento `legacy`" in (tmp_path / "old" / "summary.md").read_text()


# ----------------------------------------------------------------------------------------------------------------
# v2 configs
# ----------------------------------------------------------------------------------------------------------------


def test_v2_configs_cover_all_tasks_pairs_halves_and_use_the_new_design():
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "configs/examples/update_kernel_cka_v2"
    tasks = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
    cfgs = {p.stem: uk.parse_config(json.loads(p.read_text())) for p in sorted(d.glob("*.json"))}
    want = {f"{p}_{t.lower()}" for p in ("A_C", "B_C") for t in tasks}
    want |= {f"A_B_{t.lower()}_part{h}" for t in tasks for h in (1, 2)}
    assert set(cfgs) == want
    for name, c in cfgs.items():
        assert c.sampling == "stratified" and c.mean_shift and c.split == "test" and c.seed == 89 and c.num_images == 1024
        assert c.jackknife_chunks == 16 and c.random_draws == 19 and c.m2 is False and c.batch_size == 32
        assert c.num_images % (c.batch_size * c.jackknife_chunks) == 0
        assert c.out_dir == f"/data/merge_and_rebase_results/update_kernel_cka_v2/{name}"
        assert c.other_task == tasks[(tasks.index(c.task) + 1) % 8]  # fixed cyclic choice
        assert (c.source.model, c.source.pretrained) == (("ViT-B-16" if name.startswith(("A_", )) else "ViT-L-14"), "datacomp_xl_s13b_b90k")
        assert (c.target.model, c.target.pretrained) == (
            ("ViT-L-14", "datacomp_xl_s13b_b90k") if name.startswith("A_B") else ("ViT-B-16", "openai"))
        assert set(c.source.experts) == set(tasks) == set(c.target.experts)
        for p in (*c.source.experts.values(), *c.target.experts.values()):
            if Path("/data/merge_and_rebase_ckpts").exists():
                assert Path(p).exists(), p
    for t in tasks:
        p1, p2 = cfgs[f"A_B_{t.lower()}_part1"], cfgs[f"A_B_{t.lower()}_part2"]
        assert p1.blocks == tuple(range(12)) and p2.blocks == tuple(range(12, 24))
    assert cfgs["A_C_dtd"].blocks is None and cfgs["B_C_dtd"].blocks is None


# ----------------------------------------------------------------------------------------------------------------
# additions: native-space means, control / task-centred / pairings in the aggregator, reorder in _eval_batches
# ----------------------------------------------------------------------------------------------------------------


def test_native_space_means_are_accumulated_per_chunk(monkeypatch):
    r = _pass(monkeypatch, null_across_classes=False)
    nat = uk.mean_vectors(r.col.native_means[0])
    assert set(nat) == {"m_t", "k_t", "m_o", "k_o", "base_m", "base_k"}
    assert nat["m_t"]["cls"].shape == (2, 5) and nat["m_t"]["n"].tolist() == [12.0, 12.0]
    dh = r.x @ (r.w["k_o"] - r.eye)  # native update [N, tokens, d]
    assert torch.allclose(nat["k_o"]["cls"].sum(0), dh[:, 0].sum(0), atol=1e-9)
    assert torch.allclose(nat["k_o"]["patch"][0], dh[:12, 1:].sum((0, 1)), atol=1e-9)
    assert torch.allclose(nat["base_m"]["patch"].sum(0), r.x[:, 1:].sum((0, 1)), atol=1e-9)


def _planted_control(seed=0):
    vm, vk = _planted(seed=seed)
    gen = torch.Generator().manual_seed(seed + 99)
    vc = torch.randn(vm.shape, generator=gen, dtype=DT)
    return vm, vk, vc


def test_matching_control_task_centred_double_centred_and_pairings():
    vm, vk, vc = _planted_control()
    tasks = [f"t{i}" for i in range(8)]
    out = mm.block_matching(vm, vk, tasks, vc=vc, n_pairings=5)
    assert "mean_col_excess" not in out and out["task_centred"]["C"] and out["double_centred"]["C"]
    row = torch.tensor([out["row_excess"][t]["value"] for t in tasks], dtype=DT)
    col = torch.tensor([out["col_excess"][t]["value"] for t in tasks], dtype=DT)
    assert float(row.mean()) == pytest.approx(float(col.mean()), abs=1e-12)  # the same statistic
    # control: image-only matrix has no diagonal; excess - control paired jackknife against a direct computation
    ctl = out["control_random_update"]
    assert ctl["mean_row_excess"]["value"] == pytest.approx(0.0, abs=0.25)
    assert out["mean_row_excess"]["value"] - ctl["mean_row_excess"]["value"] == pytest.approx(ctl["excess_minus_control"]["value"])
    loo = []
    for d in range(6):
        keep = [x for x in range(6) if x != d]
        c1 = mm.corr_matrix(vm[:, keep].sum(1), vk[:, keep].sum(1))
        c2 = mm.corr_matrix(vm[:, keep].sum(1), vc[:, keep].sum(1))
        loo.append(float(mm.diag_excess(c1)[0].mean() - mm.diag_excess(c2)[0].mean()))
    ref = uk.jackknife_ci(ctl["excess_minus_control"]["value"], loo)
    assert ctl["excess_minus_control"]["lo"] == pytest.approx(ref["lo"], abs=1e-10)
    assert ctl["excess_minus_control"]["hi"] == pytest.approx(ref["hi"], abs=1e-10)
    # task-centred: direct computation (mean vectors over tasks removed after the anchor centring)
    def cen(x):
        x = x / 1.0
        x = x - x.mean(1, keepdim=True)
        x = x - x.mean(0, keepdim=True)
        return x / x.norm(dim=1, keepdim=True)
    sm, sk = vm.sum(1), vk.sum(1)
    want = cen(sm) @ cen(sk).T
    assert torch.allclose(torch.tensor(out["task_centred"]["C"], dtype=DT), want, atol=1e-12)
    # double-centred: rows and columns of C average to zero
    dc = torch.tensor(out["double_centred"]["C"], dtype=DT)
    assert torch.allclose(dc.mean(0), torch.zeros(8, dtype=DT), atol=1e-12) and torch.allclose(dc.mean(1), torch.zeros(8, dtype=DT), atol=1e-12)
    # SE over random pairings: the point estimate is invariant, the SE has a spread
    sp = out["se_pairings"]
    assert sp["n_pairings"] == 5 and sp["min"] <= sp["mean"] <= sp["max"] and sp["max"] > 0
    assert out["task_centred"]["se_pairings"]["n_pairings"] == 5
    # no control without vc
    assert "control_random_update" not in mm.block_matching(vm, vk, tasks, ci=False)
    # counts: unequal image counts change the task-centred matrix only through the per-task means
    cnt = torch.ones(8, 6, dtype=DT) * torch.arange(1, 9, dtype=DT)[:, None]
    a = mm.block_matching(vm * cnt[:, :, None], vk * cnt[:, :, None], tasks, counts=cnt, ci=False)
    b = mm.block_matching(vm, vk, tasks, ci=False)
    assert torch.allclose(torch.tensor(a["task_centred"]["C"], dtype=DT), torch.tensor(b["task_centred"]["C"], dtype=DT), atol=1e-10)


def test_matching_runs_use_saved_control_vectors():
    vm, vk, vc = _planted_control(seed=3)
    tasks = ["a", "b", "c", "d"]
    runs = []
    for i, tk in enumerate(tasks):
        run = _fake_run(tk, vm[i][:4, :24], vk[i][:4, :24])
        run["blocks"][0]["tags"]["k_rand"] = {"cls": vc[i][:4, :24].float(), "patch": (0.5 * vc[i][:4, :24]).float()}
        runs.append(run)
    res = mm.matching_runs(runs)
    assert "control_random_update" in res["blocks"][0]["kinds"]["displacement"]["all"]
    assert "control_random_update" not in res["blocks"][0]["kinds"]["base_position"]["all"]


def _fake_loader_env(monkeypatch, n_total=200, n_classes=5):
    """A fake HF dataset whose image ``i`` is the constant tensor ``i`` and whose label is sorted by index (class-ordered)."""
    from types import SimpleNamespace

    from torch.utils.data import Dataset

    from merge_and_rebase.data import vision_loaders
    from merge_and_rebase.eval.datasets import vision8_14_20

    class DS(Dataset):
        def __init__(self):
            self.split = [{"image": torch.full((2, 2), float(i)), "id": i} for i in range(n_total)]
            self.image_key = "image"

        def __len__(self):
            return n_total

        def __getitem__(self, i):
            return torch.full((3, 2, 2), float(i)), i * n_classes // n_total

    monkeypatch.setattr(vision_loaders, "load_hf_splits", lambda *a, **k: {"test": None})
    monkeypatch.setattr(vision_loaders, "build_vision_loaders", lambda **k: SimpleNamespace(test=SimpleNamespace(dataset=DS()),
                                                                                              val=SimpleNamespace(dataset=DS())))
    monkeypatch.setitem(vision8_14_20.SUITES, "vision8", SimpleNamespace(resolver=lambda task: ("p", None, {"test": "test"})))


def test_eval_batches_reorder_keeps_the_set_labels_and_batch_sizes(monkeypatch):
    _fake_loader_env(monkeypatch)
    spec = {"model": "m", "pretrained": "p", "experts": {"T": "x"}}
    base = {"source": spec, "target": spec, "task": "T", "state_dir": "s", "out_dir": "o", "controls": ["null"],
            "num_images": 100, "batch_size": 16, "jackknife_chunks": 4, "seed": 5}
    clf = __import__("types").SimpleNamespace(preprocess="pp")
    outs = {}
    for mode in ("legacy", "stratified"):
        cfg = uk.parse_config({**base, "sampling": mode})
        batches, meta = uk._eval_batches(cfg, "T", clf)
        outs[mode] = (batches, meta)
        assert [x.shape[0] for x, _ in batches] == [16] * 6 + [4]
        sel = meta["selection"]
        vals = torch.cat([x[:, 0, 0, 0] for x, _ in batches]).long().tolist()  # image ids as loaded
        assert vals == sel["dataset_positions"] == sel["hf_indices"]  # the record follows the images
        labels = torch.cat([y for _, y in batches]).tolist()
        assert labels == sel["labels"] == [v * 5 // 200 for v in vals]  # labels travel with their images
        assert len(sel["chunk_ids"]) == 100 and meta["n_chunks"] == 4
    assert sorted(outs["legacy"][1]["selection"]["dataset_positions"]) == sorted(outs["stratified"][1]["selection"]["dataset_positions"])
    assert outs["legacy"][1]["selection"]["dataset_positions"] == sorted(outs["legacy"][1]["selection"]["dataset_positions"])
    assert outs["stratified"][1]["selection"]["dataset_positions"] != outs["legacy"][1]["selection"]["dataset_positions"]
    assert outs["stratified"][1]["chunk_class_balance"]["max_abs_dev_from_proportional"] <= 2.0
    assert outs["legacy"][1]["chunk_class_balance"]["max_abs_dev_from_proportional"] > 8
    # reproducible with the seed; different seed, different order; the image-set hash differs only through the order
    again, meta2 = uk._eval_batches(uk.parse_config({**base, "sampling": "stratified"}), "T", clf)
    assert meta2["selection"] == outs["stratified"][1]["selection"]
    other = uk._eval_batches(uk.parse_config({**base, "sampling": "stratified", "seed": 6}), "T", clf)[1]
    assert other["selection"]["dataset_positions"] != outs["stratified"][1]["selection"]["dataset_positions"]


def test_reorder_batches_helper():
    xs = [(torch.arange(i, i + 4).float()[:, None], torch.tensor([0, 0, 1, 1])) for i in (0, 4, 8)]
    b, idx, cid = uk.reorder_batches(xs, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 3, 1, stratified=True)
    assert [x.shape[0] for x, _ in b] == [4, 4, 4] and sorted(idx) == list(range(10, 22))
    assert [int(v) + 10 for v in torch.cat([x for x, _ in b])[:, 0]] == idx
    b0, idx0, _ = uk.reorder_batches(xs, list(range(12)), 3, 1, stratified=False)
    assert idx0 == list(range(12)) and torch.equal(b0[1][0], xs[1][0])
