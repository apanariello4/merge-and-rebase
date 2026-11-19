"""Metodo K Phase 1, LLM path: ragged strata, halves, nested k, unit alignment, nulls, r* rules, coverage (synthetic)."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.diagnostics import metodoK_phase1 as v1
from merge_and_rebase.eval.diagnostics import metodoK_phase1_llm as llm

F64 = torch.float64


class FakeFamily:
    """Just what ``text_units`` / ``pair_batch`` need from a family adapter."""

    def content_mask(self, batch):
        return batch["attention_mask"].bool()

    def extract_calibration_batch(self, batch):
        return {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}


def make_docs(n_docs: int, d_m: int, d_k: int, max_units: int, seed: int = 0):
    """Ragged synthetic documents: per document ``[n_i, d]`` rows of two correlated models."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(3, max_units + 1, (n_docs,), generator=g)
    a = torch.randn(3, d_m, generator=g, dtype=F64)
    b = torch.randn(3, d_k, generator=g, dtype=F64)
    docs = []
    for n in counts.tolist():
        z = torch.randn(n, 3, generator=g, dtype=F64)
        docs.append((z @ a + 0.5 * torch.randn(n, d_m, generator=g, dtype=F64),
                     z @ b + 0.5 * torch.randn(n, d_k, generator=g, dtype=F64)))
    return docs, counts


def pad_batch(docs, lo, hi):
    n_slots = max(docs[p][0].shape[0] for p in range(lo, hi))
    pm = torch.zeros(hi - lo, n_slots, docs[0][0].shape[1], dtype=F64)
    pk = torch.zeros(hi - lo, n_slots, docs[0][1].shape[1], dtype=F64)
    real = torch.zeros(hi - lo, n_slots, dtype=torch.bool)
    for i, p in enumerate(range(lo, hi)):
        n = docs[p][0].shape[0]
        pm[i, :n], pk[i, :n], real[i, :n] = docs[p][0], docs[p][1], True
    return pm, pk, real


def accumulate(docs, counts, grid, nulls, n_perm, seed=0, bs=2, keep=None):
    halves = v1.half_labels(grid, seed)
    acc = llm.RaggedStratifiedSigma(docs[0][0].shape[1], docs[0][1].shape[1], grid, n_perm, nulls, seed=seed)
    for b, lo in enumerate(range(0, grid.n_max, bs)):
        pos = torch.arange(lo, lo + bs)
        pm, pk, real = pad_batch(docs, lo, lo + bs)
        valid = real if keep is None else real & keep[lo:lo + bs, : real.shape[1]]
        level = v1.token_levels(llm.unit_ranks(pos, counts[lo:lo + bs], real.shape[1], seed), grid.ks)
        a = int(v1.image_stratum(pos[:1], grid))
        acc.update(a, pm, pk, valid, real, level, halves[pos], pos, counts[lo:lo + bs], b)
    return acc, halves


def direct_sums(docs, counts, grid, halves, ia, il, seed):
    n_doc, k = grid.n_images[ia], grid.ks[il]
    out = {h: [[], []] for h in (0, 1)}
    for p in range(n_doc):
        n = int(counts[p])
        ranks = llm.unit_ranks(torch.tensor([p]), torch.tensor([n]), n, seed)[0]
        sel = ranks < k
        out[int(halves[p])][0].append(docs[p][0][sel])
        out[int(halves[p])][1].append(docs[p][1][sel])
    res = {}
    for h in (0, 1):
        x, y = torch.cat(out[h][0]), torch.cat(out[h][1])
        res[h] = (x.T @ x, y.T @ y, x.T @ y, x.shape[0])
    return res


def test_ragged_strata_sums_equal_direct_sums():
    grid = v1.Grid((4, 8), (1, 3, 8))
    docs, counts = make_docs(8, 5, 4, 8)
    acc, halves = accumulate(docs, counts, grid, [], 0)
    for ia, il in grid.cells():
        sums, _, units = acc.cell(ia, il)
        ref = direct_sums(docs, counts, grid, halves, ia, il, 0)
        for h in (0, 1):
            assert torch.allclose(sums.mm[h], ref[h][0]) and torch.allclose(sums.kk[h], ref[h][1])
            assert torch.allclose(sums.mk[h], ref[h][2])
            assert int(sums.n[h]) == ref[h][3] == int(units[h])


def test_filtered_rows_are_dropped_and_counted():
    grid = v1.Grid((4, 8), (1, 8))
    docs, counts = make_docs(8, 3, 3, 8, seed=1)
    keep = torch.rand(8, 8, generator=torch.Generator().manual_seed(5)) > 0.3
    acc, _ = accumulate(docs, counts, grid, [], 0, keep=keep)
    sums, _, units = acc.cell(1, 1)
    assert int(sums.n.sum()) < int(units.sum()) == int(counts.sum())


def test_halves_disjoint_by_document_and_balanced_per_stratum():
    grid = v1.Grid((8, 16, 32), (1, 4))
    h = v1.half_labels(grid, 3)
    lo = 0
    for hi in grid.n_images:
        assert abs(int((h[lo:hi] == 0).sum()) - int((h[lo:hi] == 1).sum())) <= 1
        lo = hi
    # every unit of a document lies in one half: accumulating one document alone fills one half only
    docs, counts = make_docs(8, 3, 3, 6, seed=2)
    g2 = v1.Grid((4, 8), (1, 6))
    acc, halves = accumulate(docs, counts, g2, [], 0)
    sums, _, _ = acc.cell(0, 1)
    assert [int(x) for x in sums.n] == [int(counts[:4][halves[:4] == hh].sum()) for hh in (0, 1)]


def test_k_subsets_are_nested_and_batch_independent():
    counts = torch.tensor([5, 9, 3, 12])
    pos = torch.arange(4)
    r = llm.unit_ranks(pos, counts, 12, seed=7)
    for i, n in enumerate(counts.tolist()):
        assert sorted(r[i, :n].tolist()) == list(range(n))  # a permutation of the document's units
    sel = {k: r < k for k in (1, 4, 16)}
    assert bool((sel[1] <= sel[4]).all()) and bool((sel[4] <= sel[16]).all())
    alone = llm.unit_ranks(torch.tensor([1]), counts[1:2], 9, seed=7)
    assert torch.equal(alone[0, :9], r[1, :9])  # depends on (seed, position, count) only
    assert int(((r < 1) & llm.slot_valid(counts, 12)).sum()) == 4  # k = 1: one unit per document


def test_to_padded_roundtrip():
    mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]], dtype=torch.bool)
    bank = torch.arange(int(mask.sum()), dtype=torch.float32)[None, :, None].repeat(1, 1, 2)
    out, counts = llm.to_padded(bank, mask)
    assert counts.tolist() == [2, 4, 1] and out.shape == (3, 4, 2)
    assert out[0, :2, 0].tolist() == [0.0, 1.0] and out[1, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert out[2, 0, 0].item() == 6.0 and float(out[2, 1:].abs().sum()) == 0.0
    with pytest.raises(ValueError):
        llm.to_padded(bank[:, :3], mask)


# ---------------------------------------------------------------------------------------------------- nulls


def test_position_kept_null_matches_the_vision_accumulator_on_equal_length_documents():
    n_docs, t = 8, 6
    g = torch.Generator().manual_seed(4)
    pm = torch.randn(n_docs, t, 5, generator=g, dtype=F64)
    pk = torch.randn(n_docs, t, 4, generator=g, dtype=F64)
    grid = v1.Grid((4, 8), (1, 6))
    halves = v1.half_labels(grid, 0)
    ref = v1.StratifiedSigma(5, 4, grid, 3)
    mine = llm.RaggedStratifiedSigma(5, 4, grid, 3, [llm.PositionKeptNull()], seed=0)
    valid = torch.ones(2, t, dtype=torch.bool)
    for b, lo in enumerate(range(0, 8, 2)):
        pos = torch.arange(lo, lo + 2)
        level = v1.token_levels(v1.token_ranks(pos, t, 0), grid.ks)
        a = int(v1.image_stratum(pos[:1], grid))
        half = halves[pos]
        ref.update(a, pm[lo:lo + 2], pk[lo:lo + 2], valid, level, half, v1.null_partners(half, 3, 0, b))
        counts = torch.full((2,), t)
        mine.update(a, pm[lo:lo + 2], pk[lo:lo + 2], valid, valid, level, half, pos, counts, b)
    for ia, il in grid.cells():
        s_ref, s_mine = ref.cell(ia, il), mine.cell(ia, il)
        assert torch.allclose(s_ref.mm, s_mine[0].mm) and torch.allclose(s_ref.mk, s_mine[0].mk)
        assert torch.allclose(s_ref.null_mk, s_mine[1]["N2"][0]) and torch.equal(s_ref.null_n, s_mine[1]["N2"][1])
        assert torch.allclose(s_ref.icc_b, s_mine[0].icc_b)


def test_random_unit_null_pairs_other_documents_with_a_permutation_of_their_units():
    counts = torch.tensor([6, 4, 5, 3])
    b, s, d = 4, 6, 2
    pk = torch.zeros(b, s, d, dtype=F64)
    for i in range(b):  # row (i, u) carries the id 10 i + u
        pk[i, :, 0] = 10 * i + torch.arange(s)
    valid = llm.slot_valid(counts, s)
    half = torch.tensor([0, 0, 1, 1])
    null = llm.RandomUnitNull()
    ctx = null.prepare(half, torch.arange(4), counts, 0, 5, 11)
    for r in range(5):
        pk_r, v_r = null.rows(ctx, r, pk, valid)
        part = ctx["partners"][r]
        for i in range(b):
            j = int(part[i])
            assert j != i and half[j] == half[i]
            ids = pk_r[i, :, 0][v_r[i]].long()
            assert bool(((ids // 10) == j).all())  # the partner's units, never the own document's
            assert bool((ids % 10 < counts[j]).all()) and ids.numel() == len(set(ids.tolist()))  # distinct real units
            assert ids.numel() == min(int(counts[i]), int(counts[j]))
    a1 = null.rows(ctx, 0, pk, valid)
    a2 = null.rows(null.prepare(half, torch.arange(4), counts, 0, 5, 11), 0, pk, valid)
    assert torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1])  # deterministic
    assert not torch.equal(null.rows(ctx, 1, pk, valid)[0], a1[0])  # replicates differ


def test_null_registry_is_pluggable():
    class Shift(llm.NullScheme):
        name = "NX"

        def prepare(self, half, positions, counts, batch_index, n_perm, seed):
            return None

        def rows(self, ctx, r, pk, valid):
            return pk.roll(1, 1), valid.roll(1, 1)

    docs, counts = make_docs(8, 3, 3, 6, seed=3)
    grid = v1.Grid((4, 8), (1, 6))
    acc, _ = accumulate(docs, counts, grid, [llm.RandomUnitNull(), Shift()], 2)
    _, nul, _ = acc.cell(1, 1)
    assert set(nul) == {"N1", "NX"} and nul["NX"][0].shape == (2, 2, 3, 3)
    assert [type(n).__name__ for n in llm.make_nulls(["N1", "N2"])] == ["RandomUnitNull", "PositionKeptNull"]
    with pytest.raises(ValueError):
        llm.make_nulls(["bogus"])


# ---------------------------------------------------------------------------------------------------- r* rules


def test_rstar_rules():
    opts = llm.LlmAnalysisOptions(null_quantile=0.95, prefix_frac=0.95)
    null = torch.linspace(0.0, 0.5, 100)[None, :]  # pooled 95% quantile ('higher') = 0.4798...
    thr = float(torch.quantile(null.flatten(), 0.95, interpolation="higher"))
    g = torch.tensor([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], dtype=F64)
    out = llm.rstar_rules(g, null, opts)
    assert out["threshold"] == pytest.approx(thr)
    assert out["r_star_contig"] == 2  # the old rule stops at the first miss
    assert out["n_above"] == 5
    assert out["r_star_excess"] == pytest.approx(5 - 0.05 * 5)
    # prefix rule with frac 0.95 needs (almost) all indices above: r = 2
    assert out["r_star_prefix"] == 2
    lax = llm.rstar_rules(g, null, llm.LlmAnalysisOptions(prefix_frac=0.8))
    assert lax["r_star_prefix"] == 6  # 5 of the first 6 indices are above (0.83 >= 0.8); 5 of 7 is 0.71
    none = llm.rstar_rules(torch.zeros(4, dtype=F64), null, opts)
    assert none["r_star_prefix"] == 0 and none["r_star_contig"] == 0


def _cell(n, k, r, gamma):
    return {"N": n, "k": k, "T_nominal": n * k, "status": "ok", "r_star": r, "gamma_out_vec": torch.as_tensor(gamma, dtype=F64)}


def test_selection_absolute_over_top_r_ref():
    big = _cell(4096, 128, 3, [0.9, 0.8, 0.7, 0.2])
    mid = _cell(1024, 128, 3, [0.91, 0.79, 0.71, 0.9])  # differs a lot only beyond r_ref = 3
    small = _cell(256, 1, 1, [0.5, 0.1, 0.1, 0.1])
    sel = llm.select_cell_abs([small, mid, big])
    assert sel["cell"]["N"] == 1024 and sel["r_ref"] == 3
    assert llm.select_cell_abs([small, mid, big], tol_r=0.0)["cell"]["N"] == 1024
    assert llm.select_cell_abs([mid, big], tol_g=0.001)["cell"]["N"] == 4096  # only the largest is stable
    degenerate = {**small, "status": "degenerate"}
    assert llm.select_cell_abs([degenerate])["cell"] is None
    glob = llm.select_global_abs([[small, mid, big], [mid, big]])
    assert glob["cell"] is None or glob["cell"]["N"] in (1024, 4096)


# ---------------------------------------------------------------------------------------------------- analysis, coverage


def synthetic_acc(n_docs=32, t=30, d_m=6, d_k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, d_m, generator=g, dtype=F64)
    b = torch.randn(3, d_k, generator=g, dtype=F64)
    docs = []
    for _ in range(n_docs):
        z = torch.randn(t, 3, generator=g, dtype=F64)
        docs.append((z @ a + 0.3 * torch.randn(t, d_m, generator=g, dtype=F64),
                     z @ b + 0.3 * torch.randn(t, d_k, generator=g, dtype=F64)))
    counts = torch.full((n_docs,), t)
    grid = v1.Grid((16, 32), (1, 30))
    acc, _ = accumulate(docs, counts, grid, [llm.RandomUnitNull(), llm.PositionKeptNull()], 9)
    return acc, a, grid


def test_analysis_and_coverage_on_synthetic_data():
    acc, a, grid = synthetic_acc()
    sums, nul, units = acc.cell(1, 1)
    opts = llm.LlmAnalysisOptions(n_null=9, icc_top=3)
    gram_shared = a.T @ a  # energy of E in the 3-dim shared subspace of P_m
    gram_iso = torch.eye(6, dtype=F64)
    info, tens = llm.analyze_cell_llm(sums, nul, opts, {"shared": gram_shared, "iso": gram_iso}, units)
    assert info["status"] == "ok" and info["T_realised"] == 32 * 30
    assert info["r_star"] == info["r_star_prefix"] and 1 <= info["r_star"] <= 5
    assert "N2_r_star" in info and "r_star_contig" in info and "r_star_excess" in info
    assert info["null_primary"] == "N1" and tens["null_gamma_out"].shape == (9, info["q"])
    cov = info["coverage"]
    assert cov["shared"]["cov_total"] > cov["iso"]["cov_total"]  # E lives in the shared directions
    assert 0.0 <= cov["iso"]["cov_total"] <= 1.0 and cov["shared"]["cov_total"] > 0.8
    assert info["coverage_isotropic_reference"] == info["r_star"] / info["rank_m"]
    assert info["n_eff_trace_real"] <= info["rows"] and info["mean_units_per_doc"] == pytest.approx(30.0)
    # the shared structure is far above both nulls
    assert info["gamma_out_top1"] > info["threshold"] and info["gamma_out_top1"] > info["N2_threshold"]


def test_degenerate_cell_reports_no_gamma():
    acc, _, _ = synthetic_acc()
    sums, nul, units = acc.cell(0, 0)  # 16 rows: half 8 rows <= d_m + d_k
    info, tens = llm.analyze_cell_llm(sums, nul, llm.LlmAnalysisOptions(n_null=9), None, units)
    assert info["status"] == "degenerate" and tens == {} and "gamma_out_top1" not in info


# ---------------------------------------------------------------------------------------------------- unit alignment


def _word_batch(texts, offsets_ids):
    """Hand-made tokenization: per text the (start, end, id) of its tokens; returns a padded batch with word units."""
    from merge_and_rebase.data.text_units import token_word_units

    t_max = max(len(o) for o in offsets_ids) + 1
    ids = torch.zeros(len(texts), t_max, dtype=torch.long)
    att = torch.zeros_like(ids)
    units = torch.full_like(ids, -1)
    for b, (text, toks) in enumerate(zip(texts, offsets_ids, strict=True)):
        offs = [(s, e) for s, e, _ in toks]
        tid = [i for _, _, i in toks]
        ids[b, : len(tid)], att[b, : len(tid)] = torch.tensor(tid), 1
        units[b, : len(tid)] = torch.tensor(token_word_units(text, offs, tid, [1] * len(tid)))
    return {"input_ids": ids, "attention_mask": att, "word_unit": units}


def test_word_unit_alignment_on_hand_made_tokenizations():
    from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY

    text = "the cat sat down"  # words 0..3 end at characters 3, 7, 11, 16
    # tokenizer A: BOS-like first token (empty span: never a unit, but the first content token), one token per word
    a = [(0, 0, 9), (0, 3, 1), (4, 7, 2), (8, 11, 3), (12, 16, 4)]
    # tokenizer B: no BOS; "sat" is two tokens (the last one ends the word); "down" is two tokens as well
    b = [(0, 3, 11), (4, 7, 12), (8, 9, 13), (9, 11, 14), (12, 14, 15), (14, 16, 16)]
    fam = FakeFamily()
    ba, bb = _word_batch([text], [a]), _word_batch([text], [b])
    wa, wb, counts = llm_pair(ba, bb, fam)
    ma, mb = wa[ROW_MASK_KEY], wb[ROW_MASK_KEY]
    assert counts["common"] == int(ma.sum()) == int(mb.sum())
    from merge_and_rebase.rebase.methods.anchors.text_units import unit_keys

    assert unit_keys(ma, ba["word_unit"]) == unit_keys(mb, bb["word_unit"])
    keys = unit_keys(ma, ba["word_unit"])
    # B has no BOS: its first content token (the word "the") is excluded, so "the" is missing in the common set
    assert [w for _, w in keys] == [1, 2, 3]
    assert int(mb[0, 3]) == 1 and int(mb[0, 5]) == 1 and int(mb[0, 1]) == 1 and int(mb[0, 0]) == 0  # last tokens


def llm_pair(ba, bb, fam):
    from merge_and_rebase.rebase.methods.anchors.text_units import pair_unit_batches

    return pair_unit_batches(ba, bb, fam)


def test_word_alignment_with_real_tokenizers_if_cached():
    transformers = pytest.importorskip("transformers")
    try:
        tq = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
        tl = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizers not cached")
    from merge_and_rebase.data.llm_calibration import build_text_calibration_loader

    texts = ["The quick brown fox, jumped over 12,345 lazy dogs (twice)!\nNew line: \"quoted\" text-with-hyphens.",
             "Antidisestablishmentarianism is a long word; naïve café résumé über 日本語 mix."]
    tok_pad = {"pad_token": "<pad>"}
    for tok in (tq, tl):
        if tok.pad_token_id is None:
            tok.add_special_tokens(tok_pad) if False else setattr(tok, "pad_token", tok.eos_token)
    kw = {"texts": texts, "batch_size": 2, "max_length": 48, "word_units": True}
    bq = next(iter(build_text_calibration_loader(tokenizer=tq, **kw)))
    bl = next(iter(build_text_calibration_loader(tokenizer=tl, **kw)))
    pb = llm.pair_batch(bl, bq, FakeFamily(), "word")
    assert pb.stats["common"] == int(pb.counts.sum()) > 0
    assert pb.stats["common"] <= min(pb.stats["source"], pb.stats["target"])
    for tok, batch, mask in ((tl, bl, pb.mask_m), (tq, bq, pb.mask_k)):
        for b in range(2):
            words = texts[b].split()
            for pos in mask[b].nonzero().flatten().tolist():
                w = words[int(batch["word_unit"][b, pos])]
                piece = tok.decode([int(batch["input_ids"][b, pos])]).strip()
                assert piece and w.endswith(piece)  # the row is the last token of the word
    ws = [[int(bl["word_unit"][b, p]) for p in pb.mask_m[b].nonzero().flatten()] for b in range(2)]
    wq = [[int(bq["word_unit"][b, p]) for p in pb.mask_k[b].nonzero().flatten()] for b in range(2)]
    assert ws == wq  # same words, same order, in both banks


def test_token_rows_pair_batch_and_guard():
    ids = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
    att = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    fam = FakeFamily()
    pb = llm.pair_batch({"input_ids": ids, "attention_mask": att}, {"input_ids": ids.clone(), "attention_mask": att}, fam, "token")
    assert pb.counts.tolist() == [2, 1]  # first content token of each document excluded
    assert torch.equal(pb.mask_m, pb.mask_k)
    with pytest.raises(ValueError):
        llm.pair_batch({"input_ids": ids, "attention_mask": att}, {"input_ids": ids + 1, "attention_mask": att}, fam, "token")


# ---------------------------------------------------------------------------------------------------- config


def _payload(**kw):
    base = {"source": {"model": "a", "instruct": "a-it"}, "target": {"model": "b"}, "out_dir": "/tmp/x"}
    return {**base, **kw}


def test_parse_config_guards_and_ks_all():
    cfg = llm.parse_config(_payload())
    assert cfg.ks == (1, 16, "all") and llm.resolve_ks(cfg.ks, 128) == (1, 16, 128)
    with pytest.raises(ValueError):
        llm.parse_config(_payload(ifeval_part="test"))
    with pytest.raises(ValueError):
        llm.parse_config(_payload(tasks=["gsm8k"]))
    with pytest.raises(ValueError):
        llm.parse_config(_payload(n_images=[100], batch_size=16))
    with pytest.raises(ValueError):
        llm.parse_config(_payload(nulls=["N9"]))
    with pytest.raises(ValueError):
        llm.parse_config({**_payload(), "source": {"model": "a"}})


def test_estimate_cost_scales_with_nulls():
    grid = v1.Grid((256, 1024, 4096), (1, 16, 128))
    one = llm.estimate_cost_llm(896, 1536, grid, 19, 1, 28, mean_units=100.0, params_m=5e8, params_k=1.5e9, blocks_per_pass=4)
    two = llm.estimate_cost_llm(896, 1536, grid, 19, 2, 28, mean_units=100.0, params_m=5e8, params_k=1.5e9, blocks_per_pass=4)
    assert two["per_block_accumulator_gb"] > one["per_block_accumulator_gb"] > 4.0
    assert one["n_passes"] == 7 and one["forward_flops_total"] == one["forward_flops_per_pass"] * 7


def test_float32_compute_with_float64_storage_matches_float64():
    docs, counts = make_docs(8, 4, 3, 6, seed=9)
    grid = v1.Grid((4, 8), (1, 6))
    halves = v1.half_labels(grid, 0)
    accs = [llm.RaggedStratifiedSigma(4, 3, grid, 2, [llm.RandomUnitNull()], seed=0, compute_dtype=dt)
            for dt in (None, torch.float32)]
    for b, lo in enumerate(range(0, 8, 2)):
        pos = torch.arange(lo, lo + 2)
        pm, pk, real = pad_batch(docs, lo, lo + 2)
        level = v1.token_levels(llm.unit_ranks(pos, counts[lo:lo + 2], real.shape[1], 0), grid.ks)
        a = int(v1.image_stratum(pos[:1], grid))
        for acc in accs:
            acc.update(a, pm, pk, real, real, level, halves[pos], pos, counts[lo:lo + 2], b)
    s64, n64, _ = accs[0].cell(1, 1)
    s32, n32, _ = accs[1].cell(1, 1)
    assert s32.mm.dtype == torch.float64
    assert torch.allclose(s64.mm, s32.mm, rtol=1e-4, atol=1e-4) and torch.allclose(n64["N1"][0], n32["N1"][0], rtol=1e-4, atol=1e-4)
    assert torch.equal(n64["N1"][1], n32["N1"][1])


def test_write_outputs_roundtrip(tmp_path):
    import csv
    import json

    acc, a, grid = synthetic_acc()
    opts = llm.LlmAnalysisOptions(n_null=9, icc_top=3)
    cells, tensors = [], {}
    for ia, il in grid.cells():
        sums, nul, units = acc.cell(ia, il)
        info, t = llm.analyze_cell_llm(sums, nul, opts, {"ifeval": a.T @ a}, units)
        cells.append(info)
        if t:
            tensors[f"b0_N{info['N']}_k{info['k']}"] = t
    blk = {"target_block": 0, "source_block": 0, "cells": cells, "selection": {"cell": None, "reason": "x"}}
    last = cells[-1]
    summary = {"setting": {"pair": "t", "tasks": ["ifeval"]}, "blocks": [blk], "selection_global": {"cell": None},
               "r_star_profile_largest_cell": {"cell": {"N": 32, "k": 30}, "blocks": [
                   {"target_block": 0, "source_block": 0, "d_sum": 11, **{k: last.get(k) for k in (
                       "status", "r_star", "r_star_contig", "r_star_excess", "n_above", "q", "rank_m")}}]}}
    llm.write_outputs(summary, tensors, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"summary.json", "summary.md", "blocks.csv", "tensors.pt"}
    rows = list(csv.DictReader(open(tmp_path / "blocks.csv")))
    assert len(rows) == len(cells) and "N2_r_star_prefix" in rows[0] and "cov_total_ifeval" in rows[0]
    assert json.loads((tmp_path / "summary.json").read_text())["blocks"][0]["cells"][-1]["r_star"] == last["r_star"]
    assert "r* by depth" in (tmp_path / "summary.md").read_text()
    assert set(torch.load(tmp_path / "tensors.pt")) == set(tensors)
