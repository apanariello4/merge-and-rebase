"""Tests of the LLM readout diagnostic (``eval.diagnostics.ariadne_norm.llm_readout``) on toy Qwen2 decoders.

Source (2 layers, width 24), target (3 layers, width 32) and a separate native fine-tuned target are distinct tiny
models; nothing here touches a checkpoint or the network.
"""

from __future__ import annotations

import copy
import inspect
import json

import pytest
import torch
from _llm_fixtures import perturbed_copy, tiny_qwen2

from merge_and_rebase.eval.diagnostics.ariadne_norm import llm_readout as lr
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.methods._ariadne.fit import _task_vector_sha256
from merge_and_rebase.rebase.model_families import infer_family

VOCAB = 64


def _docs(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        p = int(torch.randint(3, 7, (1,), generator=g))
        r = int(torch.randint(2, 6, (1,), generator=g))
        out.append(
            {
                "doc_index": 10 + i,
                "prompt_ids": torch.randint(2, VOCAB, (p,), generator=g).tolist(),
                "response_ids": torch.randint(2, VOCAB, (r,), generator=g).tolist(),
            }
        )
    return out


def _make_setup(native_equals_target=False):
    src = tiny_qwen2(layers=2, hidden=24, seed=0)
    src_ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_qwen2(layers=3, hidden=32, seed=2)
    fam = infer_family(src)
    base_sd = {k: v.clone() for k, v in tgt.state_dict().items()}
    materialize_missing_projection_biases(tgt, base_sd, family_adapter=fam, components=("mlp.c_proj",))
    native = copy.deepcopy(tgt) if native_equals_target else perturbed_copy(tgt, scale=0.05, seed=3)
    keys = [k for k in base_sd if "down_proj" in k]
    assert len(keys) == 6
    torch.manual_seed(5)
    updates = []
    for name, scale, best in (("original", 0.05, 0.6), ("sequential", 0.03, 0.8)):
        tv = {k: scale * torch.randn_like(base_sd[k], dtype=torch.float32) for k in keys}
        updates.append(lr.Update(name, tv, best, _task_vector_sha256(tv)))
    q = {j: torch.randn(24, 32) / 5 for j in range(3)}
    setup = lr.ReadoutSetup(
        device=torch.device("cpu"),
        fam=fam,
        target=tgt,
        mounted=copy.deepcopy(tgt),
        source_base=src,
        source_ft=src_ft,
        native=native,
        pairing=[0, 0, 1],
        q=q,
        target_base_state=base_sd,
        updates=updates,
        pad_id=0,
    )
    return setup


@pytest.fixture(scope="module")
def setup():
    return _make_setup()


CFG = {
    "batch_size": 2,
    "head_chunk": 5,
    "vocab_chunk": 3,
    "vocab_top_k": 8,
    "alpha_grid": [0.0, 0.5, 1.0],
    "vocab_beta_grid": [0.0, 1.0],
    "inject_betas": [0.5, 1.0],
    "m4_alphas": [0.5, 1.0],
    "split_seed": 0,
}


@pytest.fixture(scope="module")
def measured(setup):
    docs = _docs()
    before = lr.model_hashes(setup)
    record = lr.run_measure(setup, docs, CFG)
    assert lr.model_hashes(setup) == before
    return record, docs


# ---- head, injection, projection --------------------------------------------------------------------------------


def test_head_equals_plain_forward(setup):
    docs = _docs(3)
    cb = lr.collate(docs, 0, "cpu")
    store, logits = lr.run_model(setup.target, setup.fam, cb["ids"], cb["mask"], [2], logits=True)
    head = lr.Head(setup.target, setup.fam)
    with torch.no_grad():
        diff = (head(store[2]) - logits).abs()[cb["mask"].bool()].max()
    assert float(diff) < 1e-4


def test_injection_beta0_equals_base_and_beta_changes(setup):
    cb = lr.collate(_docs(3), 0, "cpu")
    base, _ = lr.run_model(setup.target, setup.fam, cb["ids"], cb["mask"], [2])
    for j in range(3):
        vec = torch.randn(*cb["ids"].shape, 32)
        zero, _ = lr.run_model(setup.target, setup.fam, cb["ids"], cb["mask"], [2], replace={j: 0.0 * vec})
        assert torch.equal(zero[2], base[2])
        moved, _ = lr.run_model(setup.target, setup.fam, cb["ids"], cb["mask"], [2], replace={j: 0.5 * vec})
        assert not torch.equal(moved[2], base[2])


def test_aligned_plus_orthogonal_is_delta_and_orthogonal_to_d():
    torch.manual_seed(0)
    delta, d = torch.randn(7, 16), torch.randn(7, 16)
    d[3] = 0.0
    aligned, orth = lr.project_on(delta, d)
    assert torch.allclose(aligned + orth, delta.double(), atol=1e-12)
    assert float((orth * d.double()).sum(-1).abs().max()) < 1e-10
    assert torch.equal(orth[3], delta[3].double()) and float(aligned[3].abs().sum()) == 0.0


# ---- vocabulary target ------------------------------------------------------------------------------------------


def test_rmsnorm_jacobian_matches_autograd(setup):
    norm = setup.target.model.norm
    torch.manual_seed(1)
    h = torch.randn(2, 32, dtype=torch.float64)
    a = torch.randn(2, 5, 32, dtype=torch.float64)
    jac = lr.rmsnorm_jacobian_rows(a, h, norm.weight.double(), float(norm.variance_epsilon))
    for r in range(2):
        jn = torch.autograd.functional.jacobian(lambda x: norm.double()(x[None])[0], h[r])  # [D, D]: d y_i / d h_k
        assert torch.allclose(jac[r], a[r] @ jn, atol=1e-9)
    norm.float()


def test_vocab_dual_solve_equals_primal_ridge():
    torch.manual_seed(2)
    jac = torch.randn(4, 6, 20, dtype=torch.float64)
    dl = torch.randn(4, 6, dtype=torch.float64)
    lam = 1e-3 * (jac * jac).sum((-1, -2)) / 6
    assert torch.allclose(lr.ridge_dual(jac, dl, lam), lr.ridge_primal(jac, dl, lam), atol=1e-9)


def test_vocab_target_delta_h_shapes_and_fit(setup):
    torch.manual_seed(3)
    h = torch.randn(5, 32)
    head = lr.Head(setup.target, setup.fam)
    ft, base = torch.randn(5, VOCAB), torch.randn(5, VOCAB)
    dh, info = lr.vocab_target_delta_h(
        h, ft, base, head(h), head.lm.weight, head.norm.weight, head.eps, top_k=8, ridge_rel=1e-3
    )
    assert dh.shape == (5, 32) and torch.isfinite(dh).all()
    assert 8 <= info["mean_k"] <= 24 and 0 <= info["mean_relative_fit_error"] < 1


# ---- metrics ----------------------------------------------------------------------------------------------------


def test_token_metrics_match_brute_force():
    torch.manual_seed(4)
    nat, var = torch.randn(6, VOCAB), torch.randn(6, VOCAB)
    tgt = torch.randint(0, VOCAB, (6,))
    got = lr.token_metrics(nat, var, tgt)
    for r in range(6):
        pn = torch.softmax(nat[r].double(), -1)
        pv = torch.softmax(var[r].double(), -1)
        assert got[r, 0] == pytest.approx(float((pn * (pn.log() - pv.log())).sum()), abs=1e-5)
        assert got[r, 1] == float(nat[r].argmax() == var[r].argmax())
        assert got[r, 2] == pytest.approx(float(-pv[tgt[r]].log()), abs=1e-5)
    same = lr.token_metrics(nat, nat, tgt)
    assert float(same[:, 0].abs().max()) < 1e-6 and float(same[:, 1].min()) == 1.0


def test_spearman_and_ranks():
    assert lr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert lr.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert list(lr.ranks([3.0, 1.0, 3.0])) == [2.5, 1.0, 2.5]
    assert lr.spearman([1, 1, 1], [1, 2, 3]) != lr.spearman([1, 1, 1], [1, 2, 3])  # nan


def test_split_is_deterministic_and_a_partition():
    a, b = lr.split_indices(141, 0), lr.split_indices(141, 0)
    assert a == b
    assert sorted(a[0] + a[1]) == list(range(141)) and len(a[0]) == 71 and len(a[1]) == 70
    assert lr.split_indices(141, 1) != a


# ---- full toy run -----------------------------------------------------------------------------------------------


def test_run_measure_gates_and_structure(measured):
    record, docs = measured
    assert all(g["passed"] for g in record["gates"].values()), record["gates"]
    assert record["n_docs"] == 5
    assert record["n_response_tokens"] == sum(len(d["response_ids"]) for d in docs)
    assert record["inject_layers"] == sorted({round(f * 2) for f in (0.2, 0.4, 0.6, 0.8)} | {2})
    # full at alpha 0 is the base variant
    base = record["m2"]["variants"]["base"]["curve"]["0"]["all"]
    full0 = record["m2"]["updates"]["original"]["full"]["curve"]["0"]["all"]
    assert full0 == base
    # M4 rows: original at its alphas plus every update at its best alpha
    got = {(r["update"], r["alpha"]) for r in record["m4"]["rows"]}
    assert ("original", 0.5) in got and ("original", 0.6) in got and ("sequential", 0.8) in got


def test_m1_matches_hidden_states_brute_force(setup, measured):
    record, docs = measured
    cb = lr.collate(docs, 0, "cpu")
    kw = {"input_ids": cb["ids"], "attention_mask": cb["mask"], "output_hidden_states": True, "use_cache": False}
    with torch.no_grad():
        hs_b, hs_n = setup.target(**kw).hidden_states, setup.native(**kw).hidden_states
        hs_sb, hs_sf = setup.source_base(**kw).hidden_states, setup.source_ft(**kw).hidden_states
    rows = lr.build_rows(cb["P"], cb["T"])
    sel = rows["response"]
    gi = (rows["b"][sel], rows["t"][sel])
    d0 = ((hs_sf[1] - hs_sb[1]) @ setup.q[0])[gi]  # layer 0 pairs with source layer 0 (not the last source layer)
    n0 = (hs_n[1] - hs_b[1])[gi]
    expect = float((d0 * n0).sum() / (d0.norm() * n0.norm()))
    got = record["m1"]["0"]["response"]["D|N"]
    assert got["cos_global"] == pytest.approx(expect, abs=1e-5)
    assert got["norm_ratio_a_over_b"] == pytest.approx(float(d0.norm() / n0.norm()), rel=1e-5)
    assert got["n_rows"] == int(sel.sum())


def test_m1_dt_against_mounted_forward(setup, measured):
    record, docs = measured
    u = setup.updates[0]
    mounted = copy.deepcopy(setup.target)
    with torch.no_grad():
        params = dict(mounted.named_parameters())
        for k, v in u.tv.items():
            params[k].copy_(setup.target_base_state[k] + u.best_alpha * v)
    cb = lr.collate(docs, 0, "cpu")
    kw = {"input_ids": cb["ids"], "attention_mask": cb["mask"], "output_hidden_states": True, "use_cache": False}
    with torch.no_grad():
        hm, hb, hn = (m(**kw).hidden_states for m in (mounted, setup.target, setup.native))
    rows = lr.build_rows(cb["P"], cb["T"])
    sel = rows["response"]
    gi = (rows["b"][sel], rows["t"][sel])
    dt, n1 = (hm[2] - hb[2])[gi], (hn[2] - hb[2])[gi]
    got = record["m1"]["1"]["response"][f"original@{u.best_alpha:g}|dT|N"]
    assert got["cos_global"] == pytest.approx(float((dt * n1).sum() / (dt.norm() * n1.norm())), abs=1e-5)
    assert got["coef_of_a_on_b"] == pytest.approx(float((dt * n1).sum() / (n1 * n1).sum()), rel=1e-4)


def test_selected_param_minimises_selection_kl(measured):
    record, _ = measured
    for fam in ("target", "native_push", "vocab"):
        s = record["m2"]["variants"][fam]
        kls = {float(p): v["selection"]["kl"] for p, v in s["curve"].items()}
        assert kls[s["selected_param"]] == min(kls.values())


def test_native_equal_target_gives_zero_kl():
    s = _make_setup(native_equals_target=True)
    record = lr.run_measure(s, _docs(4), CFG)
    base = record["m2"]["variants"]["base"]["curve"]["0"]["all"]
    assert base["kl"] < 1e-9 and base["top1"] == 1.0
    # N = 0: injecting it is a no-op
    inj = record["m3"]["N@layer0"]["curve"]
    assert all(v["all"]["kl"] < 1e-9 for v in inj.values())


# ---- generation, files, write-once ------------------------------------------------------------------------------


def test_greedy_generate_matches_manual_argmax_and_stops_at_eos():
    model = tiny_qwen2(layers=2, hidden=24, seed=7)
    prompts = [[5, 6, 7], [9, 10, 11, 12, 13]]
    with torch.no_grad():
        first = model(input_ids=torch.tensor([prompts[0]])).logits[0, -1].argmax().item()
    out = lr.greedy_generate(model, prompts, eos_ids=[first], pad_id=0, max_new_tokens=6, batch_size=2, device="cpu")
    assert out[0] == [first]  # eos is the very first greedy token: kept, then stop
    out = lr.greedy_generate(
        model, prompts, eos_ids=[VOCAB + 5], pad_id=0, max_new_tokens=4, batch_size=1, device="cpu"
    )
    ids = torch.tensor([prompts[1]])
    with torch.no_grad():
        for _ in range(4):
            ids = torch.cat([ids, model(input_ids=ids).logits[:, -1:].argmax(-1)], 1)
    assert out[1] == ids[0, len(prompts[1]) :].tolist()


def test_check_tokenizers_detects_difference():
    class Tok:
        def __init__(self, shift):
            self.shift = shift

        def encode(self, text, add_special_tokens=False):
            return [ord(c) + self.shift for c in text]

    assert lr.check_tokenizers({"a": Tok(0), "b": Tok(0)}, ["ab", "c"])["identical"]
    with pytest.raises(ValueError, match="differs"):
        lr.check_tokenizers({"a": Tok(0), "b": Tok(1)}, ["ab"])


def test_write_once_and_response_sha(tmp_path):
    p = tmp_path / "x.json"
    lr.write_new(p, "1")
    with pytest.raises(FileExistsError):
        lr.write_new(p, "2")
    assert p.read_text() == "1"
    (tmp_path / lr.RESPONSES_NAME).write_text(json.dumps({"doc_index": 1}) + "\n")
    (tmp_path / (lr.RESPONSES_NAME + ".sha256")).write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="differs from the recorded"):
        lr.load_responses(tmp_path)
    (tmp_path / (lr.RESPONSES_NAME + ".sha256")).write_text(lr.sha256_file(tmp_path / lr.RESPONSES_NAME) + "\n")
    rows, digest = lr.load_responses(tmp_path)
    assert rows == [{"doc_index": 1}] and digest == lr.sha256_file(tmp_path / lr.RESPONSES_NAME)


def test_stages_refuse_existing_results(tmp_path):
    (tmp_path / "measure.json").write_text("{}")
    with pytest.raises(FileExistsError):
        lr.stage_measure({"out_dir": str(tmp_path)}, "cpu", b"{}", "cfg.json")
    (tmp_path / lr.RESPONSES_NAME).write_text("")
    with pytest.raises(FileExistsError):
        lr.stage_generate({"out_dir": str(tmp_path)}, "cpu", b"{}", "cfg.json")


def test_attach_m4_scores_and_spearman():
    rows = [
        {
            "update": "original",
            "alpha": a,
            "report": {"kl": k, "top1": 1 - k, "nll": k},
            "all": {"kl": k, "top1": 1 - k, "nll": k},
        }
        for a, k in ((0.25, 0.4), (0.5, 0.3), (0.75, 0.2), (1.0, 0.1))
    ]
    summary = {
        "harness_results_by_alpha": {
            f"{a:g}": {"ifeval_inst_level_strict_acc": 0.5 + a / 10} for a in (0.25, 0.5, 0.75)
        }
    }
    out = lr.attach_m4_scores(rows, {"original": summary})
    assert out["n_points"] == 3 and rows[3]["ifeval"] is None
    assert out["report"]["kl"] == pytest.approx(-1.0) and out["report"]["top1"] == pytest.approx(1.0)


def test_build_llm_inputs_sink_is_optional_and_default_none():
    from merge_and_rebase.eval.diagnostics.ariadne_norm.llm import build_llm_inputs

    assert inspect.signature(build_llm_inputs).parameters["context_sink"].default is None
