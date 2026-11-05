"""Phase 0.4 of PIANO_METODO_K: seed config generator, alpha patience off, full test curve, per-block table."""

from __future__ import annotations

import copy
import json

import pytest

from merge_and_rebase.eval.diagnostics.metodoK_blocks import block_rows, drift_config, write_table
from merge_and_rebase.eval.metodoK import make_configs as gen
from merge_and_rebase.eval.vision_rebase.alpha_search import AlphaSearchSpec, run_alpha_search
from merge_and_rebase.rebase.config_schema import canonicalize
from merge_and_rebase.rebase.run_config import ALPHA_PATIENCE_DISABLED, resolve_run_config

BASE_ANCHORS = {
    "suite": "vision8",
    "tasks": "EuroSAT,GTSRB",
    "models": {"source": {"model": "ViT-B-16", "pretrained": "x"}, "target": {"model": "ViT-L-14", "pretrained": "x"},
               "tuned_checkpoints": {"EuroSAT": "a.pt", "GTSRB": "b.pt"}},
    "method": {"name": "anchors", "params": {
        "anchor_pool": "tiny_imagenet", "num_anchors": 2048, "anchor_batches": 16, "registration_batches": 10,
        "pool_seed": 0, "anchor_seed": 0, "canonical_tokens": 197, "state_dir": "results/anchors_state/tiny_A16_T2048",
        "seed": 89, "frame_map": "gpa", "synthesis": "ridge_preserve", "rho_g": 0.1}},
    "alpha": {"search": True, "min": 0.0, "max": 2.0, "step": 0.5, "selection": "per_task", "split": "val",
              "patience": 1000},
    "save": {"transported_task_vectors_policy": "auto"},
    "seed": 89,
    "logging": {"local_log_dir": "/x/y", "run_name": "O2"},
}
BASE_ARIADNE = {**copy.deepcopy(BASE_ANCHORS), "method": {"name": "ariadne", "params": {"seed": 89}}}


def _gen(base, s, name):
    return gen.make_seed_config(base, s, name=name, exp="e")


def test_seeds_vary_together_and_are_identical_across_methods():
    for s in (0, 1, 2):
        a, b = _gen(BASE_ANCHORS, s, "O2"), _gen(BASE_ANCHORS, s, "B2")
        p = a["method"]["params"]
        assert {p[k] for k in ("anchor_seed", "pool_seed", "seed", "map_seed")} == {s}
        assert b["method"]["params"]["seed"] == p["seed"] and b["method"]["params"]["anchor_seed"] == s
        assert _gen(BASE_ARIADNE, s, "A")["method"]["params"]["seed"] == s
        assert a["seed"] == 89  # val-split seed fixed


def test_state_dir_per_seed_save_dropped_outputs_on_data():
    cfgs = [_gen(BASE_ANCHORS, s, "O2") for s in (0, 1, 2)]
    dirs = [c["method"]["params"]["state_dir"] for c in cfgs]
    assert len(set(dirs)) == 3
    assert all(d.startswith("/data/merge_and_rebase_results/anchors_state_metodoK/seed") for d in dirs)
    assert all(d.endswith("tiny_A16_T2048") for d in dirs)
    assert all("save" not in c for c in cfgs)
    assert all(c["logging"]["local_log_dir"].startswith("/data/merge_and_rebase_results/e/runs/") for c in cfgs)
    assert len({c["logging"]["run_name"] for c in cfgs}) == 3
    assert "save" in gen.make_seed_config(BASE_ANCHORS, 0, name="O2", exp="e", keep_save=True)
    assert "state_dir" not in _gen(BASE_ARIADNE, 1, "A")["method"]["params"]
    assert BASE_ANCHORS["method"]["params"]["seed"] == 89  # base not mutated


def test_state_tag_collision_with_different_frame_is_rejected():
    other = copy.deepcopy(BASE_ANCHORS)
    other["method"]["params"]["num_anchors"] = 512
    with pytest.raises(ValueError, match="different frames"):
        gen.check_state_tags([_gen(BASE_ANCHORS, 0, "a"), _gen(other, 0, "b")])


def test_generated_config_resolves_with_patience_off_and_test_curve():
    cfg = dict(canonicalize(_gen(BASE_ANCHORS, 1, "O2")))
    cfg.setdefault("block_extension_enabled", True)
    resolved = resolve_run_config(cfg)
    assert resolved.alpha.patience == ALPHA_PATIENCE_DISABLED
    assert resolved.cfg["alpha_test_curve"] is True


def test_patience_default_unchanged():
    base = dict(canonicalize(copy.deepcopy(BASE_ANCHORS)))
    base["alpha_patience"] = 1000
    base.setdefault("block_extension_enabled", True)
    assert resolve_run_config(base).alpha.patience == 1000
    del base["alpha_patience"]
    assert resolve_run_config(base).alpha.patience == 0  # legacy default
    assert resolve_run_config(base).cfg.get("alpha_test_curve") is None


# ---- alpha search with a fake evaluator -------------------------------------------------------------------------


class _Log:
    def __init__(self):
        self.events = []

    def log_event(self, name, **kw):
        self.events.append((name, kw))


class _Ev:
    """Two tasks. Val rises to a peak at alpha=0.5 then dips and recovers (a patience-0 search stops at the dip)."""

    target_base_sd: dict = {}
    rebased_deltas: list = []

    def __init__(self, val, test):
        self.val, self.test, self.test_calls = val, test, []

    def _curve(self, split):
        return self.val if split == "val" else self.test

    def eval_baseline(self, split, idx, alpha):
        return 0.1

    def eval_baseline_indices(self, split, indices, alpha):
        return {i: 0.1 for i in indices}

    def eval_rebased_indices(self, split, indices, alpha):
        if split == "test":
            self.test_calls.append(round(alpha, 6))
        return {i: self._curve(split)(round(alpha, 6)) for i in indices}

    def eval_single_tv_indices(self, split, indices, alpha):
        raise AssertionError


class _Plan:
    can_eval_untransported = [False, False]
    single_tv_deltas = None


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
VAL = {0.0: 0.2, 0.25: 0.4, 0.5: 0.6, 0.75: 0.5, 1.0: 0.9, 1.25: 0.7, 1.5: 0.3}


def _spec(selection, patience, test_curve, log=None):
    return AlphaSearchSpec(
        per_task=[{"task": "A"}, {"task": "B"}], alphas=ALPHAS, selection=selection, patience=patience,
        search_split="val", global_alpha_search=True, merge_mode="none", merge_method_name="x", merge_params={},
        merge_weights=[1.0, 1.0], method_label="m", run_logger=log or _Log(), test_curve=test_curve,
    )


@pytest.mark.parametrize("selection", ["shared", "per_task"])
def test_patience_cuts_curve_and_null_patience_keeps_it_full(selection):
    cut = run_alpha_search(_spec(selection, 0, False), _Ev(VAL.get, VAL.get), _Plan())
    full = run_alpha_search(_spec(selection, ALPHA_PATIENCE_DISABLED, True), _Ev(VAL.get, VAL.get), _Plan())
    assert cut.best_alpha == pytest.approx(0.5)  # stopped at the dip, missed the true peak
    assert full.best_alpha == pytest.approx(1.0)
    assert [r["alpha"] for r in full.alpha_curve_val] == ALPHAS
    assert [r["alpha"] for r in full.alpha_curve_test] == ALPHAS
    assert cut.alpha_curve_val is None and cut.alpha_curve_test is None


@pytest.mark.parametrize("selection", ["shared", "per_task"])
def test_selection_is_independent_of_the_test_curve(selection):
    adversarial = {a: 1.0 - v for a, v in VAL.items()}  # test is maximal where val is minimal
    runs = {}
    for flag, test in [(False, VAL), (True, VAL), (True, adversarial)]:
        ev = _Ev(VAL.get, test.get)
        res = run_alpha_search(_spec(selection, ALPHA_PATIENCE_DISABLED, flag), ev, _Plan())
        runs[(flag, id(test))] = res
        if flag:  # full test curve evaluated at every grid alpha
            assert sorted(set(ev.test_calls)) == ALPHAS
    base, same, adv = runs.values()
    assert base.best_alpha == same.best_alpha == adv.best_alpha == pytest.approx(1.0)
    assert base.selected_alpha_by_task == adv.selected_alpha_by_task
    assert base.rebase_test_accs == [VAL[1.0]] * 2 and adv.rebase_test_accs == [pytest.approx(0.1)] * 2
    assert [r["per_task_rebased"]["A"] for r in adv.alpha_curve_test] == pytest.approx([1.0 - VAL[a] for a in ALPHAS])
    # the val curve stored next to it is the val one
    assert [r["per_task_rebased"]["A"] for r in adv.alpha_curve_val] == pytest.approx([VAL[a] for a in ALPHAS])


# ---- per-block table ---------------------------------------------------------------------------------------------


def _run_json():
    blocks = [
        {"anchors_source_block": j // 2, "anchors_e_norm": 10.0 * (j + 1), "anchors_d_norm": 15.0 * (j + 1),
         "anchors_c_source": 2.0, "anchors_c_target": 4.0}
        for j in range(3)
    ]
    return {"tasks": ["EuroSAT"], "selected_alpha_by_task": {"EuroSAT": 0.5}, "direct_residual": {
        "diagnostics_by_task": {"EuroSAT": blocks}}}


def test_block_rows_ratios_drift_and_update_cosine():
    import torch

    run = _run_json()
    drift = {"results": {"EuroSAT": {"per_alpha": {
        "0.5": {str(j): {"norm_ratio": 1.0 + j, "cosine": 0.9} for j in range(3)},
        "1": {str(j): {"norm_ratio": 99.0, "cosine": 0.0} for j in range(3)}}}}}
    key = "visual.transformer.resblocks.{}.mlp.c_proj.weight"
    w = torch.randn(4, 4)
    run_tv = {"EuroSAT": {key.format(j): w for j in range(3)}}
    ref_tv = {"EuroSAT": {key.format(0): w, key.format(1): -w, key.format(2): torch.eye(4)}}
    rows = block_rows(run, run_tvs=run_tv, ref_tvs=ref_tv, drift=drift)
    assert len(rows) == 3
    assert rows[0]["ratio_raw"] == pytest.approx(1.5)
    assert rows[0]["ratio_c"] == pytest.approx(1.5 * 2.0 / 4.0)  # (||D||/c_k)/(||E||/c_m)
    assert [r["drift_norm_ratio"] for r in rows] == [1.0, 2.0, 3.0] and rows[0]["drift_alpha"] == 0.5
    assert rows[0]["update_cos_ref"] == pytest.approx(1.0) and rows[1]["update_cos_ref"] == pytest.approx(-1.0)
    assert all(r["source_block"] == j // 2 for j, r in enumerate(rows))
    assert block_rows(run)[0]["update_cos_ref"] is None


def test_block_table_files_and_drift_config(tmp_path):
    rows = block_rows(_run_json())
    write_table(rows, tmp_path / "t.json", {"k": 1})
    assert len(json.loads((tmp_path / "t.json").read_text())["rows"]) == 3
    assert (tmp_path / "t.csv").read_text().splitlines()[0].startswith("task,block,source_block")
    cfg = drift_config({"tasks": "EuroSAT", "selected_alpha_by_task": [0.5]}, "r.json", "tv", "o.json")
    assert cfg["alphas"] == {"EuroSAT": [0.5]}


def test_ratio_c_uses_the_filtered_c_the_decoder_used_when_recorded():
    run = _run_json()
    old = block_rows(run)[0]
    assert old["c_source_used"] == 2.0 and old["c_target_used"] == 4.0  # fallback: registration c
    for d in run["direct_residual"]["diagnostics_by_task"]["EuroSAT"]:
        d["anchors_c_source_used"], d["anchors_c_target_used"] = 1.0, 8.0
    new = block_rows(run)[0]
    assert new["ratio_c"] == pytest.approx(1.5 * 1.0 / 8.0) and new["c_source"] == 2.0 and new["c_target"] == 4.0


def test_generator_frame_keys_are_the_anchor_frame_fields_minus_the_seeds():
    from merge_and_rebase.rebase.methods.anchors.config import FRAME_FIELDS

    assert set(gen.FRAME_KEYS) | set(gen.FRAME_SEED_KEYS) == set(FRAME_FIELDS)
    assert "anchor_tokens_per_image" in gen.FRAME_KEYS
