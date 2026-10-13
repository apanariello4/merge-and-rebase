"""CPU LLM rebase candidates must use an immutable base and explicit task weights."""

from __future__ import annotations

import pytest
import torch
from golden._hashing import hash_tensor_dict
from golden.test_llm_main_golden import CASES, World, _base_cfg, _launch

from merge_and_rebase.eval.llm_rebase import cli

CASE = "theseus_same_depth_alpha_search_save_merged_base_ckpts"


def _run(root, monkeypatch, *, alphas=(0.5, 1.0, 0.0), weights=(0.7, 0.3)):
    world_kw, make_config = CASES[CASE]
    config = _base_cfg(
        root, **make_config(root), weights=weights,
        hyperparam_search={"alpha": {"values": list(alphas)}},
    )
    return _launch(config, root, monkeypatch, World(**world_kw))


@pytest.mark.parametrize("base_checkpoints", [False, True])
def test_runtime_snapshots_survive_model_updates(tmp_path, monkeypatch, base_checkpoints):
    build = cli.build_runtime
    snapshots = []

    def check_build(*args, **kwargs):
        rt = build(*args, **kwargs)
        for model, snapshot in (
            (rt.source_llm.model, rt.source_base_sd), (rt.target_llm.model, rt.target_base_sd)
        ):
            before = hash_tensor_dict(snapshot)
            original = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            snapshots.append((before, hash_tensor_dict(snapshot)))
            model.load_state_dict(original)
        return rt

    monkeypatch.setattr(cli, "build_runtime", check_build)
    overrides = {} if base_checkpoints else {"source_base_ckpt": None, "target_base_ckpt": None}
    world_kw, make_config = CASES[CASE]
    cfg = _base_cfg(tmp_path, **{**make_config(tmp_path), **overrides})
    _launch(cfg, tmp_path, monkeypatch, World(**world_kw))
    assert all(before == after for before, after in snapshots)


@pytest.mark.parametrize("alphas", [(0.0, 0.5, 1.0), (1.0, 0.5, 0.0)])
def test_search_candidates_match_independent_runs_and_saved_winner(tmp_path, monkeypatch, alphas):
    sweep = _run(tmp_path / "sweep", monkeypatch, alphas=alphas)
    for alpha, actual in zip(alphas, sweep.harness, strict=True):
        independent = _run(tmp_path / f"single_{alpha}", monkeypatch, alphas=(alpha,))
        assert actual["weights"] == independent.harness[0]["weights"], f"alpha={alpha} depends on search order"
    best_alpha = sweep.recorders[0].summary["best_alpha"]
    winner = sweep.harness[list(alphas).index(best_alpha)]["weights"]
    saved = torch.load(tmp_path / "sweep" / "merged" / "m.pt", map_location="cpu", weights_only=True)
    assert hash_tensor_dict(saved) == winner


@pytest.mark.parametrize("weights", [[], [0.7], [0.7, 0.3, 0.2]])
def test_weight_list_length_must_match_tasks_before_transport(tmp_path, monkeypatch, weights):
    with pytest.raises(ValueError, match="weights length must match tuned checkpoints"):
        _run(tmp_path, monkeypatch, alphas=(0.5,), weights=weights)
    assert not _launch.last_calls.tuned
    assert not _launch.last_calls.harness


def test_scalar_weight_remains_equivalent_to_explicit_weights(tmp_path, monkeypatch):
    scalar = _run(tmp_path / "scalar", monkeypatch, alphas=(0.5,), weights=0.5)
    explicit = _run(tmp_path / "explicit", monkeypatch, alphas=(0.5,), weights=[0.5, 0.5])
    assert scalar.harness[0]["weights"] == explicit.harness[0]["weights"]
