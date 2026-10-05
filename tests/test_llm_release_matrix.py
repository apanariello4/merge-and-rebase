"""LLM release matrix (Phase 7 S13): every transport on tiny real Qwen2 / Qwen3 decoders through ``llm_rebase.main``.

Methods: THESEUS, theseus_gqa, BiCo, Ariadne. Pairs: Qwen2 same depth, Qwen2 extend (2->3), Qwen3 same depth,
Qwen2->Qwen3 same depth. Each cell must run, be deterministic (two runs, identical digests of summary, evaluated
weights and the saved merged state), and leave untouched what it must not change:
- Ariadne writes only the target ``down_proj`` weights (and their materialized biases);
- Qwen2 -> Qwen3 never moves the target's Qwen3-only ``q_norm`` / ``k_norm`` (no source counterpart).
"""

from __future__ import annotations

import pytest
import torch
from _llm_fixtures import tiny_qwen2, tiny_qwen3
from golden.test_llm_main_golden import World, _base_cfg, run_case

_METHODS = {
    "theseus": {"method": "theseus", "method_params": {"num_batches": 2}},
    "theseus_gqa": {"method": "theseus_gqa", "method_params": {"num_batches": 2}},
    "bico": {"method": "bico", "method_params": {"num_batches": 2, "seq_align": "mean"}},
    "ariadne": {"method": "ariadne", "ariadne_params": {"preset": "ariadne", "num_batches": 2, "seed": 0}},
}
_PAIRS = {
    "qwen2_same": dict(src_family="qwen2", tgt_family="qwen2"),
    "qwen2_extend": dict(src_family="qwen2", tgt_family="qwen2", tgt_layers=3),
    "qwen3_same": dict(src_family="qwen3", tgt_family="qwen3"),
    "qwen2_to_qwen3": dict(src_family="qwen2", tgt_family="qwen3"),
}


def _target_base_state(world_kw: dict) -> dict[str, torch.Tensor]:
    world = World(**world_kw)
    spec = world.tgt
    builder = tiny_qwen3 if spec["family"] == "qwen3" else tiny_qwen2
    model = builder(
        layers=spec["layers"], kv_heads=spec["kv_heads"], hidden=spec["hidden"], heads=spec["heads"], seed=spec["seed"]
    )
    return {k: v.detach().float() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("pair", sorted(_PAIRS))
@pytest.mark.parametrize("method", sorted(_METHODS))
def test_llm_release_matrix(method, pair, tmp_path, monkeypatch):
    world_kw = _PAIRS[pair]
    digests = []
    merged = None
    for run in ("a", "b"):
        root = tmp_path / run
        cfg = _base_cfg(root, **_METHODS[method], save_merged=str(root / "merged.pt"), depth_defaults="method")
        digests.append(run_case(cfg, root, monkeypatch, World(**world_kw)))
        merged = torch.load(root / "merged.pt", map_location="cpu", weights_only=False)
    assert digests[0] == digests[1], "non-deterministic run"
    state = merged.get("state_dict", merged) if isinstance(merged, dict) else merged
    base = _target_base_state(world_kw)

    changed = {k for k, v in base.items() if k in state and not torch.equal(state[k].float(), v)}
    assert changed, "the merged model is identical to the target base"
    if method == "ariadne":
        assert all(".mlp.down_proj." in k for k in changed), sorted(changed)
    if pair == "qwen2_to_qwen3":
        assert not any(k.endswith(("q_norm.weight", "k_norm.weight")) for k in changed), sorted(changed)
