"""llm_merge loads a dense local HF model directory used as a tuned reference (P7.S12c bug fix).

Before the fix, a dense model directory was routed to the PEFT-adapter materialization path; now it loads as a
full state dict, as io.text_checkpoints.load_aligned_tuned_from_ref does.
"""

from __future__ import annotations

import torch
from _llm_fixtures import tiny_qwen2

from merge_and_rebase.eval import llm_merge
from merge_and_rebase.eval.llm_rebase import common
from merge_and_rebase.models.text_lm import TextBuildConfig


def test_dense_local_dir_loads_as_full_state_dict(tmp_path):
    base = tiny_qwen2()
    tuned = tiny_qwen2()
    with torch.no_grad():
        for p in tuned.parameters():
            p.add_(0.01)
    tuned.save_pretrained(tmp_path / "tuned")
    base_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    aligned = llm_merge._load_aligned_tuned_from_ref(
        ckpt_ref=str(tmp_path / "tuned"),
        base_sd=base_sd,
        build_cfg=TextBuildConfig(model_name_or_path=str(tmp_path / "tuned"), device="cpu"),
        model=base,
    )
    key = "model.layers.0.mlp.down_proj.weight"
    assert torch.equal(aligned[key], tuned.state_dict()[key].float())
    # the base model was not touched by an adapter materialization
    assert torch.equal(base.state_dict()[key], base_sd[key])


def test_shared_helpers_are_the_same_objects():
    assert llm_merge._resolve_tasks is common.resolve_tasks
    assert llm_merge._inject_task_head is common.inject_task_head
