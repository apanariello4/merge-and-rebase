"""``refine='joint'`` (anchors/refine.py): joint, label-free function-space refinement of the c_proj updates.

The toy network has token mixing (single-head softmax self-attention) and LayerNorm, so the blocks are NOT affine:
the per-block ridge (independent fit) leaves a known error, and the refinement has something to fix.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.capture import capture_tokens
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.fit import fit_direct_residual
from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.refine import refine_blocks, refine_joint

WIDTH, DEPTH, TOKENS = 8, 4, 6
KEY = "visual.transformer.resblocks.{}.mlp.c_proj"


class _Attn(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        a = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)
        return self.out_proj(a @ v)


class _Block(torch.nn.Module):
    def __init__(self, width, ratio=2):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = _Attn(width)
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(width, ratio * width)), ("gelu", torch.nn.GELU()),
            ("c_proj", torch.nn.Linear(ratio * width, width))]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth, ratio=2):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width, ratio) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth, ratio=2):
        super().__init__()
        self.visual = _Visual(width, depth, ratio)

    def encode_image(self, x):
        return self.visual(x)


def _batches(n_batches, seed, size=4, shift=3.0):
    """Task images carry a strong offset in the first two input coordinates; generic images (shift 0) do not."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(size * n_batches, TOKENS, 4, generator=g)
    images[..., :2] += shift
    return list(DataLoader(TensorDataset(images, torch.zeros(size * n_batches)), batch_size=size, shuffle=False))


def _toy(depth=DEPTH, scale=0.15, seed=3, n_batches=6):
    torch.manual_seed(seed)
    base = _Model(WIDTH, depth).eval()
    truth = deepcopy(base)
    with torch.no_grad():
        for blk in truth.visual.transformer.resblocks:
            blk.mlp.c_proj.weight.add_(scale * torch.randn_like(blk.mlp.c_proj.weight))
            blk.mlp.c_proj.bias.add_(0.1 * torch.randn_like(blk.mlp.c_proj.bias))
    batches = _batches(n_batches, seed + 1)
    req = {str(j): (j, "boundary") for j in range(depth)}
    b_out = capture_tokens(base, batches, req, "cpu")
    t_out = capture_tokens(truth, batches, req, "cpu")
    desired = {j: [t - b for t, b in zip(t_out[str(j)], b_out[str(j)], strict=True)] for j in range(depth)}
    captured = {"target_batches": batches, "target_base_outputs_by_position": {j: b_out[str(j)] for j in range(depth)}}
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    return base, sd, captured, desired


def _independent(model, sd, captured, desired):
    pairing = DiscreteLayerPairing.compute(DEPTH, DEPTH)
    cfg = DirectResidualConfig(components=("mlp.c_proj",), ridge_estimator="fixed_relative", ridge_relative=1e-3,
                               num_batches=6)
    return fit_direct_residual(model, sd, captured, desired, pairing, config=cfg, device="cpu")[0]


def _ratio_last(model, sd, captured, desired, corr):
    edited = deepcopy(model)
    new = {k: v.clone() for k, v in sd.items()}
    for k, v in corr.items():
        new[k] = new[k] + v
    edited.load_state_dict(new)
    batches = captured["target_batches"][-2:]
    j = DEPTH - 1
    out = capture_tokens(edited, batches, {"o": (j, "boundary")}, "cpu")["o"]
    r = torch.cat([(o - b).reshape(-1) for o, b in zip(out, captured["target_base_outputs_by_position"][j][-2:], strict=True)]).double()
    d = torch.cat([x.reshape(-1) for x in desired[j][-2:]]).double()
    return float(r.norm() / d.norm())


def _gen():
    return _batches(3, 77, shift=0.0)


KW = dict(blocks=[1, 3], steps=40, lr=5e-3, batch=4, holdout=2, eval_every=10, seed=0, use_checkpoint=False)


def _run(extra=None, desired_=None, **over):
    model, sd, captured, desired = _toy()
    corr = _independent(model, sd, captured, desired)
    kw = {**KW, **over}
    out = refine_joint(model, sd, captured, desired_ or desired, corr, _gen(), **{**kw, **(extra or {})})
    return model, sd, captured, desired, corr, out


def test_refinement_lowers_heldout_task_loss_and_moves_ratio_toward_one():
    model, sd, captured, desired, corr, (new, per_block, info) = _run(rho=0.1)
    assert info["heldout_task_best"] < info["heldout_task_init"]
    last = per_block[DEPTH - 1]
    assert abs(last["refine_ratio_after"] - 1.0) < abs(last["refine_ratio_before"] - 1.0)
    assert last["refine_rel_err_after"] < last["refine_rel_err_before"]
    # the independent fit overshoots in depth (every block rewrites the cumulative change), the refinement removes it
    r_before, r_after = _ratio_last(model, sd, captured, desired, corr), _ratio_last(model, sd, captured, desired, new)
    assert r_before > 1.1 and abs(r_after - 1.0) < abs(r_before - 1.0)
    assert info["curve"][0]["step"] == 0 and info["best_step"] > 0
    assert set(per_block) == set(range(DEPTH)) and per_block[0]["refine_dw_change"] > 0
    assert all(k in per_block[1] for k in ("refine_generic_drift_before", "refine_generic_drift_after",
                                           "refine_cosine_before", "refine_desired_over_stream_var"))
    for k, v in model.state_dict().items():  # the model is restored
        assert torch.equal(v, sd[k])
    assert all(not hasattr(m.mlp.c_proj, "forward") or "forward" not in m.mlp.c_proj.__dict__
               for m in model.visual.transformer.resblocks)


def test_preserve_term_reduces_generic_drift():
    *_, (_, pb0, _) = _run(rho=0.0)
    *_, (_, pb1, info1) = _run(rho=50.0)
    # rho = 0 has no generic rows: compare through a run that measures drift with the term off (tiny rho)
    *_, (_, pb_off, _) = _run(rho=1e-12)
    assert "refine_generic_drift_after" not in pb0[1]
    total = lambda pb, when: sum(pb[j][f"refine_generic_drift_{when}"] ** 2 for j in (1, 3))  # noqa: E731  (L_keep / rho)
    assert total(pb1, "after") < total(pb_off, "after") and total(pb1, "after") < total(pb1, "before")
    assert pb1[3]["refine_generic_drift_after"] < pb_off[3]["refine_generic_drift_after"]
    assert info1["rho"] == 50.0


def test_final_target_uses_only_the_last_measurement_block():
    *_, (_, pb, info) = _run(target="final", rho=0.1)
    assert info["task_blocks"] == [3] and info["blocks"] == [1, 3]
    assert pb[3]["refine_in_task_loss"] and not pb[1]["refine_in_task_loss"]


def test_deterministic():
    a = _run(rho=0.1)[-1][0]
    b = _run(rho=0.1)[-1][0]
    for k in a:
        assert torch.equal(a[k], b[k])


def test_holdout_batches_never_enter_the_gradient():
    """Corrupting the held-out batches' D changes the held-out loss but not one training loss of the trajectory."""
    model, sd, captured, desired = _toy()
    corr = _independent(model, sd, captured, desired)
    bad = {j: [t.clone() for t in v] for j, v in desired.items()}
    for j in bad:
        for k in (4, 5):
            bad[j][k] = bad[j][k] + 5.0 * torch.randn_like(bad[j][k])
    _, _, i_ok = refine_joint(model, sd, captured, desired, corr, _gen(), rho=0.1, **KW)
    _, _, i_bad = refine_joint(model, sd, captured, bad, corr, _gen(), rho=0.1, **KW)
    assert i_ok["holdout_batches"] == [4, 5] and i_ok["train_batches"] == [0, 1, 2, 3]
    tr_ok = [c["train_task"] for c in i_ok["curve"][1:]]
    tr_bad = [c["train_task"] for c in i_bad["curve"][1:]]
    assert tr_ok == tr_bad  # bit-identical training losses: the trajectory only depends on the training batches
    assert i_ok["curve"][0]["heldout_task"] != i_bad["curve"][0]["heldout_task"]


def test_masked_rows_are_ignored():
    model, sd, captured, desired = _toy()
    corr = _independent(model, sd, captured, desired)
    masks = {j: [torch.ones(4, TOKENS, dtype=torch.bool) for _ in range(6)] for j in range(DEPTH)}
    for j in masks:
        for k in range(6):
            masks[j][k][:, 0] = False  # token 0 (a 'dropped' row) is masked everywhere
    bad = {j: [t.clone() for t in v] for j, v in desired.items()}
    for j in bad:
        for k in range(6):
            bad[j][k][:, 0] = bad[j][k][:, 0] + 100.0
    a = refine_joint(model, sd, captured, desired, corr, _gen(), rho=0.1, row_masks=masks, **KW)
    b = refine_joint(model, sd, captured, bad, corr, _gen(), rho=0.1, row_masks=masks, **KW)
    for k in a[0]:
        assert torch.equal(a[0][k], b[0][k])
    unmasked = refine_joint(model, sd, captured, bad, corr, _gen(), rho=0.1, **KW)
    assert not all(torch.equal(unmasked[0][k], a[0][k]) for k in a[0])


def test_checkpointing_gives_the_same_updates():
    a = _run(rho=0.1, use_checkpoint=False)[-1][0]
    b = _run(rho=0.1, use_checkpoint=True)[-1][0]
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-6)


def test_blocks_resolution_and_config():
    assert refine_blocks("all", [0, 1, 2, 3], 4) == [0, 1, 2, 3]
    assert refine_blocks((0.5, 1.0), [0, 1, 2, 3], 4) == [2, 3]
    d = AnchorsConfig()
    assert d.refine == "none" and len(d.refine_depths) == 6 and d.refine_depths[-1] == 1.0
    cfg = parse_anchors_config({"refine": "joint", "refine_depths": [0.5, 1.0], "depth_fit": "tracking"})
    assert cfg.refine_depths == (0.5, 1.0) and cfg.to_dict()["refine_depths"] == [0.5, 1.0]
    assert parse_anchors_config({"refine": "joint", "refine_depths": "all"}).refine_depths == "all"
    for bad in ({"refine": "x"}, {"refine": "joint", "refine_target": "x"}, {"refine": "joint", "refine_depths": "some"},
                {"refine": "joint", "refine_holdout_batches": 10}, {"refine": "joint", "refine_steps": 0},
                {"refine": "joint", "compile_max_depth": 0.5}, {"refine": "joint", "depth_fit": "sequential"}):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)


# ---------------------------------------------------------------- end to end through the method


def _method():
    from tests.test_anchors_method import _cfg, _prepare, _setup  # noqa: PLC0415

    return _cfg, _prepare, _setup


def test_default_refine_none_is_bit_identical_and_joint_plumbs_through_the_method():
    _cfg, _prepare, _setup = _method()
    setup = _setup(2, 4)
    ref = _prepare(_cfg(depth_fit="tracking"), setup)
    same = _prepare(_cfg(depth_fit="tracking", refine="none"), setup)
    for k, v in ref.unit_task_vector.items():
        assert torch.equal(same.unit_task_vector[k], v)
    assert "refine" not in same.extra["anchors"]
    out = _prepare(_cfg(depth_fit="tracking", refine="joint", refine_steps=16, refine_lr=0.01, refine_batch=2, refine_holdout_batches=1,
                        refine_eval_every=3, refine_depths=(0.5, 1.0), refine_rho=0.5, refine_generic_batches=2), setup)
    info = out.extra["anchors"]["refine"]
    assert info["steps"] == 16 and info["blocks"] == [2, 3]
    assert set(out.unit_task_vector) == set(ref.unit_task_vector)
    assert all("refine_ratio_after" in r for r in out.diagnostics)
    assert info["heldout_objective_best"] <= info["heldout_objective_init"]


# ---------------------------------------------------------------- step size, forward consistency, restoration


def _wide(width=256, ratio=4, depth=2, scale=0.05, seed=5):
    """A toy with c_proj d_in = ratio * width >= 1024 (the realistic regime for the step size)."""
    torch.manual_seed(seed)
    base = _Model(width, depth, ratio).eval()
    truth = deepcopy(base)
    with torch.no_grad():
        for blk in truth.visual.transformer.resblocks:
            blk.mlp.c_proj.weight.add_(scale * torch.randn_like(blk.mlp.c_proj.weight))
    batches = _batches(4, seed + 1, size=8)
    req = {str(j): (j, "boundary") for j in range(depth)}
    b_out, t_out = capture_tokens(base, batches, req, "cpu"), capture_tokens(truth, batches, req, "cpu")
    desired = {j: [t - b for t, b in zip(t_out[str(j)], b_out[str(j)], strict=True)] for j in range(depth)}
    captured = {"target_batches": batches, "target_base_outputs_by_position": {j: b_out[str(j)] for j in range(depth)}}
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    pairing = DiscreteLayerPairing.compute(depth, depth)
    cfg = DirectResidualConfig(components=("mlp.c_proj",), ridge_estimator="fixed_relative", ridge_relative=1e-2,
                               num_batches=4)
    corr = fit_direct_residual(base, sd, captured, desired, pairing, config=cfg, device="cpu")[0]
    return base, sd, captured, desired, corr


def test_one_step_lowers_the_training_loss_at_realistic_width():
    base, sd, captured, desired, corr = _wide()
    assert corr[KEY.format(0) + ".weight"].shape[1] >= 1024
    for rho in (0.0, 0.1):
        _, _, info = refine_joint(base, sd, captured, desired, corr, _gen(), blocks=[0, 1], rho=rho, steps=5, batch=8,
                                  holdout=1, eval_every=5, seed=0, use_checkpoint=True)
        assert info["step1"]["loss_after"] < info["step1"]["loss_before"], info["step1"]
        assert max(info["forward_check"].values()) < 1e-5
        assert info["curve"][-1]["heldout_task"] <= info["curve"][0]["heldout_task"] * 1.05  # does not blow up at the default lr


def test_the_patched_forward_equals_the_mounted_model_with_and_without_checkpointing():
    *_, (_, _, info) = _run(rho=0.1)
    assert info["forward_check"]["patched_vs_mounted"] < 1e-5 and info["forward_check"]["checkpoint_vs_plain"] < 1e-6


def test_model_is_restored_when_the_run_raises(monkeypatch):
    import merge_and_rebase.rebase.methods.anchors.refine as R

    model, sd, captured, desired = _toy()
    corr = _independent(model, sd, captured, desired)
    real, calls = R._forward, {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] > 8:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(R, "_forward", boom)
    with pytest.raises(RuntimeError, match="boom"):
        refine_joint(model, sd, captured, desired, corr, _gen(), rho=0.1, **KW)
    for k, v in model.state_dict().items():
        assert torch.equal(v, sd[k])
    assert all("forward" not in m.__dict__ and "forward" not in m.mlp.c_proj.__dict__
               for m in model.visual.transformer.resblocks)
    assert all(p.requires_grad for p in model.parameters())


def test_final_target_scales_rho_and_curve_has_train_subset_and_per_depth_values():
    *_, (_, _, info) = _run(target="final", rho=0.3)
    assert info["rho_effective"] == pytest.approx(0.3 * 1 / 2)
    c = info["curve"][-1]
    assert set(c["heldout_by_depth"]) == {"1", "3"} and set(c["train_subset_by_depth"]) == {"1", "3"}
    assert "train_subset_task" in c and "train_subset_task" in info["curve"][0]


# ---------------------------------------------------------------- method level


def test_method_refine_fits_on_the_remaining_batches_and_reports_refined_rows(monkeypatch):
    import merge_and_rebase.rebase.methods.anchors.method as M

    _cfg, _prepare, _setup = _method()
    setup = _setup(2, 4)
    # (a) default: refine is never called and the result does not depend on the refine options
    monkeypatch.setattr(M, "_run_refine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refine must not run")))
    ref3 = _prepare(_cfg(depth_fit="tracking"), setup)
    opts = _prepare(_cfg(depth_fit="tracking", refine_steps=5, refine_rho=3.0, refine_depths="all"), setup)
    for k, v in ref3.unit_task_vector.items():
        assert torch.equal(opts.unit_task_vector[k], v)
    # (b) with refine on, the depth fit sees num_batches - holdout batches: bypassing the refinement returns exactly the
    # fit of a 2-batch run
    monkeypatch.undo()
    orig = M._run_refine
    seen = {}

    def spy(*a, **k):
        out = orig(*a, **k)
        seen["out"] = out
        return out

    monkeypatch.setattr(M, "_run_refine", lambda *a, **k: (a[4], a[5], {}))
    bypass = _prepare(_cfg(depth_fit="tracking", refine="joint", refine_holdout_batches=1), setup)
    two = _prepare(_cfg(depth_fit="tracking", num_batches=2), setup)
    for k, v in two.unit_task_vector.items():
        assert torch.equal(bypass.unit_task_vector[k], v)
    # (c) the real refinement: saved / alpha-scaled vector equals the refined one, rows describe the refined update
    monkeypatch.setattr(M, "_run_refine", spy)
    kw = dict(depth_fit="tracking", refine="joint", refine_steps=16, refine_lr=0.01, refine_batch=2, refine_holdout_batches=1,
              refine_eval_every=3, refine_depths=(0.5, 1.0), refine_rho=0.5, refine_generic_batches=2)
    out = _prepare(_cfg(**kw), setup)
    new = seen["out"][0]
    moved = not all(torch.equal(new[k], bypass.unit_task_vector[k]) for k in new)
    assert moved == (out.extra["anchors"]["refine"]["best_step"] > 0)  # step 0 (the fit) is returned unless a step is better
    for k, v in new.items():
        assert torch.equal(out.unit_task_vector[k], v) and torch.equal(out.task_vector[k], v)
    half = _prepare(_cfg(**kw, strength=0.5), setup)
    for k, v in half.unit_task_vector.items():
        assert torch.equal(half.task_vector[k], 0.5 * v)
    for r in out.diagnostics:
        p = int(r["position"])
        assert r["correction_norm"] == pytest.approx(r["refine_dw_norm_after"])
        assert "prefit_correction_norm" in r and "residual_norm_after" not in r
        w = out.unit_task_vector[f"visual.transformer.resblocks.{p}.mlp.c_proj.weight"]
        assert float(torch.linalg.norm(w.double())) == pytest.approx(r["correction_norm"], rel=1e-5)
    assert "generic_holdout_in_sample" in out.extra["anchors"]["refine"]


def test_refine_refuses_unsupported_combinations():
    for bad in ({"refine": "joint", "update_rank": 2}, {"refine": "joint", "map_rows": "task"},
                {"refine": "joint", "depth_fit": "spread", "measure_depths": [0.5]},
                {"refine": "joint", "refine_warmup": 1.0}):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)
    parse_anchors_config({"refine": "joint", "depth_fit": "spread", "measure_depths": [0.5, 1.0]})
