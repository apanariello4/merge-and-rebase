"""Depth truncation diagnostic: block mask, k = depth == untruncated (bitwise), k = 0 == base, selection and gates."""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.diagnostics.ariadne_norm.truncation import (
    block_index,
    non_block_keys,
    select_first_max,
    select_like_run,
    truncate_task_vector,
    truncation_task,
)
from merge_and_rebase.merge.methods._common import axpy_state_dict

DEPTH = 4
P = "visual.transformer.resblocks"


def _toy():
    g = torch.Generator().manual_seed(0)
    base, tau = {}, {}
    for j in range(DEPTH):
        for name in ("mlp.c_proj.weight", "mlp.c_proj.bias", "attn.out_proj.weight"):
            base[f"{P}.{j}.{name}"] = torch.randn(3, 3, generator=g)
            tau[f"{P}.{j}.{name}"] = torch.randn(3, 3, generator=g)
    base["visual.proj"] = torch.randn(3, 3, generator=g)
    tau["visual.proj"] = torch.randn(3, 3, generator=g)
    base["visual.ln_post.weight"] = torch.randn(3, generator=g)  # not touched by tau
    return base, tau


def test_block_index():
    assert block_index(f"{P}.7.mlp.c_proj.weight") == 7
    assert block_index(f"{P}.12.attn.out_proj.bias") == 12
    assert block_index("visual.proj") is None and block_index("visual.ln_post.weight") is None


@pytest.mark.parametrize("keep_non_block", [False, True])
def test_mask_zeroes_exactly_the_right_keys(keep_non_block):
    _, tau = _toy()
    for k in range(DEPTH + 1):
        out = truncate_task_vector(tau, k, keep_non_block=keep_non_block)
        assert set(out) == set(tau)
        for key, value in out.items():
            j = block_index(key)
            kept = keep_non_block if j is None else j < k
            assert torch.equal(value, tau[key] if kept else torch.zeros_like(tau[key])), (k, key)
    assert non_block_keys(tau) == ["visual.proj"]


def test_k_depth_is_bitwise_untruncated_and_k0_is_base():
    base, tau = _toy()
    tau_only_blocks = {k: v for k, v in tau.items() if block_index(k) is not None}
    full = truncate_task_vector(tau_only_blocks, DEPTH)
    assert all(torch.equal(full[k], tau_only_blocks[k]) for k in tau_only_blocks)
    full_nb = truncate_task_vector(tau, DEPTH, keep_non_block=True)
    assert all(torch.equal(full_nb[k], tau[k]) for k in tau)
    for alpha in (0.0, 0.3, 1.0):
        a = axpy_state_dict(base, full_nb, alpha)
        b = axpy_state_dict(base, tau, alpha)
        assert all(torch.equal(a[k], b[k]) for k in b)
    z = axpy_state_dict(base, truncate_task_vector(tau, 0), 0.7)
    assert all(torch.equal(z[k], base[k]) for k in base)


def _evaluate_factory(base, calls):
    """Toy 'accuracy': a smooth function of the distance of the mounted state to a hidden optimum; val != test."""
    target = {k: v + 0.5 * (i % 3) for i, (k, v) in enumerate(base.items())}

    def evaluate(sd, split):
        calls.append(split)
        d = sum(float((sd[k] - target[k]).pow(2).sum()) for k in base)
        return round(1.0 / (1.0 + d), 6) - (0.01 if split == "test" else 0.0)

    return evaluate


def test_truncation_task_curves_selection_and_gates():
    base, tau = _toy()
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    calls: list[str] = []
    evaluate = _evaluate_factory(base, calls)
    star = 0.5
    # the "reference run" = the untruncated vector evaluated at its own selected alpha
    ref_test = evaluate(axpy_state_dict(base, tau, star), "test")
    ref_base = evaluate(base, "test")
    reference = {"test_rebased": ref_test, "test_baseline": ref_base, "baseline_alpha": 0.0}
    out = truncation_task(
        tau=tau,
        base_state=base,
        depth=DEPTH,
        ks=[0, 2, DEPTH],
        alphas=alphas,
        evaluate=evaluate,
        alpha_star=star,
        search_split="val",
        patience=10,
        reference=reference,
        keep_non_block=False,  # visual.proj is non-block: k = depth is NOT the full vector here
        n_test=100,
    )
    assert out["non_block_keys"] == ["visual.proj"]
    assert out["k_equals_untruncated_vector"] == {"0": False, "2": False, str(DEPTH): False}
    assert out["gates"]["g1_full_reproduces_reference"]["full_vector_source"].startswith("untrunc")
    assert out["gates"]["g1_full_reproduces_reference"]["exact"] and out["failed_gates"] == []
    assert out["gates"]["g0_alpha0_reproduces_base"]["exact"]
    assert out["gates"]["g0_alpha0_reproduces_base"]["mounted_state_equals_target_base_bitwise"]
    # k = 0: the whole curve is the base accuracy at every alpha
    assert out["curves"]["0"]["val"] == [out["zero_alpha_acc"]["val"]] * len(alphas)
    assert out["curves"]["0"]["test"] == [ref_base] * len(alphas)
    sel = out["selection"][str(DEPTH)]["full_grid"]
    i = select_first_max(out["curves"][str(DEPTH)]["val"])
    assert sel["selected_alpha"] == alphas[i] and sel["test_acc"] == out["curves"][str(DEPTH)]["test"][i]

    # keep_non_block=True and no non-block key issue: k = depth is bitwise tau and is reused (no extra eval)
    out2 = truncation_task(
        tau=tau,
        base_state=base,
        depth=DEPTH,
        ks=[DEPTH],
        alphas=alphas,
        evaluate=evaluate,
        alpha_star=star,
        search_split="val",
        patience=10,
        reference=reference,
        keep_non_block=True,
    )
    g1 = out2["gates"]["g1_full_reproduces_reference"]
    assert g1["k_depth_equals_tau_bitwise"] and g1["full_vector_source"] == "k=depth" and g1["exact"]

    # a wrong reference makes the gate fail
    bad = truncation_task(
        tau=tau,
        base_state=base,
        depth=DEPTH,
        ks=[DEPTH],
        alphas=alphas,
        evaluate=evaluate,
        alpha_star=star,
        search_split="val",
        patience=10,
        reference={**reference, "test_rebased": ref_test + 0.01},
        keep_non_block=True,
    )
    assert bad["failed_gates"] == ["g1_full_reproduces_reference"]


def test_run_rule_selection_stops_at_patience():
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    accs = [0.5, 0.6, 0.55, 0.5, 0.5, 0.9]
    assert alphas[select_like_run(alphas, accs, patience=1)] == 0.1  # stops after 2 bad steps, never sees 0.9
    assert alphas[select_like_run(alphas, accs, patience=10)] == 0.5
    assert select_first_max([0.5, 0.7, 0.7]) == 1


def test_ks_validation():
    base, tau = _toy()
    with pytest.raises(ValueError):
        truncation_task(
            tau=tau,
            base_state=base,
            depth=DEPTH,
            ks=[DEPTH + 1],
            alphas=[0.0, 1.0],
            evaluate=lambda s, p: 0.0,
            alpha_star=1.0,
            search_split="val",
            patience=1,
            reference={},
        )
