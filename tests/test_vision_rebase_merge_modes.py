from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.vision_rebase import (
    _average_visual_state_dicts,
    _ckpt_visual_base_coverage,
    _infer_ckpt_base,
    _merge_direction,
    _pseudo_tuned,
    _relative_visual_state_distance,
    _resolve_merge_mode_config,
    _scale_deltas_by,
    _select_dedicated_brace_loader,
    _state_dict_sha256,
)


def _clipish_base(depth: int, width: int) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for i in range(depth):
        sd[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"] = torch.randn(3 * width, width)
        sd[f"visual.transformer.resblocks.{i}.ln_1.weight"] = torch.randn(width)
    sd["visual.ln_pre.weight"] = torch.randn(width)
    sd["visual.conv1.weight"] = torch.randn(3 * width, 16, 16)
    sd["visual.class_embedding"] = torch.randn(width)
    sd["token_embedding.weight"] = torch.randn(100, width)
    sd["logit_scale"] = torch.tensor(4.6)
    return sd


def _sd(*values: float) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor(list(values))}


def _approx(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.allclose(a[k], b[k]) for k in a)


def test_pseudo_tuned_adds_each_delta_to_base() -> None:
    base = _sd(1.0, 2.0, 3.0)
    deltas = [
        {"w": torch.tensor([1.0, 1.0, 1.0])},
        {"w": torch.tensor([-2.0, 0.0, 2.0])},
    ]
    tuned = _pseudo_tuned(base, deltas)
    assert len(tuned) == 2
    assert torch.allclose(tuned[0]["w"], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(tuned[1]["w"], torch.tensor([-1.0, 2.0, 5.0]))


def test_independent_endpoint_average_uses_common_visual_float_tensors() -> None:
    bases, keys = _average_visual_state_dicts(
        {
            "Cars": {
                "visual.w": torch.tensor([1.0, 3.0]),
                "visual.counter": torch.tensor(1, dtype=torch.int64),
                "text.w": torch.tensor([100.0]),
            },
            "DTD": {
                "visual.w": torch.tensor([3.0, 5.0]),
                "visual.counter": torch.tensor(2, dtype=torch.int64),
                "text.w": torch.tensor([200.0]),
            },
        }
    )
    assert keys == ["visual.w"]
    assert torch.allclose(bases["visual.w"], torch.tensor([2.0, 4.0]))

    assert _relative_visual_state_distance(
        {"visual.w": torch.tensor([1.0, 3.0])},
        bases,
        keys,
    ) == pytest.approx(((1.0**2 + 1.0**2) / (2.0**2 + 4.0**2)) ** 0.5)


def test_merge_direction_matches_manual_weighted_sum() -> None:
    base = _sd(1.0, 2.0, 3.0)
    d1 = {"w": torch.tensor([1.0, 1.0, 1.0])}
    d2 = {"w": torch.tensor([-2.0, 0.0, 2.0])}
    direction = _merge_direction(
        base_sd=base,
        deltas=[d1, d2],
        merge_method_name="task_arithmetic",
        weights=[2.0, 0.5],
        merge_params={},
    )
    assert torch.allclose(direction["w"], 2.0 * d1["w"] + 0.5 * d2["w"])


def test_merge_direction_zero_weight_task_excluded() -> None:
    base = _sd(1.0)
    d1 = {"w": torch.tensor([5.0])}
    d2 = {"w": torch.tensor([-3.0])}
    direction = _merge_direction(
        base_sd=base,
        deltas=[d1, d2],
        merge_method_name="task_arithmetic",
        weights=[1.0, 0.0],
        merge_params={},
    )
    assert torch.allclose(direction["w"], torch.tensor([5.0]))


def test_plain_merge_method_path_yields_same_direction() -> None:
    """A merge()-only method (no prepare) must produce the same direction."""
    from merge_and_rebase.merge.registry import get_method

    task_arith = get_method("task_arithmetic")

    class PlainTaskArithmeticStub:
        def merge(self, *, base, tuned, weights=None, alpha=1.0, **kwargs):
            return task_arith.merge(base=base, tuned=tuned, weights=weights, alpha=alpha, **kwargs)

    base = _sd(1.0, -1.0)
    deltas = [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([0.5, -1.0])}]
    weights = [1.0, 1.0]

    # Force the non-prepared branch by wrapping through the stub's own math.
    tuned = _pseudo_tuned(base, deltas)
    merged = PlainTaskArithmeticStub().merge(base=base, tuned=tuned, weights=weights)
    plain_direction = {k: v - base[k] for k, v in merged.items() if k in base}

    direction_registry = _merge_direction(
        base_sd=base,
        deltas=list(deltas),
        merge_method_name="task_arithmetic",
        weights=weights,
        merge_params={},
    )
    assert _approx(direction_registry, plain_direction)


def _identity_transport(deltas: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [dict(d) for d in deltas]


def test_modes_equivalent_for_linear_transport_and_task_arithmetic() -> None:
    """rebase_then_merge vs merge_then_rebase coincide for linear transports."""
    source_base = _sd(0.0, 0.0, 0.0)
    target_base = _sd(10.0, 10.0, 10.0)
    deltas = [
        {"w": torch.tensor([1.0, 2.0, 3.0])},
        {"w": torch.tensor([-1.0, 0.5, 2.0])},
        {"w": torch.tensor([0.25, -0.75, 1.0])},
    ]
    weights = [1.0, 0.5, 2.0]

    transported = _identity_transport(deltas)
    rebase_then_merge_dir = _merge_direction(
        base_sd=target_base,
        deltas=transported,
        merge_method_name="task_arithmetic",
        weights=weights,
        merge_params={},
    )

    merged_source_dir = _merge_direction(
        base_sd=source_base,
        deltas=list(deltas),
        merge_method_name="task_arithmetic",
        weights=weights,
        merge_params={},
    )
    merge_then_rebase_dir = _identity_transport([merged_source_dir])[0]

    assert set(rebase_then_merge_dir) == set(merge_then_rebase_dir)
    for key in rebase_then_merge_dir:
        assert torch.allclose(rebase_then_merge_dir[key], merge_then_rebase_dir[key])


def _prune_top1_transport(deltas: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Nonlinear transport: keep only each delta's largest-magnitude entry."""
    out: list[dict[str, torch.Tensor]] = []
    for d in deltas:
        v = d["w"]
        idx = int(torch.argmax(v.abs()))
        pruned = torch.zeros_like(v)
        pruned[idx] = v[idx]
        out.append({"w": pruned})
    return out


def test_modes_diverge_for_nonlinear_transport_and_ties() -> None:
    """Disjoint pruning + ties (sign/top-k nonlinearities) breaks mode equivalence."""
    source_base = _sd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    target_base = _sd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    deltas = [
        {"w": torch.tensor([100.0, -40.0, 30.0, -20.0, 10.0, -5.0])},   # dominant coord 0
        {"w": torch.tensor([30.0, 90.0, -45.0, 25.0, -15.0, 8.0])},     # dominant coord 1
        {"w": torch.tensor([-25.0, 35.0, 80.0, -50.0, 12.0, -7.0])},    # dominant coord 2
    ]
    weights = [1.0, 1.0, 1.0]
    merge_kwargs = dict(merge_method_name="ties_merge", weights=weights, merge_params={})

    rebase_then_merge_dir = _merge_direction(
        base_sd=target_base,
        deltas=_prune_top1_transport(deltas),
        **merge_kwargs,  # type: ignore[arg-type]
    )
    pre_merged = _merge_direction(
        base_sd=source_base,
        deltas=list(deltas),
        **merge_kwargs,  # type: ignore[arg-type]
    )
    merge_then_rebase_dir = _prune_top1_transport([pre_merged])[0]

    assert set(rebase_then_merge_dir) == set(merge_then_rebase_dir) == {"w"}
    # Post-pruning each task keeps a single distinct coordinate, so the merged
    # direction is supported on exactly {0, 1, 2}; merging before pruning keeps
    # magnitude information of all six coordinates (topk=1.0 retains them and
    # resolves signs across tasks). The two orderings cannot coincide here.
    post_support = int((rebase_then_merge_dir["w"].abs() > 0).sum())
    assert post_support < 6, f"expected sparse post-direction, got support {post_support}"
    assert not torch.allclose(rebase_then_merge_dir["w"], merge_then_rebase_dir["w"])


def test_resolve_merge_mode_config_defaults() -> None:
    mode, name, params, global_search = _resolve_merge_mode_config({}, "shared")
    assert (mode, name, params, global_search) == ("none", "task_arithmetic", {}, True)


def test_resolve_merge_mode_config_reads_keys() -> None:
    cfg = {
        "merge_mode": "rebase_then_merge",
        "merge_method": "ties_merge",
        "merge_params": {"merging_type": "sum"},
        "global_alpha_search": False,
    }
    mode, name, params, global_search = _resolve_merge_mode_config(cfg, "per_task")
    assert (mode, name, params, global_search) == ("rebase_then_merge", "ties_merge", {"merging_type": "sum"}, False)


def test_resolve_merge_mode_config_allows_hierarchical_per_task() -> None:
    mode, _, _, global_search = _resolve_merge_mode_config({"merge_mode": "rebase_then_merge"}, "per_task")
    assert mode == "rebase_then_merge"
    assert global_search is True


def test_explicit_transport_then_merge_alias_allows_hierarchical_alpha() -> None:
    mode, _, _, _ = _resolve_merge_mode_config(
        {"merge_mode": "brace_transport_then_merge"}, "per_task"
    )
    assert mode == "brace_transport_then_merge"


@pytest.mark.parametrize(
    "mode",
    ["brace_merge_then_transport", "merge_then_brace_then_transport"],
)
def test_single_transport_campaign_modes_require_shared_alpha(mode: str) -> None:
    with pytest.raises(ValueError, match="requires alpha_selection='shared'"):
        _resolve_merge_mode_config({"merge_mode": mode}, "per_task")


def test_brace_and_transport_calibration_loaders_are_distinct() -> None:
    brace_loader = object()
    transport_loader = object()
    assert (
        _select_dedicated_brace_loader(
            brace_loader=brace_loader,
            transport_loader=transport_loader,
            correction_enabled=True,
        )
        is brace_loader
    )


def test_shared_brace_transport_calibration_loader_is_rejected() -> None:
    shared_loader = object()
    with pytest.raises(RuntimeError, match="separate loader instances"):
        _select_dedicated_brace_loader(
            brace_loader=shared_loader,
            transport_loader=shared_loader,
            correction_enabled=True,
        )


def test_state_dict_hash_detects_mutation_and_ignores_mapping_order() -> None:
    left = {"visual.b": torch.tensor([2.0]), "visual.a": torch.tensor([1.0])}
    right = {"visual.a": left["visual.a"].clone(), "visual.b": left["visual.b"].clone()}
    before = _state_dict_sha256(left)
    assert before == _state_dict_sha256(right)
    right["visual.a"].add_(1.0)
    assert before != _state_dict_sha256(right)


@pytest.mark.parametrize("bad_mode", ["merge", "rebase_then", "", "bogus"])
def test_resolve_merge_mode_config_rejects_unknown_mode(bad_mode: str) -> None:
    with pytest.raises(ValueError, match="merge_mode"):
        _resolve_merge_mode_config({"merge_mode": bad_mode}, "shared")


def test_resolve_merge_mode_config_rejects_per_task_alpha_for_merge_then_rebase() -> None:
    with pytest.raises(ValueError, match="merge_then_rebase"):
        _resolve_merge_mode_config({"merge_mode": "merge_then_rebase"}, "per_task")


def test_resolve_merge_mode_config_rejects_non_bool_global_alpha_search() -> None:
    with pytest.raises(ValueError, match="global_alpha_search"):
        _resolve_merge_mode_config({"merge_mode": "rebase_then_merge", "global_alpha_search": "yes"}, "shared")


def test_scale_deltas_by_scales_each_delta() -> None:
    deltas = [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([3.0, -4.0])}]
    scaled = _scale_deltas_by(deltas, [0.5, 2.0])
    assert torch.allclose(scaled[0]["w"], torch.tensor([0.5, 1.0]))
    assert torch.allclose(scaled[1]["w"], torch.tensor([6.0, -8.0]))


def test_scale_deltas_by_identity_alpha_returns_same_dicts() -> None:
    deltas = [{"w": torch.tensor([1.0])}]
    scaled = _scale_deltas_by(deltas, [1.0])
    assert scaled[0] is deltas[0]


def test_scale_deltas_by_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        _scale_deltas_by([{"w": torch.tensor([1.0])}], [1.0, 2.0])


def test_hierarchical_uniform_alpha_equals_scaled_merge() -> None:
    """Uniform per-task alpha => merge(scaled deltas) == alpha * merge(deltas)."""
    base = _sd(1.0, 1.0)
    deltas = [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([-0.5, 3.0])}]
    weights = [1.0, 2.0]
    alpha = 0.7

    hier = _merge_direction(
        base_sd=base,
        deltas=_scale_deltas_by(deltas, [alpha, alpha]),
        merge_method_name="task_arithmetic",
        weights=weights,
        merge_params={},
    )
    plain = _merge_direction(
        base_sd=base,
        deltas=list(deltas),
        merge_method_name="task_arithmetic",
        weights=weights,
        merge_params={},
    )
    assert torch.allclose(hier["w"], alpha * plain["w"])


# ---------------------------------------------------------------------------
# Automatic checkpoint-base detection (mixed merging)
# ---------------------------------------------------------------------------


def test_infer_ckpt_base_classifies_source_checkpoint() -> None:
    source_base = _clipish_base(depth=2, width=8)
    target_base = _clipish_base(depth=3, width=16)
    source_ckpt = _clipish_base(depth=2, width=8)
    assert (
        _infer_ckpt_base(source_ckpt, source_base_sd=source_base, target_base_sd=target_base) == "source"
    )


def test_infer_ckpt_base_classifies_target_checkpoint_as_native() -> None:
    source_base = _clipish_base(depth=2, width=8)
    target_base = _clipish_base(depth=3, width=16)
    target_ckpt = _clipish_base(depth=3, width=16)
    assert (
        _infer_ckpt_base(target_ckpt, source_base_sd=source_base, target_base_sd=target_base) == "target"
    )


def test_infer_ckpt_base_prefers_source_on_same_shapes() -> None:
    base = _clipish_base(depth=2, width=8)
    ckpt = _clipish_base(depth=2, width=8)
    assert _infer_ckpt_base(ckpt, source_base_sd=base, target_base_sd=base) == "source"


def test_infer_ckpt_base_returns_none_on_no_match() -> None:
    source_base = _clipish_base(depth=2, width=8)
    target_base = _clipish_base(depth=3, width=16)
    garbage = {"foo.bar": torch.randn(3, 4)}
    assert _infer_ckpt_base(garbage, source_base_sd=source_base, target_base_sd=target_base) is None


def test_ckpt_visual_base_coverage_full_and_zero() -> None:
    source_base = _clipish_base(depth=2, width=8)
    target_base = _clipish_base(depth=3, width=16)
    source_ckpt = _clipish_base(depth=2, width=8)
    assert _ckpt_visual_base_coverage(source_ckpt, source_base) == pytest.approx(1.0)
    assert _ckpt_visual_base_coverage(source_ckpt, target_base) == 0.0


def test_ckpt_visual_base_coverage_partial() -> None:
    base = _clipish_base(depth=2, width=8)
    partial = _clipish_base(depth=1, width=8)  # missing resblock 1
    coverage = _ckpt_visual_base_coverage(partial, base)
    assert 0.0 < coverage < 1.0


def test_resolve_merge_mode_config_rejects_unknown_merge_method() -> None:
    with pytest.raises(KeyError, match="no_such_merge"):
        _resolve_merge_mode_config({"merge_mode": "none", "merge_method": "no_such_merge"}, "shared")


def test_resolve_merge_mode_config_rejects_non_dict_params() -> None:
    with pytest.raises(ValueError, match="merge_params"):
        _resolve_merge_mode_config({"merge_mode": "merge_then_rebase", "merge_params": [1, 2]}, "shared")
