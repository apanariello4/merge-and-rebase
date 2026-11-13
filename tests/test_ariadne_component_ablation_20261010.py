"""Opt-in Ariadne ablations ``residual_target="native_oracle"`` and ``fit_blocks`` (closed-form fit, vision).

Pinned: the default path is unchanged (``fit_blocks`` covering every block reproduces the default hash, the sequential
default hash equals the pinned value), unfitted blocks get no update, a sequential fit skips unfitted blocks without
mounting them (later blocks still see the true drift), the oracle target equals ``T^ft_j - T^base_j`` computed by hand and
no Procrustes map enters it, and config validation / serialization.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import (
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_sequential_streaming,
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)

PINNED_DEFAULT_SHA = "f4c90f41ad834619d630c1e0d6a58644bc52987689973702b5708b95ba957fd0"


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(torch.tanh(x))


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_1 = torch.nn.Identity()
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        x_mid = x + self.ls_1(self.attn(x))
        return x_mid + self.ls_2(self.mlp(x_mid))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n)), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


BASE = dict(activation_storage="streaming", num_batches=3, ridge_relative=0.05)
IND = dict(BASE, components=("mlp.c_proj",))
SEQ = dict(IND, depth_fit="sequential")


def _sha(d):
    h = hashlib.sha256()
    for key in sorted(d):
        h.update(key.encode())
        h.update(d[key].numpy().tobytes())
    return h.hexdigest()


def _prepare(source_base, target_base, data, pairing, cfg):
    return prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu"
    )


def _fit(overrides, regime=(2, 4), sequential=False, native=None, scramble_q=False):
    source_base, source_ft, target_base, data, pairing, sd = _setup(*regime)
    cfg = parse_direct_residual_config({**(SEQ if sequential else IND), **overrides})
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    if scramble_q:
        prepared["q_by_position"] = {j: torch.randn_like(q) for j, q in prepared["q_by_position"].items()}
    fit = fit_direct_residual_sequential_streaming if sequential else fit_direct_residual_streaming
    kwargs = {} if native is None else {"native_model": native(target_base)}
    tau, rows = fit(target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu", **kwargs)
    return tau, rows, prepared, target_base


def _positions_with_keys(tau):
    return sorted({int(k.split("resblocks.")[1].split(".")[0]) for k in tau})


# ---- fit_blocks ----------------------------------------------------------------------------------------------------


def test_default_sequential_hash_unchanged():
    tau, _, _, _ = _fit({}, sequential=True)
    assert _sha(tau) == PINNED_DEFAULT_SHA


@pytest.mark.parametrize("sequential", [False, True])
def test_fit_blocks_all_reproduces_default(sequential):
    default, _, _, _ = _fit({}, sequential=sequential)
    full, _, _, _ = _fit({"fit_blocks": [0, 1, 2, 3]}, sequential=sequential)
    assert _sha(default) == _sha(full)


def test_fit_blocks_subset_leaves_other_blocks_untouched_independent():
    default, _, _, _ = _fit({})
    sub, rows, _, _ = _fit({"fit_blocks": [1, 3]})
    assert _positions_with_keys(sub) == [1, 3]
    assert sorted({r["position"] for r in rows}) == [1, 3]
    for key, value in sub.items():  # independent: a fitted block is identical to the full fit
        assert torch.equal(value, default[key])


def test_fit_blocks_sequential_skips_without_mounting():
    # Only block 2 fitted: nothing upstream is mounted, so it equals the independent fit of block 2 and U_2 = 0.
    ind, _, _, _ = _fit({"fit_blocks": [2]})
    seq, rows, _, _ = _fit({"fit_blocks": [2]}, sequential=True)
    assert _positions_with_keys(seq) == [2]
    assert rows[0]["upstream_effect_norm"] == 0.0
    for key, value in seq.items():
        torch.testing.assert_close(value, ind[key], rtol=1e-5, atol=1e-6)
    # Blocks 0 and 2 fitted: block 0 is mounted and block 2 sees its true drift; block 1 stays untouched.
    seq02, rows02, _, _ = _fit({"fit_blocks": [0, 2]}, sequential=True)
    assert _positions_with_keys(seq02) == [0, 2]
    assert rows02[0]["upstream_effect_norm"] == 0.0 and rows02[1]["upstream_effect_norm"] > 0.0
    full, _, _, _ = _fit({}, sequential=True)
    for key in seq02:
        if "resblocks.0." in key:
            assert torch.equal(seq02[key], full[key])


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"fit_blocks": []}, "non-empty"),
        ({"fit_blocks": [2, 1]}, "sorted and unique"),
        ({"fit_blocks": [1, 1]}, "sorted and unique"),
        ({"fit_blocks": [-1]}, "non-negative"),
        ({"fit_blocks": [True]}, "non-negative"),
        ({"fit_blocks": [0], "depth_fit": "sequential", "sequential_impl": "block_carry"}, "fit_blocks requires"),
        ({"fit_blocks": [0], "tv_scaling": "global"}, "fit_blocks requires"),
    ],
)
def test_fit_blocks_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        parse_direct_residual_config({**IND, **bad})


def test_fit_blocks_out_of_range_rejected_at_fit():
    with pytest.raises(ValueError, match="out of range"):
        _fit({"fit_blocks": [0, 9]})


# ---- native_oracle -------------------------------------------------------------------------------------------------

ORACLE = {
    "residual_target": "native_oracle",
    "native_target_ckpt_template": "unused/{task}.pt",
}


def _native(target_base, seed=5):
    native = _tuned_copy(target_base, seed, scale=0.1)
    return native.eval()


def _block_outputs(model, images):
    x = model.visual.input(images)
    outs = []
    for block in model.visual.transformer.resblocks:
        x = block(x)
        outs.append(x.detach())
    return outs


def test_oracle_target_equals_native_minus_base_by_hand():
    tau, rows, prepared, target_base = _fit(ORACLE, native=_native)
    native = _native(target_base)
    expected = [0.0] * 4
    with torch.no_grad():
        for batch in prepared["target_batches"]:
            for j, (nat, base) in enumerate(
                zip(_block_outputs(native, batch[0]), _block_outputs(target_base, batch[0]), strict=True)
            ):
                expected[j] += float(((nat - base).double() ** 2).sum())
    assert len(rows) == 4
    for row in rows:
        assert row["desired_norm"] == pytest.approx(expected[row["position"]] ** 0.5, rel=1e-5)
        assert row["residual_target"] == "native_oracle" and row["procrustes_used"] is False
    assert all(torch.isfinite(v).all() for v in tau.values())


def test_oracle_uses_no_procrustes_map():
    tau, _, _, _ = _fit(ORACLE, native=_native)
    scrambled, _, _, _ = _fit(ORACLE, native=_native, scramble_q=True)
    assert _sha(tau) == _sha(scrambled)


def test_oracle_fits_exact_native_update_of_one_component():
    # Native = base with block 1's c_proj changed: D*_1 is exactly H dW^T + db, so the (tiny-ridge) fit recovers it.
    def native(target_base):
        nat = deepcopy(target_base)
        torch.manual_seed(3)
        with torch.no_grad():
            nat.visual.transformer.resblocks[1].mlp.c_proj.weight.add_(0.3 * torch.randn(5, 10))
            nat.visual.transformer.resblocks[1].mlp.c_proj.bias.add_(0.1)
        return nat.eval()

    _, rows, _, _ = _fit({**ORACLE, "ridge_estimator": "none"}, native=native)
    by_pos = {r["position"]: r for r in rows}
    assert by_pos[1]["relative_residual_after"] < 1e-4
    assert by_pos[1]["relative_residual_before"] == pytest.approx(1.0)
    # blocks downstream of 1 see a changed input, so their D* is nonzero but block 0 (upstream) has D* == 0.
    assert by_pos[0]["desired_norm"] == 0.0


def test_oracle_with_fit_blocks():
    tau, rows, _, _ = _fit({**ORACLE, "fit_blocks": [2]}, native=_native)
    assert _positions_with_keys(tau) == [2] and [r["position"] for r in rows] == [2]


def test_oracle_needs_native_model():
    with pytest.raises(ValueError, match="native_model"):
        _fit(ORACLE)


@pytest.mark.parametrize(
    "bad",
    [
        {"native_target_ckpt_template": "x/{task}.pt"},  # template without the oracle target
        {**ORACLE, "native_target_ckpt_template": "no_placeholder.pt"},
        {**ORACLE, "depth_fit": "sequential"},
        {**ORACLE, "block_split": "joint"},
        {**ORACLE, "tv_scaling": "global"},
        {**ORACLE, "fidelity_holdout": True},
        {**ORACLE, "merge_mode": "merge_in_source_then_fit"},
    ],
)
def test_oracle_validation(bad):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**IND, **bad})


def test_oracle_is_vision_only_for_decoders():
    with pytest.raises(NotImplementedError, match="vision-only"):
        resolve_ariadne_decoder_config(ORACLE, family_adapter=object())


# ---- serialization -------------------------------------------------------------------------------------------------


def test_new_fields_serialized_only_when_non_default():
    default = direct_residual_config_dict(parse_direct_residual_config(IND))
    for name in ("native_target_ckpt_template", "fit_blocks"):
        assert name not in default
    cfg = parse_direct_residual_config({**IND, "fit_blocks": [0, 2]})
    assert direct_residual_config_dict(cfg)["fit_blocks"] == (0, 2)
    cfg = parse_direct_residual_config({**IND, **ORACLE})
    out = direct_residual_config_dict(cfg)
    assert out["residual_target"] == "native_oracle" and out["native_target_ckpt_template"] == "unused/{task}.pt"


# ---- pair_target_split ---------------------------------------------------------------------------------------------


def _hand_D(prepared, regime, pairing):
    """Unsplit transported change per target block and batch, by hand: ``(f - b) @ Q_j`` on the source blocks pi(j)."""
    source_base, source_ft, _, _, _, _ = _setup(*regime)
    out = {j: [] for j in range(len(pairing.pairing))}
    with torch.no_grad():
        for batch in prepared["target_batches"]:
            base = _block_outputs(source_base, batch[0])
            ft = _block_outputs(source_ft, batch[0])
            for j, i in enumerate(pairing.pairing):
                out[j].append((ft[i] - base[i]) @ prepared["q_by_position"][j])
    return out


def _norm(tensors):
    return float(sum((t.double() ** 2).sum() for t in tensors)) ** 0.5


def test_share_targets_are_d_over_m_by_hand():
    _, rows, prepared, _ = _fit({"pair_target_split": "per_block_share"})
    pairing = DiscreteLayerPairing.compute(2, 4)
    D = _hand_D(prepared, (2, 4), pairing)
    for row in rows:
        j = row["position"]
        assert row["target_group_size"] == 2 and row["target_scale"] == 0.5
        assert row["desired_norm"] == pytest.approx(_norm(D[j]) / 2, rel=1e-5)


def test_interpolate_targets_by_hand():
    _, rows, prepared, _ = _fit({"pair_target_split": "interpolate"}, sequential=True)
    pairing = DiscreteLayerPairing.compute(2, 4)
    D = _hand_D(prepared, (2, 4), pairing)
    zero = [torch.zeros_like(t) for t in D[0]]
    prev = {0: zero, 1: zero, 2: D[1], 3: D[1]}
    by_pos = {r["position"]: r for r in rows}
    for j in range(4):
        i = j % 2
        want = [p + (i + 1) / 2 * (d - p) for p, d in zip(prev[j], D[j], strict=True)]
        assert by_pos[j]["desired_norm"] == pytest.approx(_norm(want), rel=1e-5)
        assert by_pos[j]["target_interp_weight"] == (i + 1) / 2
    # last block of a group keeps D_j exactly (same value the unsplit fit uses)
    _, plain, _, _ = _fit({}, sequential=True)
    for j in (1, 3):
        assert by_pos[j]["desired_norm"] == plain[j]["desired_norm"]


@pytest.mark.parametrize("split, sequential", [("per_block_share", False), ("interpolate", True)])
@pytest.mark.parametrize("regime", [(3, 3), (4, 2)])
def test_split_group_size_one_is_bit_identical(split, sequential, regime):
    plain, _, _, _ = _fit({}, regime=regime, sequential=sequential)
    split_tau, rows, _, _ = _fit({"pair_target_split": split}, regime=regime, sequential=sequential)
    assert _sha(plain) == _sha(split_tau)
    assert all(r["target_group_size"] == 1 for r in rows)


@pytest.mark.parametrize(
    "bad",
    [
        {"pair_target_split": "bogus"},
        {"pair_target_split": "per_block_share", "depth_fit": "sequential"},
        {"pair_target_split": "interpolate"},
        {"pair_target_split": "interpolate", "depth_fit": "sequential", "sequential_impl": "block_carry"},
        {"pair_target_split": "per_block_share", "tv_scaling": "global"},
    ],
)
def test_split_validation(bad):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**IND, **bad})


def test_split_serialized_only_when_non_default():
    assert "pair_target_split" not in direct_residual_config_dict(parse_direct_residual_config(IND))
    cfg = parse_direct_residual_config({**IND, "pair_target_split": "per_block_share"})
    assert direct_residual_config_dict(cfg)["pair_target_split"] == "per_block_share"


# ---- gradient component fits ---------------------------------------------------------------------------------------
# Toy: a pre-LN transformer block with a stock nn.MultiheadAttention (packed in_proj), like open_clip's, in float64 so the
# float32-roundoff floor of the optimisers stays far below the 1e-4 gates.

from merge_and_rebase.rebase.methods._ariadne.gradient_fit import fit_direct_residual_gradient_streaming  # noqa: E402
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase  # noqa: E402


class _MhaBlock(torch.nn.Module):
    def __init__(self, width, time_major=False):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, 2, batch_first=not time_major)
        self.ls_1 = torch.nn.Identity()
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.ls_1(self.attn(h, h, h, need_weights=False)[0])
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class _MhaVisual(torch.nn.Module):
    def __init__(self, width, depth, time_major):
        super().__init__()
        self.time_major = time_major
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_MhaBlock(width, time_major) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        if self.time_major:
            x = x.transpose(0, 1)
        for block in self.transformer.resblocks:
            x = block(x)
        if self.time_major:
            x = x.transpose(0, 1)
        return x.mean(dim=1)


class _MhaModel(torch.nn.Module):
    def __init__(self, width, depth, time_major=False):
        super().__init__()
        self.visual = _MhaVisual(width, depth, time_major)

    def encode_image(self, x):
        return self.visual(x)


def _mha_setup(regime=(2, 4), width=8, seed=21, time_major=False, ft_scale=0.2, zero_mlp=False):
    torch.manual_seed(seed)
    source_base = _MhaModel(width, regime[0], time_major).double().eval()
    target_base = _MhaModel(width, regime[1], time_major).double().eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            for p in (block.mlp.c_proj.weight, block.attn.out_proj.weight, block.attn.in_proj_weight):
                p.add_(ft_scale * torch.randn_like(p))
    if zero_mlp:
        with torch.no_grad():
            for block in target_base.visual.transformer.resblocks:
                block.mlp.c_proj.weight.zero_()
                block.mlp.c_proj.bias.zero_()
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(6, 5, 4, generator=generator).double()
    data = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=2, shuffle=False)
    pairing = DiscreteLayerPairing.compute(*regime)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


# gradient_fit_tol is a LOSS tolerance: the weight error scales like its square root (times the conditioning), so the
# 1e-4 weight gates need a much tighter value than the 1e-9 default.
GRAD = dict(BASE, ridge_estimator="empirical_bayes", gradient_fit_tol=1e-14)


def _mha_fit(overrides, **setup_kwargs):
    source_base, source_ft, target_base, data, pairing, sd = _mha_setup(**setup_kwargs)
    cfg = parse_direct_residual_config({**GRAD, **overrides})
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    fit = (
        fit_direct_residual_gradient_streaming if cfg.component_fit != "closed_form" else fit_direct_residual_streaming
    )
    tau, rows = fit(target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu")
    return tau, rows


def _rel(a, b):
    return float(torch.linalg.norm((a - b).double()) / (torch.linalg.norm(b.double()) + 1e-30))


def _assert_tau_close(a, b, tol=1e-4):
    assert set(a) == set(b)
    for key in a:
        assert _rel(a[key], b[key]) <= tol, (key, _rel(a[key], b[key]))


ARMS_GRAD = ["gradient_exact", "gradient_linearized"]


@pytest.mark.parametrize("arm", ARMS_GRAD)
def test_gradient_cproj_equals_closed_form(arm):
    closed, _ = _mha_fit({"components": ("mlp.c_proj",)})
    grad, rows = _mha_fit({"components": ("mlp.c_proj",), "component_fit": arm})
    _assert_tau_close(grad, closed)  # weights and biases
    assert any(k.endswith("c_proj.bias") for k in grad)
    for row in rows:
        assert row["gradient_fit_converged"] and row["relative_residual_after"] < 1.0
        assert row["relative_residual_after"] == pytest.approx(row["relative_residual_after_linearized"], rel=1e-6)


@pytest.mark.parametrize("arm", ARMS_GRAD)
def test_gradient_out_proj_with_zeroed_mlp_equals_closed_form(arm):
    closed, _ = _mha_fit({"components": ("attn.out_proj",)}, zero_mlp=True)
    grad, _ = _mha_fit({"components": ("attn.out_proj",), "component_fit": arm}, zero_mlp=True)
    _assert_tau_close(grad, closed)


def test_gradient_out_proj_includes_mlp_reaction():
    closed, _ = _mha_fit({"components": ("attn.out_proj",)})
    grad, rows = _mha_fit({"components": ("attn.out_proj",), "component_fit": "gradient_exact"})
    assert max(_rel(grad[k], closed[k]) for k in grad) > 1e-3  # the downstream MLP changes the optimum
    assert all(r["relative_residual_after"] < 1.0 for r in rows)


def test_linearized_matches_exact_to_first_order():
    errors = {}
    for scale in (1e-4, 1.0):
        exact, _ = _mha_fit({"components": ("attn.out_proj",), "component_fit": "gradient_exact"}, ft_scale=scale)
        lin, _ = _mha_fit({"components": ("attn.out_proj",), "component_fit": "gradient_linearized"}, ft_scale=scale)
        errors[scale] = max(_rel(exact[k], lin[k]) for k in exact)
    assert errors[1e-4] < 1e-3
    assert errors[1.0] > 10 * errors[1e-4]


def test_gradient_v_slice_is_written_into_the_packed_rows():
    exact, rows = _mha_fit({"components": ("attn.v_proj",), "component_fit": "gradient_exact"})
    # With the MLP zeroed the block output is affine in (dW_v, db_v): exact == linearized. With a live MLP the reaction
    # downstream of the attention makes the two objectives differ.
    kw = {"components": ("attn.v_proj",)}
    e0, _ = _mha_fit({**kw, "component_fit": "gradient_exact"}, zero_mlp=True)
    l0, _ = _mha_fit({**kw, "component_fit": "gradient_linearized"}, zero_mlp=True)
    _assert_tau_close(e0, l0)
    lin, _ = _mha_fit({**kw, "component_fit": "gradient_linearized"})
    assert max(_rel(exact[k], lin[k]) for k in exact) > 1e-3
    key = "visual.transformer.resblocks.0.attn.in_proj_weight"
    bias_key = "visual.transformer.resblocks.0.attn.in_proj_bias"
    assert exact[key].shape == (24, 8) and exact[bias_key].shape == (24,)
    assert exact[key][:16].abs().max() == 0 and exact[key][16:].abs().max() > 0
    assert exact[bias_key][:16].abs().max() == 0
    assert all(r["relative_residual_after"] < 1.0 for r in rows)


def test_gradient_qk_and_qkv_and_c_fc_groups():
    for comps, expected_rows in (
        (("attn.q_proj", "attn.k_proj"), 2),
        (("attn.q_proj", "attn.k_proj", "attn.v_proj"), 3),
    ):
        tau, rows = _mha_fit({"components": comps, "component_fit": "gradient_exact"})
        key = "visual.transformer.resblocks.0.attn.in_proj_weight"
        zero_rows = tau[key][16:].abs().max() == 0 if "attn.v_proj" not in comps else False
        assert zero_rows or "attn.v_proj" in comps
        assert len(rows) == 4 * expected_rows
        assert all(r["relative_residual_after"] < r["relative_residual_before"] for r in rows)
        assert all(r["gradient_fit_group"] == list(comps) for r in rows)
    tau, rows = _mha_fit({"components": ("mlp.c_fc",), "component_fit": "gradient_exact"})
    assert "visual.transformer.resblocks.3.mlp.c_fc.weight" in tau
    assert all(r["relative_residual_after"] < 1.0 for r in rows)


@pytest.mark.parametrize("arm", ARMS_GRAD)
def test_match_norm_hits_closed_form_cproj_norm(arm):
    closed, _ = _mha_fit({"components": ("mlp.c_proj",)})
    tau, rows = _mha_fit({"components": ("mlp.c_fc",), "component_fit": arm, "gradient_ridge": "match_norm"})
    for row in rows:
        j = row["position"]
        target = float(torch.linalg.norm(closed[f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"].double()))
        assert row["match_norm_target"] == pytest.approx(target, rel=1e-5)
        assert abs(row["correction_norm"] - target) <= 0.01 * target
        assert row["match_norm_relative_gap"] <= 0.01


def test_gradient_time_major_layout_is_equivalent():
    batch_major, _ = _mha_fit({"components": ("mlp.c_proj",), "component_fit": "gradient_exact"})
    time_major, _ = _mha_fit({"components": ("mlp.c_proj",), "component_fit": "gradient_exact"}, time_major=True)
    # different models would differ; same seed -> same weights, so the fit must be layout independent
    _assert_tau_close(time_major, batch_major)


def test_gradient_with_oracle_fit_blocks_and_share():
    source_base, source_ft, target_base, data, pairing, sd = _mha_setup()
    cfg = parse_direct_residual_config(
        {**GRAD, "components": ("mlp.c_proj",), "component_fit": "gradient_exact", "fit_blocks": [1, 3]}
    )
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    tau, rows = fit_direct_residual_gradient_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    assert _positions_with_keys(tau) == [1, 3] and [r["position"] for r in rows] == [1, 3]
    share = {
        **GRAD,
        "components": ("mlp.c_proj",),
        "component_fit": "gradient_exact",
        "pair_target_split": "per_block_share",
    }
    tau_s, rows_s = _mha_fit(share)
    _, rows_u = _mha_fit({**share, "pair_target_split": "none"})
    for a, b in zip(rows_s, rows_u, strict=True):
        assert a["desired_norm"] == pytest.approx(b["desired_norm"] / 2, rel=1e-9)
    native = deepcopy(target_base)
    with torch.no_grad():
        native.visual.transformer.resblocks[2].mlp.c_proj.weight.add_(0.1)
    oracle_cfg = parse_direct_residual_config(
        {**GRAD, "components": ("mlp.c_proj",), "component_fit": "gradient_exact", **ORACLE}
    )
    prepared = _prepare(source_base, target_base, data, pairing, oracle_cfg)
    tau_o, rows_o = fit_direct_residual_gradient_streaming(
        target_base,
        sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=oracle_cfg,
        device="cpu",
        native_model=native.eval(),
    )
    by_pos = {r["position"]: r for r in rows_o}
    assert by_pos[0]["desired_norm"] == 0.0 and by_pos[2]["desired_norm"] > 0.0
    assert by_pos[2]["relative_residual_after"] < 0.3  # D*_2 is exactly affine in (dW, db); the ridge shrinks it
    assert all(r["procrustes_used"] is False for r in rows_o)


def test_gradient_target_state_restored_and_defaults_untouched():
    source_base, source_ft, target_base, data, pairing, sd = _mha_setup()
    before = {k: v.clone() for k, v in target_base.state_dict().items()}
    cfg = parse_direct_residual_config({**GRAD, "components": ("mlp.c_proj",), "component_fit": "gradient_linearized"})
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    fit_direct_residual_gradient_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    for k, v in target_base.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(p.grad is None for p in target_base.parameters())


@pytest.mark.parametrize(
    "bad",
    [
        {"component_fit": "bogus"},
        {"component_fit": "gradient_exact", "components": ("attn.q_proj",)},
        {"component_fit": "gradient_exact", "components": ("attn.out_proj", "mlp.c_proj")},
        {"component_fit": "gradient_exact", "components": ("mlp.c_proj",), "depth_fit": "sequential"},
        {"component_fit": "gradient_exact", "components": ("mlp.c_proj",), "ridge_estimator": "fixed_relative"},
        {"component_fit": "gradient_exact", "components": ("mlp.c_proj",), "fidelity_holdout": True},
        {"component_fit": "gradient_exact", "components": ("mlp.c_proj",), "gradient_ridge": "bogus"},
        {"component_fit": "closed_form", "components": ("mlp.c_proj",), "gradient_ridge": "match_norm"},
        {"component_fit": "closed_form", "components": ("mlp.c_fc",)},
        {"component_fit": "gradient_exact", "components": ("mlp.c_proj",), "gradient_fit_max_iters": 0},
        {"component_fit": "gradient_exact", "components": ("mlp.c_proj",), "gradient_fit_tol": 0.0},
    ],
)
def test_gradient_validation(bad):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**GRAD, **bad})


def test_gradient_fields_serialized_only_when_non_default():
    default = direct_residual_config_dict(parse_direct_residual_config(IND))
    for name in ("component_fit", "gradient_fit_max_iters", "gradient_fit_tol", "gradient_ridge"):
        assert name not in default
    cfg = parse_direct_residual_config({**GRAD, "components": ("mlp.c_fc",), "component_fit": "gradient_exact"})
    assert direct_residual_config_dict(cfg)["component_fit"] == "gradient_exact"


def test_gradient_is_vision_only_and_streaming_only():
    with pytest.raises(NotImplementedError, match="vision-only"):
        resolve_ariadne_decoder_config(
            {**GRAD, "components": ("mlp.c_proj",), "component_fit": "gradient_exact"}, family_adapter=object()
        )
    source_base, source_ft, target_base, data, pairing, sd = _mha_setup()
    resident = parse_direct_residual_config(
        {**GRAD, "components": ("mlp.c_proj",), "component_fit": "gradient_exact", "activation_storage": "resident"}
    )
    with pytest.raises(NotImplementedError, match="streaming"):
        AriadneRebase().prepare(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target_base,
            target_base_sd=sd,
            source_loader=data,
            target_loader=data,
            pairing=pairing,
            config=resident,
            device="cpu",
        )
