"""Tests for the opt-in ``sequential_points`` Ariadne ablation (two correction points per block, vision).

``depth_fit="sequential"`` with ``components = (attn.out_proj, mlp.c_proj)`` fits, inside block ``j`` and after the
blocks ``< j`` are mounted, first O (attention output projection) and then D (MLP output projection):
``end_end``: O against the block-end target ``D_j``; ``mid_end``: O against a mid-block target ``M_j`` (residual stream
after the attention add) through its own polar Procrustes map. Pinned here: default behaviour unchanged (hash and
capture requests), the mid residual, the block-0 equality with an independent O fit, the telescoping exactness of
every substep, config validation/serialization and state restoration.

The toy block is a true pre-residual block (``x_mid = x + attn(x)``; ``out = x_mid + mlp(x_mid)``), unlike the parallel
toy of `tests/test_ariadne_sequential_depth_fit_20261005.py`, so that the mid point is a distinct stream.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import merge_and_rebase.rebase.methods._ariadne.streaming as streaming_module
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import _procrustes_from_cross
from merge_and_rebase.rebase.methods._ariadne.capture import iter_capture_tokens
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
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase


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
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


BASE = dict(activation_storage="streaming", depth_fit="sequential", num_batches=3, ridge_relative=0.05)
SEQ_D = dict(BASE, components=("mlp.c_proj",))
SEQ_O = dict(BASE, components=("attn.out_proj",))
TWO = dict(BASE, components=("attn.out_proj", "mlp.c_proj"))
IND_O = {k: v for k, v in SEQ_O.items() if k != "depth_fit"}
ARMS = ["end_end", "mid_end"]
REGIMES = [
    pytest.param((2, 4), id="extend"),
    pytest.param((4, 2), id="shrink"),
    pytest.param((3, 3), id="same_arch"),
]


def _sha(d):
    h = hashlib.sha256()
    for key in sorted(d):
        h.update(key.encode())
        h.update(d[key].numpy().tobytes())
    return h.hexdigest()


def _state_sha(model):
    return _sha({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()})


def _prepare(source_base, target_base, data, pairing, cfg, **kwargs):
    return prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
        **kwargs,
    )


def _fit_two(regime, points, **overrides):
    source_base, source_ft, target_base, data, pairing, sd = _setup(*regime)
    cfg = parse_direct_residual_config({**TWO, "sequential_points": points, **overrides})
    prepared = _prepare(source_base, target_base, data, pairing, cfg, mid_alignment=points == "mid_end")
    tau, rows = fit_direct_residual_sequential_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    return tau, rows, prepared, (source_base, source_ft, target_base, data, pairing, sd)


# ---- default path unchanged -------------------------------------------------------------------------------------

# sha256 of the sequential D-only task vector of the toy (2 -> 4), recorded from the code before this ablation existed.
PINNED_DEFAULT_SHA = "f4c90f41ad834619d630c1e0d6a58644bc52987689973702b5708b95ba957fd0"
PINNED_PREPARE_KEYS = {
    "q_by_position",
    "procrustes_source",
    "alignment_map",
    "alignment_row_weighting",
    "procrustes_diagnostics",
    "activation_q64_by_position",
    "gradient_q64_by_position",
    "source_mean_by_position",
    "target_mean_by_position",
    "fingerprints",
    "source_batches",
    "target_batches",
    "calibration",
    "distinct_source_indices",
}


def test_default_single_hash_unchanged():
    source_base, source_ft, target_base, data, pairing, sd = _setup(2, 4)
    cfg = parse_direct_residual_config(SEQ_D)
    assert cfg.sequential_points == "single"
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    tau, rows = fit_direct_residual_sequential_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    assert _sha(tau) == PINNED_DEFAULT_SHA
    assert all("sequential_points" not in r and "correction_point" not in r for r in rows)
    via_method = AriadneRebase().prepare(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=sd,
        source_loader=data,
        target_loader=data,
        pairing=pairing,
        config=cfg,
        device="cpu",
    )
    assert _sha(via_method.task_vector) == PINNED_DEFAULT_SHA


def test_prepare_without_mid_alignment_unchanged(monkeypatch):
    source_base, _, target_base, data, pairing, _ = _setup(2, 4)
    cfg = parse_direct_residual_config(SEQ_D)
    requested = []
    real = streaming_module.iter_capture_tokens

    def spy(model, batches, requests, *args, **kwargs):
        requested.append(dict(requests))
        return real(model, batches, requests, *args, **kwargs)

    monkeypatch.setattr(streaming_module, "iter_capture_tokens", spy)
    default = _prepare(source_base, target_base, data, pairing, cfg)
    explicit = _prepare(source_base, target_base, data, pairing, cfg, mid_alignment=False)
    assert set(default) == set(explicit) == PINNED_PREPARE_KEYS
    for j in default["q_by_position"]:
        assert torch.equal(default["q_by_position"][j], explicit["q_by_position"][j])
    assert default["fingerprints"] == explicit["fingerprints"]
    assert all(kind == "boundary" for requests in requested for _, kind in requests.values())


# ---- mid residual ------------------------------------------------------------------------------------------------


def test_mid_residual_equals_block_input_plus_attention_branch():
    source_base, _, target_base, data, pairing, _ = _setup(3, 3)
    cfg = parse_direct_residual_config(TWO | {"sequential_points": "mid_end"})
    prepared = _prepare(source_base, target_base, data, pairing, cfg, mid_alignment=True)
    assert set(prepared) == PINNED_PREPARE_KEYS | {"q_mid_by_position", "mid_fingerprints"}

    def mids(model, images, j):
        x = model.visual.input(images)
        for block in model.visual.transformer.resblocks[:j]:
            x = block(x)
        return x + model.visual.transformer.resblocks[j].attn(x)

    with torch.no_grad():
        for j in range(3):
            i = pairing.pairing[j]
            cross = None
            xs, ys = [], []
            for k, batch in enumerate(prepared["target_batches"]):
                tgt_mid = mids(target_base, batch[0], j)
                out = tgt_mid.double()
                assert prepared["mid_fingerprints"][(k, j)] == (float(out.sum()), float((out**2).sum()))
                xs.append(mids(source_base, prepared["source_batches"][k][0], i).reshape(-1, 5).double())
                ys.append(tgt_mid.reshape(-1, 5).double())
            x, y = torch.cat(xs), torch.cat(ys)
            cross = (x - x.mean(0)).T @ (y - y.mean(0))
            assert torch.allclose(prepared["q_mid_by_position"][j], _procrustes_from_cross(cross).float(), atol=1e-4), j
            assert prepared["q_mid_by_position"][j].dtype == torch.float32


def test_mid_alignment_vision_only_and_polar_only():
    source_base, _, target_base, data, pairing, _ = _setup(2, 2)
    cfg = parse_direct_residual_config(SEQ_D)
    with pytest.raises(ValueError):
        _prepare(source_base, target_base, data, pairing, cfg, mid_alignment=True, alignment_map="ridge")
    with pytest.raises(ValueError):
        _prepare(source_base, target_base, data, pairing, cfg, mid_alignment=True, procrustes_rank=2)
    with pytest.raises(NotImplementedError, match="vision-only"):
        _prepare(source_base, target_base, data, pairing, cfg, mid_alignment=True, family_adapter=object())


# ---- block 0 -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", REGIMES)
def test_end_end_block0_o_equals_independent_o_bitwise(regime):
    tau, rows, prepared, (source_base, source_ft, target_base, data, pairing, sd) = _fit_two(regime, "end_end")
    cfg_i = parse_direct_residual_config(IND_O)
    ind, ind_rows = fit_direct_residual_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg_i, device="cpu"
    )
    keys = [k for k in ind if ".0." in k]
    assert keys and all(".attn.out_proj." in k for k in keys)
    for key in keys:
        assert torch.equal(ind[key], tau[key]), key
    o0 = next(r for r in rows if r["position"] == 0 and r["component"] == "attn.out_proj")
    assert o0["upstream_effect_norm"] == 0.0
    assert o0["relative_residual_after"] == ind_rows[0]["relative_residual_after"]


# ---- telescoping -------------------------------------------------------------------------------------------------


def _mounted_error(arm_data, tau, prepared, j, keep, point):
    """Brute force relative error at ``point`` of block ``j`` with only the ``keep`` entries of ``tau`` mounted."""
    source_base, source_ft, target_base, _, pairing, sd = arm_data
    mounted, pristine = deepcopy(target_base), deepcopy(target_base)
    mounted.load_state_dict({k: v + tau[k].to(v) if k in tau and keep(k) else v for k, v in sd.items()}, strict=True)
    pristine.load_state_dict(sd, strict=True)
    i = pairing.pairing[j]
    req = {"in": (j, "block_input"), "attn": (j, "attn_proj"), "out": (j, "boundary")}
    src_req = {"in": (i, "block_input"), "attn": (i, "attn_proj"), "out": (i, "boundary")}
    batches, src_batches = prepared["target_batches"], prepared["source_batches"]
    gens = (
        iter_capture_tokens(source_base, src_batches, src_req, "cpu"),
        iter_capture_tokens(source_ft, src_batches, src_req, "cpu"),
        iter_capture_tokens(mounted, batches, req, "cpu"),
        iter_capture_tokens(pristine, batches, req, "cpu"),
    )
    num = den = 0.0
    for sb, sf, m, p in zip(*gens, strict=True):
        if point == "mid":
            q = prepared["q_mid_by_position"][j]
            desired = ((sf["in"] + sf["attn"]) - (sb["in"] + sb["attn"])) @ q
            effect = (m["in"] + m["attn"]) - (p["in"] + p["attn"])
        else:
            desired = (sf["out"] - sb["out"]) @ prepared["q_by_position"][j]
            effect = m["out"] - p["out"]
        num += float(((effect - desired).double() ** 2).sum())
        den += float((desired.double() ** 2).sum())
    return num**0.5 / den**0.5


def _block_of(key):
    return int(re.search(r"resblocks\.(\d+)\.", key).group(1))


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("points", ARMS)
def test_telescoping_every_substep(regime, points):
    tau, rows, prepared, arm_data = _fit_two(regime, points)
    depth = arm_data[4].target_depth
    assert [(r["position"], r["component"]) for r in rows] == [
        (j, c) for j in range(depth) for c in ("attn.out_proj", "mlp.c_proj")
    ]
    for row in rows:
        j = row["position"]
        assert row["sequential_points"] == points and row["depth_fit"] == "sequential"
        assert row["relative_residual_after"] > 1e-3  # non-degenerate
        assert row["correction_point"] == (
            "mid" if points == "mid_end" and row["component"] == "attn.out_proj" else "end"
        )
        if row["component"] == "mlp.c_proj":
            # Block end after block j is fully mounted (blocks > j cannot affect boundary j).
            err = _mounted_error(arm_data, tau, prepared, j, lambda k, j=j: _block_of(k) <= j, "end")
            assert err == pytest.approx(row["relative_residual_after"], rel=1e-5), (j, "D")
        elif points == "mid_end":
            # Mid point after blocks < j and O_j are mounted (D_j cannot affect the mid stream).
            keep = lambda k, j=j: _block_of(k) < j or (_block_of(k) == j and ".attn.out_proj." in k)  # noqa: E731
            err = _mounted_error(arm_data, tau, prepared, j, keep, "mid")
            assert err == pytest.approx(row["relative_residual_after"], rel=1e-5), (j, "O")
    assert rows[0]["upstream_effect_norm"] == 0.0 and rows[-1]["upstream_effect_norm"] > 0.0


@pytest.mark.parametrize("points", ARMS)
def test_task_vector_keys_cover_both_components(points):
    tau, _, _, (_, _, _, _, pairing, sd) = _fit_two((2, 4), points)
    for j in range(pairing.target_depth):
        for comp in ("attn.out_proj", "mlp.c_proj"):
            base = f"visual.transformer.resblocks.{j}.{comp}"
            assert f"{base}.weight" in tau and f"{base}.bias" in tau
    assert set(tau) <= set(sd)


def test_two_points_differ_from_single_and_each_other():
    tau_e, _, _, arm = _fit_two((2, 4), "end_end")
    tau_m, _, _, _ = _fit_two((2, 4), "mid_end")
    source_base, source_ft, target_base, data, pairing, sd = arm
    cfg = parse_direct_residual_config(SEQ_D)
    prepared = _prepare(source_base, target_base, data, pairing, cfg)
    tau_s, _ = fit_direct_residual_sequential_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    assert _sha(tau_e) != _sha(tau_m)
    assert not set(tau_s) >= set(tau_e)


def test_method_dispatch_passes_mid_alignment():
    source_base, source_ft, target_base, data, pairing, sd = _setup(2, 4)
    for points in ARMS:
        cfg = parse_direct_residual_config({**TWO, "sequential_points": points})
        prep = AriadneRebase().prepare(
            source_base_model=source_base,
            source_ft_model=source_ft,
            target_model=target_base,
            target_base_sd=sd,
            source_loader=data,
            target_loader=data,
            pairing=pairing,
            config=cfg,
            device="cpu",
        )
        assert {r["sequential_points"] for r in prep.diagnostics} == {points}
        tau, _, _, _ = _fit_two((2, 4), points)
        assert _sha(tau) == _sha(prep.task_vector)


# ---- validation, serialization, state ----------------------------------------------------------------------------


@pytest.mark.parametrize("points", ARMS)
@pytest.mark.parametrize(
    "overrides",
    [
        {"depth_fit": "independent"},
        {"components": ("mlp.c_proj",)},
        {"components": ("attn.out_proj",)},
        {"block_split": "joint"},
        {"block_split": "backfit"},
        {"sequential_impl": "block_carry"},
        {"procrustes_source": "gradient"},
        {"activation_storage": "resident"},
    ],
)
def test_two_points_reject_incompatible_options(points, overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**TWO, "sequential_points": points, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"procrustes_rank": 2},
        {"procrustes_energy": 0.9},
        {"alignment_map": "ridge"},
        {"alignment_row_weighting": "cls_balanced"},
    ],
)
def test_mid_end_rejects_non_plain_alignment(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**TWO, "sequential_points": "mid_end", **overrides})
    if "alignment_map" not in overrides and "alignment_row_weighting" not in overrides:
        # end_end only needs the default fit, so the same options stay legal there as for any sequential fit.
        parse_direct_residual_config({**TWO, "sequential_points": "end_end", **overrides})


def test_single_rejects_two_components_and_unknown_value():
    with pytest.raises(ValueError):
        parse_direct_residual_config({**TWO, "sequential_points": "single"})
    with pytest.raises(ValueError):
        parse_direct_residual_config({**TWO, "sequential_points": "bogus"})
    with pytest.raises(ValueError):
        parse_direct_residual_config({**SEQ_D, "sequential_points": "end_end"})


def test_serialization_only_when_non_default():
    assert "sequential_points" not in direct_residual_config_dict(parse_direct_residual_config(SEQ_D))
    explicit = parse_direct_residual_config({**SEQ_D, "sequential_points": "single"})
    assert "sequential_points" not in direct_residual_config_dict(explicit)
    for points in ARMS:
        cfg = parse_direct_residual_config({**TWO, "sequential_points": points})
        assert direct_residual_config_dict(cfg)["sequential_points"] == points


def test_mid_end_rejected_on_decoders():
    class _Family:  # only needs to be non-None for the config-level check
        pass

    cfg = {**TWO, "sequential_points": "mid_end"}
    with pytest.raises(NotImplementedError, match="vision-only"):
        resolve_ariadne_decoder_config(cfg, _Family())


@pytest.mark.parametrize("points", ARMS)
def test_target_state_restored(points):
    source_base, source_ft, target_base, data, pairing, sd = _setup(2, 4)
    before = _state_sha(target_base)
    cfg = parse_direct_residual_config({**TWO, "sequential_points": points})
    prepared = _prepare(source_base, target_base, data, pairing, cfg, mid_alignment=points == "mid_end")
    fit_direct_residual_sequential_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    assert _state_sha(target_base) == before
    # Also on the error path (a mutated base state trips the fingerprint check).
    bad = {k: v.clone() for k, v in sd.items()}
    key = next(k for k in bad if "visual.input.weight" in k)
    bad[key] = bad[key] + 1.0
    with pytest.raises(RuntimeError, match="not the native base"):
        fit_direct_residual_sequential_streaming(
            target_base, bad, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
        )
    assert _state_sha(target_base) == before
