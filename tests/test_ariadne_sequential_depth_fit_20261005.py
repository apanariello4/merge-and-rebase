"""Tests for the opt-in ``depth_fit="sequential"`` Ariadne fit (blocks fitted in order across depth).

Block j is fitted against the target already carrying the fitted updates of blocks < j, with residual target
``E_j = D_j - U_j`` (``U_j`` the upstream effect at boundary j). Pinned here: default behaviour unchanged, block 0
equals the independent fit bit-for-bit, the exactness property (mounted realized error at boundary j equals block
j's fit residual), depth-1 equality, config validation/serialization, state restoration and a bias-free decoder.

Fixture duplicated from `tests/test_direct_residual_streaming.py` per this suite's no-cross-test-import convention.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_qwen2
from torch.utils.data import DataLoader, Dataset, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.methods._ariadne.capture import iter_capture_tokens
from merge_and_rebase.rebase.methods._ariadne.config import (
    DirectResidualConfig,
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    _streaming_desired,
    _streaming_source_iters,
    fit_direct_residual_sequential_streaming,
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase
from merge_and_rebase.rebase.model_families import infer_family


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


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
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


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


def _setup(source_depth, target_depth, width=5, seed=11, same_object=False):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = source_base if (same_object and source_depth == target_depth) else _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _fit_streaming(source_base, source_ft, target_base, data, pairing, target_base_sd, config):
    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=config.num_batches,
        seed=config.seed,
        device="cpu",
    )
    result = fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config,
        device="cpu",
    )
    return result, prepared


SEQ = dict(
    activation_storage="streaming",
    depth_fit="sequential",
    num_batches=3,
    ridge_relative=0.05,
    components=("mlp.c_proj",),
)
IND = {k: v for k, v in SEQ.items() if k != "depth_fit"}
REGIMES = [
    pytest.param(2, 4, id="extend"),
    pytest.param(4, 2, id="shrink"),
    pytest.param(3, 3, id="same_arch"),
]


def _sha(d):
    h = hashlib.sha256()
    for key in sorted(d):
        h.update(key.encode())
        h.update(d[key].numpy().tobytes())
    return h.hexdigest()


def _state_sha(model):
    return _sha({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()})


def _fit_both(source_base, source_ft, target_base, data, pairing, target_base_sd, overrides=None):
    """Independent and sequential fits from one shared Pass A."""
    cfg_i = parse_direct_residual_config({**IND, **(overrides or {})})
    cfg_s = parse_direct_residual_config({**SEQ, **(overrides or {})})
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg_i.num_batches, seed=cfg_i.seed, device="cpu"
    )
    args = (target_base, target_base_sd, source_base, source_ft, prepared, pairing)
    ind = fit_direct_residual_streaming(*args, config=cfg_i, device="cpu")
    seq = fit_direct_residual_sequential_streaming(*args, config=cfg_s, device="cpu")
    return ind, seq, prepared


def _mounted_relative_errors(
    target_model, base_sd, tau, prepared, pairing, source_base, source_ft, family_adapter=None
):
    """Brute force: ||T_j(base+tau) - T_j(base) - D_j|| / ||D_j|| per boundary on the calibration batches."""
    positions = list(range(pairing.target_depth))
    mounted = deepcopy(target_model)
    mounted.load_state_dict({k: v + tau[k].to(v) if k in tau else v for k, v in base_sd.items()}, strict=True)
    base = deepcopy(target_model)
    base.load_state_dict(base_sd, strict=True)
    req = {str(j): (j, "boundary") for j in positions}
    gens = _streaming_source_iters(source_base, source_ft, prepared, pairing, "cpu", family_adapter)
    gens["m"] = iter_capture_tokens(mounted, prepared["target_batches"], req, "cpu", family_adapter=family_adapter)
    gens["b"] = iter_capture_tokens(base, prepared["target_batches"], req, "cpu", family_adapter=family_adapter)
    num = dict.fromkeys(positions, 0.0)
    den = dict.fromkeys(positions, 0.0)
    for sb, sf, m, b in zip(*(gens[n] for n in ("sb", "sf", "m", "b")), strict=True):
        desired = _streaming_desired(prepared, pairing, "transported_delta", {"sb": sb, "sf": sf}, b, positions)
        for j in positions:
            num[j] += float(((m[str(j)] - b[str(j)] - desired[j]).double() ** 2).sum())
            den[j] += float((desired[j].double() ** 2).sum())
    return {j: num[j] ** 0.5 / den[j] ** 0.5 for j in positions}


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_default_depth_fit_hash_unchanged(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    cfg = parse_direct_residual_config({**IND})
    assert cfg.depth_fit == "independent"
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=3, seed=cfg.seed, device="cpu"
    )
    direct, _ = fit_direct_residual_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
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
    assert _sha(direct) == _sha(via_method.task_vector)
    assert all("depth_fit" not in r and "upstream_effect_norm" not in r for r in via_method.diagnostics)


def test_method_dispatch_uses_sequential_when_configured():
    source_base, source_ft, target_base, data, pairing, sd = _setup(2, 4)
    cfg = parse_direct_residual_config(SEQ)
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
    assert {r["depth_fit"] for r in prep.diagnostics} == {"sequential"}
    (ind, _), (seq, _), _ = _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    assert _sha(seq) == _sha({k: v for k, v in prep.task_vector.items()})
    assert _sha(seq) != _sha(ind)


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_block0_matches_independent_bitwise(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    (ind, ind_rows), (seq, seq_rows), _ = _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    assert set(ind) == set(seq)
    block0 = [k for k in ind if ".0." in k]
    assert block0
    for key in block0:
        assert torch.equal(ind[key], seq[key]), key
    assert seq_rows[0]["upstream_effect_norm"] == 0.0
    # Later blocks genuinely differ (non-trivial upstream effect), otherwise the test would be vacuous.
    assert seq_rows[-1]["upstream_effect_norm"] > 0.0
    assert any(not torch.equal(ind[k], seq[k]) for k in ind if ".0." not in k)
    # Row schema: independent schema plus the two new keys.
    assert set(seq_rows[0]) == set(ind_rows[0]) | {"depth_fit", "upstream_effect_norm"}
    assert [r["position"] for r in seq_rows] == list(range(target_depth))


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_mounted_error_equals_fit_residual(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    _, (seq, rows), prepared = _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    realized = _mounted_relative_errors(target_base, sd, seq, prepared, pairing, source_base, source_ft)
    for row in rows:
        j = row["position"]
        assert row["relative_residual_after"] > 1e-3  # non-degenerate
        assert realized[j] == pytest.approx(row["relative_residual_after"], rel=1e-5), j


def test_independent_fit_is_not_exact_when_mounted():
    """Negative control: the independent fit's mounted error differs from its per-block residual (upstream effects)."""
    source_base, source_ft, target_base, data, pairing, sd = _setup(2, 4)
    (ind, rows), _, prepared = _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    realized = _mounted_relative_errors(target_base, sd, ind, prepared, pairing, source_base, source_ft)
    assert any(realized[r["position"]] != pytest.approx(r["relative_residual_after"], rel=1e-3) for r in rows)


@pytest.mark.parametrize("source_depth, target_depth", [(1, 1), (3, 1)])
def test_single_target_block_equals_independent(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    (ind, ind_rows), (seq, seq_rows), _ = _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    assert set(ind) == set(seq)
    assert all(torch.equal(ind[k], seq[k]) for k in ind)
    assert _sha(ind) == _sha(seq)
    assert ind_rows[0]["relative_residual_after"] == seq_rows[0]["relative_residual_after"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"activation_storage": "resident"},
        {"block_split": "joint"},
        {"components": ("attn.out_proj", "mlp.c_proj")},
        {"tv_scaling": "global"},
        {"endpoint_construction": "sequential_source_endpoints", "activation_storage": "resident"},
        {"merge_mode": "merge_in_source_then_fit"},
        {"procrustes_source": "gradient"},
        {"residual_target": "transported_endpoint"},
        {"fidelity_holdout": True},
        {"realization_diagnostics": True},
        {"depth_fit": "bogus"},
    ],
)
def test_sequential_rejects_incompatible_options(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**SEQ, **overrides})


def test_depth_fit_serialization_only_when_non_default():
    default = DirectResidualConfig()
    assert "depth_fit" not in direct_residual_config_dict(default)
    assert "depth_fit" not in direct_residual_config_dict(parse_direct_residual_config({"depth_fit": "independent"}))
    assert direct_residual_config_dict(parse_direct_residual_config(SEQ))["depth_fit"] == "sequential"


def test_target_state_restored():
    source_base, source_ft, target_base, data, pairing, sd = _setup(2, 4)
    before = _state_sha(target_base)
    _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    assert _state_sha(target_base) == before
    # Also on the error path (mutated base state trips the fingerprint check).
    cfg = parse_direct_residual_config(SEQ)
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=3, seed=cfg.seed, device="cpu"
    )
    bad = {k: v.clone() for k, v in sd.items()}
    key = next(k for k in bad if "visual.input.weight" in k)
    bad[key] = bad[key] + 1.0
    with pytest.raises(RuntimeError, match="not the native base"):
        fit_direct_residual_sequential_streaming(
            target_base, bad, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
        )
    assert _state_sha(target_base) == before


# ---- bias-free decoder (Qwen2) with missing_bias="materialize" ---------------------------------------------------


class _Texts(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.sample_ids = [f"t{i}" for i in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def _text_loader(texts, max_length=12):
    tok = local_tokenizer(CALIB_TEXTS, padding_side="right")

    def collate(batch):
        return dict(tok(batch, padding="max_length", max_length=max_length, return_tensors="pt"))

    return DataLoader(_Texts(texts), batch_size=2, shuffle=False, collate_fn=collate)


@pytest.mark.parametrize("source_layers, target_layers", [(2, 3), (3, 2)])
def test_decoder_bias_free_materialize_exactness(source_layers, target_layers):
    texts = CALIB_TEXTS * 4
    src = tiny_qwen2(layers=source_layers, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_qwen2(layers=target_layers, seed=2)
    fam = infer_family(src)
    base_sd = {k: v.clone() for k, v in tgt.state_dict().items()}
    added = materialize_missing_projection_biases(tgt, base_sd, family_adapter=fam, components=("mlp.c_proj",))
    assert len(added) == target_layers
    cfg = resolve_ariadne_decoder_config(
        {"depth_fit": "sequential", "num_batches": len(texts) // 2, "alignment_map": "ridge"}, fam
    )
    assert cfg.depth_fit == "sequential" and cfg.missing_bias == "materialize"
    pairing = DiscreteLayerPairing.compute(source_layers, target_layers)
    before = _state_sha(tgt)
    prepared = prepare_direct_residual_streaming(
        src,
        tgt,
        _text_loader(texts),
        _text_loader(texts),
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
        family_adapter=fam,
        alignment_map="ridge",
    )
    tau, rows = fit_direct_residual_sequential_streaming(
        tgt, base_sd, src, ft, prepared, pairing, config=cfg, device="cpu", family_adapter=fam
    )
    assert _state_sha(tgt) == before
    assert any(k.endswith("down_proj.bias") for k in tau)
    realized = _mounted_relative_errors(tgt, base_sd, tau, prepared, pairing, src, ft, family_adapter=fam)
    for row in rows:
        assert row["relative_residual_after"] > 1e-3
        assert realized[row["position"]] == pytest.approx(row["relative_residual_after"], rel=1e-4), row["position"]


# ---- sequential_strength (2026-10-23): upstream blocks mounted at s times their unit update ---------------------------


def _mounted_relative_errors_at(target_model, base_sd, tau, prepared, pairing, source_base, source_ft, s, family_adapter=None):
    """||T_j(base + s tau) - T_j(base) - s D_j|| / ||s D_j|| per boundary (s = 1 is _mounted_relative_errors)."""
    positions = list(range(pairing.target_depth))
    mounted = deepcopy(target_model)
    mounted.load_state_dict({k: v + s * tau[k].to(v) if k in tau else v for k, v in base_sd.items()}, strict=True)
    base = deepcopy(target_model)
    base.load_state_dict(base_sd, strict=True)
    req = {str(j): (j, "boundary") for j in positions}
    gens = _streaming_source_iters(source_base, source_ft, prepared, pairing, "cpu", family_adapter)
    gens["m"] = iter_capture_tokens(mounted, prepared["target_batches"], req, "cpu", family_adapter=family_adapter)
    gens["b"] = iter_capture_tokens(base, prepared["target_batches"], req, "cpu", family_adapter=family_adapter)
    num = dict.fromkeys(positions, 0.0)
    den = dict.fromkeys(positions, 0.0)
    for sb, sf, m, b in zip(*(gens[n] for n in ("sb", "sf", "m", "b")), strict=True):
        desired = _streaming_desired(prepared, pairing, "transported_delta", {"sb": sb, "sf": sf}, b, positions)
        for j in positions:
            num[j] += float(((m[str(j)] - b[str(j)] - s * desired[j]).double() ** 2).sum())
            den[j] += float(((s * desired[j]).double() ** 2).sum())
    return {j: num[j] ** 0.5 / den[j] ** 0.5 for j in positions}


def _fit_both_strength(source_base, source_ft, target_base, data, pairing, target_base_sd, s):
    """As _fit_both, with sequential_strength=s on the sequential config only."""
    cfg_i = parse_direct_residual_config(IND)
    cfg_s = parse_direct_residual_config({**SEQ, "sequential_strength": s})
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg_i.num_batches, seed=cfg_i.seed, device="cpu"
    )
    args = (target_base, target_base_sd, source_base, source_ft, prepared, pairing)
    ind = fit_direct_residual_streaming(*args, config=cfg_i, device="cpu")
    seq = fit_direct_residual_sequential_streaming(*args, config=cfg_s, device="cpu")
    return ind, seq, prepared


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_sequential_strength_one_is_bitwise_default(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    _, (seq, _), _ = _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    _, (seq1, rows1), _ = _fit_both_strength(source_base, source_ft, target_base, data, pairing, sd, 1.0)
    assert _sha(seq) == _sha(seq1)
    assert "sequential_strength" not in rows1[0]


@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_sequential_strength_block0_is_independent_and_later_blocks_change(source_depth, target_depth):
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    (ind, _), (seq1, _), _ = _fit_both(source_base, source_ft, target_base, data, pairing, sd)
    _, (seqh, rows), _ = _fit_both_strength(source_base, source_ft, target_base, data, pairing, sd, 0.5)
    for key in [k for k in ind if ".0." in k]:  # the solve is linear in the target: (0.5 D_0) / 0.5 = D_0 up to rounding
        torch.testing.assert_close(seqh[key], ind[key], rtol=1e-5, atol=1e-7)
    later = [k for k in ind if ".0." not in k]
    assert any(not torch.allclose(seqh[k], seq1[k], rtol=1e-4, atol=1e-6) for k in later)
    assert all(r["sequential_strength"] == 0.5 for r in rows)


@pytest.mark.parametrize("s", [0.25, 0.5, 2.0])
@pytest.mark.parametrize("source_depth, target_depth", REGIMES)
def test_sequential_strength_mounted_error_equals_fit_residual(source_depth, target_depth, s):
    """Exactness at the fitted strength: mounting s * tau leaves, at boundary j, exactly block j's fit residual vs s D_j."""
    source_base, source_ft, target_base, data, pairing, sd = _setup(source_depth, target_depth)
    _, (seq, rows), prepared = _fit_both_strength(source_base, source_ft, target_base, data, pairing, sd, s)
    realized = _mounted_relative_errors_at(target_base, sd, seq, prepared, pairing, source_base, source_ft, s)
    for row in rows:
        j = row["position"]
        assert realized[j] == pytest.approx(row["relative_residual_after"], rel=1e-5), j


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequential_strength": 0.0},
        {"sequential_strength": -0.5},
        {"sequential_strength": float("nan")},
        {"sequential_strength": True},
        {"sequential_strength": 0.5, "depth_fit": "independent"},
        {"sequential_strength": 0.5, "sequential_impl": "block_carry"},
        {"sequential_strength": 0.5, "sequential_points": "end_end", "components": ("attn.out_proj", "mlp.c_proj")},
    ],
)
def test_sequential_strength_rejects(overrides):
    with pytest.raises(ValueError):
        parse_direct_residual_config({**SEQ, **overrides})


def test_sequential_strength_serialized_only_when_non_default():
    assert "sequential_strength" not in direct_residual_config_dict(DirectResidualConfig())
    assert direct_residual_config_dict(parse_direct_residual_config({**SEQ, "sequential_strength": 0.4}))["sequential_strength"] == 0.4


@pytest.mark.parametrize("source_layers, target_layers", [(2, 3), (3, 2)])
def test_decoder_sequential_strength_exactness(source_layers, target_layers):
    """Decoder (tiny Qwen2, fitted bias): at s = 0.5 the mounted s * tau leaves each block's fit residual vs s D_j."""
    texts = CALIB_TEXTS * 4
    src = tiny_qwen2(layers=source_layers, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_qwen2(layers=target_layers, seed=2)
    fam = infer_family(src)
    base_sd = {k: v.clone() for k, v in tgt.state_dict().items()}
    materialize_missing_projection_biases(tgt, base_sd, family_adapter=fam, components=("mlp.c_proj",))
    cfg = resolve_ariadne_decoder_config(
        {"depth_fit": "sequential", "num_batches": len(texts) // 2, "alignment_map": "ridge", "sequential_strength": 0.5}, fam
    )
    pairing = DiscreteLayerPairing.compute(source_layers, target_layers)
    prepared = prepare_direct_residual_streaming(
        src, tgt, _text_loader(texts), _text_loader(texts), pairing, num_batches=cfg.num_batches, seed=cfg.seed,
        device="cpu", family_adapter=fam, alignment_map="ridge",
    )
    tau, rows = fit_direct_residual_sequential_streaming(
        tgt, base_sd, src, ft, prepared, pairing, config=cfg, device="cpu", family_adapter=fam
    )
    assert any(k.endswith("down_proj.bias") for k in tau)
    realized = _mounted_relative_errors_at(tgt, base_sd, tau, prepared, pairing, src, ft, 0.5, family_adapter=fam)
    for row in rows:
        assert row["sequential_strength"] == 0.5
        assert realized[row["position"]] == pytest.approx(row["relative_residual_after"], rel=1e-4), row["position"]
