"""Tests for the opt-in Ariadne ``depth_fit="joint"`` (case J ~ I; ariadne_linear.tex Sec. 6.3).

All target blocks are solved together for ``sum_j ||S_j - D_j||^2 + sum_k lambda_k ||dW_k||^2`` with
``S_j = sum_{k<=j} (H_k dW_k + 1 b_k^T) L_k`` by preconditioned CG started from the independent solution.
(a) the CG result equals an explicit dense float64 solve of the normal equations built from hook-recomputed ``H_k``, ``D_j``
(vision extend / shrink / same-arch, with and without a LayerScale ``L``; decoder toy with right / left padding);
(b) one target block equals the independent fit; (c) the objective is non-increasing and below the independent solution's;
(d) the independent path is untouched (default serialization, hash equality at zero CG iterations); (e) config rejections.
Fixtures duplicated per the no-cross-test-import rule.
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
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY
from merge_and_rebase.rebase.methods._ariadne.config import direct_residual_config_dict, parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.joint import fit_direct_residual_joint_streaming
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.model_families import infer_family

WIDTH = 5


class _LayerScale(torch.nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.gamma = torch.nn.Parameter(0.5 + torch.rand(width, generator=gen))

    def forward(self, x):
        return x * self.gamma


class _Block(torch.nn.Module):
    def __init__(self, width, layerscale):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.out_proj = torch.nn.Linear(width, width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = _LayerScale(width, 7) if layerscale else torch.nn.Identity()

    def forward(self, x):
        return x + self.attn.out_proj(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth, layerscale):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width, layerscale) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth, layerscale):
        super().__init__()
        self.visual = _Visual(width, depth, layerscale)

    def encode_image(self, x):
        return self.visual(x)


def _setup(source_depth, target_depth, layerscale=False, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(WIDTH, source_depth, layerscale).eval()
    target_base = _Model(WIDTH, target_depth, layerscale).eval()
    source_ft = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.2 * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(0.2 * torch.randn_like(block.attn.out_proj.weight))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(6, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=2, shuffle=False)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, sd


BASE = dict(
    activation_storage="streaming",
    depth_fit="joint",
    num_batches=3,
    ridge_estimator="empirical_bayes",
    components=("mlp.c_proj",),
    joint_tol=1e-12,
    joint_max_iters=200,
)
REGIMES = [pytest.param(2, 4, id="extend"), pytest.param(4, 2, id="shrink"), pytest.param(3, 3, id="same_arch")]
KEY = "visual.transformer.resblocks.{}.mlp.c_proj"


def _sha(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _prepared(setup, cfg):
    source_base, _ft, target_base, data, pairing, _sd = setup
    return prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=cfg.num_batches, seed=cfg.seed, device="cpu"
    )


def _run_joint(setup, overrides=None):
    source_base, source_ft, target_base, _data, pairing, sd = setup
    cfg = parse_direct_residual_config({**BASE, **(overrides or {})})
    prepared = _prepared(setup, cfg)
    tau, rows = fit_direct_residual_joint_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )
    return tau, rows, prepared


def _run_independent(setup, overrides=None):
    source_base, source_ft, target_base, _data, pairing, sd = setup
    cfg = parse_direct_residual_config(
        {**BASE, "depth_fit": "independent", "joint_tol": 1e-6, "joint_max_iters": 50, **(overrides or {})}
    )
    prepared = _prepared(setup, cfg)
    return fit_direct_residual_streaming(
        target_base, sd, source_base, source_ft, prepared, pairing, config=cfg, device="cpu"
    )


def _boundaries(model, batches):
    blocks = model.visual.transformer.resblocks
    out = [[] for _ in blocks]
    handles = [
        b.register_forward_hook(lambda _m, _i, v, *, s=out[k]: s.append(v.detach().double()))
        for k, b in enumerate(blocks)
    ]
    try:
        with torch.no_grad():
            for batch in batches:
                model.encode_image(batch[0])
    finally:
        for h in handles:
            h.remove()
    return [torch.cat([x.reshape(-1, x.shape[-1]) for x in o]) for o in out]


def _cproj_inputs(model, batches, j):
    store = []
    h = model.visual.transformer.resblocks[j].mlp.c_proj.register_forward_hook(
        lambda _m, inputs, _v: store.append(inputs[0].detach().double())
    )
    try:
        with torch.no_grad():
            for batch in batches:
                model.encode_image(batch[0])
    finally:
        h.remove()
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in store])


def _dense_solve(hs, ds, outs, lams, with_bias):
    """Full normal equations ``(A^T A + Lambda) u = A^T d`` over u = row-major vec of every ``[dW_k; b_k^T]`` (float64)."""
    n_blocks = len(hs)
    ht = [torch.cat([h, torch.ones(h.shape[0], 1, dtype=h.dtype)], dim=1) if with_bias else h for h in hs]
    sizes = [ht[k].shape[1] * outs[k].shape[0] for k in range(n_blocks)]
    offsets = [sum(sizes[:k]) for k in range(n_blocks)]
    blocks = [torch.kron(ht[k].contiguous(), outs[k].T.contiguous()) for k in range(n_blocks)]  # row-major vec(Ht P L) = (Ht kron L^T) vec(P)
    total = sum(sizes)
    rows_per_block = hs[0].shape[0] * ds[0].shape[1]
    a = torch.zeros(n_blocks * rows_per_block, total, dtype=torch.float64)
    d = torch.cat([x.reshape(-1) for x in ds])
    for j in range(n_blocks):
        for k in range(j + 1):
            a[j * rows_per_block : (j + 1) * rows_per_block, offsets[k] : offsets[k] + sizes[k]] = blocks[k]
    lam = torch.zeros(total, dtype=torch.float64)
    for k in range(n_blocks):
        m = hs[k].shape[1]
        d_w = outs[k].shape[0]
        pen = torch.zeros(ht[k].shape[1], d_w, dtype=torch.float64)
        pen[:m] = lams[k]
        lam[offsets[k] : offsets[k] + sizes[k]] = pen.reshape(-1)
    u = torch.linalg.solve(a.T @ a + torch.diag(lam), a.T @ d)
    return [u[offsets[k] : offsets[k] + sizes[k]].reshape(ht[k].shape[1], outs[k].shape[0]) for k in range(n_blocks)]


def _rel(a, b):
    return float((a.double() - b.double()).norm() / max(float(b.double().norm()), 1e-30))


# (a) explicit dense solve ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layerscale", [False, True], ids=["identity_L", "layerscale_L"])
@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_cg_matches_dense_normal_equations_vision(source_depth, target_depth, layerscale):
    setup = _setup(source_depth, target_depth, layerscale)
    source_base, source_ft, target_base, data, pairing, _sd = setup
    tau, rows, prepared = _run_joint(setup)
    batches = list(data)
    sb, sf = _boundaries(source_base, batches), _boundaries(source_ft, batches)
    hs = [_cproj_inputs(target_base, batches, j) for j in range(target_depth)]
    ds = [
        (sf[pairing.pairing[j]] - sb[pairing.pairing[j]]) @ prepared["q_by_position"][j].double()
        for j in range(target_depth)
    ]
    outs = []
    for j in range(target_depth):
        ls = target_base.visual.transformer.resblocks[j].ls_2
        gamma = getattr(ls, "gamma", torch.ones(WIDTH))
        outs.append(torch.diag(gamma.detach().double()))
    lams = [float(r["ridge"]) for r in sorted(rows, key=lambda r: r["position"])]
    want = _dense_solve(hs, ds, outs, lams, with_bias=True)
    for j in range(target_depth):
        got_w = tau[KEY.format(j) + ".weight"].double().T
        got_b = tau[KEY.format(j) + ".bias"].double()
        assert _rel(got_w, want[j][:-1]) < 1e-6, f"dW j={j}"
        assert _rel(got_b, want[j][-1]) < 1e-6, f"b j={j}"
    run = rows[0]["joint_run"]
    assert run["converged"] and run["iterations"] >= 1


# (b) one block == independent ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth", [1, 2])
def test_single_target_block_equals_independent(source_depth):
    setup = _setup(source_depth, 1)
    tau_joint, rows, _ = _run_joint(setup)
    tau_ind, _rows_ind = _run_independent(setup)
    assert set(tau_joint) == set(tau_ind)
    for k in tau_ind:
        assert _rel(tau_joint[k], tau_ind[k]) < 1e-5, k
    assert rows[0]["joint_cumulative_rel_error"] == pytest.approx(rows[0]["independent_cumulative_rel_error"], rel=1e-5)


# (c) objective monotone and below the independent solution ----------------------------------------------------------


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_objective_non_increasing_and_below_independent(source_depth, target_depth):
    setup = _setup(source_depth, target_depth)
    _tau, rows, _ = _run_joint(setup, {"joint_tol": 1e-9})
    run = rows[0]["joint_run"]
    obj = run["objective_history"]
    assert len(obj) == run["iterations"] + 1 == len(run["residual_history"])
    assert all(b <= a * (1 + 1e-9) + 1e-12 for a, b in zip(obj, obj[1:], strict=False)), obj
    assert run["objective_joint"] <= run["objective_independent"] * (1 + 1e-9)
    if target_depth > 1:
        assert run["objective_joint"] < run["objective_independent"]
    # The recursion's final objective agrees with the exact objective from the evaluation sweep.
    assert obj[-1] == pytest.approx(run["objective_joint"], rel=1e-5)
    assert obj[0] == pytest.approx(run["objective_independent"], rel=1e-5)
    assert all(r["depth_fit"] == "joint" for r in rows)
    assert rows[-1]["joint_cumulative_rel_error"] <= rows[-1]["independent_cumulative_rel_error"] * (1 + 1e-6)


# (d) the independent path is untouched ---------------------------------------------------------------------------------


def test_default_serialization_and_zero_iteration_hash():
    default = direct_residual_config_dict(parse_direct_residual_config({}))
    assert not {"joint_weights", "joint_tol", "joint_max_iters"} & set(default)
    explicit = direct_residual_config_dict(
        parse_direct_residual_config({"joint_weights": "uniform", "joint_tol": 1e-6, "joint_max_iters": 50})
    )
    assert explicit == default
    joint_cfg = direct_residual_config_dict(parse_direct_residual_config({**BASE}))
    assert joint_cfg["depth_fit"] == "joint" and joint_cfg["joint_tol"] == 1e-12

    setup = _setup(2, 3)
    # A huge tolerance stops before the first CG iteration: the mounted correction is the independent one (to float32).
    tau_joint, rows, _ = _run_joint(setup, {"joint_tol": 1e9})
    tau_ind, _ = _run_independent(setup)
    assert rows[0]["joint_run"]["iterations"] == 0
    assert _sha(tau_joint) == _sha(tau_ind)


# (e) rejections ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"components": ("attn.out_proj", "mlp.c_proj")},
        {"block_target": "increment"},
        {"block_split": "joint"},
        {"residual_target": "transported_endpoint"},
        {"residual_target": "native_oracle"},
        {"pair_target_split": "per_block_share"},
        {"fit_blocks": (0,)},
        {"post_mlp_norm": "pre_norm"},
        {"v_fit": "preimage"},
        {"component_fit": "gradient_exact"},
        {"activation_storage": "resident"},
        {"ridge_estimator": "none"},
        {"stats_device": "cpu"},
        {"realization_diagnostics": True},
        {"joint_weights": "last"},
        {"joint_tol": 0.0},
        {"joint_tol": float("nan")},
        {"joint_tol": True},
        {"joint_max_iters": 0},
        {"joint_max_iters": 1.5},
        {"depth_fit": "independent", "joint_tol": 1e-9},
        {"depth_fit": "independent", "joint_max_iters": 10},
        {"depth_fit": "sequential", "joint_tol": 1e-9},
        {"depth_fit": "bogus"},
    ],
)
def test_config_rejections(overrides):
    base = {k: v for k, v in BASE.items() if k not in {"joint_tol", "joint_max_iters"}}
    with pytest.raises((ValueError, TypeError)):
        parse_direct_residual_config({**base, **overrides})


def test_valid_joint_config_parses():
    cfg = parse_direct_residual_config({**BASE})
    assert (cfg.depth_fit, cfg.joint_weights, cfg.joint_tol, cfg.joint_max_iters) == ("joint", "uniform", 1e-12, 200)


# decoder toy (padded batches) ---------------------------------------------------------------------------------------

TEXTS = CALIB_TEXTS * 2
DEC_REGIMES = [pytest.param(2, 3, id="extend"), pytest.param(3, 2, id="shrink")]


class _Texts(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.sample_ids = [f"t{i}" for i in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def _loader(side="right", max_length=12):
    tok = local_tokenizer(CALIB_TEXTS, padding_side=side)

    def collate(batch):
        return dict(tok(batch, padding="max_length", max_length=max_length, return_tensors="pt"))

    return DataLoader(_Texts(TEXTS), batch_size=2, shuffle=False, collate_fn=collate)


def _dec_cfg(**kw):
    base = dict(
        num_batches=len(TEXTS) // 2,
        components=("mlp.c_proj",),
        activation_storage="streaming",
        ridge_estimator="empirical_bayes",
        alignment_map="ridge",
        missing_bias="skip",
        exact_form=False,
        depth_fit="joint",
        joint_tol=1e-12,
        joint_max_iters=300,
    )
    return parse_direct_residual_config({**base, **kw})


def _dec_boundaries_and_h(model, batches):
    layers = model.model.layers
    out = [[] for _ in layers]
    hin = [[] for _ in layers]
    state = {}
    handles = []
    for k, layer in enumerate(layers):
        handles.append(
            layer.register_forward_hook(
                lambda _m, _i, v, *, k=k: out[k].append(
                    (v[0] if isinstance(v, tuple) else v).detach().double()[state["mask"]]
                )
            )
        )
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                lambda _m, inputs, *, k=k: hin[k].append(inputs[0].detach().double()[state["mask"]])
            )
        )
    try:
        with torch.no_grad():
            for batch in batches:
                state["mask"] = batch[ROW_MASK_KEY]
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(o) for o in out], [torch.cat(h) for h in hin]


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("source_depth,target_depth", DEC_REGIMES)
def test_cg_matches_dense_normal_equations_decoder(source_depth, target_depth, side):
    src = tiny_qwen2(layers=source_depth, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_qwen2(layers=target_depth, seed=2)
    fam = infer_family(src)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    cfg = _dec_cfg()
    prepared = prepare_direct_residual_streaming(
        src,
        tgt,
        _loader(side),
        _loader(side),
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
        family_adapter=fam,
        source_ft_model=ft,
        alignment_map="ridge",
    )
    assert sum(int((~b[ROW_MASK_KEY]).sum()) for b in prepared["source_batches"]) > 0, "fixture must contain padding"
    sd = {k: v.clone() for k, v in tgt.state_dict().items()}
    tau, rows = fit_direct_residual_joint_streaming(
        tgt, sd, src, ft, prepared, pairing, config=cfg, device="cpu", family_adapter=fam
    )
    sb, _ = _dec_boundaries_and_h(src, prepared["source_batches"])
    sf, _ = _dec_boundaries_and_h(ft, prepared["source_batches"])
    _, hs = _dec_boundaries_and_h(tgt, prepared["target_batches"])
    ds = [
        (sf[pairing.pairing[j]] - sb[pairing.pairing[j]]) @ prepared["q_by_position"][j].double()
        for j in range(target_depth)
    ]
    d_model = ds[0].shape[1]
    outs = [torch.eye(d_model, dtype=torch.float64)] * target_depth
    lams = [float(r["ridge"]) for r in sorted(rows, key=lambda r: r["position"])]
    want = _dense_solve(hs, ds, outs, lams, with_bias=False)
    assert not any(k.endswith(".bias") for k in tau)
    for j in range(target_depth):
        got = tau[f"model.layers.{j}.mlp.down_proj.weight"].double().T
        assert _rel(got, want[j]) < 1e-6, f"dW j={j}"
    run = rows[0]["joint_run"]
    assert run["converged"] and run["bias"] is False
    obj = run["objective_history"]
    assert all(b <= a * (1 + 1e-9) + 1e-12 for a, b in zip(obj, obj[1:], strict=False))
    assert run["objective_joint"] <= run["objective_independent"] * (1 + 1e-9)
