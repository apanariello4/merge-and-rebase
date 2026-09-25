"""DirectResidualConfig.fidelity_holdout diagnostic (dr_ablation_v2_20260925).

Covers: config validation; that turning the diagnostic on/off leaves tau
bit-identical (it must never feed any fit); that the drawn holdout sample
indices are disjoint from the calibration indices tau was fit on (asserted via
paired_calibration's own recorded index lists, not merely assumed); an
exact-fit construction where e_local==0 on both splits; that e_mounted==
e_local for a single-component model (no other component's effect to sum
into the mounted delta); and resident/streaming agreement.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    compute_fidelity_holdout_diagnostics,
    draw_fidelity_holdout_calibration,
    fit_direct_residual,
    fit_direct_residual_streaming,
    parse_direct_residual_config,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.eval.target_informed_runtime import _task_vector_sha256
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

# ---- config validation -----------------------------------------------------------


def test_config_accepts_fidelity_holdout():
    cfg = parse_direct_residual_config({"fidelity_holdout": True, "fidelity_holdout_batches": 4})
    assert cfg.fidelity_holdout is True
    assert cfg.fidelity_holdout_batches == 4


def test_config_default_fidelity_holdout_off():
    cfg = parse_direct_residual_config(None)
    assert cfg.fidelity_holdout is False
    assert cfg.fidelity_holdout_batches == 10


def test_config_fidelity_holdout_must_be_bool():
    with pytest.raises(TypeError):
        parse_direct_residual_config({"fidelity_holdout": 1})


def test_config_fidelity_holdout_batches_must_be_positive():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"fidelity_holdout_batches": 0})
    with pytest.raises(ValueError):
        parse_direct_residual_config({"fidelity_holdout_batches": -3})


# ---- fixture ---------------------------------------------------------------------


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


N_CLASSES = 6


def _loader(n, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n) % N_CLASSES), batch_size=2, shuffle=False)


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _setup(source_depth, target_depth, *, n=40, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(n=n, seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


# ---- disjointness ------------------------------------------------------------------


def test_holdout_indices_disjoint_from_calibration():
    setup = _setup(2, 4, n=40)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    src, tgt, meta = draw_fidelity_holdout_calibration(data, data, num_batches=5, holdout_batches=4, seed=89)
    assert meta["calibration_holdout_disjoint"] is True
    assert len(src) == 4
    assert len(tgt) == 4


def test_holdout_requires_seed():
    setup = _setup(2, 4, n=40)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    with pytest.raises(ValueError):
        draw_fidelity_holdout_calibration(data, data, num_batches=5, holdout_batches=4, seed=None)


def test_holdout_raises_when_not_enough_batches_available():
    setup = _setup(2, 4, n=12)  # only 6 batches of size 2 available
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    with pytest.raises(ValueError):
        draw_fidelity_holdout_calibration(data, data, num_batches=5, holdout_batches=4, seed=89)


# ---- tau bit-identical regardless of fidelity_holdout -----------------------------


@pytest.mark.parametrize("streaming", [False, True])
def test_fidelity_holdout_leaves_tau_bit_identical(streaming):
    setup = _setup(2, 4, n=40)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    kwargs = dict(num_batches=3, ridge_relative=0.05, fidelity_holdout_batches=4)
    if streaming:
        kwargs["activation_storage"] = "streaming"
    off_config = DirectResidualConfig(**kwargs, fidelity_holdout=False)
    on_config = DirectResidualConfig(**kwargs, fidelity_holdout=True)

    if streaming:
        prepared_off = prepare_direct_residual_streaming(
            source_base,
            target_base,
            data,
            data,
            pairing,
            num_batches=off_config.num_batches,
            seed=off_config.seed,
            device="cpu",
            source_ft_model=source_ft,
        )
        corr_off, _ = fit_direct_residual_streaming(
            target_base, target_base_sd, source_base, source_ft, prepared_off, pairing, config=off_config, device="cpu"
        )
        prepared_on = prepare_direct_residual_streaming(
            source_base,
            target_base,
            data,
            data,
            pairing,
            num_batches=on_config.num_batches,
            seed=on_config.seed,
            device="cpu",
            source_ft_model=source_ft,
        )
        corr_on, _ = fit_direct_residual_streaming(
            target_base, target_base_sd, source_base, source_ft, prepared_on, pairing, config=on_config, device="cpu"
        )
    else:
        captured = capture_paired_boundary_activations(
            source_base,
            source_ft,
            target_base,
            data,
            data,
            pairing,
            num_batches=off_config.num_batches,
            seed=off_config.seed,
            device="cpu",
        )
        desired = compute_desired_effects(captured, pairing, residual_target=off_config.residual_target)
        corr_off, _ = fit_direct_residual(
            target_base, target_base_sd, captured, desired, pairing, config=off_config, device="cpu"
        )
        corr_on, _ = fit_direct_residual(
            target_base, target_base_sd, captured, desired, pairing, config=on_config, device="cpu"
        )
    # fit_direct_residual[_streaming] itself never reads config.fidelity_holdout
    # (the diagnostic is computed separately, in vision_rebase.py, strictly
    # after tau is returned), so this must hold regardless -- but assert it
    # explicitly as the regression gate the spec requires.
    assert _task_vector_sha256(corr_off) == _task_vector_sha256(corr_on)
    for key in corr_off:
        assert torch.equal(corr_off[key], corr_on[key])


# ---- exact-fit construction: e_local == 0 -------------------------------------------


def test_e_local_zero_in_exact_fit_case():
    """Constructed exact-fit case: an UNTUNED source (`source_ft = source_base`
    exactly) makes D_j = (S_1 - S_0) @ Q_j identically zero for every image,
    calibration or holdout alike -- not merely small, exactly zero, by
    construction, regardless of Q_j or of what images are drawn. The zero
    correction (target_corrections = {} -- fit_direct_residual's own
    genuine output for a zero-desired-effect fit; no need to hand-construct
    one) then satisfies H_j @ 0 + 0 == D_j == 0 exactly, on both splits: the
    diagnostic's own numerator/denominator bookkeeping must recognize this
    (via the "||D_j|| == 0" guard, since 0/0 is otherwise undefined) rather
    than reporting a spurious nonzero ratio.
    """
    setup = _setup(1, 1, n=40, width=5)
    source_base, _source_ft_unused, target_base, data, pairing, target_base_sd = setup
    source_ft = deepcopy(source_base)  # untuned: zero fine-tuning delta

    config = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        components=("mlp.c_proj",),
        fidelity_holdout=True,
        fidelity_holdout_batches=3,
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=config.num_batches,
        seed=config.seed,
        device="cpu",
    )
    diag_out: dict = {}
    desired = compute_desired_effects(
        captured, pairing, residual_target=config.residual_target, diagnostics_out=diag_out
    )
    for j in range(pairing.target_depth):
        for batch in desired[j]:
            assert torch.equal(batch, torch.zeros_like(batch))
    corr, _rows = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu"
    )
    for key, value in corr.items():
        assert torch.equal(value, torch.zeros_like(value)), key

    q_by_position = {j: diag_out[j]["q"] for j in range(pairing.target_depth)}
    mu_s_by_position = {j: diag_out[j]["mu_s"] for j in range(pairing.target_depth)}
    mu_t_by_position = {j: diag_out[j]["mu_t"] for j in range(pairing.target_depth)}
    result = compute_fidelity_holdout_diagnostics(
        source_base,
        source_ft,
        target_base,
        target_base_sd,
        corr,
        data,
        data,
        pairing,
        config=config,
        q_by_position=q_by_position,
        mu_s_by_position=mu_s_by_position,
        mu_t_by_position=mu_t_by_position,
        device="cpu",
    )
    for split in ("calibration", "holdout"):
        for j, row in result["splits"][split]["e_local_by_position"].items():
            assert row["mlp.c_proj"]["e_local"] == 0.0, (split, j, row)
            assert row["mlp.c_proj"]["numerator"] == pytest.approx(0.0, abs=1e-9), (split, j, row)
            assert row["mlp.c_proj"]["desired_norm"] == pytest.approx(0.0, abs=1e-9), (split, j, row)


# ---- e_mounted == e_local for a single-block model --------------------------------


def test_e_mounted_equals_e_local_single_component():
    setup = _setup(1, 1, n=40, width=5)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    config = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        components=("mlp.c_proj",),
        fidelity_holdout=True,
        fidelity_holdout_batches=3,
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=config.num_batches,
        seed=config.seed,
        device="cpu",
    )
    diag_out: dict = {}
    desired = compute_desired_effects(
        captured, pairing, residual_target=config.residual_target, diagnostics_out=diag_out
    )
    corr, _rows = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu"
    )
    q_by_position = {j: diag_out[j]["q"] for j in range(pairing.target_depth)}
    mu_s_by_position = {j: diag_out[j]["mu_s"] for j in range(pairing.target_depth)}
    mu_t_by_position = {j: diag_out[j]["mu_t"] for j in range(pairing.target_depth)}
    result = compute_fidelity_holdout_diagnostics(
        source_base,
        source_ft,
        target_base,
        target_base_sd,
        corr,
        data,
        data,
        pairing,
        config=config,
        q_by_position=q_by_position,
        mu_s_by_position=mu_s_by_position,
        mu_t_by_position=mu_t_by_position,
        device="cpu",
    )
    for split in ("calibration", "holdout"):
        for j in result["splits"][split]["e_local_by_position"]:
            e_local = result["splits"][split]["e_local_by_position"][j]["mlp.c_proj"]["e_local"]
            e_mounted = result["splits"][split]["e_mounted_by_position"][j]["e_mounted"]
            assert e_mounted == pytest.approx(e_local, rel=1e-4, abs=1e-6), (split, j)


# ---- resident/streaming agreement ---------------------------------------------------


def test_fidelity_holdout_resident_streaming_agree():
    setup = _setup(2, 4, n=40)
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    common = dict(num_batches=3, ridge_relative=0.05, fidelity_holdout=True, fidelity_holdout_batches=3)
    config_r = DirectResidualConfig(**common)
    config_s = DirectResidualConfig(**common, activation_storage="streaming")

    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=config_r.num_batches,
        seed=config_r.seed,
        device="cpu",
    )
    diag_out: dict = {}
    desired = compute_desired_effects(
        captured, pairing, residual_target=config_r.residual_target, diagnostics_out=diag_out
    )
    corr_r, _rows_r = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=config_r, device="cpu"
    )
    q_r = {j: diag_out[j]["q"] for j in range(pairing.target_depth)}
    mu_s_r = {j: diag_out[j]["mu_s"] for j in range(pairing.target_depth)}
    mu_t_r = {j: diag_out[j]["mu_t"] for j in range(pairing.target_depth)}
    result_r = compute_fidelity_holdout_diagnostics(
        source_base,
        source_ft,
        target_base,
        target_base_sd,
        corr_r,
        data,
        data,
        pairing,
        config=config_r,
        q_by_position=q_r,
        mu_s_by_position=mu_s_r,
        mu_t_by_position=mu_t_r,
        device="cpu",
    )

    prepared = prepare_direct_residual_streaming(
        source_base,
        target_base,
        data,
        data,
        pairing,
        num_batches=config_s.num_batches,
        seed=config_s.seed,
        device="cpu",
        source_ft_model=source_ft,
    )
    corr_s, _rows_s = fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config_s,
        device="cpu",
    )
    q_s = prepared["q_by_position"]
    mu_s_s = {j: m.float() for j, m in prepared["source_mean_by_position"].items()}
    mu_t_s = {j: m.float() for j, m in prepared["target_mean_by_position"].items()}
    result_s = compute_fidelity_holdout_diagnostics(
        source_base,
        source_ft,
        target_base,
        target_base_sd,
        corr_s,
        data,
        data,
        pairing,
        config=config_s,
        q_by_position=q_s,
        mu_s_by_position=mu_s_s,
        mu_t_by_position=mu_t_s,
        device="cpu",
    )

    assert result_r["holdout_indices_sha256"] == result_s["holdout_indices_sha256"]
    assert result_r["calibration_indices_sha256"] == result_s["calibration_indices_sha256"]
    for split in ("calibration", "holdout"):
        for j in result_r["splits"][split]["e_mounted_by_position"]:
            a = result_r["splits"][split]["e_mounted_by_position"][j]["e_mounted"]
            b = result_s["splits"][split]["e_mounted_by_position"][j]["e_mounted"]
            assert a == pytest.approx(b, rel=1e-4, abs=1e-6), (split, j)
