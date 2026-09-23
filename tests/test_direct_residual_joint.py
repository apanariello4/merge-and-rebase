"""Tests for Direct Residual's ``block_split='joint'`` closed-form joint ridge
fit (``component_target='block_boundary'`` only).

``block_split='none'`` (default, historical) fits every requested residual
writer independently against the SAME block-boundary target ``D_j``.
``'backfit'`` resolves the resulting double-target overshoot with iterative
intra-block Gauss-Seidel replay of the block's own nonlinearity (see
``tests/test_direct_residual_backfit.py``). ``'joint'`` instead solves ONE
closed-form ridge on the STACKED ``(attn.out_proj, mlp.c_proj)`` features in a
single linear-algebra step, under the explicit first-order approximation that
the MLP does not respond to a change in ``attn.out_proj`` -- see
``target_informed_runtime._fit_block_boundary_joint``'s docstring for the
exact objective, the lambda_c convention (each component's own standalone
``block_split='none'`` ridge), and the bias convention (the whole joint
intercept goes to ``mlp.c_proj.bias``).

This module deliberately has no cross-test imports (same convention as
``test_direct_residual_backfit.py``/``test_direct_residual_component_
coverage.py``): every fixture below is self-contained.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
    parse_direct_residual_config,
)
from merge_and_rebase.eval.target_informed_runtime import (
    _JointBlockRidgeStatistics,
    paired_calibration,
)
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")),
]

# --------------------------------------------------------------------------
# Fixture 1: realistic GELU block (IDENTICAL to test_direct_residual_backfit
# .py's own fixture, so the block_split='none' golden hashes recorded in
# test_direct_residual_component_coverage.py remain the reference for
# "joint reduces to none with one component").
# --------------------------------------------------------------------------


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


def _setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _fit(source_depth, target_depth, config, **setup_kwargs):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(
        source_depth, target_depth, **setup_kwargs
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
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=config,
        device="cpu",
    )
    return corrections, diagnostics, target_base, target_base_sd


def _state_dict_sha256(d: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# 0. Config/parser.
# --------------------------------------------------------------------------


def test_block_split_joint_parses():
    cfg = parse_direct_residual_config({"components": ["attn.out_proj", "mlp.c_proj"], "block_split": "joint"})
    assert cfg.block_split == "joint"


def test_block_split_joint_requires_block_boundary():
    with pytest.raises(ValueError, match="block_boundary"):
        parse_direct_residual_config(
            {"component_target": "output_local", "components": ["attn.out_proj"], "block_split": "joint"}
        )


def test_block_split_joint_rejects_internal_components():
    # Internal components (q/k/v/c_fc) are unreachable under block_boundary at
    # all -- a separate, pre-existing rejection reached before block_split is
    # ever checked -- so this exercises that block_split='joint' does not
    # relax it.
    with pytest.raises(ValueError, match="unknown components"):
        parse_direct_residual_config({"components": ["attn.q_proj"], "block_split": "joint"})


def test_block_split_invalid_value_rejected():
    with pytest.raises(ValueError, match="block_split"):
        parse_direct_residual_config({"block_split": "sideways"})


# --------------------------------------------------------------------------
# (a) Golden hash for block_split='none' is unaffected by adding 'joint'
# (independent of the full regression suites for none/output_local/backfit,
# which were run separately and are unchanged -- see the task report).
# --------------------------------------------------------------------------

_GOLDEN_EXTEND = "e6f02b6922df921f802cb02ca245a60f6f988d44a70a2425061c7cd10f2e6dbc"
_GOLDEN_SHRINK = "111eafdb947b940ebee0366366c5d726830ff6661d8f78222f0a1c67f13dfe94"
_GOLDEN_SAME_ARCH = "3b0600d6b14887867dc83d8d026903dea27bf0de20baaaa5055848fb0385c9d3"


@pytest.mark.parametrize(
    "source_depth, target_depth, golden",
    [
        (2, 4, _GOLDEN_EXTEND),
        (4, 2, _GOLDEN_SHRINK),
        (3, 3, _GOLDEN_SAME_ARCH),
    ],
)
def test_golden_hash_unaffected_by_joint_block_split(source_depth, target_depth, golden):
    cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05)  # block_split defaults to "none"
    corrections, _diag, _model, _sd = _fit(source_depth, target_depth, cfg)
    assert _state_dict_sha256(corrections) == golden


# --------------------------------------------------------------------------
# (b) Single-component joint == block_split='none', bitwise (the joint path
# delegates to the identical ResidualSufficientStatistics call for a single
# requested component -- see _fit_block_boundary_joint's docstring).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
@pytest.mark.parametrize("components", [("attn.out_proj",), ("mlp.c_proj",)])
def test_single_component_joint_matches_none_bitwise(source_depth, target_depth, components):
    none_cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=components, block_split="none")
    joint_cfg = DirectResidualConfig(num_batches=3, ridge_relative=0.05, components=components, block_split="joint")
    corrections_none, diag_none, _m1, _sd1 = _fit(source_depth, target_depth, none_cfg)
    corrections_joint, diag_joint, _m2, _sd2 = _fit(source_depth, target_depth, joint_cfg)
    assert set(corrections_none) == set(corrections_joint)
    for key in corrections_none:
        torch.testing.assert_close(corrections_none[key], corrections_joint[key], rtol=0, atol=0)
    for row_none, row_joint in zip(diag_none, diag_joint, strict=True):
        assert row_none["ridge"] == row_joint["ridge"]
        assert row_none["residual_norm_after"] == row_joint["residual_norm_after"]


# --------------------------------------------------------------------------
# Toy linear-additive block: out = x + out_proj(x) + c_proj(x). Both writers
# consume the block's own raw input x directly (no ln_1/ln_2, no GELU), so
# H_out = H_cproj = x exactly and there is no block nonlinearity between them
# -- the joint model's J_M=0 approximation is then EXACT, not merely a
# first-order one, so the joint closed form, a hand-derived stacked ridge
# solve, and the converged safeguarded backfit must all agree.
# --------------------------------------------------------------------------


class _ToyAttn(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _ToyMlp(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.c_proj(x)


class _ToyBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _ToyAttn(width)
        self.mlp = _ToyMlp(width)

    def forward(self, x):
        return x + self.attn(x) + self.mlp(x)


class _ToyVisual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_ToyBlock(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _ToyModel(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _ToyVisual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _toy_setup(depth=1, width=6, seed=7):
    torch.manual_seed(seed)
    source_base = _ToyModel(width, depth).eval()
    target_base = _ToyModel(width, depth).eval()
    tuned = deepcopy(source_base)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.attn.out_proj.weight.add_(0.3 * torch.randn_like(block.attn.out_proj.weight))
            block.attn.out_proj.bias.add_(0.1 * torch.randn_like(block.attn.out_proj.bias))
            block.mlp.c_proj.weight.add_(0.3 * torch.randn_like(block.mlp.c_proj.weight))
            block.mlp.c_proj.bias.add_(0.1 * torch.randn_like(block.mlp.c_proj.bias))
    generator = torch.Generator().manual_seed(seed + 2)
    images = torch.randn(24, 5, 4, generator=generator)
    data = DataLoader(TensorDataset(images, torch.arange(24)), batch_size=4, shuffle=False)
    pairing = DiscreteLayerPairing.compute(depth, depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, tuned, target_base, data, pairing, target_base_sd


def _stacked_joint_ridge_with_intercept(
    h_o: torch.Tensor,
    h_d: torch.Tensor,
    d_rows: torch.Tensor,
    lam_o: float,
    lam_d: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent, from-scratch closed-form joint ridge (genuinely
    reimplemented from the stacked normal equations here, NOT calling
    ``_JointBlockRidgeStatistics`` or anything else production code uses):

        min_{Wo,Wd,b} ||Ho Wo^T + Hd Wd^T + 1 b^T - D||_F^2
                      + lam_o ||Wo||_F^2 + lam_d ||Wd||_F^2

    Centered normal equations with block-diagonal ridge, solved directly via
    ``torch.linalg.solve`` on the stacked design -- a brute-force reference
    distinct in implementation from ``_JointBlockRidgeStatistics.solve``'s own
    streaming-Gram machinery, though mathematically the same objective.
    """
    ho, hd, d = h_o.double(), h_d.double(), d_rows.double()
    n, d_o = ho.shape
    _, d_d = hd.shape
    a = torch.cat([ho, hd], dim=1)
    mu_a = a.mean(dim=0)
    mu_d = d.mean(dim=0)
    ac = a - mu_a
    dc = d - mu_d
    lam_vec = torch.cat(
        [
            torch.full((d_o,), float(lam_o), dtype=torch.float64),
            torch.full((d_d,), float(lam_d), dtype=torch.float64),
        ]
    )
    gram = ac.T @ ac + torch.diag(lam_vec)
    x = torch.linalg.solve(gram, ac.T @ dc)
    b = mu_d - x.T @ mu_a
    w_o = x[:d_o, :].T.float()
    w_d = x[d_o:, :].T.float()
    return w_o, w_d, b.float(), b.float()


def _single_component_ridge(h_rows: torch.Tensor, d_rows: torch.Tensor, ridge_relative: float) -> float:
    """Reproduces the SAME lambda ``ResidualSufficientStatistics.solve`` would
    return for one component fit alone against ``d_rows`` -- used only to
    corroborate ``_JointBlockRidgeStatistics``'s lambda_c against an
    independently-derived value in test (e).
    """
    h = h_rows.double()
    n, d_in = h.shape
    mu_h = h.mean(dim=0)
    hc = h - mu_h
    trace_sc = float(torch.trace(hc.T @ hc).item())
    return float(ridge_relative) * trace_sc / d_in


def test_toy_linear_additive_block_joint_matches_derived_closed_form_and_converged_backfit():
    ridge_relative = 0.1
    source_base, source_ft, target_base, data, pairing, target_base_sd = _toy_setup()

    joint_cfg = DirectResidualConfig(
        num_batches=6,
        ridge_relative=ridge_relative,
        components=("attn.out_proj", "mlp.c_proj"),
        block_split="joint",
        seed=89,
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=joint_cfg.num_batches,
        seed=joint_cfg.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    joint_corrections, joint_diagnostics = fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=joint_cfg,
        device="cpu",
    )

    backfit_cfg = DirectResidualConfig(
        num_batches=6,
        ridge_relative=ridge_relative,
        components=("attn.out_proj", "mlp.c_proj"),
        block_split="backfit",
        backfit_max_iters=2000,
        backfit_tol=1e-9,
        seed=89,
    )
    backfit_corrections, _backfit_diag = fit_direct_residual(
        deepcopy(target_base),
        target_base_sd,
        captured,
        desired,
        pairing,
        config=backfit_cfg,
        device="cpu",
    )

    _sb, target_batches, _meta = paired_calibration(data, data, num_batches=joint_cfg.num_batches, seed=joint_cfg.seed)
    with torch.no_grad():
        x_batches = [target_base.visual.input(b[0]) for b in target_batches]
    h_rows = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_batches], dim=0)
    d_rows = torch.cat([d.reshape(-1, d.shape[-1]) for d in desired[0]], dim=0)

    # Both writers share the identical input in this fixture, so lam_o==lam_d
    # by construction: use the same brute-force lambda for each.
    lam = _single_component_ridge(h_rows, d_rows, ridge_relative)
    w_o_ref, w_d_ref, b_o_ref, b_d_ref = _stacked_joint_ridge_with_intercept(h_rows, h_rows, d_rows, lam, lam)

    key_out = "visual.transformer.resblocks.0.attn.out_proj.weight"
    key_cproj = "visual.transformer.resblocks.0.mlp.c_proj.weight"
    bias_out = "visual.transformer.resblocks.0.attn.out_proj.bias"
    bias_cproj = "visual.transformer.resblocks.0.mlp.c_proj.bias"

    # Joint vs. the hand-derived stacked closed form: exact (float32 solve
    # noise only), no Gauss-Seidel iteration is involved on either side.
    torch.testing.assert_close(joint_corrections[key_out], w_o_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(joint_corrections[key_cproj], w_d_ref, rtol=1e-4, atol=1e-5)
    # Bias convention: the whole intercept sits on mlp.c_proj.bias; out_proj's
    # bias gets an exact zero delta (see _fit_block_boundary_joint's docstring).
    torch.testing.assert_close(
        joint_corrections[bias_out],
        torch.zeros_like(joint_corrections[bias_out]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(joint_corrections[bias_cproj], b_o_ref, rtol=1e-4, atol=1e-5)

    # Joint vs. the converged safeguarded backfit: agrees on the COMBINED
    # per-writer effect (Wo+bo, Wd+bd combined into the total S=Wo+Wd since
    # backfit's own bias split convention differs from joint's -- see
    # test_direct_residual_backfit.py's docstring: backfit only pins down
    # Wo+Wd and bo+bd, not the individual split).
    joint_combined_w = joint_corrections[key_out] + joint_corrections[key_cproj]
    backfit_combined_w = backfit_corrections[key_out] + backfit_corrections[key_cproj]
    torch.testing.assert_close(joint_combined_w, backfit_combined_w, rtol=0, atol=2e-3)
    joint_combined_b = joint_corrections[bias_out] + joint_corrections[bias_cproj]
    backfit_combined_b = backfit_corrections[bias_out] + backfit_corrections[bias_cproj]
    torch.testing.assert_close(joint_combined_b, backfit_combined_b, rtol=0, atol=2e-3)


# --------------------------------------------------------------------------
# (d) Brute-force check on a small random problem, unrelated to any Direct
# Residual fixture: builds synthetic H_O/H_D/D directly and compares
# _JointBlockRidgeStatistics.solve against numpy/torch's OWN direct solve of
# the stacked normal equations with block-diagonal lambda (a second,
# independent reimplementation from test (c)'s, at generic -- not necessarily
# equal -- per-component lambdas and non-square d_out).
# --------------------------------------------------------------------------


def test_joint_block_ridge_statistics_matches_brute_force_stacked_solve():
    torch.manual_seed(0)
    n, d_o, d_d, d_out = 200, 7, 5, 4
    h_o = torch.randn(n, d_o, dtype=torch.float64)
    h_d = torch.randn(n, d_d, dtype=torch.float64)
    d_rows = torch.randn(n, d_out, dtype=torch.float64)
    lam_o, lam_d = 0.37, 1.9

    stats = _JointBlockRidgeStatistics([d_o, d_d], d_out)
    # Streamed across two unequal-sized batches, to also exercise the
    # accumulation path (not a single update call).
    stats.update([h_o[:80], h_d[:80]], d_rows[:80])
    stats.update([h_o[80:], h_d[80:]], d_rows[80:])
    x, beta, _diag = stats.solve([lam_o, lam_d])

    a = torch.cat([h_o, h_d], dim=1)
    mu_a = a.mean(dim=0)
    mu_d = d_rows.mean(dim=0)
    ac = a - mu_a
    dc = d_rows - mu_d
    lam_vec = torch.cat(
        [
            torch.full((d_o,), lam_o, dtype=torch.float64),
            torch.full((d_d,), lam_d, dtype=torch.float64),
        ]
    )
    x_ref = torch.linalg.solve(ac.T @ ac + torch.diag(lam_vec), ac.T @ dc)
    beta_ref = mu_d - x_ref.T @ mu_a

    torch.testing.assert_close(x, x_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(beta, beta_ref, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------
# (e) lambda_c from the joint path equals the single-component solver's own
# diag["ridge"], for both components.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_depth, target_depth", [(2, 4), (4, 2), (3, 3)])
def test_joint_lambda_matches_single_component_solver_ridge(source_depth, target_depth):
    ridge_relative = 0.07
    joint_cfg = DirectResidualConfig(
        num_batches=3,
        ridge_relative=ridge_relative,
        components=("attn.out_proj", "mlp.c_proj"),
        block_split="joint",
    )
    _corrections, joint_diagnostics, _model, _sd = _fit(source_depth, target_depth, joint_cfg)

    for component in ("attn.out_proj", "mlp.c_proj"):
        standalone_cfg = DirectResidualConfig(
            num_batches=3,
            ridge_relative=ridge_relative,
            components=(component,),
            block_split="none",
        )
        _c, standalone_diag, _m, _s = _fit(source_depth, target_depth, standalone_cfg)
        by_position = {(row["position"], row["component"]): row for row in joint_diagnostics}
        for row in standalone_diag:
            joint_row = by_position[(row["position"], component)]
            assert joint_row["ridge"] == pytest.approx(row["ridge"], rel=1e-12, abs=1e-12)
            assert joint_row["joint_lambda"][component] == pytest.approx(row["ridge"], rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------
# (f) open_clip integration end-to-end: finite outputs for extend/shrink/
# same_arch, device-parametrized.
# --------------------------------------------------------------------------

from open_clip.transformer import VisionTransformer  # noqa: E402


class _CLIPLike(torch.nn.Module):
    def __init__(self, visual: VisionTransformer):
        super().__init__()
        self.visual = visual

    def encode_image(self, x):
        return self.visual(x)


def _make_vit(*, image_size, patch_size, width, layers, heads, seed):
    torch.manual_seed(seed)
    vt = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        mlp_ratio=2.0,
        ls_init_value=None,
        output_dim=width,
        pool_type="tok",
    )
    return _CLIPLike(vt).eval()


def _tuned_vit_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.mlp.c_fc.weight.add_(scale * torch.randn_like(block.mlp.c_fc.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
            block.attn.in_proj_weight.add_(scale * torch.randn_like(block.attn.in_proj_weight))
    return tuned


class _IdentityTensorDataset(TensorDataset):
    def __init__(self, images, sample_ids):
        super().__init__(images, torch.arange(len(sample_ids)))
        self.sample_ids = list(sample_ids)


def _vit_loader(image_size, n=6, seed=0, sample_ids=None):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, image_size, image_size, generator=generator)
    ids = sample_ids if sample_ids is not None else [str(i) for i in range(n)]
    return DataLoader(_IdentityTensorDataset(images, ids), batch_size=2, shuffle=False)


_VIT_DIRECTIONS = {
    "extend": dict(
        source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
        target=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3),
    ),
    "shrink": dict(
        source=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3),
        target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
    ),
    "same_arch": dict(
        source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
        target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
    ),
}


def _vit_direction_setup(direction, seed=101):
    spec = _VIT_DIRECTIONS[direction]
    source_base = _make_vit(seed=seed, **spec["source"])
    target_base = _make_vit(seed=seed + 1, **spec["target"])
    source_ft = _tuned_vit_copy(source_base, seed=seed + 2)
    shared_ids = [str(i) for i in range(6)]
    source_loader = _vit_loader(spec["source"]["image_size"], seed=seed + 3, sample_ids=shared_ids)
    target_loader = _vit_loader(spec["target"]["image_size"], seed=seed + 4, sample_ids=shared_ids)
    pairing = DiscreteLayerPairing.compute(spec["source"]["layers"], spec["target"]["layers"])
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
@pytest.mark.parametrize("device", DEVICES)
def test_block_boundary_joint_open_clip_end_to_end_finite(direction, device):
    source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd = _vit_direction_setup(
        direction
    )
    cfg = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        components=("attn.out_proj", "mlp.c_proj"),
        block_split="joint",
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        source_loader,
        target_loader,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device=device,
    )
    desired = compute_desired_effects(captured, pairing)
    corrections, diagnostics = fit_direct_residual(
        target_base,
        target_base_sd,
        captured,
        desired,
        pairing,
        config=cfg,
        device=device,
    )
    assert diagnostics
    for key, value in corrections.items():
        assert torch.isfinite(value).all(), key
        assert (".attn.out_proj." in key) or (".mlp.c_proj." in key), key
    for row in diagnostics:
        assert math.isfinite(row["joint_residual_relative"]) or set(cfg.components) == {row["component"]}


# --------------------------------------------------------------------------
# (g) The live/pretrained target model is never mutated by the joint fit.
# --------------------------------------------------------------------------


def test_joint_never_mutates_the_live_target_model():
    cfg = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        components=("attn.out_proj", "mlp.c_proj"),
        block_split="joint",
    )
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(2, 4)
    before_hash = _state_dict_sha256(target_base.state_dict())
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    fit_direct_residual(target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu")
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash


def test_joint_single_component_never_mutates_the_live_target_model():
    cfg = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        components=("mlp.c_proj",),
        block_split="joint",
    )
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(2, 4)
    before_hash = _state_dict_sha256(target_base.state_dict())
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    fit_direct_residual(target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu")
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash


# --------------------------------------------------------------------------
# realization_diagnostics is analysis-only for 'joint' too.
# --------------------------------------------------------------------------


def test_joint_realization_diagnostics_does_not_change_fitted_corrections():
    base_kwargs = dict(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"), block_split="joint"
    )
    cfg_off = DirectResidualConfig(realization_diagnostics=False, **base_kwargs)
    cfg_on = DirectResidualConfig(realization_diagnostics=True, **base_kwargs)
    corrections_off, _diag_off, _m1, _sd1 = _fit(2, 4, cfg_off)
    corrections_on, _diag_on, _m2, _sd2 = _fit(2, 4, cfg_on)
    assert _state_dict_sha256(corrections_off) == _state_dict_sha256(corrections_on)


def test_joint_realization_diagnostics_adds_expected_fields():
    cfg = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        components=("attn.out_proj", "mlp.c_proj"),
        block_split="joint",
        realization_diagnostics=True,
    )
    _corrections, diagnostics, _model, _sd = _fit(2, 4, cfg)
    expected = {
        "fit_relative_residual",
        "target_norm",
        "update_norm",
        "relative_update_norm",
        "realized_target_norm_ratio",
    }
    for row in diagnostics:
        assert expected <= set(row), row.keys()
        assert all(torch.isfinite(torch.tensor(float(row[k]))) for k in expected)


def test_joint_rejects_nonidentity_layerscale():
    class _LSBlock(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.attn = _Attention(width)
            self.mlp = torch.nn.Sequential(OrderedDict([("c_proj", torch.nn.Linear(width, width))]))
            self.mlp.c_proj = torch.nn.Linear(width, width)
            self.ls_1 = torch.nn.Identity()

            class _Scale(torch.nn.Module):
                def __init__(self, w):
                    super().__init__()
                    self.gamma = torch.nn.Parameter(torch.full((w,), 2.0))

                def forward(self, x):
                    return x * self.gamma

            self.ls_2 = _Scale(width)

        def forward(self, x):
            return x + self.ls_1(self.attn(x)) + self.ls_2(self.mlp(x))

    class _LSVisual(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.input = torch.nn.Linear(4, width)
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList([_LSBlock(width) for _ in range(depth)])

        def forward(self, images):
            x = self.input(images)
            for block in self.transformer.resblocks:
                x = block(x)
            return x.mean(dim=1)

    class _LSModel(torch.nn.Module):
        def __init__(self, width, depth):
            super().__init__()
            self.visual = _LSVisual(width, depth)

        def encode_image(self, x):
            return self.visual(x)

    torch.manual_seed(3)
    source_base = _LSModel(5, 2).eval()
    target_base = _LSModel(5, 2).eval()
    source_ft = deepcopy(source_base)
    with torch.no_grad():
        for block in source_ft.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(0.1 * torch.randn_like(block.mlp.c_proj.weight))
    data = _loader(seed=5)
    pairing = DiscreteLayerPairing.compute(2, 2)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    cfg = DirectResidualConfig(
        num_batches=3,
        ridge_relative=0.05,
        components=("attn.out_proj", "mlp.c_proj"),
        block_split="joint",
    )
    captured = capture_paired_boundary_activations(
        source_base,
        source_ft,
        target_base,
        data,
        data,
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
    )
    desired = compute_desired_effects(captured, pairing)
    with pytest.raises(ValueError, match="LayerScale"):
        fit_direct_residual(target_base, target_base_sd, captured, desired, pairing, config=cfg, device="cpu")
