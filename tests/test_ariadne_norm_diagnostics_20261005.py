"""Regression tests for the Ariadne update-norm diagnostics (``eval.diagnostics.ariadne_norm``).

The diagnostics must (a) leave the fit bit-identical, (b) reproduce the solver's own lambda, norms and residual from
their closed-form spectral expressions, (c) split the statistics exactly into odd/even halves, and (d) measure, by
mounting, the same single-block realization the statistics predict analytically.

Fixture duplicated from `tests/test_direct_residual_streaming.py` per this suite's no-cross-test-import convention.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.diagnostics.ariadne_norm.analysis import (
    EigenSystem,
    centered_system,
    label_free_alpha,
    official_lambda,
    row_concentration,
)
from merge_and_rebase.eval.diagnostics.ariadne_norm.inputs import DiagnosticInputs
from merge_and_rebase.eval.diagnostics.ariadne_norm.instrument import FitRecorder
from merge_and_rebase.eval.diagnostics.ariadne_norm.runner import run_task
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.fit import (
    ResidualSufficientStatistics,
    _realized_pred_sq_from_stats,
    _task_vector_sha256,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)


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


def _loader(n=8, seed=0):
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


def _config(**overrides):
    raw = {
        "components": ["mlp.c_proj"],
        "activation_storage": "streaming",
        "ridge_estimator": "empirical_bayes",
        "num_batches": 4,
        "seed": 3,
        **overrides,
    }
    return parse_direct_residual_config(raw)


def _setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned_copy(source_base, seed + 1)
    data = _loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, data, pairing, target_base_sd


def _key(j):
    return f"visual.transformer.resblocks.{j}.mlp.c_proj.weight"


REGIMES = [
    pytest.param(2, 4, id="extend"),
    pytest.param(4, 2, id="shrink"),
    pytest.param(3, 3, id="same_arch"),
]


def _fit(setup, config, stats_factory=None):
    source_base, source_ft, target_base, data, pairing, target_base_sd = setup
    prepared = prepare_direct_residual_streaming(
        source_base, target_base, data, data, pairing, num_batches=config.num_batches, seed=config.seed, device="cpu"
    )
    return fit_direct_residual_streaming(
        target_base,
        target_base_sd,
        source_base,
        source_ft,
        prepared,
        pairing,
        config=config,
        device="cpu",
        stats_factory=stats_factory,
    )


def _recorder(config, row_positions=()):
    return FitRecorder(
        ridge_estimator=config.ridge_estimator,
        ridge_relative=config.ridge_relative,
        exact_form=config.exact_form,
        row_positions=row_positions,
        token_index_fn=lambda k, n: torch.arange(5).repeat(n // 5),
    )


@pytest.mark.parametrize(("sd", "td"), REGIMES)
@pytest.mark.parametrize("estimator", ["empirical_bayes", "fixed_relative"])
def test_instrumented_fit_is_bit_identical(sd, td, estimator):
    config = _config(ridge_estimator=estimator)
    plain, plain_rows = _fit(_setup(sd, td), config)
    recorder = _recorder(config, row_positions=[0])
    instrumented, rows = _fit(_setup(sd, td), config, stats_factory=recorder.factory)
    assert _task_vector_sha256(plain) == _task_vector_sha256(instrumented)
    assert [r["correction_norm"] for r in plain_rows] == [r["correction_norm"] for r in rows]
    assert sorted(recorder.records) == list(range(td))


@pytest.mark.parametrize(("sd", "td"), REGIMES)
def test_spectral_analysis_reproduces_solver(sd, td):
    config = _config()
    recorder = _recorder(config)
    _fit(_setup(sd, td), config, stats_factory=recorder.factory)
    for record in recorder.records.values():
        n3 = record["n3"]
        assert n3["lam"] == pytest.approx(n3["solver_lam"], rel=1e-12)
        assert n3["at_official"]["correction_norm"] == pytest.approx(n3["solver_correction_norm"], rel=1e-9)
        assert n3["at_official"]["bias_norm"] == pytest.approx(n3["solver_bias_norm"], rel=1e-7, abs=1e-9)
        assert n3["at_official"]["residual_norm_after"] == pytest.approx(
            n3["solver_residual_norm_after"], rel=1e-7, abs=1e-9
        )
        shares = n3["tail_shares"]["0.1"]
        assert 0.0 <= shares["correction_norm_sq_share"] <= 1.0
        # L-curve: the correction norm is non-increasing in the ridge.
        norms = [p["correction_norm"] for p in n3["l_curve"][1:]]
        assert all(a >= b - 1e-12 for a, b in zip(norms, norms[1:], strict=False))


def test_split_halves_partition_the_statistics():
    config = _config()
    captured = {}

    class _Spy(FitRecorder):
        def factory(self, position, component, device):
            stats = super().factory(position, component, device)
            captured[position] = stats
            return stats

    recorder = _Spy(
        ridge_estimator=config.ridge_estimator,
        ridge_relative=config.ridge_relative,
        exact_form=True,
        row_positions=(),
        token_index_fn=lambda k, n: torch.zeros(n, dtype=torch.long),
    )
    original_solve = type(next(iter([recorder.factory(0, "mlp.c_proj", "cpu")]))).solve
    halves_seen = {}

    def solve_spy(self, **kwargs):
        halves_seen[self.position] = [h for h in self.halves]
        return original_solve(self, **kwargs)

    import merge_and_rebase.eval.diagnostics.ariadne_norm.instrument as inst

    captured.clear()
    old = inst.InstrumentedStatistics.solve
    inst.InstrumentedStatistics.solve = solve_spy
    try:
        _fit(_setup(2, 4), config, stats_factory=recorder.factory)
    finally:
        inst.InstrumentedStatistics.solve = old
    for j, stats in captured.items():
        a, b = halves_seen[j]
        assert a.n_rows + b.n_rows == stats.n_rows
        assert torch.allclose(a.s + b.s, stats.s, rtol=1e-12, atol=1e-12)
        assert torch.allclose(a.b + b.b, stats.b, rtol=1e-12, atol=1e-12)
        assert a.sum_e2 + b.sum_e2 == pytest.approx(stats.sum_e2, rel=1e-12)


def test_row_concentration_full_refit_matches_solver():
    torch.manual_seed(0)
    h = torch.randn(200, 6)
    e = h @ torch.randn(6, 3) + 0.1 * torch.randn(200, 3)
    t_out = torch.eye(3)
    stats = ResidualSufficientStatistics()
    stats.update(h, e, None, t_out)
    x, diag = stats.solve(ridge_relative=0.01, ridge_estimator="empirical_bayes", exact_form=True)
    rec = row_concentration(
        h, e, torch.arange(200) % 10, t_out, ridge_estimator="empirical_bayes", ridge_relative=0.01, exact_form=True
    )
    assert rec["correction_norm"] == pytest.approx(diag["correction_norm"], rel=1e-6)
    assert rec["baseline_relative_residual"] == pytest.approx(
        diag["residual_norm_after"] / diag["residual_norm_before"], rel=1e-6
    )
    eig = EigenSystem(centered_system(stats))
    lam = official_lambda(centered_system(stats), ridge_estimator="empirical_bayes", ridge_relative=0.01)
    assert eig.point(lam)["correction_norm"] == pytest.approx(diag["correction_norm"], rel=1e-9)
    assert rec["first_token"]["row_share"] == pytest.approx(0.1)


def _inputs(sd, td, config):
    source_base, source_ft, target_base, data, pairing, target_base_sd = _setup(sd, td)
    generic = [(torch.randn(2, 5, 4), torch.zeros(2)) for _ in range(2)]
    return DiagnosticInputs(
        family="toy",
        task="toy",
        device="cpu",
        config=config,
        pairing=pairing,
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_state=target_base_sd,
        source_loader=data,
        target_loader=data,
        generic_target_batches=generic,
        token_index=lambda batch, n: torch.arange(5).repeat(n // 5),
        component_key=_key,
        source_component_key=_key,
        alpha_star=0.5,
    )


@pytest.mark.parametrize(("sd", "td"), REGIMES)
def test_run_task_end_to_end(sd, td):
    config = _config()
    plain, _ = _fit(_setup(sd, td), config)
    inp = _inputs(sd, td, config)
    inp.reference_task_vector = plain
    record, arrays, tv = run_task(inp, alpha_grid=[0.5, 1.0], row_positions=[0])
    assert record["n0"]["task_vector"]["reference"]["passed"]
    assert record["n0"]["task_vector"]["reference"]["sha256_equal"]
    assert set(record["n5"]) == {0}
    # Mounting block j alone changes its own boundary by exactly H dW + 1 beta (the statistics' prediction).
    recorder = _recorder(config)
    _fit(_setup(sd, td), config, stats_factory=recorder.factory)
    for j in range(td):
        own = record["n2"]["single_block"][j]
        assert own["own_r"] is not None
    # Single-block own realization vs analytic prediction from the sufficient statistics.
    captured = {}

    class _Keep(FitRecorder):
        def factory(self, position, component, device):
            s = super().factory(position, component, device)
            captured[position] = s
            return s

    keep = _Keep(
        ridge_estimator=config.ridge_estimator,
        ridge_relative=config.ridge_relative,
        exact_form=True,
        row_positions=(),
        token_index_fn=lambda k, n: torch.zeros(n, dtype=torch.long),
    )
    _fit(_setup(sd, td), config, stats_factory=keep.factory)
    for j, stats in captured.items():
        pred = (
            _realized_pred_sq_from_stats(stats, tv[_key(j)], tv[_key(j).replace(".weight", ".bias")], stats._t_out)
            ** 0.5
        )
        d_norm = record["n2"]["joint_by_alpha"][1.0][j]["desired_norm"]
        assert record["n2"]["single_block"][j]["own_r"] == pytest.approx(pred / d_norm, rel=1e-4)
    assert record["n2"]["label_free_alpha"]["alpha_hat"] in (0.5, 1.0)
    assert set(record["n6"]) == {0.5, 1.0}
    assert all(k.startswith("pos") for k in arrays)


def test_label_free_alpha_picks_minimum():
    curve = {0.5: {0: {"relative_error": 0.2}}, 1.0: {0: {"relative_error": 0.9}}}
    assert label_free_alpha(curve)["alpha_hat"] == 0.5


def test_state_sha_handles_zero_dim_and_bf16_parameters():
    from merge_and_rebase.eval.diagnostics.ariadne_norm.runner import _state_sha

    model = torch.nn.Module()
    model.logit_scale = torch.nn.Parameter(torch.tensor(4.6))  # 0-dim, as in OpenCLIP
    model.w = torch.nn.Parameter(torch.randn(3, 2).to(torch.bfloat16))
    first = _state_sha(model)
    assert first == _state_sha(model)
    with torch.no_grad():
        model.logit_scale.add_(1e-3)
    assert _state_sha(model) != first
