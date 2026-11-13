"""Opt-in one-pass Ariadne fit (`DirectResidualConfig.onepass`) vs the streaming fit it is equivalent to.

CPU, tiny models (the fixtures of ``test_ariadne_storage_parity``). Covered: ΔW and bias of the one-pass fit equal the
streaming fit's (float64 statistics: rtol 1e-5 / atol 1e-6; float32: per-tensor relative Frobenius difference <= 1e-4)
for extend-like and shrink-like pairings, with equal and with different source/target token counts (the resize), several
batches and ``streaming_position_chunk`` on/off in the reference; the blocked upper-triangular Gram equals the full
Gram; the config guard; the decoder rejection; the default (``onepass=False``) is untouched.
"""

from __future__ import annotations

import pytest
import torch
from test_ariadne_storage_parity import MAIN, N_CLASSES, WIDTH, _Model, _tuned
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.config import (
    DirectResidualConfig,
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.onepass import gram_blocked
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase

RTOL, ATOL = 1e-5, 1e-6
BATCH_SIZE, NUM_BATCHES = 8, 4


class _Images(TensorDataset):
    """Tensor dataset with explicit sample ids, so source and target (different token counts) pair by identity."""

    def __init__(self, images, labels):
        super().__init__(images, labels)
        self.sample_ids = list(range(len(labels)))


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fit(depths, tokens, overrides, *, width=WIDTH, tweak=None, base_sd_hook=None, loaders=None):
    """One ``AriadneRebase().prepare`` on the tiny stacks; ``depths=(source, target)``, ``tokens=(source, target)``."""
    source_depth, target_depth = depths
    torch.manual_seed(5)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    if tweak is not None:
        tweak(target_base)
    source_ft = _tuned(source_base, 6, 0.05)
    n = BATCH_SIZE * NUM_BATCHES
    gen = torch.Generator().manual_seed(7)
    labels = torch.arange(n) % N_CLASSES
    if loaders is None:
        loaders = [
            DataLoader(_Images(torch.randn(n, t, 4, generator=gen), labels), batch_size=BATCH_SIZE, shuffle=False)
            for t in tokens
        ]
    cfg = parse_direct_residual_config({**MAIN, "activation_storage": "streaming", **overrides})
    return AriadneRebase().prepare(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=(base_sd_hook or (lambda sd: sd))({k: v.clone() for k, v in target_base.state_dict().items()}),
        source_loader=loaders[0],
        target_loader=loaders[1],
        pairing=DiscreteLayerPairing.compute(source_depth, target_depth),
        config=cfg,
        device="cpu",
    )


def _rel_frob(a, b):
    return float(torch.linalg.norm(a.double() - b.double()) / torch.linalg.norm(b.double()))


DIRECTIONS = {"extend": (2, 4), "shrink": (4, 2)}
TOKEN_CASES = {"same_tokens": (17, 17), "resize_up": (10, 17), "resize_down": (17, 10)}


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("tokens", list(TOKEN_CASES))
@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_onepass_fp64_matches_streaming(direction, tokens, chunk):
    ref = _fit(DIRECTIONS[direction], TOKEN_CASES[tokens], {"streaming_position_chunk": chunk})
    one = _fit(DIRECTIONS[direction], TOKEN_CASES[tokens], {"onepass": True, "num_batches": NUM_BATCHES})
    assert set(one.unit_task_vector) == set(ref.unit_task_vector)
    assert any(k.endswith(".bias") for k in ref.unit_task_vector)
    for key, value in ref.unit_task_vector.items():
        assert one.unit_task_vector[key].dtype == value.dtype
        torch.testing.assert_close(one.unit_task_vector[key], value, rtol=RTOL, atol=ATOL, msg=key)
    # same row set and fields the fit reports, analysis-only residual fields not measured
    assert [r["position"] for r in one.diagnostics] == [r["position"] for r in ref.diagnostics]
    for r_one, r_ref in zip(one.diagnostics, ref.diagnostics, strict=True):
        assert r_one["n_rows"] == r_ref["n_rows"]
        assert r_one["source_position"] == r_ref["source_position"]
        assert r_one["procrustes_rank"] == r_ref["procrustes_rank"]
        assert r_one["ridge"] == pytest.approx(r_ref["ridge"], rel=1e-6)
        assert r_one["residual_norm_after"] is None and r_one["correction_rank"] is None
    assert one.extra["calibration"]["indices"] == ref.extra["calibration"]["indices"]


@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_onepass_fp32_matches_streaming(direction):
    ref = _fit(DIRECTIONS[direction], (10, 17), {})
    one = _fit(DIRECTIONS[direction], (10, 17), {"onepass": True, "onepass_stats_dtype": "float32"})
    for key, value in ref.unit_task_vector.items():
        assert _rel_frob(one.unit_task_vector[key], value) <= 1e-4, key
    assert one.timing["correction_fit"]["onepass_stats_dtype"] == "float32"
    assert one.timing["correction_fit"]["onepass_tf32_stats_matmul"] is False


def test_onepass_timing_keys_and_strength():
    one = _fit((2, 4), (17, 17), {"onepass": True, "strength": 0.5})
    fit = one.timing["correction_fit"]
    assert fit["onepass_fit_seconds"] == fit["correction_fit_seconds"] > 0
    assert one.timing["alignment_calibration"]["alignment_calibration_seconds"] == 0.0
    for key, value in one.unit_task_vector.items():
        torch.testing.assert_close(one.task_vector[key], 0.5 * value)
    assert {r["position"] for r in one.diagnostics} == {0, 1, 2, 3}


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("d", [16, 13, 7])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_blocked_gram_equals_full_gram(k, d, dtype):
    a = torch.randn(50, d, dtype=dtype, generator=torch.Generator().manual_seed(1))
    tol = dict(rtol=1e-12, atol=1e-12) if dtype == torch.float64 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gram_blocked(a, k), a.T @ a, **tol)
    # accumulation over several batches
    s = torch.zeros(d, d, dtype=dtype)
    from merge_and_rebase.rebase.methods._ariadne.onepass import gram_mirror_upper_, gram_upper_update_

    for chunk in a.split(17):
        gram_upper_update_(s, chunk, k)
    torch.testing.assert_close(gram_mirror_upper_(s, k), a.T @ a, **tol)


def test_blocked_gram_more_blocks_than_columns():
    a = torch.randn(9, 3, dtype=torch.float64)
    torch.testing.assert_close(gram_blocked(a, 8), a.T @ a)


ONEPASS = {"onepass": True}
BASE = {**MAIN, "activation_storage": "streaming"}


@pytest.mark.parametrize(
    "bad",
    [
        {"depth_fit": "sequential"},
        {"block_split": "backfit"},
        {"components": ["attn.out_proj"]},
        {"components": ["attn.out_proj", "mlp.c_proj"]},
        {"procrustes_rank": 3},
        {"procrustes_energy": 0.9},
        {"alignment_map": "ridge"},
        {"alignment_row_weighting": "cls_balanced"},
        {"residual_target": "transported_endpoint"},
        {"realization_diagnostics": True},
        {"fidelity_holdout": True},
        {"activation_storage": "resident"},
        {"tv_scaling": "global"},
    ],
)
def test_onepass_guard_rejects(bad):
    # block_split='backfit' is already refused by the streaming validation (before the one-pass guard is reached)
    with pytest.raises(ValueError, match="onepass|block_split"):
        parse_direct_residual_config({**BASE, **ONEPASS, **bad})


def test_onepass_dtype_validation():
    with pytest.raises(ValueError, match="onepass_stats_dtype"):
        parse_direct_residual_config({**BASE, **ONEPASS, "onepass_stats_dtype": "bfloat16"})
    with pytest.raises(ValueError, match="requires onepass"):
        parse_direct_residual_config({**BASE, "onepass_stats_dtype": "float32"})
    assert parse_direct_residual_config({**BASE, **ONEPASS, "onepass_stats_dtype": "float32"}).onepass


def test_decoder_resolver_rejects_onepass():
    class _Adapter:
        name = "x"
        HAS_POST_MLP_NORM = False

    with pytest.raises(ValueError):
        resolve_ariadne_decoder_config({"preset": "ariadne", "onepass": True}, _Adapter())


def test_default_is_off_and_not_serialized():
    cfg = parse_direct_residual_config(BASE)
    assert cfg.onepass is False and cfg.onepass_stats_dtype == "float64"
    assert "onepass" not in direct_residual_config_dict(cfg)
    assert "onepass_stats_dtype" not in direct_residual_config_dict(cfg)
    assert direct_residual_config_dict(DirectResidualConfig()) == direct_residual_config_dict(
        DirectResidualConfig(onepass=False)
    )
    on = parse_direct_residual_config({**BASE, **ONEPASS})
    assert direct_residual_config_dict(on)["onepass"] is True


def test_default_path_has_no_onepass_timing_keys():
    ref = _fit((2, 4), (17, 17), {})
    assert "onepass_fit_seconds" not in ref.timing["correction_fit"]
    assert ref.timing["alignment_calibration"]["alignment_calibration_seconds"] > 0


def _massive_channels(model):
    """Eight c_proj-input channels per block with mean ~20 and std ~0.05 (mean >> std, as in real GELU features)."""
    with torch.no_grad():
        for block in model.visual.transformer.resblocks:
            block.mlp.c_fc.weight[:8] *= 0.05
            block.mlp.c_fc.bias[:8] = 20.0


def test_onepass_fp32_large_mean_channels(monkeypatch):
    """fp32 statistics stay within 1e-4 of the fp64 streaming fit when channels have mean >> std (first-batch shift)."""
    kw = dict(width=128, tweak=_massive_channels)
    ref = _fit((2, 4), (10, 17), {}, **kw)
    one = _fit((2, 4), (10, 17), {"onepass": True, "onepass_stats_dtype": "float32"}, **kw)
    assert one.timing["correction_fit"]["onepass_shift_by_first_batch"] is True
    for key, value in ref.unit_task_vector.items():
        assert _rel_frob(one.unit_task_vector[key], value) <= 1e-4, key


def test_onepass_rejects_non_base_target():
    def perturb(sd):
        key = next(k for k in sd if k.endswith("mlp.c_proj.weight"))
        sd[key] = sd[key] + 1e-3
        return sd

    with pytest.raises(ValueError, match="target_base_sd"):
        _fit((2, 4), (17, 17), {"onepass": True}, base_sd_hook=perturb)


def test_onepass_warns_on_ignored_options(caplog):
    with caplog.at_level("WARNING"):
        one = _fit((2, 4), (17, 17), {"onepass": True, "streaming_position_chunk": 2})
    assert one.timing["correction_fit"]["onepass_ignored_options"] == {"streaming_position_chunk": 2}
    assert any("ignores" in r.message for r in caplog.records)
    assert one.timing["alignment_calibration"]["alignment_calibration_peak_memory_bytes"] is None
    assert one.timing["correction_fit"]["onepass_skipped_work"]


def test_onepass_matches_streaming_on_precomputed_calibration(tmp_path):
    """onepass + calibration_data='precomputed': same images as streaming, same fit (merge of the two branches)."""
    from merge_and_rebase.eval.vision_rebase.context import _TaskContext
    from merge_and_rebase.eval.vision_rebase.precomputed_calibration import build_precomputed_context

    n, tokens = BATCH_SIZE * NUM_BATCHES, 12  # [N, 3, 4, 4] images -> (N, 12, 4) inputs of the tiny stack
    images = torch.randn(n + 5, 3, 4, 4, generator=torch.Generator().manual_seed(11))
    torch.save({"images": images}, tmp_path / "DTD.pt")
    template = _TaskContext(
        loaders=None, source_loaders=None, classnames=["a"], build_cfg_task=None, source_build_cfg_task=None
    )
    ctx, meta = build_precomputed_context(
        "DTD",
        cfg={"calibration_images_dir": str(tmp_path)},
        num_batches=NUM_BATCHES,
        batch_size=BATCH_SIZE,
        template=template,
    )
    assert meta["n_used"] == n

    def collate(batch):
        x = torch.stack([b[0] for b in batch]).reshape(len(batch), tokens, 4)
        return x, torch.tensor([b[1] for b in batch])

    def loaders():
        return [
            DataLoader(ctx.source_loaders.train.dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate),
            DataLoader(ctx.loaders.train.dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate),
        ]

    for direction in DIRECTIONS.values():
        common = {"calibration_data": "precomputed", "num_batches": NUM_BATCHES}
        ref = _fit(direction, (tokens, tokens), common, loaders=loaders())
        one = _fit(direction, (tokens, tokens), {**common, "onepass": True}, loaders=loaders())
        assert ref.extra["calibration"]["indices"] == list(range(n))
        assert one.extra["calibration"]["indices"] == ref.extra["calibration"]["indices"]
        assert set(one.unit_task_vector) == set(ref.unit_task_vector)
        for key, value in ref.unit_task_vector.items():
            torch.testing.assert_close(one.unit_task_vector[key], value, rtol=RTOL, atol=ATOL, msg=key)
