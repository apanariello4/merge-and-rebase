"""``calibration_streaming`` (anchors/calibration_streaming.py): host RAM independent of the calibration token count.

The streamed fit (batch-wise sufficient statistics) must return the resident fit's task vector up to float summation
order, for (a) the kernel route (D3 held-out canonical weights, ridge_preserve, independent depth fit, registered frame)
and (b) the bridge to Ariadne (``map_rows='task'``, Gram-form Procrustes / truncations, ridge_preserve). Tiny Llama
models, CPU. The memory test counts every live tensor byte at each ridge update and checks that it does not grow with
the number of calibration tokens.
"""

from __future__ import annotations

import gc
from copy import deepcopy

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_llama

from merge_and_rebase.data.llm_calibration import build_text_calibration_loader
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne import fit as ariadne_fit
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, AnchorsRebase, parse_anchors_config
from merge_and_rebase.rebase.methods.anchors import calibration_streaming as cs
from merge_and_rebase.rebase.model_families import infer_family

torch.set_num_threads(2)

WORDS = sorted({w for t in CALIB_TEXTS for w in t.split()})
POOL_TEXTS = [" ".join(WORDS[(7 * i + 3 * k) % len(WORDS)] for k in range(8 + i % 4)) for i in range(32)]
MAX_LENGTH = 14


def _task_texts(n):
    return [" ".join(WORDS[(5 * i + 2 * k + 1) % len(WORDS)] for k in range(8 + i % 3)) for i in range(n)]


def _loader(texts):
    return build_text_calibration_loader(
        tokenizer=local_tokenizer(CALIB_TEXTS), texts=texts, batch_size=4, max_length=MAX_LENGTH
    )


def _setup(source_layers=2, target_layers=3, target_hidden=32):
    src = tiny_llama(layers=source_layers, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    fam = infer_family(src)
    tgt = deepcopy(tiny_llama(layers=target_layers, hidden=target_hidden, inter=2 * target_hidden, seed=2))
    sd = {k: v.detach().clone() for k, v in tgt.state_dict().items()}
    materialize_missing_projection_biases(tgt, sd, family_adapter=fam, components=("mlp.c_proj",))
    return src, ft, tgt, sd, fam, DiscreteLayerPairing.compute(source_layers, target_layers)


def _prepare(cfg, setup, n_task=24):
    src, ft, tgt, sd, fam, pairing = setup
    return AnchorsRebase().prepare(
        source_base_model=src, source_ft_model=ft, target_model=tgt, target_base_sd=sd,
        source_loader=_loader(_task_texts(n_task)), target_loader=_loader(_task_texts(n_task)), pairing=pairing,
        config=cfg, device="cpu", source_anchor_loader=_loader(POOL_TEXTS), target_anchor_loader=_loader(POOL_TEXTS),
        source_model_id="A", target_model_id="B", state=None, family_adapter=fam,
    )


KERNEL = dict(
    anchor_pool="wikitext103", num_anchors=120, anchor_batches=6, registration_batches=2, num_batches=3,
    missing_bias="materialize", synthesis="ridge_preserve", rho_g=0.1, depth_fit="independent",
    kernel_decode="canonical_soft", canonical_soft_weights="heldout", anchor_sampling="k_per_document",
    anchor_tokens_per_image=5, ridge_estimator="fixed_relative", ridge_relative=0.01,
)
BRIDGE = dict(
    anchor_pool="wikitext103", num_anchors=16, anchor_batches=2, registration_batches=2, num_batches=3,
    missing_bias="materialize", synthesis="ridge_preserve", rho_g=0.1, frame_map="pairwise", map_rows="task",
    spectral_filter="procrustes_cross", rescale_target=False, ridge_estimator="fixed_relative", ridge_relative=0.01,
)


def _assert_same(resident, streamed, *, rtol=2e-5):
    a, b = resident.unit_task_vector, streamed.unit_task_vector
    assert set(a) == set(b) and any(float(v.abs().max()) > 0 for v in a.values())
    worst = 0.0
    for key in a:
        torch.testing.assert_close(b[key], a[key], rtol=rtol, atol=rtol * float(a[key].abs().max()))
        worst = max(worst, float((b[key] - a[key]).abs().max() / (a[key].abs().max() + 1e-30)))
    assert len(resident.diagnostics) == len(streamed.diagnostics)
    for r0, r1 in zip(resident.diagnostics, streamed.diagnostics, strict=True):
        assert set(r0) == set(r1)
        for k, v in r0.items():
            if isinstance(v, float) and k != "effect_before_norm":
                assert r1[k] == pytest.approx(v, rel=1e-4, abs=1e-9), k
    return worst


# ---------------------------------------------------------------- config surface


def test_default_is_off_and_not_recorded():
    cfg = AnchorsConfig()
    assert cfg.calibration_streaming is False and cfg.calibration_streaming_chunk == 4
    assert "calibration_streaming" not in cfg.to_dict() and "calibration_streaming_chunk" not in cfg.to_dict()
    on = parse_anchors_config({"calibration_streaming": True, "calibration_streaming_chunk": 2})
    assert on.to_dict()["calibration_streaming"] is True and on.to_dict()["calibration_streaming_chunk"] == 2


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"depth_fit": "sequential"}, "depth_fit='independent'"),
        ({"calibration_streaming_chunk": 0}, "chunk"),
        ({"gain_calibration": "lsq"}, "gain_calibration"),
        ({"refine": "joint"}, "refine"),
        ({"map_rows": "task", "spectral_filter": "tikhonov"}, "spectral_filter"),
        ({"map_rows": "task", "spectral_filter": "procrustes_cross", "frame_map": "gpa"}, "frame_map|gpa"),
    ],
)
def test_streaming_validation_refuses_unstreamed_combinations(bad, match):
    with pytest.raises(ValueError, match=match):
        parse_anchors_config({"calibration_streaming": True, **bad})


def test_streaming_refused_on_vision_run_time():
    cfg = AnchorsConfig(calibration_streaming=True)
    with pytest.raises(ValueError, match="decoders"):
        cs.draw_streaming_calibration(None, None, cfg, None)


# ---------------------------------------------------------------- equivalence


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_kernel_ridge_preserve_independent_streaming_equals_resident(chunk):
    setup = _setup(2, 3, 16)  # T = 120 anchors; each held-out half (60) exceeds d_m + d_k = 48
    resident = _prepare(AnchorsConfig(**KERNEL), setup)
    streamed = _prepare(AnchorsConfig(**KERNEL, calibration_streaming=True, calibration_streaming_chunk=chunk), setup)
    worst = _assert_same(resident, streamed)
    assert worst < 2e-5
    assert streamed.extra["calibration"]["n_rows_content"] == resident.extra["calibration"]["n_rows_content"]
    assert streamed.extra["anchors"]["config"]["calibration_streaming"] is True
    # the held-out canonical decode and the source coverage reach the diagnostics identically
    assert "anchors_coverage" in streamed.diagnostics[0]


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"map_num_rows": 40},  # the seeded row subsample of the frame, applied by global row offset
        {"spectral_filter": "truncate_cross", "spectral_rank": 6},
        {"spectral_filter": "truncate_target", "rescale_target": True},
        {"synthesis": "ridge", "rho_g": 0.0},
    ],
)
def test_ariadne_bridge_task_rows_streaming_equals_resident(extra):
    setup = _setup(2, 3, 32)
    resident = _prepare(AnchorsConfig(**{**BRIDGE, **extra}), setup)
    streamed = _prepare(AnchorsConfig(**{**BRIDGE, **extra}, calibration_streaming=True, calibration_streaming_chunk=2), setup)
    # the resident frame is rounded to float32; the streamed one is float64 moments
    assert _assert_same(resident, streamed, rtol=1e-4) < 1e-4


def test_streaming_restores_the_target_model():
    setup = _setup()
    before = {k: v.clone() for k, v in setup[2].state_dict().items()}
    _prepare(AnchorsConfig(**BRIDGE, calibration_streaming=True), setup)
    after = setup[2].state_dict()
    assert set(before) == set(after) and all(torch.equal(before[k], after[k]) for k in before)


# ---------------------------------------------------------------- memory does not scale with the token count


def _live_tensor_bytes() -> int:
    seen: set[int] = set()
    total = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == "cpu" and obj.numel():
                st = obj.untyped_storage()
                if st.data_ptr() not in seen:
                    seen.add(st.data_ptr())
                    total += st.nbytes()
        except Exception:  # noqa: S112  (freed / wrapped objects)
            continue
    return total


def _peak_live_bytes(cfg, n_task, monkeypatch) -> tuple[int, int]:
    """(peak live tensor bytes at ridge updates, content rows) of one prepare()."""
    peak = {"v": 0}
    orig = ariadne_fit.ResidualSufficientStatistics.update

    def spy(self, *a, **k):
        out = orig(self, *a, **k)
        peak["v"] = max(peak["v"], _live_tensor_bytes())
        return out

    monkeypatch.setattr(ariadne_fit.ResidualSufficientStatistics, "update", spy)
    prepared = _prepare(cfg, _setup(), n_task=n_task)
    return peak["v"], int(prepared.extra["calibration"]["n_rows_content"])


def test_streaming_live_memory_does_not_grow_with_tokens_but_resident_does(monkeypatch):
    # the resident fit's banks scale with rows (here ~ rows * (3 blocks * (2*32 + 64 + 32)) * 4 B); the streamed
    # statistics do not: 3 vs 12 calibration batches (4x the rows) changes the streamed peak by < one batch of rows.
    kw = dict(BRIDGE, map_rows="frame", spectral_filter="tikhonov", num_batches=3)
    kw_big = dict(kw, num_batches=12)
    gc.collect()
    res_small, rows_small = _peak_live_bytes(AnchorsConfig(**kw), 48, monkeypatch)
    res_big, rows_big = _peak_live_bytes(AnchorsConfig(**kw_big), 48, monkeypatch)
    st_small, _ = _peak_live_bytes(AnchorsConfig(**kw, calibration_streaming=True), 48, monkeypatch)
    st_big, _ = _peak_live_bytes(AnchorsConfig(**kw_big, calibration_streaming=True), 48, monkeypatch)
    assert rows_big > 3 * rows_small
    bank_per_row = 4 * (2 * 2 * 32 + 3 * 32 + 3 * 64 + 3 * 32 + 3 * 32)  # rough lower bound of the resident banks
    assert res_big - res_small > 0.5 * bank_per_row * (rows_big - rows_small)
    assert abs(st_big - st_small) < 0.1 * bank_per_row * (rows_big - rows_small)
