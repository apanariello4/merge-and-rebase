"""Per-phase cost accounting (utils.cost_accounting): recorder semantics, and that
recording never changes a numerical result (THESEUS/BiCo transported deltas hash-identical
with and without an active recorder). Direct Residual's resident/streaming invariance is
pinned in test_direct_residual_component_coverage.py / test_direct_residual_streaming_parity.py.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.registry import get_method
from merge_and_rebase.utils.cost_accounting import (
    PHASES,
    PhaseCostRecorder,
    cost_excluded,
    cost_phase,
    cost_phase_decorator,
    recording,
)

# ---- recorder semantics -------------------------------------------------------


def test_cost_phase_is_a_noop_without_recorder():
    calls = []

    @cost_phase_decorator("transformation")
    def fn(x):
        calls.append(x)
        return x + 1

    with cost_phase("activation_collection"), cost_excluded():
        assert fn(1) == 2
    assert calls == [1]


def test_unknown_phase_rejected():
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder), pytest.raises(ValueError, match="unknown cost phase"):
        with cost_phase("prepare"):
            pass


def test_phases_split_wall_time_without_double_counting():
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder):
        with cost_phase("activation_collection"):
            time.sleep(0.05)
            with cost_phase("transformation"):  # nested: suspends collection
                time.sleep(0.05)
            time.sleep(0.05)
        with cost_phase("transport"):
            time.sleep(0.02)
        with cost_excluded():
            with cost_phase("transformation"):  # excluded work is charged to nothing
                time.sleep(0.05)
        time.sleep(0.02)  # unattributed
    summary = recorder.summary()
    phases = summary["phases"]
    assert set(phases) == set(PHASES)
    assert phases["activation_collection"]["segments"] == 2
    assert 0.09 <= phases["activation_collection"]["seconds"] < 0.2
    assert 0.045 <= phases["transformation"]["seconds"] < 0.09
    assert phases["transformation"]["segments"] == 1
    assert 0.018 <= phases["transport"]["seconds"] < 0.06
    assert summary["excluded_seconds"] >= 0.045
    assert summary["unattributed_seconds"] >= 0.018
    attributed = sum(p["seconds"] for p in phases.values())
    covered = attributed + summary["excluded_seconds"] + summary["unattributed_seconds"]
    assert covered <= summary["total_seconds"] + 1e-6


def test_exclusive_phase_absorbs_inner_phases():
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder):
        with cost_phase("transformation", exclusive=True):
            with cost_phase("activation_collection"):
                time.sleep(0.03)
    phases = recorder.summary()["phases"]
    assert phases["activation_collection"]["segments"] == 0
    assert phases["activation_collection"]["seconds"] == 0.0
    assert phases["transformation"]["seconds"] >= 0.03


def test_reentering_the_open_phase_extends_its_segment():
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder):
        with cost_phase("transformation"):
            with cost_phase("transformation"):
                pass
    assert recorder.summary()["phases"]["transformation"]["segments"] == 1


def test_host_peak_is_charged_to_the_allocating_phase():
    recorder = PhaseCostRecorder("cpu")
    nbytes = 256 * 1024 * 1024
    with recording(recorder):
        with cost_phase("activation_collection"):
            small = np.ones(1024)
        with cost_phase("transformation"):
            big = np.ones(nbytes // 8)  # touched pages: counted in RSS
            big += 1.0
            del big
    phases = recorder.summary()["phases"]
    assert small.sum() == 1024
    assert phases["transformation"]["host_peak_rss_delta_bytes"] >= 0.8 * nbytes
    assert phases["activation_collection"]["host_peak_rss_delta_bytes"] < 0.5 * nbytes
    assert phases["transformation"]["host_peak_rss_bytes"] >= phases["transformation"]["host_peak_rss_delta_bytes"]


def test_peaks_since_mark_spans_segments():
    recorder = PhaseCostRecorder("cpu")
    nbytes = 192 * 1024 * 1024
    with recording(recorder):
        mark = recorder.mark()
        with cost_phase("transformation"):
            big = np.ones(nbytes // 8)
            big += 1.0
            del big
        with cost_phase("transport"):
            pass
        _cuda_peak, host_peak = recorder.peaks_since(mark)
        later = recorder.mark()
        _cuda_later, host_later = recorder.peaks_since(later)
    assert host_peak >= host_later + 0.8 * nbytes


# ---- THESEUS / BiCo: recording leaves transported deltas bit-identical ------


class _TinyVisual(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.ln(self.fc1(x)))


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _recipe(model, batch):
    images, labels = batch
    return F.cross_entropy(model.encode_image(images), labels.long()), []


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(3)
    data = torch.randn(16, 6, generator=generator)
    labels = torch.arange(16) % 5
    return DataLoader(TensorDataset(data, labels), batch_size=4, shuffle=False)


def _hash(delta: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(delta):
        h.update(key.encode())
        h.update(delta[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _transport(method_name: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    source_model = _TinyModel(hid_dim=8)
    target_model = _TinyModel(hid_dim=7)
    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}
    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }
    kwargs = {"source_recipe": _recipe, "target_recipe": _recipe} if method_name == "bico" else {}
    return get_method(method_name).transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=_loader(),
        target_dataloader=_loader(),
        device="cpu",
        seq_align="mean",
        num_batches=2,
        strict=True,
        **kwargs,
    )


@pytest.mark.parametrize("method_name", ["theseus", "bico"])
def test_recording_leaves_transport_bit_identical(method_name):
    plain = _transport(method_name)
    recorder = PhaseCostRecorder("cpu")
    with recording(recorder), cost_phase("transport"):
        recorded = _transport(method_name)
    assert _hash(plain) == _hash(recorded)
    phases = recorder.summary()["phases"]
    # method.transport runs prepare internally: its collection and statistics are
    # charged to their own phases even inside the enclosing transport phase.
    assert phases["activation_collection"]["segments"] >= 2
    assert phases["transformation"]["segments"] >= 1
    assert phases["transport"]["seconds"] > 0.0
