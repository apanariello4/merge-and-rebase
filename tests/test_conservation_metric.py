"""Conservation metric helpers (eval/diagnostics/conservation.py)."""

import pytest
import torch

from merge_and_rebase.eval.diagnostics.conservation import accuracy_delta, find_vector
from merge_and_rebase.merge.methods._common import axpy_state_dict


def test_zero_update_has_zero_delta():
    base = {"a": 0.5, "b": 0.7, "c": 0.9}
    out = accuracy_delta(dict(base), base, "a")
    assert out["delta_by_task"] == {"b": 0.0, "c": 0.0} and out["mean_delta"] == 0.0 and out["max_drop"] == 0.0
    sd = {"w": torch.randn(3, 3)}
    assert torch.equal(axpy_state_dict(sd, {"w": torch.zeros(3, 3)}, 0.7)["w"], sd["w"])


def test_delta_in_points_excludes_the_transferred_task():
    out = accuracy_delta({"a": 0.9, "b": 0.6, "c": 0.95}, {"a": 0.5, "b": 0.7, "c": 0.9}, "a")
    assert out["delta_by_task"] == pytest.approx({"b": -10.0, "c": 5.0})
    assert out["mean_delta"] == pytest.approx(-2.5) and out["max_drop"] == pytest.approx(-10.0)


def test_find_vector(tmp_path):
    (tmp_path / "GTSRB_anchors_transported_native.pt").write_bytes(b"x")
    assert find_vector(tmp_path, "GTSRB").name == "GTSRB_anchors_transported_native.pt"
    with pytest.raises(FileNotFoundError):
        find_vector(tmp_path, "EuroSAT")
