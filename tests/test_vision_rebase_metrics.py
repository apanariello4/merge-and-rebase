from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import merge_and_rebase.eval.vision_rebase.cli as vision_rebase_cli
import merge_and_rebase.eval.vision_rebase.source_lmc as source_lmc
from merge_and_rebase.eval.print_utils import (
    _latex_percent_cells,
    _latex_ratio_cells,
    print_latex_task_rows,
)
from merge_and_rebase.eval.rebase_metrics import audit_rebase_summary
from merge_and_rebase.eval.vision_rebase import (
    _evaluate_cross_task_source_lmc,
    _evaluate_source_lmc,
    _norm_acc,
)


def test_normalized_rebase_ratio_can_exceed_one() -> None:
    # Transport can improve over an untransported task vector.
    assert _norm_acc(0.639, 0.582) == pytest.approx(1.0979381443)


class _ToyEvalClassifier:
    def __init__(self, *, model, **_kwargs) -> None:
        self.model = model
        self._zs_text_features = None
        self._zs_text_fingerprint = None

    def build_zeroshot_text_features(self, *_args, **_kwargs) -> None:
        self._zs_text_features = torch.ones(1)

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def test_source_lmc_reports_area_below_chord_without_zip_length_error(monkeypatch) -> None:
    monkeypatch.setattr(vision_rebase_cli, "OpenClipClassifier", _ToyEvalClassifier)
    monkeypatch.setattr(source_lmc, "OpenClipClassifier", _ToyEvalClassifier)
    model = nn.Linear(1, 2)
    endpoint_a = {key: value.detach().clone() for key, value in model.state_dict().items()}
    endpoint_b = {key: value.detach().clone() for key, value in model.state_dict().items()}
    endpoint_b["bias"] = endpoint_b["bias"] + torch.tensor([0.0, 1.0])
    loaders = SimpleNamespace(
        val=[(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]))],
        test=[],
    )

    result = _evaluate_source_lmc(
        model=model,
        restore_sd=endpoint_a,
        endpoint_a_sd=endpoint_a,
        endpoint_b_sd=endpoint_b,
        clf_source=SimpleNamespace(tokenizer=None, preprocess=None, normalize=False, logit_scale=None),
        loaders_obj=loaders,
        classnames_task=[],
        source_build_cfg_task=None,
        split="val",
        first_n_batches=None,
        alphas=[0.0, 0.5, 1.0],
        device="cpu",
    )

    assert result["max_loss_barrier"] >= 0.0
    assert "area_below_loss_chord" in result


def test_cross_task_source_lmc_evaluates_both_task_contexts(monkeypatch) -> None:
    monkeypatch.setattr(vision_rebase_cli, "OpenClipClassifier", _ToyEvalClassifier)
    monkeypatch.setattr(source_lmc, "OpenClipClassifier", _ToyEvalClassifier)
    model = nn.Linear(1, 2)
    endpoint_a = {key: value.detach().clone() for key, value in model.state_dict().items()}
    endpoint_b = {key: value.detach().clone() for key, value in model.state_dict().items()}
    endpoint_b["bias"] = endpoint_b["bias"] + torch.tensor([0.0, 1.0])
    loaders = SimpleNamespace(
        val=[(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]))],
        test=[],
    )
    contexts = [
        {"task": "task_a", "classnames": [], "source_build_cfg_task": None, "source_loaders": loaders},
        {"task": "task_b", "classnames": [], "source_build_cfg_task": None, "source_loaders": loaders},
    ]

    result = _evaluate_cross_task_source_lmc(
        model=model,
        restore_sd=endpoint_a,
        endpoint_a_sd=endpoint_a,
        endpoint_b_sd=endpoint_b,
        clf_source=SimpleNamespace(tokenizer=None, preprocess=None, normalize=False, logit_scale=None),
        task_contexts=contexts,
        split="val",
        first_n_batches=None,
        alphas=[0.0, 0.5, 1.0],
        device="cpu",
    )

    assert set(result["per_task_loss"]) == {"task_a", "task_b"}
    assert len(result["average_loss"]) == 3
    assert set(result["per_task_loss_chord_gap"]) == {"task_a", "task_b"}
    assert result["max_per_task_loss_barrier"] >= 0.0
    assert torch.equal(model.state_dict()["bias"], endpoint_a["bias"])


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_normalized_rebase_ratio_rejects_invalid_raw_accuracy(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _norm_acc(value, 0.582)


def test_rebase_summary_audit_keeps_raw_accuracy_and_ratio_distinct() -> None:
    payload = {
        "test_results": {
            "per_task_baseline_accuracy": {"PCAM": 0.582},
            "per_task_absolute_accuracy": {"PCAM": 0.639},
            "per_task_normalized_accuracy_ratio": {"PCAM": 0.639 / 0.582},
        }
    }

    messages = audit_rebase_summary(payload)
    assert messages[0] == "PCAM: absolute=0.639000, baseline=0.582000, ratio=1.097938"


def test_rebase_summary_audit_rejects_ratio_as_raw_accuracy() -> None:
    payload = {
        "test_results": {
            "per_task_baseline_accuracy": {"PCAM": 0.582},
            "per_task_absolute_accuracy": {"PCAM": 1.098},
            "per_task_normalized_accuracy_ratio": {"PCAM": 1.098},
        }
    }

    with pytest.raises(ValueError, match="raw accuracy"):
        audit_rebase_summary(payload)


def test_rebase_summary_audit_warns_non_fatally_on_ratio_above_one() -> None:
    """A ratio > 1.0 surfaces a non-fatal WARNING, not a raise (decimal-ratio semantics)."""
    payload = {
        "test_results": {
            "per_task_baseline_accuracy": {"PCAM": 0.582, "STL10": 0.95},
            "per_task_absolute_accuracy": {"PCAM": 0.639, "STL10": 0.94},
            "per_task_normalized_accuracy_ratio": {"PCAM": 0.639 / 0.582, "STL10": 0.94 / 0.95},
        }
    }

    messages = audit_rebase_summary(payload)

    data_lines = [m for m in messages if "WARNING" not in m]
    assert any("PCAM" in m and "ratio=1.097938" in m for m in data_lines)
    assert any("STL10" in m and "ratio=0.989474" in m for m in data_lines)

    warnings = [m for m in messages if "WARNING" in m]
    assert len(warnings) == 1
    assert "PCAM" in warnings[0]
    assert "> 1.0" in warnings[0]
    assert "not a percentage" in warnings[0]


def test_latex_percent_cells_formats_raw_accuracy_as_percent() -> None:
    cells = _latex_percent_cells([0.639, 0.94])
    assert float(cells[0]) == pytest.approx(63.90)
    assert float(cells[1]) == pytest.approx(94.00)
    assert float(cells[2]) == pytest.approx(78.95)


def test_latex_ratio_cells_formats_ratio_as_decimal_not_percent() -> None:
    # Ratios above 1.0 are legitimate (transport beats baseline) and stay decimal.
    cells = _latex_ratio_cells([1.098, 0.950])
    assert len(cells) == 3
    assert float(cells[0]) == pytest.approx(1.098)
    assert float(cells[1]) == pytest.approx(0.950)
    assert float(cells[2]) == pytest.approx((1.098 + 0.950) / 2)
    # Never the ×100 form.
    assert all(float(c) < 110.0 for c in cells)


def test_print_latex_task_rows_routes_norm_through_ratio_formatter() -> None:
    """The norm row must carry the ratio as a decimal, not multiplied by 100."""
    per_task = [{"task": "PCAM"}, {"task": "STL10"}]
    merged_accs = [0.639, 0.94]
    norm_accs = [1.098, 0.950]

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_latex_task_rows(per_task, merged_accs, norm_accs)
    out = buf.getvalue()

    def _row(label: str) -> list[str]:
        line = next(ln for ln in out.splitlines() if ln.lstrip().startswith(label))
        line = line.rstrip()
        if line.endswith(r"\\"):
            line = line[:-2].rstrip()
        _, _, rest = line.partition(": ")
        return [c.strip() for c in rest.split("&")]

    top1_cells = _row("top1")
    norm_cells = _row("norm")

    # top1 is still percent-valued (raw accuracy in [0, 1] * 100).
    assert float(top1_cells[0]) == pytest.approx(63.90)
    assert float(top1_cells[1]) == pytest.approx(94.00)

    # norm is the decimal ratio, not the percentage: 1.098 stays 1.098, never 109.80.
    assert float(norm_cells[0]) == pytest.approx(1.098)
    assert float(norm_cells[1]) == pytest.approx(0.950)
    assert float(norm_cells[0]) < 10.0  # would be ~109.8 if the ×100 bug were present
