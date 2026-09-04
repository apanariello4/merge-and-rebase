from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from merge_and_rebase.eval.brace_diagnostics import BRACEDiagnosticCollector
from merge_and_rebase.eval.block_extension import BlockExtender


def test_collector_is_opt_in_and_serializes_cpu_fp32(tmp_path):
    model = nn.Module()
    model.visual = nn.Module()
    model.visual.weight = nn.Parameter(torch.ones(2, dtype=torch.float16))

    # No collector means no side effects on the extender.
    extender = BlockExtender(model, model, "cpu", verbose=False, show_progress=False)
    extender._diagnostic_context = {
        "structural_step": 1,
        "final_block": 2,
        "source_block": 0,
    }
    extender._record_correction("base", "q", torch.eye(2), torch.zeros(2))
    assert extender.diagnostic_collector is None

    collector = BRACEDiagnosticCollector(tmp_path / "run", {"task": "Cars"})
    collector.record_map(
        mode="independent",
        endpoint="base",
        structural_step=1,
        final_block=2,
        source_block=0,
        component="q",
        W=torch.eye(2, dtype=torch.float16),
        b=torch.zeros(2, dtype=torch.float16),
    )
    collector.save_endpoint("base", model)
    collector.finalize()

    payload = torch.load(tmp_path / "run" / "maps.pt", weights_only=False)
    assert payload["maps"][0]["W"].dtype == torch.float32
    assert payload["maps"][0]["W"].device.type == "cpu"
    endpoint = torch.load(tmp_path / "run" / "endpoint_base.pt", weights_only=False)
    assert endpoint["visual.weight"].dtype == torch.float32
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["map_records"] == 1
    assert (tmp_path / "run" / "COMPLETE").exists()


def test_collector_rejects_nonempty_namespace(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    (path / "historical").write_text("keep")
    with pytest.raises(FileExistsError):
        BRACEDiagnosticCollector(path)


def test_shared_ft_records_exact_base_map(tmp_path):
    collector = BRACEDiagnosticCollector(tmp_path / "run")
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([0.5, -0.25])
    collector.record_map(
        mode="shared",
        endpoint="base",
        structural_step=1,
        final_block=1,
        source_block=0,
        component="q",
        W=W,
        b=b,
    )
    # _record_corrections is the same path used after shared FT application.
    extender = object.__new__(BlockExtender)
    extender.diagnostic_collector = collector
    extender.diagnostic_mode = "shared"
    extender._diagnostic_context = {"structural_step": 1, "final_block": 1, "source_block": 0}
    extender._record_corrections("ft", {"q": (W, b)})
    assert torch.equal(collector.maps[0]["W"], collector.maps[1]["W"])
    assert torch.equal(collector.maps[0]["b"], collector.maps[1]["b"])
    assert collector.maps[1]["endpoint"] == "ft"
