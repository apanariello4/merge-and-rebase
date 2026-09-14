"""Artifact-integrity tests for the crossed swap merge stage.

The merge stage consumes transported deltas that a separate single-vector run
produced.  A merged row and its single-vector reference row are comparable only
if both consumed the identical transported vector, so the hash gate and the
reference lookup are the two things worth testing without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from merge_and_rebase.eval.vision_brace_swap_merge import (
    _cell_name,
    read_transported_cell,
    single_vector_reference,
)
from merge_and_rebase.eval.vision_brace_tv_swap import state_dict_sha256


def _write_delta_bank(root: Path, *, method: str, tasks: list[str], cells: list[tuple[str, str]]) -> None:
    for index, task in enumerate(tasks):
        task_dir = root / method / task
        task_dir.mkdir(parents=True)
        records = {}
        for activation_bank, vector_bank in cells:
            cell = _cell_name(activation_bank, vector_bank)
            delta = {"visual.w": torch.full((3,), float(index) + len(cell))}
            torch.save(delta, task_dir / f"{cell}.pt")
            records[cell] = {
                "path": str(task_dir / f"{cell}.pt"),
                "sha256": state_dict_sha256(delta),
                "activation_bank": activation_bank,
                "vector_bank": vector_bank,
            }
        (task_dir / "metadata.json").write_text(
            json.dumps({"task": task, "method": method, "cells": records}), encoding="utf-8"
        )
        (task_dir / "COMPLETE").touch()


def test_read_transported_cell_returns_the_named_cell_for_every_task(tmp_path: Path) -> None:
    tasks = ["Cars", "DTD"]
    _write_delta_bank(
        tmp_path, method="bico", tasks=tasks,
        cells=[("shared", "shared"), ("shared", "skip")],
    )

    deltas = read_transported_cell(
        tmp_path, method_name="bico", tasks=tasks, activation_bank="shared", vector_bank="skip"
    )

    assert sorted(deltas) == tasks
    # The shared/skip cell must not be served the shared/shared vector: the two
    # differ only by file, and confusing them would silently void the ablation.
    other = read_transported_cell(
        tmp_path, method_name="bico", tasks=tasks, activation_bank="shared", vector_bank="shared"
    )
    assert not torch.equal(deltas["Cars"]["visual.w"], other["Cars"]["visual.w"])


def test_read_transported_cell_rejects_a_delta_that_does_not_match_its_recorded_hash(tmp_path: Path) -> None:
    _write_delta_bank(tmp_path, method="theseus", tasks=["Cars"], cells=[("shared", "shared")])
    task_dir = tmp_path / "theseus" / "Cars"
    torch.save({"visual.w": torch.ones(3) * 99.0}, task_dir / "shared__shared.pt")

    with pytest.raises(ValueError, match="hash mismatch"):
        read_transported_cell(
            tmp_path, method_name="theseus", tasks=["Cars"], activation_bank="shared", vector_bank="shared"
        )


def test_read_transported_cell_rejects_an_unfinished_bank(tmp_path: Path) -> None:
    _write_delta_bank(tmp_path, method="theseus", tasks=["Cars"], cells=[("shared", "shared")])
    (tmp_path / "theseus" / "Cars" / "COMPLETE").unlink()

    with pytest.raises(FileNotFoundError, match="Incomplete transported-delta bank"):
        read_transported_cell(
            tmp_path, method_name="theseus", tasks=["Cars"], activation_bank="shared", vector_bank="shared"
        )


def test_read_transported_cell_rejects_a_cell_the_bank_never_crossed(tmp_path: Path) -> None:
    _write_delta_bank(tmp_path, method="bico", tasks=["Cars"], cells=[("shared", "shared")])

    with pytest.raises(KeyError, match="independent"):
        read_transported_cell(
            tmp_path, method_name="bico", tasks=["Cars"], activation_bank="shared", vector_bank="independent"
        )


def _write_swap_summary(root: Path, *, method: str, task: str, rows: list[dict]) -> None:
    task_dir = root / method / task
    task_dir.mkdir(parents=True)
    (task_dir / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_single_vector_reference_selects_the_matching_crossed_row(tmp_path: Path) -> None:
    _write_swap_summary(
        tmp_path, method="bico", task="Cars",
        rows=[
            {"activation_bank": "shared", "vector_bank": "shared", "selected_test_accuracy": 0.9},
            {"activation_bank": "shared", "vector_bank": "skip", "selected_test_accuracy": 0.8},
            {"activation_bank": "skip", "vector_bank": "shared", "selected_test_accuracy": 0.7},
        ],
    )

    reference = single_vector_reference(
        tmp_path, method_name="bico", tasks=["Cars"], activation_bank="shared", vector_bank="skip"
    )

    assert reference == {"Cars": 0.8}


def test_single_vector_reference_is_optional(tmp_path: Path) -> None:
    assert single_vector_reference(
        None, method_name="bico", tasks=["Cars"], activation_bank="shared", vector_bank="skip"
    ) is None


def test_single_vector_reference_rejects_an_ambiguous_summary(tmp_path: Path) -> None:
    _write_swap_summary(
        tmp_path, method="bico", task="Cars",
        rows=[
            {"activation_bank": "shared", "vector_bank": "skip", "selected_test_accuracy": 0.8},
            {"activation_bank": "shared", "vector_bank": "skip", "selected_test_accuracy": 0.6},
        ],
    )

    with pytest.raises(ValueError, match="found 2"):
        single_vector_reference(
            tmp_path, method_name="bico", tasks=["Cars"], activation_bank="shared", vector_bank="skip"
        )
