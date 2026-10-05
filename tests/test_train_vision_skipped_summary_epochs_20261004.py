"""The skipped-task summary keeps a recorded epoch 0 (release review fix, 2026-10-04: ``or -1`` mapped 0 to -1)."""

from __future__ import annotations

from pathlib import Path

from merge_and_rebase.finetune.train_vision import _build_skipped_existing_task_summary


def _summary(tmp_path: Path, existing):
    paths = {
        "summary_path": tmp_path / "summary.json",
        "task_dir": tmp_path,
        "best_ckpt_path": tmp_path / "best.pt",
        "last_ckpt_path": tmp_path / "last.pt",
    }
    return _build_skipped_existing_task_summary(
        task="DTD",
        strategy="full",
        strategy_cfg={},
        regularization_cfg={},
        save_format="pt",
        save_checkpoints=True,
        save_last_epoch=False,
        checkpoint_paths=paths,
        existing_summary=existing,
    )


def test_epoch_zero_is_kept_and_missing_is_minus_one(tmp_path):
    out = _summary(tmp_path, {"best_epoch": 0, "last_epoch": 0})
    assert (out["best_epoch"], out["last_epoch"]) == (0, 0)
    out = _summary(tmp_path, {"best_epoch": 3})
    assert (out["best_epoch"], out["last_epoch"]) == (3, -1)
    assert _summary(tmp_path, None)["best_epoch"] == -1
