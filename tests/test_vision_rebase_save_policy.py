"""``save_transported_tvs``: opt-in policy for saving transported task vectors (P5.11d).

``"if_dir_given"`` (the default, also when the key is absent) is the historical behaviour: nothing is saved
unless ``save_transported_tvs_dir`` is given, and the summary does not mention the policy. ``"auto"`` derives
``<summary_dir>/transported_tvs`` and additionally saves the merged single-transport delta under a new name.
Driven through the real ``main()`` on the golden tiny offline world.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from golden._hashing import hash_json, hash_tensor_dict
from golden.test_main_golden import CASES, World, _base_cfg, _launch

PER_TASK_CASE = "theseus_equal_depth_none_fixed_alpha_save_tvs"
MERGED_CASE = "theseus_merge_then_rebase"
MERGED_FILE = "merged_merge_then_rebase_theseus_transported_native.pt"


def _run(case_name: str, root: Path, monkeypatch, **overrides):
    case = CASES[case_name]
    world = World(**case.world)
    cfg = _base_cfg(world, root, **{**case.cfg, **overrides})
    rec = _launch(cfg, root, monkeypatch, world)[0]
    assert rec.status == "success" and rec.summary is not None
    return rec


def _pt_hashes(directory: Path) -> dict[str, str]:
    return {
        p.name: hash_tensor_dict(torch.load(p, map_location="cpu", weights_only=True))
        for p in sorted(directory.glob("*.pt"))
    }


def test_default_policy_is_unchanged_and_not_recorded(tmp_path, monkeypatch):
    absent = _run(PER_TASK_CASE, tmp_path / "absent", monkeypatch, save_transported_tvs_dir=str(tmp_path / "a"))
    explicit = _run(
        PER_TASK_CASE,
        tmp_path / "explicit",
        monkeypatch,
        save_transported_tvs_dir=str(tmp_path / "b"),
        save_transported_tvs="if_dir_given",
    )
    assert "save_policy" not in absent.summary
    assert explicit.summary["save_policy"] == {
        "save_transported_tvs": "if_dir_given",
        "transported_tvs_dir": str(tmp_path / "b"),
    }
    # Same files, same tensors; the summaries differ only by the additive key (and the volatile paths).
    assert _pt_hashes(tmp_path / "a") == _pt_hashes(tmp_path / "b")
    assert _pt_hashes(tmp_path / "a")
    stripped = {k: v for k, v in explicit.summary.items() if k != "save_policy"}
    assert list(stripped) == list(absent.summary)

    def comparable(summary):
        # The artifact paths differ by run directory; compare their file names instead.
        out = dict(summary)
        out["transported_artifacts"] = {
            t: [Path(p).name for p in ps] for t, ps in summary["transported_artifacts"].items()
        }
        return out

    assert hash_json(comparable(stripped)) == hash_json(comparable(absent.summary))


def test_if_dir_given_without_a_dir_saves_nothing(tmp_path, monkeypatch):
    rec = _run(PER_TASK_CASE, tmp_path / "run", monkeypatch, save_transported_tvs="if_dir_given")
    assert not list((tmp_path / "run").rglob("*.pt"))
    assert rec.summary["transported_artifacts"] == {}
    assert rec.summary["save_policy"] == {"save_transported_tvs": "if_dir_given", "transported_tvs_dir": None}


def test_auto_derives_the_directory_from_the_summary_dir(tmp_path, monkeypatch):
    rec = _run(
        PER_TASK_CASE, tmp_path / "run", monkeypatch, save_transported_tvs="auto", save_transported_tvs_legacy=False
    )
    summary_dir = Path(rec.metadata["summary_path"]).parent
    saved_dir = summary_dir / "transported_tvs"
    names = sorted(p.name for p in saved_dir.glob("*.pt"))
    assert names == ["DTD_theseus_transported_native.pt", "MNIST_theseus_transported_native.pt"]
    assert rec.summary["save_policy"] == {"save_transported_tvs": "auto", "transported_tvs_dir": str(saved_dir)}
    assert sorted(rec.summary["transported_artifacts"]) == ["DTD", "MNIST"]
    # Per-task vectors are bitwise those of the historical explicit-dir run.
    explicit = _run(
        PER_TASK_CASE,
        tmp_path / "explicit",
        monkeypatch,
        save_transported_tvs_dir=str(tmp_path / "e"),
        save_transported_tvs_legacy=False,
    )
    assert explicit.summary["transported_artifacts"]
    assert _pt_hashes(saved_dir) == _pt_hashes(tmp_path / "e")


def test_auto_also_saves_the_merged_single_transport_delta_write_once(tmp_path, monkeypatch):
    saved = tmp_path / "tvs"
    rec = _run(
        MERGED_CASE, tmp_path / "run", monkeypatch, save_transported_tvs="auto", save_transported_tvs_dir=str(saved)
    )
    assert [p.name for p in saved.glob("*.pt")] == [MERGED_FILE]
    assert list(rec.summary["transported_artifacts"]) == ["merged_merge_then_rebase"]
    merged = torch.load(saved / MERGED_FILE, map_location="cpu", weights_only=True)
    assert merged and all(torch.isfinite(t).all() for t in merged.values())
    # The default policy never writes the merged delta.
    default = _run(MERGED_CASE, tmp_path / "default", monkeypatch, save_transported_tvs_dir=str(tmp_path / "d"))
    assert not list((tmp_path / "d").glob("*.pt"))
    assert default.summary["transported_artifacts"] == {}
    # Write-once: a second run into the same directory refuses to overwrite the merged vector.
    with pytest.raises(FileExistsError, match="refusing to overwrite merged transported vector"):
        _run(
            MERGED_CASE,
            tmp_path / "again",
            monkeypatch,
            save_transported_tvs="auto",
            save_transported_tvs_dir=str(saved),
        )


def test_unknown_policy_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="save_transported_tvs must be one of"):
        _run(PER_TASK_CASE, tmp_path / "run", monkeypatch, save_transported_tvs="always")
