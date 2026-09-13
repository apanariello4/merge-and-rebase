from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_brace_source_merge_cache.py"
SPEC = importlib.util.spec_from_file_location("brace_cache_check", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _fixture(tmp_path: Path, *, missing_test: bool = False) -> tuple[Path, Path]:
    cache = tmp_path / "hf"
    (cache / "hub/models--laion--CLIP-ViT-B-16-DataComp.XL-s13B-b90K/snapshots/r").mkdir(parents=True)
    (cache / "hub/models--laion--CLIP-ViT-B-16-DataComp.XL-s13B-b90K/snapshots/r/open_clip_pytorch_model.bin").write_bytes(b"weights")
    for task, dirname in module.DATASET_DIRS.items():
        root = cache / "datasets" / dirname / "default/0.0.0/r"
        root.mkdir(parents=True)
        (root / f"{task.lower()}-train.arrow").write_bytes(b"train")
        if not missing_test:
            (root / f"{task.lower()}-test.arrow").write_bytes(b"test")
    checkpoints = {}
    for task in module.VISION8:
        path = tmp_path / task / "full_best_ep.pt"
        path.parent.mkdir()
        path.write_bytes(b"checkpoint")
        checkpoints[task] = str(path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"tasks": "all", "source_clip_model": "ViT-B-16", "source_clip_pretrained": "datacomp_xl_s13b_b90k", "tuned_ckpts": checkpoints}))
    return config, cache


def test_preflight_records_model_checkpoints_and_all_splits(tmp_path: Path) -> None:
    config, cache = _fixture(tmp_path)
    audit = module.preflight(config, cache_root=cache)
    assert audit["ok"] is True
    assert audit["model"]["path"].endswith("open_clip_pytorch_model.bin")
    assert set(audit["datasets"]) == set(module.VISION8)
    assert all(audit["datasets"][task]["splits"]["test"] for task in module.VISION8)
    assert all(item["readable"] for item in audit["checkpoints"].values())


def test_preflight_rejects_missing_arrow_split(tmp_path: Path) -> None:
    config, cache = _fixture(tmp_path, missing_test=True)
    audit = module.preflight(config, cache_root=cache)
    assert audit["ok"] is False
    assert any("has no readable cached test Arrow" in error for error in audit["errors"])


def test_preflight_rejects_model_config_without_weight_file(tmp_path: Path) -> None:
    config, cache = _fixture(tmp_path)
    model = cache / "hub/models--laion--CLIP-ViT-B-16-DataComp.XL-s13B-b90K/snapshots/r/open_clip_pytorch_model.bin"
    model.unlink()
    (model.parent / "config.json").write_text("{}")
    audit = module.preflight(config, cache_root=cache)
    assert audit["ok"] is False
    assert any("model file not found" in error for error in audit["errors"])


def test_cli_refuses_to_overwrite_audit(tmp_path: Path) -> None:
    config, cache = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    output.write_text("old")
    # The preflight uses HF_HOME when no cache root is passed.
    import os
    old = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = str(cache)
    try:
        with pytest.raises(SystemExit) as exc:
            module.main(["--config", str(config), "--output", str(output)])
        assert exc.value.code == 2
    finally:
        if old is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = old
