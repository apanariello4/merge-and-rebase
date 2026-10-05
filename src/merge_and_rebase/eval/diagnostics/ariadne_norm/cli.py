"""``python -m merge_and_rebase.eval.diagnostics.ariadne_norm --config <json>``.

Config keys:

- ``family``: ``"vision"`` or ``"llm"``.
- ``reference_summary``: summary JSON of the reference run whose fit is reproduced and measured.
- ``tasks``: list of task names, or ``"all"`` (every task of the reference run).
- ``out_dir``: result directory; must not exist (results are never overwritten).
- ``alpha_grid``: alphas for the joint realization curve (N2).
- ``row_positions``: target positions whose row banks are kept for N5, or ``"auto"`` (4 positions spread over depth).
- ``generic_num_batches``: off-task batches for N6. Family-specific keys are documented in the adapter modules.

Per task: ``<out_dir>/<task>/record.json``, ``arrays.pt`` (spectra) and the reproduced unit task vector under
``<out_dir>/transported_tvs``. ``<out_dir>/run.json`` records config hash, code fingerprint and timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import socket
import time
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.run_logging import _code_fingerprint

from .runner import run_task


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def auto_row_positions(depth: int) -> list[int]:
    return sorted({0, depth // 3, (2 * depth) // 3, depth - 1})


def _build_inputs(cfg: dict, device: str):
    if cfg["family"] == "vision":
        from .vision import build_vision_inputs

        return build_vision_inputs(cfg, device)
    if cfg["family"] == "llm":
        from .llm import build_llm_inputs

        return build_llm_inputs(cfg, device)
    raise ValueError(f"unknown family {cfg['family']!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    out_dir = Path(cfg["out_dir"])
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; ariadne_norm results are write-once")
    out_dir.mkdir(parents=True)
    run_meta: dict[str, Any] = {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": cfg,
        "code_fingerprint": _code_fingerprint(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "torch": torch.__version__,
        "tasks": {},
    }
    tv_dir = out_dir / "transported_tvs"
    tv_dir.mkdir()
    started = time.perf_counter()
    for inp in _build_inputs(cfg, args.device):
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rows = cfg.get("row_positions", "auto")
        row_positions = auto_row_positions(inp.pairing.target_depth) if rows == "auto" else [int(r) for r in rows]
        record, arrays, tv = run_task(
            inp, alpha_grid=[float(a) for a in cfg["alpha_grid"]], row_positions=row_positions
        )
        task_dir = out_dir / inp.task
        task_dir.mkdir()
        (task_dir / "record.json").write_text(json.dumps(_jsonable(record), indent=1, sort_keys=True))
        torch.save(arrays, task_dir / "arrays.pt")
        torch.save(tv, tv_dir / f"{inp.task}_ariadne_transported_native.pt")
        run_meta["tasks"][inp.task] = {
            "seconds": time.perf_counter() - t0,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "n0_task_vector_passed": (record["n0"]["task_vector"]["reference"] or {}).get("passed"),
            "n0_rows_passed": (record["n0"]["rows"] or {}).get("passed"),
        }
        (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))
        print(f"[ariadne_norm] {inp.task}: done in {run_meta['tasks'][inp.task]['seconds']:.0f}s", flush=True)
    run_meta["total_seconds"] = time.perf_counter() - started
    (out_dir / "run.json").write_text(json.dumps(_jsonable(run_meta), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
