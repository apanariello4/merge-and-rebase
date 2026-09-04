"""Opt-in, side-effect-free artifact capture for BRACE endpoint diagnostics.

The collector deliberately stores only fitted affine maps and visual endpoint
weights.  Activation banks are never retained or serialized here.  It is
disabled unless a caller explicitly constructs and passes an instance to
``run_block_extension``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


class BRACEDiagnosticCollector:
    """Collect fitted maps and endpoint state dicts for one diagnostic run.

    ``output_dir`` is treated as a new artifact namespace.  A non-empty
    directory is rejected so a rerun cannot silently overwrite historical
    results.  All tensors are detached, copied to CPU, and converted to FP32
    at capture time.
    """

    def __init__(self, output_dir: str | os.PathLike[str], metadata: Mapping[str, Any] | None = None):
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Diagnostic output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self._maps: list[dict[str, Any]] = []
        self._endpoints: list[str] = []
        self._finalized = False

    @staticmethod
    def _cpu_fp32(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", dtype=torch.float32).clone()

    @property
    def maps(self) -> list[dict[str, Any]]:
        """In-memory map records (useful for tests and immediate inspection)."""

        return self._maps

    def record_map(
        self,
        *,
        mode: str,
        endpoint: str,
        structural_step: int,
        final_block: int,
        source_block: int,
        component: str,
        W: torch.Tensor,
        b: torch.Tensor,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot record diagnostics after finalize().")
        self._maps.append(
            {
                "mode": str(mode),
                "endpoint": str(endpoint),
                "structural_step": int(structural_step),
                "final_block": int(final_block),
                "source_block": int(source_block),
                "component": str(component),
                "W": self._cpu_fp32(W),
                "b": self._cpu_fp32(b),
            }
        )

    def save_endpoint(self, endpoint: str, model: torch.nn.Module) -> Path:
        """Save only ``model.visual`` parameters as a CPU FP32 state dict."""

        if self._finalized:
            raise RuntimeError("Cannot save diagnostics after finalize().")
        endpoint = str(endpoint)
        state = {
            key: self._cpu_fp32(value)
            for key, value in model.state_dict().items()
            if key.startswith("visual.") and torch.is_tensor(value)
        }
        if not state:
            raise ValueError("Model has no visual state to save.")
        path = self.output_dir / f"endpoint_{endpoint}.pt"
        if path.exists():
            raise FileExistsError(f"Diagnostic endpoint already exists: {path}")
        tmp = path.with_name(path.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        self._endpoints.append(endpoint)
        return path

    def finalize(self) -> Path:
        """Atomically write map payload, metadata, and a completion marker."""

        if self._finalized:
            return self.output_dir / "metadata.json"
        maps_path = self.output_dir / "maps.pt"
        metadata_path = self.output_dir / "metadata.json"
        complete_path = self.output_dir / "COMPLETE"
        for path in (maps_path, metadata_path, complete_path):
            if path.exists():
                raise FileExistsError(f"Diagnostic artifact already exists: {path}")

        payload = {"maps": self._maps}
        maps_tmp = maps_path.with_name(maps_path.name + ".tmp")
        torch.save(payload, maps_tmp)
        os.replace(maps_tmp, maps_path)

        metadata = dict(self.metadata)
        metadata.update({"map_records": len(self._maps), "endpoints": list(self._endpoints)})
        meta_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
        meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(meta_tmp, metadata_path)
        complete_path.touch()
        self._finalized = True
        return metadata_path
