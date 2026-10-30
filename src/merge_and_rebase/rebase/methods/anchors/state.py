"""Method state: the shared frame (pool draw, anchor rows) and the registered models, optionally on disk.

Layout of ``state_dir``::

    frame.json                  # AnchorsConfig.frame() + pool draw (dataset identity, indices, batch size)
                                #   + the anchor rows (image, canonical position); written once, never rewritten
    models/<model_id>/meta.json # architecture, block count, widths, sha256 of blocks.pt
    models/<model_id>/blocks.pt # per block j: P_j [T, d] float32, mu_j [d] float64, c_j
    models/<model_id>/norm_filter_k<k>.pt  # optional (anchor_filter='norm'): per block the median norm of the
                                #   centred registration rows, mu/c without rows above k x median, anchor flags

Registering a model writes only under ``models/<model_id>/``: the frame and every other model are never
touched (the "no recomputation" property, checked by hashing the files).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .._ariadne.capture import iter_capture_tokens, paired_calibration
from .._shared import _interp_2d_tokens
from .config import AnchorsConfig
from .core import MomentAccumulator, anchor_frame, kernel

Tensor = torch.Tensor


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def vision_block_count(model) -> int:
    return len(model.visual.transformer.resblocks)


def to_canonical(tokens: Tensor, canonical_tokens: int) -> Tensor:
    """Resample ``[B, N, d]`` tokens to the canonical grid (CLS kept, square patch grid bilinear)."""
    return _interp_2d_tokens(tokens, int(canonical_tokens))


@dataclass
class RegisteredModel:
    model_id: str
    meta: dict[str, Any]
    p: list[Tensor]  # per block, [T, d] float32 as stored
    mu: list[Tensor]  # per block, [d] float64
    c: list[float]  # per block
    extras: dict[str, Any] = field(default_factory=dict)  # optional per-model statistics (e.g. norm filters)

    @property
    def depth(self) -> int:
        return len(self.p)

    def block(self, j: int) -> tuple[Tensor, Tensor, float]:
        """``(P_j float64, K_j float64, c_j)``."""
        p = self.p[j].to(torch.float64)
        return p, kernel(p), self.c[j]


class AnchorState:
    """Frame + registered models. In memory when ``state_dir`` is None, else mirrored on disk."""

    def __init__(self, config: AnchorsConfig, state_dir: str | os.PathLike | None = None) -> None:
        self.config = config
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.frame: dict[str, Any] | None = None
        self.models: dict[str, RegisteredModel] = {}
        if self.state_dir is not None:
            self._load_frame()

    # ------------------------------------------------------------------ frame

    def _load_frame(self) -> None:
        path = self.state_dir / "frame.json"
        if not path.exists():
            return
        with open(path) as fh:
            frame = json.load(fh)
        expected = self.config.frame()
        stored = frame["config"]
        mismatch = {k: (stored.get(k), v) for k, v in expected.items() if stored.get(k) != v}
        if mismatch:
            raise ValueError(
                f"anchors state at {self.state_dir} was built with a different frame (stored, requested): {mismatch}"
            )
        self.frame = frame

    def _establish_frame(self, pool_meta: dict[str, Any], batch_sizes: list[int]) -> None:
        cfg = self.config
        n_images = sum(batch_sizes[: cfg.anchor_batches])
        positions = list(range(cfg.canonical_tokens))
        if cfg.anchor_positions == "patch":
            positions = positions[1:]
        n_candidates = n_images * len(positions)
        if cfg.num_anchors > n_candidates:
            raise ValueError(
                f"num_anchors={cfg.num_anchors} exceeds the {n_candidates} candidate rows "
                f"({n_images} anchor images x {len(positions)} positions)"
            )
        gen = torch.Generator().manual_seed(int(cfg.anchor_seed))
        if cfg.anchor_sampling == "whole_images":
            if cfg.num_anchors % len(positions):
                raise ValueError(
                    f"anchor_sampling='whole_images' needs num_anchors a multiple of {len(positions)} positions, "
                    f"got {cfg.num_anchors}"
                )
            images = sorted(int(i) for i in torch.randperm(n_images, generator=gen)[: cfg.num_anchors // len(positions)])
            rows = [[img, pos] for img in images for pos in positions]
        elif cfg.anchor_sampling == "one_per_image":
            if cfg.num_anchors > n_images:
                raise ValueError(
                    f"anchor_sampling='one_per_image' needs num_anchors <= {n_images} anchor images, got {cfg.num_anchors}"
                )
            images = torch.randperm(n_images, generator=gen)[: cfg.num_anchors]
            picks = torch.randint(len(positions), (cfg.num_anchors,), generator=gen)
            rows = sorted([int(i), positions[int(k)]] for i, k in zip(images, picks, strict=True))
        else:
            perm = torch.randperm(n_candidates, generator=gen)
            chosen = sorted(int(i) for i in perm[: cfg.num_anchors])
            rows = [[i // len(positions), positions[i % len(positions)]] for i in chosen]
        self.frame = {
            "config": cfg.frame(),
            "pool": {
                "dataset_identity": pool_meta["dataset_identity"],
                "indices": list(pool_meta["indices"]),
                "batch_size": int(pool_meta["batch_size"]),
                "batch_sizes": list(batch_sizes),
            },
            "anchor_rows": rows,
        }
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            _json_dump(self.state_dir / "frame.json", self.frame)

    def _check_pool(self, pool_meta: dict[str, Any], batch_sizes: list[int]) -> None:
        pool = self.frame["pool"]
        if pool_meta["dataset_identity"] != pool["dataset_identity"]:
            raise ValueError("anchor pool dataset differs from the one the frame was built on")
        if list(pool_meta["indices"]) != pool["indices"] or list(batch_sizes) != pool["batch_sizes"]:
            raise ValueError("anchor pool draw (indices / batch sizes) differs from the frame's")

    # ------------------------------------------------------------------ models

    def has_model(self, model_id: str) -> bool:
        if model_id in self.models:
            return True
        return self.state_dir is not None and (self.state_dir / "models" / model_id / "meta.json").exists()

    def model(self, model_id: str) -> RegisteredModel:
        if model_id not in self.models:
            if self.state_dir is None or not self.has_model(model_id):
                raise KeyError(f"model {model_id!r} is not registered")
            self.models[model_id] = self._load_model(model_id)
        return self.models[model_id]

    def model_meta_hash(self, model_id: str) -> str:
        """Content hash of a registered model's frame: the stored ``blocks_sha256``, else a hash of its tensors."""
        reg = self.model(model_id)
        stored = reg.meta.get("blocks_sha256")
        if stored:
            return str(stored)
        h = hashlib.sha256()
        for t in (*reg.p, *reg.mu):
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        h.update(repr(list(reg.c)).encode())
        return h.hexdigest()

    def _model_dir(self, model_id: str) -> Path:
        if "/" in model_id or model_id in {"", ".", ".."}:
            raise ValueError(f"invalid model_id {model_id!r}")
        return self.state_dir / "models" / model_id

    def _load_model(self, model_id: str) -> RegisteredModel:
        d = self._model_dir(model_id)
        with open(d / "meta.json") as fh:
            meta = json.load(fh)
        if _file_sha256(d / "blocks.pt") != meta["blocks_sha256"]:
            raise ValueError(f"anchors state for {model_id!r} is corrupted (blocks.pt hash mismatch)")
        blocks = torch.load(d / "blocks.pt", map_location="cpu", weights_only=True)
        return RegisteredModel(model_id, meta, blocks["p"], blocks["mu"], [float(c) for c in blocks["c"]])

    def _save_model(self, reg: RegisteredModel) -> None:
        d = self._model_dir(reg.model_id)
        d.mkdir(parents=True, exist_ok=True)
        blocks_path = d / "blocks.pt"
        torch.save({"p": reg.p, "mu": reg.mu, "c": torch.tensor(reg.c, dtype=torch.float64)}, blocks_path)
        reg.meta["blocks_sha256"] = _file_sha256(blocks_path)
        _json_dump(d / "meta.json", reg.meta)

    def load_norm_filter(self, model_id: str, *, k: float) -> dict[str, Any]:
        """The stored high-norm filter of a registered model (for models that are not live in this run, e.g. the
        extra models of a GPA universe). Raises if it was never computed (``norm_filter`` needs the model)."""
        reg = self.model(model_id)
        key = f"norm_k{k:g}"
        if key not in reg.extras:
            path = None if self.state_dir is None else self._model_dir(model_id) / f"norm_filter_k{k:g}.pt"
            if path is None or not path.exists():
                raise KeyError(f"no stored norm filter k={k:g} for {model_id!r}; compute it with norm_filter()")
            reg.extras[key] = torch.load(path, map_location="cpu", weights_only=True)
        return reg.extras[key]

    @torch.no_grad()
    def norm_filter(self, model, model_id: str, pool_loader, *, k: float, device) -> dict[str, Any]:
        """High-norm token filter of a registered model (AGENT_NEXT step 2, variant F1), computed once and stored.

        Replays the registration images of the frame's pool draw: per block, the norms ``||x - mu||`` of the
        centred registration rows, their median ``m``, and ``mu_k``, ``c_k`` recomputed without the rows above
        ``k * m``; an anchor is flagged when its own centred norm (``||P_row|| c``) is above ``k * m``. Stored as
        ``models/<id>/norm_filter_k<k>.pt``; nothing else in the state is written.
        """
        reg = self.model(model_id)
        if f"norm_k{k:g}" in reg.extras:
            return reg.extras[f"norm_k{k:g}"]
        path = None if self.state_dir is None else self._model_dir(model_id) / f"norm_filter_k{k:g}.pt"
        if path is not None and path.exists():
            out = torch.load(path, map_location="cpu", weights_only=True)
            reg.extras[f"norm_k{k:g}"] = out
            return out
        cfg = self.config
        batches, _, pool_meta = paired_calibration(
            pool_loader, pool_loader, num_batches=cfg.anchor_batches + cfg.registration_batches, seed=cfg.pool_seed
        )
        batch_sizes = [int(b[0].shape[0]) if isinstance(b, (tuple, list)) else int(b.shape[0]) for b in batches]
        self._check_pool(pool_meta, batch_sizes)
        reg_batches = batches[cfg.anchor_batches :]
        depth = reg.depth
        requests = {str(j): (j, "boundary") for j in range(depth)}
        norms: list[list[Tensor]] = [[] for _ in range(depth)]
        for values in iter_capture_tokens(model, reg_batches, requests, device):
            for j in range(depth):
                canon = to_canonical(values[str(j)], cfg.canonical_tokens).double()
                norms[j].append((canon - reg.mu[j].to(canon.device)).norm(dim=-1).reshape(-1).cpu())
        medians = torch.tensor([float(torch.cat(n).median()) for n in norms], dtype=torch.float64)
        moments = [MomentAccumulator() for _ in range(depth)]
        for values in iter_capture_tokens(model, reg_batches, requests, device):
            for j in range(depth):
                canon = to_canonical(values[str(j)], cfg.canonical_tokens).double()
                rows = canon.reshape(-1, canon.shape[-1])
                keep = (rows - reg.mu[j].to(rows.device)).norm(dim=-1) <= float(k) * float(medians[j])
                moments[j].update(rows[keep])
        finals = [m.finalize() for m in moments]
        flags = torch.stack(
            [reg.p[j].double().norm(dim=1) * float(reg.c[j]) > float(k) * float(medians[j]) for j in range(depth)]
        )
        out = {
            "k": float(k),
            "median": medians,
            "mu": [f[0] for f in finals],
            "c": torch.tensor([f[1] for f in finals], dtype=torch.float64),
            "anchor_flags": flags,  # [depth, T] bool
        }
        if path is not None:
            torch.save(out, path)
        reg.extras[f"norm_k{k:g}"] = out
        return out

    @torch.no_grad()
    def register_model(
        self,
        model,
        model_id: str,
        pool_loader,
        *,
        device,
        model_meta: dict[str, Any] | None = None,
    ) -> RegisteredModel:
        """Step 1: place the anchors in ``model`` (base weights) at every block; two passes over the pool draw.

        Idempotent: an already registered ``model_id`` is returned as stored, without any forward pass.
        Vision (OpenCLIP layout) only.
        """
        if self.has_model(model_id):
            return self.model(model_id)
        cfg = self.config
        batches, _, pool_meta = paired_calibration(
            pool_loader,
            pool_loader,
            num_batches=cfg.anchor_batches + cfg.registration_batches,
            seed=cfg.pool_seed,
        )
        if len(batches) != cfg.anchor_batches + cfg.registration_batches:
            raise ValueError(
                f"anchor pool too small: {len(batches)} batches drawn, "
                f"{cfg.anchor_batches + cfg.registration_batches} required"
            )
        batch_sizes = [int(b[0].shape[0]) if isinstance(b, (tuple, list)) else int(b.shape[0]) for b in batches]
        if self.frame is None:
            self._establish_frame(pool_meta, batch_sizes)
        else:
            self._check_pool(pool_meta, batch_sizes)
        depth = vision_block_count(model)
        requests = {str(j): (j, "boundary") for j in range(depth)}
        rows = self.frame["anchor_rows"]
        offsets = [0]
        for size in batch_sizes:
            offsets.append(offsets[-1] + size)
        anchor_rows: list[list[Tensor]] = [[] for _ in range(depth)]
        moments = [MomentAccumulator() for _ in range(depth)]
        native_tokens = None
        width = None
        for b, values in enumerate(iter_capture_tokens(model, batches, requests, device)):
            if b < cfg.anchor_batches:
                lo, hi = offsets[b], offsets[b + 1]
                sel = [(img - lo, pos) for img, pos in rows if lo <= img < hi]
                if not sel:
                    continue
                img_idx = torch.tensor([s[0] for s in sel])
                pos_idx = torch.tensor([s[1] for s in sel])
            for j in range(depth):
                tokens = values[str(j)]
                native_tokens = int(tokens.shape[1])
                width = int(tokens.shape[2])
                canon = to_canonical(tokens, cfg.canonical_tokens)
                if b < cfg.anchor_batches:
                    anchor_rows[j].append(canon[img_idx, pos_idx].to(torch.float64))
                else:
                    moments[j].update(canon)
        p_blocks, mu_blocks, c_blocks = [], [], []
        for j in range(depth):
            s_anc = torch.cat(anchor_rows[j], dim=0)
            if s_anc.shape[0] != cfg.num_anchors:
                raise RuntimeError(f"collected {s_anc.shape[0]} anchor rows at block {j}, expected {cfg.num_anchors}")
            mu, c = moments[j].finalize()
            p_blocks.append(anchor_frame(s_anc, mu, c).float())
            mu_blocks.append(mu)
            c_blocks.append(c)
        meta = {
            "model_id": model_id,
            "depth": depth,
            "width": width,
            "native_tokens": native_tokens,
            "registration_rows_per_block": moments[0].n,
            **(model_meta or {}),
        }
        reg = RegisteredModel(model_id, meta, p_blocks, mu_blocks, c_blocks)
        if self.state_dir is not None:
            self._save_model(reg)
        self.models[model_id] = reg
        return reg
