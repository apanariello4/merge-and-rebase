"""Method state: the shared frame (pool draw, anchor rows) and the registered models, optionally on disk.

Layout of ``state_dir``::

    frame.json                  # AnchorsConfig.frame() + pool draw (dataset identity, indices, batch size)
                                #   + the anchor rows (image, canonical position); written once, never rewritten
                                #   (decoders: rows (sequence, token position), pool["tokens"] = sha256 of the
                                #   content token ids, so every registered model must see identical tokens;
                                #   text_units='word': rows (sequence, word index), pool["units"] = sha256 of the
                                #   raw texts, tokenizers may differ: the anchors are words present in every model
                                #   registered with the frame, text_units.py)
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

from .._ariadne.capture import ROW_MASK_KEY, iter_capture_tokens, paired_calibration
from .._ariadne.layouts import _layout_for
from .._shared import _interp_2d_tokens
from .config import AnchorsConfig
from .core import MomentAccumulator, anchor_frame, kernel
from .text_units import WORD_UNIT_KEY, own_unit_batch, texts_sha256, unit_keys

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


def to_rows(tokens: Tensor, canonical_tokens: int, text: bool) -> Tensor:
    """The anchor-frame rows of captured tokens: the canonical grid (vision), or the decoder capture itself.

    A decoder capture under a row mask is a ``[1, N, d]`` bank of content rows (padding and the first token already
    dropped by ``paired_calibration``); it has no grid, so it is never resampled.
    """
    if not text:
        return to_canonical(tokens, canonical_tokens)
    if tokens.dim() != 3 or tokens.shape[0] != 1:
        raise ValueError(f"decoder rows must be a [1, N, d] content-row bank, got {tuple(tokens.shape)}")
    return tokens


def block_count(model, family_adapter=None) -> int:
    """Number of residual blocks (OpenCLIP resblocks, or the decoder layers of ``family_adapter``)."""
    return vision_block_count(model) if family_adapter is None else _layout_for(family_adapter).block_count(model)


def _batch_size(batch) -> int:
    if isinstance(batch, dict):
        return int(batch["input_ids"].shape[0])
    return int(batch[0].shape[0]) if isinstance(batch, (tuple, list)) else int(batch.shape[0])


def text_pool_tokens(batches: list[Any], family_adapter) -> dict[str, Any]:
    """Identity of the token rows a decoder pool draw feeds the models: sha256 of every batch's row mask (shape and
    values) and of the token ids on it. Padding ids are left out (base and instruct tokenizers of one vocabulary
    may pad with different ids); the content ids, their positions and the BOS exclusion must match exactly."""
    h = hashlib.sha256()
    rows = 0
    for batch in batches:
        mask = batch[ROW_MASK_KEY].bool().cpu()
        ids = family_adapter.extract_calibration_batch(batch)["input_ids"].cpu()
        h.update(repr(tuple(mask.shape)).encode())
        h.update(mask.to(torch.uint8).contiguous().numpy().tobytes())
        h.update(ids[mask].to(torch.int64).contiguous().numpy().tobytes())
        rows += int(mask.sum())
    return {"token_rows_sha256": h.hexdigest(), "content_rows": rows, "exclude_first_token": True}


def text_pool_units(pool_loader) -> dict[str, Any]:
    """Identity of a word-unit pool (``text_units='word'``): the sha256 of its raw texts. Units are defined on the
    text alone, so it is the same under every tokenizer; which units a model has is checked anchor by anchor."""
    return {"unit": "word", "texts_sha256": texts_sha256(pool_loader), "exclude_first_token": True}


_POOL_GRID = {"pool2": 2, "pool4": 4}


def units_per_image(unit: str, canonical_tokens: int) -> int:
    """Number of anchor units of kind ``unit`` in one image on the canonical grid (CLS + g x g patches)."""
    if unit in ("image_mean", "cls"):
        return 1
    g = _patch_side(canonical_tokens)
    return max(1, round(g / _POOL_GRID[unit])) ** 2


def _patch_side(canonical_tokens: int) -> int:
    g = int(round((int(canonical_tokens) - 1) ** 0.5))
    if g * g != int(canonical_tokens) - 1:
        raise ValueError(f"centroid anchors need a square patch grid, got {canonical_tokens} canonical tokens")
    return g


def unit_features(canon: Tensor, unit: str) -> Tensor:
    """``[B, U, d]`` centroid rows of the canonical tokens ``[B, 1 + g*g, d]`` (AGENT_KERNEL.md R1a/R1b)."""
    if unit == "cls":
        return canon[:, :1]
    patches = canon[:, 1:]
    if unit == "image_mean":
        return patches.mean(dim=1, keepdim=True)
    b, n, d = patches.shape
    g = _patch_side(n + 1)
    grid = patches.reshape(b, g, g, d).permute(0, 3, 1, 2)
    side = max(1, round(g / _POOL_GRID[unit]))
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid.float(), side).to(patches.dtype)
    return pooled.permute(0, 2, 3, 1).reshape(b, side * side, d)


@dataclass
class RegisteredModel:
    model_id: str
    meta: dict[str, Any]
    p: list[Tensor]  # per block, [T, d] float32 as stored
    mu: list[Tensor]  # per block, [d] float64
    c: list[float]  # per block
    extras: dict[str, Any] = field(default_factory=dict)  # optional per-model statistics (e.g. norm filters)
    # image / document id of every anchor row of ``p`` ([T] int64), attached by AnchorState from the frame (not stored
    # per model); None for a model built outside a state. Used by the held-out canonical weights (split by id).
    anchor_ids: Tensor | None = field(default=None, compare=False, repr=False)

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

    def _establish_frame(
        self,
        pool_meta: dict[str, Any],
        batch_sizes: list[int],
        *,
        text_masks: list[Tensor] | None = None,
        tokens: dict[str, Any] | None = None,
        unit_candidates: list[list[int]] | None = None,
    ) -> None:
        if unit_candidates is not None:
            rows = self._draw_text_rows(unit_candidates, "anchor-batch words present in every registered tokenization")
        elif text_masks is None:
            rows = self._image_anchor_rows(batch_sizes)
        else:
            rows = self._text_anchor_rows(text_masks)
        identity_key = "units" if unit_candidates is not None else "tokens"
        self.frame = {
            "config": self.config.frame(),
            "pool": {
                "dataset_identity": pool_meta["dataset_identity"],
                "indices": list(pool_meta["indices"]),
                "batch_size": int(pool_meta["batch_size"]),
                "batch_sizes": list(batch_sizes),
                **({} if tokens is None else {identity_key: tokens}),
            },
            "anchor_rows": rows,
        }
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            _json_dump(self.state_dir / "frame.json", self.frame)

    def _text_anchor_rows(self, masks: list[Tensor]) -> list[list[int]]:
        """Decoder frame: ``num_anchors`` (sequence, token position) rows drawn uniformly (``anchor_seed``) among the
        content rows of the anchor sequences (row masks: no padding, no first token), in capture order."""
        cfg = self.config
        candidates: list[list[int]] = []
        offset = 0
        for mask in masks:
            candidates += [[offset + s, p] for s, p in torch.nonzero(mask.bool().cpu()).tolist()]
            offset += int(mask.shape[0])
        return self._draw_text_rows(candidates, f"content token rows of the {offset} anchor sequences")

    def _draw_text_rows(self, candidates: list[list[int]], what: str) -> list[list[int]]:
        """``num_anchors`` rows drawn uniformly (``anchor_seed``) among the ordered ``candidates``, kept in order."""
        cfg = self.config
        if cfg.num_anchors > len(candidates):
            raise ValueError(f"num_anchors={cfg.num_anchors} exceeds the {len(candidates)} {what}")
        gen = torch.Generator().manual_seed(int(cfg.anchor_seed))
        if cfg.anchor_sampling == "k_per_document":  # k rows from each of T / k documents (candidate[0] = document id)
            k = int(cfg.anchor_tokens_per_image)
            by_doc: dict[int, list[list[int]]] = {}
            for cand in candidates:
                by_doc.setdefault(int(cand[0]), []).append(cand)
            docs = sorted(d for d, rows in by_doc.items() if len(rows) >= k)
            need = cfg.num_anchors // k
            if need > len(docs):
                raise ValueError(
                    f"anchor_sampling='k_per_document' needs {need} documents with >= {k} candidate rows, got {len(docs)}"
                )
            picked = sorted(docs[int(i)] for i in torch.randperm(len(docs), generator=gen)[:need])
            out: list[list[int]] = []
            for d in picked:
                rows = by_doc[d]
                out += [rows[int(i)] for i in sorted(torch.randperm(len(rows), generator=gen)[:k].tolist())]
            return out
        chosen = sorted(int(i) for i in torch.randperm(len(candidates), generator=gen)[: cfg.num_anchors])
        return [candidates[i] for i in chosen]

    def _image_anchor_rows(self, batch_sizes: list[int]) -> list[list[int]]:
        cfg = self.config
        n_images = sum(batch_sizes[: cfg.anchor_batches])
        positions = list(range(cfg.canonical_tokens))
        if cfg.anchor_unit != "token":  # centroid units: "positions" index the units of one image
            positions = list(range(units_per_image(cfg.anchor_unit, cfg.canonical_tokens)))
        elif cfg.anchor_positions == "patch":
            positions = positions[1:]
        n_candidates = n_images * len(positions)
        if cfg.anchor_sampling == "k_per_document":
            raise ValueError("anchor_sampling='k_per_document' applies to decoder (text) pools; use 'k_per_image'")
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
        elif cfg.anchor_sampling == "k_per_image":
            k = int(cfg.anchor_tokens_per_image)
            if k > len(positions) or cfg.num_anchors // k > n_images:
                raise ValueError(
                    f"anchor_sampling='k_per_image' needs k={k} <= {len(positions)} positions and "
                    f"num_anchors / k = {cfg.num_anchors // k} <= {n_images} anchor images"
                )
            images = sorted(int(i) for i in torch.randperm(n_images, generator=gen)[: cfg.num_anchors // k])
            rows = []
            for img in images:
                picks = sorted(int(i) for i in torch.randperm(len(positions), generator=gen)[:k])
                rows += [[img, positions[i]] for i in picks]
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
        return rows

    def _check_pool(
        self, pool_meta: dict[str, Any], batch_sizes: list[int], tokens: dict[str, Any] | None = None
    ) -> None:
        pool = self.frame["pool"]
        if pool_meta["dataset_identity"] != pool["dataset_identity"]:
            raise ValueError("anchor pool dataset differs from the one the frame was built on")
        if list(pool_meta["indices"]) != pool["indices"] or list(batch_sizes) != pool["batch_sizes"]:
            raise ValueError("anchor pool draw (indices / batch sizes) differs from the frame's")
        if self.config.text_units == "word":
            if tokens != pool.get("units"):
                raise ValueError(
                    "anchor pool texts differ from the frame's (word units are defined on the raw texts, so every "
                    "model must read the same pool texts)"
                )
            return
        if (tokens is None) != ("tokens" not in pool) or (tokens is not None and tokens != pool["tokens"]):
            raise ValueError(
                "anchor pool token rows differ from the frame's (tokenizer, max_length, padding or modality): "
                "anchors pairs token rows by position, so every model must see identical tokens"
            )

    def _word_draw(self, pool_loader, family_adapter):
        """Word-unit pool draw: the seeded batches, each with the row mask of ALL this tokenization's own units, the
        pool metadata, and the ``[sequence, word]`` keys of the anchor batches' units (global sequence, row-major)."""
        cfg = self.config
        raw, _, pool_meta = paired_calibration(
            pool_loader, pool_loader, num_batches=cfg.anchor_batches + cfg.registration_batches, seed=cfg.pool_seed
        )
        if len(raw) != cfg.anchor_batches + cfg.registration_batches:
            raise ValueError(
                f"anchor pool too small: {len(raw)} batches drawn, "
                f"{cfg.anchor_batches + cfg.registration_batches} required"
            )
        batches, keys, offset = [], [], 0
        for b, batch in enumerate(raw):
            wrapped, own = own_unit_batch(batch, family_adapter)
            batches.append(wrapped)
            if b < cfg.anchor_batches:
                keys += [[offset + s, w] for s, w in own]
            offset += _batch_size(wrapped)
        return batches, pool_meta, keys

    def _draw_words(self, pool_loader, family_adapter, frame_loaders) -> tuple[list[Any], list[int], dict[str, Any]]:
        """``_draw`` for ``text_units='word'`` (see there)."""
        batches, pool_meta, keys = self._word_draw(pool_loader, family_adapter)
        batch_sizes = [_batch_size(b) for b in batches]
        units = text_pool_units(pool_loader)
        if self.frame is None:
            present = {tuple(k) for k in keys}
            for other in frame_loaders:
                if text_pool_units(other) != units:
                    raise ValueError("frame_loaders: a pool loader was built on different texts")
                _, other_meta, other_keys = self._word_draw(other, family_adapter)
                if list(other_meta["indices"]) != list(pool_meta["indices"]):
                    raise ValueError("frame_loaders: a pool loader gives a different pool draw")
                present &= {tuple(k) for k in other_keys}
            candidates = [k for k in keys if tuple(k) in present]
            self._establish_frame(pool_meta, batch_sizes, tokens=units, unit_candidates=candidates)
        else:
            self._check_pool(pool_meta, batch_sizes, units)
            missing = sorted({tuple(r) for r in self.frame["anchor_rows"]} - {tuple(k) for k in keys})
            if missing:
                raise ValueError(
                    f"{len(missing)} of the frame's {len(self.frame['anchor_rows'])} anchor words have no last token "
                    f"in this model's tokenization (first (sequence, word): {missing[:3]}); the anchors are drawn "
                    "among the words of the models registered with the frame (register them together: frame_loaders)"
                )
        return batches, batch_sizes, units

    def _draw(
        self, pool_loader, family_adapter, frame_loaders=()
    ) -> tuple[list[Any], list[int], dict[str, Any] | None]:
        """The frame's pool draw (``anchor_batches + registration_batches`` batches, ``pool_seed``), checked against
        the frame when it exists (else the frame is established by the caller). Decoders: row masks attached (padding
        and the first token of every sequence dropped) and the token identity of the draw.

        ``text_units='word'``: each model's row mask holds its own word units (``text_units.py``); the identity is
        the sha256 of the pool texts. When this call establishes the frame, the anchors are drawn among the
        anchor-batch words present in this tokenization AND in the pool under every loader of ``frame_loaders``
        (the other models' tokenizers); a model registered later must have a last token for every anchor word."""
        cfg = self.config
        if family_adapter is not None and cfg.text_units == "word":
            return self._draw_words(pool_loader, family_adapter, frame_loaders)
        kwargs = {} if family_adapter is None else {"family_adapter": family_adapter, "exclude_first_token": True}
        batches, _, pool_meta = paired_calibration(
            pool_loader,
            pool_loader,
            num_batches=cfg.anchor_batches + cfg.registration_batches,
            seed=cfg.pool_seed,
            **kwargs,
        )
        if len(batches) != cfg.anchor_batches + cfg.registration_batches:
            raise ValueError(
                f"anchor pool too small: {len(batches)} batches drawn, "
                f"{cfg.anchor_batches + cfg.registration_batches} required"
            )
        batch_sizes = [_batch_size(b) for b in batches]
        tokens = None if family_adapter is None else text_pool_tokens(batches, family_adapter)
        if self.frame is None:
            masks = None if family_adapter is None else [b[ROW_MASK_KEY] for b in batches[: cfg.anchor_batches]]
            self._establish_frame(pool_meta, batch_sizes, text_masks=masks, tokens=tokens)
        else:
            self._check_pool(pool_meta, batch_sizes, tokens)
        return batches, batch_sizes, tokens

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
        self._attach_ids(self.models[model_id])
        return self.models[model_id]

    def _attach_ids(self, reg: RegisteredModel) -> None:
        """Row ids of the frame (first column of ``anchor_rows``; rows are drawn sorted by id, which is the row order of
        ``p``: the registration pass concatenates the anchor batches in id order)."""
        if reg.anchor_ids is None and self.frame is not None:
            reg.anchor_ids = torch.tensor([int(r[0]) for r in self.frame["anchor_rows"]], dtype=torch.long)

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
    def norm_filter(
        self, model, model_id: str, pool_loader, *, k: float, device, family_adapter=None
    ) -> dict[str, Any]:
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
        text = family_adapter is not None
        batches, _, _ = self._draw(pool_loader, family_adapter)
        reg_batches = batches[cfg.anchor_batches :]
        depth = reg.depth
        requests = {str(j): (j, "boundary") for j in range(depth)}
        norms: list[list[Tensor]] = [[] for _ in range(depth)]
        for values in iter_capture_tokens(model, reg_batches, requests, device, family_adapter=family_adapter):
            for j in range(depth):
                canon = to_rows(values[str(j)], cfg.canonical_tokens, text).double()
                norms[j].append((canon - reg.mu[j].to(canon.device)).norm(dim=-1).reshape(-1).cpu())
        medians = torch.tensor([float(torch.cat(n).median()) for n in norms], dtype=torch.float64)
        moments = [MomentAccumulator() for _ in range(depth)]
        for values in iter_capture_tokens(model, reg_batches, requests, device, family_adapter=family_adapter):
            for j in range(depth):
                canon = to_rows(values[str(j)], cfg.canonical_tokens, text).double()
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
        family_adapter=None,
        frame_loaders=(),
    ) -> RegisteredModel:
        """Step 1: place the anchors in ``model`` (base weights) at every block; two passes over the pool draw.

        Idempotent: an already registered ``model_id`` is returned as stored, without any forward pass.
        ``family_adapter=None``: vision (OpenCLIP layout, canonical token grid). With a decoder family adapter the
        rows are the content tokens of the pool sequences (padding and first token dropped), never resampled;
        ``text_units='word'``: the last tokens of the pool words (``frame_loaders``: see ``_draw``).
        """
        if self.has_model(model_id):
            return self.model(model_id)
        cfg = self.config
        text = family_adapter is not None
        words = text and cfg.text_units == "word"
        batches, batch_sizes, _ = self._draw(pool_loader, family_adapter, frame_loaders)
        depth = block_count(model, family_adapter)
        requests = {str(j): (j, "boundary") for j in range(depth)}
        rows = self.frame["anchor_rows"]
        offsets = [0]
        for size in batch_sizes:
            offsets.append(offsets[-1] + size)
        anchor_rows: list[list[Tensor]] = [[] for _ in range(depth)]
        moments = [MomentAccumulator() for _ in range(depth)]
        centroid = cfg.anchor_unit != "token"
        unit_moments = [MomentAccumulator() for _ in range(depth)] if centroid else None
        native_tokens = None
        width = None
        captures = iter_capture_tokens(model, batches, requests, device, family_adapter=family_adapter)
        for b, values in enumerate(captures):
            if b < cfg.anchor_batches:
                lo, hi = offsets[b], offsets[b + 1]
                sel = [(img - lo, pos) for img, pos in rows if lo <= img < hi]
                if not sel:
                    continue
                if words:  # (sequence, word) -> index in the [1, N, d] bank of this model's own unit rows
                    mask = batches[b][ROW_MASK_KEY].bool().cpu()
                    bank = {key: n for n, key in enumerate(unit_keys(mask, batches[b][WORD_UNIT_KEY]))}
                    img_idx = torch.zeros(len(sel), dtype=torch.long)
                    pos_idx = torch.tensor([bank[(s, w)] for s, w in sel])
                elif text:  # (sequence, position) -> index in the [1, N, d] content-row bank (row-major mask order)
                    mask = batches[b][ROW_MASK_KEY].bool().cpu()
                    bank_index = torch.full(mask.shape, -1, dtype=torch.long)
                    bank_index[mask] = torch.arange(int(mask.sum()))
                    img_idx = torch.zeros(len(sel), dtype=torch.long)
                    pos_idx = torch.tensor([int(bank_index[s, p]) for s, p in sel])
                    if bool((pos_idx < 0).any()):
                        raise RuntimeError("a frame anchor row is not a content row of the pool draw")
                else:
                    img_idx = torch.tensor([s[0] for s in sel])
                    pos_idx = torch.tensor([s[1] for s in sel])
            for j in range(depth):
                tokens = values[str(j)]
                native_tokens = int(tokens.shape[1])
                width = int(tokens.shape[2])
                canon = to_rows(tokens, cfg.canonical_tokens, text)
                units = unit_features(canon, cfg.anchor_unit) if centroid else canon
                if b < cfg.anchor_batches:
                    # native (float32) rows: anchor_frame casts to float64, exactly; keeps T x d x depth x 4 B, not x 8 B
                    anchor_rows[j].append(units[img_idx, pos_idx])
                else:
                    moments[j].update(canon)
                    if centroid:
                        unit_moments[j].update(units)
        p_blocks, mu_blocks, c_blocks = [], [], []
        for j in range(depth):
            s_anc = torch.cat(anchor_rows[j], dim=0)
            anchor_rows[j] = []  # release block j's rows as soon as its frame is built (host RAM)
            if s_anc.shape[0] != cfg.num_anchors:
                raise RuntimeError(f"collected {s_anc.shape[0]} anchor rows at block {j}, expected {cfg.num_anchors}")
            mu, c = moments[j].finalize()
            if centroid:  # centred by the mean of the same unit, scaled by the token spread (the scale of E)
                mu = unit_moments[j].finalize()[0]
            p_blocks.append(anchor_frame(s_anc, mu, c).float())
            mu_blocks.append(mu)
            c_blocks.append(c)
        meta = {
            "model_id": model_id,
            "depth": depth,
            "width": width,
            "native_tokens": native_tokens,
            "registration_rows_per_block": moments[0].n,
            **({"anchor_unit": cfg.anchor_unit, "mu_kind": "unit_mean"} if centroid else {}),
            # decoders: native_tokens is the content-row count of the last pool batch, not a grid
            **(
                {"rows": "words" if words else "tokens", "max_length": int(batches[0][ROW_MASK_KEY].shape[1])}
                if text
                else {}
            ),
            **(model_meta or {}),
        }
        reg = RegisteredModel(model_id, meta, p_blocks, mu_blocks, c_blocks)
        if self.state_dir is not None:
            self._save_model(reg)
        self.models[model_id] = reg
        self._attach_ids(reg)
        return reg
