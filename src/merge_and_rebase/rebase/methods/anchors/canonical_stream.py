"""Streamed second moments of the canonical decoder over ALL tokens of the anchor images (no stored rows).

``canonical_source='sigma_all_tokens'`` (vision): the anchor images of the frame (``anchor_batches`` batches of the pool
draw, ``canonical_images`` of them) go through the source and the target once; at every needed block the rows (every
canonical position of every image, ``to_canonical``) are centred and scaled with the REGISTRATION ``mu`` / ``c`` of the
frame (fixed, disjoint from the anchors) BEFORE they are squared, in float64, per half of the image ids (``split_ids``,
seed ``anchor_seed``): ``Sigma_mm[i]``, ``Sigma_kk[j]`` per block and ``Sigma_mk[i, j]`` per block pair. The result is a
few ``d x d`` matrices per block, independent of the number of rows; ``heldout_soft_phi_from_sigma`` turns the halves into
the same D3 decoder as the row-based path (``canonical_stats``).

Numerics. Centring before squaring means there is no ``sum x x^T - n mu mu^T`` cancellation (the residual stream has a
large common mean); the accumulation is float64 over float32 activations, so the error vs the exact Gram of the same
float32 rows is at the float64 summation level (~1e-13 relative for 8e5 rows); the activations' own float32 rounding
(~1e-7) is common to the row-based path. ``canonical_factors_from_sigma`` resolves a direction of the Gram down to
``eps_mach * lambda_max / lambda_i``; directions below ``PINV_RTOL`` are dropped as in the frame decoders.

Cache. ``<state_dir>/sigma/<source>__<target>__<key>.pt`` holds the statistics of one (source, target, frame, pairs,
seed, images) combination; it is reused by every run on the same pair (``key`` hashes all of them and the models'
stored-frame hashes).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .._ariadne.capture import _layout_for, iter_capture_tokens
from .canonical_stats import split_ids
from .state import to_canonical

Tensor = torch.Tensor


class StreamedSigma:
    """Per-half ``Sigma_mm[i]``, ``Sigma_kk[j]``, ``Sigma_mk[(i, j)]`` of a set of block pairs, accumulated in float64."""

    def __init__(self, pairs: Sequence[tuple[int, int]], mu_m, c_m, mu_k, c_k, halves, device="cpu") -> None:
        self.pairs = sorted({(int(i), int(j)) for i, j in pairs})
        self.src = sorted({i for i, _ in self.pairs})
        self.tgt = sorted({j for _, j in self.pairs})
        self.mu_m = {i: mu_m[i].to(device, torch.float64) for i in self.src}
        self.mu_k = {j: mu_k[j].to(device, torch.float64) for j in self.tgt}
        self.c_m = {i: float(c_m[i]) for i in self.src}
        self.c_k = {j: float(c_k[j]) for j in self.tgt}
        self.halves, self.device = halves, device
        z = lambda a, b: [torch.zeros(a, b, dtype=torch.float64, device=device) for _ in range(2)]  # noqa: E731
        self.mm = {i: z(len(self.mu_m[i]), len(self.mu_m[i])) for i in self.src}
        self.kk = {j: z(len(self.mu_k[j]), len(self.mu_k[j])) for j in self.tgt}
        self.mk = {(i, j): z(len(self.mu_m[i]), len(self.mu_k[j])) for i, j in self.pairs}
        self.n = [0, 0]

    @torch.no_grad()
    def update(self, rows_m: Mapping[int, Tensor], rows_k: Mapping[int, Tensor], ids: Tensor) -> None:
        """``rows_m[i]`` ``[n, d_m]`` / ``rows_k[j]`` ``[n, d_k]`` raw rows (same ``n`` rows, same order), ``ids`` ``[n]``."""
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        masks = [torch.isin(ids, h) for h in self.halves]
        if not bool((masks[0] | masks[1]).all()):
            raise ValueError("a row id belongs to neither half")
        pm = {i: (rows_m[i].to(self.device, torch.float64) - self.mu_m[i]) / self.c_m[i] for i in self.src}
        pk = {j: (rows_k[j].to(self.device, torch.float64) - self.mu_k[j]) / self.c_k[j] for j in self.tgt}
        for h, mask in enumerate(masks):
            if not bool(mask.any()):
                continue
            mk_ = mask.to(self.device)
            sm = {i: x[mk_] for i, x in pm.items()}
            sk = {j: x[mk_] for j, x in pk.items()}
            for i, x in sm.items():
                self.mm[i][h] += x.T @ x
            for j, x in sk.items():
                self.kk[j][h] += x.T @ x
            for i, j in self.pairs:
                self.mk[(i, j)][h] += sm[i].T @ sk[j]
            self.n[h] += int(mask.sum())

    def sigma(self, i: int, j: int, half: int) -> tuple[Tensor, Tensor, Tensor, int]:
        return self.mm[i][half].cpu(), self.kk[j][half].cpu(), self.mk[(i, j)][half].cpu(), self.n[half]

    def to_payload(self) -> dict[str, Any]:
        cpu = lambda ts: [t.cpu() for t in ts]  # noqa: E731
        return {"pairs": self.pairs, "n": list(self.n),
                "mm": {i: cpu(v) for i, v in self.mm.items()}, "kk": {j: cpu(v) for j, v in self.kk.items()},
                "mk": {k: cpu(v) for k, v in self.mk.items()}}


class LoadedSigma:
    """Read-only view of a cached payload (same ``sigma`` accessor)."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.p = payload
        self.pairs = [tuple(x) for x in payload["pairs"]]

    def sigma(self, i: int, j: int, half: int) -> tuple[Tensor, Tensor, Tensor, int]:
        p = self.p
        return p["mm"][i][half], p["kk"][j][half], p["mk"][(i, j)][half], int(p["n"][half])


def cache_key(config, source, target, pairs: Sequence[tuple[int, int]]) -> str:
    """Hash of everything the statistics depend on: the frame, the number of images, the split seed, the pairs, both models'
    stored-frame hashes (registration ``mu`` / ``c`` and the anchors' draw)."""
    desc = {"frame": config.frame(), "images": config.canonical_images, "seed": config.anchor_seed,
            "canonical_tokens": config.canonical_tokens, "pairs": sorted(map(list, pairs)),
            "blocks": [source.meta.get("blocks_sha256"), target.meta.get("blocks_sha256")]}
    return hashlib.sha256(json.dumps(desc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def stream_sigma(
    source_model, target_model, source, target, source_batches, target_batches, pairs, *, config, device,
    stats_device="cpu",
) -> StreamedSigma:
    """One pass of the anchor images through the source and the target (the target at its base weights)."""
    offset = 0
    n_images = sum(int(_layout_for(None).batch_size(b)) for b in source_batches)
    if config.canonical_images is not None and n_images != int(config.canonical_images):
        raise ValueError(f"canonical_images={config.canonical_images} but the frame's anchor batches hold {n_images} "
                         "images (anchor_batches x batch size): set canonical_images accordingly")
    uniq = torch.arange(n_images)
    halves = split_ids(uniq, config.anchor_seed)
    acc = StreamedSigma(pairs, source.mu, source.c, target.mu, target.c, halves, device=stats_device)
    req_m = {str(i): (i, "boundary") for i in acc.src}
    req_k = {str(j): (j, "boundary") for j in acc.tgt}
    gen_m = iter_capture_tokens(source_model, source_batches, req_m, device)
    gen_k = iter_capture_tokens(target_model, target_batches, req_k, device)
    canon = int(config.canonical_tokens)
    for vm, vk in zip(gen_m, gen_k, strict=True):
        rows_m = {i: to_canonical(vm[str(i)], canon) for i in acc.src}
        rows_k = {j: to_canonical(vk[str(j)], canon) for j in acc.tgt}
        b = int(next(iter(rows_m.values())).shape[0])
        ids = (offset + torch.arange(b)).repeat_interleave(canon)
        acc.update({i: x.reshape(-1, x.shape[-1]) for i, x in rows_m.items()},
                   {j: x.reshape(-1, x.shape[-1]) for j, x in rows_k.items()}, ids)
        offset += b
    return acc


def streamed_sigma_for_pair(
    state, source, target, source_model, target_model, target_base_sd, source_batches, target_batches, pairs, *,
    config, device, stats_device="cpu",
):
    """Cached statistics (``state_dir/sigma``) or one streaming pass; returns an object with ``sigma(i, j, half)``."""
    key = cache_key(config, source, target, pairs)
    path = None
    if state.state_dir is not None:
        path = Path(state.state_dir) / "sigma" / f"{source.model_id}__{target.model_id}__{key}.pt"
        if path.exists():
            return LoadedSigma(torch.load(path, map_location="cpu", weights_only=False))
    current = {k: v.detach().cpu().clone() for k, v in target_model.state_dict().items()}
    try:
        target_model.load_state_dict(dict(target_base_sd), strict=True)
        acc = stream_sigma(source_model, target_model, source, target, source_batches, target_batches, pairs,
                           config=config, device=device, stats_device=stats_device)
    finally:
        target_model.load_state_dict(current, strict=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(acc.to_payload(), tmp)
        tmp.replace(path)
    return acc
