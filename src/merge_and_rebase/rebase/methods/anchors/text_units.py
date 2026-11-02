"""Word-unit rows of the decoder path (``text_units='word'``): models with DIFFERENT tokenizers on the same texts.

Units are defined on the raw text (``data/text_units.py``): word ``k`` of sequence ``s``. The calibration and pool
loaders (``build_text_calibration_loader(word_units=True)``) carry, per token, the index of the word whose last token
it is (``word_unit``, -1 elsewhere). The activation of a unit in a model is the activation of that last token.

Everything below reuses Ariadne's capture unchanged: a model's rows are selected by the row mask attached to its own
batches (``ROW_MASK_KEY``, read by ``iter_capture_tokens`` and by the shared ridge), here one mask PER MODEL over its
own tokenization, marking the last tokens of the selected units. Within a sequence the last tokens of words are in
word order in every tokenization, so two masks that select the same units give two ``[1, N, d]`` banks whose rows
are the same units in the same order (checked).

A unit is never a padding token, a special token, or the first content token of its sequence (the attention sink,
as on the token path); a unit cut by truncation or whose last token crosses the word boundary is missing in that
model only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .._ariadne.capture import ROW_MASK_KEY, _first_content_position, capture_tokens, paired_calibration

Tensor = torch.Tensor

#: Batch key of the per-token word index (``build_text_calibration_loader(word_units=True)``).
WORD_UNIT_KEY = "word_unit"


def _unit_ids(batch: Mapping[str, Any]) -> Tensor:
    ids = batch.get(WORD_UNIT_KEY) if isinstance(batch, Mapping) else None
    if not isinstance(ids, torch.Tensor):
        raise ValueError(
            "text_units='word' needs calibration / pool batches with word units "
            "(build_text_calibration_loader(word_units=True))"
        )
    return ids.long().cpu()


def model_unit_mask(batch: Mapping[str, Any], family_adapter) -> tuple[Tensor, Tensor]:
    """``(mask, ids)``: bool ``[B, T]`` at the last token of every unit present in this tokenization (no padding, no
    special token, not the first content token), and the word index of every position (-1 off units)."""
    ids = _unit_ids(batch)
    content = family_adapter.content_mask(batch).bool().cpu()
    if tuple(ids.shape) != tuple(content.shape):
        raise ValueError(f"word units {tuple(ids.shape)} do not match the batch {tuple(content.shape)}")
    mask = (ids >= 0) & content & ~_first_content_position(content)
    return mask, ids


def unit_keys(mask: Tensor, ids: Tensor) -> list[tuple[int, int]]:
    """``(sequence in batch, word)`` of every selected position, in row-major (capture bank) order."""
    return [(int(s), int(ids[s, p])) for s, p in torch.nonzero(mask).tolist()]


def _wrap(batch: Mapping[str, Any], family_adapter, mask: Tensor) -> dict[str, Any]:
    wrapped = dict(family_adapter.extract_calibration_batch(batch))
    wrapped[ROW_MASK_KEY] = mask
    wrapped[WORD_UNIT_KEY] = _unit_ids(batch)  # not a model input (extract_calibration_batch drops it)
    return wrapped


def own_unit_batch(batch: Mapping[str, Any], family_adapter) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """One model alone (pool draw): its batch with the row mask of all its own units, and their keys."""
    mask, ids = model_unit_mask(batch, family_adapter)
    if not bool(mask.any()):
        raise ValueError("pool batch has no word unit after padding / special / first-token exclusion")
    return _wrap(batch, family_adapter, mask), unit_keys(mask, ids)


def pair_unit_batches(
    source_batch: Mapping[str, Any], target_batch: Mapping[str, Any], family_adapter
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Source and target batches of the same texts with the row masks of the units present in BOTH tokenizations
    (each mask on its own tokens), and the unit counts ``{source, target, common}``."""
    ms, ids_s = model_unit_mask(source_batch, family_adapter)
    mt, ids_t = model_unit_mask(target_batch, family_adapter)
    if ms.shape[0] != mt.shape[0]:
        raise ValueError(f"source / target batches hold {ms.shape[0]} / {mt.shape[0]} sequences")
    keep_s = torch.zeros_like(ms)
    keep_t = torch.zeros_like(mt)
    for b in range(ms.shape[0]):
        common = sorted(set(ids_s[b][ms[b]].tolist()) & set(ids_t[b][mt[b]].tolist()))
        if not common:
            continue
        sel = torch.tensor(common, dtype=torch.long)
        keep_s[b] = ms[b] & torch.isin(ids_s[b], sel)
        keep_t[b] = mt[b] & torch.isin(ids_t[b], sel)
    if unit_keys(keep_s, ids_s) != unit_keys(keep_t, ids_t):  # word order = token order in both tokenizations
        raise RuntimeError("source and target unit rows are not in the same order")
    if not bool(keep_s.any()):
        raise ValueError("calibration batch has no word unit present in both tokenizations")
    counts = {"source": int(ms.sum()), "target": int(mt.sum()), "common": int(keep_s.sum())}
    return _wrap(source_batch, family_adapter, keep_s), _wrap(target_batch, family_adapter, keep_t), counts


def texts_sha256(loader) -> str:
    """Identity of the raw texts a word-unit loader was built on (set by ``build_text_calibration_loader``)."""
    sha = getattr(getattr(loader, "dataset", None), "texts_sha256", None)
    if not sha:
        raise ValueError("text_units='word' needs loaders built with build_text_calibration_loader(word_units=True)")
    return str(sha)


def paired_word_calibration(
    source_loader, target_loader, *, num_batches: int, seed: int | None, family_adapter
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """``paired_calibration`` (same seeded draw, same metadata) with word-unit rows: each batch pair carries the
    row masks of the units present in both tokenizations. The metadata adds the unit counts."""
    if texts_sha256(source_loader) != texts_sha256(target_loader):
        raise ValueError("source / target calibration loaders were built on different texts")
    raw_s, raw_t, metadata = paired_calibration(source_loader, target_loader, num_batches=num_batches, seed=seed)
    src, tgt = [], []
    totals = {"source": 0, "target": 0, "common": 0}
    for a, b in zip(raw_s, raw_t, strict=True):
        wa, wb, counts = pair_unit_batches(a, b, family_adapter)
        src.append(wa)
        tgt.append(wb)
        for key in totals:
            totals[key] += counts[key]
    metadata = dict(metadata)
    metadata.update(
        {
            "rows": "word_units",
            "texts_sha256": texts_sha256(source_loader),
            "n_rows_content": totals["common"],
            "n_units_source": totals["source"],
            "n_units_target": totals["target"],
            "n_units_missing_in_target": totals["source"] - totals["common"],
            "n_units_missing_in_source": totals["target"] - totals["common"],
            "exclude_first_token": True,
        }
    )
    return src, tgt, metadata


def capture_paired_word_activations(
    source_base_model,
    source_ft_model,
    target_base_model,
    source_loader,
    target_loader,
    pairing,
    *,
    num_batches: int,
    seed: int | None,
    device,
    family_adapter,
) -> dict[str, Any]:
    """``capture_paired_boundary_activations`` on word-unit rows: the source pair on the source tokenization, the
    target on its own, each bank holding the common units in the same order. Same result layout (the target
    batches, with the target's unit row masks, are the ridge rows of the shared fit)."""
    source_batches, target_batches, metadata = paired_word_calibration(
        source_loader, target_loader, num_batches=num_batches, seed=seed, family_adapter=family_adapter
    )
    src_req = {str(i): (i, "boundary") for i in sorted(set(int(i) for i in pairing.pairing))}
    tgt_req = {str(j): (j, "boundary") for j in range(pairing.target_depth)}
    base = capture_tokens(source_base_model, source_batches, src_req, device, family_adapter=family_adapter)
    ft = capture_tokens(source_ft_model, source_batches, src_req, device, family_adapter=family_adapter)
    tgt = capture_tokens(target_base_model, target_batches, tgt_req, device, family_adapter=family_adapter)
    return {
        "source_base_outputs": {int(k): v for k, v in base.items()},
        "source_ft_outputs": {int(k): v for k, v in ft.items()},
        "target_base_outputs_by_position": {int(k): v for k, v in tgt.items()},
        "target_batches": target_batches,
        "calibration": metadata,
    }
