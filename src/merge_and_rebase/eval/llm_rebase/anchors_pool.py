"""Generic text pool of an LLM ``anchors`` run (AGENT_NEXT_PHASE.md L1): the anchor and registration sequences.

The dataset is ``method.params.anchor_pool`` (``wikitext103``: WikiText-103 raw, validation split); the run key
``anchors_pool`` holds the text-side settings, all optional:

    min_chars    600   paragraphs at least this long (drops the " = Heading = " lines and short fragments)
    n_sequences  None  pool size (None: exactly the frame's draw, (anchor_batches + registration_batches) x batch_size)
    seed         0     seeded choice of the paragraphs
    max_length   None  tokens per sequence, BOS included (None: calibration_max_length)
    batch_size   None  (None: calibration_batch_size)

Every model registered in one anchors state must see this same pool: the frame stores the draw identity (the
loader's sample ids hash the texts) and the sha256 of its content token rows (``text_units='word'``: of its raw
texts, the tokenizers may differ), and refuses a model that differs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...data.llm_calibration import build_text_calibration_loader

#: anchor_pool name -> (HF dataset path, config, split, text column).
POOL_DATASETS = {"wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation", "text")}

_KEYS = {"min_chars", "n_sequences", "seed", "max_length", "batch_size"}


def resolve_anchors_pool(
    cfg: Mapping[str, Any], anchors_cfg: Any, *, calib_batch_size: int, calib_max_length: int
) -> dict[str, Any]:
    """The pool record of the run: dataset from ``anchors_cfg.anchor_pool``, the rest from ``cfg['anchors_pool']``."""
    raw = dict(cfg.get("anchors_pool") or {})
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ValueError(f"unknown anchors_pool keys: {unknown}; allowed: {sorted(_KEYS)}")
    if anchors_cfg.anchor_pool not in POOL_DATASETS:
        raise ValueError(f"anchor_pool {anchors_cfg.anchor_pool!r} has no text dataset ({sorted(POOL_DATASETS)})")
    batch_size = int(raw.get("batch_size") or calib_batch_size)
    draw = (int(anchors_cfg.anchor_batches) + int(anchors_cfg.registration_batches)) * batch_size
    n_sequences = int(raw.get("n_sequences") or draw)
    if n_sequences < draw:
        raise ValueError(f"anchors_pool.n_sequences={n_sequences} is smaller than the frame's draw of {draw}")
    path, name, split, column = POOL_DATASETS[anchors_cfg.anchor_pool]
    return {
        "dataset": anchors_cfg.anchor_pool,
        "hf_path": path,
        "hf_name": name,
        "split": split,
        "text_column": column,
        "min_chars": int(raw.get("min_chars", 600)),
        "n_sequences": n_sequences,
        "seed": int(raw.get("seed", 0)),
        "max_length": int(raw.get("max_length") or calib_max_length),
        "batch_size": batch_size,
    }


def load_pool_texts(pool: Mapping[str, Any]) -> list[str]:
    """``n_sequences`` paragraphs of at least ``min_chars`` characters, a seeded choice in dataset order."""
    import datasets
    import torch

    ds = datasets.load_dataset(pool["hf_path"], pool["hf_name"], split=pool["split"])
    paragraphs = [t for t in ds[pool["text_column"]] if isinstance(t, str) and len(t) >= int(pool["min_chars"])]
    if len(paragraphs) < int(pool["n_sequences"]):
        raise ValueError(
            f"anchor pool: {len(paragraphs)} paragraphs of >= {pool['min_chars']} chars, need {pool['n_sequences']}"
        )
    order = torch.randperm(len(paragraphs), generator=torch.Generator().manual_seed(int(pool["seed"])))
    return [paragraphs[i] for i in sorted(order[: int(pool["n_sequences"])].tolist())]


def build_anchor_pool_loaders(
    source_tokenizer: Any, target_tokenizer: Any, pool: Mapping[str, Any], *, word_units: bool = False
):
    """``(source loader, target loader)``: the same pool texts under each model's tokenizer (``word_units``: the
    loaders also carry the per-token word index, ``text_units='word'``)."""
    kwargs = dict(texts=load_pool_texts(pool), batch_size=int(pool["batch_size"]), max_length=int(pool["max_length"]))
    if word_units:
        kwargs["word_units"] = True
    return (
        build_text_calibration_loader(tokenizer=source_tokenizer, **kwargs),
        build_text_calibration_loader(tokenizer=target_tokenizer, **kwargs),
    )
