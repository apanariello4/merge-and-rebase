"""Text-defined units of a calibration sequence (word units), and where each one ends in a given tokenization.

A unit is defined on the raw TEXT only, never on a tokenizer: the default unit is a word, a maximal run of
non-whitespace characters (``\\S+``), identified by its index in the text and its character end offset. Units are
therefore the same for every model that reads the text, whatever its tokenizer.

The token that represents unit ``k`` in a tokenization is the LAST token of the unit: the token whose character span
ends exactly at the unit's end offset (fast tokenizer ``offset_mapping``). When no token ends there (a token crosses the
word boundary, e.g. ``".\\n"``), or that token was cut by truncation, the unit is MISSING in that tokenization only;
the units themselves are never redefined. Special tokens (BOS, chat markers registered as special) and padding are
never units.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

_WORD = re.compile(r"\S+")

#: Value of ``token_word_units`` at a token that is not the last token of any unit.
NOT_A_UNIT = -1


def word_end_offsets(text: str) -> list[int]:
    """Character end offset of every word (``\\S+`` match) of ``text``, in text order (word index = list index)."""
    return [m.end() for m in _WORD.finditer(text)]


def token_word_units(
    text: str,
    offsets: Sequence[Sequence[int]],
    input_ids: Sequence[int],
    attention_mask: Sequence[int] | None = None,
    special_ids: Iterable[int] = (),
) -> list[int]:
    """Per token position, the index of the word whose last token it is, else ``NOT_A_UNIT``.

    ``offsets`` are the tokenizer's ``(start, end)`` character spans of ``text``; a token is the last token of word
    ``k`` when its end equals word ``k``'s end offset and its span is non-empty. When several tokens end at the same
    character (a character split into several byte tokens), the last of them represents the word. Padding (attention
    0) and special ids are never units. Words with no such token (boundary-crossing token, truncation) are missing.
    """
    ends = {end: k for k, end in enumerate(word_end_offsets(text))}
    special = set(int(i) for i in special_ids)
    out = [NOT_A_UNIT] * len(offsets)
    last_pos: dict[int, int] = {}
    for pos, ((start, end), tok) in enumerate(zip(offsets, input_ids, strict=True)):
        if attention_mask is not None and not attention_mask[pos]:
            continue
        if int(tok) in special or int(end) <= int(start):
            continue
        k = ends.get(int(end))
        if k is None:
            continue
        if k in last_pos:  # an earlier token ended at the same character: the later one is the unit's last token
            out[last_pos[k]] = NOT_A_UNIT
        out[pos] = k
        last_pos[k] = pos
    return out


def tokenizer_special_ids(tokenizer: Any) -> set[int]:
    """Ids of every special token of ``tokenizer`` (BOS/EOS/PAD/... and added tokens registered as special)."""
    ids = {int(i) for i in (getattr(tokenizer, "all_special_ids", None) or [])}
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for idx, tok in decoder.items():
        if bool(getattr(tok, "special", False)):
            ids.add(int(idx))
    return ids


def tokenize_word_units(
    tokenizer: Any, texts: Sequence[str], *, max_length: int, add_special_tokens: bool = True
) -> tuple[Any, list[list[int]]]:
    """``(encoding, word units)`` of ``texts`` tokenized exactly as ``build_text_calibration_loader`` pads them
    (truncation and padding to ``max_length``); the offsets are dropped from the returned encoding."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("word units need a fast tokenizer (character offsets); got a slow one.")
    special_kwargs = {} if add_special_tokens else {"add_special_tokens": False}
    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
        return_offsets_mapping=True,
        **special_kwargs,
    )
    special = tokenizer_special_ids(tokenizer)
    units = [
        token_word_units(text, offsets, ids, attention, special)
        for text, offsets, ids, attention in zip(
            texts, enc["offset_mapping"], enc["input_ids"], enc["attention_mask"], strict=True
        )
    ]
    enc.pop("offset_mapping")
    return enc, units
