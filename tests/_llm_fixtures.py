"""Offline fixtures shared by the LLM characterization tests (Phase 7, S1).

Sharing mechanism: ``tests/`` has no ``__init__.py``, so pytest (prepend import
mode) puts ``tests/`` itself on ``sys.path`` for every test module that lives in
it. Test files therefore do ``from _llm_fixtures import ...``, which works under
both ``python -m pytest`` and plain ``pytest`` and does not depend on the
repository root being importable as ``tests``.

Nothing here touches the network: tokenizers are built in memory, models are
random-initialised from a config, and the real Qwen tokenizer is loaded with
``local_files_only=True`` (skip if it is not in the cache).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch

PAD = "<|pad|>"
UNK = "<|unk|>"

# Short texts: every one fits in 12 tokens under the whitespace tokenizer, so
# max_length 12 vs 24 differ only in the amount of padding, never in content.
CALIB_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "premise one entails hypothesis two",
    "a short sentence",
    "numbers and words 1 2 3 and more words",
    "calibration text number five",
    "tiny decoder",
]


def local_tokenizer(texts: list[str], padding_side: str = "right") -> Any:
    """Whitespace/WordLevel tokenizer wrapped as a PreTrainedTokenizerFast (no files).

    ``<|pad|>`` is a registered special token, so it tokenizes to the pad id even
    when it appears inside a text (used to test pad-id-as-content behaviour).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words: list[str] = []
    for text in texts:
        for word in text.split():
            if word not in (PAD, UNK) and word not in words:
                words.append(word)
    vocab = {PAD: 0, UNK: 1}
    for word in words:
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token=PAD,
        unk_token=UNK,
        padding_side=padding_side,
    )


def _init_seeded(cls: Any, config: Any, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return cls(config).eval()


def tiny_qwen2(
    *,
    layers: int = 2,
    hidden: int = 32,
    heads: int = 4,
    kv_heads: int = 2,
    inter: int = 64,
    vocab: int = 64,
    seed: int = 0,
    tie_word_embeddings: bool = True,
) -> torch.nn.Module:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=inter,
        vocab_size=vocab,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    return _init_seeded(Qwen2ForCausalLM, config, seed)


def tiny_qwen3(
    *,
    layers: int = 2,
    hidden: int = 32,
    heads: int = 4,
    kv_heads: int = 2,
    inter: int = 64,
    head_dim: int = 8,
    vocab: int = 64,
    seed: int = 0,
    tie_word_embeddings: bool = True,
) -> torch.nn.Module:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=inter,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    return _init_seeded(Qwen3ForCausalLM, config, seed)


def perturbed_copy(model: torch.nn.Module, *, scale: float = 0.05, seed: int = 1) -> torch.nn.Module:
    """Deterministic 'fine-tuned' copy: base + scale * N(0, 1) on every parameter."""
    ft = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(scale * torch.randn_like(p))
    return ft


_REAL_TOKENIZER_CANDIDATES = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B")


def real_qwen_tokenizer_or_skip(padding_side: str = "right") -> Any:
    """Cached Qwen2.5 tokenizer (``local_files_only``); skips the test if not cached.

    Mirrors ``TextLM.build``: pad token falls back to EOS when unset, so pad id ==
    EOS id (``<|endoftext|>``) -- masks must come from attention_mask only.
    """
    from transformers import AutoTokenizer

    last: Exception | None = None
    for name in _REAL_TOKENIZER_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True, padding_side=padding_side)
        except Exception as exc:  # not cached / offline
            last = exc
            continue
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        return tok
    pytest.skip(f"no cached Qwen tokenizer ({last!r})")
