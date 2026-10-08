"""Generated-answer calibration: each calibration text becomes ``prompt + the source fine-tune's own greedy answer``.

The effect of an instruct / reasoning fine-tune shows up while the model *writes* the answer; a prompt-only corpus
reaches those positions only through the last prompt token. Teacher-forcing the fine-tune's own answer puts response
positions in the corpus. Opt-in (config ``calibration_generated_answer``); with the key absent nothing here runs.

Text contract (exact): ``text = prompt + tokenizer.decode(answer_ids, skip_special_tokens=True)`` where ``prompt`` is
the string the corpus held before (with ``harness_apply_chat_template`` the already rendered chat prompt, generation
prompt included, so the answer follows it exactly as evaluation would see it) and ``answer_ids`` are the greedy tokens
up to, excluding, the first stop token (EOS; with a chat template also its end-of-turn token). No separator is
inserted. The string is re-tokenized later by the calibration loader, so the ids that are teacher-forced may differ from
the generated ids at the prompt/answer seam or for tokenizers that do not round-trip decode(encode(.)); this is
deliberate (source and target tokenize the same string) and is the reference behaviour.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Defaults of the reference sweep config (``align_with_gen_response``: max_new_tokens 256, batch_size 16).
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 16
_KEYS = frozenset({"enabled", "max_new_tokens", "batch_size", "model", "cache_path"})


@dataclass(frozen=True)
class GeneratedAnswerConfig:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    batch_size: int = DEFAULT_BATCH_SIZE
    model: str | None = None
    cache_path: str | None = None


def parse_generated_answer_cfg(raw: Any) -> GeneratedAnswerConfig | None:
    """Normalize ``config['calibration_generated_answer']``; ``None`` (absent / null / false / enabled=false) = off."""
    if raw is None or raw is False:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("calibration_generated_answer must be a mapping (or null).")
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"unknown calibration_generated_answer fields: {sorted(unknown)}; allowed: {sorted(_KEYS)}")
    if not bool(raw.get("enabled", True)):
        return None
    cfg = GeneratedAnswerConfig(
        max_new_tokens=int(raw.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        batch_size=int(raw.get("batch_size", DEFAULT_BATCH_SIZE)),
        model=None if raw.get("model") is None else str(raw["model"]),
        cache_path=None if raw.get("cache_path") is None else str(raw["cache_path"]),
    )
    if cfg.max_new_tokens <= 0 or cfg.batch_size <= 0:
        raise ValueError("calibration_generated_answer: max_new_tokens and batch_size must be > 0.")
    return cfg


def select_generator_ref(cfg: GeneratedAnswerConfig, tuned_refs: Sequence[str]) -> str:
    if cfg.model is not None:
        return cfg.model
    refs = sorted({str(r) for r in tuned_refs})
    if len(refs) != 1:
        raise ValueError(
            f"calibration_generated_answer needs 'model' when there is not exactly one tuned body; got {refs}."
        )
    return refs[0]


def generate_answers(
    prompts: Sequence[str],
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    batch_size: int,
    apply_chat_template: bool,
    device: str,
) -> tuple[list[str], dict[str, Any]]:
    """Greedy answers (decoded, stop token excluded) for ``prompts``, left-padded batches, neutral GenerationConfig."""
    import torch

    from ..eval.llm_rebase.harness import task_decoding_config

    gen_config = task_decoding_config(tokenizer, apply_chat_template=apply_chat_template)
    if gen_config.pad_token_id is None:  # no pad token at all: padding needs some id, EOS is never read back
        gen_config.pad_token_id = gen_config.eos_token_id[0] if gen_config.eos_token_id else 0
    gen_config.max_new_tokens = int(max_new_tokens)
    stop = set(gen_config.eos_token_id or [])
    # Installed on the model (as harness_decoding="task" does): generate() merges a *passed* generation_config with the
    # model's own one (transformers 5.0 let a checkpoint's repetition_penalty through), an installed one is used as is.
    saved_generation_config = getattr(model, "generation_config", None)
    model.generation_config = gen_config
    saved_side = tokenizer.padding_side
    saved_pad = tokenizer.pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    was_training = model.training
    model.to(device).eval()
    answers: list[str] = []
    lengths: list[int] = []
    hit_limit = 0
    try:
        for start in range(0, len(prompts), int(batch_size)):
            chunk = [str(p) for p in prompts[start : start + int(batch_size)]]
            # A rendered chat prompt carries its own special tokens; plain text gets the tokenizer's defaults, the
            # same ones the calibration loader adds to it.
            enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=not apply_chat_template)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc)
            for row in out[:, enc["input_ids"].shape[1] :].tolist():
                cut = next((i for i, t in enumerate(row) if t in stop), None)
                hit_limit += int(cut is None)
                ids = row if cut is None else row[:cut]
                lengths.append(len(ids))
                answers.append(tokenizer.decode(ids, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = saved_side
        tokenizer.pad_token = saved_pad
        model.train(was_training)
        model.generation_config = saved_generation_config
    stats = {
        "n": len(answers),
        "mean_response_tokens": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "n_hit_max_new_tokens": hit_limit,
    }
    return answers, stats


def _identity(
    *,
    ref: str,
    dtype: str,
    cfg: GeneratedAnswerConfig,
    apply_chat_template: bool,
    system_instruction: Any,
    chat_template_sha256: str | None,
    prompts: Sequence[str],
) -> dict[str, Any]:
    return {
        "model": ref,
        "dtype": dtype,
        "max_new_tokens": cfg.max_new_tokens,
        "apply_chat_template": bool(apply_chat_template),
        "system_instruction": system_instruction if apply_chat_template else None,
        "chat_template_sha256": chat_template_sha256,
        "prompts_sha1": hashlib.sha1("\x00".join(prompts).encode()).hexdigest(),
    }


def build_generated_texts(
    prompts: Sequence[str],
    *,
    cfg: GeneratedAnswerConfig,
    ref: str,
    generator_factory: Callable[[str], Any],
    tokenizer: Any,
    dtype: str,
    device: str,
    apply_chat_template: bool,
    system_instruction: Any = None,
    chat_template_sha256: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """``prompt + answer`` for every prompt (cache hit: no model is built) and the generation record."""
    identity = _identity(
        ref=ref,
        dtype=dtype,
        cfg=cfg,
        apply_chat_template=apply_chat_template,
        system_instruction=system_instruction,
        chat_template_sha256=chat_template_sha256,
        prompts=prompts,
    )
    path = Path(cfg.cache_path) if cfg.cache_path else None
    if path is not None and path.is_file():
        try:
            cached = json.loads(path.read_text())
        except (OSError, ValueError):
            cached = {}
        if cached.get("identity") == identity and len(cached.get("texts", ())) == len(prompts):
            print(f"  calibration answers loaded from {path}")
            return list(cached["texts"]), {**cached["stats"], "cached": True}
        print(f"  calibration answer cache at {path} does not match this run; regenerating")
    print(f"  generating calibration answers with {ref} (max_new_tokens={cfg.max_new_tokens})")
    model = generator_factory(ref)
    try:
        answers, stats = generate_answers(
            prompts,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.batch_size,
            apply_chat_template=apply_chat_template,
            device=device,
        )
    finally:
        del model
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    texts = [p + a for p, a in zip(prompts, answers, strict=True)]
    print(
        f"  generated {stats['n']} answers, mean {stats['mean_response_tokens']:.0f} tokens, "
        f"{stats['n_hit_max_new_tokens']} hit max_new_tokens"
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({"identity": identity, "texts": texts, "stats": stats}, fh)
        os.replace(tmp, path)
    return texts, {**stats, "cached": False}


def truncated_count(tokenizer: Any, texts: Sequence[str], max_length: int, *, add_special_tokens: bool) -> int:
    ids = tokenizer(list(texts), truncation=False, add_special_tokens=add_special_tokens)["input_ids"]
    n = sum(1 for x in ids if len(x) > int(max_length))
    if n:
        warnings.warn(
            f"calibration_generated_answer: {n}/{len(ids)} prompt+answer texts exceed calibration_max_length="
            f"{int(max_length)} tokens and will be truncated (the end of the answer is lost); max_length was not "
            "changed.",
            RuntimeWarning,
            stacklevel=2,
        )
    return n
