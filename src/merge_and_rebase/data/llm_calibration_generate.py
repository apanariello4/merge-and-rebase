"""Generated-answer calibration: each calibration text becomes ``prompt + the source fine-tune's own greedy answer``.

The effect of an instruct / reasoning fine-tune shows up while the model *writes* the answer; a prompt-only corpus
reaches those positions only through the last prompt token. Teacher-forcing the fine-tune's own answer puts response
positions in the corpus. Opt-in (config ``calibration_generated_answer``); with the key absent nothing here runs.

Text contract (exact): ``text = prompt + separator + answer`` with ``answer = tokenizer.decode(answer_ids,
skip_special_tokens=True)`` and ``separator`` = ``""`` unless configured, where ``prompt`` is
the string the corpus held before (with ``harness_apply_chat_template`` the already rendered chat prompt, generation
prompt included, so the answer follows it exactly as evaluation would see it) and ``answer_ids`` are the greedy tokens
up to, excluding, the first stop token (EOS; with a chat template also its end-of-turn token). The string is re-tokenized later by the calibration loader, so the ids that are teacher-forced may differ from
the generated ids at the prompt/answer seam or for tokenizers that do not round-trip decode(encode(.)); this is
deliberate (source and target tokenize the same string) and is the reference behaviour.

Opt-in extensions (all default to the behaviour above; sub-keys of ``calibration_generated_answer``):

* ``generation_chat_template`` (+ ``generation_system_instruction``): the generator is prompted with the chat-template
  rendering of the prompt (generation prompt added, no extra special tokens, stops at EOS / end-of-turn) while the
  calibration text stays the PLAIN ``prompt + separator + answer``. Needs ``harness_apply_chat_template`` false (with it
  on the corpus is already chat-rendered and the coupled mode above applies).
* ``separator``: inserted between prompt and answer. Composition-time option: not part of the cache identity.
* ``shuffle_answers_seed``: CONTROL, pairs every prompt with another prompt's answer (seeded derangement, n >= 2).
  Composition-time option: not part of the cache identity.
* ``include_prompt`` (default true): false makes each calibration text the answer alone (``text = answer``); the prompt
  is still what the answer is generated from. Composition-time option: not part of the cache identity. An old-format
  cache (texts only) is reused by recovering each answer as ``text[len(prompt):]`` (its texts are ``prompt + answer``).

The cache keeps the raw ``answers`` next to ``texts``, so separator / shuffle cells reuse one generation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Defaults of the reference sweep config (``align_with_gen_response``: max_new_tokens 256, batch_size 16).
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 16
_KEYS = frozenset(
    {
        "enabled",
        "max_new_tokens",
        "batch_size",
        "model",
        "cache_path",
        "generation_chat_template",
        "generation_system_instruction",
        "separator",
        "shuffle_answers_seed",
        "include_prompt",
    }
)


@dataclass(frozen=True)
class GeneratedAnswerConfig:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    batch_size: int = DEFAULT_BATCH_SIZE
    model: str | None = None
    cache_path: str | None = None
    generation_chat_template: bool = False
    generation_system_instruction: str | None = None
    separator: str = ""
    shuffle_answers_seed: int | None = None
    include_prompt: bool = True

    @property
    def uses_extensions(self) -> bool:
        """Any non-default new option (gates the additive provenance keys; defaults stay byte-identical)."""
        return (
            self.generation_chat_template
            or self.generation_system_instruction is not None
            or self.separator != ""
            or self.shuffle_answers_seed is not None
            or not self.include_prompt
        )


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
        generation_chat_template=bool(raw.get("generation_chat_template", False)),
        generation_system_instruction=(
            None if raw.get("generation_system_instruction") is None else str(raw["generation_system_instruction"])
        ),
        separator=str(raw.get("separator", "")),
        shuffle_answers_seed=None if raw.get("shuffle_answers_seed") is None else int(raw["shuffle_answers_seed"]),
        include_prompt=bool(raw.get("include_prompt", True)),
    )
    if not cfg.include_prompt and cfg.separator != "":
        raise ValueError("calibration_generated_answer: separator has no meaning with include_prompt=false.")
    if cfg.generation_system_instruction is not None and not cfg.generation_chat_template:
        raise ValueError("calibration_generated_answer: generation_system_instruction needs generation_chat_template.")
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
    identity = {
        "model": ref,
        "dtype": dtype,
        "max_new_tokens": cfg.max_new_tokens,
        "apply_chat_template": bool(apply_chat_template),
        "system_instruction": system_instruction if apply_chat_template else None,
        "chat_template_sha256": chat_template_sha256,
        "prompts_sha1": hashlib.sha1("\x00".join(prompts).encode()).hexdigest(),
    }
    if cfg.generation_chat_template:  # added only in the new mode: existing caches keep their identity
        identity["generation_chat_template"] = True
        identity["generation_system_instruction"] = cfg.generation_system_instruction
    return identity


def derange(answers: Sequence[str], seed: int) -> list[str]:
    """Answers re-paired so no prompt keeps its own: a seeded random cycle through all indices (n >= 2)."""
    n = len(answers)
    if n < 2:
        raise ValueError("calibration_generated_answer: shuffle_answers_seed needs at least 2 calibration texts.")
    order = list(range(n))
    random.Random(int(seed)).shuffle(order)
    out = [""] * n
    for k, i in enumerate(order):
        out[i] = answers[order[(k + 1) % n]]
    return out


def _compose(prompts: Sequence[str], answers: Sequence[str], cfg: GeneratedAnswerConfig) -> list[str]:
    if cfg.shuffle_answers_seed is not None:
        answers = derange(answers, cfg.shuffle_answers_seed)
    if not cfg.include_prompt:
        empty = sum(1 for a in answers if a == "")
        if empty:
            raise ValueError(f"calibration_generated_answer: include_prompt=false with {empty} empty answers.")
        return list(answers)
    return [p + cfg.separator + a for p, a in zip(prompts, answers, strict=True)]


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
    """``prompt + separator + answer`` for every prompt (cache hit: no model is built) and the generation record
    (carries the raw ``answers`` under ``"answers"``)."""
    if cfg.generation_chat_template and apply_chat_template:
        raise ValueError(
            "calibration_generated_answer.generation_chat_template cannot be combined with harness_apply_chat_template: "
            "the corpus is already chat-rendered (the answer is generated from that rendering); use the existing "
            "coupled mode by dropping generation_chat_template."
        )
    if cfg.shuffle_answers_seed is not None and len(prompts) < 2:
        raise ValueError("calibration_generated_answer: shuffle_answers_seed needs at least 2 calibration texts.")
    needs_answers = cfg.separator != "" or cfg.shuffle_answers_seed is not None or not cfg.include_prompt
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
        answers_c = cached.get("answers")
        if cached.get("identity") == identity and len(cached.get("texts", ())) == len(prompts):
            if answers_c is not None and len(answers_c) == len(prompts):
                print(f"  calibration answers loaded from {path}")
                return _compose(prompts, answers_c, cfg), {**cached["stats"], "cached": True, "answers": answers_c}
            if not needs_answers:  # old-format file: its texts are exactly prompt + answer
                print(f"  calibration answers loaded from {path}")
                return list(cached["texts"]), {**cached["stats"], "cached": True, "answers": None}
            texts_c = list(cached["texts"])
            if not cfg.include_prompt and all(t.startswith(p) for p, t in zip(prompts, texts_c, strict=True)):
                answers_c = [t[len(p) :] for p, t in zip(prompts, texts_c, strict=True)]
                print(f"  calibration answers recovered from the old-format cache {path}")
                return _compose(prompts, answers_c, cfg), {**cached["stats"], "cached": True, "answers": answers_c}
            print(f"  calibration answer cache at {path} has no raw answers; regenerating")
        else:
            print(f"  calibration answer cache at {path} does not match this run; regenerating")
    print(f"  generating calibration answers with {ref} (max_new_tokens={cfg.max_new_tokens})")
    gen_prompts = list(prompts)
    if cfg.generation_chat_template:
        from .llm_calibration import make_chat_template_fn

        render = make_chat_template_fn(tokenizer)
        system = cfg.generation_system_instruction
        gen_prompts = [
            render(([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}])
            for p in prompts
        ]
    model = generator_factory(ref)
    try:
        answers, stats = generate_answers(
            gen_prompts,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.batch_size,
            apply_chat_template=apply_chat_template or cfg.generation_chat_template,
            device=device,
        )
    finally:
        del model
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        f"  generated {stats['n']} answers, mean {stats['mean_response_tokens']:.0f} tokens, "
        f"{stats['n_hit_max_new_tokens']} hit max_new_tokens"
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(
                {
                    "identity": identity,
                    "texts": [p + a for p, a in zip(prompts, answers, strict=True)],
                    "stats": stats,
                    "answers": answers,
                },
                fh,
            )
        os.replace(tmp, path)
    return _compose(prompts, answers, cfg), {**stats, "cached": False, "answers": answers}


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
