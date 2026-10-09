"""Dataset-backed calibration text for LLM block extension and transport.

Block extension and the Theseus/BiCo transports fit their maps on activations
collected from a calibration corpus. The vision path has always drawn that
corpus from a real dataset (`build_vision_calibration_loader`, driven by
`block_extension_params.calibration_dataset` / `calibration_split`); 
this module is the LLM counterpart. Calibration text is resolved, in order, from:

1. explicit `calibration_prompts` in the config (escape hatch for a curated
   bank),
2. `block_extension_params.calibration_dataset` -- an HF dataset path or spec,
   mirroring the vision path,
3. the evaluated lm-harness task(s), rendered through the task's own
   `doc_to_text` so calibration sees exactly the prompt format eval scores.

For (3) the docs are split deterministically into a calibration slice and an
eval slice. The eval slice is handed back as explicit doc indices so the
harness can be told to score only those, keeping the two sets disjoint even
for a single-split benchmark like ifeval.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

_TEXT_COLUMN_CANDIDATES = (
    "text",
    "prompt",
    "question",
    "content",
    "sentence",
    "instruction",
    "input",
    "article",
    "document",
)


#: lm-eval's own default few-shot sampler seed (``DEFAULT_OTHER_SEED``; ``harness.run`` never overrides it).
_LM_EVAL_FEWSHOT_SEED = 1234


class CalibrationTextList(list):
    """``list[str]`` that remembers how its texts must be tokenized.

    Chat-template-rendered texts already carry their special tokens, so they must be tokenized with
    ``add_special_tokens=False``; every consumer that does ``build_text_calibration_loader(texts=...calibration().texts)``
    then picks this up without a signature change. Plain lists (and ``CalibrationTextList(add_special_tokens=True)``)
    keep the historical ``tokenizer(...)`` defaults.
    """

    def __init__(
        self,
        items: Sequence[str] = (),
        *,
        add_special_tokens: bool = True,
        content_spans: Sequence[Sequence[tuple[int, int]]] | None = None,
    ) -> None:
        super().__init__(items)
        self.add_special_tokens = bool(add_special_tokens)
        #: Per text, the character spans of its message contents (opt-in ``calibration_fit_mask``; None = not recorded).
        self.content_spans = None if content_spans is None else [[(int(a), int(b)) for a, b in sp] for sp in content_spans]
        if self.content_spans is not None and len(self.content_spans) != len(self):
            raise ValueError(f"{len(self.content_spans)} content span lists for {len(self)} calibration texts.")


def tokenizer_chat_template_sha256(tokenizer: Any) -> str:
    """sha256 of ``tokenizer.chat_template`` (a dict of named templates is hashed in canonical JSON form)."""
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise ValueError("Tokenizer has no chat_template; cannot apply a chat template to the calibration text.")
    if not isinstance(template, str):
        template = json.dumps(template, sort_keys=True)
    return hashlib.sha256(template.encode()).hexdigest()


def make_chat_template_fn(tokenizer: Any) -> Any:
    """Equivalent of lm-eval's ``HFLM.apply_chat_template`` (0.4.12) for a bare HF tokenizer.

    Same call (``tokenize=False``, ``continue_final_message=not add_generation_prompt``) and same fallback on a
    jinja ``TemplateError`` (retry without the system turn). ``HFLM.chat_template_args`` is empty unless
    ``enable_thinking`` is given, which this harness never does.
    """
    import jinja2

    def apply(chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        apply.last_messages = chat_history  # the messages actually rendered (after the system-turn fallback below)
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": not add_generation_prompt,
        }
        try:
            return tokenizer.apply_chat_template(chat_history, **kwargs)
        except jinja2.exceptions.TemplateError:
            chat_history = [m for m in chat_history if m["role"] != "system"]
            apply.last_messages = chat_history
            return tokenizer.apply_chat_template(chat_history, **kwargs)

    apply.last_messages = []
    return apply


def _render_user_text(chat_template_fn: Any, text: str, system_instruction: str | None) -> str:
    messages = ([{"role": "system", "content": system_instruction}] if system_instruction else []) + [
        {"role": "user", "content": text}
    ]
    return chat_template_fn(messages, add_generation_prompt=True)


def resolve_calibration_apply_chat_template(cfg: Mapping[str, Any]) -> bool:
    """Whether calibration text is chat-rendered: ``calibration_apply_chat_template`` when set (bool), else it follows
    ``harness_apply_chat_template`` (the historical coupling). Evaluation always follows ``harness_apply_chat_template``."""
    raw = cfg.get("calibration_apply_chat_template", None)
    if raw is None:
        return bool(cfg.get("harness_apply_chat_template", False))
    if not isinstance(raw, bool):
        raise ValueError(f"calibration_apply_chat_template must be a bool or null, got {raw!r}.")
    return raw


FIT_MASK_MODES = ("user_content",)


def resolve_calibration_fit_mask(cfg: Mapping[str, Any], *, chat_rendered: bool, method_is_ariadne: bool) -> str | None:
    """``calibration_fit_mask`` (null | "user_content"); every incompatible combination is a ``ValueError``."""
    mode = cfg.get("calibration_fit_mask", None)
    if mode is None:
        return None
    if mode not in FIT_MASK_MODES:
        raise ValueError(f"calibration_fit_mask must be null or one of {list(FIT_MASK_MODES)}, got {mode!r}.")
    if not chat_rendered:
        raise ValueError(
            "calibration_fit_mask needs chat-rendered calibration text (harness_apply_chat_template or "
            "calibration_apply_chat_template true): the mask separates message contents from template tokens."
        )
    if not method_is_ariadne:
        raise ValueError(
            "calibration_fit_mask is implemented for the ARIADNE fit only; THESEUS / BiCo / BRACE would silently "
            "ignore it, so it is refused."
        )
    return str(mode)


def locate_content_spans(rendered: str, messages: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Character spans of every non-system message content inside ``rendered``.

    The chat template inserts each content verbatim, in message order, so the spans are found by a left-to-right
    search with a cursor: each content is looked up after the end of the previous one (the system turn first, which
    only moves the cursor; it may be absent when the template rejected the system role). User AND assistant contents
    (few-shot multi-turn examples) are all spans: only template markers, role names, the system turn, template
    newlines and the generation header lie outside. Raises ``ValueError`` when a content is not found verbatim (a
    template that rewrites it, e.g. ``| trim``) and, for a single-message conversation, when it occurs more than once
    after the system turn (the locator would be ambiguous).
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    contents = [m for m in messages if str(m.get("role")) != "system"]
    for message in messages:
        content = str(message.get("content", ""))
        if not content:
            continue
        idx = rendered.find(content, cursor)
        if str(message.get("role")) == "system":
            if idx >= 0:
                cursor = idx + len(content)
            continue
        if idx < 0:
            raise ValueError(
                f"calibration_fit_mask: the {message.get('role')!r} message content {content[:60]!r} is not found "
                "verbatim in the rendered chat text (the chat template rewrites it)."
            )
        if len(contents) == 1 and rendered.find(content, idx + 1) >= 0:
            raise ValueError(
                f"calibration_fit_mask: the user content {content[:60]!r} occurs more than once in the rendered text "
                "after the system turn, so its span is ambiguous."
            )
        spans.append((idx, idx + len(content)))
        cursor = idx + len(content)
    return spans


class _RecordingChatFn:
    """Chat-template callable that remembers, per rendered string, the messages the template received."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.calls: dict[str, list[dict[str, Any]]] = {}

    def __call__(self, chat_history: Any, *args: Any, **kwargs: Any) -> str:
        out = self.fn(chat_history, *args, **kwargs)
        self.calls[out] = [dict(m) for m in getattr(self.fn, "last_messages", chat_history)]
        return out

    def spans(self, rendered: str) -> list[tuple[int, int]]:
        if rendered not in self.calls:
            raise ValueError("calibration_fit_mask: a calibration text was not produced by the chat template callable.")
        return locate_content_spans(rendered, self.calls[rendered])


def _content_token_masks(
    enc: Mapping[str, Any], spans: Sequence[Sequence[tuple[int, int]]]
) -> list[list[int]]:
    """Per sequence, 1 for every real token whose character offsets overlap a content span (token-boundary overlap
    counts: a token that straddles a content edge is kept), 0 for template tokens and padding."""
    out: list[list[int]] = []
    for offsets, attention, sp in zip(enc["offset_mapping"], enc["attention_mask"], spans, strict=True):
        out.append(
            [
                int(bool(att) and e > s and any(s < b and e > a for a, b in sp))
                for (s, e), att in zip(offsets, attention, strict=True)
            ]
        )
    return out


def fit_mask_provenance(
    tokenizer: Any, texts: Sequence[str], *, mode: str, max_length: int, tokenizer_name: str = "source"
) -> dict[str, Any]:
    """Run-summary record of ``calibration_fit_mask``: mode, total / kept real tokens and sha256 of the masks."""
    masks, attention = tokenize_fit_masks(tokenizer, texts, max_length=max_length)
    total = int(sum(sum(int(a) for a in row) for row in attention))
    kept = int(sum(sum(row) for row in masks))
    flat = bytes(v for row in masks for v in row)
    return {
        "calibration_fit_mask": mode,
        "n_tokens_total": total,
        "n_tokens_kept": kept,
        "kept_share": (kept / total) if total else 0.0,
        "mask_sha256": hashlib.sha256(flat).hexdigest(),
    }


def tokenize_fit_masks(tokenizer: Any, texts: Sequence[str], *, max_length: int) -> tuple[list[list[int]], Any]:
    """``(fit masks, attention masks)`` of ``texts`` tokenized exactly as ``build_text_calibration_loader`` does."""
    spans = getattr(texts, "content_spans", None)
    if spans is None:
        raise ValueError("calibration_fit_mask needs content spans recorded while the calibration text was rendered.")
    enc = _tokenize_for_loader(tokenizer, list(texts), max_length, getattr(texts, "add_special_tokens", True), True)
    return _content_token_masks(enc, spans), enc["attention_mask"]


def _tokenize_for_loader(tokenizer: Any, prompt_list: list[str], max_length: int, add_special_tokens: bool, offsets: bool):
    special_kwargs = {} if add_special_tokens else {"add_special_tokens": False}
    if offsets:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("calibration_fit_mask needs a fast tokenizer (character offsets); got a slow one.")
        special_kwargs["return_offsets_mapping"] = True
    return tokenizer(
        prompt_list, truncation=True, max_length=int(max_length), padding="max_length", **special_kwargs
    )


#: Probe conversations rendered under both templates when they differ (single turn, and multi-turn few-shot).
_TEMPLATE_PROBES = (
    [{"role": "user", "content": "Question: probe\nAnswer:"}],
    [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1 #### 1"},
        {"role": "user", "content": "Q2"},
    ],
)


def check_calibration_tokenizers(
    source_tokenizer: Any,
    target_tokenizer: Any,
    texts: Sequence[str],
    *,
    add_special_tokens: bool,
    system_instruction: str | None = None,
) -> dict[str, str]:
    """Source and target calibration batches are tokenized separately: require identical ids for every text.

    The texts are rendered once, with the source template. Different template strings are accepted only when
    they render the same prompts: Qwen2.5 base / Instruct / Math templates differ solely in the default system
    prompt they insert when none is given, so a mismatch requires an explicit ``system_instruction``, and both
    templates must render identical probe conversations with it. Returns the chat-template sha256 of each
    tokenizer; raises ``ValueError`` on a rendering or token-id mismatch.
    """
    sha = {
        "source": tokenizer_chat_template_sha256(source_tokenizer),
        "target": tokenizer_chat_template_sha256(target_tokenizer),
    }
    if sha["source"] != sha["target"]:
        if not system_instruction:
            raise ValueError(
                "Chat-template calibration: source and target tokenizers have different chat_template "
                f"(sha256 source={sha['source'][:12]}, target={sha['target'][:12]}) and no system instruction "
                "is set, so each template inserts its own default system prompt; set harness_system_instruction."
            )
        src_fn, tgt_fn = make_chat_template_fn(source_tokenizer), make_chat_template_fn(target_tokenizer)
        for probe in _TEMPLATE_PROBES:
            conversation = [{"role": "system", "content": system_instruction}, *probe]
            if src_fn(conversation) != tgt_fn(conversation):
                raise ValueError(
                    "Chat-template calibration: source and target chat templates render the same conversation "
                    f"differently (sha256 source={sha['source'][:12]}, target={sha['target'][:12]}); the two "
                    "models would be calibrated and evaluated on different prompts."
                )
    kwargs = {"add_special_tokens": bool(add_special_tokens), "truncation": False}
    src_ids = source_tokenizer(list(texts), **kwargs)["input_ids"]
    tgt_ids = target_tokenizer(list(texts), **kwargs)["input_ids"]
    bad = [i for i, (a, b) in enumerate(zip(src_ids, tgt_ids, strict=True)) if list(a) != list(b)]
    if bad:
        raise ValueError(
            f"Chat-template calibration: source and target tokenizers disagree on input_ids for {len(bad)}/"
            f"{len(src_ids)} calibration texts (first indices {bad[:5]}); paired calibration requires identical tokens."
        )
    return sha


class CalibrationTexts:
    """Resolved calibration corpus plus the eval doc indices it leaves free.

    `eval_samples` maps an lm-harness task name to the doc indices that were
    NOT used for calibration. It is empty whenever calibration came from a
    source independent of the evaluated task, in which case the harness should
    score the task in full.
    """

    def __init__(
        self,
        texts: list[str],
        *,
        source: str,
        eval_samples: dict[str, list[int]] | None = None,
        notes: list[str] | None = None,
        decoupled: bool = False,
        chat_template: Mapping[str, Any] | None = None,
        content_spans: Sequence[Sequence[tuple[int, int]]] | None = None,
    ) -> None:
        if not texts:
            raise ValueError(f"Calibration source {source!r} produced no text.")
        if content_spans is not None and chat_template is None:
            raise ValueError("content spans exist only for chat-rendered calibration text.")
        #: Chat-template rendering record (None = off, historical plain text). ``sha256`` is filled by
        #: ``check_calibration_tokenizers`` once the source/target tokenizers have been compared.
        self.chat_template = dict(chat_template) if chat_template is not None else None
        if self.chat_template is not None:
            texts = CalibrationTextList(texts, add_special_tokens=False, content_spans=content_spans)
        self.texts = texts
        self.source = source
        self.eval_samples = dict(eval_samples or {})
        #: Recorded (not raised) deviations, e.g. a harness task without a gold target under ``include_target``.
        self.notes = list(notes or [])
        #: True when the corpus was chosen independently of the evaluated task (so no eval hold-out exists).
        self.decoupled = bool(decoupled)
        #: Generated-answer calibration record (None = off); set by ``TextCalibrationCache`` after it replaced the texts.
        self.generated_answer: dict[str, Any] | None = None

    def provenance(self) -> dict[str, Any]:
        """Additive run-summary record: corpus fingerprint, size, hold-out hash and any recorded deviations."""
        holdout = json.dumps({k: list(v) for k, v in sorted(self.eval_samples.items())}, sort_keys=True)
        record = {
            "source": self.source,
            "n_sequences": len(self.texts),
            "corpus_sha256": hashlib.sha256("\x00".join(self.texts).encode()).hexdigest(),
            "holdout_sha256": hashlib.sha256(holdout.encode()).hexdigest() if self.eval_samples else None,
            "holdout_sizes": {k: len(v) for k, v in sorted(self.eval_samples.items())},
            "decoupled_from_eval": self.decoupled,
            "notes": list(self.notes),
            "chat_template_applied": self.chat_template is not None,
            "chat_template_sha256": (self.chat_template or {}).get("sha256"),
            "chat_template": {k: v for k, v in (self.chat_template or {}).items() if k != "sha256"} or None,
            "rendered_texts_sha256": hashlib.sha256(json.dumps(list(self.texts)).encode()).hexdigest(),
        }
        if self.generated_answer is not None:  # only present when the opt-in feature ran
            record["generated_answer"] = dict(self.generated_answer)
        return record

    def __len__(self) -> int:
        return len(self.texts)

    def describe(self) -> str:
        held_out = ", ".join(
            f"{task}: {len(idx)} eval docs" for task, idx in sorted(self.eval_samples.items())
        )
        suffix = f" (held out -> {held_out})" if held_out else ""
        return f"{len(self.texts)} sequences from {self.source}{suffix}"


def resolve_calibration_texts(
    *,
    prompts: Sequence[str] | None = None,
    calibration_dataset: str | Mapping[str, Any] | None = None,
    calibration_split: str = "val",
    harness_tasks: Sequence[str] | None = None,
    n_sequences: int,
    seed: int = 0,
    include_target: bool = False,
    apply_chat_template: bool = False,
    system_instruction: str | None = None,
    tokenizer: Any = None,
    num_fewshot: int | Sequence[int] | Mapping[str, int] = 0,
    fewshot_seed: int = _LM_EVAL_FEWSHOT_SEED,
    record_content_spans: bool = False,
) -> CalibrationTexts:
    """Resolve the calibration corpus for an LLM run.

    Parameters
    ----------
    prompts : Explicit prompt bank from `config['calibration_prompts']`.
    calibration_dataset : HF dataset path, or a spec mapping with `path` and
        optional `name`/`split`/`text_column`.
    calibration_split : Split of a ``calibration_dataset`` to calibrate on. An
        lm-harness source ignores it: its calibration docs are always held out
        of the evaluation (``eval_samples``), since the harness scores the same
        task the text is drawn from.
    harness_tasks : Evaluated lm-harness tasks, used as the default source.
    n_sequences : How many sequences the run will actually consume
        (`n_batches * batch_size`). The calibration slice is sized to cover
        this where the source allows it.
    seed : Seed for the deterministic calibration/eval partition.
    include_target : Harness source only (default off): append each doc's gold target (``doc_to_target``; an int
        index is mapped through ``doc_to_choice``) to its rendered prompt. Tasks without a gold target fall back to
        the prompt alone and the fallback is recorded in ``CalibrationTexts.notes``.
    apply_chat_template : Opt-in (default off = byte-identical plain text). Render calibration text the way the
        chat-template evaluation does, so the activations see the evaluation's prompt format. Harness source: each
        calibration doc goes through ``task.fewshot_context(doc, num_fewshot, system_instruction,
        apply_chat_template=True, fewshot_as_multiturn=num_fewshot > 0, chat_template=...)`` (lm-eval's own path;
        same hold-out indices as with the flag off). Prompts / HF dataset: one user turn (after the optional system
        turn) with the generation prompt. Needs ``tokenizer`` (the source tokenizer); tokenize with
        ``add_special_tokens=False`` (``CalibrationTexts.texts`` carries that).
    system_instruction : System turn used with ``apply_chat_template`` (lm-eval's ``system_instruction``).
    num_fewshot : Harness source with ``apply_chat_template`` only; same shapes as ``harness_num_fewshot``.
    fewshot_seed : Seed of lm-eval's few-shot sampler for the calibration docs (default = lm-eval's evaluation seed).
    record_content_spans : Opt-in (needs ``apply_chat_template``; ``calibration_fit_mask``): record, per rendered text,
        the character spans of its message contents (``CalibrationTextList.content_spans``). Texts are unchanged.
    """
    if n_sequences <= 0:
        raise ValueError("n_sequences must be > 0.")
    chat_fn = None
    if record_content_spans and not apply_chat_template:
        raise ValueError("record_content_spans needs apply_chat_template (content spans exist only in chat-rendered text).")
    if apply_chat_template:
        if include_target:
            raise NotImplementedError("calibration_include_target is not supported with harness_apply_chat_template.")
        if tokenizer is None:
            raise ValueError("apply_chat_template calibration needs the source tokenizer (tokenizer=...).")
        chat_fn = make_chat_template_fn(tokenizer)
        if record_content_spans:
            chat_fn = _RecordingChatFn(chat_fn)
        # Fails early (clear message) when the tokenizer has no template.
        tokenizer_chat_template_sha256(tokenizer)

    if prompts:
        bad = [i for i, p in enumerate(prompts) if p is None or not str(p).strip()]
        if bad:
            raise ValueError(f"config['calibration_prompts'] has empty entries at indices {bad[:10]}.")
        return _finalize_rendered(
            CalibrationTexts([str(p) for p in prompts], source="config['calibration_prompts']", decoupled=True),
            chat_fn,
            system_instruction,
            record_content_spans,
        )

    if calibration_dataset is not None:
        return _finalize_rendered(
            _from_hf_dataset(
                calibration_dataset,
                calibration_split=calibration_split,
                n_sequences=n_sequences,
            ),
            chat_fn,
            system_instruction,
            record_content_spans,
        )

    if harness_tasks:
        return _from_harness_tasks(
            list(harness_tasks),
            calibration_split=calibration_split,
            n_sequences=n_sequences,
            seed=seed,
            include_target=include_target,
            chat_template_fn=chat_fn,
            system_instruction=system_instruction,
            num_fewshot=num_fewshot,
            fewshot_seed=fewshot_seed,
            record_content_spans=record_content_spans,
        )

    raise ValueError(
        "No calibration source configured. Set one of: "
        "config['calibration_prompts'], "
        "config['block_extension_params']['calibration_dataset'], "
        "or config['harness_tasks']."
    )


def _finalize_rendered(
    resolved: CalibrationTexts, chat_fn: Any, system_instruction: str | None, record_content_spans: bool = False
) -> CalibrationTexts:
    """Chat-template flag on: one rendered user turn per raw text (flag off: ``resolved`` unchanged)."""
    if chat_fn is None:
        return resolved
    rendered = [_render_user_text(chat_fn, t, system_instruction) for t in resolved.texts]
    return CalibrationTexts(
        rendered,
        content_spans=[chat_fn.spans(r) for r in rendered] if record_content_spans else None,
        source=resolved.source,
        eval_samples=resolved.eval_samples,
        notes=resolved.notes,
        decoupled=resolved.decoupled,
        chat_template={"system_instruction": system_instruction, "sha256": None, "num_fewshot": 0},
    )


def _from_hf_dataset(
    spec: str | Mapping[str, Any],
    *,
    calibration_split: str,
    n_sequences: int,
) -> CalibrationTexts:
    import datasets

    if isinstance(spec, str):
        spec = {"path": spec}
    if not isinstance(spec, Mapping):
        raise TypeError("calibration_dataset must be a dataset path or a mapping spec.")

    path = spec.get("path", spec.get("dataset", None))
    if not path:
        raise ValueError("calibration_dataset spec needs a 'path'.")
    name = spec.get("name", spec.get("config", None))
    split = str(spec.get("split", calibration_split))
    text_column = spec.get("text_column", None)
    text_columns = spec.get("text_columns", None)
    text_template = spec.get("text_template", None)
    if text_columns is not None and text_column is not None:
        raise ValueError("calibration_dataset: set only one of 'text_column' and 'text_columns'.")

    ds = (
        datasets.load_dataset(str(path), str(name), split=split)
        if name
        else datasets.load_dataset(str(path), split=split)
    )

    if text_columns is not None:
        # Opt-in multi-column rendering (e.g. question + answer). Default (single text column) is unchanged.
        columns = [str(c) for c in text_columns]
        missing = [c for c in columns if c not in ds.column_names]
        if not columns or missing:
            raise ValueError(f"calibration_dataset text_columns {columns} not all in {ds.column_names}.")
        template = str(text_template) if text_template is not None else "\n".join("{" + c + "}" for c in columns)
        texts = []
        for row in ds:
            values = {c: row.get(c, None) for c in columns}
            if all(isinstance(v, str) and v.strip() for v in values.values()):
                texts.append(template.format(**values))
            if len(texts) >= n_sequences:
                break
        return CalibrationTexts(texts, source=f"{path}[{split}].{'+'.join(columns)}", decoupled=True)

    column = text_column or _pick_text_column(ds.column_names)
    if column is None:
        raise ValueError(
            f"Could not infer a text column for calibration_dataset {path!r} "
            f"(columns: {ds.column_names}). Set 'text_column' explicitly."
        )

    texts: list[str] = []
    for row in ds:
        value = row.get(column, None)
        if isinstance(value, str) and value.strip():
            texts.append(value)
        if len(texts) >= n_sequences:
            break

    return CalibrationTexts(texts, source=f"{path}[{split}].{column}", decoupled=True)


def _pick_text_column(columns: Sequence[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in _TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _from_harness_tasks(
    tasks: list[str],
    *,
    calibration_split: str,
    n_sequences: int,
    seed: int,
    include_target: bool = False,
    chat_template_fn: Any = None,
    system_instruction: str | None = None,
    num_fewshot: int | Sequence[int] | Mapping[str, int] = 0,
    fewshot_seed: int = _LM_EVAL_FEWSHOT_SEED,
    record_content_spans: bool = False,
) -> CalibrationTexts:
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    # Always hold out: the calibration docs come from the very task the harness then scores, so they never reach
    # the evaluation (``calibration_split`` used to switch this off for any split but "val"/"validation"/"dev").
    del calibration_split

    texts: list[str] = []
    content_spans: list[list[tuple[int, int]]] = []
    notes: list[str] = []
    eval_samples: dict[str, list[int]] = {}
    fewshot_by_task: dict[str, int] = {}
    if chat_template_fn is not None:
        from ..eval.llm_rebase.harness import _resolve_fewshot_by_task

        fewshot_by_task = _resolve_fewshot_by_task(list(tasks), num_fewshot)
    per_task = max(1, -(-n_sequences // max(1, len(tasks))))

    for task_name in tasks:
        loaded = manager.load_task_or_group([task_name])
        task = loaded.get(task_name, None)
        if task is None or not hasattr(task, "doc_to_text"):
            # A group expands to several subtasks; calibrating on a group's
            # aggregate is ambiguous, so skip rather than guess.
            continue

        docs = list(_task_docs(task))
        if not docs:
            continue

        order = list(range(len(docs)))
        random.Random(seed).shuffle(order)

        n_calib = min(per_task, max(1, len(order) - 1))
        calib_idx = sorted(order[:n_calib])
        eval_samples[task_name] = sorted(order[n_calib:])

        n_without_target = 0
        if chat_template_fn is not None:
            # lm-eval's own rendering (what build_all_requests sends with apply_chat_template=True). Its few-shot
            # sampler is one stateful Random(seed) consumed in doc order (and lm-eval's
            # sampler excludes the eval doc when fewshot split == test split), so reseed per task and walk the
            # calibration docs in sorted order: deterministic, though not the shots the eval docs receive.
            n_shot = fewshot_by_task[task_name]
            if hasattr(task, "set_fewshot_seed"):
                task.set_fewshot_seed(seed=int(fewshot_seed))
            for i in calib_idx:
                ctx = task.fewshot_context(
                    docs[i],
                    num_fewshot=n_shot,
                    system_instruction=system_instruction,
                    apply_chat_template=True,
                    fewshot_as_multiturn=n_shot > 0,
                    chat_template=chat_template_fn,
                )
                # multiple-input tasks (e.g. winogrande) return one context per choice
                for text in ctx if isinstance(ctx, list) else [ctx]:
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
                        if record_content_spans:
                            content_spans.append(chat_template_fn.spans(text))
            continue
        for i in calib_idx:
            rendered = task.doc_to_text(docs[i])
            if isinstance(rendered, str) and rendered.strip():
                if include_target:
                    target = _gold_target_text(task, docs[i])
                    if target is None:
                        n_without_target += 1
                    else:
                        rendered = f"{rendered}{getattr(getattr(task, 'config', None), 'target_delimiter', ' ')}{target}"
                texts.append(rendered)
        if include_target and n_without_target:
            notes.append(
                f"include_target: {n_without_target}/{len(calib_idx)} docs of {task_name} have no gold target; "
                "their prompt was used alone."
            )

    if not texts:
        raise ValueError(
            f"lm-harness tasks {tasks} yielded no calibration text. "
            "Point block_extension_params.calibration_dataset at a dataset instead."
        )

    return CalibrationTexts(
        texts,
        source=f"lm-harness {'+'.join(tasks)}[holdout]",
        eval_samples=eval_samples,
        notes=notes,
        content_spans=content_spans if record_content_spans else None,
        chat_template=(
            {
                "system_instruction": system_instruction,
                "sha256": None,
                "num_fewshot": dict(fewshot_by_task),
                "fewshot_as_multiturn": any(n > 0 for n in fewshot_by_task.values()),
                "fewshot_seed": int(fewshot_seed),
            }
            if chat_template_fn is not None
            else None
        ),
    )


def _gold_target_text(task: Any, doc: Any) -> str | None:
    """The doc's gold target as text, or ``None`` when the task has none (e.g. generative tasks like ifeval)."""
    try:
        target = task.doc_to_target(doc)
    except Exception:  # noqa: BLE001 - tasks without a target raise task-specific errors
        return None
    if isinstance(target, bool):
        return None
    if isinstance(target, int):
        try:
            choices = task.doc_to_choice(doc)
            return str(choices[target])
        except Exception:  # noqa: BLE001
            return None
    if isinstance(target, str) and target.strip():
        return target
    return None


def _task_docs(task: Any) -> Sequence[Any]:
    """Return exactly the docs lm-eval would score, in lm-eval's own order.

    The indices handed back as `eval_samples` are resolved by lm-eval against
    `Task.eval_docs` (`build_all_requests` filters `enumerate(self.eval_docs)`),
    so the calibration slice has to be carved from that same sequence. Reading
    `eval_docs` rather than reimplementing its test/validation preference keeps
    the two in lockstep if lm-eval ever changes that preference.
    """
    try:
        return list(task.eval_docs)
    except (AttributeError, ValueError, KeyError):
        # Task has neither test nor validation docs; lm-eval could not score it
        # either, so there is no index space to stay disjoint from.
        return []


class TokenizedPromptDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]], sample_ids: list[str] | None = None) -> None:
        self.features = features
        # Identity of the underlying examples, not of this tokenization. The
        # source and target calibration loaders tokenize the SAME texts with
        # different tokenizers, so they are different objects holding different
        # token ids; paired calibration has to recognise them as the same
        # examples replayed under two preprocessors, which is exactly what it
        # falls back to sample_ids for.
        self.sample_ids = list(sample_ids) if sample_ids is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        feat = dict(self.features[int(idx)])
        feat["labels"] = list(feat["input_ids"])
        return feat


def build_text_calibration_loader(
    *,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 2,
    max_length: int = 128,
    add_special_tokens: bool | None = None,
    fit_mask: str | None = None,
) -> DataLoader:
    # None: follow the corpus (``CalibrationTextList``: False for chat-template-rendered text, else True).
    if add_special_tokens is None:
        add_special_tokens = getattr(texts, "add_special_tokens", True)
    prompt_list = list(texts)
    if not prompt_list:
        raise ValueError("Calibration loader needs at least one text sequence.")
    # Default (True) keeps the historical call (tokenizer's own default add_special_tokens) untouched.
    if fit_mask is not None:
        # Opt-in ``calibration_fit_mask``: one extra per-token ``fit_mask`` (1 = inside a message content).
        if fit_mask not in FIT_MASK_MODES:
            raise ValueError(f"fit_mask must be one of {list(FIT_MASK_MODES)} or None, got {fit_mask!r}.")
        spans = getattr(texts, "content_spans", None)
        if spans is None:
            raise ValueError("fit_mask needs chat-rendered texts with recorded content spans (record_content_spans).")
        enc = _tokenize_for_loader(tokenizer, prompt_list, max_length, add_special_tokens, True)
        fit_masks = _content_token_masks(enc, spans)
        enc.pop("offset_mapping")
    else:
        fit_masks = None
        special_kwargs = {} if add_special_tokens else {"add_special_tokens": False}
        enc = tokenizer(
            prompt_list,
            truncation=True,
            max_length=int(max_length),
            padding="max_length",
            **special_kwargs,
        )
    features: list[dict[str, Any]] = []
    for i in range(len(prompt_list)):
        feat = {k: v[i] for k, v in enc.items()}
        if fit_masks is not None:
            feat["fit_mask"] = fit_masks[i]
        features.append(feat)

    # Stable across processes: str.__hash__ is salted per interpreter, which
    # would make these ids non-reproducible if they were ever persisted.
    sample_ids = [f"{i}:{hashlib.sha1(t.encode()).hexdigest()[:12]}" for i, t in enumerate(prompt_list)]
    dataset = TokenizedPromptDataset(features, sample_ids=sample_ids)

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        feats = [{k: v for k, v in row.items() if k not in ("labels", "fit_mask")} for row in batch]
        padded = tokenizer.pad(feats, return_tensors="pt", padding="max_length", max_length=int(max_length))
        padded["labels"] = padded["input_ids"].clone()
        if "attention_mask" in padded:
            padded["labels"][padded["attention_mask"] == 0] = -100
        if fit_masks is not None:  # same length as input_ids (tokenized to max_length), padding already 0
            padded["fit_mask"] = torch.tensor([row["fit_mask"] for row in batch], dtype=torch.bool)
        return padded

    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
    )


def tokenization_stats(tokenizer: Any, texts: Sequence[str], max_length: int) -> dict[str, Any]:
    """Token-level provenance of a calibration corpus under ``max_length`` padding (summary record)."""
    add_special_tokens = getattr(texts, "add_special_tokens", True)
    lengths = [
        len(ids) for ids in tokenizer(list(texts), truncation=False, add_special_tokens=add_special_tokens)["input_ids"]
    ]
    n = len(lengths)
    kept = [min(length, int(max_length)) for length in lengths]
    total_slots = n * int(max_length)
    return {
        "n_sequences": n,
        "max_length": int(max_length),
        "mean_length": (sum(kept) / n) if n else 0.0,
        "n_truncated": sum(1 for length in lengths if length > int(max_length)),
        "pad_fraction": (1.0 - sum(kept) / total_slots) if total_slots else 0.0,
    }
