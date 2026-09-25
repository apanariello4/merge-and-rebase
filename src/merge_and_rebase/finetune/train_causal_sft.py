"""Supervised fine-tuning of a causal LM on prompt/answer pairs, to build a task vector.

Why this exists
---------------
Proposal 1 needs a source pair ``(base, base + tau)`` where ``tau`` is a genuine
task vector: small relative to the weights (projection delta ratio roughly
0.01-0.15, see ``scripts/task_vector_check.py``) and worth something on its own
model. No math pair on disk satisfies both. ``Qwen2.5-1.5B -> Qwen2.5-Math-1.5B``
is continued pretraining (delta ~1.5x the weights), and ``Math-1.5B ->
Math-1.5B-Instruct`` transports instruction tuning, which *costs* 2.4 pp on
MATH-500. This script manufactures the missing pair: a short, full-parameter SFT
of a *base* model on one task's training split, so the delta is SFT-sized by
construction and its value is measurable.

``finetune/train_text.py`` cannot do this: it trains classification heads.

What it does
------------
- Renders every row as ``prompt_template`` + ``target_template`` and trains the
  next-token loss **on the target tokens only** (prompt tokens are masked to
  -100), followed by EOS so the model learns to stop.
- The default templates are lm-eval's own ``gsm8k`` rendering
  (``doc_to_text: "Question: {{question}}\\nAnswer:"``, target delimiter ``" "``,
  ``doc_to_target: "{{answer}}"``), so the fine-tune is scored in exactly the
  format it was trained in -- the instruct-format confound that sank the
  earlier ``-Instruct`` arms cannot arise.
- Holds out the last ``--holdout`` rows of the (seed-shuffled) split for a
  validation loss; the held-out row indices are written to the metadata.
- Trains with **fp32 master weights** (bf16 autocast for speed) and saves the
  checkpoint in **fp32**. This matters: at SFT learning rates most per-element
  updates are below bf16's ~0.4% relative resolution, so a bf16 checkpoint
  would round a large part of the task vector away before anyone measured it.
- Writes ``sft_meta.json`` next to the weights: every hyperparameter, the data
  fingerprint, the held-out indices, and the loss trajectory.

Usage
-----
::

    python -m merge_and_rebase.finetune.train_causal_sft \\
        --model Qwen/Qwen2.5-0.5B \\
        --dataset openai/gsm8k --dataset-config main --split train \\
        --out /path/to/qwen2.5-0.5b-gsm8k-sft \\
        --epochs 3 --lr 1e-5 --batch-size 8 --grad-accum 4 --max-length 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import string
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

GSM8K_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
GSM8K_TARGET_TEMPLATE = " {answer}"

IGNORE_INDEX = -100


def _template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def render_example(
    row: Mapping[str, Any], *, prompt_template: str, target_template: str
) -> tuple[str, str]:
    """Return ``(prompt, target)`` for one dataset row."""
    fields = _template_fields(prompt_template) | _template_fields(target_template)
    missing = sorted(fields - set(row))
    if missing:
        raise KeyError(f"row is missing template fields {missing}; has {sorted(row)}")
    values = {k: row[k] for k in fields}
    return prompt_template.format(**values), target_template.format(**values)


def tokenize_example(
    tokenizer: Any, prompt: str, target: str, *, max_length: int
) -> dict[str, list[int]] | None:
    """Token ids with the loss restricted to ``target`` + EOS.

    Prompt and target are tokenized separately and concatenated, so the
    boundary is exact rather than recovered from offsets. Returns None when the
    prompt alone fills ``max_length`` (nothing left to learn from); a long
    target is truncated, and EOS is then dropped with it.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) >= max_length:
        return None
    ids = (prompt_ids + target_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + target_ids)[:max_length]
    return {"input_ids": ids, "labels": labels}


def collate(rows: Sequence[Mapping[str, list[int]]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch; pad positions are masked from attention and loss."""
    width = max(len(r["input_ids"]) for r in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        input_ids[i, :n] = torch.tensor(r["input_ids"])
        labels[i, :n] = torch.tensor(r["labels"])
        attention[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def split_indices(n: int, *, holdout: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic seed-shuffled ``(train, holdout)`` row indices."""
    if not 0 <= holdout < n:
        raise ValueError(f"holdout must be in [0, {n}), got {holdout}")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return sorted(order[holdout:]), sorted(order[:holdout])


def _lr_at(step: int, *, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def _eval_loss(model, batches, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = batch["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum",
        )
        total += float(loss)
        count += int((shift_labels != IGNORE_INDEX).sum())
    model.train()
    return total / max(1, count)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--dataset-config", default="main")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prompt-template", default=GSM8K_PROMPT_TEMPLATE)
    ap.add_argument("--target-template", default=GSM8K_TARGET_TEMPLATE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args(argv)

    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 master weights: see the module docstring on bf16 rounding.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.config.use_cache = False
    model.train()

    ds = datasets.load_dataset(args.dataset, args.dataset_config, split=args.split)
    train_idx, holdout_idx = split_indices(len(ds), holdout=args.holdout, seed=args.seed)

    def encode(indices):
        rows, dropped = [], 0
        for i in indices:
            prompt, target = render_example(
                ds[i], prompt_template=args.prompt_template, target_template=args.target_template
            )
            enc = tokenize_example(tokenizer, prompt, target, max_length=args.max_length)
            if enc is None:
                dropped += 1
            else:
                rows.append(enc)
        return rows, dropped

    train_rows, train_dropped = encode(train_idx)
    holdout_rows, _ = encode(holdout_idx)
    truncated = sum(1 for r in train_rows if r["labels"][-1] != tokenizer.eos_token_id)
    fingerprint = hashlib.sha1(
        json.dumps([ds[i] for i in train_idx[:50]], sort_keys=True, default=str).encode()
    ).hexdigest()
    print(
        f"train rows {len(train_rows)} (dropped {train_dropped}, target truncated {truncated}), "
        f"holdout {len(holdout_rows)}"
    )

    def batches(rows, shuffle, epoch_seed=0):
        order = list(range(len(rows)))
        if shuffle:
            random.Random(epoch_seed).shuffle(order)
        for s in range(0, len(order), args.batch_size):
            yield collate([rows[j] for j in order[s : s + args.batch_size]], pad_token_id=tokenizer.pad_token_id)

    holdout_batches = list(batches(holdout_rows, shuffle=False))
    micro_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = math.ceil(micro_per_epoch * args.epochs / args.grad_accum)
    warmup = max(1, int(args.warmup_frac * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    initial_val = _eval_loss(model, holdout_batches, args.device) if holdout_batches else float("nan")
    print(f"step 0 holdout loss {initial_val:.4f}")
    history.append({"step": 0, "holdout_loss": initial_val})

    step, micro, running, t0 = 0, 0, 0.0, time.time()
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        for batch in batches(train_rows, shuffle=True, epoch_seed=args.seed + epoch):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                batch["labels"][:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            (loss / args.grad_accum).backward()
            running += float(loss)
            micro += 1
            if micro % args.grad_accum:
                continue
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, total=total_steps, warmup=warmup, peak=args.lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                row = {"step": step, "train_loss": running / (args.log_every * args.grad_accum),
                       "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - t0}
                running = 0.0
                if step == total_steps or step % (args.log_every * 5) == 0:
                    row["holdout_loss"] = _eval_loss(model, holdout_batches, args.device)
                history.append(row)
                print(json.dumps(row))
            if step >= total_steps:
                break
        epoch += 1

    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    meta = {
        "script": "merge_and_rebase.finetune.train_causal_sft",
        "args": vars(args),
        "saved_dtype": "float32",
        "dataset_rows": len(ds),
        "train_rows": len(train_rows),
        "train_rows_dropped": train_dropped,
        "train_targets_truncated": truncated,
        "holdout_indices": holdout_idx,
        "train_fingerprint_first50_sha1": fingerprint,
        "optimizer_steps": total_steps,
        "warmup_steps": warmup,
        "history": history,
        "final_holdout_loss": history[-1].get("holdout_loss"),
        "initial_holdout_loss": initial_val,
    }
    (out / "sft_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
