"""The causal-LM SFT used to manufacture an SFT-sized math task vector.

What has to hold for the resulting delta to be a fair P1 source:

- the training text is lm-eval's own gsm8k rendering, so the tuned model is not
  scored in a format it never saw;
- the loss covers the answer (+ EOS) only, never the prompt or padding;
- the checkpoint is saved in fp32, so small SFT updates are not rounded away;
- the held-out split is deterministic and recorded.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from merge_and_rebase.finetune import train_causal_sft as sft


class _CharTok:
    """One token per character, EOS = 1, pad = 0."""

    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_default_templates_match_lm_eval_gsm8k():
    """lm-eval gsm8k: doc_to_text 'Question: {{question}}\\nAnswer:', delimiter ' ', target '{{answer}}'."""
    prompt, target = sft.render_example(
        {"question": "2+3?", "answer": "2+3=5\n#### 5"},
        prompt_template=sft.GSM8K_PROMPT_TEMPLATE, target_template=sft.GSM8K_TARGET_TEMPLATE,
    )
    assert prompt + target == "Question: 2+3?\nAnswer: 2+3=5\n#### 5"


def test_render_refuses_missing_fields():
    with pytest.raises(KeyError, match="answer"):
        sft.render_example({"question": "q"}, prompt_template="{question}", target_template="{answer}")


def test_loss_covers_target_and_eos_only():
    enc = sft.tokenize_example(_CharTok(), "Q:", " ab", max_length=64)
    assert enc["input_ids"] == [ord("Q"), ord(":"), ord(" "), ord("a"), ord("b"), 1]
    assert enc["labels"] == [-100, -100, ord(" "), ord("a"), ord("b"), 1]


def test_long_rows_truncate_the_target_and_drop_prompt_only_rows():
    enc = sft.tokenize_example(_CharTok(), "Q:", " abcdef", max_length=4)
    assert len(enc["input_ids"]) == 4 and enc["labels"][:2] == [-100, -100]
    assert sft.tokenize_example(_CharTok(), "QQQQ", " a", max_length=4) is None


def test_collate_masks_padding_from_loss_and_attention():
    batch = sft.collate(
        [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}], pad_token_id=0
    )
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_split_is_deterministic_disjoint_and_complete():
    a = sft.split_indices(50, holdout=7, seed=3)
    assert a == sft.split_indices(50, holdout=7, seed=3)
    train, hold = a
    assert len(hold) == 7 and not set(train) & set(hold) and sorted(train + hold) == list(range(50))
    assert sft.split_indices(50, holdout=7, seed=4) != a


def test_lr_schedule_warms_up_then_decays_to_zero():
    lrs = [sft._lr_at(s, total=100, warmup=10, peak=1.0) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(b <= a + 1e-12 for a, b in zip(lrs[10:], lrs[11:]))
    assert lrs[-1] < 0.01


def test_end_to_end_saves_an_fp32_checkpoint_that_moved(tmp_path, monkeypatch):
    """A tiny random Qwen2 trains for a few steps on CPU and is saved in fp32 with metadata."""
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    vocab = {chr(i): i - 31 for i in range(32, 127)}
    vocab.update({"<pad>": 0, "<eos>": 96})
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    raw.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>")
    base_dir = tmp_path / "base"
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=97, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    AutoModelForCausalLM.from_config(config).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)

    rows = [{"question": f"{i}+1?", "answer": f"{i}+1={i + 1}\n#### {i + 1}"} for i in range(24)]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: rows
    monkeypatch.setitem(sys.modules, "datasets", fake)

    out = tmp_path / "sft"
    sft.main([
        "--model", str(base_dir), "--out", str(out), "--device", "cpu", "--epochs", "2",
        "--lr", "1e-3", "--batch-size", "4", "--grad-accum", "1", "--holdout", "4",
        "--max-length", "64", "--log-every", "2",
    ])

    meta = json.loads((out / "sft_meta.json").read_text())
    assert meta["saved_dtype"] == "float32" and meta["train_rows"] == 20 and len(meta["holdout_indices"]) == 4
    assert meta["final_holdout_loss"] < meta["initial_holdout_loss"]
    tuned = AutoModelForCausalLM.from_pretrained(out, dtype=None)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=None)
    assert all(p.dtype == torch.float32 for p in tuned.parameters())
    key = "model.layers.0.mlp.down_proj.weight"
    assert not torch.equal(tuned.state_dict()[key], base.state_dict()[key])
