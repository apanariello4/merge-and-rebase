from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.eval import llm_rebase
from merge_and_rebase.eval.llm_rebase import (
    _build_text_calibration_loader,
    _prepare_resized_task_delta,
)


class _TinyTokenizer:
    def __call__(self, prompts, *, truncation, max_length, padding):
        assert truncation and padding == "max_length"
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for prompt in prompts:
            tokens = [idx + 1 for idx, _ in enumerate(prompt.split())][:max_length]
            ids.append(tokens + [0] * (max_length - len(tokens)))
            masks.append([1] * len(tokens) + [0] * (max_length - len(tokens)))
        return {"input_ids": ids, "attention_mask": masks}

    def pad(self, features, *, return_tensors, padding, max_length):
        assert return_tensors == "pt" and padding == "max_length"
        return {
            key: torch.tensor([row[key][:max_length] for row in features])
            for key in features[0]
        }


def test_text_calibration_masks_only_padded_labels() -> None:
    loader = _build_text_calibration_loader(
        tokenizer=_TinyTokenizer(),
        texts=["one token", "one two three"],
        batch_size=2,
        max_length=5,
    )
    batch = next(iter(loader))

    valid = batch["attention_mask"].bool()
    assert torch.equal(batch["labels"][valid], batch["input_ids"][valid])
    assert torch.all(batch["labels"][~valid] == -100)


class _DepthModel(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(depth)])


class _DepthFamily:
    def transportable_keys(self, state_dict):
        return {key for key in state_dict if key.startswith("layers.")}


def test_resized_task_contexts_extend_fresh_source_copies_and_keep_new_transport_keys(monkeypatch) -> None:
    def fake_extend(*, source_base_model, source_ft_model, target_layers_total, **kwargs):
        for model in (source_base_model, source_ft_model):
            while len(model.layers) < target_layers_total:
                model.layers.append(nn.Linear(3, 3, bias=False))
        return len(source_base_model.layers)

    monkeypatch.setattr(llm_rebase, "run_block_extension_llm", fake_extend)
    base = _DepthModel(depth=2)
    family = _DepthFamily()

    first_base = _DepthModel(depth=2)
    first_base.load_state_dict(base.state_dict())
    first_ft = _DepthModel(depth=2)
    first_ft.load_state_dict(base.state_dict())
    first_ft.layers[0].weight.data.add_(1.0)

    second_base = _DepthModel(depth=2)
    second_base.load_state_dict(base.state_dict())
    second_ft = _DepthModel(depth=2)
    second_ft.load_state_dict(base.state_dict())
    second_ft.layers[0].weight.data.add_(2.0)

    first = _prepare_resized_task_delta(
        source_base_model=first_base,
        source_ft_model=first_ft,
        calibration_loader=object(),
        target_layers_total=3,
        config=object(),
        family_adapter=family,
        device="cpu",
    )
    second = _prepare_resized_task_delta(
        source_base_model=second_base,
        source_ft_model=second_ft,
        calibration_loader=object(),
        target_layers_total=3,
        config=object(),
        family_adapter=family,
        device="cpu",
    )

    # The template remains reusable; each task copy is extended to the target
    # depth before its task vector is transported to the target width.
    assert len(base.layers) == 2
    assert len(first.source_model.layers) == len(second.source_model.layers) == 3
    assert "layers.2.weight" in first.transport_keys
    assert "layers.2.weight" in second.transport_keys
    assert "layers.2.weight" in first.source_base
    assert not torch.equal(first.delta["layers.0.weight"], second.delta["layers.0.weight"])
