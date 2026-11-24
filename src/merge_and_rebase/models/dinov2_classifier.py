"""DINOv2 classification without a text encoder or zero-shot dependencies."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model


class DinoPreprocess:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]


class DinoV2Classifier(nn.Module):
    def __init__(self, backbone: Dinov2Model, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(2 * backbone.config.hidden_size, num_classes)
        self.linear_probe = False

    def set_training_mode(self, mode: str):
        if mode not in {"full", "linear_probe", "backbone_only"}:
            raise ValueError("DINOv2 supports full, linear_probe, and backbone_only training modes.")
        self.linear_probe = mode == "linear_probe"
        self.backbone.requires_grad_(not self.linear_probe)
        self.head.requires_grad_(mode != "backbone_only")
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.linear_probe:
            self.backbone.eval()
        return self

    def forward(self, images):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.linear_probe):
            tokens = self.backbone(pixel_values=images).last_hidden_state
            features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1)
        return self.head(features)

    @classmethod
    def pretrained(cls, model_id: str, revision: str | None = None):
        backbone = Dinov2Model.from_pretrained(model_id, revision=revision)
        processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        return backbone, processor

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any], task: str | None = None):
        task_payload = payload["tasks"][task] if "tasks" in payload else payload
        model = cls(Dinov2Model(Dinov2Config.from_dict(payload["model_config"])), len(task_payload["classnames"]))
        model.backbone.load_state_dict(payload["backbone_state"], strict=True)
        model.head.load_state_dict(task_payload["head_state"], strict=True)
        return model
