"""Typed state passed between LLM model loading, calibration and evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from ...hyperparam_search import SearchPlanner
from ...models.text_lm import TextBuildConfig, TextLM
from ...rebase.model_families.base import ModelFamilyAdapter
from ...rebase.run_config import ResolvedRunConfig, RunPlan
from .splits import HarnessSlices

if TYPE_CHECKING:
    from .context import TextCalibrationCache
    from .stages import LlmTaskContext

TensorDict = dict[str, torch.Tensor]


@dataclass
class LlmModels:
    source_llm: TextLM
    target_llm: TextLM
    source_build_cfg: TextBuildConfig
    source_base_sd: TensorDict
    target_base_sd: TensorDict
    source_family: ModelFamilyAdapter | None
    target_family: ModelFamilyAdapter | None
    device: str
    num_labels: int
    tp_keys: set[str] | None
    full_fp_keys: set[str] | None
    load_tuned: Callable[..., Mapping[str, torch.Tensor]]

    @property
    def family_adapter(self) -> ModelFamilyAdapter | None:
        return self.target_family or self.source_family


@dataclass(frozen=True)
class CalibrationSettings:
    prompts: list[str] | None
    dataset: str | Mapping[str, Any] | None
    n_sequences: int | None
    include_target: bool
    batch_size: int
    max_length: int


@dataclass
class LlmCalibration:
    cache: TextCalibrationCache
    batch_size: int
    max_length: int
    num_batches: int | None
    blockext_loader: DataLoader | None = None


@dataclass
class LlmEvaluation:
    tasks: list[str]
    tuned_ref_list: list[str]
    suite_name: str | None
    is_harness_only: bool
    harness_tasks: list[str]
    harness_num_fewshot: int | list[int] | dict[str, int]
    harness_batch_size: str
    harness_limit: int | float | None
    slices: HarnessSlices
    run_before_rebase_eval: bool
    eval_before_rebase_only: bool
    eval_mode: str
    head_key_pattern: str
    task_heads_path: str | None
    search_planner: SearchPlanner
    baseline_results: dict[str, dict[str, float]] = field(default_factory=dict)

    def baseline_summary(self) -> dict[str, Any] | None:
        if not self.baseline_results:
            return None
        if len(self.baseline_results) == 1:
            return next(iter(self.baseline_results.values()))
        return dict(self.baseline_results)

    def evaluate_before_rebase(self, models: LlmModels, model: torch.nn.Module, label: str) -> None:
        from .harness import run as run_harness

        print(f"\nEvaluating source model ({label}) with lm-harness (before rebase)...")
        results = run_harness(
            tasks=list(self.harness_tasks),
            model=model,
            tokenizer=models.source_llm.tokenizer,
            device=models.device,
            num_fewshot=self.harness_num_fewshot,
            batch_size=self.harness_batch_size,
            limit=self.harness_limit,
            samples=self.slices.search,
        )
        for task_name, acc in results.items():
            print(f"  [before rebase / {label}] {task_name}: {acc:.4f}")
        self.baseline_results[label] = results


@dataclass
class LlmRuntime:
    cfg: dict[str, Any]
    models: LlmModels
    calibration: LlmCalibration
    evaluation: LlmEvaluation
    resolved: ResolvedRunConfig
    plan: RunPlan
    task_contexts: dict[str, LlmTaskContext]
    ignored_block_extension_fields: list[str]
