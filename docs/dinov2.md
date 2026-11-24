# DINOv2 classification and backbone merging

DINOv2 uses the existing `finetune.train_vision` and `eval.vision_merge` commands. Select `backbone.name: dinov2` and a Hugging Face model ID such as `facebook/dinov2-base`. Vision8, vision14, and vision20 reuse the repository's dataset definitions, class ordering, and seeded validation/test split policy.

## Install and quick start

```bash
uv sync --extra data --extra dev
uv run python -m merge_and_rebase.finetune.train_vision \
  --vision-config configs/examples/dinov2/vision8_full.yaml
uv run python -m merge_and_rebase.eval.vision_merge \
  --config configs/examples/dinov2/vision8_merge.json
```

The first training run downloads the backbone, image processor, and datasets. `common.backbone.revision` can pin the model to a Hub commit. Training configured for CUDA fails clearly when CUDA is unavailable; pass `--device cpu` for an explicit CPU run.

## Training modes

A randomly initialized, task-specific linear classifier consumes the concatenation of the final CLS token and mean patch-token features. The pretrained DINOv2 checkpoint supplies the visual backbone, rather than a classifier for these datasets.

| `common.strategy.name` | Classifier | Backbone |
|---|---|---|
| `full` | Trained with the backbone | Fine-tuned from the shared pretrained base |
| `linear_probe` | Trained independently for each task | Original pretrained weights; frozen and kept in evaluation mode |
| `classifier_then_backbone` | Trained in stage 1, then frozen | Frozen in stage 1, fine-tuned in stage 2 |

The two-stage method restores the best validation-selected classifier after stage 1. Stage 2 creates a fresh optimizer containing only backbone parameters. Both stages use the task's epoch budget by default; `common.train.classifier_stage` overrides stage-1 settings, including `epochs`, `lr`, and `weight_decay`. Stage-1 defaults are learning rate `1e-3` and zero weight decay. Joint/backbone examples use learning rate `1e-5` and weight decay `0.1`.

A head-only baseline is the stage-1 model, or a separate `linear_probe` run. It always uses the original pretrained backbone. Final two-stage results use the fine-tuned stage-2 backbone. DINOv2 has no text encoder or zero-shot task classifier, so zero-shot accuracy is not defined for this runtime.

### Suites and task selection

Runnable examples live in `configs/examples/dinov2/`. Use the `vision8_*.yaml` presets for eight tasks, or `vision20_*.yaml` for the complete twenty-task epoch configuration. Vision14 is a subset of vision20:

```bash
uv run python -m merge_and_rebase.finetune.train_vision \
  --vision-config configs/examples/dinov2/vision20_full.yaml --suite vision14
uv run python -m merge_and_rebase.finetune.train_vision \
  --vision-config configs/examples/dinov2/vision20_linear_probe.yaml
uv run python -m merge_and_rebase.finetune.train_vision \
  --vision-config configs/examples/dinov2/vision20_classifier_then_backbone.yaml
```

`--datasets CIFAR10,CIFAR100` selects individual tasks. Every task starts independently from the same pretrained backbone with a new task head. It does not start from the previous task or from a merged model. Settings under `datasets.<task>` override `common`; use separate output directories for different pretrained bases. All tasks within one training configuration share the backbone configuration and output root.

### Data, optimization, and seeds

Images use the saved checkpoint processor with RGB conversion and deterministic resizing/cropping; no additional training augmentation is enabled. `data.train_preprocess: eval` and the HF loader profile are supported. Custom local data roots are not supported by this runtime.

`seed` controls training initialization. `data.split_seed` controls dataset partitions and defaults to `seed`. The repository constructs disjoint validation/test subsets from the supplied evaluation split, using `data.val_fraction` (default 0.1) for validation and the remainder for held-out test. Training uses the supplied training split. Keep the split seed and validation fraction fixed across training and merging; merge evaluation rejects a changed validation fraction.

Training supports FP32, Adam/AdamW/SGD, cosine/linear/constant schedules, warmup, gradient clipping, accumulation, an optional `train.max_train_batches` limit, and optional early stopping. The examples use a microbatch of 32 with four accumulation steps, for effective batch size 128. Use 16 and eight respectively to lower peak GPU memory.

## Training artifacts and inference

For `output.out_dir: checkpoints/dinov2/full`, the structure is:

```text
checkpoints/dinov2/full/
  base.pt
  Cars/
    full_best.pt
    full_last.pt
    full.json
    progress.json
```

Best/last checkpoints and `progress.json` are written after each epoch. Final `<strategy>.json` files contain validation-selected test metrics and zero-based best/last epoch indices. Two-stage summaries also contain `stages.classifier` and `stages.backbone`. Stage-1 checkpoints use the additional `_classifier` suffix, for example `classifier_then_backbone_classifier_best.pt`.

`output.save_last_epoch` controls last-checkpoint output; `output.save_checkpoints` controls model checkpoint output. `--skip-existing-task-vectors` skips existing best checkpoint files; it does not resume optimizer state or prove that a previously interrupted task completed all configured epochs.

Original training checkpoints contain `backbone_state`, `head_state`, ordered `classnames`, model/processor configuration, task/strategy, data settings, seed, base identity, and `base_checkpoint`. The base reference is an absolute local path by default. If relocating training checkpoints, update this reference, or use a valid `hf-hub:org/repo/path/base.pt` reference. The exact base weights must match across merged tasks.

```python
import torch
import transformers
from PIL import Image
from merge_and_rebase.models.dinov2_classifier import DinoPreprocess, DinoV2Classifier

payload = torch.load("checkpoints/dinov2/full/Cars/full_best.pt", map_location="cpu", weights_only=True)
model = DinoV2Classifier.from_checkpoint(payload).eval()
processor_config = payload["processor"]
processor = getattr(transformers, processor_config["image_processor_type"]).from_dict(processor_config)
image = Image.new("RGB", (256, 256))  # Replace with an actual input image.
with torch.inference_mode():
    logits = model(DinoPreprocess(processor)(image).unsqueeze(0))
class_name = payload["classnames"][logits.argmax(-1).item()]
```

The [published vision8 DINOv2 exports](https://huggingface.co/merge-and-rebase/checkpoints/tree/main/finetune/dinov2-base/facebook) contain weights and essential inference metadata. Download them with `hf_hub_download` and use the same classifier loader. Those inference-only exports omit training provenance and metric files; use original training checkpoints, their exact `base.pt`, and task summaries for merge experiments.

## Merge backbones and keep task heads

Merge configurations are JSON with `backbone.name: dinov2`, a suite, `tuned_ckpts` mapping task IDs to original best checkpoints, and a `base_ckpt`. Each suite has separate joint and two-stage presets:

```bash
uv run python -m merge_and_rebase.eval.vision_merge \
  --config configs/examples/dinov2/vision14_merge.json
uv run python -m merge_and_rebase.eval.vision_merge \
  --config configs/examples/dinov2/vision20_classifier_then_backbone_merge.json
```

The vision14/20 examples reference the corresponding `vision20_<strategy>.yaml` training output paths. Use a task subset supported by the selected suite via `tasks` (a CSV string or list). Every requested task needs its own checkpoint.

Only the backbone enters the merge method. During evaluation the runtime loads the original head for each task ID, checks its class ordering, and asserts exact head equality after inference. A saved merged checkpoint contains the shared backbone plus all original task heads and their class names. It reloads independently of the training files:

```python
payload = torch.load("checkpoints/dinov2/vision20/classifier_then_backbone_merged.pt", map_location="cpu", weights_only=True)
model = DinoV2Classifier.from_checkpoint(payload, task="MNIST").eval()
```

### Merge methods and weight averaging

All currently registered dense prepared merge methods are covered by tiny DINOv2 compatibility tests. Useful method configurations include:

| Method | `method` | `method_params` |
|---|---|---|
| Task arithmetic | `task_arithmetic` | `{}` |
| TIES | `ties_merge` | `{"topk": 0.2, "merging_type": "mean", "low_memory": true}` |
| ISO-C | `isoc_merge` | `{"vector_1d_merge": "zero"}` |
| TSV | `tsv_merge` | `{"sv_reduction": 0.05, "svd_dtype": "float64"}` for 20 tasks |
| Weighted average | `weighted_average` | `{}` with `alpha: 1.0` and `alpha_search: false` |

For TSV, use rank fraction `1/N` for N tasks (1/8, 1/14, or 1/20). Matrix methods retain their own tensor-selection and non-matrix fallback rules. Full method documentation is in [Merge Methods](methods.md).

Arithmetic weight averaging also has an equivalent task-vector recipe: `base + (1/N) * sum(tuned_i - base)`. Configure `method: task_arithmetic`, `alpha_search: false`, and `alpha: 1/N` for that recipe. The `weighted_average` registry method already divides by the sum of weights, so its plain arithmetic average uses `alpha: 1.0`.

### Alpha selection and metrics

The example merge presets search alpha from 0 through 2 inclusive in steps of 0.1, with no early stopping. They choose one global alpha on validation data and evaluate the selected backbone on held-out test data. `alpha_early_stop` optionally stops a sweep when the selection score decreases. Alpha 0 restores the pretrained backbone with the existing trained task heads; it is a head-only model when those heads were fitted on the frozen pretrained features.

To normalize against separately fine-tuned experts, set `selection_metric: mean_normalized_top1` and `single_task_summaries` to the task-to-training-JSON mapping. The runtime computes an arithmetic mean across tasks:

- `mean_top1`: mean of per-task accuracy.
- `mean_normalized_top1`: mean of each task's accuracy divided by its corresponding single-task expert accuracy.

Validation normalization uses each summary's `best_val_top1`; test normalization uses `test_top1`. The ratio of the two suite means is a separate statistic and can differ from the mean per-task ratios. Without normalized selection, the default selection metric is `mean_top1`.

`alpha` sets a fixed coefficient; `alphas` supplies an explicit list. Set `alpha_search: false` when using either. `fixed_eval_split: val` or `test` evaluates exactly one coefficient on the requested split and skips the other split. It never selects alpha on test data. Search progress is written to a sibling `<results-stem>.progress.json`; final JSON includes per-task and mean accuracies and the alpha candidates.

## Multiple GPUs and CPU preparation

Independent training variants or merge runs can use separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m merge_and_rebase.finetune.train_vision --vision-config configs/examples/dinov2/vision20_full.yaml
CUDA_VISIBLE_DEVICES=1 uv run python -m merge_and_rebase.finetune.train_vision --vision-config configs/examples/dinov2/vision20_classifier_then_backbone.yaml
```

These commands each use one GPU; the DINOv2 runtime does not launch distributed training. Start independent commands in separate terminals or with a job scheduler.

ISO/TSV CPU SVD preparation can dominate runtime. Set the same `exclusive_merge_lock` path in their JSON configurations, for example `results/dinov2/cpu_prepare.lock`, to serialize preparation across processes. The lock is released before dataset loading and GPU alpha/test evaluation, allowing concurrent GPU evaluation. File locks are advisory and require a shared filesystem with working POSIX `flock` support.

## Supported scope and verification

DINOv2 training supports the three strategies above with `standard` forward mode and empty `strategy.params`. Text tuning, PEFT, CLIP-specific regularizers/attention patches, specialized forward modes, post-merge training, rebasing, and joint method-parameter/Sobol searches are outside this runtime's supported scope. Use the scalar alpha controls for DINOv2 merging.

Offline regression checks use tiny random DINOv2 models to cover gradient routing, frozen-backbone/head invariants, class ordering, checkpoint round trips, all registered dense merge methods, task-head retention, validation-only alpha selection, split-specific fixed evaluation, and CPU lock release on success/failure. Run them with:

```bash
uv run pytest tests/test_dinov2.py -q
```
