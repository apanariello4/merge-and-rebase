# Fine-Tuning

## Vision

Vision fine-tuning uses YAML files in `src/merge_and_rebase/finetune/configs/`.

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --datasets CIFAR10,CIFAR100,EuroSAT
```

Use `--suite vision8` to select a named benchmark suite. Outputs default to `src/checkpoints/finetune/<model>/<pretrained>/<task>/` and include checkpoints, summaries, and append-only event logs.

Available strategies are `full`, `linear_probe`, and `peft_lora`. Forward modes, including `linearized_ntk`, are configured through the strategy configuration. The main vision presets include `vision.yaml`, `vision-peft.yaml`, `vision-ntk.yaml`, and two-stage variants.

### Fine-tuning methods

A method combines three blocks of `common.strategy` (plus an optional `common.regularization`):

| Key | Values |
|-----|--------|
| `strategy.name` | `full`, `linear_probe`, `peft_lora` |
| `strategy.forward_mode` | `standard`, `linearized_ntk` |
| `strategy.params.parameterization` | `weights` (default), `delta` |
| `strategy.params.trainable_params` | `all_trainable` (default), `regularized_only` |
| `regularization.name` | `distillation`, `kfac_ggn`, `ekfac_ggn`, `composite` |

Every method uses the same entrypoint. Only the config changes:

```bash
python -m merge_and_rebase.finetune.train_vision --vision-config <config.yaml> --suite vision8
```

**Standard.** Non-linear fine-tuning of all weights (`full`) or of LoRA adapters (`peft_lora`), trained with cross-entropy. Presets: `vision.yaml`, `vision-peft.yaml`, `vision-vitl.yaml`, and `vision-muon.yaml` (Muon optimizer). The `vision-two-stage*.yaml` variants first fine-tune the text embeddings (`strategy.text_embeddings_finetune`) and then the image encoder.

**Linearized (NTK).** With `forward_mode: linearized_ntk`, the model is replaced by its first-order Taylor expansion around the pretrained weights θ₀: f(x; θ₀ + τ) ≈ f(x; θ₀) + J_θ f(x; θ₀)·τ. Training only happens in the tangent space, which makes the task vectors τ more disentangled when you merge them. Presets: `vision-ntk.yaml`, `vision-peft-ntk.yaml`. The cost is about one JVP per forward pass, so the presets use `batch_size: 16` with `accumulate_grad_batches: 8`, which keeps the effective batch size at 128.

**TAK (`configs/TAK/`).** Linearized fine-tuning with `parameterization: delta` (the trainable parameters are the displacement τ from θ₀) and a `kfac_ggn` regularizer. The regularizer penalizes τ under a K-FAC approximation of the Gauss-Newton/Fisher matrix, estimated on the other tasks (`train_percent` of their data, cached in `cache_dir`). The penalty strength is set by `reg_lambda`. There is one file per backbone and suite:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/TAK/vision-tak-kfac-vitb32-vision8.yaml
```

**Distillation.** The `distillation` regularizer adds an MSE (or another loss) between student and teacher activations at the configured `locations`, for example `image_features`. The teacher is either `frozen` (pretrained or `initialization.checkpoint`) or `online`: it is trained alongside the student, with its own `strategy`, `train`, and optional nested `regularization`. With `along_path.enabled`, the student is evaluated at θ₀ + α·τ, where α is sampled from `alpha_range`. Under `sampling: curriculum`, the interval starts at α = 1 and widens during training. Under `sampling: uniform`, α is drawn uniformly from the full range. This keeps the model well-behaved when the task vector is rescaled.

**DELTA (`configs/DELTA/Full FT`, `configs/DELTA/LoRA`).** A `composite` of two regularizers:

1. `distillation` from an online *linearized* teacher, itself regularized with `ekfac_ggn`.
2. An `ekfac_ggn` penalty on the student, with ImageNet21KP as the reference dataset.

The student stays non-linear (`forward_mode: standard`, `parameterization: delta`) and is distilled along the path. The LoRA variant saves only the adapters (`save_format: peft`). The directory names contain spaces, so quote the path:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config "src/merge_and_rebase/finetune/configs/DELTA/Full FT/vision-delta-vitb32-vision8.yaml"
```

The TAK and DELTA configs already set `datasets_order` for their suite (`vision8`, `vision14`, `vision20`), so `--suite` is optional. Use `--reference-suite` or `--reference-datasets` to change the regularizer's reference tasks. Use `--force-recompute` to rebuild the Fisher/K-FAC caches.

## Text

```bash
python -m merge_and_rebase.finetune.train_text \
  --text-config src/merge_and_rebase/finetune/configs/text-peft.yaml \
  --suite nli6
```

Text configurations support full fine-tuning, linear probing, LoRA adapters, task heads, and PEFT export.

## Regularization

Regularizers are available in `train_vision` only; `train_text` does not read a `regularization` block. Set `common.regularization.name` to exactly one registered regularizer. The training loss per step is

```text
loss = task_loss + regularizer.apply(...)
```

| Name | Penalty |
|------|---------|
| `distillation` | Feature/logit matching against a frozen or online teacher, optionally along the path θ₀ + α·τ. |
| `kfac_ggn` | Quadratic penalty on τ under a K-FAC approximation of the GGN, estimated on reference tasks. |
| `ekfac_ggn` | Same as `kfac_ggn` with the eigenvalue-corrected (EK-FAC) approximation. |
| `composite` | Sum of several of the above (see below). |

For `kfac_ggn` and `ekfac_ggn`, the penalty scale is `reg_lambda`, with `full_block_scaler` and `projection_scaler` weighting the individual curvature terms. Curvature statistics are computed once on `train_percent` of each reference task and cached in `cache_dir`. The cache is reused until a relevant parameter changes or `force_recompute` / `--force-recompute` is set.

Reference tasks are resolved in this order: `--reference-datasets`, `--reference-suite`, `regularization.reference_datasets`, `regularization.reference_suite`, then the tasks of `--suite` (or of the training run). The task being trained is always excluded. Training on a single dataset requires an explicit reference selection.

### Composite regularizer

`composite` lets one run combine several regularizers, as DELTA does. Each entry in `regularizers` is a complete regularizer config with its own `name`:

```yaml
regularization:
  name: composite
  regularizers:
    - name: distillation
      locations: [{student: image_features, teacher: image_features, loss: mse, weight: 1.0}]
      teacher: {mode: online, ...}
    - name: ekfac_ggn
      reg_lambda: 500.0
      reference_datasets: [ImageNet21KP]
```

Semantics (`finetune/regularizers/composite.py`):

- **Loss.** The child losses are summed without an extra weight. Weight each child through its own parameters (`reg_lambda`, per-location `weight`).
- **Preparation.** Each child runs its own `finalize_model` and `prepare`, in list order, so every child keeps its own cache and reference tasks.
- **Batch overrides.** At most one child may change the training batch (for example, `distillation` with `along_path` sampling α). Two overriding children raise an error.
- **Optimizers.** Optimizers owned by children, such as an online teacher's, are collected and stepped together with the student's.
- **Checkpoints.** Extra checkpoint payloads from children are merged. Duplicate keys raise an error.

Nesting is allowed: a `distillation` teacher can carry its own `regularization` block, which is how the DELTA teacher is trained with `ekfac_ggn`.

## Logging

All entrypoints write local structured logs. Configure optional Weights & Biases logging with:

```yaml
logging:
  use_wandb: false
  project: null
  entity: null
  tags: []
  mode: online
  local_log_dir: null
  log_every_n_steps: 50
```

See [methods](methods.md) for the runtime behavior of merge, transport, and post-merge methods.
