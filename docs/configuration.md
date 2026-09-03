# Configuration Reference

Configurations are JSON for evaluation and rebasin, and YAML or JSON for fine-tuning. CLI arguments override populated configuration fields.

## Vision Merge

The smallest useful merge configuration identifies the OpenCLIP base, method, and task checkpoints.

```json
{
  "suite": "vision8",
  "clip_model": "ViT-B-32",
  "clip_pretrained": "openai",
  "method": "task_arithmetic",
  "alpha": 0.3,
  "tuned_ckpts": {"Cars": "hf-hub:org/repository/Cars/full.pt"}
}
```

| Field | Type | Meaning |
|---|---|---|
| `suite` | string | Named dataset suite, such as `vision8`, `vision14`, or `vision20`. |
| `clip_model` | string | OpenCLIP architecture. |
| `clip_pretrained` | string | OpenCLIP pretrained-tag identifier. |
| `method` | string | Registered merge method from [methods.md](methods.md). |
| `method_params` | object | Method-specific parameters. |
| `tuned_ckpts` | object | Mapping from task name to local, HTTPS, or `hf-hub:` checkpoint reference. |
| `weights` | array or `null` | Per-task merge weights; `null` uses the method default. |
| `alpha` | number | Fixed interpolation scale. |
| `alpha_search` | boolean | Enable a scalar alpha sweep. |
| `text_features_source` | string | `auto`, `zero_shot`, or `tuned_ckpt`. |
| `batch_size`, `num_workers`, `val_fraction`, `seed` | scalar | Evaluation data-loader and split settings. |
| `strict_load` | boolean | Require strict compatible checkpoint loading. |
| `postmerge` | object | Optional post-merge adaptation block. |

Block extension calibration is task-local by default: `calibration_split` selects
the active task's loader. To use one task-independent support set for every
task, configure `calibration_dataset` inside `block_extension_params`. It can
name a supported task such as `ImageNet1K`, or point directly to a Hugging Face
dataset such as Tiny ImageNet:

```json
{
  "block_extension_enabled": true,
  "block_extension_params": {
    "calibration_dataset": {
      "path": "zh-plus/tiny-imagenet",
      "split": "train",
      "max_samples": 2048
    },
    "n_batches_act": 5,
    "calibration_split": "train"
  }
}
```

For a named task, use `"calibration_dataset": "ImageNet1K"` and select
`train`, `val`, or `test` with `calibration_split`; direct Hugging Face specs
select their raw split with the spec's `split` field. The alias
`calibration_task` is also accepted for named datasets. When
`calibration_dataset`/`calibration_task` is omitted, existing task-dependent
behavior is unchanged.

Use `hyperparam_search` instead of a fixed alpha for multi-parameter search:

```json
{
  "hyperparam_search": {
    "strategy": "sequential",
    "alpha": {"min": 0.0, "max": 1.0, "step": 0.1},
    "method_params": {"topk": [0.1, 0.2, 0.5]}
  }
}
```

`strategy` is either `sequential` or `sobol`. Sequential search enumerates supplied candidates; Sobol search accepts bounded numeric method parameters plus `num_samples`, `refinement_steps`, and `refine_factor`.

## Post-Merge Block

Add `postmerge` to a vision merge configuration to run adaptation after initial merging.

```json
{
  "postmerge": {
    "method": "adamerging",
    "alpha_mode": "task",
    "steps": 500,
    "lr": 0.001,
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "init_alpha": 0.3
  }
}
```

All post-merge methods accept `steps`, `lr`, `beta1`, `beta2`, `weight_decay`, `log_every`, and `device`. `adamerging` also accepts `alpha_mode` (`task` or `layer`) and alpha bounds. See [methods.md](methods.md) for supported methods and restrictions.

## Vision Rebasin

```json
{
  "suite": "vision8",
  "source_clip_model": "ViT-B-32",
  "source_clip_pretrained": "openai",
  "target_clip_model": "ViT-B-32",
  "target_clip_pretrained": "laion2b_s34b_b79k",
  "method": "gradfix",
  "method_params": {"mask_mode": "normal", "vote": "max"},
  "alpha": 1.0,
  "tuned_ckpts": {"Cars": "path-or-reference"}
}
```

| Field | Meaning |
|---|---|
| `source_clip_model`, `source_clip_pretrained` | Source base defining the task vector. |
| `target_clip_model`, `target_clip_pretrained` | Target base receiving the transported vector. |
| `method` and `method_params` | Registered transport method and its parameters. |
| `tasks` | `all` or a task-name subset. |
| `alpha`, `alpha_search`, `alpha_selection` | Fixed scale or search; selection is `shared` or `per_task`. |
| `weights` | Task-vector composition weights where supported. |
| `grad_batch_size`, `grad_imgs_per_class`, `grad_num_batches` | GradFix data budget controls. |
| `save_merged`, `save_transported_tvs_dir` | Optional output locations for derived checkpoints or task vectors. |

## Fine-Tuning

Fine-tuning configurations contain a `common` section and optional per-dataset overrides. The canonical starting point is [vision.yaml](https://github.com/apanariello4/merge-and-rebase/blob/main/src/merge_and_rebase/finetune/configs/vision.yaml).

| Section | Important fields |
|---|---|
| `common.backbone` | `name`, `clip_model`, `clip_pretrained`. |
| `common.data` | `batch_size`, `num_workers`, `val_fraction`, `pin_memory`. |
| `common.train` | `epochs`, optimizer, `lr`, `weight_decay`, accumulation, scheduler, clipping, early stopping. |
| `common.strategy` | strategy `name`, `forward_mode`, `forward_mode_params`, and strategy `params`. |
| `common.output` | output directory, save format, and last-epoch checkpoint behavior. |
| `common.logging` | local log path and optional Weights & Biases configuration. |
| `datasets` | per-task overrides for any `common` field. |

Use `--suite`, `--datasets`, `--device`, and logging flags to override a fine-tuning configuration at runtime.
