# merge-and-rebase

![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/llm-transformers-F9AB00?logo=huggingface&logoColor=white)
![NLI](https://img.shields.io/badge/text-NLI%20merging-0A7F5A)
![OpenCLIP](https://img.shields.io/badge/backbone-OpenCLIP-1F6FEB)
![Status](https://img.shields.io/badge/status-research%20code-6F42C1)
[![BibTex](https://img.shields.io/badge/citation-bibtex-4E9A06?logo=bibtex&logoColor=white)](#citation)

`merge-and-rebase` is a research codebase for model merging, task-vector transport, and configurable fine-tuning across vision and text models. It is built for fast iteration on checkpoint merging, rebasing, and evaluation workflows.

## Highlights

- Config-driven OpenCLIP fine-tuning across multiple datasets
- Vision checkpoint merging with `weighted_average`, `task_arithmetic`, `ties_merge`, `dare_merge`, `tsv_merge`, `isoc_merge`, `isocts_merge`, `cart_merge`, and `pcb`/`pcb_merge`
- Zero-shot and merged evaluation on full benchmark test sets
- Advanced task-vector fine-tuning techniques such as TAK and DELTA
- Task-vector transport utilities under `merge_and_rebase.rebase`
- GradFix-based rebasing for cross-base transfer in vision models
- Text fine-tuning and merge evaluation for NLI-style LLM setups

---

## Setup

Create an environment with `uv` and install the package in editable mode:

```bash
uv venv .venv
source .venv/bin/activate
```

Minimal install:

```bash
uv pip install -e .
```

Install with dataset dependencies for fine-tuning and evaluation:

```bash
uv pip install -e ".[data]"
```

Install with development and test extras:

```bash
uv pip install -e ".[dev,data,test]"
```

---

## Vision Fine-Tuning

Vision fine-tuning is driven by config files. The CLI is used to choose the config and optionally restrict datasets.

### Config

Default vision config:

`src/merge_and_rebase/finetune/configs/vision.yaml`

This file defines:
- model backbone and pretrained variant
- training hyperparameters
- strategy choice, such as `linear_probe` or `full`
- dataset list and suite selection
- per-dataset overrides

### Running Fine-Tuning

Run all datasets defined by the config:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml
```

Restrict to a subset of datasets:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --datasets CIFAR10,CIFAR100,EuroSAT
```

Or run a named suite:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --suite vision8
```

For linear-attention LoRA runs, use the preset config:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision-peft-linear.yaml \
  --suite vision8
```

Outputs are saved to `src/checkpoints/finetune/<model>/<pretrained>/<task>/` by default.

Each task produces:
- `<strategy>.pt`: fine-tuned model checkpoint
- `<strategy>.json`: training log with metrics and hyperparameters

Each CLI run also writes:
- a structured run summary JSON
- a sibling `*.events.jsonl` file with append-only metric events

### Text Fine-Tuning

Text fine-tuning follows the same `common` plus per-dataset override structure and supports:
- `full`
- `linear_probe`
- `peft_lora`
- task-wise head extraction into `heads.pt`
- PEFT adapter export with `save_format: peft`

Starter configs:
- `src/merge_and_rebase/finetune/configs/text.yaml`
- `src/merge_and_rebase/finetune/configs/text-peft.yaml`

Run:

```bash
python -m merge_and_rebase.finetune.train_text \
  --text-config src/merge_and_rebase/finetune/configs/text-peft.yaml \
  --suite nli6
```

### Logging

All main CLI entrypoints support the same logging config block:

```yaml
logging:
  use_wandb: false
  project: null
  entity: null
  tags: []
  mode: online
  local_log_dir: null
  log_every_n_steps: 50
  run_name: null
```

By default, local structured logs are always written. If `local_log_dir` is unset, run summaries go to the entrypoint's natural output area when one exists, otherwise to `src/.cache/run_logs/<entrypoint>/`.

All entrypoints also accept CLI overrides:
- `--use-wandb` / `--no-use-wandb`
- `--wandb-project`
- `--wandb-entity`
- `--wandb-tags tag1,tag2`
- `--wandb-mode online|offline|disabled`
- `--local-log-dir`
- `--run-name`
- `--log-every-n-steps`

Example enabling W&B for vision fine-tuning:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --use-wandb \
  --wandb-project merge-and-rebase \
  --wandb-tags vision,finetune
```

### Available Strategies

Strategies live in `src/merge_and_rebase/finetune/strategies/`.

Currently available:
- `full`: full fine-tuning of all model parameters
- `linear_probe`: train only a linear classification head on frozen features
- `ntk`: first-order linearized fine-tuning via JVP around initialization
- `peft_lora`: LoRA adapters on the image encoder

### Extending Strategies

To add a new strategy:

1. Create a new file in `src/merge_and_rebase/finetune/strategies/`.
2. Implement `configure(...)` to return the optimizer, info dict, and scheduler.
3. Register the strategy in `registry.py`.

Schedulers and optimizers are owned by the strategy implementation.

### Available Regularizers

Regularizers live in `src/merge_and_rebase/finetune/regularizers/`.

Currently available:
- `distillation`: teacher-student knowledge distillation
- `kfac_ggn`: curvature regularization that caches second-order statistics of reference tasks and penalizes updates that interfere with those tasks
- `ekfac_ggn`: a variant of the `kfac_ggn` penalty that applies a more fine-grained approximation of the Hessian

### Extending Regularizers

To add a new regularizer:

1. Create a new file in `src/merge_and_rebase/finetune/regularizers/`.
2. Implement `prepare(...)` to build any cached state and return the prepared regularizer plus an info dict.
3. Implement `apply(...)` to return the regularization loss for each training step.
4. Optionally implement `finalize_model(...)` if the regularizer needs to patch or prepare the model before training starts.
5. Register the regularizer in `registry.py`.

Prepared regularizers can also expose auxiliary optimizer bundles, batch overrides, checkpoint payloads, and cleanup hooks when needed.

---

## Merge and Evaluate

Checkpoint merging and evaluation code lives in `src/merge_and_rebase/merge` and `src/merge_and_rebase/eval`.

Task-vector rebasing and transport code lives in `src/merge_and_rebase/rebase`.

Built-in merge methods:
- `weighted_average`: weighted checkpoint averaging relative to a base model
- `task_arithmetic`: sum of task vectors relative to a base model
- `ties_merge`: TIES-style sparse sign-resolved task-vector merge
- `dare_merge`: DARE-style random sparsification with optional rescaling
- `tsv_merge`: task-singular-vector merge for matrix-valued weights
- `isoc_merge`: isotropic composition on matrix-valued deltas
- `isocts_merge`: isotropic merge with common and task-specific subspaces
- `cart_merge`: low-rank CART merge for 2D parameters
- `pcb` and `pcb_merge`: PCB merge on flattened task vectors

For `tsv_merge`, `isoc_merge`, and `isocts_merge`, 2D parameters use the method-specific SVD rule. 1D parameters default to zero task-vector deltas, matching the current repo behavior. To use the paper-style average for 1D parameters instead, set:

```json
{
  "method_params": {
    "vector_1d_merge": "average"
  }
}
```

The functional API in `merge_and_rebase.merge.methods.functional` exposes the raw-matrix helpers for the functional merge methods.

Supported evaluation features:
- alpha search for merged evaluation with caching
- multi-parameter search across `alpha` and merge `method_params` with sequential grids or Sobol refinement

Evaluation is always performed on the full test set of each dataset.

Typical evaluation entrypoint:

```bash
python -m merge_and_rebase.eval.vision_merge --config <config_file>
```

By default, vision merge checks each tuned checkpoint for `tuned_text_features` and uses them when present. If they are missing, it falls back to zero-shot templates.

To override this behavior:

```json
{
  "text_features_source": "zero_shot"
}
```

Or force strict checkpoint-only text features:

```json
{
  "text_features_source": "tuned_ckpt"
}
```

Command-line parameters always override config values.

Multi-parameter search is configured with `hyperparam_search`. Sequential search evaluates ordered method-parameter combinations and sweeps `alpha` inside each combination. Sobol search samples a coarse low-discrepancy set, then refines around the best region.

Example:

```json
{
  "method": "ties_merge",
  "hyperparam_search": {
    "strategy": "sequential",
    "alpha": {"min": 0.0, "max": 1.0, "step": 0.1},
    "method_params": {
      "topk": [0.1, 0.2, 0.5, 1.0]
    }
  }
}
```

Sobol example:

```json
{
  "method": "cart_merge",
  "hyperparam_search": {
    "strategy": "sobol",
    "num_samples": 16,
    "refinement_steps": 1,
    "refine_factor": 0.5,
    "alpha": {"min": 0.0, "max": 1.0, "step": 0.1},
    "method_params": {
      "pruning_rank": {"min": 1, "max": 8, "step": 1, "type": "int"},
      "scaling_coeffs": {"min": 0.0, "max": 1.0, "step": 0.1}
    }
  }
}
```

For Llama-style causal LMs on NLI tasks such as SNLI, MNLI, SICK, QNLI, RTE, and SciTail:

```bash
python -m merge_and_rebase.eval.llm_merge \
  --model-name-or-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tasks snli,mnli,sick,qnli,rte,scitail \
  --eval-mode head_logits \
  --task-heads /path/to/heads.pt \
  --tuned-ckpts /path/to/tuned_a.pt /path/to/tuned_b.pt \
  --method task_arithmetic \
  --alpha-search --alpha-min 0.0 --alpha-max 1.0 --alpha-step 0.1
```

Starter config:

`configs/llm_merge_llama3_8b_knots_hf.json`

---

## Rebasin

The `rebase` package transports a merged task vector from a source base model to a target base model.

Core API:
- `merge_task_vectors(...)`: computes `Δ_merge = Σ_i w_i (tuned_i - source_base)`
- `transport_task_vector(...)`: transports `Δ_merge` from source to target coordinates
- `rebase_merged_task_vectors(...)`: end-to-end helper that merges, transports, and applies the result

Built-in transport methods:
- `identity`: no-op transport where `Δ' = Δ`
- `orthogonal_shift`: removes the component of `Δ` aligned with `(target_base - source_base)`
- `gradfix`: masks each task-vector component using gradient signs computed on the target model
- `theseus`: transports matrix-shaped updates with activation-aligned orthogonal Procrustes maps

Minimal Python usage:

```python
from merge_and_rebase.rebase import rebase_merged_task_vectors

rebased_sd = rebase_merged_task_vectors(
  source_base=source_base_sd,
  target_base=target_base_sd,
  tuned=[task1_sd, task2_sd],
  weights=[0.5, 0.5],
  alpha=1.0,
  transport_method="theseus",  # or "identity", "orthogonal_shift", "gradfix"
)
```

To add a new transport rule, create a class in `src/merge_and_rebase/rebase/methods/` that implements `transport(...)`, then register it in that module.

### GradFix Rebase

GradFix masks each task vector before applying it to a different pretrained base in a cross-base A to B transport setting. For each parameter, the sign of its task-vector component is compared to the gradient sign obtained on B's training data. Mismatches are either zeroed out in `normal` mode or forced to the gradient sign in `force` mode.

Pipeline:

```text
θ_A (source base)  +  tuned checkpoints  ->  task vectors Δ_i
                                                    |
gradient signs on B  -------------------->  GradFix mask
                                                    |
                                         masked_Δ_i per task
                                                    |
                                       Σ_i w_i · masked_Δ_i
                                                    |
                                    θ_B + α · composed_Δ  ->  eval
```

Multi-task run over all 8 datasets:

```bash
python -m merge_and_rebase.eval.vision_rebase \
  --config configs/vision8_gradfix_rebase.json
```

`configs/vision8_gradfix_rebase.json` sets `"tasks": "all"`, includes all 8 tuned checkpoints, and sweeps alpha from 0.0 to 2.0.

Single-task example, such as Cars:

```bash
python -m merge_and_rebase.eval.vision_rebase \
  --config configs/vision_gradfix_single_task.json
```

`configs/vision_gradfix_single_task.json` sets `"tasks": "Cars"` and lists only the Cars checkpoint in `tuned_ckpts`.

You can also select a subset of tasks:

```bash
python -m merge_and_rebase.eval.vision_rebase \
  --config configs/vision8_gradfix_rebase.json \
  --tasks Cars,DTD
```

CLI flags always override config values.

Key config fields:

| Field | Description | Default |
|-------|-------------|---------|
| `source_clip_model/pretrained` | Base model A, relative to which task vectors are defined | `ViT-B-32 / openai` |
| `target_clip_model/pretrained` | Base model B, on which rebased vectors are applied | `ViT-B-32 / laion2b_s34b_b79k` |
| `tasks` | `"all"` or comma-separated task names such as `"Cars"` or `"Cars,DTD"` | `"all"` |
| `mask_mode` | `"normal"` to zero disagreeing signs, or `"force"` to override them | `"normal"` |
| `vote` | Gradient sign voting mode: `"mean"` or `"max"` | `"mean"` |
| `alpha_search` | Enable a linear alpha sweep | `false` |
| `alpha_selection` | `"shared"` for one best alpha across tasks, or `"per_task"` for one best alpha per task | `"shared"` |
| `alpha` | Fixed scaling factor when `alpha_search` is disabled | `1.0` |
| `weights` | Per-task composition weights, with `null` meaning uniform weights | `null` |
| `tuned_ckpts` | Mapping from task name to checkpoint path | — |

---

## Citation

If you use this repository in your work, please cite it as:

```bibtex
@software{panariello2026merge_and_rebase,
  author = {Panariello, Aniello and Rinaldi, Filippo},
  title = {merge-and-rebase},
  year = {2026},
  url = {https://github.com/apanariello4/merge-and-rebase},
  version = {0.1.0}
}
```

GitHub citation metadata is also available in `CITATION.cff`.
