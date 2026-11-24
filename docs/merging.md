# Merging and Evaluation

Merge methods combine independently fine-tuned checkpoints relative to a shared base model. Vision evaluation uses full dataset test splits.

```bash
python -m merge_and_rebase.eval.vision_merge \
  --config configs/vision8_task_arithmetic_hf_release.json
```

The equivalent language-model entrypoint is `python -m merge_and_rebase.eval.llm_merge`; see [configs/llm_merge_llama3_8b_knots_hf.json](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/llm_merge_llama3_8b_knots_hf.json) for a starting configuration.

## DINOv2

The same entrypoint supports DINOv2 with `backbone.name: dinov2`. It merges only backbones and selects each unchanged classification head by task ID. See the [DINOv2 guide](dinov2.md) for vision8/14/20 presets, fixed-alpha weight averaging, normalized validation alpha search, inference, and shared CPU-preparation locks.

## Configuration

At minimum, a vision merge configuration specifies the base model, `method`, `tuned_ckpts`, and either a fixed `alpha` or search configuration.

```json
{
  "method": "ties_merge",
  "alpha": 0.3,
  "method_params": {"topk": 0.2},
  "tuned_ckpts": {"Cars": "path-or-checkpoint-reference"}
}
```

Use `text_features_source: "zero_shot"` to force zero-shot text features, or `"tuned_ckpt"` to require checkpoint-provided text features.

## Search

`hyperparam_search` supports deterministic sequential grids and Sobol exploration. Sequential search evaluates each method-parameter candidate and sweeps alpha within it; Sobol samples then refines a bounded continuous region.

```json
{
  "method": "ties_merge",
  "hyperparam_search": {
    "strategy": "sequential",
    "alpha": {"min": 0.0, "max": 1.0, "step": 0.1},
    "method_params": {"topk": [0.1, 0.2, 0.5, 1.0]}
  }
}
```

Method names and parameters are summarized in [methods.md](methods.md). For reproducible checkpoint acquisition, see [artifacts.md](artifacts.md).
