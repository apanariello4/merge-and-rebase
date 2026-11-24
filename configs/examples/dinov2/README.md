# DINOv2 classification examples

See the [DINOv2 workflow guide](../../../docs/dinov2.md) for training modes, split policy, checkpoint formats, metric definitions, model loading, and supported options.

Install with `uv sync --extra data --extra dev`, then run from the repository root:

```bash
# Full backbone + classifier fine-tuning on vision8.
uv run python -m merge_and_rebase.finetune.train_vision --vision-config configs/examples/dinov2/vision8_full.yaml

# Train just a linear head on the original, frozen pretrained backbone.
uv run python -m merge_and_rebase.finetune.train_vision --vision-config configs/examples/dinov2/vision8_linear_probe.yaml

# Fit the head, freeze it, then fine-tune the backbone.
uv run python -m merge_and_rebase.finetune.train_vision --vision-config configs/examples/dinov2/vision8_classifier_then_backbone.yaml

# Merge vision8 joint checkpoints; select alpha on normalized validation accuracy.
uv run python -m merge_and_rebase.eval.vision_merge --config configs/examples/dinov2/vision8_merge.json
```

## Vision14 and vision20

The three `vision20_<strategy>.yaml` presets cover all twenty tasks. Use `--suite vision14` with one of these presets to train only vision14, or `--datasets MNIST` for a single-task run. The output directory stays the one in the YAML.

```bash
uv run python -m merge_and_rebase.finetune.train_vision --vision-config configs/examples/dinov2/vision20_full.yaml --suite vision14
uv run python -m merge_and_rebase.finetune.train_vision --vision-config configs/examples/dinov2/vision20_classifier_then_backbone.yaml
uv run python -m merge_and_rebase.eval.vision_merge --config configs/examples/dinov2/vision14_merge.json
uv run python -m merge_and_rebase.eval.vision_merge --config configs/examples/dinov2/vision20_classifier_then_backbone_merge.json
```

Each suite has `vision<N>_merge.json` for joint checkpoints and `vision<N>_classifier_then_backbone_merge.json` for the two-stage checkpoints. The vision14/20 merge presets reference the `checkpoints/dinov2/vision20/<strategy>/` training directories. All merge presets use a validation alpha grid from 0 through 2 in steps of 0.1, then evaluate the selected model on held-out test data. Original task heads are retained.

Change `common.backbone.model_id` for another standard DINOv2 size, and optionally pin `revision`. The presets use FP32 and effective batch size 128 (32 images, four accumulation steps). Learning rates and epoch budgets are starting configurations. Set the microbatch to 16 and accumulation to 8 for lower GPU memory usage at the same effective batch size.
