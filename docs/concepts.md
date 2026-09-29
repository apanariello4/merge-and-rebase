# Concepts

## Base Model and Task Vector

For a base checkpoint `theta_base` and a task-specific checkpoint `theta_task`, a task vector is the parameter update:

```text
delta_task = theta_task - theta_base
```

Merge methods combine task vectors or task checkpoints, then apply a scalar `alpha` to interpolate from the base. This requires compatible parameter keys and tensor shapes.

## Fine-Tuning for Mergeable Task Vectors

How a task vector is trained affects how well it merges. The fine-tuning workflow supports several regimes (details and launch commands in [fine-tuning.md](fine-tuning.md#fine-tuning-methods)):

- **Standard.** Non-linear fine-tuning. Task vectors can interfere when they are summed.
- **Linearized (NTK).** Training in the tangent space of θ₀, `f(x; θ₀) + J f(x; θ₀)·τ`. The model output is linear in τ, so it is also linear in the merge coefficients.
- **TAK.** Linearized training plus a K-FAC curvature penalty (`kfac_ggn`) computed on the *other* tasks. This discourages τ from moving along directions that matter for those tasks.
- **DELTA.** A non-linear student distilled from an online linearized teacher along the path θ₀ + α·τ, combined with an EK-FAC penalty on a generic reference dataset (ImageNet21KP). The aim is to get the merge-friendly behavior of linearized training while keeping a standard forward pass at inference time.

Checkpoints record the forward mode they were trained with. Evaluation with `forward_mode: auto` (the default in `vision_merge`) uses `linearized_ntk` only if *every* input checkpoint was trained linearized. TAK checkpoints are therefore evaluated linearized, and DELTA checkpoints are evaluated with the standard forward pass.

## Merging and Rebasin

Merging combines independently specialized updates defined relative to the same base. Rebasin transports one task vector from a source base coordinate system to a target base coordinate system. They share checkpoint and task-vector interfaces, but the current rebasin evaluator transports one task vector at a time; a fully configurable multi-task merge-then-transport pipeline is future work.

## Preparation and Application

Methods can implement two phases:

- `prepare`: expensive, alpha-independent work such as SVDs, activation alignment, or gradient collection.
- `apply`: cheap construction of a result for a chosen alpha.

The evaluation code uses this split to avoid repeating preparation during alpha search. Change method parameters or checkpoint inputs when you need a new prepared state.

## Metrics

Vision entrypoints report raw test accuracy. Some analyses additionally report normalized values relative to a selected baseline; these are ratios, not percentages bounded by 100. Always label raw accuracy and normalization denominator separately.

## Checkpoint Compatibility

Local and hosted checkpoints are structurally validated by matching supported wrappers, normalizing common key prefixes, and checking tensor keys and shapes. Structural compatibility does not prove a checkpoint's claimed training data, split, seed, or base revision. Use the manifest's hashes and provenance metadata for released artifacts.
