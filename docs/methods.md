# Methods Reference

All merge methods accept a common base checkpoint, fine-tuned checkpoints, optional task weights, and `alpha`. Expensive work belongs in `prepare`; `apply` then evaluates alpha candidates without rebuilding the method state. Put method-specific settings in the configuration's `method_params` object. Bounds below are enforced unless marked as an option or recommendation.

## Merge Methods

| Name | Class | Summary | Key `method_params` |
|---|---|---|---|
| `weighted_average` | `WeightedAverageMerge` | Averages checkpoints, then interpolates from the base. | `normalize`. |
| `task_arithmetic` | `TaskArithmeticMerge` | Weighted sum of task vectors. | None. |
| `ties_merge` | `TIESMerge` | Prunes task vectors and resolves conflicting signs. | `topk`, `merging_type`, `low_memory`. |
| `dare_merge` | `DAREMerge` | Randomly sparsifies each task vector before aggregation. | `drop_rate`, `seed`, `rescale`, `low_memory`. |
| `tsv_merge` | `TSVMerge` | Merges matrix updates through truncated singular-vector components. | `sv_reduction`, `max_rank`, `svd_dtype`, `vector_1d_merge`. |
| `isoc_merge` | `IsoCMerge` | Isotropizes the weighted matrix update. | `vector_1d_merge`. |
| `isocts_merge` | `IsoCTSMerge` | Separates common and task-specific matrix subspaces. | `common_space_fraction`, `vector_1d_merge`. |
| `cart_merge` | `CARTMerge` | Applies low-rank Cartesian task-vector composition. | `pruning_rank`, `scaling_coeffs`. |
| `pcb` / `pcb_merge` | `PCBMerge` | Balances flattened task vectors with intra/inter-task signals. | `clamp_min_ratio`, `clamp_max_ratio`, `att_ratio`, `lam`. |
| `actmerge` | `ActMerge` | Uses task-vector activation geometry for dense matrices. | Method-specific matrix options. |
| `dc_merge` | `DCMerge` | Dense task-delta composition. | Method-specific composition options. |
| `wudi` / `wudi_merge` | `WUDIMerge` | Solves a per-matrix WUDI objective. | `vector_1d_merge`. |

Matrix-oriented methods use their method-specific rule for 2D tensors. `vector_1d_merge: "average"` enables averaged 1D deltas where supported; the default is typically zero task-vector deltas.

### Weighted Average

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `normalize` | `"sumw"` | `"sumw"`, `"n"` | Divide the weighted checkpoint sum by the weight sum or by the number of checkpoints. |

### Task Arithmetic

Task Arithmetic has no method-specific parameters. It adds weighted task vectors, then applies the common `alpha`.

### TIES

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `topk` | `1.0` | Fraction `(0, 1]` or percentage `(0, 100]`; values at least `1` retain all entries | Per-task magnitude fraction retained before sign resolution. |
| `merging_type` | `"mean"` | `"mean"`, `"sum"`, `"max"` | Aggregation rule for entries whose sign agrees with the resolved sign. |
| `low_memory` | `false` | Boolean | Process tensors incrementally instead of materializing the flattened task-vector bank. |

### DARE

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `drop_rate` | `0.9` | `[0, 1)` | Fraction of task-vector entries removed. `p` is an alias; `keep_ratio` maps to `1 - drop_rate`. |
| `seed` | `null` | Integer or `null` | Random-mask seed. Set it to make sparsification reproducible. |
| `rescale` | `true` | Boolean | Divide retained entries by their keep probability to preserve the expected update. |
| `low_memory` | `false` | Boolean | Process tensors incrementally instead of materializing the flattened task-vector bank. |

### TSV

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `vector_1d_merge` | `"zero"` | `"zero"`, `"average"` | Keep one-dimensional tensors at the base value or average their deltas. |
| `sv_reduction` | `1 / n_tasks` | `(0, 1]` | Fraction of available singular components retained per matrix. |
| `max_rank` | `null` | Positive integer or `null` | Optional upper bound on retained rank after `sv_reduction`. |
| `svd_dtype` | `"float64"` | `"float32"`/`"fp32"`, `"float64"`/`"fp64"` | Precision used for the singular-value decomposition. |
| `accum_dtype` | `"float32"` | Standard Torch floating dtype | Precision used while accumulating the merged direction. |
| `progress` | `true` | Boolean | Show per-tensor progress. |
| `low_memory`, `key_batch_size` | `false`, `null` | Boolean; non-negative integer or `null` | Accepted for compatibility; TSV now always processes one key at a time, so these do not change execution. |

### IsoC and IsoCTS

| Method | Parameter | Default | Range / values | Description |
|---|---|---:|---|---|
| IsoC | `vector_1d_merge` | `"zero"` | `"zero"`, `"average"` | Keep one-dimensional tensors at the base value or average their deltas. |
| IsoCTS | `common_space_fraction` | `0.8` | Recommended `[0, 1]` | Fraction of matrix rank reserved for the common subspace; the remainder is divided among tasks. |
| IsoCTS | `vector_1d_merge` | `"zero"` | `"zero"`, `"average"` | Keep one-dimensional tensors at the base value or average their deltas. |

### CART

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `pruning_rank` | `0.5` | Positive number | Values in `(0, 1]` retain that fraction of singular components; values above `1` request an absolute rank, clipped to the available rank. |
| `scaling_coeffs` | `0.5` | Number | Scales the summed low-rank residual before the common `alpha` is applied. |

### PCB

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `clamp_min_ratio` | `0.01` | `[0, 1)` | Lower quantile removed when clamping absolute task-vector entries. |
| `clamp_max_ratio` | `0.01` | `[0, 1)` and `clamp_min_ratio + clamp_max_ratio < 1` | Upper quantile removed when clamping absolute task-vector entries. |
| `att_ratio` | `0.05` | `(0, 1]` | Fraction used to derive the attention/balancing scale. |
| `lam` | `1.2` | Number | Multiplier applied to the task weights in the final aggregation. |

### ACTMerge

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `merge_all_2d` | Context-dependent | Boolean | Merge every floating 2D tensor with ACTMerge. By default this is enabled for PEFT subspaces and otherwise excludes embeddings, token heads, and text projection. |
| `non_matrix_merge` | `"average"` | `"zero"`, `"average"` | Fallback for non-matrix tensors. |
| `vector_1d_merge` | `non_matrix_merge` | `"zero"`, `"average"` | Explicit fallback for one-dimensional tensors. |
| `ridge` | `0.0` | `>= 0` | Tikhonov regularization in the matrix pseudoinverse. |
| `pinv_atol` | `0.0` | `>= 0` | Absolute cutoff for the pseudoinverse. |
| `pinv_rtol` | `null` | `null` or `>= 0` | Relative cutoff for the pseudoinverse. |
| `work_dtype` | `"float32"` | `"float32"`/`"fp32"`, `"float64"`/`"fp64"` | Precision used for matrix operations. |

### DC-Merge

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `vector_1d_merge` | `"zero"` | `"zero"`, `"average"` | Keep one-dimensional tensors at the base value or average their deltas. |
| `sv_reduction` | `1 / n_tasks` | `(0, 1]` | Fraction of singular components retained in each task delta. |
| `max_rank` | `null` | Positive integer or `null` | Optional upper bound on retained rank. |
| `svd_dtype` | `"float64"` | `"float32"`/`"fp32"`, `"float64"`/`"fp64"` | Precision used for decompositions. |
| `energy_smoothing` | `"average"` | `"none"`, `"average"`, `"linear"` | Smoothing rule applied to retained singular values. |
| `energy_smoothing_strength` | `1.0` | `[0, 1]` | Interpolation amount between original and smoothed singular values. |
| `mask_mode` | `"block"` | `"block"`, `"none"` | Preserve task-specific blocks in the core space or use the full core. |
| `whiten_eps` | `1e-6` | `> 0` | Numerical floor when whitening the coordinate bases. |
| `cover_merge_method` | `"task_arithmetic"` | `"task_arithmetic"`, `"weighted_average"`, `"ties_merge"`, `"wudi"`, `"dare_merge"`, `"pcb"`, `"cart_merge"` | Functional method used to merge the core matrices. |
| `cover_merge_params` | `{}` | Mapping | Parameters forwarded to `cover_merge_method`; see that method's table. |

### WUDI

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `vector_1d_merge` | `"average"` | `"zero"`, `"average"` | Fallback for one-dimensional tensors. |
| `solver` | `"closed_form"` | `"closed_form"`, `"gd"` | Use the analytic solution or optimize the WUDI objective with gradient descent. `variant` and `mode` are aliases. |
| `ridge` | `1e-8` | `>= 0` | Ridge regularization for the closed-form linear solve. |
| `work_dtype` | `"float32"` | `"float32"`/`"fp32"`, `"float64"`/`"fp64"` | Precision used for matrix operations. |
| `steps` | `300` | Positive integer; `gd` only | Number of Adam steps. `num_steps` and `iters` are aliases. |
| `lr` | `1e-5` | `> 0`; `gd` only | Adam learning rate. `learning_rate` is an alias. |
| `weight_decay` | `0.0` | `>= 0`; `gd` only | Adam weight decay. |

## Transport Methods

| Name | Class | Summary |
|---|---|---|
| `identity` | `IdentityTransport` | Applies compatible task-vector tensors unchanged. |
| `orthogonal_shift` | `OrthogonalShiftTransport` | Removes the component aligned with the base-model shift. |
| `gradfix` | `GradFixRebase` | Uses target-model gradient signs to mask or correct task-vector components. |
| `theseus` | `TheseusRebase` | Builds activation-aligned coordinate transforms for matrix updates. |
| `transfusion` | `TransFusionRebase` | Uses Transformer-aware weight permutations. |
| `bico` | `BiCoRebase` | Estimates bilinear input/output coordinate maps from activations and gradients. |

### Identity and Orthogonal Shift

Identity has no method-specific parameters. Orthogonal Shift accepts `beta` (default `1.0`, any number), which controls the amount of base-shift projection removed, and `eps` (default `1e-12`, non-negative numerical threshold). With `beta: 1.0`, each compatible tensor block is orthogonal to its source-to-target base shift.

### GradFix

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `mask_mode` | `"normal"` | `"normal"`, `"force"` | Select the regular sign mask or force its correction mode when applying the delta. |
| `vote` | `"mean"` | `"mean"`, `"majority"`, `"max"` | Aggregation rule for gradient signs collected across batches. |
| `device` | `"cuda"` | Torch device string | Device used when collecting gradient signs. |

### Theseus and BiCo

| Method | Parameter | Default | Range / values | Description |
|---|---|---:|---|---|
| Theseus / BiCo | `seq_align` | `"interpolate2d"` | Sequence-alignment mode accepted by the activation collector | Align source and target token sequences before estimating layer maps. |
| Theseus / BiCo | `center_acts` | `false` | Boolean | Center activation features before computing transforms. |
| Theseus / BiCo | `whiten_power` | `0.0` | `[0, 0.5]` | Strength of covariance whitening in transform estimation. |
| Theseus / BiCo | `whiten_eps` | `1e-6` | `> 0` | Numerical floor used by whitening. |
| Theseus / BiCo | `n_batches` | `null` | Positive integer or `null` | Number of batches used for statistics; `num_batches` is an alias. |
| Theseus / BiCo | `seed` | `0` | Integer | Controls deterministic batch sampling. |
| Theseus / BiCo | `batch_size` | `null` | Positive integer or `null` | Optional per-batch cap during statistic collection. |
| Theseus / BiCo | `patch_qkv` | `true` | Boolean | Split fused QKV blocks while calculating and applying transforms. `split_qkv` is an alias. |
| Theseus | `covariance_mode` | `"activations"` | `"activations"`, `"data_free"` | Estimate transforms from loaders or use data-free base-model covariance. |
| BiCo | `transform_granularity` | `"param"` | `"param"` | Granularity supported by the current implementation. |

### TransFusion

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `seed` | `42` | Integer | Identifies a reproducible permutation-cache entry. |
| `perm_cache_source`, `perm_cache_target` | `"source"`, `"target"` | Strings | Source and target tags used to name cached permutations. |
| `perm_cache_dir` | `"permutations"` | Path | Directory used to cache permutations. |
| `perm_cache_mode` | `"auto"` | `"auto"`, `"load"`, `"save"` | Load an existing cache entry, save a new one, or do both when possible. |
| `max_iter` | `100` | Positive integer | Maximum permutation-matcher iterations. |
| `layer_iteration_order` | `"random"` | Matcher ordering string | Order in which the permutation matcher visits layers. |
| `intra_head` | `true` | Boolean | Enable attention-head-internal permutations. |
| `sanity_check_functional_equivalent` | `false` | Boolean | Evaluate source zero-shot behavior before and after permutation. |
| `sanity_check_first_n_batches` | `null` | Positive integer or `null` | Optional batch cap for the sanity check. |

## Post-Merge Methods

| Name | Class | Summary |
|---|---|---|
| `adamerging` | `AdaMergingPostMerge` | Optimizes task or layer alpha values against an entropy objective. |
| `task_vector_finetune` | `TaskVectorFinetunePostMerge` | Fine-tunes each task delta after composition. |
| `merged_delta_finetune` | `MergedDeltaFinetunePostMerge` | Fine-tunes one merged visual delta. |
| `vision_head_probe` | `VisionHeadProbePostMerge` | Fine-tunes only final vision-head task-vector tensors. |

All current post-merge methods are vision-only and require `peft_subspace: "full"`. The optimizer settings below apply to every post-merge method unless noted otherwise.

| Parameter | Default | Range / values | Description |
|---|---:|---|---|
| `device` | Model device | Torch device string | Device used for optimization. |
| `steps` | `500` | Integer `>= 0` | Number of optimization updates. |
| `lr` | `1e-3` | `> 0` | Adam learning rate. |
| `beta1`, `beta2` | `0.9`, `0.999` | Adam beta values | First- and second-moment coefficients passed to `torch.optim.Adam`. |
| `weight_decay` | `0.0` | Number | Adam weight decay. |
| `log_every` | `25` | Integer | Emit progress every positive multiple; `0` disables logging. |
| `loss` | `"ce"` | `"ce"`, `"entropy"` | Vision training objective. |
| `entropy_temperature` | `1.0` | `> 0` | Temperature for the entropy objective. |
| `batches_per_task` | `2` | Positive integer | Batches drawn from each task per optimizer step. |
| `max_batches_per_task` | `null` | Positive integer or `null` | Optional cap on batches consumed per task. |

| Method | Additional parameter | Default | Range / values | Description |
|---|---|---:|---|---|
| AdaMerging | `alpha_mode` | `"task"` | `"task"`, `"layer"` | Learn one coefficient per task or per task/layer pair. |
| AdaMerging | `alpha_min`, `alpha_max` | `0.0`, `1.0` | Numbers with `alpha_min <= alpha_max` | Bounds applied to learned coefficients. |
| AdaMerging | `init_alpha` | `0.3` | Number | Initial value before bound parameterization. |
| Vision Head Probe | `init_alpha` | `0.3` | Number | Initial scale applied to task deltas before optimizing final head tensors. |

The source classes document method-specific behavior beside their implementations. The rendered API reference is generated directly from those class docstrings and signatures: [merge methods](api/merge-methods.md), [rebasin methods](api/rebase-methods.md), and [post-merge methods](api/postmerge-methods.md).
