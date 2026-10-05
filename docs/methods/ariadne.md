# Ariadne

Ariadne is the method of the paper. It transfers the effect of a fine-tuning run from a *source* model to a
*target* model that may differ in depth, width or block design, **without transporting any parameter**. It
was previously called *Direct Residual*; the legacy name `direct_residual` is still accepted
(see [Legacy names](#legacy-names-and-compatibility)).

Implementation: `merge_and_rebase.rebase.methods.ariadne` (registered rebase method `ariadne`).

## What Ariadne does

Given a source base model, its fine-tuned counterpart and an untouched target base model, Ariadne measures
what fine-tuning did to the source residual stream on a small calibration set, expresses that effect in the
coordinates of the target residual stream, and then fits the target's residual-writing projection so that it
reproduces that effect. The returned task vector contains **only the fitted projection update**; every other
target parameter is left unchanged (zero in the task vector).

For every target block `j` that has a paired source block `i(j)`:

1. **Block-boundary states.** Collect the residual-stream states after source block `i(j)` for the source
   base (`S_base`) and source fine-tuned (`S_ft`) models, and after target block `j` for the target base
   (`T_base`). Rows index calibration examples and tokens.
2. **Procrustes map.** Center `S_base` and `T_base` and solve the rectangular Procrustes problem

    ```
    Q_j = argmin_R || S~_base R - T~_base ||_F^2     (R a partial isometry, d_source x d_target)
    S~_base^T T~_base = U Sigma V^T,   Q_j = U V^T
    ```

3. **Desired effect in target coordinates.** The same map is applied to the fine-tuning effect:

    ```
    D_j = (S_ft - S_base) Q_j                         (n x d_target)
    ```

    `D_j` is the displacement the target must realize in its residual stream after block `j`. It depends
    only on the source activations up to an orthogonal change of basis, because Procrustes absorbs the
    inverse rotation.
4. **Ridge fit of the target projection.** For each fitted component `c` (in the main method, the MLP
   output projection), let `H` be the activations entering it, collected on the **untouched target base**,
   and `L` the fixed linear map from the component output to the residual stream (identity for decoders;
   a per-channel LayerScale scaling in some vision blocks). Solve for an additive weight update `dW` and bias
   update `beta`:

    ```
    min  || (H dW + 1 beta^T) L - D_j ||_F^2 + lambda ||dW||_F^2
    ```

    The exact solution is a Sylvester system `S dW G + lambda dW = B` with `S = H~^T H~`, `G = L L^T`,
    `B = H~^T D~ L^T`, solved in closed form by two eigendecompositions. Centering removes the intercept
    from the weight solve; `beta` is recovered afterwards.
5. **Task vector.** `tau_target = { dW_j, beta_j }` for all fitted positions and components. The adapted
   model is `target_base + alpha * tau_target`, with `alpha` the usual task-vector strength.

All positions are fitted **independently**: every `H` is captured from the untouched target base, nothing is
mounted during fitting, and the updates are assembled after all solves complete. No regression depends on
another's fitted parameters.

## Inputs

| Input | Role |
| --- | --- |
| Source base model | Reference state `S_base` and Procrustes source |
| Source fine-tuned model | Fine-tuned state `S_ft` (and hence the effect to transfer) |
| Target base model | Activations `H`, `T_base`, and the parameters that receive the update |
| Paired calibration data | The same inputs are fed to source and target so rows are paired |

Calibration data are consumed in `num_batches` batches (a per-experiment field, see below). For vision
models, if source and target token grids differ the source tokens are interpolated to the target grid before
rows are paired. For decoder models, each model's rows are masked by its own attention mask, so **padding
never enters any statistic**; if source and target rows cannot be paired after masking (tokenizer mismatch)
Ariadne raises an error instead of falling back to rows that include padding.


### Depth pairing

In the method, the target-to-source block map `pi` can be any (possibly partial) map. This implementation
provides two maps, selected with `depth_pairing`:

- `"relative"` (default): uniform nearest index, `i(j) = round( j (D_source - 1) / (D_target - 1) )`, with
  Python's round-half-to-even and `D_source`/`D_target` the numbers of blocks. Several target blocks may share a
  source block (extension) and some source blocks may go unused (reduction).
- `"spread_duplicate"`: every source block maps to its own target block in order; the extra target blocks are
  duplicates of source blocks chosen at evenly spread positions (the default BRACE spread schedule), and a
  reduction merges evenly spread groups of source blocks.

`reversed`, `shift_plus1` and `shift_minus1` are ablations only (see the option table).

## The main method

The paper configuration is selected with a named preset:

```json
{
  "method": "ariadne",
  "ariadne_params": {
    "preset": "ariadne",
    "num_batches": 10,
    "seed": 89
  }
}
```

The preset `"ariadne"` sets exactly three fields:

| Field | Value set by the preset |
| --- | --- |
| `components` | `["mlp.c_proj"]` (decoder: `mlp.down_proj`) |
| `activation_storage` | `"streaming"` |
| `ridge_estimator` | `"empirical_bayes"` |

Together with the dataclass defaults that are part of the main method (`component_target="block_boundary"`,
`alignment_map="polar"`, `alignment_row_weighting="uniform"`, `procrustes_source="activation"`,
`depth_pairing="relative"`, `exact_form=true`, `endpoint_construction="native_delta"`,
`residual_target="transported_delta"`, `tv_scaling="none"`), this is the method described in the paper:
block-boundary target, polar activation Procrustes, a single fitted residual-writing projection per block,
empirical-Bayes ridge, uniform relative depth pairing.

Rules for presets:

- Explicit keys in the same mapping override preset fields.
- Every field the preset does not list keeps its dataclass default.
- **`num_batches` and `seed` are per-experiment budgets and are never part of a preset.** State both in
  every config. The dataclass defaults (`num_batches=10`, `seed=89`) are historical and only apply if the
  keys are omitted; an example that omits them is not reproducible by reading the config alone.
- The preset name is recorded in the run summary; the resolved fields are recorded under `config`.
- Dataclass defaults are unchanged and historical (for example `ridge_estimator="fixed_relative"`,
  `activation_storage="resident"`, both projections), so configs without a preset keep reproducing past
  results.

### Empirical-Bayes ridge

With `ridge_estimator="empirical_bayes"` the ridge is set from the calibration statistics rather than tuned:

```
lambda = tr(S)/(n-1) * tr(G)/d_out
```

The second factor accounts for a non-identity output map `L` and equals 1 when `L = I`. In this mode
`ridge_relative` is not used.

## Options

`DirectResidualConfig` (the dataclass name predates the rename) is parsed strictly: unknown keys are
rejected. Status legend: **main** = part of the paper method; **ablation** = supported, not the main
method; **retired** = rejected with an error; **kept** = supported but not used in current experiments.
"Vision" marks options that are only meaningful for the vision pipeline.

| Field | Default | Allowed values | Status |
| --- | --- | --- | --- |
| `preset` | none | `"ariadne"` | main |
| `components` | `["attn.out_proj", "mlp.c_proj"]` | non-empty, repeat-free subset of `attn.out_proj`, `mlp.c_proj` | main: `["mlp.c_proj"]` (set by preset); `attn.out_proj` is an ablation |
| `ridge_estimator` | `"fixed_relative"` | `fixed_relative`, `empirical_bayes`, `none` | main: `empirical_bayes` (preset); `fixed_relative` is the historical default; `none` is an exact solve that errors if the system is ill-conditioned |
| `ridge_relative` | `0.01` | finite, > 0 | used by `fixed_relative` only |
| `strength` | `1.0` | finite, >= 0 | main (task-vector strength; the per-task alpha search is run by the caller) |
| `num_batches` | `10` | positive integer | main; **set explicitly per experiment** |
| `seed` | `89` | integer | main; **set explicitly per experiment** |
| `exact_form` | `true` | bool | main: fit the exact residual contribution including the intercept; `false` is the first-order form |
| `missing_bias` | `"error"` | `error`, `materialize`, `skip` | main when the projection has no bias: `materialize` creates the bias key (needed by `exact_form`); `skip` requires `exact_form=false` |
| `component_target` | `"block_boundary"` | `block_boundary` | main. `output_local` and `output_total` are **retired** and raise an error |
| `alignment_map` | `"polar"` | `polar`, `ridge`, `random_isometry` | main: `polar`; others are ablations |
| `alignment_row_weighting` | `"uniform"` | `uniform`, `cls_balanced`, `delta_magnitude` | main: `uniform`; others are ablations (polar only; `ridge`/`random_isometry` require `uniform`) |
| `alignment_seed` | `0` | integer | ablation: seed base for `random_isometry` (per-position seed derived from it) |
| `procrustes_source` | `"activation"` | `activation`, `gradient` | main: `activation`. `gradient` (vision only) fits `Q_j` on boundary gradients of a zero-shot contrastive loss; ablation |
| `residual_target` | `"transported_delta"` | `transported_delta`, `transported_endpoint` | main: `transported_delta`, `D_j = (S_ft - S_base) Q_j`. `transported_endpoint` is an ablation that differs by the Procrustes residual (activation Procrustes only) |
| `depth_pairing` | `"relative"` | `relative`, `spread_duplicate`, `reversed`, `shift_plus1`, `shift_minus1` | main: `relative`; the others are ablations that change both `D_j` and `Q_j` |
| `streaming_fingerprint_tol` | `1e-9` | positive float | streaming check that the target model was not mutated between capture passes; serialized only when non-default |
| `copy_shape_matching_source_deltas` | `false` | bool | ablation: add shape-matching source deltas of non-fitted parameters to the task vector; serialized only when true |
| `activation_storage` | `"resident"` | `resident`, `streaming` | main: `streaming` (preset). See [storage paths](#storage-paths) |
| `streaming_position_chunk` | `null` | `null` or positive integer | streaming only: fit target positions in chunks (one capture sweep each) to bound host memory |
| `block_split` | `"none"` | `none`, `backfit`, `joint` | main: `none`. `joint` is an ablation (closed-form joint ridge over attention and MLP projections; equals `none` for one component). `backfit` is kept but not used in current experiments |
| `backfit_max_iters` | `20` | positive integer | `backfit` only |
| `backfit_tol` | `1e-4` | finite, > 0 | `backfit` only |
| `merge_mode` | `"per_task_then_merge"` | `per_task_then_merge`, `merge_in_source_then_fit` | main: `per_task_then_merge`. The other merges task deltas on the source base and fits once (vision orchestration) |
| `endpoint_construction` | `"native_delta"` | `native_delta`, `sequential_source_endpoints`, `sequential_delta_on_synthesized_base` | main: `native_delta`. Sequential modes are ablations requiring `components=["mlp.c_proj"]`, resident storage, activation Procrustes, per-task fits, no TV scaling |
| `calibration_data` | `"task_local"` | `task_local`, `tiny_imagenet`, `vision8_mix` | vision only; `task_local` is the default, the others are ablations and need `procrustes_source="activation"` |
| `tv_scaling` | `"none"` | `none`, `global`, `per_block` | ablation: label-free rescaling of the task vector before the alpha search; not allowed with `block_split != none` |
| `tv_scaling_iters` | `3` | positive integer | `per_block` only |
| `cascade_order` | `"independent"` | `independent`, `bottom_top`, `top_bottom` | no-op: positions never cascade; kept for schema parity |
| `realization_diagnostics` | `false` | bool | analysis only: adds diagnostic fields to the per-component rows |
| `fidelity_holdout` | `false` | bool | diagnostic only, never read by a fit: measures reproduction of `D_j` on a disjoint held-out slice |
| `fidelity_holdout_batches` | `10` | positive integer | with `fidelity_holdout` |


Cross-field validation (all raise `ValueError` at parse time):

- `block_split` in {`backfit`, `joint`} needs `block_boundary` and residual-writing components only.
- `activation_storage="streaming"` needs `block_boundary` and `block_split="none"`.
- Non-default alignment options need `procrustes_source="activation"`, `residual_target="transported_delta"`.
- `procrustes_source="gradient"` and non-default `calibration_data` are mutually exclusive.
- Sequential endpoint modes need the restrictions listed in the table.

## Storage paths

| | `resident` | `streaming` |
| --- | --- | --- |
| Memory | Full per-batch activation banks in host RAM; grows with `num_batches` | Sufficient statistics accumulated per batch; host RAM is O(1) in `num_batches` |
| Supported endpoints | all (including `joint`, `backfit`, sequential endpoints) | `native_delta`, `block_split="none"` only |
| Default | yes (historical) | selected by the `ariadne` preset |

The two paths solve the same problem and agree to about 1e-7 relative error. They are **not bitwise
identical**: the Procrustes cross-covariance is accumulated in a different summation order (per-batch
accumulation versus a single concatenated pass), so `Q_j`, and everything downstream of it, differs at the
level of floating-point summation. Compare resident and streaming numbers only as matched ablations, not as
entries of one table, unless equality has been demonstrated for the run in question.

`streaming_position_chunk` trades extra capture sweeps for lower peak host memory in the streaming path.

## Diagnostics

Every fitted position reports a row in the run summary. Always present (both storage paths):

- `procrustes_rank`: numerical rank of the centered cross-covariance the map was solved from;
- `procrustes_min_dim`: `min(d_source, d_target)`, the rank for a full-rank cross-covariance;
- `procrustes_q_non_unique`: `true` when `rank < min_dim`, i.e. the polar factor `U V^T` is not unique. This
  is a **flag only**: the fitted `Q_j` is not changed, and a run-level warning lists the affected positions.
  It is most relevant for gradient Procrustes, where the cross-covariance can be rank deficient; random
  isometries are reported with their rank but are never flagged.

Alignment diagnostics (`compute_alignment_diagnostics`, streaming equivalent) **describe the map that was
actually configured** (polar, ridge or random isometry, with the configured row weights), not always the
polar map. They report, per position, the Procrustes residual norm and relative error, the norm of the
transported delta, the ratio between them, the in-range and out-of-range parts of the residual, and the mean
offset. They never feed back into a fit and are computed outside the timed fit phase.

Realization diagnostics (`realization_diagnostics`) and held-out fidelity (`fidelity_holdout`) are opt-in and
analysis-only. Summaries also carry the git commit and dirty fingerprint, the seed, and calibration-sample
identifiers, so that transported-delta and task-vector hashes can be compared between runs.

## Legacy names and compatibility

| Legacy | Current |
| --- | --- |
| `"method": "direct_residual"` | `"method": "ariadne"` (registry alias; same object) |
| `direct_residual_params` | `ariadne_params` (alias of the params block) |

- Specifying both `direct_residual_params` and `ariadne_params` in one config is an error.
- The old import path `merge_and_rebase.eval.direct_residual` was removed; import from
  `merge_and_rebase.rebase.methods.ariadne` (public API) — internal stage modules live in
  `merge_and_rebase.rebase.methods._ariadne`.
- The config dataclass keeps its historical name `DirectResidualConfig` and the parser
  `parse_direct_residual_config`; they are exported from the new package.
- The `ariadne` preset was added without changing any dataclass default, so existing configs and results are
  unaffected.

## Decoder (LLM) support

Ariadne runs on Hugging Face decoders (Llama, Qwen2/2.5, Qwen3) through `python -m merge_and_rebase.eval.llm_rebase`
with `"method": "ariadne"`. Absent keys of `ariadne_params` get decoder defaults (the dataclass defaults are not
changed):

- **Component:** the residual-writing MLP projection only (`mlp.c_proj` maps to `mlp.down_proj` through the model
  family adapter).
- **Storage:** `streaming`. **Ridge:** `empirical_bayes`.
- **Bias:** `exact_form=true` with `missing_bias="materialize"`: a zero `down_proj.bias` is added once to the
  target model and base state, and the task vector carries its fitted value. To keep the stock architecture use
  `missing_bias="skip"` with `exact_form=false` (first-order, intercept-free fit).
  **Deviation from the paper:** the paper states that a bias-free component is fitted with `beta = 0` on
  uncentered `H` and `D`, which is the `missing_bias="skip"`, `exact_form=false` configuration. The decoder
  default deliberately materializes and fits the bias instead; select the paper's formulation explicitly when
  reproducing it.
- **Calibration:** text from the LLM calibration loader; padding rows are removed with each model's attention
  mask before any statistic is accumulated; an unpairable mask (e.g. tokenizer mismatch) is an error.
- **Task vector:** only the fit. `copy_shape_matching_source_deltas` (default off) is an ablation that adds
  shape-matching source deltas of the other parameters.
- **Depth:** no structural prestep; Ariadne pairs blocks with `depth_pairing`.
- **Rejected on decoders:** `procrustes_source="gradient"`, `tv_scaling`, `fidelity_holdout`, `block_split`
  other than `none`, weighted alignment rows, sequential endpoints, non-default `calibration_data`,
  `merge_mode="merge_in_source_then_fit"`, `transport_delta_source`/`delta_norm_match`. Mixture-of-experts
  decoders fail fast.

## Reproducibility notes

- Fix `num_batches` and `seed` in every config; they define the calibration sample.
- Run directories are never overwritten; commit before launching any run whose numbers may reach a table.
- Save the transported task vectors of every new run so that delta hashes can be compared later.
