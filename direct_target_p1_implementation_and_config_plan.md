# Direct-target P1: implementation and configuration plan

## Objective

Evaluate whether a source fine-tuning effect can be reconstructed directly in
the native target model, without fitting or applying a THESEUS/BiCo parameter
transport.  The returned task vector is made solely from target-space residual
projection corrections.

The current runner keeps `method: "theseus"` as a registry and BRACE-routing
gate.  In `target_residual_completion.mode: "direct_target"`, it must not fit
or call `method.transport`; the method name is not evidence of transport.

## Implementation delivered

1. **Transport-free branch.** `vision_rebase.py` sets `prepared = None` and
   `transported_delta = {}` for direct-target P1.  It then invokes
   `complete_residuals_direct`, whose fitted target corrections are the whole
   task vector.  `gamma=0` is therefore exactly the native target base.

2. **Layout-only structural preprocessing.** Direct depth conversion needs a
   realized ancestry layout but not source-side ARIADNE activation collection,
   ridge fitting, or correction.  `skip_correction: true` is now valid with
   direct-target P1 and is reported as `direct_target_layout_only`.

3. **Shrink.** Decoder reduction publishes `build_reduction_layout(...)`.
   Direct shrink addresses every final target position (`target_scope: "all"`)
   using each collapsed span's terminal source boundary.  It rejects
   `target_scope: "inserted"` and interpolated trajectories before activation
   capture.

4. **Same architecture.** A depth-preserving direct path creates an identity
   layout over all target blocks, captures paired source/target references,
   skips parameter transport, and fits the target projections directly.  No
   source structural operation is run.

5. **Observability.** Direct runs print `Direct-target P1 (parameter transport
   skipped)` and record near-zero `transport_seconds`; generic labels have been
   clarified so they cannot be mistaken for an executed THESEUS transport.

## Configuration recipes

All direct runs require:

```json
"method": "theseus",
"block_extension_enabled": true,
"block_extension_params": {
  "n_batches_act": 10,
  "calibration_split": "val",
  "skip_correction": true,
  "lmc_mode": "independent",
  "target_residual_completion": {
    "enabled": true,
    "mode": "direct_target",
    "component": "c_proj.weight",
    "components": ["mlp.c_proj"],
    "ridge_relative": 0.1,
    "strength": 1.0,
    "num_batches": 10,
    "exact_form": true,
    "missing_bias": "error",
    "cascade_order": "bottom_top"
  }
},
"alpha_min": 0.0,
"alpha_max": 5.0,
"alpha_step": 0.1
```

### B/16 -> L/14 extension

Set `target_layers_total: 24`, use an extension strategy such as
`duplicate_per_weight`, and set:

```json
"target_scope": "inserted",
"target_trajectory": "step"
```

The layout contains the newly inserted target blocks.  The fitted corrections
are restricted to those positions.

### L/14 -> B/16 shrink

Set `target_layers_total: 12`, use the reduction strategy
`interpolate_per_weight`, and set:

```json
"target_scope": "all",
"target_trajectory": "step"
```

Every final B/16 block is a collapsed source span, so inserted-only and
interpolate semantics are invalid and deliberately rejected.

### B/16 -> B/16 cross-dataset / same architecture

Set `target_layers_total: 12` and:

```json
"target_scope": "all",
"target_trajectory": "step"
```

This uses the identity layout: target position `j` reproduces the source
fine-tuning effect at source position `j` in target coordinates.

## Acceptance checks for every run

1. The resolved config contains `mode: "direct_target"` and
   `skip_correction: true`.
2. The log says `Direct-target P1 ... parameter transport skipped`.
3. `transport_timings.<task>.transport_seconds` is effectively zero; no
   THESEUS fit/covariance diagnostics are emitted.
4. The summary contains `target_residual_completion` diagnostics for every
   intended target position.
5. The alpha curve covers 51 points from `0.0` through `5.0`, and alpha zero
   reproduces the native target base.

## Recommended experiment order

1. Run same-architecture direct target to validate cross-pretraining transfer
   without a depth operation.
2. Run extension direct target to validate inserted-position synthesis.
3. Run shrink direct target to validate span-end ancestry and all-position
   synthesis.
4. Compare each with a separately labeled historical THESEUS baseline; do not
   reuse a generic `method` field as the experimental-arm label.
