# Changelog

## Unreleased — `release/ariadne-main`

Published numbers: every refactor step was gated by golden SHA-256 pins (`tests/golden/`); intended changes are
listed under "Declared changes" in `tests/golden/HASHES.md`. Results of default configurations are byte-identical
unless a change below says otherwise.

### New
- **Ariadne** (formerly Direct Residual) is a registered rebase method, `rebase/methods/ariadne.py` (internals in
  `rebase/methods/_ariadne/`), with `direct_residual` as a pure alias. Preset `"preset": "ariadne"` selects the paper
  configuration (D-only, streaming, empirical-Bayes ridge); `num_batches` and `seed` are never part of the preset.
  See `docs/methods/ariadne.md`.
- Ariadne, THESEUS, `theseus_gqa` and BiCo run on HF decoders (Llama, Qwen2/2.5, Qwen3) through
  `python -m merge_and_rebase.eval.llm_rebase`; padding rows never enter any statistic; Ariadne materializes missing
  `down_proj` biases by default (`missing_bias="skip"` + `exact_form=false` keeps the stock architecture).
- BiCo depth change uses the BiCo-paper discrete index match (vision and LLM), statistics on the reindexed stack.
- Per-method depth defaults behind `depth_defaults: "legacy" | "method"`; `block_extension_params.depth_rule`.
- Ariadne `depth_pairing="spread_duplicate"`.
- LLM calibration options (default off): `text_columns`/`text_template`, `calibration_include_target`, top-level
  `calibration_dataset` decoupled from the evaluation hold-out, `calibration_n_sequences`; calibration provenance in
  the summary.
- `save_transported_tvs: "auto"`; example configs in `configs/examples/`.
- Annotated reference configs `configs/examples/reference/{vision,llm}/<method>.yaml` (Ariadne, THESEUS, `theseus_gqa`,
  BiCo): every accepted key with role, default and status, guarded by `tests/test_reference_configs.py`. The LLM example
  configs no longer carry `save_transported_tvs`, which is a vision-only key and was a no-op there.

### Changed
- Vision and LLM entrypoints are packages (`eval/vision_rebase/`, `eval/llm_rebase/`) on one shared per-task pipeline
  (`rebase/run_config.py`, `rebase/prestep.py`, `rebase/orchestration.py`); LLM runs per task (lower peak memory).
- BRACE block extension is one core with vision/decoder adapters in `rebase/block_extension/`.
- Shared transport helpers in `rebase/methods/_shared.py`; no method imports another method's private helpers.
- Diagnostic CLIs moved to `eval/diagnostics/` (`python -m merge_and_rebase.eval.diagnostics.<name>`).
- Run summaries: method label "Ariadne" for both spellings; additive keys `canonical_method`, `preset`,
  `depth_rule_resolved`, `ignored_block_extension_fields`, `save_policy` (only when set), `calibration_provenance`,
  `materialized_bias_keys` (LLM Ariadne).

### Behaviour changes (declared)
- A depth-changing THESEUS/BiCo config without `depth_defaults` or an explicit `skip_correction` /
  `depth_alignment` / `depth_rule` raises `ConfigMeaningChangedError` naming both fixes.
- Padding rows are removed from BRACE decoder correction statistics (changes decoder BRACE results).
- THESEUS/BiCo padding hardening: all-pad batches contribute no rows, mask mismatches raise, mean/cls pooling is
  masked; `seq_align="mean"` no longer crashes on decoders.
- Ariadne alignment diagnostics describe the configured alignment map; Procrustes rank and a non-unique-Q flag are
  reported per position.
- Fixes: `source_only` no longer crashes; target-architecture checkpoints with `auto_detect_ckpt_base=false` are
  refused instead of yielding an empty task vector; CPU fp32 base snapshots no longer alias the model;
  `independent_endpoint_average` without effect is rejected; unknown `merge_method` and wrong-length `weights` raise
  `ValueError`; the discrete-index protocol label; decoder `layer_types` truncated on shrink; `llm_merge` loads dense
  HF model directories as full state dicts; more tuned bodies than task names is an error.

### Retired / removed
- Ariadne `component_target="output_local" | "output_total"` (error: retired).
- Target-informed completion (`target_residual_completion`, `joint_blockwise_correction`, `direct_p1_correction`,
  LLM `direct_target` / `transport_residual` modes) — archived; the config keys raise "retired".
- One-off BRACE campaign entrypoints (swap/tv-swap/source-merge/diagnostics, `vision_lmc`, `transport_then_merge`) —
  archived. Dead cross-task/all-task LMC options are deprecated.
- Removed import paths: `merge_and_rebase.eval.direct_residual`, `eval.block_extension`, `eval.block_extension_llm`,
  `eval.target_informed_runtime`, `eval.target_residual_completion`, `eval.vision_rebase_*`, `eval.llm_common`,
  `eval.lm_harness_runner`, `eval.llm_eval_only` (use the package paths above).
- `depth_pairing="brace_ancestry"` (an intermediate name) raises "renamed to 'spread_duplicate'".
