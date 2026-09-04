# Independent/Shared diagnostic campaign (2026-08-31)

This campaign is restricted to the vision8 ViT-B/16 DataComp extension
12→24. It records endpoint correction maps and corrected endpoints for the
Independent versus Shared diagnostic. It does not change BRACE behavior and
does not save activation banks.

The primary config uses `interpolate_per_weight`, ten seeded validation batches,
batch size 16, and `lambda_id` values 0.1, 1, 10, and 150. The separate
`duplicate_per_weight` config is a lambda-150 companion solely for comparison
with the historical duplicate merge results; it must not be pooled with the
primary interpolation lambda trend.

Before submission, verify both `$HF_HOME` and `$HF_HOME/datasets` exist on the
compute node and perform a one-task smoke run. The launchers set
`TRANSFORMERS_OFFLINE=1`, refuse an existing `COMPLETED` marker, and require a
per-task manifest. Do not use `HF_DATASETS_OFFLINE=1`.

Expected invocation is the dedicated module
`merge_and_rebase.eval.vision_ind_shared_diagnostic` with `--config`,
`--tasks`, `--output-dir`, `--ridge-identity`, and `--diagnostic-root`.

The diagnostic root is dated and outside the repository result tables. Every
run should retain its config path/hash, source checkpoint paths/hashes, job ID,
and output manifest. Never delete or overwrite historical results.

Launchers (submission intentionally omitted):

* `scripts/run_ind_shared_diagnostic.slurm`
* `scripts/run_ind_shared_diagnostic_companion.slurm`
