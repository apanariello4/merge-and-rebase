# Rebasin

Rebasin moves a fine-tuned task vector from a source base model to a different target base model (different
width and/or depth). Two entrypoints share one per-task pipeline (`rebase/run_config.py`, `rebase/prestep.py`,
`rebase/orchestration.py`): resolve the config once → optional depth prestep → method → save → merge → alpha
search → evaluation.

```bash
python -m merge_and_rebase.eval.vision_rebase --config configs/examples/vision8_ariadne_b16_to_l14.json   # OpenCLIP
python -m merge_and_rebase.eval.llm_rebase    --config configs/examples/qwen2.5_0.5b_to_1.5b_ariadne.json  # HF decoders
```

## Methods

| Method | What it does | Vision | LLM (Llama, Qwen2/2.5, Qwen3) | Width change | Depth change |
|---|---|---|---|---|---|
| `ariadne` (alias `direct_residual`) | fits the target's residual-writing projections from paired activations; no parameter transport ([details](methods/ariadne.md)) | yes | yes | yes | yes, by `depth_pairing` |
| `theseus` | activation-aligned parameter transport | yes | yes | yes | via BRACE prestep |
| `theseus_gqa` | THESEUS with head-aware attention for GQA decoders (choose explicitly) | — | yes | yes | via BRACE prestep |
| `bico` | bidirectional coupling (activations + gradients) | yes | `bico` | yes | via BiCo discrete index match |
| `transfusion`, `gradfix`, `identity`, `orthogonal_shift` | other transports (see [methods](methods.md)) | yes | `gradfix`, `identity`, `orthogonal_shift` (same size) | — | — |

Mixture-of-experts decoders are rejected.

## Depth change

When source and target depths differ, each method handles the difference in its own way. Example configs state it
explicitly, so the config alone says what runs:

| Method | Config line | What it does to the source layers |
|---|---|---|
| THESEUS (`theseus`, `theseus_gqa`) | `depth_alignment: {rule: "interpolate_layers"}` | Inserts blended copies of source layers (each is the 50/50 weight average of a layer and the next one) until the source has the target's depth; no correction is fitted. 24 to 28 layers copies layers 0, 5, 11 and 17. |
| BiCo | `depth_alignment: {rule: "index_match"}` | Target layer `j` copies source layer `round(j (D_s-1)/(D_t-1))`, with no blending and no fit; BiCo's statistics are collected on the re-indexed source and fine-tuned models. 24 to 28 layers uses source layers 3, 9, 14 and 20 twice. |
| Ariadne | `ariadne_params: {depth_pairing: "relative"}` | Adds no layers; each target layer is fitted from a source layer chosen by `depth_pairing` (`relative` uses the same index formula as BiCo; `spread_duplicate` is the other option). |

Every run prints one line naming what it actually executes, for example
`Depth handling: theseus -> interpolate_layers (inserts blended copies of source layers [0, 5, 11, 17], 24 -> 28)`,
`... bico -> index_match (...)`, or `... none (source and target both have 24 layers)`.

`interpolate_layers` and `index_match` are plain names for the internal rules `brace` and `discrete_index_match`
(both internal names stay accepted). `interpolate_layers` means exactly "interpolated layers, no correction": combining
it with `skip_correction: false` or another `extension_strategy` is an error, and `brace` is the name to use for those
experimental variants. `depth_rule` (`block_extension_params.depth_rule`, or `depth_alignment.rule` in the nested
layout) takes `method_default`, `interpolate_layers`, `index_match`, `brace` or `discrete_index_match`. Setting
BRACE-only keys under `index_match` warns that they are ignored; it never switches the rule.

`"depth_defaults": "method"` (`depth_alignment.defaults`) is still accepted: it picks the rule per method as in the
table, which is why the examples no longer use it. `"depth_defaults": "legacy"` reproduces the depth rule from before
the per-method defaults (BRACE for BiCo); it does not bring back the ridge correction, which is off by default for
every method (see below). A depth-changing THESEUS/BiCo config that sets neither `depth_defaults` nor an explicit
rule stops with an error naming both fixes, so old configs never silently change meaning.

### BRACE correction (experimental)

`skip_correction` defaults to `true` on every path: BRACE then only inserts or merges layers and reads no calibration
data. `skip_correction: false` enables a ridge regression per inserted layer that makes its output match the original
layer's activations (the keys `ridge_identity`, `ridge_weight`, `n_cascade_iters`, `correction_scope`,
`target_shared_correction` and the other correction-only keys configure it). This correction is experimental: it needs
several times the memory of the default (about 175 GiB against 56 GiB host memory on Qwen2.5 0.5B to 1.5B) and logs
a warning when enabled. On the Qwen2.5 0.5B to 1.5B IFEval runs it never beat skipping it (one-seed results on 141
documents). The example and reference configs therefore do
not set any BRACE key. Configs from before the release that relied on the old default (correction on) must now set
`skip_correction: false` explicitly.

## Core fields

| Field | Meaning |
|---|---|
| `source_clip_model`/`source_clip_pretrained`, `target_clip_model`/`target_clip_pretrained` | vision source and target bases |
| `source_model_name_or_path`, `target_model_name_or_path`, `tuned_bodies` | LLM source/target bases and fine-tuned references |
| `tuned_ckpts`, `tasks` | vision fine-tuned checkpoints and task subset |
| `method`, `method_params` / `ariadne_params` | method and its parameters; always set `num_batches` and `seed` explicitly |
| `alpha` / `alpha_search` | fixed task-vector scale or an alpha sweep |
| `save_transported_tvs` | vision: `"if_dir_given"` (default: save only with `save_transported_tvs_dir`) or `"auto"` (always save next to the summary) |
| `calibration_dataset`, `calibration_include_target`, `calibration_n_sequences` | LLM calibration text (decoupled from the evaluation hold-out; defaults unchanged) |

Example configs: `configs/examples/`. Every accepted key, annotated with its role, default and status (main /
ablation / diagnostic / legacy), per method and modality: `configs/examples/reference/` (see its `README.md`).
