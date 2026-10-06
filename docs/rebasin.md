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

When source and target depths differ, each method uses its own rule, configured under `block_extension_params`:

| Method | Default rule (`"depth_defaults": "method"`) |
|---|---|
| THESEUS (`theseus`, `theseus_gqa`) | BRACE block extension, `extension_strategy="interpolate_per_weight"`, `skip_correction=true` (`false` selectable) |
| BiCo | BiCo-paper discrete index match `i(j) = round(j (D_s-1)/(D_t-1))`; source base and fine-tuned models are reindexed and BiCo's statistics are collected on the reindexed stack |
| Ariadne | no prestep; blocks are paired by `ariadne_params.depth_pairing` (`relative` default, `spread_duplicate`) |

`block_extension_params.depth_rule` (`method_default` / `brace` / `discrete_index_match`) overrides the rule; the
legacy top-level `depth_alignment` is an alias. `"depth_defaults": "legacy"` reproduces the behaviour before the
per-method defaults. A depth-changing THESEUS/BiCo config that sets neither `depth_defaults` nor an explicit
choice stops with an error naming both fixes, so old configs never silently change meaning.

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
