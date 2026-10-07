# Reference configs

One annotated YAML per main rebase method and modality. Each file lists every key the code accepts, with a comment
giving the key's role, allowed values and status.

| file | method | setup |
|---|---|---|
| `vision/ariadne.yaml` | Ariadne (preset `ariadne`) | CLIP ViT-B/16 → ViT-L/14, vision8 |
| `vision/theseus.yaml` | THESEUS + BRACE depth prestep | same |
| `vision/bico.yaml` | BiCo + discrete index match | same |
| `llm/ariadne.yaml` | Ariadne (preset `ariadne`) | Qwen2.5-0.5B → 1.5B, ifeval |
| `llm/theseus.yaml` | THESEUS + BRACE depth prestep | same |
| `llm/theseus_gqa.yaml` | THESEUS with head-aware GQA maps | same |
| `llm/bico.yaml` | BiCo + discrete index match | same |

```bash
python -m merge_and_rebase.eval.vision_rebase --config configs/examples/reference/vision/theseus.yaml
python -m merge_and_rebase.eval.llm_rebase    --config configs/examples/reference/llm/ariadne.yaml
```

Configs are read with `yaml.safe_load`, so `.yaml` works wherever `.json` does.

The files use the canonical nested names (`method: {name, params}`, `depth_alignment: {rule, brace}`, `models`,
`merge`, `alpha`, `save`, `load`), defined in `src/merge_and_rebase/rebase/config_schema.py` together with the
table of the legacy flat names (`method_params`, `block_extension_params`, `alpha_search`, ...). Legacy names still
work: the entrypoints accept them, warn once, and record them in the run metadata (`legacy_config_keys`).

## How to read a file

- **Active lines** are the recommended setup. They are identical to the JSON twin in `configs/examples/`, and the
  `llm/theseus_gqa.yaml` twin is the THESEUS JSON with only `method.name` changed.
- **Commented lines** `# key: value  # ...` list every other accepted key at the value that reproduces the active run.
  Uncommenting one line therefore changes exactly one factor, which is one ablation.
- **Tags:**
  - `[main]`: part of the method as published.
  - `[ablation]`: alternative arm.
  - `[diagnostic]`: extra measurement that never changes the task vector.
  - `[legacy]`: alias kept for old configs.
  - `[deprecated]`: emits a warning.
  - `[BRACE-only]`: inert under `depth_alignment.rule: discrete_index_match`, the BiCo default, and warns when set.
  - `[vision-only]`: the decoder ignores the key, or rejects a non-default value.
  - `[default: method | preset | decoder | stage]`: the value comes from that layer, not from the dataclass default.
- `num_batches` and `seed` are always stated explicitly, and no default stands in for them. On LLMs the top-level
  `seed` is not forwarded to THESEUS/BiCo, so set `method.params.seed`.

`tests/test_reference_configs.py` keeps these files honest. For each file it checks that:
- it resolves through the real resolvers without warnings and equals its JSON twin;
- it lists every field of `BlockExtensionConfig` (under `depth_alignment.brace`), `DirectResidualConfig` and the
  THESEUS/BiCo `prepare` keywords (under `method.params`), with the true defaults (legacy aliases excluded);
- every listed top-level key is read by the entrypoint;
- uncommenting any one line still resolves to the same depth plan.

## Known gaps (not changed by these files)

- Unknown top-level keys are silently ignored (unknown THESEUS/BiCo `method.params` keys and unknown keys inside the
  canonical blocks raise). A top-level typo therefore runs the default. Diff the run summary's resolved config
  against the file.
- `--alpha-early-stop` (both CLIs) and the LLM `--prealign-*` flags are parsed but never read.
- The default top-level `seed` is 42 on vision and 0 on LLMs.
