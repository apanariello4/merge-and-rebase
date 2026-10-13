# Getting Started

## Environment

The project requires Python 3.12 or later. `uv` is the supported environment manager.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[data,dev]"
```

Install only the package with `uv pip install -e .`. The `data` extra adds Hugging Face dataset and Hub support; `dev` adds test and lint tooling.

## Repository Layout

- `configs/`: runnable evaluation and rebasin configurations.
- `src/merge_and_rebase/finetune/configs/`: vision and text fine-tuning configurations.
- `src/merge_and_rebase/merge/`: checkpoint merge methods and subspaces.
- `src/merge_and_rebase/rebase/`: task-vector transport methods.
- `src/merge_and_rebase/eval/`: vision, language-model, connectivity, and rebasin entrypoints.
- `artifacts/manifest.json`: released checkpoint bundles.

## Common Commands

```bash
# Run source and test quality checks.
uv run ruff check src tests

# Run the test suite.
uv run pytest -q

# Preview the documentation site locally.
uv sync --extra docs
uv run mkdocs serve

# Fine-tune a Vision8 subset.
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --suite vision8
```

Continue with [artifacts and checkpoints](artifacts.md), [fine-tuning](fine-tuning.md), or [merging](merging.md).
