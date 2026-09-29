# merge-and-rebase

![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![OpenCLIP](https://img.shields.io/badge/backbone-OpenCLIP-1F6FEB)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-222?logo=githubpages&logoColor=white)](https://apanariello4.github.io/merge-and-rebase/)
[![Docs deployment](https://github.com/apanariello4/merge-and-rebase/actions/workflows/docs.yml/badge.svg)](https://apanariello4.github.io/merge-and-rebase/)

`merge-and-rebase` is a research codebase for fine-tuning, model merging, task-vector transport, and evaluation across vision and text models.

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[data,dev]"
```

Use `uv pip install -e .` when dataset and development dependencies are not needed.

## Quick Start

Run the released Vision8 Task Arithmetic example. Checkpoints referenced through `hf-hub:` are downloaded automatically.

```bash
python -m merge_and_rebase.eval.vision_merge \
  --config configs/vision8_task_arithmetic_hf_release.json
```

## Official Implementations

This repository hosts the official implementation of:

| Method | Paper | Venue | Entry point | Configs |
|---|---|---|---|---|
| **GradFix** | [Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](https://arxiv.org/abs/2510.09658) | ICLR 2026 | `merge_and_rebase.eval.vision_rebase` | [`configs/vision8_gradfix_hf.json`](configs/vision8_gradfix_hf.json) |
| **TAK** | [Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](https://arxiv.org/abs/2602.17385) | ICLR 2026 | `merge_and_rebase.finetune.train_vision` | [`finetune/configs/TAK/`](src/merge_and_rebase/finetune/configs/TAK) |
| **DELTA** | [Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](https://arxiv.org/abs/2605.18993) | ICML 2026 | `merge_and_rebase.finetune.train_vision` | [`finetune/configs/DELTA/`](src/merge_and_rebase/finetune/configs/DELTA) (`Full FT/`, `LoRA/`) |
| **Theseus** | [Transporting Task Vectors across Different Architectures without Training](https://arxiv.org/abs/2602.12952) | ICML 2026 | `merge_and_rebase.eval.vision_rebase` | [`configs/vision8_theseus_all.json`](configs/vision8_theseus_all.json) |

Example launches:

```bash
# GradFix / Theseus: transport task vectors to a new base model
python -m merge_and_rebase.eval.vision_rebase --config configs/vision8_gradfix_hf.json
python -m merge_and_rebase.eval.vision_rebase --config configs/vision8_theseus_all.json

# TAK / DELTA: fine-tune task vectors (configs named <method>-<backbone>-<suite>.yaml)
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/TAK/vision-tak-kfac-vitb16-vision8.yaml
python -m merge_and_rebase.finetune.train_vision \
  --vision-config "src/merge_and_rebase/finetune/configs/DELTA/Full FT/vision-delta-vitb16-vision8.yaml"
```

Fine-tuned TAK and DELTA checkpoints are then merged with `merge_and_rebase.eval.vision_merge` (see [Merging](docs/merging.md)).

## Documentation

- [Documentation site](https://apanariello4.github.io/merge-and-rebase/): rendered guides, tutorials, and API reference.
- [Getting started](docs/getting-started.md): environments, repository layout, and common commands.
- [Tutorials](docs/tutorials/reproduce-vision8.md): end-to-end runs with released and local checkpoints.
- [Concepts](docs/concepts.md): task vectors, bases, preparation, metrics, and compatibility.
- [Configuration reference](docs/configuration.md): merge, rebasin, post-merge, and fine-tuning fields.
- [Artifacts and checkpoints](docs/artifacts.md): released checkpoints, manifests, validation, and local checkpoints.
- [Fine-tuning](docs/fine-tuning.md): vision and text configurations, strategies, regularizers, and logging.
- [Merging](docs/merging.md): merge methods, evaluation, alpha search, and hyperparameter search.
- [Rebasin](docs/rebasin.md): transport methods and Vision rebasin configurations.
- [Methods reference](docs/methods.md): registered methods, configuration parameters, and source-level APIs.
- [Repository overview slides](docs/repo-overview-slides.md).

## Citation

```bibtex
@software{panariello2026merge_and_rebase,
  author = {Panariello, Aniello and Rinaldi, Filippo and Porrello, Angelo and van de Weijer, Joost and Calderara, Simone},
  title = {Merge-and-Rebase: A Unified Framework and Evaluation Benchmark for Fine-Tuning, Model Merging, and Rebasin},
  year = {2026},
  url = {https://github.com/apanariello4/merge-and-rebase},
  version = {0.1.0}
}
```

GitHub citation metadata is available in `CITATION.cff`.
