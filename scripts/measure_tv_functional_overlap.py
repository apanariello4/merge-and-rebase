"""Measure target-feature overlap and non-additivity of transported task deltas."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch

from merge_and_rebase.data.vision_loaders import build_vision_loaders, load_hf_splits
from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier


def _load(directory: Path, task: str) -> dict[str, torch.Tensor]:
    return torch.load(directory / f"{task}_theseus_transported_native.pt", map_location="cpu", weights_only=True)


def _set_delta(
    state: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor] | None = None,
    first_scale: float = 1.0,
    second_scale: float = 1.0,
) -> None:
    with torch.inference_mode():
        for key, value in base.items():
            state[key].copy_(value.to(device=state[key].device) + first_scale * first.get(key, 0).to(device=state[key].device))
            if second is not None:
                state[key].add_(second_scale * second.get(key, 0).to(device=state[key].device))


def _reset(state: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> None:
    with torch.inference_mode():
        for key, value in base.items():
            state[key].copy_(value.to(device=state[key].device))


def _summary(values: dict[str, float]) -> dict[str, float]:
    first_norm = values["first_sq"] ** 0.5
    second_norm = values["second_sq"] ** 0.5
    additive_norm = values["additive_sq"] ** 0.5
    interaction_norm = values["interaction_sq"] ** 0.5
    return {
        "feature_delta_cosine": values["dot"] / (first_norm * second_norm),
        "first_feature_delta_norm": first_norm,
        "second_feature_delta_norm": second_norm,
        "additive_feature_delta_norm": additive_norm,
        "interaction_norm": interaction_norm,
        "relative_nonadditivity": interaction_norm / additive_norm,
        "images": int(values["images"]),
    }


def _norm(delta: dict[str, torch.Tensor]) -> float:
    return sum(float(value.float().square().sum()) for value in delta.values()) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--independent-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs=2, default=("DTD", "SVHN"))
    parser.add_argument("--batches-per-task", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--normalize-deltas", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite = SUITES["vision8"]
    clf = OpenClipClassifier.build(
        OpenClipBuildConfig(model_name="ViT-L-14", pretrained="datacomp_xl_s13b_b90k", device=args.device)
    )
    model = clf.model.eval()
    state = model.state_dict()
    deltas = {
        mode: (_load(directory, args.tasks[0]), _load(directory, args.tasks[1]))
        for mode, directory in (("shared", args.shared_dir), ("independent", args.independent_dir))
    }
    keys = set.intersection(*(set(delta) for pair in deltas.values() for delta in pair))
    base = {key: state[key].detach().cpu().clone() for key in keys}
    scales = {
        mode: (
            args.scale / _norm(first) if args.normalize_deltas else args.scale,
            args.scale / _norm(second) if args.normalize_deltas else args.scale,
        )
        for mode, (first, second) in deltas.items()
    }
    totals: dict[str, dict[str, dict[str, float]]] = {
        mode: defaultdict(lambda: defaultdict(float)) for mode in deltas
    }

    for task in args.tasks:
        hf_path, hf_config, split_map = suite.resolver(task)
        loaders = build_vision_loaders(
            hf_ds=load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values()))),
            hf_path=hf_path,
            preprocess=clf.preprocess,
            ft_epochs=1,
            split_map=split_map,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            val_fraction=0.1,
            seed=89,
        )
        for batch in islice(loaders.val, args.batches_per_task):
            images = batch[0].to(args.device)
            with torch.inference_mode():
                _reset(state, base)
                base_features = model.encode_image(images).float()
                for mode, (first, second) in deltas.items():
                    first_scale, second_scale = scales[mode]
                    _set_delta(state, base, first, first_scale=first_scale)
                    first_features = model.encode_image(images).float() - base_features
                    _set_delta(state, base, second, first_scale=second_scale)
                    second_features = model.encode_image(images).float() - base_features
                    _set_delta(state, base, first, second, first_scale=first_scale, second_scale=second_scale)
                    interaction = model.encode_image(images).float() - base_features - first_features - second_features
                    additive = first_features + second_features
                    for group in (task, "pooled"):
                        values = totals[mode][group]
                        values["first_sq"] += float(first_features.square().sum())
                        values["second_sq"] += float(second_features.square().sum())
                        values["dot"] += float((first_features * second_features).sum())
                        values["additive_sq"] += float(additive.square().sum())
                        values["interaction_sq"] += float(interaction.square().sum())
                        values["images"] += len(images)

    output = {
        "definition": "Feature deltas use target visual embeddings on the pooled DTD/SVHN validation probes. Interaction is f(base + DTD + SVHN) - f(base) - [f(base + DTD) - f(base)] - [f(base + SVHN) - f(base)].",
        "delta_normalization": "each delta has unit parameter norm" if args.normalize_deltas else f"raw deltas scaled by {args.scale}",
        "strategies": {
            mode: {group: _summary(values) for group, values in groups.items()}
            for mode, groups in totals.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
