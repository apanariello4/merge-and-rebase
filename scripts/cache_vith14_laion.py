#!/usr/bin/env python3
"""Download or offline-verify the exact ViT-H/14 LAION-2B OpenCLIP model."""

from __future__ import annotations

import argparse
import os


MODEL_NAME = "ViT-H-14"
PRETRAINED = "laion2b_s32b_b79k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Require all model files to be present locally and disable Hub network access.",
    )
    args = parser.parse_args()

    if args.verify:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device="cpu",
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Verified {MODEL_NAME}/{PRETRAINED}: "
        f"{parameter_count:,} parameters; HF_HOME={os.environ.get('HF_HOME', '<default>')}"
    )


if __name__ == "__main__":
    main()
