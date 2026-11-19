"""Reference accuracies for the anchors round trip (source == target): the native expert and the expert restricted to
its ``mlp.c_proj`` deltas.

For a model ``M`` (open_clip name + pretrained) and its Vision8 experts, evaluates on the TEST split of every task the
run protocol uses (same zero-shot heads, loaders, ``seed``, ``val_fraction`` as ``eval.vision_rebase`` and
``conservation``):

- ``base``: the pretrained model;
- ``expert``: the full fine-tuned checkpoint (the native expert; also the 'ceiling_native_*' runs of the rebuttal);
- ``cproj_only``: base + the expert's deltas on ``visual.transformer.resblocks.*.mlp.c_proj.{weight,bias}`` ONLY
  (every other parameter at base): the best the synthesis step could do if it reproduced the expert's c_proj
  update exactly (an upper reference for ``synthesis`` in the round trip, not a baseline of another method family).

    python -m merge_and_rebase.eval.diagnostics.expert_cproj_restriction --model ViT-B-16 \
        --pretrained datacomp_xl_s13b_b90k --ckpt-dir <dir with <task>/best.pt> --tasks EuroSAT,GTSRB,DTD --out OUT.json

Eval only (no fit); one model on the GPU. Not covered by an automated test (needs the CLIP weights and datasets).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.utils import eval_task_top1
from merge_and_rebase.eval.vision_rebase.context import _build_task_context
from merge_and_rebase.eval.vision_rebase.pipeline import _owned_cpu_fp32
from merge_and_rebase.io.ckpt import load_ckpt, load_into_model
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

CPROJ_KEY = re.compile(r"(?:^|\.)visual\.transformer\.resblocks\.\d+\.mlp\.c_proj\.(?:weight|bias)$")


def cproj_restricted(base: dict[str, torch.Tensor], expert: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``base`` with the expert's ``mlp.c_proj`` weight / bias; every other tensor is the base's."""
    out = dict(base)
    hit = [k for k in expert if CPROJ_KEY.search(k) and k in base]
    if not hit:
        raise KeyError("the expert checkpoint holds no visual.transformer.resblocks.*.mlp.c_proj tensors")
    for k in hit:
        out[k] = expert[k].to(dtype=base[k].dtype, device=base[k].device)
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tasks", default="EuroSAT,GTSRB,DTD")
    ap.add_argument("--suite", default="vision8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=89)
    args = ap.parse_args(argv)

    cfg = {"seed": args.seed, "val_fraction": 0.1, "batch_size": 32, "num_workers": 0, "no_humanize": True}
    build = OpenClipBuildConfig(model_name=args.model, pretrained=args.pretrained, device=args.device)
    clf = OpenClipClassifier.build(build)
    base_sd = _owned_cpu_fp32(dict(clf.model.state_dict()))
    suite = SUITES[args.suite]
    result: dict[str, Any] = {"model": [args.model, args.pretrained], "ckpt_dir": args.ckpt_dir, "seed": args.seed,
                              "by_task": {}}
    for task in args.tasks.split(","):
        ctx = _build_task_context(
            task, suite=suite, cfg=cfg, clf_target=clf, clf_source=clf, source_cfg=build, target_cfg=build,
            use_humanized_classnames=False, need_source_loaders=False,
        )

        def evaluate(ctx=ctx) -> float:
            return eval_task_top1(clf=clf, loaders=ctx.loaders, classnames=ctx.classnames,
                                  build_cfg_task=ctx.build_cfg_task, device=args.device, split="test")

        expert = load_ckpt(str(Path(args.ckpt_dir) / task.lower() / "best.pt"))
        entry = {}
        for name, sd in (("base", base_sd), ("expert", {**base_sd, **{k: v for k, v in expert.items() if k in base_sd}}),
                         ("cproj_only", cproj_restricted(base_sd, expert))):
            load_into_model(clf.model, sd, strict=True)
            entry[name] = evaluate()
        result["by_task"][task] = entry
        print(f"[expert_cproj_restriction] {task}: " + " ".join(f"{k} {100 * v:.2f}" for k, v in entry.items()), flush=True)
    load_into_model(clf.model, base_sd, strict=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
