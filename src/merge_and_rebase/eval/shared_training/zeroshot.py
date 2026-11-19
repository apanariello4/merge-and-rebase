"""Zero-shot Vision8 TEST accuracy of registered models (AGENT_ADDENDUM_CONSENSUS.md section 4: agreement vs quality).

    python -m merge_and_rebase.eval.shared_training.zeroshot --out-dir DIR MODEL_ID [MODEL_ID ...]

Writes ``DIR/zeroshot_vision8_<model_id>.json`` (same loaders/heads as the transport runs: seed 89, val_fraction 0.1).
"""
import argparse
import json
from pathlib import Path

import torch

from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES

from .data import build_classifier, task_data
from .evaluate import top1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    for m in a.models:
        out = Path(a.out_dir) / f"zeroshot_vision8_{m}.json"
        if out.exists():
            continue
        clf = build_classifier(m, a.device)
        acc = {}
        for t in SUITES["vision8"].tasks:
            td = task_data(t, {m: clf})
            acc[t] = top1(clf[0], td.contexts[m], td.text_features[m], "test", a.device)
        out.write_text(json.dumps({"model": m, "test_accuracy": acc, "mean": sum(acc.values()) / len(acc)}, indent=1))
        print(m, {t: round(v, 4) for t, v in acc.items()}, flush=True)
        del clf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
